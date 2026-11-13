"""Resolve planar conductor faces and their incident solution domains."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
from typing import Any

from scgsim.geometry.grid.grid import boolean_polygon_region
from scgsim.geometry.grid.loops import (
    gdstk_surface_region,
    geometry_refs_from_gdstk_region,
    loop_centroid,
    loop_signature,
)
from scgsim.geometry.interpretation.envelope import auto_vacuum_sheet_z
from scgsim.geometry.models.common import INTERFACE_KIND_ORDER, RouteLiteral
from scgsim.geometry.models.entities import entity_z_range_um, is_vacuum_solution_entity
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.models.topology import InterfacePlanRecord
from scgsim.geometry.planning.domain_planning.hosts import auto_vacuum_host_component_id
from scgsim.geometry.planning.domain_planning.occupancy import (
    solution_bounds,
    solution_entity_xy_region,
    solution_id_for_entity_coverage,
    solution_region_entities,
)
from scgsim.geometry.planning.entities import entity_by_id, unique_ids
from scgsim.geometry.planning.spatial import bounds_contains_point, same_coordinate_z
from scgsim.geometry.planning.surface_records import (
    interface_kinds,
    is_route_a_sheet_interface,
)
from scgsim.semantics import (
    EvidenceResult,
    SemanticEvidenceFacade,
    conductor_solution_interface_kind,
)


def planar_side_solution_regions(
    build_input: GeometryBuildInput,
    *,
    owner_id: str,
    occupied_region: tuple[Any, ...],
    plane_z_um: float,
    side: str,
) -> tuple[tuple[str, tuple[Any, ...]], ...]:
    """Resolve exact nonoverlapping solution coverage on one planar side."""
    import gdstk  # noqa: PLC0415 -- defer the optional polygon engine until this operation reaches native geometry.

    records: list[tuple[str, tuple[Any, ...]]] = []
    for solution in solution_region_entities(build_input):
        z_min_um = float(solution.geometry["z_min_um"])
        z_max_um = float(solution.geometry["z_max_um"])
        on_boundary = (
            same_coordinate_z(z_max_um, plane_z_um)
            if side == "bottom"
            else same_coordinate_z(z_min_um, plane_z_um)
        )
        # Endpoint equivalence and strict interior are mutually exclusive.
        contains_plane = (
            z_min_um < plane_z_um < z_max_um
            and (not same_coordinate_z(z_min_um, plane_z_um))
            and (not same_coordinate_z(z_max_um, plane_z_um))
        )
        if not on_boundary and (not contains_plane):
            continue
        overlap = boolean_polygon_region(
            gdstk, occupied_region, solution_entity_xy_region(gdstk, solution), "and"
        )
        if overlap:
            records.append((solution.semantic_id, overlap))
    for index, (left_id, left_region) in enumerate(records):
        for right_id, right_region in records[index + 1 :]:
            if boolean_polygon_region(gdstk, left_region, right_region, "and"):
                raise ValueError(
                    f"{owner_id} {side} local adjacency is ambiguous between {left_id!r} and {right_id!r}."
                )
    covered: tuple[Any, ...] = ()
    for _, region in records:
        covered = (
            region
            if not covered
            else boolean_polygon_region(gdstk, covered, region, "or")
        )
    uncovered = boolean_polygon_region(gdstk, occupied_region, covered, "not")
    if uncovered:
        raise ValueError(f"{owner_id} {side} has no local solution coverage.")
    return tuple(records)


@dataclass(frozen=True)
class ConductorFaceRequest:
    """Carry one conductor face query and its route-specific solution context."""

    build_input: GeometryBuildInput
    _: KW_ONLY
    route: RouteLiteral | None
    entity: SemanticEntitySpec
    shell_part: str
    geometry_refs: Sequence[Mapping[str, Any]]
    representation_context: RepresentationContext | None = None


def conductor_face_solution_pieces(
    request: ConductorFaceRequest,
) -> tuple[tuple[str, dict[str, Any]], ...]:
    """Partition Route-A/B planar faces by exact local solution adjacency."""
    if request.route not in {"A", "B"} and (
        not (request.route is None and request.representation_context is not None)
    ):
        adjacent_id = _conductor_face_adjacent_solution_id(
            request.build_input, request.entity, request.shell_part
        )
        return tuple(
            (
                (adjacent_id, dict(geometry_ref))
                for geometry_ref in request.geometry_refs
            )
        )
    z_min_um, z_max_um = entity_z_range_um(request.entity)
    plane_z_um = z_min_um if request.shell_part == "bottom" else z_max_um
    records: list[tuple[str, dict[str, Any]]] = []
    for geometry_ref in request.geometry_refs:
        occupied_region = gdstk_surface_region(geometry_ref)
        adjacent_regions = planar_side_solution_regions(
            request.build_input,
            owner_id=request.entity.semantic_id,
            occupied_region=occupied_region,
            plane_z_um=plane_z_um,
            side=request.shell_part,
        )
        for adjacent_id, region in adjacent_regions:
            records.extend(
                (
                    (adjacent_id, child)
                    for child in geometry_refs_from_gdstk_region(geometry_ref, region)
                )
            )
    return tuple(
        sorted(
            records, key=lambda item: (item[0], loop_signature(item[1]["outer_loop"]))
        )
    )


def route_a_sheet_plane_z_um_from_solutions(
    entity: SemanticEntitySpec, solution_entities: Sequence[SemanticEntitySpec]
) -> float:
    """Find the sheet plane coincident with an incident solution boundary."""
    z_min_um, z_max_um = entity_z_range_um(entity)
    point = loop_centroid(entity.geometry["outer_loop"])
    for face_z_um, solution_edge_key in (
        (z_min_um, "z_max_um"),
        (z_max_um, "z_min_um"),
    ):
        for solution in solution_entities:
            if not bounds_contains_point(solution_bounds(solution), point):
                continue
            if not same_coordinate_z(
                float(solution.geometry[solution_edge_key]), face_z_um
            ):
                continue
            if not is_vacuum_solution_entity(solution):
                return face_z_um
    return z_min_um


def route_a_sheet_boundary_volume_ids_from_solutions(
    entity: SemanticEntitySpec, solution_entities: Sequence[SemanticEntitySpec]
) -> tuple[str, ...]:
    """Collect solution volumes incident to the conductor's sheet faces."""
    z_min_um, z_max_um = entity_z_range_um(entity)
    point = loop_centroid(entity.geometry["outer_loop"])
    ids: list[str] = []
    for face_z_um, solution_edge_key in (
        (z_min_um, "z_max_um"),
        (z_max_um, "z_min_um"),
    ):
        for solution in solution_entities:
            if not bounds_contains_point(solution_bounds(solution), point):
                continue
            if same_coordinate_z(
                float(solution.geometry[solution_edge_key]), face_z_um
            ):
                ids.append(solution.semantic_id)
                break
    return unique_ids(ids)


def interface_surface_kinds(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral | None,
    interface: InterfacePlanRecord,
    route_a_evidence: tuple[EvidenceResult, ...] = (),
    representation_context: RepresentationContext | None = None,
) -> tuple[str, ...]:
    """Resolve ordered surface kinds for one recognized interface."""
    raw_kinds = interface.metadata.get("interface_kinds")
    if raw_kinds is not None:
        if isinstance(raw_kinds, str):
            raw_kinds = (raw_kinds,)
        return tuple((kind for kind in INTERFACE_KIND_ORDER if kind in set(raw_kinds)))
    if is_route_a_sheet_interface(
        route, interface, representation_context=representation_context
    ):
        derived = (
            tuple((result.classification for result in route_a_evidence))
            if route_a_evidence
            else tuple(
                (
                    _conductor_solution_interface_kind(
                        entity_by_id(build_input, boundary_id)
                    )
                    for boundary_id in route_a_sheet_boundary_volume_ids(
                        build_input,
                        entity_by_id(build_input, interface.owner_semantic_ids[0]),
                    )
                )
            )
        )
        return tuple((kind for kind in INTERFACE_KIND_ORDER if kind in set(derived)))
    kinds = set(interface_kinds(interface))
    return tuple((kind for kind in INTERFACE_KIND_ORDER if kind in kinds))


def route_a_sheet_boundary_volume_ids(
    build_input: GeometryBuildInput, entity: SemanticEntitySpec
) -> tuple[str, ...]:
    """Collect the distinct solution owners adjacent to both conductor faces."""
    return unique_ids(
        (
            _conductor_face_adjacent_solution_id(build_input, entity, "bottom"),
            _conductor_face_adjacent_solution_id(build_input, entity, "top"),
        )
    )


def route_a_sheet_plane_z_um(
    build_input: GeometryBuildInput,
    entity: SemanticEntitySpec,
    *,
    representation_context: RepresentationContext | None = None,
) -> float:
    """Resolve a Route A sheet plane from its host and solution geometry."""
    if any(
        (
            solution.metadata.get("auto_vacuum_group_id")
            == entity.host_void_semantic_id
            for solution in solution_region_entities(build_input)
        )
    ):
        return auto_vacuum_sheet_z(
            build_input, entity, representation_context=representation_context
        )
    z_min_um, z_max_um = entity_z_range_um(entity)
    for boundary_id, z_um in (
        (_conductor_face_adjacent_solution_id(build_input, entity, "bottom"), z_min_um),
        (_conductor_face_adjacent_solution_id(build_input, entity, "top"), z_max_um),
    ):
        if not is_vacuum_solution_entity(entity_by_id(build_input, boundary_id)):
            return z_um
    return z_min_um


def _conductor_face_adjacent_solution_id(
    build_input: GeometryBuildInput, entity: SemanticEntitySpec, face: str
) -> str:
    z_min_um, z_max_um = entity_z_range_um(entity)
    face_z_um = z_min_um if face == "bottom" else z_max_um
    auto_parent, auto_component = auto_vacuum_host_component_id(
        build_input, entity=entity, relation=face, z_um=face_z_um
    )
    if auto_component is not None:
        return auto_component
    if auto_parent:
        non_auto_solution = solution_id_for_entity_coverage(
            build_input, entity, z_um=face_z_um, mode=face, include_auto_vacuum=False
        )
        if non_auto_solution is not None:
            return non_auto_solution
        adjacent_solution = solution_id_for_entity_coverage(
            build_input, entity, z_um=face_z_um, mode=face
        )
        if (
            adjacent_solution is not None
            and entity_by_id(build_input, adjacent_solution).metadata.get(
                "auto_vacuum_group_id"
            )
            == entity.host_void_semantic_id
        ):
            return adjacent_solution
        containing_solution = solution_id_for_entity_coverage(
            build_input, entity, z_um=face_z_um, mode="containing"
        )
        if (
            containing_solution is not None
            and entity_by_id(build_input, containing_solution).metadata.get(
                "auto_vacuum_group_id"
            )
            == entity.host_void_semantic_id
        ):
            return containing_solution
        raise ValueError(
            f"{entity.semantic_id} {face} has no local auto-vacuum component."
        )
    exact = solution_id_for_entity_coverage(
        build_input, entity, z_um=face_z_um, mode=face
    )
    if exact is not None:
        return exact
    containing = solution_id_for_entity_coverage(
        build_input, entity, z_um=face_z_um, mode="containing"
    )
    if containing is not None:
        return containing
    raise ValueError(f"{entity.semantic_id} {face} face has no adjacent solution")


def _conductor_solution_interface_kind(
    solution: SemanticEntitySpec,
    *,
    semantic_facts: SemanticEvidenceFacade | None = None,
) -> str:
    material_kind = (
        semantic_facts.material_kind(solution.semantic_id)
        if semantic_facts is not None
        else solution.material_kind
    )
    if not isinstance(material_kind, str):
        raise ValueError(f"{solution.semantic_id} has no snapshotted material kind")
    return conductor_solution_interface_kind(material_kind)
