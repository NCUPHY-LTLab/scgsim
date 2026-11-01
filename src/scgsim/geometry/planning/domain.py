"""Plan the substrate-air style interfaces between solution entities, and Route A substrate support.

Owns the horizontal interfaces where two solution entities (dielectric or vacuum) meet,
with conductors and sheets cut out, and the check that every Route A face metal sits on
exactly one dielectric. Ownership comes from source geometry, never from guessed native
coordinates. Auto vacuum, adjacency and boundary planning live in their own modules.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

from scgsim.geometry._primitives.constants import _INTERFACE_KIND_ORDER
from scgsim.geometry._primitives.entities import _entity_by_id, _entity_z_range_um
from scgsim.geometry._primitives.entity_validation import _is_solution_entity
from scgsim.geometry._primitives.loops import (
    _boolean_gdstk_region,
    _clean_loop,
    _gdstk_surface_region,
    _geometry_ref_from_gdstk_polygon,
    _geometry_refs_from_gdstk_region,
    _loop_centroid,
    _polygon_area,
    _simple_interior_hole_loops,
)
from scgsim.geometry._primitives.spatial import (
    _geometry_ref_surface_z_um,
    _intersect_bounds,
    _same_z,
)
from scgsim.geometry._primitives.surface_records import (
    _interface_kinds,
    _is_route_a_sheet_interface,
    _RouteASheetPatch,
    _surface_contribution_provenance,
)
from scgsim.geometry.models.common import RouteLiteral
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.models.topology import InterfacePlanRecord, SurfacePlanRecord
from scgsim.geometry.planning.conductor_faces import (
    _conductor_entities_on_solution_plane,
)
from scgsim.geometry.planning.evidence import solution_solution_evidence
from scgsim.geometry.planning.solution_adjacency import (
    _conductor_solution_interface_kind,
    _route_a_sheet_boundary_volume_ids,
)
from scgsim.geometry.planning.solution_boundaries import (
    _plan_solution_domain_boundary_surfaces,
)
from scgsim.geometry.planning.solution_geometry import (
    _entity_occupied_region,
    _solution_bounds,
    _solution_entities,
    _solution_entity_xy_region,
)
from scgsim.semantics import EvidenceResult, SemanticEvidenceFacade
from scgsim.semantics.route_a import geometry_z_range, record_geometry, same_z


def verified_route_a_substrate_support(
    build_input: GeometryBuildInput,
    stack: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    """Bind each face metal to one positive-area dielectric contact plane.

    The normalized face polygons and modeled substrate region (rectangle or
    explicit outer/hole loops) are intersected as regions. A metal overhang
    remains exposed; neither bounds nor a point/edge touch establishes contact.
    """
    import gdstk

    layers = stack.get("layers")
    if isinstance(layers, str | bytes) or not isinstance(layers, Sequence):
        raise TypeError("Route A requires a sequence of stack layers.")
    polygons = {polygon.polygon_id: polygon for polygon in build_input.polygons}
    substrates = tuple(
        entity
        for entity in build_input.entities
        if _is_solution_entity(entity) and entity.material_kind == "dielectric"
    )
    result: dict[str, dict[str, str]] = {}
    for record in layers:
        if not isinstance(record, Mapping):
            raise TypeError("Route A stack layers must contain mappings.")
        if record.get("part_role") != "face_metal":
            continue
        source_id = record.get("semantic_id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("face_metal requires semantic_id for substrate support.")
        face_z_min, face_z_max = geometry_z_range(
            record_geometry(record, source_id), source_id
        )
        if face_z_max <= face_z_min:
            raise ValueError(f"{source_id} physical face-metal thickness must be > 0.")
        face_regions = _face_metal_regions(
            gdstk=gdstk,
            build_input=build_input,
            polygons=polygons,
            source_id=source_id,
        )
        matches = _substrate_contact_matches(
            gdstk=gdstk,
            substrates=substrates,
            face_regions=face_regions,
            face_z_min=face_z_min,
            face_z_max=face_z_max,
        )
        if len(matches) != 1:
            raise ValueError(
                f"{source_id} requires exactly one positive-area dielectric substrate "
                f"contact at its physical face; matched {matches!r} among "
                f"{tuple(entity.semantic_id for entity in substrates)!r}."
            )
        side, substrate_id = matches[0]
        result[source_id] = {"side": side, "substrate_id": substrate_id}
    return result


def _face_metal_regions(
    gdstk: Any,
    build_input: GeometryBuildInput,
    polygons: Mapping[str, Any],
    source_id: str,
) -> list[Any]:
    """Return the xy region (holes removed) of every polygon of the face metal `source_id`."""
    face_entities = tuple(
        entity
        for entity in build_input.entities
        if entity.semantic_id == source_id
        or entity.metadata.get("semantic_group_id") == source_id
        or entity.metadata.get("source_semantic_id") == source_id
    )
    face_regions: list[Any] = []
    for entity in face_entities:
        for polygon_id in entity.polygon_ids:
            polygon = polygons.get(polygon_id)
            if polygon is None:
                raise ValueError(
                    f"{source_id} references missing normalized polygon {polygon_id!r}."
                )
            exterior = (gdstk.Polygon(_clean_loop(polygon.exterior)),)
            holes = tuple(gdstk.Polygon(_clean_loop(hole)) for hole in polygon.holes)
            face_regions.extend(_boolean_gdstk_region(gdstk, exterior, holes, "not"))
    if not face_regions:
        raise ValueError(
            f"{source_id} has no normalized face-metal polygon for substrate attachment."
        )
    return face_regions


def _substrate_contact_matches(
    gdstk: Any,
    substrates: Sequence[SemanticEntitySpec],
    face_regions: Sequence[Any],
    face_z_min: float,
    face_z_max: float,
) -> list[tuple[str, str]]:
    """Return (side, substrate id) for each substrate that meets the face metal with area.

    ``lower`` means the substrate lies below the face metal (its top is at the metal's
    bottom); ``upper`` means it lies above.
    """
    matches: list[tuple[str, str]] = []
    for substrate in substrates:
        sub_z_min, sub_z_max = _entity_z_range_um(substrate)
        sides = (
            *(("lower",) if same_z(sub_z_max, face_z_min) else ()),
            *(("upper",) if same_z(sub_z_min, face_z_max) else ()),
        )
        if not sides:
            continue
        substrate_region = _solution_entity_xy_region(gdstk, substrate)
        if _boolean_gdstk_region(gdstk, face_regions, substrate_region, "and"):
            matches.extend((side, substrate.semantic_id) for side in sides)
    return matches


def _interface_surface_kinds(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral | None,
    interface: InterfacePlanRecord,
    route_a_evidence: tuple[EvidenceResult, ...] = (),
    representation_context: RepresentationContext | None = None,
) -> tuple[str, ...]:
    raw_kinds = interface.metadata.get("interface_kinds")
    if raw_kinds is not None:
        if isinstance(raw_kinds, str):
            raw_kinds = (raw_kinds,)
        return tuple(kind for kind in _INTERFACE_KIND_ORDER if kind in set(raw_kinds))
    if _is_route_a_sheet_interface(
        route, interface, representation_context=representation_context
    ):
        derived = (
            tuple(result.classification for result in route_a_evidence)
            if route_a_evidence
            else tuple(
                _conductor_solution_interface_kind(
                    _entity_by_id(build_input, boundary_id)
                )
                for boundary_id in _route_a_sheet_boundary_volume_ids(
                    build_input,
                    _entity_by_id(build_input, interface.owner_semantic_ids[0]),
                )
            )
        )
        return tuple(kind for kind in _INTERFACE_KIND_ORDER if kind in set(derived))
    kinds = set(_interface_kinds(interface))
    return tuple(kind for kind in _INTERFACE_KIND_ORDER if kind in kinds)


def _plan_substrate_air_surfaces(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral | None,
    semantic_facts: SemanticEvidenceFacade | None = None,
    route_a_sheet_patches: Sequence[_RouteASheetPatch] = (),
    representation_context: RepresentationContext | None = None,
) -> tuple[SurfacePlanRecord, ...]:
    """Plan the surfaces between neighbouring solution entities, then the domain walls.

    These are the substrate-air (SA), air-air (AA) and substrate-substrate (SS)
    interfaces. Each is a horizontal plane shared by a lower and an upper solution.
    Conductors and Route A sheets sitting on that plane are cut out of it, because
    their own surfaces take over there.
    """
    import gdstk

    records: list[SurfacePlanRecord] = list(
        _plan_solution_domain_boundary_surfaces(
            build_input,
            route=route,
            representation_context=representation_context,
        )
    )
    surface_index = 0
    for lower, upper, z_um, _ in _solution_interface_planes(build_input):
        plane = _SolutionPlane(lower, upper, z_um)
        interface_region = _boolean_gdstk_region(
            gdstk,
            _solution_entity_xy_region(gdstk, lower),
            _solution_entity_xy_region(gdstk, upper),
            "and",
        )
        if not interface_region:
            continue
        interface_evidence = solution_solution_evidence(
            semantic_facts,
            contribution_id=(
                f"solution-solution:{lower.semantic_id}:{upper.semantic_id}:z={z_um}"
            ),
            patch_id=(f"planned:{lower.semantic_id}:{upper.semantic_id}:z={z_um}"),
            lower_id=lower.semantic_id,
            upper_id=upper.semantic_id,
            side=_solution_plane_physical_side(lower, upper),
        )
        for patch in _sorted_interface_patches(interface_region):
            base_geometry_ref = _geometry_ref_from_gdstk_polygon(
                {"plane": {"axis": "z", "value_um": z_um}},
                patch,
            )
            plane_conductors = _conductor_entities_on_solution_plane(
                build_input,
                route=route,
                representation_context=representation_context,
                lower=lower,
                upper=upper,
                z_um=z_um,
                base_region=(patch,),
            )
            solution_geometry_refs = _solution_interface_geometry_refs(
                base_geometry_ref,
                plane_conductors,
                conductor_geometry_refs=_sheet_geometry_refs_on_patch(
                    gdstk, plane, patch, route_a_sheet_patches
                ),
            )
            for geometry_ref in solution_geometry_refs:
                records.append(
                    _solution_interface_record(
                        plane,
                        interface_evidence,
                        geometry_ref=geometry_ref,
                        surface_index=surface_index,
                        route=route,
                    )
                )
                surface_index += 1
    return tuple(records)


class _SolutionPlane(NamedTuple):
    """A horizontal plane shared by a lower and an upper solution entity."""

    lower: SemanticEntitySpec
    upper: SemanticEntitySpec
    z_um: float


def _solution_plane_physical_side(
    lower: SemanticEntitySpec, upper: SemanticEntitySpec
) -> str:
    """Tell the semantic core which side of the dielectric faces the vacuum, if any."""
    material_pair = (lower.material_kind, upper.material_kind)
    if material_pair == ("dielectric", "vacuum"):
        return "top"
    if material_pair == ("vacuum", "dielectric"):
        return "bottom"
    return "shared_plane"


def _sorted_interface_patches(interface_region: Sequence[Any]) -> tuple[Any, ...]:
    """Order patches by rounded centroid, then area, so surface ids are numbered stably."""
    return tuple(
        sorted(
            interface_region,
            key=lambda polygon: (
                round(_loop_centroid(_clean_loop(polygon.points))[0], 9),
                round(_loop_centroid(_clean_loop(polygon.points))[1], 9),
                round(abs(_polygon_area(polygon.points)), 9),
            ),
        )
    )


def _sheet_geometry_refs_on_patch(
    gdstk: Any,
    plane: _SolutionPlane,
    patch: Any,
    route_a_sheet_patches: Sequence[_RouteASheetPatch],
) -> tuple[Mapping[str, Any], ...]:
    """Return the Route A sheet patches that lie on this plane and overlap the patch."""
    return tuple(
        sheet_patch.geometry_ref
        for sheet_patch in route_a_sheet_patches
        if sheet_patch.boundary_volume_ids
        == (plane.lower.semantic_id, plane.upper.semantic_id)
        and _same_z(
            _geometry_ref_surface_z_um(sheet_patch.geometry_ref),
            plane.z_um,
        )
        and _boolean_gdstk_region(
            gdstk,
            _gdstk_surface_region(sheet_patch.geometry_ref),
            (patch,),
            "and",
        )
    )


def _solution_interface_record(
    plane: _SolutionPlane,
    interface_evidence: EvidenceResult,
    *,
    geometry_ref: Mapping[str, Any],
    surface_index: int,
    route: RouteLiteral | None,
) -> SurfacePlanRecord:
    """Build one solution-interface surface from its evidence and geometry."""
    lower, upper = plane.lower, plane.upper
    kind = interface_evidence.classification
    owner_ids = interface_evidence.source_owner_ids
    interface_id = f"{kind}__{owner_ids[0]}__{owner_ids[1]}__{surface_index:04d}"
    return SurfacePlanRecord(
        surface_id=f"SURF__{interface_id}",
        owner_semantic_id=owner_ids[0],
        surface_role="solution_interface",
        geometry_ref=geometry_ref,
        interface_id=interface_id,
        valid_routes=() if route is None else (route,),
        solver_use="solver_active",
        metadata={
            "interface_kinds": (kind,),
            "owner_semantic_ids": owner_ids,
            "boundary_volume_ids": (
                lower.semantic_id,
                upper.semantic_id,
            ),
            "source_provenance": _surface_contribution_provenance(
                parent_interface_id=(
                    f"solution-interface:{lower.semantic_id}:{upper.semantic_id}"
                ),
                patch_id=interface_evidence.contribution_id,
                contributions=(interface_evidence,),
            ),
        },
    )


def _solution_interface_geometry_refs(
    parent_geometry_ref: Mapping[str, Any],
    plane_conductors: Sequence[SemanticEntitySpec],
    *,
    conductor_geometry_refs: Sequence[Mapping[str, Any]] = (),
) -> tuple[dict[str, Any], ...]:
    """Create live solution-interface patches after removing conductors."""
    if not plane_conductors and not conductor_geometry_refs:
        return (dict(parent_geometry_ref),)

    import gdstk

    hole_loops = _simple_interior_hole_loops(
        parent_geometry_ref,
        plane_conductors,
        conductor_geometry_refs=conductor_geometry_refs,
    )
    if hole_loops is not None:
        return ({**dict(parent_geometry_ref), "hole_loops": hole_loops},)

    base_region = _gdstk_surface_region(parent_geometry_ref)
    conductor_region = tuple(
        polygon
        for entity in plane_conductors
        for polygon in _entity_occupied_region(gdstk, entity)
    ) + tuple(
        polygon
        for geometry_ref in conductor_geometry_refs
        for polygon in _gdstk_surface_region(geometry_ref)
    )
    live_region = _boolean_gdstk_region(
        gdstk,
        base_region,
        conductor_region,
        "not",
    )
    return _geometry_refs_from_gdstk_region(parent_geometry_ref, live_region)


def _solution_interface_planes(
    build_input: GeometryBuildInput,
) -> tuple[
    tuple[SemanticEntitySpec, SemanticEntitySpec, float, Mapping[str, float]],
    ...,
]:
    records: list[
        tuple[SemanticEntitySpec, SemanticEntitySpec, float, Mapping[str, float]]
    ] = []
    solutions = _solution_entities(build_input)
    for index, first in enumerate(solutions):
        for second in solutions[index + 1 :]:
            first_bounds = _solution_bounds(first)
            second_bounds = _solution_bounds(second)
            overlap = _intersect_bounds(first_bounds, second_bounds)
            if overlap is None:
                continue
            if _same_z(
                float(first.geometry["z_max_um"]),
                float(second.geometry["z_min_um"]),
            ):
                records.append(
                    (first, second, float(first.geometry["z_max_um"]), overlap)
                )
            elif _same_z(
                float(second.geometry["z_max_um"]),
                float(first.geometry["z_min_um"]),
            ):
                records.append(
                    (second, first, float(second.geometry["z_max_um"]), overlap)
                )
    return tuple(records)
