"""Trim finite conductor walls against exact same-Z contacts and solution boundaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from scgsim.geometry.models.common import (
    HIGH_COUNT_LOCAL_CONDUCTOR_PART_ROLES,
    RouteLiteral,
)
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.models.topology import (
    InterfacePlanRecord,
)
from scgsim.geometry.planning.domain import (
    adjacent_auto_vacuum_sidewall_component_id,
    conductor_sidewall_adjacent_solution_id,
    conductor_sidewall_ref_solution_id,
    route_conductor_contact_footprints,
    sidewall_is_auto_vacuum_envelope_edge,
    sidewall_on_solution_outer_boundary,
)
from scgsim.geometry.planning.representation import (
    planned_conductor_representation,
)
from scgsim.geometry.planning.surface_planning.contacts import (
    subtract_contact_patches_from_face,
)
from scgsim.geometry.primitives.constants import (
    TOPOLOGY_EPS_UM,
)
from scgsim.geometry.primitives.entities import (
    entity_by_id,
    entity_z_range_um,
)
from scgsim.geometry.primitives.entity_validation import is_solution_entity
from scgsim.geometry.primitives.geometry_refs import (
    route_entity_geometry_ref,
    sidewall_geometry_refs,
)
from scgsim.geometry.primitives.planar_loops import clean_loop, ring_edges
from scgsim.geometry.primitives.spatial import (
    interpolate_3d,
    interval_complement,
    line_key_2d,
    same_coordinate_z,
    segment_overlap_interval,
)


@dataclass(frozen=True)
class ConductorSidewallRequest:
    """Source conductor, representation and exclusions used to resolve live walls."""

    build_input: GeometryBuildInput
    route: RouteLiteral | None = field(kw_only=True)
    entity: SemanticEntitySpec = field(kw_only=True)
    representation: str = field(kw_only=True)
    adjacent_solution_id: str | None = field(kw_only=True)
    interfaces: tuple[InterfacePlanRecord, ...] = field(default=(), kw_only=True)
    excluded_footprints: Sequence[tuple[tuple[float, float], ...]] = field(
        default=(), kw_only=True
    )
    representation_context: RepresentationContext | None = field(
        default=None, kw_only=True
    )


def conductor_sidewall_geometry_refs(
    request: ConductorSidewallRequest,
) -> tuple[dict[str, Any], ...]:
    """Build sidewall references with contact and solution-domain ownership."""
    build_input = request.build_input
    route = request.route
    entity = request.entity
    representation = request.representation
    adjacent_solution_id = request.adjacent_solution_id
    interfaces = request.interfaces
    excluded_footprints = request.excluded_footprints
    representation_context = request.representation_context
    base_ref = route_entity_geometry_ref(
        build_input,
        route,
        entity,
        representation=representation,
        interfaces=interfaces,
    )
    face_refs = subtract_contact_patches_from_face(base_ref, excluded_footprints)
    refs = tuple(
        side_ref
        for face_ref in face_refs
        for side_ref in sidewall_geometry_refs(face_ref)
    )
    edge_index = _conductor_boundary_edge_index(
        build_input,
        route=route,
        entity=entity,
        representation_context=representation_context,
    )
    if route == "C":
        if adjacent_solution_id is None:
            raise ValueError(
                f"{entity.semantic_id} Route C sidewall requires a solution."
            )
        return _route_c_conductor_sidewall_geometry_refs(
            entity=entity,
            adjacent_solution=entity_by_id(build_input, adjacent_solution_id),
            refs=refs,
            edge_index=edge_index,
        )
    exposed_refs: list[dict[str, Any]] = []
    for raw_geometry_ref in refs:
        if sidewall_is_auto_vacuum_envelope_edge(
            build_input,
            entity=entity,
            geometry_ref=raw_geometry_ref,
        ):
            continue
        # Exact same-z conductor contacts remove their shared MM segment
        # before solution adjacency is selected.  One raw sidewall can span
        # contact and solution portions; each retained topology segment binds
        # an authored dielectric cavity or a ledger-bound vacuum child.
        for geometry_ref in _trim_sidewall_ref_against_route_conductors(
            geometry_ref=raw_geometry_ref,
            edge_index=edge_index,
        ):
            exact_adjacency, ref_adjacent_solution_id = (
                conductor_sidewall_ref_solution_id(
                    build_input,
                    entity=entity,
                    geometry_ref=geometry_ref,
                )
            )
            if exact_adjacency:
                if ref_adjacent_solution_id is None:
                    raise ValueError(
                        f"{entity.semantic_id} sidewall has no exact solution boundary."
                    )
            else:
                if adjacent_solution_id is None:
                    adjacent_solution_id = conductor_sidewall_adjacent_solution_id(
                        build_input,
                        entity,
                    )
                if adjacent_solution_id is None:
                    continue
                ref_adjacent_solution_id = adjacent_solution_id
            adjacent_solution = entity_by_id(build_input, ref_adjacent_solution_id)
            # The explicit parent-envelope check above remains the exterior rule
            # for exact source boundaries. A disconnected child's rectangular bounds
            # can coincide with a conductor hole edge, which is not an exterior
            # boundary and must retain its exact child adjacency.
            if not exact_adjacency and sidewall_on_solution_outer_boundary(
                geometry_ref, adjacent_solution
            ):
                ref_adjacent_solution_id = adjacent_auto_vacuum_sidewall_component_id(
                    build_input,
                    entity=entity,
                    geometry_ref=geometry_ref,
                )
                if ref_adjacent_solution_id is None:
                    continue
            exposed_refs.append(
                {
                    **geometry_ref,
                    "adjacent_solution_id": ref_adjacent_solution_id,
                }
            )
    return tuple(exposed_refs)


def _route_c_conductor_sidewall_geometry_refs(
    *,
    entity: SemanticEntitySpec,
    adjacent_solution: SemanticEntitySpec,
    refs: tuple[dict[str, Any], ...],
    edge_index: Mapping[
        tuple[tuple[float, float], float],
        Sequence[tuple[tuple[float, float], tuple[float, float], str]],
    ],
) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    for geometry_ref in refs:
        points = geometry_ref.get("quad_points", ())
        if len(points) != 4:
            result.append(dict(geometry_ref))
            continue
        start = (float(points[0][0]), float(points[0][1]))
        end = (float(points[1][0]), float(points[1][1]))
        exterior_intervals = (
            ((0.0, 1.0),)
            if sidewall_on_solution_outer_boundary(geometry_ref, adjacent_solution)
            else ()
        )
        contact_intervals = _route_c_contact_intervals(
            entity=entity,
            start=start,
            end=end,
            edge_index=edge_index,
        )
        result.extend(
            _sidewall_subsegment_geometry_ref(
                geometry_ref,
                interval_start,
                interval_end,
                extra={"solution_exterior_boundary": True},
            )
            for interval_start, interval_end in exterior_intervals
        )
        result.extend(
            _sidewall_subsegment_geometry_ref(
                geometry_ref,
                interval_start,
                interval_end,
                extra={"adjacent_conductor_semantic_id": adjacent_id},
            )
            for (
                interval_start,
                interval_end,
                adjacent_id,
                create_surface,
            ) in contact_intervals
            if create_surface
        )
        blocked = (
            *exterior_intervals,
            *((start, end) for start, end, _, _ in contact_intervals),
        )
        result.extend(
            _sidewall_subsegment_geometry_ref(
                geometry_ref,
                interval_start,
                interval_end,
            )
            for interval_start, interval_end in interval_complement(blocked)
        )
    return tuple(result)


def _conductor_boundary_edge_index(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral | None,
    entity: SemanticEntitySpec,
    representation_context: RepresentationContext | None = None,
) -> dict[
    tuple[tuple[float, float], float],
    tuple[tuple[tuple[float, float], tuple[float, float], str], ...],
]:
    if entity.part_role in HIGH_COUNT_LOCAL_CONDUCTOR_PART_ROLES:
        return {}
    index: dict[
        tuple[tuple[float, float], float],
        list[tuple[tuple[float, float], tuple[float, float], str]],
    ] = {}
    edge_entities = (
        route_conductor_contact_footprints(
            build_input, route, representation_context=representation_context
        )
        if route is None and representation_context is not None
        else build_input.entities
    )
    for other in edge_entities:
        if not _can_trim_against_route_conductor(
            route,
            entity=entity,
            other=other,
            representation_context=representation_context,
        ):
            continue
        for other_start, other_end in _entity_boundary_edges(other):
            line_key = line_key_2d(other_start, other_end)
            if line_key is None:
                continue
            index.setdefault(line_key, []).append(
                (other_start, other_end, other.semantic_id)
            )
    return {line_key: tuple(records) for line_key, records in index.items()}


def _candidate_boundary_edges(
    edge_index: Mapping[
        tuple[tuple[float, float], float],
        Sequence[tuple[tuple[float, float], tuple[float, float], str]],
    ],
    start: tuple[float, float],
    end: tuple[float, float],
) -> Sequence[tuple[tuple[float, float], tuple[float, float], str]]:
    line_key = line_key_2d(start, end)
    if line_key is None:
        return ()
    return edge_index.get(line_key, ())


def _route_c_contact_intervals(
    entity: SemanticEntitySpec,
    start: tuple[float, float],
    end: tuple[float, float],
    edge_index: Mapping[
        tuple[tuple[float, float], float],
        Sequence[tuple[tuple[float, float], tuple[float, float], str]],
    ],
) -> tuple[tuple[float, float, str, bool], ...]:
    records: list[tuple[float, float, str, bool]] = []
    for other_start, other_end, other_semantic_id in _candidate_boundary_edges(
        edge_index,
        start,
        end,
    ):
        interval = segment_overlap_interval(start, end, other_start, other_end)
        if interval is None:
            continue
        records.append(
            (
                interval[0],
                interval[1],
                other_semantic_id,
                entity.semantic_id < other_semantic_id,
            )
        )
    return tuple(records)


def _trim_sidewall_ref_against_route_conductors(
    *,
    geometry_ref: Mapping[str, Any],
    edge_index: Mapping[
        tuple[tuple[float, float], float],
        Sequence[tuple[tuple[float, float], tuple[float, float], str]],
    ],
) -> tuple[dict[str, Any], ...]:
    points = geometry_ref.get("quad_points", ())
    if len(points) != 4:
        return (dict(geometry_ref),)
    start = (float(points[0][0]), float(points[0][1]))
    end = (float(points[1][0]), float(points[1][1]))
    covered_intervals: list[tuple[float, float]] = []
    for other_start, other_end, _ in _candidate_boundary_edges(
        edge_index,
        start,
        end,
    ):
        interval = segment_overlap_interval(start, end, other_start, other_end)
        if interval is None:
            continue
        covered_intervals.append(interval)
    if not covered_intervals:
        return (dict(geometry_ref),)
    return tuple(
        _sidewall_subsegment_geometry_ref(
            geometry_ref,
            start_parameter,
            end_parameter,
        )
        for start_parameter, end_parameter in interval_complement(
            covered_intervals,
        )
        if end_parameter - start_parameter > TOPOLOGY_EPS_UM
    )


def _can_trim_against_route_conductor(
    route: RouteLiteral | None,
    *,
    entity: SemanticEntitySpec,
    other: SemanticEntitySpec,
    representation_context: RepresentationContext | None = None,
) -> bool:
    same_parent_distinct_polygons = (
        route is None
        and representation_context is not None
        and other.semantic_id == entity.semantic_id
        and bool(entity.polygon_ids)
        and bool(other.polygon_ids)
        and set(entity.polygon_ids).isdisjoint(other.polygon_ids)
    )
    if (
        other.semantic_id == entity.semantic_id
        and not same_parent_distinct_polygons
        or is_solution_entity(other)
        or planned_conductor_representation(
            other, route=route, context=representation_context
        )
        is None
        or "outer_loop" not in other.geometry
    ):
        return False
    z_min_um, z_max_um = entity_z_range_um(entity)
    other_z_min_um, other_z_max_um = entity_z_range_um(other)
    return same_coordinate_z(z_min_um, other_z_min_um) and same_coordinate_z(
        z_max_um, other_z_max_um
    )


def _entity_boundary_edges(
    entity: SemanticEntitySpec,
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    return tuple(
        edge
        for loop in (
            entity.geometry["outer_loop"],
            *entity.geometry.get("hole_loops", ()),
        )
        for edge in ring_edges(clean_loop(loop))
    )


def _sidewall_subsegment_geometry_ref(
    geometry_ref: Mapping[str, Any],
    start_parameter: float,
    end_parameter: float,
    *,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    points = tuple(
        (float(point[0]), float(point[1]), float(point[2]))
        for point in geometry_ref["quad_points"]
    )
    return {
        **dict(geometry_ref),
        "quad_points": (
            interpolate_3d(points[0], points[1], start_parameter),
            interpolate_3d(points[0], points[1], end_parameter),
            interpolate_3d(points[3], points[2], end_parameter),
            interpolate_3d(points[3], points[2], start_parameter),
        ),
        "trimmed_by_conductor_contact": True,
        **dict(extra or {}),
    }
