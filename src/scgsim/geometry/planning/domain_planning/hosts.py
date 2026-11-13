"""Resolve authored vacuum hosts and exact sidewall component ownership."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from scgsim.geometry.grid.entities import entity_z_range_um
from scgsim.geometry.grid.loops import edge_is_covered_by_loop_edge
from scgsim.geometry.grid.planar_loops import clean_loop, ring_edges
from scgsim.geometry.grid.spatial import (
    interval_complement,
    range_within_bounds,
    same_coordinate_z,
    segment_overlap_interval,
)
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.planning.domain_planning.occupancy import (
    solution_id_for_entity_coverage,
    solution_region_entities,
)
from scgsim.geometry.planning.geometry_refs import (
    route_entity_geometry_ref,
    sidewall_geometry_refs,
)


def conductor_sidewall_adjacent_solution_id(
    build_input: GeometryBuildInput, entity: SemanticEntitySpec
) -> str | None:
    """Resolve the solution volume adjacent to a conductor sidewall."""
    z_min_um, z_max_um = entity_z_range_um(entity)
    mid_z_um = (z_min_um + z_max_um) / 2.0
    auto_parent, auto_component = auto_vacuum_host_component_id(
        build_input, entity=entity, relation="sidewall", z_um=mid_z_um
    )
    if auto_component is not None:
        return auto_component
    if auto_parent:
        if _auto_vacuum_sidewall_is_exterior_only(build_input, entity):
            return None
        raise ValueError(
            f"{entity.semantic_id} sidewall has no local auto-vacuum component."
        )
    solution_id = solution_id_for_entity_coverage(
        build_input, entity, z_um=mid_z_um, mode="containing"
    )
    if solution_id is not None:
        return solution_id
    raise ValueError(f"{entity.semantic_id} sidewall has no adjacent solution")


def auto_vacuum_host_component_id(
    build_input: GeometryBuildInput,
    *,
    entity: SemanticEntitySpec,
    relation: str,
    z_um: float,
) -> tuple[bool, str | None]:
    """Resolve an explicitly authored auto-vacuum parent to one child only.

    This uses only the parent group recorded on an auto-vacuum component, the
    exact subtractor entity ledger for that component, and the declared z
    relation. It intentionally does not sample geometry, physical labels, or
    bounding boxes, and fails rather than choosing across multiple components.
    """
    host_id = entity.host_void_semantic_id
    if not isinstance(host_id, str) or not host_id:
        raise ValueError(f"{entity.semantic_id} requires host_void_semantic_id")
    components = []
    has_parent = False
    for solution in solution_region_entities(build_input):
        metadata = solution.metadata
        if metadata.get("auto_vacuum_group_id") != host_id:
            continue
        has_parent = True
        entity_ids = _component_entity_ids(solution, relation, metadata)
        if entity.semantic_id not in entity_ids:
            continue
        z_min = float(solution.geometry["z_min_um"])
        z_max = float(solution.geometry["z_max_um"])
        if relation == "sidewall":
            matches = z_min < z_um < z_max
        elif relation == "bottom":
            matches = same_coordinate_z(z_max, z_um)
        else:
            matches = same_coordinate_z(z_min, z_um)
        if matches:
            components.append(solution.semantic_id)
    if not has_parent:
        return (False, None)
    if len(components) > 1:
        raise ValueError(
            f"{entity.semantic_id} {relation} requires exactly one structured auto-vacuum component for parent {host_id!r}; got {components!r}."
        )
    return (True, components[0] if components else None)


def conductor_sidewall_ref_solution_id(
    build_input: GeometryBuildInput,
    *,
    entity: SemanticEntitySpec,
    geometry_ref: Mapping[str, Any],
) -> tuple[bool, str | None]:
    """Bind one sidewall to its source dielectric cavity or vacuum child.

    A conductor's host names the background complement, not the material on
    every wall. Its outer wall can coincide with an authored dielectric bore;
    its inner wall remains adjacent to a ledger-bound vacuum component. Both
    use complete canonical segment coverage and the declared Z interval, not
    a point, name, bounds proxy, or a guessed conductor role. The boolean result
    distinguishes this exact binding from legacy whole-footprint coverage.
    """
    host_id = entity.host_void_semantic_id
    if not isinstance(host_id, str) or not host_id:
        raise ValueError(f"{entity.semantic_id} requires host_void_semantic_id")
    start, end, z_min_um, z_max_um = _sidewall_span(entity, geometry_ref)
    matches: list[str] = []
    has_parent = False
    for solution in solution_region_entities(build_input):
        metadata = solution.metadata
        is_host_child = metadata.get("auto_vacuum_group_id") == host_id
        if is_host_child:
            has_parent = True
            entity_ids = metadata.get("auto_vacuum_subtracting_entity_ids")
            if isinstance(entity_ids, str | bytes) or not isinstance(
                entity_ids, Sequence
            ):
                raise TypeError(
                    f"auto-vacuum component {solution.semantic_id!r} lacks an entity subtractor ledger"
                )
            if entity.semantic_id not in entity_ids:
                continue
        elif solution.material_kind != "dielectric":
            continue
        solution_z_min_um = float(solution.geometry["z_min_um"])
        solution_z_max_um = float(solution.geometry["z_max_um"])
        if is_host_child:
            if not (
                same_coordinate_z(solution_z_min_um, z_min_um)
                and same_coordinate_z(solution_z_max_um, z_max_um)
            ):
                continue
        elif not range_within_bounds(
            z_min_um, z_max_um, solution_z_min_um, solution_z_max_um
        ):
            continue
        if _sidewall_segment_is_component_boundary(
            start, end, solution, holes_only=not is_host_child
        ):
            matches.append(solution.semantic_id)
    if not has_parent and (not matches):
        return (False, None)
    if len(matches) != 1:
        raise ValueError(
            f"{entity.semantic_id} sidewall segment {start!r}->{end!r} requires exactly one source dielectric cavity or auto-vacuum child for parent {host_id!r}; got {matches!r}."
        )
    return (True, matches[0])


def adjacent_auto_vacuum_sidewall_component_id(
    build_input: GeometryBuildInput,
    *,
    entity: SemanticEntitySpec,
    geometry_ref: Mapping[str, Any],
) -> str | None:
    """Resolve an exact auto-vacuum child across an authored host boundary."""
    points = geometry_ref.get("quad_points", ())
    if len(points) != 4:
        raise ValueError(f"{entity.semantic_id} sidewall requires four quad points")
    start = (float(points[0][0]), float(points[0][1]))
    end = (float(points[1][0]), float(points[1][1]))
    z_values = tuple((float(point[2]) for point in points))
    z_min_um, z_max_um = (min(z_values), max(z_values))
    matches: list[str] = []
    for solution in solution_region_entities(build_input):
        if not bool(solution.metadata.get("is_auto_vacuum_region")):
            continue
        entity_ids = solution.metadata.get("auto_vacuum_subtracting_entity_ids")
        if isinstance(entity_ids, str | bytes) or not isinstance(entity_ids, Sequence):
            raise TypeError(
                f"auto-vacuum component {solution.semantic_id!r} lacks an entity subtractor ledger"
            )
        if entity.semantic_id not in entity_ids:
            continue
        if not (
            same_coordinate_z(float(solution.geometry["z_min_um"]), z_min_um)
            and same_coordinate_z(float(solution.geometry["z_max_um"]), z_max_um)
            and _sidewall_segment_is_component_boundary(start, end, solution)
        ):
            continue
        matches.append(solution.semantic_id)
    if len(matches) > 1:
        raise ValueError(
            f"{entity.semantic_id} sidewall segment {start!r}->{end!r} has ambiguous auto-vacuum adjacency: {matches!r}."
        )
    return matches[0] if matches else None


def _sidewall_segment_is_component_boundary(
    start: tuple[float, float],
    end: tuple[float, float],
    solution: SemanticEntitySpec,
    *,
    holes_only: bool = False,
) -> bool:
    """Require complete canonical edge coverage, retaining outer/hole roles."""
    loops = tuple(solution.geometry.get("hole_loops", ()))
    if not holes_only:
        outer_loop = solution.geometry.get("outer_loop")
        if outer_loop is None:
            raise ValueError(
                f"solution component {solution.semantic_id!r} lacks outer_loop."
            )
        loops = (outer_loop, *loops)
    intervals = tuple(
        (
            interval
            for loop in loops
            for edge_start, edge_end in ring_edges(clean_loop(loop))
            if (interval := segment_overlap_interval(start, end, edge_start, edge_end))
            is not None
        )
    )
    return not interval_complement(intervals)


def _auto_vacuum_sidewall_is_exterior_only(
    build_input: GeometryBuildInput, entity: SemanticEntitySpec
) -> bool:
    """Whether every unmatched finite sidewall is an exact parent-envelope edge."""
    z_min, z_max = entity_z_range_um(entity)
    if z_max <= z_min:
        raise ValueError(f"{entity.semantic_id} has non-positive sidewall extent.")
    base_ref = route_entity_geometry_ref(
        build_input, "B", entity, representation="cutout_boundary_shell"
    )
    sidewalls = sidewall_geometry_refs(base_ref)
    if not sidewalls:
        raise ValueError(f"{entity.semantic_id} has no sidewall edge references.")
    for sidewall in sidewalls:
        if not sidewall_is_auto_vacuum_envelope_edge(
            build_input, entity=entity, geometry_ref=sidewall
        ):
            return False
    return True


def sidewall_is_auto_vacuum_envelope_edge(
    build_input: GeometryBuildInput,
    *,
    entity: SemanticEntitySpec,
    geometry_ref: Mapping[str, Any],
) -> bool:
    """Return whether an auto-envelope edge fully covers this sidewall."""
    host_id = entity.host_void_semantic_id
    envelopes = {
        tuple((tuple((float(value) for value in point)) for point in loop))
        for solution in solution_region_entities(build_input)
        if solution.metadata.get("auto_vacuum_group_id") == host_id
        for loop in (solution.metadata.get("auto_vacuum_envelope_outer_loop"),)
        if isinstance(loop, Sequence) and (not isinstance(loop, str | bytes))
    }
    if not envelopes:
        return False
    if len(envelopes) != 1:
        raise ValueError(
            f"{entity.semantic_id} auto-vacuum parent {host_id!r} lacks one envelope loop."
        )
    quad = geometry_ref.get("quad_points")
    if not isinstance(quad, Sequence) or len(quad) != 4:
        raise ValueError(f"{entity.semantic_id} sidewall lacks exact quad geometry.")
    start = (float(quad[0][0]), float(quad[0][1]))
    end = (float(quad[1][0]), float(quad[1][1]))
    envelope = next(iter(envelopes))
    return edge_is_covered_by_loop_edge(start, end, envelope)


def _component_entity_ids(
    solution: SemanticEntitySpec, relation: str, metadata: Mapping[str, Any]
) -> Sequence[str]:
    """Read the authored component ledger for one requested boundary relation."""
    if relation == "sidewall":
        entity_ids = metadata.get("auto_vacuum_subtracting_entity_ids")
        if isinstance(entity_ids, str | bytes) or not isinstance(entity_ids, Sequence):
            raise TypeError(
                f"auto-vacuum component {solution.semantic_id!r} lacks an entity subtractor ledger"
            )
    elif relation in {"bottom", "top"}:
        boundary_entity_ids = metadata.get("auto_vacuum_boundary_entity_ids")
        if not isinstance(boundary_entity_ids, Mapping):
            raise TypeError(
                f"auto-vacuum component {solution.semantic_id!r} lacks a boundary entity ledger"
            )
        boundary_key = "top" if relation == "bottom" else "bottom"
        entity_ids = boundary_entity_ids.get(boundary_key)
        if isinstance(entity_ids, str | bytes) or not isinstance(entity_ids, Sequence):
            raise TypeError(
                f"auto-vacuum component {solution.semantic_id!r} has invalid {boundary_key!r} boundary ledger"
            )
    else:
        raise ValueError(f"unsupported auto-vacuum face relation {relation!r}")
    return entity_ids


def _sidewall_span(
    entity: SemanticEntitySpec, geometry_ref: Mapping[str, Any]
) -> tuple[tuple[float, float], tuple[float, float], float, float]:
    """Read and validate the complete sidewall segment and its Z interval."""
    points = geometry_ref.get("quad_points", ())
    if len(points) != 4:
        raise ValueError(f"{entity.semantic_id} sidewall requires four quad points")
    start = (float(points[0][0]), float(points[0][1]))
    end = (float(points[1][0]), float(points[1][1]))
    if segment_overlap_interval(start, end, start, end) is None:
        raise ValueError(f"{entity.semantic_id} sidewall has degenerate XY segment")
    z_values = tuple((float(point[2]) for point in points))
    z_min_um, z_max_um = (min(z_values), max(z_values))
    if not z_max_um > z_min_um:
        raise ValueError(f"{entity.semantic_id} sidewall has non-positive z extent")
    return (start, end, z_min_um, z_max_um)
