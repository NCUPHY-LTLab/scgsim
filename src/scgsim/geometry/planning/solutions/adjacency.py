"""Decide which solution entity (dielectric or vacuum) touches a conductor face or sidewall.

Owns adjacency lookups only. The answers feed interface ids, boundary volume ids and
physical-group owners, so they must come from source geometry and never from native
coordinate guesses.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.models.topology import SurfacePlanRecord
from scgsim.geometry.planning.solutions.auto_vacuum import auto_vacuum_sheet_z
from scgsim.geometry.planning.solutions.occupancy import (
    entity_occupied_region,
    list_solution_entities,
    solution_bounds,
    solution_entity_xy_region,
)
from scgsim.geometry.primitives.entities import (
    entity_by_id,
    entity_z_range_um,
    unique_ids,
)
from scgsim.geometry.primitives.entity_validation import (
    is_solution_entity,
    is_vacuum_solution_entity,
)
from scgsim.geometry.primitives.geometry_refs import (
    route_entity_geometry_ref,
    sidewall_geometry_refs,
)
from scgsim.geometry.primitives.loops import (
    boolean_gdstk_region,
    edge_is_covered_by_loop_edge,
    loop_centroid,
    normalized_loop,
    ring_edges,
)
from scgsim.geometry.primitives.spatial import (
    bounds_contains_point,
    geometry_ref_surface_z_um,
    interval_complement,
    is_same_z,
    range_within_bounds,
    segment_overlap_interval,
)
from scgsim.geometry.primitives.surface_records import surface_boundary_volume_ids
from scgsim.semantics import SemanticEvidenceFacade, conductor_solution_interface_kind


def conductor_face_adjacent_solution_id(
    build_input: GeometryBuildInput,
    entity: SemanticEntitySpec,
    face: str,
) -> str:
    """Id of the solution entity adjacent to a conductor's bottom or top face."""
    z_min_um, z_max_um = entity_z_range_um(entity)
    face_z_um = z_min_um if face == "bottom" else z_max_um
    auto_parent, auto_component = _auto_vacuum_host_component_id(
        build_input,
        entity=entity,
        relation=face,
        z_um=face_z_um,
    )
    if auto_component is not None:
        return auto_component
    if auto_parent:
        non_auto_solution = _solution_id_for_entity_coverage(
            build_input,
            entity,
            z_um=face_z_um,
            mode=face,
            include_auto_vacuum=False,
        )
        if non_auto_solution is not None:
            return non_auto_solution
        adjacent_solution = _solution_id_for_entity_coverage(
            build_input,
            entity,
            z_um=face_z_um,
            mode=face,
        )
        if (
            adjacent_solution is not None
            and entity_by_id(build_input, adjacent_solution).metadata.get(
                "auto_vacuum_group_id"
            )
            == entity.host_void_semantic_id
        ):
            return adjacent_solution
        # A Route-A sheet may retain its physical film thickness in source
        # coordinates while its solution side lies inside one generated vacuum
        # component. Require full polygonal coverage by that exact component.
        containing_solution = _solution_id_for_entity_coverage(
            build_input,
            entity,
            z_um=face_z_um,
            mode="containing",
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
    exact = _solution_id_for_entity_coverage(
        build_input,
        entity,
        z_um=face_z_um,
        mode=face,
    )
    if exact is not None:
        return exact
    containing = _solution_id_for_entity_coverage(
        build_input,
        entity,
        z_um=face_z_um,
        mode="containing",
    )
    if containing is not None:
        return containing
    raise ValueError(f"{entity.semantic_id} {face} face has no adjacent solution")


def conductor_sidewall_adjacent_solution_id(
    build_input: GeometryBuildInput,
    entity: SemanticEntitySpec,
) -> str | None:
    """Id of the solution beside a conductor sidewall, or None when exterior-only."""
    z_min_um, z_max_um = entity_z_range_um(entity)
    mid_z_um = (z_min_um + z_max_um) / 2.0
    auto_parent, auto_component = _auto_vacuum_host_component_id(
        build_input,
        entity=entity,
        relation="sidewall",
        z_um=mid_z_um,
    )
    if auto_component is not None:
        return auto_component
    if auto_parent:
        if _auto_vacuum_sidewall_is_exterior_only(build_input, entity):
            return None
        raise ValueError(
            f"{entity.semantic_id} sidewall has no local auto-vacuum component."
        )
    solution_id = _solution_id_for_entity_coverage(
        build_input,
        entity,
        z_um=mid_z_um,
        mode="containing",
    )
    if solution_id is not None:
        return solution_id
    raise ValueError(f"{entity.semantic_id} sidewall has no adjacent solution")


def _solution_id_for_entity_coverage(
    build_input: GeometryBuildInput,
    entity: SemanticEntitySpec,
    *,
    z_um: float,
    mode: str,
    include_auto_vacuum: bool = True,
) -> str | None:
    """Resolve one exact solution cover for a structured conductor footprint.

    This is topology coverage, not a centroid or bounds proxy: every occupied
    polygon point must be inside one candidate solution region at the declared
    z relation. Disjoint covers therefore fail instead of assigning one part
    of a conductor to an arbitrary solution volume.
    """
    import gdstk

    occupied_region = entity_occupied_region(gdstk, entity, build_input=build_input)
    if not occupied_region:
        raise ValueError(f"{entity.semantic_id} has no occupied geometry region.")
    candidates: list[str] = []
    for solution in list_solution_entities(build_input):
        if not include_auto_vacuum and bool(
            solution.metadata.get("is_auto_vacuum_region")
        ):
            continue
        if mode == "bottom":
            matches = is_same_z(float(solution.geometry["z_max_um"]), z_um)
        elif mode == "top":
            matches = is_same_z(float(solution.geometry["z_min_um"]), z_um)
        elif mode == "containing":
            matches = (
                float(solution.geometry["z_min_um"])
                < z_um
                < float(solution.geometry["z_max_um"])
            )
        else:
            raise ValueError(f"unsupported solution adjacency mode {mode!r}")
        if not matches:
            continue
        uncovered_region = boolean_gdstk_region(
            gdstk,
            occupied_region,
            solution_entity_xy_region(gdstk, solution),
            "not",
        )
        if not uncovered_region:
            candidates.append(solution.semantic_id)
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ValueError(
            f"{entity.semantic_id} {mode} footprint has ambiguous exact solution coverage: {candidates!r}"
        )
    return candidates[0]


def _auto_vacuum_host_component_id(
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
    for solution in list_solution_entities(build_input):
        if solution.metadata.get("auto_vacuum_group_id") != host_id:
            continue
        has_parent = True
        entity_ids = _auto_vacuum_ledger_entity_ids(solution, relation)
        if entity.semantic_id not in entity_ids:
            continue
        if _component_meets_relation(solution, relation, z_um):
            components.append(solution.semantic_id)
    if not has_parent:
        return False, None
    if len(components) > 1:
        raise ValueError(
            f"{entity.semantic_id} {relation} requires exactly one structured "
            f"auto-vacuum component for parent {host_id!r}; got {components!r}."
        )
    return True, components[0] if components else None


def _auto_vacuum_ledger_entity_ids(
    solution: SemanticEntitySpec, relation: str
) -> Sequence[Any]:
    """Return the obstacle ids an auto-vacuum component recorded for this face relation.

    A sidewall uses the component's subtractor ledger. A bottom face of the conductor
    touches the component's top boundary, and a top face its bottom boundary.
    """
    metadata = solution.metadata
    if relation == "sidewall":
        entity_ids = metadata.get("auto_vacuum_subtracting_entity_ids")
        if isinstance(entity_ids, str | bytes) or not isinstance(entity_ids, Sequence):
            raise TypeError(
                f"auto-vacuum component {solution.semantic_id!r} lacks an entity subtractor ledger"
            )
        return entity_ids
    if relation in {"bottom", "top"}:
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
        return entity_ids
    raise ValueError(f"unsupported auto-vacuum face relation {relation!r}")


def _component_meets_relation(
    solution: SemanticEntitySpec, relation: str, z_um: float
) -> bool:
    """Whether the component's z range reaches ``z_um`` the way the relation needs."""
    z_min = float(solution.geometry["z_min_um"])
    z_max = float(solution.geometry["z_max_um"])
    if relation == "sidewall":
        return z_min < z_um < z_max
    if relation == "bottom":
        return is_same_z(z_max, z_um)
    return is_same_z(z_min, z_um)


def _auto_vacuum_sidewall_is_exterior_only(
    build_input: GeometryBuildInput,
    entity: SemanticEntitySpec,
) -> bool:
    """Whether every unmatched finite sidewall is an exact parent-envelope edge."""
    z_min, z_max = entity_z_range_um(entity)
    if z_max <= z_min:
        raise ValueError(f"{entity.semantic_id} has non-positive sidewall extent.")
    base_ref = route_entity_geometry_ref(
        build_input,
        "B",
        entity,
        representation="cutout_boundary_shell",
    )
    sidewalls = sidewall_geometry_refs(base_ref)
    if not sidewalls:
        raise ValueError(f"{entity.semantic_id} has no sidewall edge references.")
    for sidewall in sidewalls:
        if not sidewall_is_auto_vacuum_envelope_edge(
            build_input,
            entity=entity,
            geometry_ref=sidewall,
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
        tuple(tuple(float(value) for value in point) for point in loop)
        for solution in list_solution_entities(build_input)
        if solution.metadata.get("auto_vacuum_group_id") == host_id
        for loop in (solution.metadata.get("auto_vacuum_envelope_outer_loop"),)
        if isinstance(loop, Sequence) and not isinstance(loop, str | bytes)
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
    points = geometry_ref.get("quad_points", ())
    if len(points) != 4:
        raise ValueError(f"{entity.semantic_id} sidewall requires four quad points")
    start = (float(points[0][0]), float(points[0][1]))
    end = (float(points[1][0]), float(points[1][1]))
    if segment_overlap_interval(start, end, start, end) is None:
        raise ValueError(f"{entity.semantic_id} sidewall has degenerate XY segment")
    z_values = tuple(float(point[2]) for point in points)
    z_min_um, z_max_um = min(z_values), max(z_values)
    if not z_max_um > z_min_um:
        raise ValueError(f"{entity.semantic_id} sidewall has non-positive z extent")
    segment = _SidewallSegment(start, end, z_min_um, z_max_um)
    matches: list[str] = []
    has_parent = False
    for solution in list_solution_entities(build_input):
        is_host_child = solution.metadata.get("auto_vacuum_group_id") == host_id
        if is_host_child:
            has_parent = True
        if _sidewall_meets_solution(
            solution, is_host_child=is_host_child, entity=entity, segment=segment
        ):
            matches.append(solution.semantic_id)
    if not has_parent and not matches:
        return False, None
    if len(matches) != 1:
        raise ValueError(
            f"{entity.semantic_id} sidewall segment {start!r}->{end!r} requires "
            f"exactly one source dielectric cavity or auto-vacuum child "
            f"for parent {host_id!r}; got {matches!r}."
        )
    return True, matches[0]


class _SidewallSegment(NamedTuple):
    """One conductor sidewall: its xy end points and z extent."""

    start: tuple[float, float]
    end: tuple[float, float]
    z_min_um: float
    z_max_um: float


def _sidewall_meets_solution(
    solution: SemanticEntitySpec,
    *,
    is_host_child: bool,
    entity: SemanticEntitySpec,
    segment: _SidewallSegment,
) -> bool:
    """Whether the segment lies on this solution's boundary, as a vacuum child or dielectric.

    A vacuum child of the conductor's host must list the conductor as a subtractor and
    span exactly the wall's z range. An authored dielectric must contain the wall's z
    range and have the segment on a hole boundary.
    """
    if is_host_child:
        entity_ids = solution.metadata.get("auto_vacuum_subtracting_entity_ids")
        if isinstance(entity_ids, str | bytes) or not isinstance(entity_ids, Sequence):
            raise TypeError(
                f"auto-vacuum component {solution.semantic_id!r} lacks an entity subtractor ledger"
            )
        if entity.semantic_id not in entity_ids:
            return False
    elif solution.material_kind != "dielectric":
        return False
    solution_z_min_um = float(solution.geometry["z_min_um"])
    solution_z_max_um = float(solution.geometry["z_max_um"])
    if is_host_child:
        if not (
            is_same_z(solution_z_min_um, segment.z_min_um)
            and is_same_z(solution_z_max_um, segment.z_max_um)
        ):
            return False
    elif not range_within_bounds(
        segment.z_min_um, segment.z_max_um, solution_z_min_um, solution_z_max_um
    ):
        return False
    return _sidewall_segment_is_component_boundary(
        segment.start, segment.end, solution, holes_only=not is_host_child
    )


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
    z_values = tuple(float(point[2]) for point in points)
    z_min_um, z_max_um = min(z_values), max(z_values)
    matches: list[str] = []
    for solution in list_solution_entities(build_input):
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
            is_same_z(float(solution.geometry["z_min_um"]), z_min_um)
            and is_same_z(float(solution.geometry["z_max_um"]), z_max_um)
            and _sidewall_segment_is_component_boundary(start, end, solution)
        ):
            continue
        matches.append(solution.semantic_id)
    if len(matches) > 1:
        raise ValueError(
            f"{entity.semantic_id} sidewall segment {start!r}->{end!r} has "
            f"ambiguous auto-vacuum adjacency: {matches!r}."
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
        interval
        for loop in loops
        for edge_start, edge_end in ring_edges(normalized_loop(loop))
        if (interval := segment_overlap_interval(start, end, edge_start, edge_end))
        is not None
    )
    return not interval_complement(intervals)


def component_is_boundary_attached(
    build_input: GeometryBuildInput,
    *,
    solution: SemanticEntitySpec,
    surface: SurfacePlanRecord,
) -> bool:
    """Whether this PEC surface is on a solution-domain boundary plane.

    Boundary-attached Route-A conductors form part of a solution exterior
    shell; only a component wholly internal to a solution becomes an inner
    PEC void shell.
    """
    if surface.surface_role not in {
        "cutout_boundary_shell",
        "route_a_sheet_contact_cap",
    }:
        return False
    if "quad_points" in surface.geometry_ref:
        z_values = tuple(
            float(point[2]) for point in surface.geometry_ref["quad_points"]
        )
        return any(
            is_same_z(z_um, float(solution.geometry["z_min_um"]))
            or is_same_z(z_um, float(solution.geometry["z_max_um"]))
            for z_um in z_values
        )
    if solution.semantic_id not in surface_boundary_volume_ids(
        surface,
        known_entity_ids={entity.semantic_id for entity in build_input.entities},
    ):
        return False
    z_min = float(solution.geometry["z_min_um"])
    z_max = float(solution.geometry["z_max_um"])
    try:
        z_um = geometry_ref_surface_z_um(surface.geometry_ref)
    except (KeyError, TypeError, ValueError):
        return False
    return is_same_z(z_um, z_min) or is_same_z(z_um, z_max)


def required_host_solution_id(
    build_input: GeometryBuildInput, entity: SemanticEntitySpec
) -> str:
    """Return one authored conductor host, never an inferred global vacuum."""
    host_id = entity.host_void_semantic_id
    if not isinstance(host_id, str) or not host_id:
        raise ValueError(f"{entity.semantic_id} requires host_void_semantic_id")
    host = entity_by_id(build_input, host_id)
    if not is_solution_entity(host):
        raise ValueError(
            f"{entity.semantic_id} host_void_semantic_id {host_id!r} is not a solution_region"
        )
    return host_id


def classify_conductor_solution_interface(
    solution: SemanticEntitySpec,
    *,
    semantic_facts: SemanticEvidenceFacade | None = None,
) -> str:
    """Interface kind of a solution entity, from its material kind."""
    material_kind = (
        semantic_facts.material_kind(solution.semantic_id)
        if semantic_facts is not None
        else solution.material_kind
    )
    if not isinstance(material_kind, str):
        raise ValueError(f"{solution.semantic_id} has no snapshotted material kind")
    return conductor_solution_interface_kind(material_kind)


def route_a_sheet_boundary_volume_ids(
    build_input: GeometryBuildInput,
    entity: SemanticEntitySpec,
) -> tuple[str, ...]:
    """Unique ids of the solutions adjacent to a Route-A sheet's bottom and top."""
    return unique_ids(
        (
            conductor_face_adjacent_solution_id(build_input, entity, "bottom"),
            conductor_face_adjacent_solution_id(build_input, entity, "top"),
        )
    )


def route_a_sheet_plane_z_um(
    build_input: GeometryBuildInput,
    entity: SemanticEntitySpec,
    *,
    representation_context: RepresentationContext | None = None,
) -> float:
    """Z of a Route-A sheet plane in micrometres, taken from a non-vacuum side."""
    if any(
        solution.metadata.get("auto_vacuum_group_id") == entity.host_void_semantic_id
        for solution in list_solution_entities(build_input)
    ):
        return auto_vacuum_sheet_z(
            build_input, entity, representation_context=representation_context
        )
    z_min_um, z_max_um = entity_z_range_um(entity)
    for boundary_id, z_um in (
        (
            conductor_face_adjacent_solution_id(build_input, entity, "bottom"),
            z_min_um,
        ),
        (
            conductor_face_adjacent_solution_id(build_input, entity, "top"),
            z_max_um,
        ),
    ):
        if not is_vacuum_solution_entity(entity_by_id(build_input, boundary_id)):
            return z_um
    return z_min_um


def route_a_sheet_plane_z_um_from_solutions(
    entity: SemanticEntitySpec,
    solution_entities: Sequence[SemanticEntitySpec],
) -> float:
    """Z in micrometres of a Route-A sheet plane, from the given solution entities."""
    z_min_um, z_max_um = entity_z_range_um(entity)
    point = loop_centroid(entity.geometry["outer_loop"])
    for face_z_um, solution_edge_key in (
        (z_min_um, "z_max_um"),
        (z_max_um, "z_min_um"),
    ):
        for solution in solution_entities:
            if not bounds_contains_point(solution_bounds(solution), point):
                continue
            if not is_same_z(float(solution.geometry[solution_edge_key]), face_z_um):
                continue
            if not is_vacuum_solution_entity(solution):
                return face_z_um
    return z_min_um


def route_a_sheet_boundary_volume_ids_from_solutions(
    entity: SemanticEntitySpec,
    solution_entities: Sequence[SemanticEntitySpec],
) -> tuple[str, ...]:
    """Unique ids of solutions touching a sheet's bottom and top at its centroid."""
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
            if is_same_z(float(solution.geometry[solution_edge_key]), face_z_um):
                ids.append(solution.semantic_id)
                break
    return unique_ids(ids)


def sidewall_on_solution_outer_boundary(
    geometry_ref: Mapping[str, Any],
    solution: SemanticEntitySpec,
) -> bool:
    """True when a sidewall quad's first edge lies along the solution's XY boundary."""
    points = geometry_ref.get("quad_points", ())
    if len(points) != 4:
        return False
    start = (float(points[0][0]), float(points[0][1]))
    end = (float(points[1][0]), float(points[1][1]))
    bounds = solution_bounds(solution)
    x_min = float(bounds["x_min_um"])
    x_max = float(bounds["x_max_um"])
    y_min = float(bounds["y_min_um"])
    y_max = float(bounds["y_max_um"])
    if is_same_z(start[0], x_min) and is_same_z(end[0], x_min):
        return range_within_bounds(start[1], end[1], y_min, y_max)
    if is_same_z(start[0], x_max) and is_same_z(end[0], x_max):
        return range_within_bounds(start[1], end[1], y_min, y_max)
    if is_same_z(start[1], y_min) and is_same_z(end[1], y_min):
        return range_within_bounds(start[0], end[0], x_min, x_max)
    if is_same_z(start[1], y_max) and is_same_z(end[1], y_max):
        return range_within_bounds(start[0], end[0], x_min, x_max)
    return False
