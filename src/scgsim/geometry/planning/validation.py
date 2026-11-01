"""Invariants of the compiler's plan records, checked before and after OCC lowering.

Owns: route, surface, curve, loop, volume and tag-plan checks. Each `validate_*`
function collects its errors in a fixed order and raises one `ValueError`.

Does not own: input-only validation (`source/validation.py`), the horizontal
surface-overlap check (`surface_overlap.py`) or the point lookup used for the
curve check (`point_index.py`). No numerical policy lives here.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from typing import Any

from scgsim.geometry.models.common import ROUTE_ALLOWED_REPRESENTATIONS, RouteLiteral
from scgsim.geometry.models.construction import (
    ConstructionBodyPlanRecord,
    CutHostOperationRecord,
    SurfacePartitionRecord,
)
from scgsim.geometry.models.input import GeometryBuildInput
from scgsim.geometry.models.tags import BackendEntityTagRecord, TagPlanRecord
from scgsim.geometry.models.topology import (
    CurvePlanRecord,
    InterfacePlanRecord,
    PointPlanRecord,
    SurfaceLoopRecord,
    SurfacePlanRecord,
    VolumePlanRecord,
)
from scgsim.geometry.planning.context.representation import (
    planned_conductor_representation,
)
from scgsim.geometry.planning.point_index import (
    axis_aligned_point_index,
    build_point_axis_index,
    coordinate_signature,
    point_on_segment_interior,
    segment_candidate_point_ids,
)
from scgsim.geometry.primitives.entity_validation import (
    is_solution_entity,
    requires_route_representation,
)

_INTERFACE_KINDS = {"MM", "SS", "AA", "MS", "MA", "SA"}


def validate_selected_route(
    build_input: GeometryBuildInput,
    route: RouteLiteral,
    *,
    representation_context=None,
) -> None:
    """Validate that every conductor has an explicit representation for route."""
    if representation_context is None:
        raise ValueError(
            "current compilation requires interpret_physical_geometry() and plan.context"
        )

    allowed = ROUTE_ALLOWED_REPRESENTATIONS.get(route)
    if allowed is None:
        raise ValueError(f"Unsupported route: {route}")

    errors: list[str] = []
    if route == "C" and build_input.port_sheet_regions:
        errors.append("Route C does not support Palace lumped-port sheets")
    for entity in build_input.entities:
        if not requires_route_representation(entity):
            continue
        representation = planned_conductor_representation(
            entity, route=route, context=representation_context
        )
        if representation is None:
            errors.append(f"{entity.semantic_id} does not define Route {route}")
        elif representation not in allowed:
            errors.append(
                f"{entity.semantic_id} Route {route} representation "
                f"{representation!r} is not supported"
            )
        if route == "C" and entity.geometry.get("route_ab_fused_selector_mode"):
            errors.append(
                f"{entity.semantic_id} Route C cannot use route_ab_fused_selector_mode"
            )
    if errors:
        raise ValueError("Invalid selected route: " + "; ".join(errors))


def validate_route_operation_coverage(
    *,
    construction_bodies: tuple[ConstructionBodyPlanRecord, ...],
    cut_operations: tuple[CutHostOperationRecord, ...],
    surfaces: tuple[SurfacePlanRecord, ...],
) -> None:
    """Validate Route A/B construction-body, cut, and shell-surface linkage."""
    surface_ids = {surface.surface_id for surface in surfaces}
    covered_surface_ids = {
        *surface_ids,
        *(
            surface.parent_surface_id
            for surface in surfaces
            if surface.parent_surface_id is not None
        ),
    }
    body_ids = {body.construction_body_id for body in construction_bodies}
    operation_body_ids = {
        construction_body_id
        for operation in cut_operations
        for construction_body_id in operation.construction_body_ids
    }
    operation_surface_ids = {
        surface_id
        for operation in cut_operations
        for surface_id in operation.exposed_surface_ids
    }

    errors: list[str] = []
    for body in construction_bodies:
        missing_surfaces = [
            surface_id
            for surface_id in body.expected_surface_ids
            if surface_id not in covered_surface_ids
        ]
        if missing_surfaces:
            errors.append(
                f"{body.construction_body_id} expects missing shell surfaces "
                f"{missing_surfaces!r}"
            )

    missing_operation_bodies = sorted(body_ids - operation_body_ids)
    if missing_operation_bodies:
        errors.append(
            f"construction bodies without cut operations {missing_operation_bodies!r}"
        )

    missing_operation_surfaces = sorted(operation_surface_ids - covered_surface_ids)
    if missing_operation_surfaces:
        errors.append(
            f"cut operations expose missing surfaces {missing_operation_surfaces!r}"
        )

    if errors:
        raise ValueError("Invalid route operation plan: " + "; ".join(errors))


def validate_surface_sheet_interface_coverage(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    surfaces: tuple[SurfacePlanRecord, ...],
) -> None:
    """Fail if Route A sheet conductors are not represented by interfaces."""
    sheet_ids = {
        entity.semantic_id
        for entity in build_input.entities
        if not is_solution_entity(entity)
        and entity.route_representations.get(route) == "surface_sheet"
    }
    if not sheet_ids:
        return

    covered_ids: set[str] = set()
    for surface in surfaces:
        if surface.interface_id is None:
            continue
        owner_ids = surface.metadata.get(
            "owner_semantic_ids",
            (surface.owner_semantic_id,),
        )
        if isinstance(owner_ids, str):
            owner_ids = (owner_ids,)
        covered_ids.update(str(owner_id) for owner_id in owner_ids)

    missing = sorted(sheet_ids - covered_ids)
    if missing:
        raise ValueError(
            "surface_sheet entities require planned interface surface coverage: "
            f"{missing!r}"
        )


def validate_route_volume_surface_refs(
    *,
    route: RouteLiteral,
    volumes: tuple[VolumePlanRecord, ...],
) -> None:
    """Fail fast when backend-live volumes have no planned boundaries."""
    missing = sorted(
        volume.volume_id
        for volume in volumes
        if not volume.construction_only and not volume.surface_refs
    )
    ambiguous_solution_shells = sorted(
        volume.volume_id
        for volume in volumes
        if route in {"A", "B"}
        and not volume.construction_only
        and volume.metadata.get("representation") == "solution_volume"
        and not getattr(volume, "exterior_surface_refs", ())
    )
    if missing:
        raise ValueError(
            f"Backend-live volumes require planned boundary surfaces: {missing!r}"
        )
    if ambiguous_solution_shells:
        raise ValueError(
            "solution volumes require explicit exterior shell refs: "
            f"{ambiguous_solution_shells!r}"
        )


def validate_curve_plan_coverage(
    *,
    points: tuple[PointPlanRecord, ...],
    curves: tuple[CurvePlanRecord, ...],
    surface_loops: tuple[SurfaceLoopRecord, ...],
    surfaces: tuple[SurfacePlanRecord, ...],
) -> None:
    """Validate compiler-owned point, curve, and surface-loop topology.

    Each `SurfaceLoopRecord` must be one simple directed cycle: holes are
    separate loops, and shared curves are legal across different surfaces or
    loops but may not repeat inside one loop. This is the pre-OCC contract that
    keeps self-touching boolean output from reaching backend curve-loop
    lowering.

    Errors from every step are collected, in step order, into one message.
    """
    errors: list[str] = []
    if any(not surface.construction_only for surface in surfaces) and (
        not points or not curves or not surface_loops
    ):
        errors.append(
            "backend-live surfaces require canonical points, curves, and loops"
        )

    points_by_id = _check_points(points, errors)
    _check_curves(curves, set(points_by_id), errors)
    _check_no_points_inside_line_segments(curves, points_by_id, errors)
    curves_by_id = {curve.curve_id: curve for curve in curves}
    for loop in surface_loops:
        _check_surface_loop(loop, curves_by_id, errors)
    _check_surface_loop_refs(surfaces, {loop.loop_id for loop in surface_loops}, errors)

    if errors:
        raise ValueError("Invalid curve plan coverage: " + "; ".join(errors))


def _check_points(
    points: tuple[PointPlanRecord, ...], errors: list[str]
) -> dict[str, PointPlanRecord]:
    """Point ids and coordinates must be unique. Returns the points by id."""
    point_ids: set[str] = set()
    point_signatures: dict[tuple[float, float, float], str] = {}
    points_by_id: dict[str, PointPlanRecord] = {}
    for point in points:
        if point.point_id in point_ids:
            errors.append(f"duplicate point_id: {point.point_id}")
        point_ids.add(point.point_id)
        points_by_id[point.point_id] = point
        signature = coordinate_signature(point.coordinate)
        existing = point_signatures.get(signature)
        if existing is not None:
            errors.append(
                f"duplicate point coordinate: {existing} and {point.point_id}"
            )
        point_signatures[signature] = point.point_id
    return points_by_id


def _check_curves(
    curves: tuple[CurvePlanRecord, ...], point_ids: set[str], errors: list[str]
) -> None:
    """Curve ids and geometry must be unique and their end points must exist."""
    curve_ids: set[str] = set()
    curve_signatures: dict[tuple[str, str], str] = {}
    for curve in curves:
        if curve.curve_id in curve_ids:
            errors.append(f"duplicate curve_id: {curve.curve_id}")
        curve_ids.add(curve.curve_id)
        if curve.start_point_id not in point_ids:
            errors.append(
                f"{curve.curve_id} references unknown point {curve.start_point_id}"
            )
        if curve.end_point_id not in point_ids:
            errors.append(
                f"{curve.curve_id} references unknown point {curve.end_point_id}"
            )
        if (
            curve.start_point_id == curve.end_point_id
            and curve.curve_kind == "line_segment"
        ):
            errors.append(f"{curve.curve_id} has identical start/end points")
            continue
        # A line is the same line in either direction; other curves are compared
        # by kind, geometry and parameter interval.
        signature = (
            tuple(sorted((curve.start_point_id, curve.end_point_id)))
            if curve.curve_kind == "line_segment"
            else (
                curve.curve_kind,
                json.dumps(curve.geometry, sort_keys=True),
                curve.parameter_interval,
            )
        )
        existing = curve_signatures.get(signature)
        if existing is not None:
            errors.append(f"duplicate curve geometry: {existing} and {curve.curve_id}")
        curve_signatures[signature] = curve.curve_id


def _check_no_points_inside_line_segments(
    curves: tuple[CurvePlanRecord, ...],
    points_by_id: Mapping[str, PointPlanRecord],
    errors: list[str],
) -> None:
    """A planned point on a line's interior would leave the mesh non-conformal."""
    point_axis_index = build_point_axis_index(points_by_id)
    point_line_index = axis_aligned_point_index(points_by_id)
    for curve in curves:
        if curve.curve_kind != "line_segment":
            continue
        start = points_by_id.get(curve.start_point_id)
        end = points_by_id.get(curve.end_point_id)
        if start is None or end is None:
            continue
        interior_points = [
            point_id
            for point_id in segment_candidate_point_ids(
                start.coordinate,
                end.coordinate,
                point_axis_index,
                point_line_index,
            )
            if point_id not in {curve.start_point_id, curve.end_point_id}
            and point_on_segment_interior(
                points_by_id[point_id].coordinate,
                start.coordinate,
                end.coordinate,
            )
        ]
        if interior_points:
            errors.append(
                f"{curve.curve_id} contains planned points in its interior: "
                f"{interior_points!r}"
            )


def _check_surface_loop(
    loop: SurfaceLoopRecord,
    curves_by_id: Mapping[str, CurvePlanRecord],
    errors: list[str],
) -> None:
    """One loop must be a closed simple cycle of known, signed curves."""
    if len(loop.curve_refs) < 3 and not any(
        curves_by_id.get(ref.curve_id) is not None
        and curves_by_id[ref.curve_id].curve_kind != "line_segment"
        for ref in loop.curve_refs
    ):
        errors.append(f"{loop.loop_id} requires at least three straight curves")
        return
    repeated_curve_refs = sorted(
        curve_id
        for curve_id, count in Counter(
            curve_ref.curve_id for curve_ref in loop.curve_refs
        ).items()
        if count > 1
    )
    if repeated_curve_refs:
        errors.append(
            f"{loop.loop_id} repeats curve refs inside one surface loop: "
            f"{repeated_curve_refs!r}"
        )
    directed_edges = _directed_loop_edges(loop, curves_by_id, errors)
    for index, (_, end_point_id) in enumerate(directed_edges):
        next_start_id, _ = directed_edges[(index + 1) % len(directed_edges)]
        if end_point_id != next_start_id:
            errors.append(f"{loop.loop_id} curve refs do not form a closed loop")
            break
    if len(directed_edges) == len(loop.curve_refs):
        # In a simple cycle every point is entered once and left once.
        point_use_counts: Counter[str] = Counter()
        for start_point_id, end_point_id in directed_edges:
            point_use_counts.update((start_point_id, end_point_id))
        bad_point_use_counts = {
            point_id: count
            for point_id, count in point_use_counts.items()
            if count != 2
        }
        if bad_point_use_counts:
            errors.append(
                f"{loop.loop_id} is not a simple surface loop; point use "
                f"counts {bad_point_use_counts!r}"
            )


def _directed_loop_edges(
    loop: SurfaceLoopRecord,
    curves_by_id: Mapping[str, CurvePlanRecord],
    errors: list[str],
) -> list[tuple[str, str]]:
    """(start, end) point ids of the loop's curves, following each curve's orientation."""
    directed_edges: list[tuple[str, str]] = []
    for curve_ref in loop.curve_refs:
        curve = curves_by_id.get(curve_ref.curve_id)
        if curve is None:
            errors.append(
                f"{loop.loop_id} references unknown curve {curve_ref.curve_id}"
            )
            continue
        if curve_ref.orientation not in {-1, 1}:
            errors.append(
                f"{loop.loop_id} curve {curve_ref.curve_id} has invalid "
                f"orientation {curve_ref.orientation!r}"
            )
            continue
        if curve_ref.orientation == 1:
            directed_edges.append((curve.start_point_id, curve.end_point_id))
        else:
            directed_edges.append((curve.end_point_id, curve.start_point_id))
    return directed_edges


def _check_surface_loop_refs(
    surfaces: tuple[SurfacePlanRecord, ...], loop_ids: set[str], errors: list[str]
) -> None:
    """Every backend-live surface needs an outer loop and only known loop refs."""
    for surface in surfaces:
        if surface.construction_only:
            continue
        if surface.outer_loop_ref is None:
            errors.append(f"{surface.surface_id} lacks outer_loop_ref")
        refs = (
            *((surface.outer_loop_ref,) if surface.outer_loop_ref is not None else ()),
            *surface.hole_loop_refs,
        )
        for loop_ref in refs:
            if loop_ref not in loop_ids:
                errors.append(
                    f"{surface.surface_id} references unknown loop {loop_ref}"
                )


def validate_interface_surface_source_of_truth(
    *,
    interfaces: tuple[InterfacePlanRecord, ...],
    surfaces: tuple[SurfacePlanRecord, ...],
) -> None:
    """Require every solver-relevant interface surface to come from InterfacePlan."""
    interface_ids = {interface.interface_id for interface in interfaces}
    errors: list[str] = []
    for surface in surfaces:
        if surface.construction_only:
            continue
        interface_kinds = _surface_interface_kinds(surface)
        if len(interface_kinds) > 2:
            errors.append(
                f"{surface.surface_id} has too many interface kinds {interface_kinds!r}"
            )
        if interface_kinds and surface.interface_id is None:
            errors.append(
                f"{surface.surface_id} has interface kinds "
                f"{interface_kinds!r} without InterfacePlan source"
            )
        if (
            surface.interface_id is not None
            and surface.interface_id not in interface_ids
        ):
            errors.append(
                f"{surface.surface_id} references unknown interface "
                f"{surface.interface_id}"
            )
    if errors:
        raise ValueError("Invalid interface source of truth: " + "; ".join(errors))


def validate_surface_deduplication(
    *,
    surfaces: tuple[SurfacePlanRecord, ...],
) -> None:
    """Reject two backend-live surfaces with the same canonical signature."""
    signatures: dict[str, str] = {}
    errors: list[str] = []
    for surface in surfaces:
        if surface.construction_only:
            continue
        signature = _surface_signature(surface)
        existing = signatures.get(signature)
        if existing is not None and existing != surface.surface_id:
            errors.append(
                f"duplicate live surface geometry: {existing} and {surface.surface_id}"
            )
        signatures[signature] = surface.surface_id
    if errors:
        raise ValueError("Invalid surface deduplication: " + "; ".join(errors))


def validate_volume_surface_closure(
    *,
    volumes: tuple[VolumePlanRecord, ...],
    surfaces: tuple[SurfacePlanRecord, ...],
    surface_loops: tuple[SurfaceLoopRecord, ...],
) -> None:
    """Validate volume surface references before backend lowering.

    Full closed-shell validation belongs to the canonical curve/loop plan. This
    guard rejects missing references and then checks curve incidence from the
    planned loops, before OCC is allowed to construct a volume.
    """
    surfaces_by_id = {surface.surface_id: surface for surface in surfaces}
    loops_by_id = {loop.loop_id: loop for loop in surface_loops}
    errors: list[str] = []
    for volume in volumes:
        if volume.construction_only:
            continue
        _validate_one_volume_shell(
            volume_id=volume.volume_id,
            shell_id="exterior",
            surface_refs=(
                getattr(volume, "exterior_surface_refs", ()) or volume.surface_refs
            ),
            surfaces_by_id=surfaces_by_id,
            loops_by_id=loops_by_id,
            errors=errors,
        )
        for void in getattr(volume, "inner_pec_void_shells", ()):
            _validate_one_volume_shell(
                volume_id=volume.volume_id,
                shell_id=void.shell_id,
                surface_refs=void.surface_refs,
                surfaces_by_id=surfaces_by_id,
                loops_by_id=loops_by_id,
                errors=errors,
            )
    if errors:
        raise ValueError("Invalid volume surface closure: " + "; ".join(errors))


def _validate_one_volume_shell(  # noqa: PLR0913 - keyword-only; each argument is one shell fact
    *,
    volume_id: str,
    shell_id: str,
    surface_refs: tuple[Any, ...],
    surfaces_by_id: Mapping[str, SurfacePlanRecord],
    loops_by_id: Mapping[str, SurfaceLoopRecord],
    errors: list[str],
) -> None:
    curve_use_count: Counter[str] = Counter()
    for surface_ref in surface_refs:
        surface = surfaces_by_id.get(surface_ref.surface_id)
        if surface is None:
            errors.append(
                f"{volume_id}/{shell_id} references unknown surface "
                f"{surface_ref.surface_id}"
            )
            continue
        loop_ids = (
            *((surface.outer_loop_ref,) if surface.outer_loop_ref else ()),
            *surface.hole_loop_refs,
        )
        if not loop_ids:
            errors.append(
                f"{volume_id}/{shell_id} uses non-canonical surface "
                f"{surface.surface_id}"
            )
        for loop_id in loop_ids:
            loop = loops_by_id.get(loop_id)
            if loop is None:
                errors.append(
                    f"{volume_id}/{shell_id} references unknown loop {loop_id}"
                )
                continue
            curve_use_count.update(curve_ref.curve_id for curve_ref in loop.curve_refs)
    bad_counts = {
        curve_id: count for curve_id, count in curve_use_count.items() if count != 2
    }
    if bad_counts:
        errors.append(
            f"{volume_id}/{shell_id} shell is not closed; curve incidence "
            f"{bad_counts!r}"
        )


def validate_surface_use_counts(
    *,
    volumes: tuple[VolumePlanRecord, ...],
    surfaces: tuple[SurfacePlanRecord, ...],
) -> None:
    """Validate internal/exterior surface incidence across live volumes."""
    live_surface_ids = {
        surface.surface_id for surface in surfaces if not surface.construction_only
    }
    use_counts: Counter[str] = Counter(
        surface_ref.surface_id
        for volume in volumes
        if not volume.construction_only
        for surface_ref in volume.surface_refs
    )
    errors: list[str] = []
    for surface in surfaces:
        if surface.construction_only:
            continue
        if surface.surface_role == "lumped_port" and not surface.metadata.get(
            "route_a_boundary_port"
        ):
            continue
        if surface.metadata.get("embedded_surface_sheet") and not surface.metadata.get(
            "sheet_contact_cap"
        ):
            continue
        count = use_counts[surface.surface_id]
        if surface.surface_id not in live_surface_ids:
            errors.append(f"{surface.surface_id} is not live")
        if surface.metadata.get("route_a_boundary_port"):
            if count != 2:
                errors.append(
                    f"{surface.surface_id} Route A port use-count must be 2, "
                    f"got {count}"
                )
        elif surface.interface_id is not None and count not in {1, 2}:
            errors.append(
                f"{surface.surface_id} interface use-count must be 1 or 2, got {count}"
            )
        elif surface.interface_id is None and count != 1:
            errors.append(
                f"{surface.surface_id} exterior use-count must be 1, got {count}"
            )
    if errors:
        raise ValueError("Invalid surface use counts: " + "; ".join(errors))


def validate_surface_partition_coverage(
    *,
    interfaces: tuple[InterfacePlanRecord, ...],
    surface_partitions: tuple[SurfacePartitionRecord, ...],
    surfaces: tuple[SurfacePlanRecord, ...],
) -> None:
    """Validate that partition records point to real parent/child plan ids."""
    surfaces_by_id = {surface.surface_id: surface for surface in surfaces}
    interface_ids = {interface.interface_id for interface in interfaces}
    interface_ids.update(
        surface.interface_id for surface in surfaces if surface.interface_id is not None
    )
    errors: list[str] = []
    for partition in surface_partitions:
        if partition.parent_interface_id not in interface_ids:
            errors.append(
                f"{partition.partition_id} references unknown parent interface "
                f"{partition.parent_interface_id}"
            )
        child_surface = surfaces_by_id.get(partition.child_surface_id)
        if child_surface is None:
            errors.append(
                f"{partition.partition_id} references unknown child surface "
                f"{partition.child_surface_id}"
            )
            continue
        geometry_ref = child_surface.geometry_ref
        if (
            "outer_loop" not in geometry_ref
            and "loop_geometry_ref" not in geometry_ref
            and "quad_points" not in geometry_ref
        ):
            errors.append(f"{partition.partition_id} child surface lacks loop geometry")

    if errors:
        raise ValueError("Invalid surface partition plan: " + "; ".join(errors))


def validate_tag_plan_coverage(
    *,
    surfaces: tuple[SurfacePlanRecord, ...],
    volumes: tuple[VolumePlanRecord, ...],
    tags: tuple[TagPlanRecord, ...],
) -> None:
    """Validate that every exported geometry record has a tag plan.

    Interface surfaces, domain exterior boundary surfaces, and non-construction
    volumes must have `TagPlanRecord`s before OCC assigns dim-tags.
    """
    live_sources = {
        ("surface", surface.surface_id)
        for surface in surfaces
        if not surface.construction_only
        and not surface.metadata.get("hidden_solver_contact")
        and (
            surface.interface_id is not None
            or surface.surface_role == "domain_boundary"
            or surface.surface_role == "lumped_port"
        )
    }
    live_sources.update(
        ("volume", volume.volume_id)
        for volume in volumes
        if not volume.construction_only
    )
    tag_sources = {(tag.source_record_kind, tag.source_record_id) for tag in tags}

    missing = sorted(live_sources - tag_sources)
    extra = sorted(tag_sources - live_sources)
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing tags for {missing!r}")
        if extra:
            parts.append(f"tags reference non-live records {extra!r}")
        raise ValueError("Invalid tag plan coverage: " + "; ".join(parts))


def validate_backend_tag_ledger(
    *,
    backend_tags: tuple[BackendEntityTagRecord, ...],
) -> None:
    """Validate plan-id to backend-tag mapping after OCC lowering."""
    source_to_dimtags: dict[tuple[str, str], set[tuple[int, int]]] = {}
    dimtag_to_sources: dict[tuple[int, int], set[tuple[str, str]]] = {}
    for tag in backend_tags:
        source = (tag.source_record_kind, tag.source_record_id)
        source_to_dimtags.setdefault(source, set()).add(tag.dim_tag)
        dimtag_to_sources.setdefault(tag.dim_tag, set()).add(source)

    errors: list[str] = []
    for source, dimtags in source_to_dimtags.items():
        if len(dimtags) > 1:
            errors.append(f"{source!r} maps to multiple backend tags {dimtags!r}")
    for dimtag, sources in dimtag_to_sources.items():
        if len(sources) > 1:
            errors.append(f"backend tag {dimtag!r} is shared by sources {sources!r}")
    if errors:
        raise ValueError("Invalid backend tag ledger: " + "; ".join(errors))


def _surface_signature(surface: SurfacePlanRecord) -> str:
    if surface.outer_loop_ref is not None:
        return json.dumps(
            {
                "outer": surface.outer_loop_ref,
                "holes": surface.hole_loop_refs,
            },
            sort_keys=True,
        )
    return json.dumps(surface.geometry_ref, sort_keys=True, default=str)


def _surface_interface_kinds(surface: SurfacePlanRecord) -> tuple[str, ...]:
    kinds = surface.metadata.get("interface_kinds", ())
    if isinstance(kinds, str):
        kinds = (kinds,)
    return tuple(sorted({str(kind) for kind in kinds if str(kind) in _INTERFACE_KINDS}))
