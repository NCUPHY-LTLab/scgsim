"""Shared polygon/loop/cutline operations beneath source and planning. Preserve holes, ordered source geometry and existing precision."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import pairwise
from math import sqrt
from typing import Any

from scgsim.geometry.models.input import SemanticEntitySpec
from scgsim.geometry.models.topology import CurveRefRecord
from scgsim.geometry.primitives.constants import TOPOLOGY_EPS_UM
from scgsim.geometry.primitives.spatial import (
    coordinate_2d_key,
    segment_overlap_interval,
)


def loops_touch_without_area(left: Any, right: Any) -> bool:
    """Reject a coplanar contact that has no finite-area footprint."""
    left_loop, right_loop = normalized_loop(left), normalized_loop(right)
    return any(
        _point_on_segment(point, start, end)
        for point in left_loop
        for start, end in ring_edges(right_loop)
    ) or any(
        _point_on_segment(point, start, end)
        for point in right_loop
        for start, end in ring_edges(left_loop)
    )


def canonical_loop_sort_key(loop: tuple[tuple[float, float], ...]) -> tuple:
    clean = normalized_loop(loop)
    candidates: list[tuple[tuple[float, float], ...]] = []
    reversed_loop = tuple(reversed(clean))
    for candidate in (clean, reversed_loop):
        for offset in range(len(candidate)):
            rotated = candidate[offset:] + candidate[:offset]
            candidates.append(rotated)
    return min(candidates)


def loop_signature(loop: Any) -> tuple[tuple[float, float], ...]:
    try:
        points = normalized_loop(loop)
    except Exception:  # noqa: BLE001 - malformed optional loop has no signature
        return ()
    return tuple(sorted(coordinate_2d_key(point) for point in points))


def same_loop_geometry(left: Any, right: Any) -> bool:
    return loop_signature(left) == loop_signature(right)


def contact_holes_are_simple(
    outer_loop: tuple[tuple[float, float], ...],
    hole_loops: Sequence[tuple[tuple[float, float], ...]],
) -> bool:
    for hole_loop in hole_loops:
        if not _loop_strictly_inside_loop(hole_loop, outer_loop):
            return False
    hole_records = tuple(
        (_loop_bounds_tuple(hole_loop), hole_loop) for hole_loop in hole_loops
    )
    for index, (left_bounds, left) in enumerate(hole_records):
        for right_bounds, right in hole_records[index + 1 :]:
            if not _bounds_tuple_may_overlap(left_bounds, right_bounds):
                continue
            if loops_share_edge_overlap(left, right):
                return False
    return True


def _loop_strictly_inside_loop(loop: Any, container: Any) -> bool:
    """Require a real interior hole, excluding any boundary touch."""
    if not _loop_inside_loop(loop, container):
        return False
    loop_points = normalized_loop(loop)
    container_points = normalized_loop(container)
    if any(
        _point_on_segment(point, start, end)
        for point in loop_points
        for start, end in ring_edges(container_points)
    ):
        return False
    return not loops_share_edge_overlap(loop_points, container_points)


def _point_on_segment(
    point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]
) -> bool:
    dx, dy = end[0] - start[0], end[1] - start[1]
    cross = (point[0] - start[0]) * dy - (point[1] - start[1]) * dx
    if abs(cross) > TOPOLOGY_EPS_UM:
        return False
    return (
        min(start[0], end[0]) - TOPOLOGY_EPS_UM
        <= point[0]
        <= max(start[0], end[0]) + TOPOLOGY_EPS_UM
        and min(start[1], end[1]) - TOPOLOGY_EPS_UM
        <= point[1]
        <= max(start[1], end[1]) + TOPOLOGY_EPS_UM
    )


def _loop_bounds_tuple(
    loop: tuple[tuple[float, float], ...],
) -> tuple[float, float, float, float]:
    return (
        min(point[0] for point in loop),
        min(point[1] for point in loop),
        max(point[0] for point in loop),
        max(point[1] for point in loop),
    )


def _bounds_tuple_may_overlap(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    """Cheap candidate filter; contact/hole validity is checked on loops."""
    return not (
        left[2] <= right[0] + TOPOLOGY_EPS_UM
        or right[2] <= left[0] + TOPOLOGY_EPS_UM
        or left[3] <= right[1] + TOPOLOGY_EPS_UM
        or right[3] <= left[1] + TOPOLOGY_EPS_UM
    )


def simple_interior_hole_loops(
    parent_geometry_ref: Mapping[str, Any],
    plane_conductors: Sequence[SemanticEntitySpec],
    *,
    conductor_geometry_refs: Sequence[Mapping[str, Any]] = (),
) -> tuple[tuple[tuple[float, float], ...], ...] | None:
    # This shortcut constructs a new hole list from conductor footprints.  An
    # already-holed solution interface must take the Boolean path so authored
    # substrate holes remain absent from the live SA surface.
    if parent_geometry_ref.get("hole_loops"):
        return None
    if any(entity.geometry.get("hole_loops") for entity in plane_conductors):
        return None
    if any(geometry_ref.get("hole_loops") for geometry_ref in conductor_geometry_refs):
        return None
    base_loop = normalized_loop(parent_geometry_ref["outer_loop"])
    hole_loops = tuple(
        normalized_loop(entity.geometry["outer_loop"])
        for entity in plane_conductors
        if "outer_loop" in entity.geometry
    ) + tuple(
        normalized_loop(geometry_ref["outer_loop"])
        for geometry_ref in conductor_geometry_refs
        if "outer_loop" in geometry_ref
    )
    if len(hole_loops) != len(plane_conductors) + len(conductor_geometry_refs):
        return None
    if any(not _loop_inside_loop(hole_loop, base_loop) for hole_loop in hole_loops):
        return None
    # Overlapping conductor footprints must be unioned before they become
    # solution-interface holes; separate overlapping holes create a third
    # coincident rim when a normalized contact-pad cap is added.
    import gdstk

    for index, left in enumerate(hole_loops):
        for right in hole_loops[index + 1 :]:
            if boolean_gdstk_region(
                gdstk,
                (gdstk.Polygon(left),),
                (gdstk.Polygon(right),),
                "and",
            ):
                return None
    all_loops = (base_loop, *hole_loops)
    for index, left in enumerate(all_loops):
        for right in all_loops[index + 1 :]:
            if loops_share_edge_overlap(left, right):
                return None
    return hole_loops


def loops_share_edge_overlap(
    left: tuple[tuple[float, float], ...],
    right: tuple[tuple[float, float], ...],
) -> bool:
    return any(
        segment_overlap_interval(left_start, left_end, right_start, right_end)
        is not None
        for left_start, left_end in ring_edges(left)
        for right_start, right_end in ring_edges(right)
    )


def edge_is_covered_by_loop_edge(
    start: tuple[float, float],
    end: tuple[float, float],
    loop: Any,
) -> bool:
    return any(
        interval is not None
        and interval[0] <= TOPOLOGY_EPS_UM
        and interval[1] >= 1.0 - TOPOLOGY_EPS_UM
        for edge_start, edge_end in ring_edges(normalized_loop(loop))
        for interval in (segment_overlap_interval(start, end, edge_start, edge_end),)
    )


def _loop_inside_loop(loop: Any, container: Any) -> bool:
    import gdstk

    clean_loop = normalized_loop(loop)
    clean_container = normalized_loop(container)
    loop_area = abs(polygon_area(clean_loop))
    intersection = boolean_gdstk_region(
        gdstk,
        (gdstk.Polygon(clean_loop),),
        (gdstk.Polygon(clean_container),),
        "and",
    )
    intersection_area = sum(
        abs(polygon_area(polygon.points)) for polygon in intersection
    )
    return loop_area - intersection_area <= max(TOPOLOGY_EPS_UM, loop_area * 1e-9)


def loop_centroid(loop: Any) -> tuple[float, float]:
    points = tuple((float(point[0]), float(point[1])) for point in loop)
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def ordered_loop_signature(
    curve_refs: tuple[CurveRefRecord, ...],
) -> tuple[tuple[str, int], ...]:
    """Return a rotation-stable loop signature without discarding orientation."""
    signature = tuple((ref.curve_id, ref.orientation) for ref in curve_refs)
    if not signature:
        return ()
    rotations = (
        signature[index:] + signature[:index] for index in range(len(signature))
    )
    return min(rotations)


def normalized_loop(loop: Any) -> tuple[tuple[float, float], ...]:
    points = tuple(
        coordinate_2d_key((float(point[0]), float(point[1]))) for point in loop
    )
    if len(points) > 1 and points[0] == points[-1]:
        points = points[:-1]
    points = _drop_near_duplicate_loop_points(points)
    points = _drop_redundant_collinear_loop_points(points)
    if len(points) < 3:
        raise ValueError("loop requires at least 3 unique points")
    if abs(polygon_area(points)) <= TOPOLOGY_EPS_UM:
        raise ValueError("loop area is below topology tolerance")
    return points


def _drop_near_duplicate_loop_points(
    points: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    cleaned: list[tuple[float, float]] = []
    for point in points:
        if cleaned and cleaned[-1] == point:
            continue
        cleaned.append(point)
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    return tuple(cleaned)


def _drop_redundant_collinear_loop_points(
    points: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    points = _drop_near_duplicate_loop_points(points)
    if len(points) <= 3:
        return points

    protected = _duplicate_edge_point_indices(points)
    changed = True
    while changed and len(points) > 3:
        changed = False
        for index, point in enumerate(points):
            if index in protected:
                continue
            previous = points[index - 1]
            following = points[(index + 1) % len(points)]
            if _point_on_segment_2d(point, previous, following):
                points = points[:index] + points[index + 1 :]
                protected = _duplicate_edge_point_indices(points)
                changed = True
                break
    return points


def _duplicate_edge_point_indices(
    points: tuple[tuple[float, float], ...],
) -> set[int]:
    edge_counts = Counter(
        tuple(sorted((start, end))) for start, end in ring_edges(points)
    )
    protected: set[int] = set()
    for index, edge in enumerate(ring_edges(points)):
        if edge_counts[tuple(sorted(edge))] > 1:
            protected.update((index, (index + 1) % len(points)))
    return protected


def _point_on_segment_2d(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> bool:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq <= TOPOLOGY_EPS_UM * TOPOLOGY_EPS_UM:
        return False
    cross = (point[0] - start[0]) * dy - (point[1] - start[1]) * dx
    if abs(cross) > TOPOLOGY_EPS_UM * sqrt(length_sq):
        return False
    dot = (point[0] - start[0]) * dx + (point[1] - start[1]) * dy
    return TOPOLOGY_EPS_UM < dot < length_sq - TOPOLOGY_EPS_UM


def ring_edges(
    ring: tuple[tuple[float, float], ...],
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    return tuple(
        (ring[index], ring[(index + 1) % len(ring)]) for index in range(len(ring))
    )


def gdstk_surface_region(geometry_ref: Mapping[str, Any]) -> tuple[Any, ...]:
    import gdstk

    outer = gdstk.Polygon(normalized_loop(geometry_ref["outer_loop"]))
    holes = tuple(
        gdstk.Polygon(normalized_loop(hole_loop))
        for hole_loop in geometry_ref.get("hole_loops", ())
    )
    if not holes:
        return (outer,)
    return boolean_gdstk_region(gdstk, (outer,), holes, "not")


def boolean_gdstk_region(
    gdstk: Any,
    left: Sequence[Any],
    right: Sequence[Any],
    operation: str,
) -> tuple[Any, ...]:
    if not left:
        return ()
    if operation == "not" and not right:
        return filter_gdstk_polygons(left)
    result = gdstk.boolean(
        left,
        right,
        operation,
        precision=1e-9,
    )
    return filter_gdstk_polygons(result or ())


def filter_gdstk_polygons(polygons: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(
        polygon
        for polygon in polygons
        if abs(polygon_area(polygon.points)) > TOPOLOGY_EPS_UM
    )


def geometry_ref_from_gdstk_polygon(
    parent_geometry_ref: Mapping[str, Any],
    polygon: Any,
) -> dict[str, Any]:
    geometry_ref = dict(parent_geometry_ref)
    outer_loop, hole_loops = split_gdstk_cutline_loop(normalized_loop(polygon.points))
    geometry_ref["outer_loop"] = outer_loop
    geometry_ref["hole_loops"] = hole_loops
    return geometry_ref


def split_gdstk_cutline_loop(
    loop: tuple[tuple[float, float], ...],
) -> tuple[
    tuple[tuple[float, float], ...],
    tuple[tuple[tuple[float, float], ...], ...],
]:
    """Normalize gdstk's cutline-encoded boundary into simple outer/hole loops.

    Gdstk represents a polygon with holes as one walk with paired, opposite
    cutline segments.  Atomic splitting and cancellation leaves exactly the
    independent boundary cycles; that representation is suitable for both the
    adapter's fused selector components and Boolean residual lowering.
    """
    atomic_edges: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for start, end in ring_edges(loop):
        points = sorted(
            {point for point in loop if _point_on_planar_segment(point, start, end)},
            key=lambda point: _planar_segment_parameter(point, start, end),
        )
        atomic_edges.extend((left, right) for left, right in pairwise(points))
    loops = simple_planar_loops_from_edges(cancel_reversed_planar_edges(atomic_edges))
    if not loops:
        raise ValueError("gdstk cutline polygon has no simple boundary loop")
    outer_index = max(
        range(len(loops)), key=lambda index: abs(polygon_area(loops[index]))
    )
    outer = loops[outer_index]
    holes = tuple(
        sorted(
            (loop_ for index, loop_ in enumerate(loops) if index != outer_index),
            key=lambda loop_: (
                min(point[0] for point in loop_),
                min(point[1] for point in loop_),
            ),
        )
    )
    if any(not _loop_inside_loop(hole, outer) for hole in holes):
        raise ValueError("gdstk cutline polygon has a non-interior boundary loop")
    return outer, holes


def _point_on_planar_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> bool:
    dx, dy = end[0] - start[0], end[1] - start[1]
    if abs((point[0] - start[0]) * dy - (point[1] - start[1]) * dx) > TOPOLOGY_EPS_UM:
        return False
    return (
        min(start[0], end[0]) - TOPOLOGY_EPS_UM
        <= point[0]
        <= max(start[0], end[0]) + TOPOLOGY_EPS_UM
        and min(start[1], end[1]) - TOPOLOGY_EPS_UM
        <= point[1]
        <= max(start[1], end[1]) + TOPOLOGY_EPS_UM
    )


def _planar_segment_parameter(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    return ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / (
        dx * dx + dy * dy
    )


def cancel_reversed_planar_edges(
    edges: Sequence[tuple[tuple[float, float], tuple[float, float]]],
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    counts: dict[tuple[tuple[float, float], tuple[float, float]], int] = {}
    for edge in edges:
        counts[edge] = counts.get(edge, 0) + 1
    for edge in tuple(counts):
        reverse = (edge[1], edge[0])
        cancelled = min(counts.get(edge, 0), counts.get(reverse, 0))
        if cancelled:
            counts[edge] -= cancelled
            counts[reverse] -= cancelled
    return tuple(
        edge
        for edge, count in counts.items()
        for _ in range(count)
        if edge[0] != edge[1]
    )


def simple_planar_loops_from_edges(
    edges: Sequence[tuple[tuple[float, float], tuple[float, float]]],
) -> tuple[tuple[tuple[float, float], ...], ...]:
    neighbors: dict[tuple[float, float], set[tuple[float, float]]] = {}
    unused: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    for start, end in edges:
        edge_key = tuple(sorted((start, end)))
        if edge_key in unused:
            raise ValueError("gdstk cutline repeats a retained boundary edge")
        unused.add(edge_key)
        neighbors.setdefault(start, set()).add(end)
        neighbors.setdefault(end, set()).add(start)
    if any(len(points) != 2 for points in neighbors.values()):
        raise ValueError("gdstk cutline boundary is not a simple loop")
    loops: list[tuple[tuple[float, float], ...]] = []
    while unused:
        start, current = next(iter(unused))
        loop = [start]
        previous = start
        while current != start:
            loop.append(current)
            choices = neighbors[current] - {previous}
            if len(choices) != 1:
                raise ValueError("gdstk cutline boundary branches")
            following = next(iter(choices))
            unused.discard(tuple(sorted((previous, current))))
            previous, current = current, following
        unused.discard(tuple(sorted((previous, current))))
        loops.append(tuple(loop))
    return tuple(loops)


def geometry_refs_from_gdstk_region(
    parent_geometry_ref: Mapping[str, Any],
    region: tuple[Any, ...],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        geometry_ref_from_gdstk_polygon(parent_geometry_ref, polygon)
        for polygon in filter_gdstk_polygons(region)
    )


def polygon_area(points: Sequence[Sequence[float]]) -> float:
    clean_points = tuple((float(point[0]), float(point[1])) for point in points)
    if len(clean_points) < 3:
        return 0.0
    return 0.5 * sum(
        x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in ring_edges(clean_points)
    )
