"""Low-level polygon conversion and selection shared by source owners.

This leaf has no adapter, occurrence, planning, or backend dependency. Keep one
implementation of source-ring and connected-polygon mechanics for the Ground
recipe and occurrence selectors.
"""

from __future__ import annotations

from scgsim.geometry._primitives.grid import polygon_boolean

from scgsim.geometry._primitives.grid import current_grid_size_um, current_grid_context

from collections.abc import Sequence
from typing import Any

from scgsim.geometry._primitives.loops import _split_gdstk_cutline_loop
from scgsim.geometry.models.input import LayoutPolygonSpec


def _fused_boolean_candidates(candidates):
    """Conservative world-frame broad phase for the expensive overlap test.

    Raw shared-edge recognition still runs in original local coordinates for
    every pair. The historical unbound gdstk path keeps its original calls.
    """
    if current_grid_context() is None:
        return None
    from math import isfinite
    from shapely import STRtree, box
    from scgsim.geometry._primitives.grid import _frame_point, _SOURCE_INTENT

    # Native near-vertex insertion is within grid/100 and node rounding within
    # half a grid step. A full step on EACH bound is a conservative candidate
    # envelope, never an overlap/contact acceptance rule.
    padding = 0.0 if _SOURCE_INTENT.get() else current_grid_size_um()
    envelopes = []
    for polygon in candidates:
        points = tuple(_frame_point(tuple(map(float, p))) for p in polygon.points)
        if not points:
            return None
        low = tuple(min(p[axis] for p in points) - padding for axis in (0, 1))
        high = tuple(max(p[axis] for p in points) + padding for axis in (0, 1))
        if not all(isfinite(v) for v in (*low, *high)):
            # No new input rejection; uncertain envelopes use the original path.
            return None
        envelopes.append(box(*low, *high))
    tree = STRtree(envelopes)
    return tuple(
        frozenset(int(other) for other in tree.query(envelope) if int(other) > index)
        for index, envelope in enumerate(envelopes)
    )


def _fused_gdstk_components(
    candidates: Sequence[Any],
) -> tuple[tuple[LayoutPolygonSpec, tuple[int, ...], float], ...]:
    """Return stable occupied components for explicit source selectors."""
    import gdstk

    overlap_candidates = _fused_boolean_candidates(candidates)
    parents = list(range(len(candidates)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parents[max(first_root, second_root)] = min(first_root, second_root)

    for index, polygon in enumerate(candidates):
        for other_index, other in enumerate(candidates[index + 1 :], index + 1):
            overlap = ()
            if overlap_candidates is None or other_index in overlap_candidates[index]:
                overlap = polygon_boolean(
                    (polygon,), (other,), "and", precision=current_grid_size_um()
                )
            if overlap or _rings_share_edge(
                _ring_from_gdstk_polygon(polygon), _ring_from_gdstk_polygon(other)
            ):
                union(index, other_index)

    members: dict[int, list[int]] = {}
    for index in range(len(candidates)):
        members.setdefault(find(index), []).append(index)
    result: list[tuple[LayoutPolygonSpec, tuple[int, ...], float]] = []
    for indexes in members.values():
        regions = polygon_boolean(
            tuple(candidates[index] for index in indexes),
            (),
            "or",
            precision=current_grid_size_um(),
        )
        # GDS permits conductor fragments to share an exact edge. gdstk's
        # boolean union may retain them separately; normalize this mechanical
        # source fact without using layout names or proximity inference.
        if current_grid_context() is None:
            regions = gdstk.offset(
                regions or (), 0.0, join="miter", precision=current_grid_size_um()
            )
        if not regions or len(regions) != 1:
            raise ValueError("fused selector component must lower to one polygon")
        outer, holes = _split_gdstk_cutline_loop(_ring_from_gdstk_polygon(regions[0]))
        source_indexes = tuple(indexes)
        result.append(
            (
                LayoutPolygonSpec(
                    polygon_id="",
                    layer="",
                    exterior=outer,
                    holes=holes,
                ),
                source_indexes,
                sum(float(candidates[index].area()) for index in source_indexes),
            )
        )
    return tuple(
        sorted(
            result,
            key=lambda record: (
                min(point[0] for point in record[0].exterior),
                min(point[1] for point in record[0].exterior),
                record[2],
            ),
        )
    )


def _ring_from_gdstk_polygon(polygon: Any) -> tuple[tuple[float, float], ...]:
    points = tuple((float(x), float(y)) for x, y in polygon.points)
    if len(points) > 1 and points[0] == points[-1]:
        points = points[:-1]
    if len(points) < 3:
        raise ValueError("GDS polygon requires at least 3 unique points")
    return points


def _ring_edges(
    loop: Sequence[tuple[float, float]],
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    return tuple(zip(loop, (*loop[1:], loop[0]), strict=True))


def _rings_share_edge(
    left: Sequence[tuple[float, float]], right: Sequence[tuple[float, float]]
) -> bool:
    return any(
        _shared_segment_length(left_start, left_end, right_start, right_end) > 1e-9
        for left_start, left_end in _ring_edges(left)
        for right_start, right_end in _ring_edges(right)
    )


def _shared_segment_length(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
) -> float:
    first_dx, first_dy = (
        first_end[0] - first_start[0],
        first_end[1] - first_start[1],
    )
    second_dx, second_dy = (
        second_end[0] - second_start[0],
        second_end[1] - second_start[1],
    )
    first_length = (first_dx * first_dx + first_dy * first_dy) ** 0.5
    second_length = (second_dx * second_dx + second_dy * second_dy) ** 0.5
    if first_length <= 1e-9 or second_length <= 1e-9:
        return 0.0
    if (
        abs(
            (second_start[0] - first_start[0]) * first_dy
            - (second_start[1] - first_start[1]) * first_dx
        )
        > 1e-9 * first_length
        or abs(first_dx * second_dy - first_dy * second_dx)
        > 1e-9 * first_length * second_length
    ):
        return 0.0
    axis = 0 if abs(first_dx) >= abs(first_dy) else 1
    first_low, first_high = sorted((first_start[axis], first_end[axis]))
    second_low, second_high = sorted((second_start[axis], second_end[axis]))
    overlap = min(first_high, second_high) - max(first_low, second_low)
    if overlap <= 1e-9:
        return 0.0
    return overlap * first_length / abs((first_dx, first_dy)[axis])


def _point_in_layout_polygon(
    point: tuple[float, float], polygon: LayoutPolygonSpec
) -> bool:
    return _point_in_ring(point, polygon.exterior) and not any(
        _point_in_ring(point, hole) for hole in polygon.holes
    )


def _point_in_ring(
    point: tuple[float, float],
    ring: Sequence[tuple[float, float]],
) -> bool:
    x, y = point
    inside = False
    j = len(ring) - 1
    for i, (xi, yi) in enumerate(ring):
        xj, yj = ring[j]
        crosses = (yi > y) != (yj > y)
        if crosses:
            x_intersect = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside
