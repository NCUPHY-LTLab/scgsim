"""Rebuild planar boundary cycles from coordinates without grid or Boolean policy."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from itertools import pairwise
from math import sqrt
from typing import Any

from scgsim.geometry._primitives.constants import _TOPOLOGY_EPS_UM
from scgsim.geometry._primitives.spatial import _coordinate_2d_key


def cutline_boundary_loops(
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
    return outer, holes


def clean_loop(loop: Any) -> tuple[tuple[float, float], ...]:
    """Round coordinate keys and remove redundant points while preserving cutlines."""
    points = tuple(
        _coordinate_2d_key((float(point[0]), float(point[1]))) for point in loop
    )
    if len(points) > 1 and points[0] == points[-1]:
        points = points[:-1]
    points = _drop_near_duplicate_loop_points(points)
    points = _drop_redundant_collinear_loop_points(points)
    if len(points) < 3:
        raise ValueError("loop requires at least 3 unique points")
    if abs(polygon_area(points)) <= _TOPOLOGY_EPS_UM:
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
    if length_sq <= _TOPOLOGY_EPS_UM * _TOPOLOGY_EPS_UM:
        return False
    cross = (point[0] - start[0]) * dy - (point[1] - start[1]) * dx
    if abs(cross) > _TOPOLOGY_EPS_UM * sqrt(length_sq):
        return False
    dot = (point[0] - start[0]) * dx + (point[1] - start[1]) * dy
    return _TOPOLOGY_EPS_UM < dot < length_sq - _TOPOLOGY_EPS_UM


def ring_edges(
    ring: tuple[tuple[float, float], ...],
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    """Return directed edges including the closing edge of a ring."""
    return tuple(
        (ring[index], ring[(index + 1) % len(ring)]) for index in range(len(ring))
    )


def filter_gdstk_polygons(polygons: Sequence[Any]) -> tuple[Any, ...]:
    """Retain polygons whose area exceeds the existing topology tolerance."""
    return tuple(
        polygon
        for polygon in polygons
        if abs(polygon_area(polygon.points)) > _TOPOLOGY_EPS_UM
    )


def _point_on_planar_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> bool:
    dx, dy = end[0] - start[0], end[1] - start[1]
    if abs((point[0] - start[0]) * dy - (point[1] - start[1]) * dx) > _TOPOLOGY_EPS_UM:
        return False
    return (
        min(start[0], end[0]) - _TOPOLOGY_EPS_UM
        <= point[0]
        <= max(start[0], end[0]) + _TOPOLOGY_EPS_UM
        and min(start[1], end[1]) - _TOPOLOGY_EPS_UM
        <= point[1]
        <= max(start[1], end[1]) + _TOPOLOGY_EPS_UM
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
    """Cancel opposite directed edge occurrences without discarding retained repeats."""
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
    """Reconstruct nonbranching cycles and reject repeated retained boundary edges."""
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


def polygon_area(points: Sequence[Sequence[float]]) -> float:
    """Return the signed shoelace area of planar coordinates."""
    clean_points = tuple((float(point[0]), float(point[1])) for point in points)
    if len(clean_points) < 3:
        return 0.0
    return 0.5 * sum(
        x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in ring_edges(clean_points)
    )
