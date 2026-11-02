"""Index canonical coordinates and select ordered segment split candidates."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Collection, Mapping, Sequence
from typing import cast

from scgsim.geometry.primitives.constants import TOPOLOGY_EPS_UM
from scgsim.geometry.primitives.spatial import coordinate_key


def segment_parameter(
    point: tuple[float, float, float],
    start: tuple[float, float, float],
    end: tuple[float, float, float],
) -> float | None:
    """Locate a point on a directed 3D segment using the existing tolerances."""
    vector = tuple(end[index] - start[index] for index in range(3))
    offset = tuple(point[index] - start[index] for index in range(3))
    length_sq = sum(value * value for value in vector)
    if length_sq <= 1e-18:
        return None
    parameter = sum(offset[index] * vector[index] for index in range(3)) / length_sq
    if parameter < -1e-9 or parameter > 1.0 + 1e-9:
        return None
    closest = tuple(start[index] + parameter * vector[index] for index in range(3))
    distance_sq = sum((point[index] - closest[index]) ** 2 for index in range(3))
    if distance_sq > 1e-18:
        return None
    return max(0.0, min(1.0, parameter))


def index_points_by_axis(
    coordinates: Collection[tuple[float, float, float]],
) -> tuple[
    tuple[tuple[float, ...], tuple[tuple[float, float, float], ...]],
    ...,
]:
    """Index the complete reusable coordinate collection along each axis."""
    indexes: list[tuple[tuple[float, ...], tuple[tuple[float, float, float], ...]]] = []
    for axis in range(3):
        items = sorted((coordinate[axis], coordinate) for coordinate in coordinates)
        indexes.append(
            (
                tuple(value for value, _ in items),
                tuple(coordinate for _, coordinate in items),
            )
        )
    return tuple(indexes)


def axis_aligned_point_index(
    coordinates: Collection[tuple[float, float, float]],
) -> dict[
    tuple[int, tuple[float, float]],
    tuple[tuple[float, ...], tuple[tuple[float, float, float], ...]],
]:
    """Index canonical coordinates by their varying axis and two fixed coordinates."""
    records: dict[
        tuple[int, tuple[float, float]],
        list[tuple[float, tuple[float, float, float]]],
    ] = {}
    for coordinate in coordinates:
        key = coordinate_key(coordinate)
        for varying_axis in range(3):
            fixed_key = cast(
                tuple[float, float],
                tuple(key[axis] for axis in range(3) if axis != varying_axis),
            )
            records.setdefault((varying_axis, fixed_key), []).append(
                (key[varying_axis], key)
            )
    return {
        line_key: (
            tuple(value for value, _ in sorted_items),
            tuple(coordinate for _, coordinate in sorted_items),
        )
        for line_key, items in records.items()
        for sorted_items in (tuple(sorted(items)),)
    }


def segment_candidate_points(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    point_axis_index: Sequence[
        tuple[tuple[float, ...], tuple[tuple[float, float, float], ...]]
    ],
    point_line_index: Mapping[
        tuple[int, tuple[float, float]],
        tuple[tuple[float, ...], tuple[tuple[float, float, float], ...]],
    ],
) -> tuple[tuple[float, float, float], ...]:
    """Select deterministic coordinate candidates for one segment before exact testing."""
    start_key = coordinate_key(start)
    end_key = coordinate_key(end)
    varying_axes = tuple(
        axis
        for axis in range(3)
        if abs(start_key[axis] - end_key[axis]) > TOPOLOGY_EPS_UM
    )
    if len(varying_axes) == 1:
        varying_axis = varying_axes[0]
        fixed_key = cast(
            tuple[float, float],
            tuple(start_key[axis] for axis in range(3) if axis != varying_axis),
        )
        values, coordinates = point_line_index.get(
            (varying_axis, fixed_key),
            ((), ()),
        )
        lower = min(start_key[varying_axis], end_key[varying_axis]) - TOPOLOGY_EPS_UM
        upper = max(start_key[varying_axis], end_key[varying_axis]) + TOPOLOGY_EPS_UM
        left = bisect_left(values, lower)
        right = bisect_right(values, upper)
        candidates = set(coordinates[left:right])
        candidates.update((start_key, end_key))
        return tuple(sorted(candidates))
    bounds = tuple(
        (
            min(start_key[axis], end_key[axis]) - TOPOLOGY_EPS_UM,
            max(start_key[axis], end_key[axis]) + TOPOLOGY_EPS_UM,
        )
        for axis in range(3)
    )
    ranges: list[tuple[int, int, int]] = []
    for axis, (values, _) in enumerate(point_axis_index):
        lower, upper = bounds[axis]
        left = bisect_left(values, lower)
        right = bisect_right(values, upper)
        ranges.append((right - left, left, axis))
    _, left, axis = min(ranges)
    right = left + min(ranges)[0]
    coordinates = point_axis_index[axis][1][left:right]
    candidates = {
        coordinate
        for coordinate in coordinates
        if all(
            bounds[coordinate_axis][0]
            <= coordinate[coordinate_axis]
            <= bounds[coordinate_axis][1]
            for coordinate_axis in range(3)
        )
    }
    candidates.update((start_key, end_key))
    return tuple(sorted(candidates))
