"""Spatial indexes that find planned points lying inside a line segment.

Owns: sorted per-axis and per-axis-aligned-line indexes over `PointPlanRecord`s
and the candidate lookup built on them, so the interior-point check does not
compare every point with every segment.

Does not own: the validation message or the decision that an interior point is an
error (`validation.py`).

Coordinates are rounded to 9 decimals (1e-9 um) so equal points compare equal.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Mapping

from scgsim.geometry.models.topology import PointPlanRecord


def coordinate_signature(
    coordinate: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Round a 3D coordinate to 9 decimals so equal points compare equal."""
    return tuple(round(float(value), 9) for value in coordinate)


def build_point_axis_index(
    points_by_id: Mapping[str, PointPlanRecord],
) -> tuple[tuple[tuple[float, ...], tuple[str, ...]], ...]:
    """Per-axis lists of sorted coordinate values and matching point ids."""
    indexes: list[tuple[tuple[float, ...], tuple[str, ...]]] = []
    for axis in range(3):
        items = sorted(
            (
                coordinate_signature(point.coordinate)[axis],
                point_id,
            )
            for point_id, point in points_by_id.items()
        )
        indexes.append(
            (
                tuple(value for value, _ in items),
                tuple(point_id for _, point_id in items),
            )
        )
    return tuple(indexes)


def axis_aligned_point_index(
    points_by_id: Mapping[str, PointPlanRecord],
) -> dict[tuple[int, tuple[float, float]], tuple[tuple[float, ...], tuple[str, ...]]]:
    """Index points by (axis, other two coordinates) to sorted values and point ids."""
    records: dict[
        tuple[int, tuple[float, float]],
        list[tuple[float, str]],
    ] = {}
    for point_id, point in points_by_id.items():
        key = coordinate_signature(point.coordinate)
        for varying_axis in range(3):
            fixed_key = tuple(key[axis] for axis in range(3) if axis != varying_axis)
            records.setdefault((varying_axis, fixed_key), []).append(
                (key[varying_axis], point_id)
            )
    return {
        line_key: (
            tuple(value for value, _ in sorted_items),
            tuple(point_id for _, point_id in sorted_items),
        )
        for line_key, items in records.items()
        for sorted_items in (tuple(sorted(items)),)
    }


def segment_candidate_point_ids(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    point_axis_index: tuple[tuple[tuple[float, ...], tuple[str, ...]], ...],
    point_line_index: Mapping[
        tuple[int, tuple[float, float]],
        tuple[tuple[float, ...], tuple[str, ...]],
    ],
) -> tuple[str, ...]:
    """Ids of points that may lie on a segment, using the line or axis index."""
    start_key = coordinate_signature(start)
    end_key = coordinate_signature(end)
    varying_axes = tuple(
        axis for axis in range(3) if abs(start_key[axis] - end_key[axis]) > 1e-9
    )
    if len(varying_axes) == 1:
        varying_axis = varying_axes[0]
        fixed_key = tuple(start_key[axis] for axis in range(3) if axis != varying_axis)
        values, point_ids = point_line_index.get(
            (varying_axis, fixed_key),
            ((), ()),
        )
        lower = min(start_key[varying_axis], end_key[varying_axis]) - 1e-9
        upper = max(start_key[varying_axis], end_key[varying_axis]) + 1e-9
        left = bisect_left(values, lower)
        right = bisect_right(values, upper)
        return point_ids[left:right]
    bounds = tuple(
        (
            min(start_key[axis], end_key[axis]) - 1e-9,
            max(start_key[axis], end_key[axis]) + 1e-9,
        )
        for axis in range(3)
    )
    ranges: list[tuple[int, int, int]] = []
    for axis, (values, _) in enumerate(point_axis_index):
        lower, upper = bounds[axis]
        left = bisect_left(values, lower)
        right = bisect_right(values, upper)
        ranges.append((right - left, left, axis))
    count, left, axis = min(ranges)
    return point_axis_index[axis][1][left : left + count]


def point_on_segment_interior(
    point: tuple[float, float, float],
    start: tuple[float, float, float],
    end: tuple[float, float, float],
) -> bool:
    """True when a point lies on a segment strictly between its endpoints."""
    vector = tuple(float(end[index]) - float(start[index]) for index in range(3))
    offset = tuple(float(point[index]) - float(start[index]) for index in range(3))
    length_sq = sum(value * value for value in vector)
    if length_sq <= 1e-18:
        return False
    parameter = sum(offset[index] * vector[index] for index in range(3)) / length_sq
    if parameter <= 1e-9 or parameter >= 1.0 - 1e-9:
        return False
    closest = tuple(
        float(start[index]) + parameter * vector[index] for index in range(3)
    )
    distance_sq = sum((float(point[index]) - closest[index]) ** 2 for index in range(3))
    return distance_sq <= 1e-18
