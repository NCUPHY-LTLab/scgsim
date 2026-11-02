"""Shared coordinate, interval, and bounds operations for geometry planning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

# Geometry-reference values remain open source metadata at this boundary.
from scgsim.geometry.primitives.constants import (
    TOPOLOGY_EPS_UM,
)


def geometry_ref_surface_z_um(geometry_ref: Mapping[str, Any]) -> float:
    """Resolve the surface plane from shell, explicit plane, or base Z data."""
    z_min_um = float(geometry_ref.get("z_min_um", geometry_ref.get("z_um", 0.0)))
    shell_part = geometry_ref.get("shell_part")
    if shell_part == "top":
        z_max_um = geometry_ref.get("z_max_um")
        if z_max_um is not None:
            return float(z_max_um)
        return z_min_um + float(geometry_ref.get("thickness_um", 0.0))
    if shell_part == "bottom":
        return z_min_um
    plane = geometry_ref.get("plane") or geometry_ref.get("contact_plane")
    if isinstance(plane, Mapping) and plane.get("axis") == "z":
        return float(plane["value_um"])
    return z_min_um


def coordinate_key(
    coordinate: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Round a 3D point to the established nine-decimal identity key."""
    return (
        round(float(coordinate[0]), 9),
        round(float(coordinate[1]), 9),
        round(float(coordinate[2]), 9),
    )


def bounds_overlap(
    first: Mapping[str, float],
    second: Mapping[str, float],
) -> bool:
    """Return whether two XY bounds overlap beyond the topology tolerance."""
    return not (
        float(first["x_max_um"]) <= float(second["x_min_um"]) + TOPOLOGY_EPS_UM
        or float(second["x_max_um"]) <= float(first["x_min_um"]) + TOPOLOGY_EPS_UM
        or float(first["y_max_um"]) <= float(second["y_min_um"]) + TOPOLOGY_EPS_UM
        or float(second["y_max_um"]) <= float(first["y_min_um"]) + TOPOLOGY_EPS_UM
    )


def coordinate_z_key(value_um: float) -> float:
    """Round a Z coordinate to the established nine-decimal identity key."""
    return round(float(value_um), 9)


def interpolate_2d(
    start: tuple[float, float],
    end: tuple[float, float],
    parameter: float,
) -> tuple[float, float]:
    """Interpolate one XY point at a segment parameter."""
    return (
        start[0] + (end[0] - start[0]) * parameter,
        start[1] + (end[1] - start[1]) * parameter,
    )


def vector_subtract(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Return the component-wise difference between two 3D vectors."""
    return (
        left[0] - right[0],
        left[1] - right[1],
        left[2] - right[2],
    )


def vector_cross(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Return the right-handed cross product of two 3D vectors."""
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def undirected_xy_edge(
    start: Sequence[float], end: Sequence[float]
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return rounded edge endpoints in direction-independent order."""
    first = coordinate_2d_key((float(start[0]), float(start[1])))
    second = coordinate_2d_key((float(end[0]), float(end[1])))
    ordered = sorted((first, second))
    return ordered[0], ordered[1]


def intersect_bounds(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> dict[str, float] | None:
    """Return the positive-area XY intersection, or None for touching bounds."""
    bounds = {
        "x_min_um": max(float(first["x_min_um"]), float(second["x_min_um"])),
        "y_min_um": max(float(first["y_min_um"]), float(second["y_min_um"])),
        "x_max_um": min(float(first["x_max_um"]), float(second["x_max_um"])),
        "y_max_um": min(float(first["y_max_um"]), float(second["y_max_um"])),
    }
    if (
        bounds["x_min_um"] >= bounds["x_max_um"]
        or bounds["y_min_um"] >= bounds["y_max_um"]
    ):
        return None
    return bounds


def bounds_contains_point(
    bounds: Mapping[str, Any],
    point: tuple[float, float],
) -> bool:
    """Return whether a point lies inside or on the XY bounds."""
    x, y = point
    return float(bounds["x_min_um"]) <= x <= float(bounds["x_max_um"]) and float(
        bounds["y_min_um"]
    ) <= y <= float(bounds["y_max_um"])


def same_coordinate_z(left: float, right: float) -> bool:
    """Compare Z coordinates using the shared topology tolerance."""
    return abs(left - right) <= TOPOLOGY_EPS_UM


def segment_overlap_interval(
    start: tuple[float, float],
    end: tuple[float, float],
    other_start: tuple[float, float],
    other_end: tuple[float, float],
) -> tuple[float, float] | None:
    """Return the positive collinear overlap as parameters on the first segment."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-18:
        return None

    def parameter(point: tuple[float, float]) -> float | None:
        cross = (point[0] - start[0]) * dy - (point[1] - start[1]) * dx
        if abs(cross) > 1e-9:
            return None
        return ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_sq

    first = parameter(other_start)
    second = parameter(other_end)
    if first is None or second is None:
        return None
    overlap_start = max(0.0, min(first, second))
    overlap_end = min(1.0, max(first, second))
    if overlap_end - overlap_start <= TOPOLOGY_EPS_UM:
        return None
    return overlap_start, overlap_end


def line_key_2d(
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[tuple[float, float], float] | None:
    """Return a direction-independent normalized XY line key."""
    start_key = tuple(round(float(value), 9) for value in start)
    end_key = tuple(round(float(value), 9) for value in end)
    direction = (
        end_key[0] - start_key[0],
        end_key[1] - start_key[1],
    )
    scale = max(abs(value) for value in direction)
    if scale <= 1e-18:
        return None
    unit = (
        round(direction[0] / scale, 9),
        round(direction[1] / scale, 9),
    )
    for component in unit:
        if abs(component) <= 1e-18:
            continue
        if component < 0:
            unit = (-unit[0], -unit[1])
        break
    offset = round(start_key[0] * unit[1] - start_key[1] * unit[0], 9)
    return unit, offset


def interval_complement(
    intervals: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    """Return uncovered unit-interval spans after clipping and merging inputs."""
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        start = max(0.0, min(1.0, start))
        end = max(0.0, min(1.0, end))
        if end - start <= TOPOLOGY_EPS_UM:
            continue
        if not merged or start > merged[-1][1] + TOPOLOGY_EPS_UM:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    exposed: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in merged:
        if start - cursor > TOPOLOGY_EPS_UM:
            exposed.append((cursor, start))
        cursor = max(cursor, end)
    if 1.0 - cursor > TOPOLOGY_EPS_UM:
        exposed.append((cursor, 1.0))
    return tuple(exposed)


def interpolate_3d(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    parameter: float,
) -> tuple[float, float, float]:
    """Interpolate one XYZ point at a segment parameter."""
    return (
        start[0] + (end[0] - start[0]) * parameter,
        start[1] + (end[1] - start[1]) * parameter,
        start[2] + (end[2] - start[2]) * parameter,
    )


def range_within_bounds(
    start: float,
    end: float,
    lower: float,
    upper: float,
) -> bool:
    """Return whether a scalar segment lies within bounds plus tolerance."""
    return lower - TOPOLOGY_EPS_UM <= min(start, end) and max(start, end) <= (
        upper + TOPOLOGY_EPS_UM
    )


def coordinate_2d_key(point: tuple[float, float]) -> tuple[float, float]:
    """Round an XY point to the established nine-decimal identity key."""
    return (round(float(point[0]), 9), round(float(point[1]), 9))
