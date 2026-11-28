"""Coordinate key and collinear-segment overlap that the grid's loop operations share."""

from __future__ import annotations

from scgsim.geometry.grid.constants import TOPOLOGY_EPS_UM


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


def coordinate_2d_key(point: tuple[float, float]) -> tuple[float, float]:
    """Round an XY point to the established nine-decimal identity key."""
    return (round(float(point[0]), 9), round(float(point[1]), 9))
