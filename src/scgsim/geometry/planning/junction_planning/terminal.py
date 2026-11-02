"""Validate rectangular terminal geometry against the shared structural contract."""

from __future__ import annotations

import math
from collections.abc import Mapping
from fractions import Fraction
from typing import Any, Protocol

from scgsim.geometry.planning.junction_planning.exact_geometry import cross


class JunctionTerminalGeometry(Protocol):
    """Describe the junction identity, direction and width used by terminal lines."""

    @property
    def junction_id(self) -> str:
        """Identify the source junction in geometry errors."""
        ...

    @property
    def direction_xy(self) -> tuple[float, float]:
        """Give the junction's axial direction in the XY plane."""
        ...

    @property
    def width_um(self) -> float:
        """Give the junction's transverse width in micrometres."""
        ...


def junction_terminal_line(
    polygon: Mapping[str, Any], junction: JunctionTerminalGeometry, z_um: float
) -> tuple[list[list[float]], float]:
    """Validate a rectangular junction sheet and return its axial terminal line."""
    exterior = tuple(
        ((float(point[0]), float(point[1])) for point in polygon["exterior"])
    )

    def rational(p):
        return tuple((Fraction(str(v)) for v in p))

    witness = tuple(
        (point for index, point in enumerate(exterior) if point != exterior[index - 1])
    )
    exterior = tuple(
        (
            point
            for index, point in enumerate(witness)
            if cross(
                rational(witness[index - 1]),
                rational(point),
                rational(witness[(index + 1) % len(witness)]),
            )
            != 0
        )
    )
    if len(exterior) != 4 or polygon.get("holes"):
        raise ValueError(
            f"junction {junction.junction_id!r} requires one rectangular polygon without holes"
        )
    dx, dy = junction.direction_xy
    px, py = (-dy, dx)
    along = tuple((x * dx + y * dy for x, y in exterior))
    across = tuple((x * px + y * py for x, y in exterior))
    along_min, along_max = (min(along), max(along))
    across_min, across_max = (min(across), max(across))
    tolerance = 1e-09 * max(
        1.0, abs(along_min), abs(along_max), abs(across_min), abs(across_max)
    )
    if along_max - along_min <= tolerance or across_max - across_min <= tolerance:
        raise ValueError(
            f"junction {junction.junction_id!r} polygon has empty projected extent"
        )
    expected = {
        (along_min, across_min),
        (along_min, across_max),
        (along_max, across_min),
        (along_max, across_max),
    }
    if any(
        (
            not any(
                (
                    abs(u - eu) <= tolerance and abs(v - ev) <= tolerance
                    for eu, ev in expected
                )
            )
            for u, v in zip(along, across, strict=False)
        )
    ):
        raise ValueError(
            f"junction {junction.junction_id!r} polygon is not rectangular in its declared direction"
        )
    width = across_max - across_min
    if not math.isclose(width, junction.width_um, rel_tol=1e-09, abs_tol=1e-09):
        raise ValueError(
            f"junction {junction.junction_id!r} width disagrees with its polygon"
        )
    across_mid = (across_min + across_max) / 2.0
    line = [
        [along_min * dx + across_mid * px, along_min * dy + across_mid * py, z_um],
        [along_max * dx + across_mid * px, along_max * dy + across_mid * py, z_um],
    ]
    return (line, along_max - along_min)
