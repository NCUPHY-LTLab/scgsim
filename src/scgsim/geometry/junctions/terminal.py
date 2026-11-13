"""Validate rectangular terminal geometry against the shared structural contract."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Protocol

from scgsim.geometry.junctions.exact_geometry import cross
from scgsim.geometry.models.refusals import Declared


def _junction(junction: object) -> Declared:
    """A junction's sheet, which the layout draws."""
    return Declared(
        f"the sheet of junction {getattr(junction, 'junction_id', None)!r}",
        "the layout: draw the sheet as one rectangle along the port direction",
    )


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


def _corner_points(ring) -> tuple[tuple[float, float], ...]:
    """A ring's corners: its points without repeats and without points on a straight run, decided exactly."""
    exterior = tuple(((float(point[0]), float(point[1])) for point in ring))

    def rational(p):
        return tuple((Fraction(str(v)) for v in p))

    witness = tuple(
        (point for index, point in enumerate(exterior) if point != exterior[index - 1])
    )
    return tuple(
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


def junction_terminal_line(
    polygon: Mapping[str, Any], junction: JunctionTerminalGeometry, z_um: float
) -> tuple[list[list[float]], float]:
    """Validate a rectangular junction sheet and return its axial terminal line."""
    exterior = _corner_points(polygon["exterior"])
    if len(exterior) != 4 or polygon.get("holes"):
        raise ValueError(
            _junction(junction).refusal(
                "it requires one rectangular polygon without holes"
            )
        )
    dx, dy = junction.direction_xy
    px, py = (-dy, dx)
    projected = _Projection.of(exterior, junction.direction_xy)
    along_min, along_max = (min(projected.along), max(projected.along))
    across_min, across_max = (min(projected.across), max(projected.across))
    tolerance = projected.tolerance
    if along_max - along_min <= tolerance or across_max - across_min <= tolerance:
        raise ValueError(_junction(junction).refusal("it has empty projected extent"))
    if not projected.corners_on_extent():
        raise ValueError(
            _junction(junction).refusal(
                "it is not rectangular in its declared direction"
            )
        )
    width = across_max - across_min
    if not math.isclose(width, junction.width_um, rel_tol=1e-09, abs_tol=1e-09):
        raise ValueError(
            _junction(junction).refusal(
                f"its width disagrees with the junction's width {junction.width_um} um"
            )
        )
    across_mid = (across_min + across_max) / 2.0
    line = [
        [along_min * dx + across_mid * px, along_min * dy + across_mid * py, z_um],
        [along_max * dx + across_mid * px, along_max * dy + across_mid * py, z_um],
    ]
    return (line, along_max - along_min)


def rectangle_along(
    corners: Sequence[Sequence[float]], direction_xy: tuple[float, float]
) -> bool:
    """Whether four corners are a rectangle with its sides along and across ``direction_xy``.

    This is the comparison ``junction_terminal_line`` refuses a sheet by, with its tolerance.
    """
    return _Projection.of(corners, direction_xy).corners_on_extent()


@dataclass(frozen=True)
class _Projection:
    """Points measured along a direction and across it, with the rectangle check's tolerance."""

    along: tuple[float, ...]
    across: tuple[float, ...]
    tolerance: float

    @classmethod
    def of(cls, points, direction_xy) -> _Projection:
        dx, dy = direction_xy
        along = tuple(x * dx + y * dy for x, y in points)
        across = tuple(-x * dy + y * dx for x, y in points)
        # Relative to the coordinates' magnitude, so a sheet far from the origin is judged alike.
        tolerance = 1e-09 * max(1.0, *(abs(v) for v in (*along, *across)))
        return cls(along, across, tolerance)

    def corners_on_extent(self) -> bool:
        """Whether every point lies on a corner of the extent, within the tolerance."""
        expected = [
            (u, v)
            for u in (min(self.along), max(self.along))
            for v in (min(self.across), max(self.across))
        ]
        return all(
            any(
                abs(u - eu) <= self.tolerance and abs(v - ev) <= self.tolerance
                for eu, ev in expected
            )
            for u, v in zip(self.along, self.across, strict=False)
        )
