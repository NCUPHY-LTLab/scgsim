"""The cut line on the model grid: its ends as lattice points and the map between the plan and the section.

The section's frame: ``u`` is the distance from the cut's start along the cut, in grid units, and ``z`` is the
stack height in micrometres, exactly the 3D model's levels.  The cut runs along x or y with both ends on the
grid, so where it crosses an edge of the model at a right angle the crossing is a lattice point and its ``u``
an exact integer: the section rounds nothing the 3D model did not already round.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from scgsim.documentation import see_also
from scgsim.geometry.exact.errors import InputError
from scgsim.geometry.exact.grid import Grid
from scgsim.geometry.exact.plane import Point, RepPoint
from scgsim.geometry.models.grid import format_box, format_point
from scgsim.geometry.models.section import CutLine

Along = int | Fraction | float
"""A position along the cut, in grid units: exact at a crossing, a float only for the place of an arc."""


@dataclass(frozen=True)
class CutFrame:
    """A cut line on one model's grid: its lattice ends and the axis it runs along."""

    cut: CutLine
    grid: Grid
    start: Point
    end: Point

    @property
    def along(self) -> int:
        """The plan coordinate ``u`` follows: 0 when the cut runs along x, 1 along y."""
        return 0 if self.start[1] == self.end[1] else 1

    @property
    def across(self) -> int:
        """The plan coordinate that is constant on the cut."""
        return 1 - self.along

    @property
    def sign(self) -> int:
        """+1 when ``u`` grows with the plan coordinate the cut runs along, -1 when it shrinks."""
        return 1 if self.end[self.along] > self.start[self.along] else -1

    @property
    def length(self) -> int:
        """The length of the cut, in grid units."""
        return abs(self.end[self.along] - self.start[self.along])

    @property
    def line(self) -> int:
        """The value of the constant plan coordinate on the cut, in grid units."""
        return self.start[self.across]

    def u_of(self, point: tuple[Along, Along]) -> Along:
        """The ``u`` of a plan point projected onto the cut, in grid units."""
        return (point[self.along] - self.start[self.along]) * self.sign

    def offset(self, point: tuple[Along, Along]) -> Along:
        """The signed distance of a plan point from the cut's line, in grid units."""
        return point[self.across] - self.line

    def plan_point(self, u: Along) -> tuple[Along, Along]:
        """The plan point at ``u`` on the cut, in grid units."""
        along = self.start[self.along] + self.sign * u
        return (along, self.line) if self.along == 0 else (self.line, along)

    def midpoint(self, u0: int, u1: int) -> RepPoint:
        """The exact plan point halfway between ``u0`` and ``u1`` on the cut, with denominator 2."""
        along = 2 * self.start[self.along] + self.sign * (u0 + u1)
        across = 2 * self.line
        x, y = (along, across) if self.along == 0 else (across, along)
        return (x, y, 2)

    def left_is_after(self, direction: Point) -> bool:
        """Whether the left of an edge travelled along ``direction`` (across the cut) is the side of larger ``u``.

        The left of ``(dx, dy)`` is ``(-dy, dx)``; across the cut it points along the cut, one way or the other.
        """
        left = (-direction[1], direction[0])
        return left[self.along] * self.sign > 0

    def point_text(self, u: Along) -> str:
        """The plan point at ``u`` in micrometres, for messages: ``(x, y) um``."""
        x, y = self.plan_point(u)
        return format_point(self.grid.to_um(x), self.grid.to_um(y), self.grid.size_um)

    def span_text(self, u0: Along, u1: Along) -> str:
        """The plan points from ``u0`` to ``u1`` in micrometres, for messages: ``(x0, y0)..(x1, y1) um``."""
        x0, y0 = self.plan_point(u0)
        x1, y1 = self.plan_point(u1)
        to_um = self.grid.to_um
        return format_box(
            (to_um(x0), to_um(y0), to_um(x1), to_um(y1)), self.grid.size_um
        )

    def text(self) -> str:
        """The cut line as messages name it."""
        return f"the cut line {self.span_text(0, self.length)}"


def cut_frame(cut: CutLine, grid: Grid) -> CutFrame:
    """The cut on the model grid; an end off the grid is refused with its place and the fix."""
    ends = []
    for name, (x_um, y_um) in (("start_um", cut.start_um), ("end_um", cut.end_um)):
        x, y = grid.lattice_units(x_um), grid.lattice_units(y_um)
        if x is None or y is None:
            raise InputError(
                f"the cut line's {name} {format_point(x_um, y_um, None)} is off the {grid.size_um} um "
                "model grid, so its crossings with the layout would not be lattice points. Fix the "
                "cut line: put both ends on the grid."
                + see_also("docs/geometry-sgb", "cut-rules")
            )
        ends.append((x, y))
    return CutFrame(cut, grid, ends[0], ends[1])
