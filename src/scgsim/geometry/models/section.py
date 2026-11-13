"""The cut line of a 2D model: the straight segment across a layout whose plane section is the model.

A 2D (cross-section) model stands for an infinitely long, uniform line with one cross-section.  The study
author declares where that cross-section is taken: a :class:`CutLine` from ``start_um`` to ``end_um`` in the
plan's assembly frame, the frame of selector points, in micrometres.  The cut runs along x or along y; SGB
checks the rest against the model it cuts (the ends on the model grid, the cut crossing the line's edges at a
right angle, no arc, both ends on the edge of the model's planar extent).

The section's own frame: ``u`` is the distance from ``start_um`` along the cut and the second coordinate is
the stack height ``z``, both in micrometres.  The same record serves every 2D consumer, so Q2D can later take
the cut a Palace study used.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Literal

from scgsim.geometry.models.grid import format_box
from scgsim.geometry.models.refusals import Declared

PlanPoint = tuple[float, float]


@dataclass(frozen=True)
class CutLine:
    """A straight cut along x or y from ``start_um`` to ``end_um`` (micrometres, assembly frame)."""

    start_um: PlanPoint
    end_um: PlanPoint

    def __post_init__(self) -> None:
        start = _plan_point(self.start_um, "start_um")
        end = _plan_point(self.end_um, "end_um")
        object.__setattr__(self, "start_um", start)
        object.__setattr__(self, "end_um", end)
        where = f"CutLine {format_box((*start, *end), None)}"
        if start == end:
            raise ValueError(
                Declared(where, "the CutLine call").refusal(
                    "it has no length; give it two different ends across the line"
                )
            )
        if start[0] != end[0] and start[1] != end[1]:
            raise ValueError(
                Declared(where, "the CutLine call").refusal(
                    "it runs neither along x nor along y; give both ends the same y (a cut "
                    "along x) or the same x (a cut along y), across a section of the line "
                    "that runs the other way"
                )
            )

    @property
    def axis(self) -> Literal["x", "y"]:
        """The plan axis the cut runs along."""
        return "x" if self.start_um[1] == self.end_um[1] else "y"

    @property
    def length_um(self) -> float:
        """The length of the cut, the width of the section, in micrometres."""
        index = 0 if self.axis == "x" else 1
        return abs(self.end_um[index] - self.start_um[index])


_CUT_LINE = Declared("CutLine", "the CutLine call")


def _plan_point(value: object, name: str) -> PlanPoint:
    """A point of the plan as two finite floats; anything else is a ``TypeError`` or ``ValueError``."""
    if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 2:
        raise TypeError(
            _CUT_LINE.expected(name, "an (x, y) pair in micrometres", value)
        )
    x, y = value
    for coordinate in (x, y):
        if isinstance(coordinate, bool) or not isinstance(coordinate, Real):
            raise TypeError(
                _CUT_LINE.expected(
                    name, "an (x, y) pair of numbers in micrometres", value
                )
            )
        if not isfinite(coordinate):
            raise ValueError(_CUT_LINE.expected(name, "finite", value))
    return (float(x), float(y))
