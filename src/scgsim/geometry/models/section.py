"""The cut line of a 2D model and the line cross-section found on it.

A 2D (cross-section) model stands for an infinitely long, uniform line with one cross-section.  A
:class:`CutLine` says where that cross-section is taken: from ``start_um`` to ``end_um`` in the plan's assembly
frame, the frame of selector points, in micrometres.  The cut runs along x or along y; SGB checks the rest
against the model it cuts (the ends on the model grid, the cut crossing the line's edges at a right angle, no
arc, both ends on the edge of the model's planar extent).

A :class:`LineCrossSection` is what lies on the cut, classified: the conductor pieces in order along it, each
with its role (``signal`` or ``reference``), the gaps between them and the dielectric and vacuum faces around
them.  SGB classifies a section into it (``scgsim.geometry.sgb.section.lines``); both 2D solvers read it, so
Palace and Q2D take the same cross-section.  It is detached from the model it was read from, and nothing in it
depends on the route the model was built on.

The section's own frame: ``u`` is the distance from ``start_um`` along the cut and the second coordinate is
the stack height ``z``, both in micrometres.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import isfinite
from numbers import Real
from typing import Literal

from scgsim.geometry.models.grid import format_box
from scgsim.refusal import Declared
from scgsim.semantics.snapshot import canonical_sha256, to_plain

PlanPoint = tuple[float, float]
Span = tuple[float, float]
"""A range ``(low, high)`` in micrometres: of ``u`` along the cut, or of the height ``z``."""
Role = Literal["signal", "reference"]
DielectricKind = Literal["dielectric", "vacuum"]
DIGEST_SCHEMA = "scgsim.line-cross-section.v1"
"""The name the digest hashes with the record: a change of the record's fields changes it too."""


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


# --------------------------------------------------------------------------- the line cross-section


@dataclass(frozen=True)
class LineConductor:
    """One conductor piece on the cut: a connected part of one Entity's cross-section.

    ``entity`` is the owning Entity and ``net`` its Net.  ``role`` is ``reference`` for the Net found on both
    outer sides of the line and ``signal`` for a piece between them.  ``u_um`` and ``z_um`` bound the piece.
    """

    entity: str
    net: str
    role: Role
    material: str
    u_um: Span
    z_um: Span


@dataclass(frozen=True)
class LineGap:
    """The bare stretch of the cut between two neighbouring conductor pieces.

    ``u_um`` runs from the wall of the piece before it to the wall of the piece after it, ``z_um`` is the film
    the two walls face each other across, and ``nets`` are the Nets of those two pieces, in order along the cut.
    """

    u_um: Span
    z_um: Span
    nets: tuple[str, str]


@dataclass(frozen=True)
class LineDielectric:
    """One dielectric or vacuum face of the section: its owning Entity, its material and the range it covers."""

    entity: str
    kind: DielectricKind
    material: str
    u_um: Span
    z_um: Span


@dataclass(frozen=True)
class LineCrossSection:
    """The classified cross-section of a line on one cut, detached from the model it was read from.

    ``conductors`` are in order along the cut, ``gaps`` lie between consecutive conductors, and ``dielectrics``
    are ordered by height, then along the cut.  Every field holds the same values whichever route the model was
    built on, so two records of one plan at one cut are equal, and so are their :attr:`digest`.
    """

    cut: CutLine
    conductors: tuple[LineConductor, ...]
    gaps: tuple[LineGap, ...]
    dielectrics: tuple[LineDielectric, ...]

    @property
    def digest(self) -> str:
        """The SHA-256 of every field as canonical JSON: equal for equal records and the same in every run.

        A simulation rebuilds the record from its own plan and compares this digest, so a plan changed after
        the record was made is found.
        """
        return canonical_sha256(
            {"schema": DIGEST_SCHEMA, "record": to_plain(asdict(self))}
        )

    def describe(self) -> str:
        """One line for a reader: the pieces and gaps with their widths, the film, what it stands on and the cut.

        For the finite-ground CPW coupon: ``ground 80 | 6 | signal 10 | 6 | ground 80 um; Al 0.1 um on Si
        500 um; cut at x = 0 um``.
        """
        parts = [f"{self.conductors[0].net} {_width(self.conductors[0].u_um):g}"]
        for gap, after in zip(self.gaps, self.conductors[1:], strict=True):
            parts += [f"{_width(gap.u_um):g}", f"{after.net} {_width(after.u_um):g}"]
        film = self.conductors[0].z_um
        materials = dict.fromkeys(c.material for c in self.conductors)
        line = f"{' | '.join(parts)} um; {'/'.join(materials)} {_width(film):g} um"
        under = [
            d
            for d in self.dielectrics
            if d.kind == "dielectric" and d.z_um[1] == film[0]
        ]
        if under:
            line += f" on {under[0].material} {_width(under[0].z_um):g} um"
        across, value = (
            ("x", self.cut.start_um[0])
            if self.cut.axis == "y"
            else ("y", self.cut.start_um[1])
        )
        return f"{line}; cut at {across} = {value:g} um"


def _width(span: Span) -> float:
    return span[1] - span[0]
