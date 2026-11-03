"""Curve carriers: the exact curves that edges of the arrangement come from.

A carrier keeps its exact definition (line through two integer points, or a
circular arc of a circle) and a lineage record naming the source entity,
polygon, ring and edge it came from, or the cut-family member that generated
it.  Arrangement edges refer to carriers by id, so every edge can be traced
back to the source curve it belongs to.

Circular arcs are noded through a polyline witness (chords with sagitta at most
a quarter of the grid): the chord pieces carry ids ``<arc id>#<k>`` and merge
back into one arc edge after dissolution (see ``arrangement.arcs``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from decimal import Decimal
from fractions import Fraction
from typing import Literal

from scgsim.geometry.sgb.plane import Point
from scgsim.geometry.sgb.records import circle_math

MEMBER_ONLY = "~m:"
"""Prefix of carrier ids that exist only to cut cells for area-weighted membership; never required."""

PIECE_SEPARATOR = "#"
"""Separator in the id of the k-th witness chord of an arc carrier: ``<arc id>#<k>``."""

WITNESS_SAGITTA = 0.25
"""Largest sagitta of a witness chord, in grid units."""


@dataclass(frozen=True)
class Lineage:
    """Where a carrier came from: a source polygon edge or a curve-family member."""

    kind: Literal["polygon_edge", "family_member"]
    entity_id: str = ""
    polygon: int = 0
    ring: int = 0
    edge: int = 0
    family_id: str = ""
    member: str = ""

    def describe(self) -> str:
        """Stable text form; for polygon edges it is the carrier id of a straight edge."""
        if self.kind == "polygon_edge":
            return f"{self.entity_id}/polygon{self.polygon}/ring{self.ring}/edge{self.edge}"
        return f"{self.family_id}/{self.member}"


@dataclass(frozen=True)
class LineCarrier:
    """A straight curve with exact integer endpoints and a lineage record."""

    id: str
    p: Point
    q: Point
    lineage: Lineage


def base_carrier_id(carrier_id: str) -> str:
    """Id of the carrier a (possibly witness-piece) carrier id belongs to."""
    return carrier_id.split(PIECE_SEPARATOR, 1)[0]


@dataclass(frozen=True)
class LineEdge:
    """A straight source edge of a ring that may also contain arcs.

    ``source_edge`` is the edge number lineage reports (the index of the edge in
    the source ring); ``-1`` numbers the edge by its position in the ring.
    """

    p: Point
    q: Point
    source_edge: int = -1

    def reversed(self) -> LineEdge:
        """The same edge travelled the other way."""
        return LineEdge(self.q, self.p, self.source_edge)


def round_to_lattice(point: tuple[Fraction, Fraction]) -> Point:
    """Nearest lattice point of a grid-unit point (half rounds up)."""
    half = Fraction(1, 2)
    return (math.floor(point[0] + half), math.floor(point[1] + half))


@dataclass(frozen=True)
class ArcCarrier:
    """A circular arc (less than 360 degrees) with exact centre in grid units.

    ``start`` and ``end`` are grid-unit points on the circle: lattice points for
    source arcs and snapped lattice points for derived pieces.  ``ccw`` is the
    direction of travel from ``start`` to ``end``.  A full circle is several
    arcs of at most a quarter turn, so no arc edge is ever a loop or a pair of
    parallel edges.  ``radius`` is a high-precision rational approximation used
    for realisation and the witness only; topology never depends on it.

    ``root`` is the id of the source carrier this arc derives from (drift is
    measured against its exact circle).
    """

    id: str
    centre: tuple[Fraction, Fraction]
    radius: Fraction
    ccw: bool
    start: tuple[Fraction, Fraction]
    end: tuple[Fraction, Fraction]
    lineage: Lineage
    root: str = ""

    @property
    def root_id(self) -> str:
        """Id of the source arc this carrier derives from (itself for a source arc)."""
        return self.root or self.id

    def reversed(self) -> ArcCarrier:
        """The same arc travelled the other way (start and end swap)."""
        return replace(self, ccw=not self.ccw, start=self.end, end=self.start)

    def sweep(self) -> tuple[Decimal, Decimal]:
        """(start angle, signed sweep) in radians as decimals; the sweep is positive when ``ccw``."""
        t0 = circle_math.vector_angle(self.centre, self.start)
        t1 = circle_math.vector_angle(self.centre, self.end)
        magnitude = circle_math.sweep_magnitude(t0, t1, ccw=self.ccw)
        return t0, magnitude if self.ccw else -magnitude

    def witness_points(self) -> list[Point]:
        """Grid points ``start .. end`` of the polyline witness (chord sagitta at most g/4).

        Computed in fixed-precision decimals (``records.circle_math``), so the lattice
        points -- and with them every id downstream -- are identical on every platform.
        """
        radius = circle_math.to_decimal(self.radius)
        _, signed_sweep = self.sweep()
        magnitude = abs(signed_sweep)
        max_step = (
            2 * circle_math.acos(1 - Decimal(WITNESS_SAGITTA) / radius)
            if radius > Decimal(WITNESS_SAGITTA)
            else circle_math.PI
        )
        steps = max(1, circle_math.ceil_div(magnitude, max_step))
        centre_x = circle_math.to_decimal(self.centre[0])
        centre_y = circle_math.to_decimal(self.centre[1])
        x = circle_math.to_decimal(self.start[0] - self.centre[0])
        y = circle_math.to_decimal(self.start[1] - self.centre[1])
        rotation = circle_math.Rotation(magnitude / steps, ccw=self.ccw)
        points: list[Point] = [round_to_lattice(self.start)]
        for _ in range(1, steps):
            x, y = rotation.apply(x, y)
            point = (
                circle_math.round_half_up(centre_x + x),
                circle_math.round_half_up(centre_y + y),
            )
            if point != points[-1]:
                points.append(point)
        end = round_to_lattice(self.end)
        if end != points[-1]:
            points.append(end)
        return points

    def piece_ids(self) -> list[str]:
        """Carrier ids of the witness chords, in order from start to end."""
        count = len(self.witness_points()) - 1
        return [f"{self.id}{PIECE_SEPARATOR}{k}" for k in range(count)]


def circle_arcs(
    circle_id: str,
    centre: Point,
    radius: int,
    ccw: bool,
    lineage: Lineage,
) -> list[ArcCarrier]:
    """A full circle with integer centre and radius as four quarter arcs (exact end points)."""
    cx, cy, r = Fraction(centre[0]), Fraction(centre[1]), Fraction(radius)
    quarter = [(cx + r, cy), (cx, cy + r), (cx - r, cy), (cx, cy - r)]
    order = list(range(4)) if ccw else [0, 3, 2, 1]
    arcs = []
    for i in range(4):
        a, b = quarter[order[i]], quarter[order[(i + 1) % 4]]
        arcs.append(
            ArcCarrier(f"{circle_id}/q{i}", (cx, cy), r, ccw, a, b, lineage, circle_id)
        )
    return arcs


RingPrimitive = LineEdge | ArcCarrier
"""A boundary primitive of a ring: a straight edge or a circular arc."""
