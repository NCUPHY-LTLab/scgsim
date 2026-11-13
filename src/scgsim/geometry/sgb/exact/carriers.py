"""Curve carriers: the exact curves that edges of the arrangement come from.

A carrier keeps its exact definition (line through two integer points, or a
circular arc of a circle) and a lineage record naming the source entity,
polygon, ring and edge it came from, or the cut-family member that generated
it.  Arrangement edges hold :class:`EdgeCarrier` values -- a carrier's id with
its role and, for a piece of a curve, the curve and the piece number -- so every
edge can be traced back to the source curve it belongs to, and no stage reads a
role or a curve back out of an id.

Circular arcs are noded through a polyline witness (chords with sagitta at most
a quarter of the grid): the chord pieces carry ids ``<arc id>#<k>`` and merge
back into one arc edge after dissolution (see ``arrangement.arcs``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from functools import lru_cache
from typing import Literal

from scgsim.geometry.sgb.exact import circle_math
from scgsim.geometry.sgb.exact.plane import Point

PIECE_SEPARATOR = "#"
"""Separator in the id of the k-th witness chord of an arc carrier: ``<arc id>#<k>``; ids are written with it,
never split at it (the curve and the piece number are fields of :class:`EdgeCarrier`)."""

WITNESS_SAGITTA = 0.25
"""Largest sagitta of a witness chord, in grid units."""

WITNESS_CACHE_SIZE = 1 << 14
"""How many distinct arc geometries keep their witness: several stages ask for the witness of the same
arc, and a banded die has a few thousand distinct arcs."""

RationalPoint = tuple[Fraction, Fraction]


BandMemberRole = Literal["edge", "corner", "capstart", "capend"]
"""How a band curve follows its support: ``edge`` offsets one primitive, ``corner`` turns round the reflex
vertex after it, ``capstart`` and ``capend`` close round the start or end of a run of distance origins."""


@dataclass(frozen=True)
class BandMember:
    """Where on its support an inset band curve comes from, and by which margin it is offset.

    ``ring`` and ``index`` locate it among the rings of the family's support region
    (``CurveFamilyRecord.support``).  ``edge`` is the offset of primitive ``index`` of that ring; ``corner`` the
    arc about the reflex vertex between primitives ``index`` and ``index + 1`` (the first one after the last);
    ``capstart`` and ``capend`` the half circle about the start or end of primitive ``index``.  ``margin`` is
    the offset, in grid units.  ``piece`` numbers the parts of a cap or corner cut into several (an arc is at
    most a quarter turn), and is ``None`` for a member in one piece.  The band numbers on the two sides of the
    curve are its seam's (``BandSeam``), and a carrier's id or member name is never read for any of this.
    """

    ring: int
    index: int
    role: BandMemberRole
    margin: int
    piece: int | None = None


@dataclass(frozen=True)
class Lineage:
    """Where a carrier came from: a source polygon edge or a curve-family member.

    ``band_member`` is set on the members of an inset band family only.
    """

    kind: Literal["polygon_edge", "family_member"]
    entity_id: str = ""
    polygon: int = 0
    ring: int = 0
    edge: int = 0
    family_id: str = ""
    member: str = ""
    band_member: BandMember | None = None

    def describe(self) -> str:
        """Stable text form; for polygon edges it is the carrier id of a straight edge."""
        if self.kind == "polygon_edge":
            return f"{self.entity_id}/polygon{self.polygon}/ring{self.ring}/edge{self.edge}"
        return f"{self.family_id}/{self.member}"


class CarrierRole(Enum):
    """What a carrier of an arrangement edge stands for, so that a stage reads it from a field, never from an id."""

    CURVE = "curve"
    """A source edge or arc, or a cut-family member: a curve SGB keeps a record of."""
    SLAB_EDGE = "slab_edge"
    """An edge of a slab arrangement entering a level arrangement (its id is ``linking.TAG`` + the edge id); it is
    how assembly finds the level edges that cover a slab edge."""
    MEMBERSHIP = "membership"
    """A slab edge that dissolved, entering a level arrangement only to cut cells for area-weighted membership;
    never required."""
    SUPPORT_BOUNDARY = "support_boundary"
    """The boundary of an inset family's support, in the arrangement of one margin (never a seam)."""
    PORT_BOUNDARY = "port_boundary"
    """A port footprint boundary, in the arrangement of one margin (a clip boundary, never an origin)."""


@dataclass(frozen=True, order=True)
class EdgeCarrier:
    """One carrier of an arrangement edge: its id, its role and, for a piece of a curve, the curve and the piece.

    Equality, hashing and order are those of ``id`` alone, so a set of carriers and its sorted order are those of
    their ids.  ``curve`` is the id of the curve a piece belongs to (an arc, for its ``k``-th witness chord), and
    ``piece`` its number along it; both are empty for a carrier that is a whole curve.  :meth:`whole` is the
    carrier of the curve itself, which dissolution merges consecutive pieces into.
    """

    id: str
    role: CarrierRole = field(default=CarrierRole.CURVE, compare=False)
    curve: str = field(default="", compare=False)
    piece: int | None = field(default=None, compare=False)

    @property
    def curve_id(self) -> str:
        """The id of the curve this carrier lies on: the curve of a piece, else its own id."""
        return self.curve or self.id

    def whole(self) -> EdgeCarrier:
        """The carrier of the curve this carrier is a piece of (itself when it is a whole curve)."""
        return self if self.piece is None else EdgeCarrier(self.curve, self.role)


def piece_carriers(
    curve_id: str, count: int, role: CarrierRole = CarrierRole.CURVE
) -> list[EdgeCarrier]:
    """The carriers of the ``count`` pieces of a curve, in order: ids ``<curve id>#<k>``."""
    return [
        EdgeCarrier(f"{curve_id}{PIECE_SEPARATOR}{k}", role, curve_id, k)
        for k in range(count)
    ]


LineRole = Literal["source_edge", "arc_chord", "family_line"]
"""What a straight carrier is: ``source_edge`` a straight edge of a prism outline (a source curve of its own),
``arc_chord`` a witness chord of a source arc (it lies on the arc's curve), ``family_line`` a straight curve of a
cut family."""


@dataclass(frozen=True)
class LineCarrier:
    """A straight curve with exact integer endpoints, a lineage record and its role.

    ``role`` is set where the carrier is made, so no stage tells a source edge from a chord or a family line by
    its id.  ``curve`` and ``piece`` are set on a witness chord of a source arc: the arc's id and the chord's
    number along it.  :meth:`edge_carrier` is what the carrier enters an arrangement as.
    """

    id: str
    p: Point
    q: Point
    lineage: Lineage
    role: LineRole
    curve: str = ""
    piece: int | None = None

    def edge_carrier(self) -> EdgeCarrier:
        """This carrier as an arrangement edge holds it."""
        return EdgeCarrier(self.id, CarrierRole.CURVE, self.curve, self.piece)


@dataclass(frozen=True)
class KnownCarrier:
    """What the catalogue records about one carrier id: its lineage and the curve it lies on.

    ``curve`` is the carrier's own id for a whole curve, and the arc's id for a witness chord of an arc.
    """

    lineage: Lineage
    curve: str


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
        t0, magnitude = _start_and_sweep(self.centre, self.start, self.end, self.ccw)
        with circle_math.working_precision():
            return t0, magnitude if self.ccw else -magnitude

    def witness_points(self) -> list[Point]:
        """Grid points ``start .. end`` of the polyline witness (chord sagitta at most g/4).

        Computed in fixed-precision decimals (``records.circle_math``), so the lattice
        points -- and with them every id downstream -- are identical on every platform.
        The witness depends only on the arc's geometry, so it is computed once per geometry.
        """
        return list(_witness(self.centre, self.radius, self.ccw, self.start, self.end))

    def pieces(self, role: CarrierRole = CarrierRole.CURVE) -> list[EdgeCarrier]:
        """The carriers of the witness chords, in order from start to end (ids ``<arc id>#<k>``)."""
        witness = _witness(self.centre, self.radius, self.ccw, self.start, self.end)
        return piece_carriers(self.id, len(witness) - 1, role)


def _start_and_sweep(
    centre: RationalPoint, start: RationalPoint, end: RationalPoint, ccw: bool
) -> tuple[Decimal, Decimal]:
    """Start angle and unsigned sweep of the arc ``start -> end`` about ``centre``."""
    t0 = circle_math.vector_angle(centre, start)
    t1 = circle_math.vector_angle(centre, end)
    return t0, circle_math.sweep_magnitude(t0, t1, ccw=ccw)


@lru_cache(maxsize=WITNESS_CACHE_SIZE)
def _witness(
    centre: RationalPoint,
    radius: Fraction,
    ccw: bool,
    start: RationalPoint,
    end: RationalPoint,
) -> tuple[Point, ...]:
    """The witness of :meth:`ArcCarrier.witness_points`: equal steps of sagitta at most g/4, rounded."""
    with circle_math.working_precision():
        exact_radius = circle_math.to_decimal(radius)
        _, magnitude = _start_and_sweep(centre, start, end, ccw)
        max_step = (
            2 * circle_math.acos(1 - Decimal(WITNESS_SAGITTA) / exact_radius)
            if exact_radius > Decimal(WITNESS_SAGITTA)
            else circle_math.PI
        )
        steps = max(1, circle_math.ceil_div(magnitude, max_step))
        centre_x = circle_math.to_decimal(centre[0])
        centre_y = circle_math.to_decimal(centre[1])
        x = circle_math.to_decimal(start[0] - centre[0])
        y = circle_math.to_decimal(start[1] - centre[1])
        rotation = circle_math.Rotation(magnitude / steps, ccw=ccw)
        points: list[Point] = [round_to_lattice(start)]
        for _ in range(1, steps):
            x, y = rotation.apply(x, y)
            point = (
                circle_math.round_half_up(centre_x + x),
                circle_math.round_half_up(centre_y + y),
            )
            if point != points[-1]:
                points.append(point)
    last = round_to_lattice(end)
    if last != points[-1]:
        points.append(last)
    return tuple(points)


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
