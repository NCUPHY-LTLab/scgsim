"""Where along a boundary edge a gap lies: the intervals of an edge that face a gap, or lie on the die edge.

Owns cutting one boundary edge of a support face (a straight edge or an arc, travelled with the face on its left)
at the ends of the gap edges and die edges of another level that lie along it, and saying which of the pieces
face a gap and which of the others lie on the die edge.

A support face on a plane other than its conductor's attachment plane (the face of a finite conductor away from
its substrate: the top on an upright die, the bottom on a flipped one) is bounded by the
same source edges as the footprint on the attachment plane, but a source edge may face a gap only along part of its
length there.  The two levels are noded separately, so their pieces of one source edge end at different vertices; the
interval a gap edge covers along the source edge is what both levels agree on.  Edges are compared along their shared
source carrier: an end point of a gap edge is a lattice vertex that lies within the snap distance of the carrier,
so positions along the edge are ordered by projection (lines, in integer arithmetic) or by angle about the circle's
centre (arcs, by the sign of a cross product, as an arc spans at most a quarter turn), never by a distance tolerance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import cmp_to_key

from scgsim.geometry.sgb.plane import Point


@dataclass(frozen=True)
class Piece:
    """A stretch of a boundary edge between two lattice points: whether it faces a gap, or else lies on the die edge."""

    start: Point
    end: Point
    faces_gap: bool
    on_die_edge: bool = False


@dataclass(frozen=True)
class LineSpan:
    """A straight edge between two lattice points."""

    start: Point
    end: Point

    def _along(self, point: Point) -> int:
        (px, py), (qx, qy) = self.start, self.end
        return (point[0] - px) * (qx - px) + (point[1] - py) * (qy - py)

    def covers(self, point: Point) -> bool:
        """Whether the projection of ``point`` on the edge lies within it."""
        length_squared = self._along(self.end)
        return 0 <= self._along(point) <= length_squared

    def before(self, p: Point, q: Point) -> bool:
        """Whether the projection of ``p`` is nearer the start than that of ``q``."""
        return self._along(p) < self._along(q)

    def part(self, start: Point, end: Point) -> LineSpan:
        """The stretch of this edge between two of its points."""
        return LineSpan(start, end)

    def same_sense(self, other: Span) -> bool:
        """Whether ``other`` is a straight edge pointing the same way."""
        if not isinstance(other, LineSpan):
            return False
        return (other.end[0] - other.start[0]) * (self.end[0] - self.start[0]) + (
            other.end[1] - other.start[1]
        ) * (self.end[1] - self.start[1]) > 0


@dataclass(frozen=True)
class ArcSpan:
    """An arc of at most a quarter turn about ``centre``, between two lattice points."""

    centre: tuple[Fraction, Fraction]
    ccw: bool
    start: Point
    end: Point

    def _cross(self, p: Point, q: Point) -> Fraction:
        px, py = p[0] - self.centre[0], p[1] - self.centre[1]
        qx, qy = q[0] - self.centre[0], q[1] - self.centre[1]
        return px * qy - py * qx

    def covers(self, point: Point) -> bool:
        """Whether the ray from the centre through ``point`` lies within the angular span."""
        from_start = self._cross(self.start, point)
        to_end = self._cross(point, self.end)
        if self.ccw:
            return from_start >= 0 and to_end >= 0
        return from_start <= 0 and to_end <= 0

    def before(self, p: Point, q: Point) -> bool:
        """Whether ``p`` is reached before ``q`` when travelling along the arc."""
        cross = self._cross(p, q)
        return cross > 0 if self.ccw else cross < 0

    def part(self, start: Point, end: Point) -> ArcSpan:
        """The stretch of this arc between two of its points."""
        return ArcSpan(self.centre, self.ccw, start, end)

    def same_sense(self, other: Span) -> bool:
        """Whether ``other`` is an arc about the same centre travelled the same way."""
        return (
            isinstance(other, ArcSpan)
            and other.centre == self.centre
            and other.ccw == self.ccw
        )


Span = LineSpan | ArcSpan
"""An edge as a stretch of its carrier, ordered along its direction of travel."""


def split_at_gaps(
    span: Span, gaps: Sequence[Span], die_edges: Sequence[Span] = ()
) -> list[Piece]:
    """The pieces of an edge, cut where a gap edge or a die edge begins or ends along it.

    A piece faces a gap iff a gap edge of the same sense covers it; a piece that faces no gap lies on the die
    edge iff a die edge of the same sense covers it.  Gap edges that touch or overlap are one gap, so an edge is
    cut only where a gap really begins or ends, and the same holds for die edges.  An edge with no gap and no die
    edge along it is one piece.
    """
    pieces: list[Piece] = []
    for start, end, faces_gap in _covered_runs(span, gaps):
        if faces_gap:
            pieces.append(Piece(start, end, True))
            continue
        pieces.extend(
            Piece(a, b, False, on_die_edge=on_die)
            for a, b, on_die in _covered_runs(span.part(start, end), die_edges)
        )
    return pieces


def _covered_runs(
    span: Span, covers: Sequence[Span]
) -> list[tuple[Point, Point, bool]]:
    """The stretches of an edge in travel order, each with whether an edge of ``covers`` covers it."""
    runs = _merged(span, [run for cover in covers if (run := _overlap(span, cover))])
    stretches: list[tuple[Point, Point, bool]] = []
    position = span.start
    for begin, finish in runs:
        if span.before(position, begin):
            stretches.append((position, begin, False))
        stretches.append((begin, finish, True))
        position = finish
    if span.before(position, span.end):
        stretches.append((position, span.end, False))
    return stretches


def _overlap(span: Span, gap: Span) -> tuple[Point, Point] | None:
    """The stretch of ``span`` a gap edge covers (``None`` when it covers none of it, or only a point)."""
    if not span.same_sense(gap):
        return None
    begins_inside, ends_inside = span.covers(gap.start), span.covers(gap.end)
    if not (begins_inside or ends_inside or gap.covers(span.start)):
        return None
    begin = gap.start if begins_inside else span.start
    finish = gap.end if ends_inside else span.end
    return (begin, finish) if span.before(begin, finish) else None


def _merged(span: Span, runs: list[tuple[Point, Point]]) -> list[tuple[Point, Point]]:
    """The runs in travel order, with those that touch or overlap joined."""

    def order(p: Point, q: Point) -> int:
        return -1 if span.before(p, q) else 1 if span.before(q, p) else 0

    merged: list[tuple[Point, Point]] = []
    for begin, finish in sorted(runs, key=cmp_to_key(lambda a, b: order(a[0], b[0]))):
        if merged and not span.before(merged[-1][1], begin):
            last_begin, last_finish = merged[-1]
            merged[-1] = (
                last_begin,
                finish if span.before(last_finish, finish) else last_finish,
            )
        else:
            merged.append((begin, finish))
    return merged
