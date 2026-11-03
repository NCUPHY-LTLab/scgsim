"""Circular arcs of source boundaries as carriers (stage S2).

Owns turning an authored circle or arc (three points on it, or a closed circle
through three anchors) into the arc carriers the arrangement nodes.  The circle is
computed exactly from rational coordinates; each arc is cut into pieces of at most
a quarter turn at lattice points, so no arc edge is ever a loop or a pair of
parallel edges.

The end points of a source arc are lattice points and stay exact.  The interior
cut points are circle points rounded to the lattice, which moves them by at most
``sqrt(2)/2`` grid units -- the same bound snap rounding gives every vertex.
"""

from __future__ import annotations

import math
from fractions import Fraction

from scgsim.geometry.sgb.errors import InputError
from scgsim.geometry.sgb.plane import Point
from scgsim.geometry.sgb.records.carriers import ArcCarrier, Lineage, round_to_lattice

RationalPoint = tuple[Fraction, Fraction]
QUARTER_TURN = math.pi / 2
PIECE_ANGLE_SLACK = (
    1e-9  # a quarter turn computed in floating point may exceed pi/2 by rounding
)


def circumcircle(
    a: RationalPoint, b: RationalPoint, c: RationalPoint, what: str
) -> tuple[RationalPoint, Fraction]:
    """Exact centre and (high-precision) radius of the circle through three points."""
    determinant = 2 * (
        a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])
    )
    if determinant == 0:
        raise InputError(
            f"{what}: the three arc points are collinear, so they define no circle"
        )
    a_sq, b_sq, c_sq = (p[0] ** 2 + p[1] ** 2 for p in (a, b, c))
    ux = (
        a_sq * (b[1] - c[1]) + b_sq * (c[1] - a[1]) + c_sq * (a[1] - b[1])
    ) / determinant
    uy = (
        a_sq * (c[0] - b[0]) + b_sq * (a[0] - c[0]) + c_sq * (b[0] - a[0])
    ) / determinant
    radius_sq = (a[0] - ux) ** 2 + (a[1] - uy) ** 2
    return (ux, uy), Fraction(math.sqrt(radius_sq))


def travels_counter_clockwise(
    a: RationalPoint, b: RationalPoint, c: RationalPoint
) -> bool:
    """Whether the arc ``a -> b -> c`` runs counter-clockwise around its circle."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) > 0


def arc_through_points(  # noqa: PLR0913 - an arc is defined by its three points and its identity
    arc_id: str,
    start: Point,
    through: RationalPoint,
    end: Point,
    lineage: Lineage,
    what: str,
) -> list[ArcCarrier]:
    """Pieces of the arc from ``start`` through ``through`` to ``end`` (lattice end points)."""
    a: RationalPoint = (Fraction(start[0]), Fraction(start[1]))
    c: RationalPoint = (Fraction(end[0]), Fraction(end[1]))
    centre, radius = circumcircle(a, through, c, what)
    ccw = travels_counter_clockwise(a, through, c)
    t_start = _angle(centre, a)
    t_end = _angle(centre, c)
    sweep = (
        (t_end - t_start) % (2 * math.pi)
        if ccw
        else -((t_start - t_end) % (2 * math.pi))
    )
    return _pieces(
        arc_id, centre, radius, ccw, start, end, t_start, sweep, lineage, what
    )


def full_circle(  # noqa: PLR0913 - a circle is defined by three anchors, a direction and its identity
    arc_id: str,
    anchors: tuple[Point, Point, Point],
    ccw: bool,
    lineage: Lineage,
    what: str,
) -> list[ArcCarrier]:
    """A full circle through three lattice anchors, starting and ending at the first anchor."""
    points = [(Fraction(p[0]), Fraction(p[1])) for p in anchors]
    centre, radius = circumcircle(points[0], points[1], points[2], what)
    sweep = 2 * math.pi if ccw else -2 * math.pi
    return _pieces(
        arc_id,
        centre,
        radius,
        ccw,
        anchors[0],
        anchors[0],
        _angle(centre, points[0]),
        sweep,
        lineage,
        what,
    )


def _angle(centre: RationalPoint, point: RationalPoint) -> float:
    return math.atan2(float(point[1] - centre[1]), float(point[0] - centre[0]))


def _pieces(  # noqa: PLR0913 - the circle, the sweep and the identity travel together
    arc_id: str,
    centre: RationalPoint,
    radius: Fraction,
    ccw: bool,
    start: Point,
    end: Point,
    t_start: float,
    sweep: float,
    lineage: Lineage,
    what: str,
) -> list[ArcCarrier]:
    count = max(1, math.ceil(abs(sweep) / QUARTER_TURN - PIECE_ANGLE_SLACK))
    cx, cy, r = float(centre[0]), float(centre[1]), float(radius)
    points: list[Point] = [start]
    for k in range(1, count):
        angle = t_start + sweep * k / count
        cut = (Fraction(cx + r * math.cos(angle)), Fraction(cy + r * math.sin(angle)))
        points.append(round_to_lattice(cut))
    points.append(end)
    if any(p == q for p, q in zip(points, points[1:], strict=False)):
        raise InputError(
            f"{what}: the arc is too small for the grid (radius {r:.2f} grid units); "
            "its pieces collapse onto one lattice point"
        )
    return [
        ArcCarrier(
            f"{arc_id}/q{k}",
            centre,
            radius,
            ccw,
            (Fraction(p[0]), Fraction(p[1])),
            (Fraction(q[0]), Fraction(q[1])),
            lineage,
            arc_id,
        )
        for k, (p, q) in enumerate(zip(points, points[1:], strict=False))
    ]
