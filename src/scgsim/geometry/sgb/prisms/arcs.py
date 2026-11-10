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

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

from scgsim.geometry.sgb.errors import InputError
from scgsim.geometry.sgb.plane import Point
from scgsim.geometry.sgb.records import circle_math
from scgsim.geometry.sgb.records.carriers import ArcCarrier, Lineage

RationalPoint = tuple[Fraction, Fraction]
QUARTER_TURN = circle_math.HALF_PI
# Authored middle points carry float noise, so a quarter turn may exceed pi/2 by a few 1e-16.
PIECE_ANGLE_SLACK = Decimal("1e-9")


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
    radius = Fraction(circle_math.sqrt(circle_math.to_decimal(radius_sq)))
    return (ux, uy), radius


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
    t_start = circle_math.vector_angle(centre, a)
    t_end = circle_math.vector_angle(centre, c)
    magnitude = circle_math.sweep_magnitude(t_start, t_end, ccw=ccw)
    return _pieces(
        _Circle(arc_id, centre, radius, ccw), (start, end), magnitude, lineage, what
    )


def full_circle(
    arc_id: str,
    anchors: tuple[Point, Point, Point],
    ccw: bool,
    lineage: Lineage,
    what: str,
) -> list[ArcCarrier]:
    """A full circle through three lattice anchors, starting and ending at the first anchor."""
    points = [(Fraction(p[0]), Fraction(p[1])) for p in anchors]
    centre, radius = circumcircle(points[0], points[1], points[2], what)
    return _pieces(
        _Circle(arc_id, centre, radius, ccw),
        (anchors[0], anchors[0]),
        circle_math.TWO_PI,
        lineage,
        what,
    )


@dataclass(frozen=True)
class _Circle:
    """The circle an arc lies on and the direction it is travelled."""

    arc_id: str
    centre: RationalPoint
    radius: Fraction
    ccw: bool


def _pieces(
    circle: _Circle,
    ends: tuple[Point, Point],
    magnitude: Decimal,
    lineage: Lineage,
    what: str,
) -> list[ArcCarrier]:
    """Cut an arc of the given sweep into pieces of at most a quarter turn at lattice points."""
    start, end = ends
    centre = circle.centre
    points: list[Point] = [start]
    with circle_math.working_precision():
        count = max(
            1, circle_math.ceil_div(magnitude - PIECE_ANGLE_SLACK, QUARTER_TURN)
        )
        x = circle_math.to_decimal(start[0] - centre[0])
        y = circle_math.to_decimal(start[1] - centre[1])
        rotation = circle_math.Rotation(magnitude / count, ccw=circle.ccw)
        centre_x, centre_y = (
            circle_math.to_decimal(centre[0]),
            circle_math.to_decimal(centre[1]),
        )
        for _ in range(1, count):
            x, y = rotation.apply(x, y)
            points.append(
                (
                    circle_math.round_half_up(centre_x + x),
                    circle_math.round_half_up(centre_y + y),
                )
            )
    points.append(end)
    if any(p == q for p, q in zip(points, points[1:], strict=False)):
        raise InputError(
            f"{what}: the arc is too small for the grid (radius {float(circle.radius):.2f} grid "
            "units); its pieces collapse onto one lattice point"
        )
    return [
        ArcCarrier(
            f"{circle.arc_id}/q{k}",
            circle.centre,
            circle.radius,
            circle.ccw,
            (Fraction(p[0]), Fraction(p[1])),
            (Fraction(q[0]), Fraction(q[1])),
            lineage,
            circle.arc_id,
        )
        for k, (p, q) in enumerate(zip(points, points[1:], strict=False))
    ]
