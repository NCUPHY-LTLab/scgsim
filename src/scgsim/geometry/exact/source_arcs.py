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

from scgsim.geometry.exact import circle_math
from scgsim.geometry.exact.carriers import ArcCarrier, Lineage
from scgsim.geometry.exact.errors import InputError
from scgsim.geometry.exact.grid import Grid
from scgsim.geometry.exact.plane import Point
from scgsim.geometry.models.grid import format_length, format_point
from scgsim.geometry.models.input import twice_area

RationalPoint = tuple[Fraction, Fraction]
QUARTER_TURN = circle_math.HALF_PI
# Authored middle points carry float noise, so a quarter turn may exceed pi/2 by a few 1e-16.
PIECE_ANGLE_SLACK = Decimal("1e-9")


def circumcircle(
    a: RationalPoint, b: RationalPoint, c: RationalPoint, what: str
) -> tuple[RationalPoint, Fraction]:
    """Exact centre and (high-precision) radius of the circle through three points.

    The binding (``prisms.curves``) refuses three points ``collinear`` finds on one line
    first, on the same exact points, naming where they lie and who must act; this refusal
    keeps a direct caller fail-closed.
    """
    determinant = 2 * twice_area(a, b, c)
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
    *,
    grid: Grid,
) -> list[ArcCarrier]:
    """Pieces of the arc from ``start`` through ``through`` to ``end`` (lattice end points).

    ``grid`` converts a refusal's location to micrometres.
    """
    a: RationalPoint = (Fraction(start[0]), Fraction(start[1]))
    c: RationalPoint = (Fraction(end[0]), Fraction(end[1]))
    centre, radius = circumcircle(a, through, c, what)
    ccw = travels_counter_clockwise(a, through, c)
    t_start = circle_math.vector_angle(centre, a)
    t_end = circle_math.vector_angle(centre, c)
    magnitude = circle_math.sweep_magnitude(t_start, t_end, ccw=ccw)
    return _pieces(
        _Circle(arc_id, centre, radius, ccw),
        (start, end),
        magnitude,
        lineage,
        _ArcSource(what, grid),
    )


def full_circle(  # noqa: PLR0913 - a circle is defined by its anchors, its direction and its identity
    arc_id: str,
    anchors: tuple[Point, Point, Point],
    ccw: bool,
    lineage: Lineage,
    what: str,
    *,
    grid: Grid,
) -> list[ArcCarrier]:
    """A full circle through three lattice anchors, starting and ending at the first anchor.

    ``grid`` converts a refusal's location to micrometres.
    """
    points = [(Fraction(p[0]), Fraction(p[1])) for p in anchors]
    centre, radius = circumcircle(points[0], points[1], points[2], what)
    return _pieces(
        _Circle(arc_id, centre, radius, ccw),
        (anchors[0], anchors[0]),
        circle_math.TWO_PI,
        lineage,
        _ArcSource(what, grid),
    )


@dataclass(frozen=True)
class _Circle:
    """The circle an arc lies on and the direction it is travelled."""

    arc_id: str
    centre: RationalPoint
    radius: Fraction
    ccw: bool


@dataclass(frozen=True)
class _ArcSource:
    """What names the arc in a refusal, and the grid that locates it in micrometres."""

    what: str
    grid: Grid


def _pieces(
    circle: _Circle,
    ends: tuple[Point, Point],
    magnitude: Decimal,
    lineage: Lineage,
    source: _ArcSource,
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
    collapsed = next(
        (p for p, q in zip(points, points[1:], strict=False) if p == q), None
    )
    if collapsed is not None:
        raise _too_small(circle, collapsed, source)
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


def _too_small(circle: _Circle, collapsed: Point, source: _ArcSource) -> InputError:
    """The refusal of an arc whose pieces collapse: its size and where it is, in micrometres."""
    grid = source.grid
    size = grid.size_um

    def point(p: tuple[Fraction, Fraction] | Point) -> str:
        return format_point(grid.to_um(p[0]), grid.to_um(p[1]), size)

    return InputError(
        f"{source.what}: the arc is too small for the grid (radius "
        f"{format_length(grid.to_um(circle.radius), size)}, centre {point(circle.centre)}); "
        f"its pieces collapse onto the lattice point {point(collapsed)}. Fix the layout: "
        "draw the arc with a radius the grid resolves, or as straight edges."
    )
