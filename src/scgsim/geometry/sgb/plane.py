"""Exact integer plane geometry shared by every stage.

Owns cross products, ring areas, exact point-in-ring tests and bounding boxes
on integer grid points.  All arithmetic is integer or ``Fraction``, so no
topology decision ever depends on a floating-point tolerance.

A *rep point* is the exact rational point ``(X / D, Y / D)`` stored as the
integer triple ``(X, Y, D)`` with ``D > 0``; this keeps point-in-polygon tests
in pure integer arithmetic.

Does not own snapping (``noding``) or faces (``arrangement``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction

Point = tuple[int, int]
Ring = tuple[Point, ...]
RepPoint = tuple[int, int, int]


def cross(o: Point, a: Point, b: Point) -> int:
    """Twice the signed area of triangle (o, a, b); positive when counter-clockwise."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def ring_twice_area(ring: Sequence[Point]) -> int:
    """Twice the signed shoelace area of a closed ring (positive when CCW)."""
    total = 0
    count = len(ring)
    for i in range(count):
        x0, y0 = ring[i]
        x1, y1 = ring[(i + 1) % count]
        total += x0 * y1 - x1 * y0
    return total


def ring_bbox(ring: Sequence[Point]) -> tuple[int, int, int, int]:
    """(xmin, ymin, xmax, ymax) of a ring."""
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return min(xs), min(ys), max(xs), max(ys)


def point_in_ring(ring: Sequence[Point], p: RepPoint) -> bool:
    """Even-odd containment of an exact rational point in a closed ring.

    The caller guarantees the point is not on the ring boundary (rep points are
    strictly interior to arrangement cells); the ray convention below is the
    standard half-open crossing rule, so shared vertices are counted once.
    """
    px, py, den = p
    inside = False
    count = len(ring)
    for i in range(count):
        ax, ay = ring[i]
        bx, by = ring[(i + 1) % count]
        ay_scaled = ay * den
        by_scaled = by * den
        if (ay_scaled > py) == (by_scaled > py):
            continue
        dy = by - ay
        lhs = px * dy
        rhs = ax * den * dy + (bx - ax) * (py - ay_scaled)
        # x_int = rhs / (den * dy) in units of 1/den; compare px < x_int.
        if (dy > 0 and lhs < rhs) or (dy < 0 and lhs > rhs):
            inside = not inside
    return inside


def fraction_to_rep(x: Fraction, y: Fraction) -> RepPoint:
    """Convert a rational point to the common-denominator integer triple."""
    den = math.lcm(x.denominator, y.denominator)
    return (int(x * den), int(y * den), den)


def segment_point_distance_sq(a: Point, b: Point, p: Point) -> Fraction:
    """Exact squared distance from integer point p to segment ab."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    len_sq = dx * dx + dy * dy
    if len_sq == 0:
        return Fraction((p[0] - a[0]) ** 2 + (p[1] - a[1]) ** 2)
    t_num = (p[0] - a[0]) * dx + (p[1] - a[1]) * dy
    if t_num <= 0:
        return Fraction((p[0] - a[0]) ** 2 + (p[1] - a[1]) ** 2)
    if t_num >= len_sq:
        return Fraction((p[0] - b[0]) ** 2 + (p[1] - b[1]) ** 2)
    cross_product = dx * (p[1] - a[1]) - dy * (p[0] - a[0])
    return Fraction(cross_product * cross_product, len_sq)


def segments_properly_cross(a1: Point, b1: Point, a2: Point, b2: Point) -> bool:
    """True when two segments cross at a point interior to both."""
    d1 = cross(a1, b1, a2)
    d2 = cross(a1, b1, b2)
    d3 = cross(a2, b2, a1)
    d4 = cross(a2, b2, b1)
    return (
        ((d1 > 0) != (d2 > 0))
        and d1 != 0
        and d2 != 0
        and ((d3 > 0) != (d4 > 0))
        and d3 != 0
        and d4 != 0
    )
