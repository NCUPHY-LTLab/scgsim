"""Compute exact polygon incidence, sections and closed boundary contact."""

from __future__ import annotations

import math
from collections import defaultdict
from fractions import Fraction
from itertools import pairwise


def region_edges(rings):
    """Read the cyclic directed edges of every supplied ring."""
    for ring in rings:
        yield from zip(ring, (*ring[1:], ring[0]), strict=False)


def cross(a, b, c):
    """Return the signed planar cross product about the first point."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def on(point, a, b):
    """Test exact collinearity and closed-segment bounds."""
    return cross(a, b, point) == 0 and all(
        (min(a[i], b[i]) <= point[i] <= max(a[i], b[i]) for i in (0, 1))
    )


def inside_ring(point, ring):
    """Classify a point as inside, on the boundary or outside a ring."""
    inside = False
    for a, b in region_edges((ring,)):
        if on(point, a, b):
            return 0
        if (a[1] > point[1]) != (b[1] > point[1]):
            x = Fraction(b[0] - a[0], b[1] - a[1]) * (point[1] - a[1]) + a[0]
            if point[0] < x:
                inside = not inside
    return 1 if inside else -1


def _inside(point, rings):
    return inside_ring(point, rings[0]) >= 0 and (
        not any((inside_ring(point, hole) == 1 for hole in rings[1:]))
    )


def section(rings, axis, coordinate, low, high):
    """Exact closed contact on a specified edge in Boolean integer coordinates."""
    other = 1 - axis
    marks = {Fraction(low), Fraction(high)}
    for a, b in region_edges(rings):
        if a[axis] == b[axis] == coordinate:
            marks.update(
                (Fraction(v[other]) for v in (a, b) if low <= v[other] <= high)
            )
        elif a[axis] != b[axis] and min(a[axis], b[axis]) <= coordinate <= max(
            a[axis], b[axis]
        ):
            v = (
                Fraction(coordinate - a[axis], b[axis] - a[axis])
                * (b[other] - a[other])
                + a[other]
            )
            if low <= v <= high:
                marks.add(v)
    ordered = sorted(marks)

    def point(v):
        return (coordinate, v) if axis == 0 else (v, coordinate)

    intervals = [
        (a, b) for a, b in pairwise(ordered) if _inside(point((a + b) / 2), rings)
    ]
    points = [v for v in ordered if _inside(point(v), rings)]
    return (intervals, points)


def covered(polygons, axis, coordinate, low, high):
    """Test complete interval coverage by the exact polygon sections."""
    intervals = sorted(
        (
            interval
            for rings in polygons
            for interval in section(rings, axis, coordinate, low, high)[0]
        )
    )
    end = Fraction(low)
    for a, b in intervals:
        if a > end:
            return False
        end = max(end, b)
    return end >= high


def _line_intervals(polygons):
    result = defaultdict(list)
    for rings in polygons:
        for a, b in region_edges(rings):
            dx, dy = (b[0] - a[0], b[1] - a[1])
            divisor = math.gcd(dx, dy)
            if not divisor:
                continue
            dx, dy = (dx // divisor, dy // divisor)
            if dx < 0 or (dx == 0 and dy < 0):
                dx, dy = (-dx, -dy)
            key = (dx, dy, dx * a[1] - dy * a[0])
            u, v = (a[0] * dx + a[1] * dy, b[0] * dx + b[1] * dy)
            result[key].append((min(u, v), max(u, v)))
    return result


def positive_boundary_contact(left, right):
    """Test whether polygon boundaries share a positive-length section."""
    a, b = (_line_intervals(left), _line_intervals(right))
    return any(
        (
            max(x0, y0) < min(x1, y1)
            for key in a.keys() & b.keys()
            for x0, x1 in a[key]
            for y0, y1 in b[key]
        )
    )


def closed_contact(left, right):
    """Test whether two polygon-region collections touch or overlap."""
    for a in left:
        for b in right:
            ax = [p[0] for p in a[0]]
            ay = [p[1] for p in a[0]]
            bx = [p[0] for p in b[0]]
            by = [p[1] for p in b[0]]
            if (
                max(ax) < min(bx)
                or max(bx) < min(ax)
                or max(ay) < min(by)
                or (max(by) < min(ay))
            ):
                continue
            if {p for ring in a for p in ring} & {p for ring in b for p in ring}:
                return True
            if any((_inside(p, b) for p in a[0])) or any((_inside(p, a) for p in b[0])):
                return True
            for p, q in region_edges(a):
                for r, s in region_edges(b):
                    c = (cross(p, q, r), cross(p, q, s), cross(r, s, p), cross(r, s, q))
                    if any((on(r, p, q), on(s, p, q), on(p, r, s), on(q, r, s))):
                        return True
                    if c[0] * c[1] < 0 and c[2] * c[3] < 0:
                        return True
    return False
