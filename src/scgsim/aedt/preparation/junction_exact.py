"""Exact planar geometry on the Boolean integer grid used by junction partitioning.

Owns: the integer-coordinate predicates and sections (orientation, containment,
contact, coverage) and the gdstk Boolean wrappers that every junction proof is
built from. Coordinates here are integers in units of `PRECISION_UM`, except where
a function says it takes `Fraction` coordinates. Predicates are exact so that proof
conditions never depend on a floating-point tolerance.

Does not own: choosing the junction cuts (`junction_partition.py`) or mapping the
proof back to model coordinates (`junction_model.py`).

A "polygon" below is a tuple of rings, outer ring first and holes after it. A
"region" is a list of such polygons.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import fields
from fractions import Fraction
from itertools import pairwise
from typing import Any

import gdstk

from scgsim.geometry.primitives.loops import (
    cancel_reversed_planar_edges,
    simple_planar_loops_from_edges,
)

# Boolean grid pitch in micrometres: coordinates are rounded to multiples of this.
PRECISION_UM = 1e-9


def plain_with_dataclasses(value: Any) -> Any:
    """Convert dataclasses, mappings and sequences to plain dicts and lists."""
    if hasattr(value, "__dataclass_fields__"):
        return {
            item.name: plain_with_dataclasses(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): plain_with_dataclasses(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain_with_dataclasses(item) for item in value]
    return value


def ring_edges(rings):
    """Yield the closed edges `(start, end)` of every ring."""
    for ring in rings:
        yield from zip(ring, (*ring[1:], ring[0]), strict=False)


def twice_signed_area(ring):
    """Twice the signed area of a ring; positive when counter-clockwise."""
    return sum(a[0] * b[1] - b[0] * a[1] for a, b in ring_edges((ring,)))


def oriented_ring(ring, index):
    """Return the ring wound counter-clockwise if outer (index 0), else clockwise."""
    return (
        ring if (twice_signed_area(ring) > 0) == (index == 0) else tuple(reversed(ring))
    )


def cross(a, b, c):
    """Twice the signed area of triangle `abc`; zero exactly when collinear."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def on(point, a, b):
    """Whether `point` lies on the closed segment `ab`."""
    return cross(a, b, point) == 0 and all(
        min(a[i], b[i]) <= point[i] <= max(a[i], b[i]) for i in (0, 1)
    )


def inside_ring(point, ring):
    """Locate `point` against one ring: 1 inside, 0 on the boundary, -1 outside."""
    inside = False
    for a, b in ring_edges((ring,)):
        if on(point, a, b):
            return 0  # Boundary, distinct from inside and outside.
        if (a[1] > point[1]) != (b[1] > point[1]):
            x = Fraction(b[0] - a[0], b[1] - a[1]) * (point[1] - a[1]) + a[0]
            if point[0] < x:
                inside = not inside
    return 1 if inside else -1


def _inside(point, rings):
    """Whether `point` is in the closed polygon: on or inside the outer ring and
    not strictly inside a hole (hole boundaries belong to the polygon)."""
    return inside_ring(point, rings[0]) >= 0 and not any(
        inside_ring(point, hole) == 1 for hole in rings[1:]
    )


def section(rings, axis, coordinate, low, high):
    """Exact closed contact on a specified edge in Boolean integer coordinates.

    Intersects the closed polygon with the line `axis == coordinate` over
    `[low, high]` (along the other axis). Returns `(intervals, points)`: the
    maximal sub-intervals inside the polygon, and every mark inside it, which
    includes isolated touching points.
    """
    other = 1 - axis
    marks = {Fraction(low), Fraction(high)}
    for a, b in ring_edges(rings):
        if a[axis] == b[axis] == coordinate:
            marks.update(Fraction(v[other]) for v in (a, b) if low <= v[other] <= high)
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
    return intervals, points


def covered(polygons, axis, coordinate, low, high):
    """Whether the union of the polygons covers `[low, high]` on the given line."""
    intervals = sorted(
        interval
        for rings in polygons
        for interval in section(rings, axis, coordinate, low, high)[0]
    )
    end = Fraction(low)
    for a, b in intervals:
        if a > end:
            return False
        end = max(end, b)
    return end >= high


def _line_intervals(polygons):
    """Group polygon edges by supporting line; map each line to its edge intervals.

    A line is keyed by its reduced direction and offset, with direction sign fixed,
    so collinear edges of different polygons share a key. Each interval is given
    by projections onto the direction.
    """
    result = defaultdict(list)
    for rings in polygons:
        for a, b in ring_edges(rings):
            dx, dy = b[0] - a[0], b[1] - a[1]
            divisor = math.gcd(dx, dy)
            if not divisor:
                continue
            dx, dy = dx // divisor, dy // divisor
            if dx < 0 or (dx == 0 and dy < 0):
                dx, dy = -dx, -dy
            key = (dx, dy, dx * a[1] - dy * a[0])
            u, v = a[0] * dx + a[1] * dy, b[0] * dx + b[1] * dy
            result[key].append((min(u, v), max(u, v)))
    return result


def positive_boundary_contact(left, right):
    """Whether two groups of polygons share boundary of positive length."""
    a, b = _line_intervals(left), _line_intervals(right)
    return any(
        max(x0, y0) < min(x1, y1)
        for key in a.keys() & b.keys()
        for x0, x1 in a[key]
        for y0, y1 in b[key]
    )


def closed_contact(left, right):
    """Whether any polygon of `left` and any of `right` share at least one point.

    Closed regions count: touching at a vertex or along an edge is contact, as is
    containment. A bounding-box test skips pairs that cannot touch.
    """
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
                or max(by) < min(ay)
            ):
                continue
            if {p for ring in a for p in ring} & {p for ring in b for p in ring}:
                return True
            if any(_inside(p, b) for p in a[0]) or any(_inside(p, a) for p in b[0]):
                return True
            for p, q in ring_edges(a):
                for r, s in ring_edges(b):
                    c = (
                        cross(p, q, r),
                        cross(p, q, s),
                        cross(r, s, p),
                        cross(r, s, q),
                    )
                    if any((on(r, p, q), on(s, p, q), on(p, r, s), on(q, r, s))):
                        return True
                    if c[0] * c[1] < 0 and c[2] * c[3] < 0:
                        return True
    return False


def region_from_polygons(polygons):
    """Convert integer polygons to gdstk polygons in micrometres, holes cut out."""
    result = []
    for rings in polygons:
        exterior = gdstk.Polygon(
            [(x * PRECISION_UM, y * PRECISION_UM) for x, y in rings[0]]
        )
        holes = [
            gdstk.Polygon([(x * PRECISION_UM, y * PRECISION_UM) for x, y in ring])
            for ring in rings[1:]
        ]
        result.extend(boolean([exterior], holes, "not"))
    return result


def boolean(left, right, operation):
    # SGB (the geometry builder) has a Boolean wrapper that filters small areas.
    # Partitioning must retain every residual, so reuse its precision and loop
    # lowering without that filter.
    """Run a gdstk Boolean at PRECISION_UM and return the result as a list."""
    return list(gdstk.boolean(left, right, operation, precision=PRECISION_UM) or ())


def rings_from_region(region):
    """Convert a gdstk region back to integer polygons, each as `(outer, *holes)`.

    Raises `ValueError` when a residual has point-connected or non-interior
    components, which the junction proof cannot represent.
    """
    result = []
    for polygon in region:
        points = tuple(
            (round(x / PRECISION_UM), round(y / PRECISION_UM))
            for x, y in polygon.points
        )
        # Atomize at every retained vertex, including vertices inside another
        # edge. This preserves SGB minimum-coherence/point-fragment semantics.
        ordered = sorted(set(points))
        xs = [point[0] for point in ordered]
        edges = []
        for a, b in ring_edges((points,)):
            candidates = ordered[
                bisect_left(xs, min(a[0], b[0])) : bisect_right(xs, max(a[0], b[0]))
            ]
            on_edge = sorted(
                (p for p in candidates if on(p, a, b)),
                key=lambda p: (
                    (p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])
                ),
            )
            edges.extend(pairwise(on_edge))
        loops = simple_planar_loops_from_edges(cancel_reversed_planar_edges(edges))

        def area(ring):
            return abs(twice_signed_area(ring))

        outer = max(loops, key=area)
        holes = tuple(ring for ring in loops if ring is not outer)
        if any(inside_ring(hole[0], outer) != 1 for hole in holes):
            raise ValueError(
                "junction residual has point-connected or non-interior components"
            )
        result.append((outer, *holes))
    return result


def rectangle(x0, y0, x1, y1):
    """The rectangle as a one-polygon region of integer rings."""
    return ((((x0, y0), (x1, y0), (x1, y1), (x0, y1)),),)
