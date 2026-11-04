"""Local feature size of faces on the exact carriers (brief section 14, sliver width).

Owns the sliver width of a face: the minimum distance between two *non-incident* boundary edges (edges that share
no vertex), measured on the realised geometry -- straight segments and circular arcs, not their chords. A face
whose edges are all pairwise incident (a triangle, a lens) falls back to the smallest distance from a vertex to a
non-incident edge (the altitude). ``2 A / P`` of the chord polygon is not used: it is not a width, and chords of an
arc misstate it.

For a wall (a vertical face), the feature size is ``min(height, distance between its vertical edges)``.

Distances between non-crossing primitives are attained at an end point of one of them or at a stationary point
along the line of centres (arc against arc) or the centre-to-segment foot (arc against line); both families of
candidates are evaluated analytically. Large faces (many holes) use a spatial hash: the answer is exact whenever it
is smaller than the hash cell size, otherwise the cell size is reported with ``bounded=True``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

TWO_PI = 2 * math.pi


@dataclass(frozen=True)
class Prim:
    """A boundary primitive in grid units: a segment or an arc (centre, radius, start angle, signed sweep)."""

    a: tuple[float, float]
    b: tuple[float, float]
    centre: tuple[float, float] | None = None
    radius: float = 0.0
    t0: float = 0.0
    sweep: float = 0.0
    ids: tuple[str, str] = ("", "")  # end vertex ids (incidence)

    @property
    def is_arc(self) -> bool:
        return self.centre is not None

    def bbox(self) -> tuple[float, float, float, float]:
        xs, ys = [self.a[0], self.b[0]], [self.a[1], self.b[1]]
        if self.is_arc:
            for k in range(4):  # axis extremes inside the span
                angle = k * math.pi / 2
                if self.in_span(angle):
                    xs.append(self.centre[0] + self.radius * math.cos(angle))
                    ys.append(self.centre[1] + self.radius * math.sin(angle))
        return min(xs), min(ys), max(xs), max(ys)

    def in_span(self, angle: float) -> bool:
        delta = (angle - self.t0) % TWO_PI if self.sweep > 0 else -((self.t0 - angle) % TWO_PI)
        return 0 <= delta <= self.sweep if self.sweep > 0 else self.sweep <= delta <= 0


def point_distance(prim: Prim, x: float, y: float) -> float:
    if not prim.is_arc:
        (ax, ay), (bx, by) = prim.a, prim.b
        dx, dy = bx - ax, by - ay
        t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)))
        return math.hypot(x - (ax + t * dx), y - (ay + t * dy))
    cx, cy = prim.centre
    if prim.in_span(math.atan2(y - cy, x - cx)):
        return abs(math.hypot(x - cx, y - cy) - prim.radius)
    return min(math.hypot(x - prim.a[0], y - prim.a[1]), math.hypot(x - prim.b[0], y - prim.b[1]))


def prim_distance(p: Prim, q: Prim) -> float:
    """Distance between two non-crossing primitives (see module docstring for the candidate set)."""
    best = min(point_distance(q, *p.a), point_distance(q, *p.b), point_distance(p, *q.a), point_distance(p, *q.b))
    for arc, other in ((p, q), (q, p)):
        if not arc.is_arc:
            continue
        for x, y in _stationary_points(arc, other):
            best = min(best, point_distance(other, x, y))
    return best


def _stationary_points(arc: Prim, other: Prim):
    """Points of ``arc`` (inside its span) where the distance to ``other`` can be stationary."""
    cx, cy = arc.centre
    if other.is_arc:
        dx, dy = other.centre[0] - cx, other.centre[1] - cy
    else:  # direction from the centre to the foot of the perpendicular on the segment's line
        (ax, ay), (bx, by) = other.a, other.b
        ex, ey = bx - ax, by - ay
        t = ((cx - ax) * ex + (cy - ay) * ey) / (ex * ex + ey * ey)
        dx, dy = ax + t * ex - cx, ay + t * ey - cy
    n = math.hypot(dx, dy)
    if n == 0:
        return
    angle = math.atan2(dy, dx)
    for a in (angle, angle + math.pi):
        if arc.in_span(a):
            yield cx + arc.radius * math.cos(a), cy + arc.radius * math.sin(a)


def feature_size(prims: list[Prim], cell: float | None = None) -> tuple[float, tuple[float, float], bool]:
    """(min distance between non-incident primitives, midpoint of that pair, bounded) in grid units.

    ``bounded`` is true when no pair lay within the hash neighbourhood and the value is only a lower bound.
    Falls back to the smallest vertex-to-non-incident-edge distance when every pair of edges is incident.
    """
    n = len(prims)
    best, where, bounded = math.inf, (0.0, 0.0), False
    boxes = [p.bbox() for p in prims]
    if n <= 300:
        pairs = ((i, j) for i in range(n) for j in range(i + 1, n))
        hashed = False
    else:
        lengths = sorted(max(b[2] - b[0], b[3] - b[1]) for b in boxes)
        cell = cell or max(1.0, 4 * lengths[n // 2])
        grid: dict[tuple[int, int], list[int]] = {}
        for i, (x0, y0, x1, y1) in enumerate(boxes):
            for cx in range(int(x0 // cell) - 1, int(x1 // cell) + 2):
                for cy in range(int(y0 // cell) - 1, int(y1 // cell) + 2):
                    grid.setdefault((cx, cy), []).append(i)
        seen: set[tuple[int, int]] = set()

        def gen():
            for members in grid.values():
                for ii, i in enumerate(members):
                    for j in members[ii + 1:]:
                        key = (i, j) if i < j else (j, i)
                        if key not in seen:
                            seen.add(key)
                            yield key

        pairs, hashed = gen(), True
    for i, j in pairs:
        p, q = prims[i], prims[j]
        if set(p.ids) & set(q.ids):
            continue
        bi, bj = boxes[i], boxes[j]
        gap = math.hypot(max(bi[0] - bj[2], bj[0] - bi[2], 0.0), max(bi[1] - bj[3], bj[1] - bi[3], 0.0))
        if gap >= best:
            continue
        d = prim_distance(p, q)
        if d < best:
            best, where = d, ((p.a[0] + q.a[0] + p.b[0] + q.b[0]) / 4, (p.a[1] + q.a[1] + p.b[1] + q.b[1]) / 4)
    if best < math.inf:
        return best, where, False
    if hashed:
        return cell, (0.0, 0.0), True
    for p in prims:  # triangle-like faces: altitude of each vertex over the edges it does not touch
        for vertex, vid in ((p.a, p.ids[0]), (p.b, p.ids[1])):
            for q in prims:
                if vid not in q.ids:
                    d = point_distance(q, *vertex)
                    if d < best:
                        best, where = d, vertex
    return best, where, False
