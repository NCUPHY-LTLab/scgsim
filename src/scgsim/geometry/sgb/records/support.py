"""The support region of a surface partition (design brief sections 5 and 6).

A *support region* is the physical face a family of inset bands partitions: a planar region
bounded by rings of exact primitives (straight edges and circular arcs, the interior on the left
of each ring) plus the margins ``m_1 < ... < m_K`` of its bands.  It owns the geometry questions
the bands need: which band a point lies in, the region's exact area and perimeter, and its
witness polygon for containment.

Containment is exact (even-odd over the witness polygon, in integer arithmetic).  Distances are
measured on the exact primitives in floating point, from the region's *distance origins*: the boundary
primitives the bands are measured from.  A conductor face has its whole boundary as origins; a solution
interface has only the parts of its boundary where it meets a conductor.  Distances decide the band of a
cell from the area it covers (``labelling.bands``), never a topological question.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property

from scgsim.geometry.sgb.plane import Point, RepPoint, point_in_ring
from scgsim.geometry.sgb.records.carriers import ArcCarrier, LineEdge, RingPrimitive

DISTANCE_SLACK = 1.5
"""How far, in grid units, a cut curve may lie from the exact offset it stands for: its end points are
snapped to the lattice and arcs are cut along witness chords, so a distance within this of a margin
does not say on which side of the cut a point lies."""


def primitive_ends(
    prim: RingPrimitive,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Start and end point of a ring primitive as floats."""
    if isinstance(prim, LineEdge):
        return (float(prim.p[0]), float(prim.p[1])), (
            float(prim.q[0]),
            float(prim.q[1]),
        )
    return (
        (float(prim.start[0]), float(prim.start[1])),
        (float(prim.end[0]), float(prim.end[1])),
    )


def reverse_ring(prims: list[RingPrimitive]) -> list[RingPrimitive]:
    """The same boundary traversed the other way (a hole of one region is the outer ring of its complement)."""
    return [p.reversed() for p in reversed(prims)]


def distance_to_primitive(prim: RingPrimitive, x: float, y: float) -> float:
    """Distance from a point to a straight edge or an arc (end points included)."""
    start, end = primitive_ends(prim)
    if isinstance(prim, LineEdge):
        dx, dy = end[0] - start[0], end[1] - start[1]
        t = max(
            0.0,
            min(1.0, ((x - start[0]) * dx + (y - start[1]) * dy) / (dx * dx + dy * dy)),
        )
        return math.hypot(x - (start[0] + t * dx), y - (start[1] + t * dy))
    cx, cy = float(prim.centre[0]), float(prim.centre[1])
    if _inside_span(prim, (cx, cy), start, end, (x, y)):
        return abs(math.hypot(x - cx, y - cy) - float(prim.radius))
    return min(
        math.hypot(x - start[0], y - start[1]), math.hypot(x - end[0], y - end[1])
    )


def _inside_span(
    arc: ArcCarrier,
    centre: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
    point: tuple[float, float],
) -> bool:
    """Whether the ray from the centre through ``point`` lies within the arc's angular span.

    Arcs here span less than a half turn (they are cut into quarter pieces), so two cross products
    decide it, with no trigonometry.
    """
    sx, sy = start[0] - centre[0], start[1] - centre[1]
    ex, ey = end[0] - centre[0], end[1] - centre[1]
    px, py = point[0] - centre[0], point[1] - centre[1]
    from_start = sx * py - sy * px
    to_end = px * ey - py * ex
    return (
        (from_start >= 0 and to_end >= 0)
        if arc.ccw
        else (from_start <= 0 and to_end <= 0)
    )


@dataclass(frozen=True)
class SupportRegion:
    """A planar region bounded by rings of exact primitives (interior on the left of each ring) and its margins.

    ``origin_mask`` says, per ring and primitive, whether the primitive is a distance origin; empty
    means every boundary primitive is one.
    """

    rings: tuple[tuple[RingPrimitive, ...], ...]
    margins: tuple[int, ...]  # grid units, strictly increasing
    origin_mask: tuple[tuple[bool, ...], ...] = ()
    _witness_cache: list[list[Point]] = field(
        default_factory=list, compare=False, repr=False
    )

    def witness_rings(self) -> list[list[Point]]:
        """Each ring as an integer polygon (arcs as their witness chords)."""
        if not self._witness_cache:
            for ring in self.rings:
                points: list[Point] = []
                for prim in ring:
                    if isinstance(prim, LineEdge):
                        points.append(prim.p)
                    else:
                        points.extend(prim.witness_points()[:-1])
                self._witness_cache.append(points)
        return self._witness_cache

    def contains(self, point: RepPoint) -> bool:
        """Exact even-odd containment of a rational point in the witness polygon."""
        return sum(point_in_ring(ring, point) for ring in self.witness_rings()) % 2 == 1

    def is_origin(self, ring_index: int, primitive_index: int) -> bool:
        """Whether a boundary primitive is a distance origin."""
        return not self.origin_mask or self.origin_mask[ring_index][primitive_index]

    @cached_property
    def origins(self) -> tuple[RingPrimitive, ...]:
        """The boundary primitives distances are measured from."""
        return tuple(
            prim
            for ri, ring in enumerate(self.rings)
            for pi, prim in enumerate(ring)
            if self.is_origin(ri, pi)
        )

    @cached_property
    def free_boundary(self) -> tuple[RingPrimitive, ...]:
        """The boundary primitives that are not distance origins (the die edge, the edge to other solution cells)."""
        return tuple(
            prim
            for ri, ring in enumerate(self.rings)
            for pi, prim in enumerate(ring)
            if not self.is_origin(ri, pi)
        )

    def distance(self, point: tuple[float, float]) -> float:
        """Distance from a point to the nearest distance origin."""
        return min(distance_to_primitive(p, point[0], point[1]) for p in self.origins)

    def distance_to_free_boundary(self, point: tuple[float, float]) -> float:
        """Distance from a point to the boundary that is not an origin (infinite when every primitive is one)."""
        return min(
            (distance_to_primitive(p, point[0], point[1]) for p in self.free_boundary),
            default=math.inf,
        )

    def band_of(self, point: RepPoint) -> int | None:
        """Band index of a point inside the region (0 = closest to the boundary), ``None`` outside."""
        if not self.contains(point):
            return None
        d = self.distance((point[0] / point[2], point[1] / point[2]))
        return sum(1 for m in self.margins if d >= m)

    def area(self) -> float:
        """Exact area (grid^2): line chords by the shoelace formula, arcs by their sector term."""
        twice = 0.0
        for ring in self.rings:
            for prim in ring:
                (sx, sy), (ex, ey) = primitive_ends(prim)
                if isinstance(prim, LineEdge):
                    twice += sx * ey - ex * sy
                else:
                    cx, cy, r = (
                        float(prim.centre[0]),
                        float(prim.centre[1]),
                        float(prim.radius),
                    )
                    twice += (
                        cx * (ey - sy) - cy * (ex - sx) + r * r * float(prim.sweep()[1])
                    )
        return twice / 2

    def perimeter(self) -> float:
        """Length of all boundary rings (grid units)."""
        total = 0.0
        for ring in self.rings:
            for prim in ring:
                (sx, sy), (ex, ey) = primitive_ends(prim)
                if isinstance(prim, LineEdge):
                    total += math.hypot(ex - sx, ey - sy)
                else:
                    total += float(prim.radius) * abs(float(prim.sweep()[1]))
        return total

    @property
    def band_count(self) -> int:
        """Number of bands: one more than the margins."""
        return len(self.margins) + 1
