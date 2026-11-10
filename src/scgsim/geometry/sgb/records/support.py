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

ORIGIN_CELL = 64.0
"""Smallest cell, in grid units, of the grid that finds the nearest distance origin of a point."""

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

    @cached_property
    def _origin_grid(self) -> _OriginGrid:
        return _OriginGrid(self.origins)

    def distance(self, point: tuple[float, float]) -> float:
        """Distance from a point to the nearest distance origin."""
        return self._origin_grid.nearest(point[0], point[1])

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


class _OriginGrid:
    """The distance origins listed under every cell their bounding box covers, for nearest-origin queries.

    A query visits the cells in square rings around its own cell, nearest ring first, and stops once
    every primitive not yet met is farther away than the best distance found.  The result is the minimum
    of :func:`distance_to_primitive` over a set that contains the nearest primitive, so it is the same
    float as the minimum over all origins.
    """

    def __init__(self, origins: tuple[RingPrimitive, ...]) -> None:
        if not origins:
            raise ValueError(
                "a support region without distance origins has no distances"
            )
        self.origins = origins
        boxes = [_bounding_box(prim) for prim in origins]
        extent = max(
            max(box[2] for box in boxes) - min(box[0] for box in boxes),
            max(box[3] for box in boxes) - min(box[1] for box in boxes),
        )
        # about sqrt(n) by sqrt(n) cells over n origins, never finer than ORIGIN_CELL
        self.size = max(ORIGIN_CELL, extent / max(4, math.isqrt(len(origins))))
        self.cells: dict[tuple[int, int], list[int]] = {}
        for index, (x0, y0, x1, y1) in enumerate(boxes):
            for column in range(self._cell(x0), self._cell(x1) + 1):
                for row in range(self._cell(y0), self._cell(y1) + 1):
                    self.cells.setdefault((column, row), []).append(index)
        self.columns = (min(c for c, _ in self.cells), max(c for c, _ in self.cells))
        self.rows = (min(r for _, r in self.cells), max(r for _, r in self.cells))

    def _cell(self, coordinate: float) -> int:
        return math.floor(coordinate / self.size)

    def nearest(self, x: float, y: float) -> float:
        """Distance from ``(x, y)`` to the nearest origin."""
        column, row = self._cell(x), self._cell(y)
        last_ring = max(
            abs(column - self.columns[0]),
            abs(column - self.columns[1]),
            abs(row - self.rows[0]),
            abs(row - self.rows[1]),
        )
        best = math.inf
        met: set[int] = set()
        for ring in range(last_ring + 1):
            # A primitive first met in this ring lies in cells of this ring or farther out, so at least
            # (ring - 1) cells from the point, wherever it sits in its own cell.  Stopping one ring later
            # than that leaves a whole cell for float rounding of the boxes and the distances.
            if (ring - 2) * self.size > best:
                break
            for cell in _ring_cells(column, row, ring):
                for index in self.cells.get(cell, ()):
                    if index not in met:
                        met.add(index)
                        best = min(
                            best, distance_to_primitive(self.origins[index], x, y)
                        )
        return best


def _bounding_box(prim: RingPrimitive) -> tuple[float, float, float, float]:
    """Bounding box of a primitive: its end points, and for an arc the axis extremes it passes."""
    start, end = primitive_ends(prim)
    xs, ys = [start[0], end[0]], [start[1], end[1]]
    if isinstance(prim, ArcCarrier):
        cx, cy, r = float(prim.centre[0]), float(prim.centre[1]), float(prim.radius)
        for extreme in ((cx + r, cy), (cx - r, cy), (cx, cy + r), (cx, cy - r)):
            if _inside_span(prim, (cx, cy), start, end, extreme):
                xs.append(extreme[0])
                ys.append(extreme[1])
    return min(xs), min(ys), max(xs), max(ys)


def _ring_cells(column: int, row: int, ring: int) -> list[tuple[int, int]]:
    """The cells at Chebyshev distance ``ring`` from ``(column, row)``."""
    if ring == 0:
        return [(column, row)]
    top_and_bottom = [
        (c, r)
        for c in range(column - ring, column + ring + 1)
        for r in (row - ring, row + ring)
    ]
    sides = [
        (c, r)
        for r in range(row - ring + 1, row + ring)
        for c in (column - ring, column + ring)
    ]
    return top_and_bottom + sides
