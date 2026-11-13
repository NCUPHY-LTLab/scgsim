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
cell from the area it covers (``arrangement.labelling.bands``), never a topological question.

The die edge is a dicing line: a straight origin that runs onto it belongs to a conductor the dicing cut, so
distances are measured from that origin continued straight past the die edge (Issue #138), as if the
conductor went on.  An arc origin that runs onto it has no single way on, and the build refuses it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from functools import cached_property

from scgsim.geometry.exact.carriers import (
    ArcCarrier,
    LineEdge,
    RingPrimitive,
    round_to_lattice,
)
from scgsim.geometry.exact.plane import Point, RepPoint, point_in_ring
from scgsim.refusal import internal

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
    means every boundary primitive is one.  ``die_edge_mask`` says, the same way, whether a primitive that is
    no origin lies on the die edge (the substrate's outer edge); empty means none does.  ``sources`` names, the
    same way, the source curves each primitive lies on: the carrier ids of the source polygon edges and arcs
    drawn along it, one per Entity whose outline runs there; a region resolved from a model always records
    them, and empty means a region drawn directly (in a test), whose primitives come from no source.
    """

    rings: tuple[tuple[RingPrimitive, ...], ...]
    margins: tuple[int, ...]  # grid units, strictly increasing
    origin_mask: tuple[tuple[bool, ...], ...] = ()
    die_edge_mask: tuple[tuple[bool, ...], ...] = ()
    sources: tuple[tuple[tuple[str, ...], ...], ...] = ()
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

    def is_die_edge(self, ring_index: int, primitive_index: int) -> bool:
        """Whether a boundary primitive that is no distance origin lies on the die edge."""
        return (
            bool(self.die_edge_mask) and self.die_edge_mask[ring_index][primitive_index]
        )

    @cached_property
    def die_cuts(self) -> tuple[DieCut, ...]:
        """Every end of a straight origin that the die edge cuts, with the origin continued past it there.

        A straight origin whose neighbour on the ring lies on the die edge is continued at that end, along its
        own line, until the end lies twice the largest margin beyond the die edge's tangent there.  Where the
        die is convex the disk of the largest margin about the new end then lies outside the die, so inside the
        die the continued edge is as far from every point as the whole line is, and each offset of it ends at
        least one margin outside the die, clear of snapping.  A longer continuation gives the same distances in
        the die.  Where the die outline comes back beside a continuation, the build is refused when the support
        is resolved (``bands.die_cuts``).

        Not cut, so its end keeps the half circle of the bands about it: a line where the die edge turns back
        over it (a die that is not convex there), which has no straight way out.  An arc origin that ends on the
        die edge is not continued either; it is listed in :attr:`cut_arcs`, and the build is refused.
        """
        reach = 2 * self.margins[-1]
        cuts: list[DieCut] = []
        for ri, ring in enumerate(self.rings):
            count = len(ring)
            for pi, prim in enumerate(ring):
                if not (self.is_origin(ri, pi) and isinstance(prim, LineEdge)):
                    continue
                for at_end, neighbour, inner, vertex in (
                    (True, (pi + 1) % count, prim.p, prim.q),
                    (False, (pi - 1) % count, prim.q, prim.p),
                ):
                    if not self.is_die_edge(ri, neighbour):
                        continue
                    direction = _direction_at(ring[neighbour], at_end=not at_end)
                    end = _past_the_die(inner, vertex, direction, reach)
                    if end is not None:
                        cuts.append(DieCut(ri, pi, at_end, vertex, end, direction))
        return tuple(cuts)

    @cached_property
    def cut_arcs(self) -> tuple[ArcCut, ...]:
        """Every end of an arc origin whose neighbour on the ring lies on the die edge: a conductor cut along an arc.

        A conductor cut by the die edge is measured as continuing past it (:attr:`die_cuts`), and a straight
        origin has one way on, along its own line.  An arc has none that the layout states, so SGB does not
        model it: the build is refused when the support is resolved (``bands.die_cuts``, Issue #138).
        """
        cuts: list[ArcCut] = []
        for ri, ring in enumerate(self.rings):
            count = len(ring)
            for pi, prim in enumerate(ring):
                if not (self.is_origin(ri, pi) and isinstance(prim, ArcCarrier)):
                    continue
                for at_end, neighbour, vertex in (
                    (True, (pi + 1) % count, prim.end),
                    (False, (pi - 1) % count, prim.start),
                ):
                    if self.is_die_edge(ri, neighbour):
                        cuts.append(ArcCut(ri, pi, at_end, round_to_lattice(vertex)))
        return tuple(cuts)

    @cached_property
    def continued_origins(self) -> dict[tuple[int, int], LineEdge]:
        """The straight origins the die edge cuts, each continued past it (:attr:`die_cuts`): ``(ring, primitive)`` -> edge."""
        continued: dict[tuple[int, int], LineEdge] = {}
        for cut in self.die_cuts:
            key = (cut.ring, cut.primitive)
            edge = continued.get(key)
            if edge is None:
                prim = self.rings[cut.ring][cut.primitive]
                assert isinstance(prim, LineEdge), "only straight origins are cut"
                edge = prim
            continued[key] = (
                LineEdge(edge.p, cut.end, edge.source_edge)
                if cut.at_end
                else LineEdge(cut.end, edge.q, edge.source_edge)
            )
        return continued

    def origin(self, ring_index: int, primitive_index: int) -> RingPrimitive:
        """A distance origin as distances are measured from it: continued past the die edge where it cuts it."""
        return self.continued_origins.get(
            (ring_index, primitive_index), self.rings[ring_index][primitive_index]
        )

    @cached_property
    def origins(self) -> tuple[RingPrimitive, ...]:
        """The boundary primitives distances are measured from, each as :meth:`origin` gives it."""
        return tuple(
            self.origin(ri, pi)
            for ri, ring in enumerate(self.rings)
            for pi in range(len(ring))
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


@dataclass(frozen=True)
class DieCut:
    """One end of a straight distance origin that the die edge cuts, and the origin's continuation past it.

    ``ring`` and ``primitive`` locate the origin and ``at_end`` says which of its ends is cut.  ``vertex`` is that
    end, on the die edge; ``end`` is where the continuation stops, past the die edge; ``die_direction`` is the die
    edge's direction of travel at the vertex, not normalised, with the die on its left.
    """

    ring: int
    primitive: int
    at_end: bool
    vertex: Point
    end: Point
    die_direction: tuple[Fraction, Fraction]


@dataclass(frozen=True)
class ArcCut:
    """One end of an arc distance origin that lies on the die edge, which SGB refuses.

    ``ring`` and ``primitive`` locate the arc and ``at_end`` says which of its ends lies on the die edge;
    ``vertex`` is that end, a lattice point.
    """

    ring: int
    primitive: int
    at_end: bool
    vertex: Point


def _direction_at(prim: RingPrimitive, *, at_end: bool) -> tuple[Fraction, Fraction]:
    """The direction of travel of a primitive at its start or end, not normalised (an arc's tangent there)."""
    if isinstance(prim, LineEdge):
        return Fraction(prim.q[0] - prim.p[0]), Fraction(prim.q[1] - prim.p[1])
    x, y = prim.end if at_end else prim.start
    rx, ry = x - prim.centre[0], y - prim.centre[1]
    return (-ry, rx) if prim.ccw else (ry, -rx)


def _past_the_die(
    inner: Point, vertex: Point, die_direction: tuple[Fraction, Fraction], reach: int
) -> Point | None:
    """``vertex`` moved on along the line from ``inner`` until it lies ``reach`` beyond the die edge's line.

    The die edge runs along ``die_direction`` at ``vertex``, with the die on its left.  The point moves in whole
    steps of the line's smallest lattice step, so it stays a lattice point exactly on the line.  ``None`` when
    the line does not leave the die there.
    """
    dx, dy = vertex[0] - inner[0], vertex[1] - inner[1]
    divisor = math.gcd(dx, dy)
    sx, sy = dx // divisor, dy // divisor
    ex, ey = die_direction
    cross = (
        sx * ey - sy * ex
    )  # |step| |die| sin(angle): positive when a step leaves the die
    if cross <= 0:
        return None
    # k steps lie k * cross / |die| beyond the die edge's line; the smallest k that puts them `reach` beyond.
    bound = reach * reach * (ex * ex + ey * ey) / (cross * cross)
    steps = math.isqrt(math.ceil(bound))
    if steps * steps < bound:
        steps += 1
    return (vertex[0] + steps * sx, vertex[1] + steps * sy)


class _OriginGrid:
    """The distance origins listed under every cell their bounding box covers, for nearest-origin queries.

    A query visits the cells in square rings around its own cell, nearest ring first, and stops once
    every primitive not yet met is farther away than the best distance found.  The result is the minimum
    of :func:`distance_to_primitive` over a set that contains the nearest primitive, so it is the same
    float as the minimum over all origins.  Only the cells of the occupied column and row range are ever
    visited, so a query far outside the origins (a large interface measured from one small conductor)
    costs at most the grid's own cell count, wherever it lies.
    """

    def __init__(self, origins: tuple[RingPrimitive, ...]) -> None:
        if not origins:
            raise ValueError(
                internal("a support region without distance origins has no distances")
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
        # No origin lies in a ring nearer than the occupied range, nor beyond its far corner.
        first_ring = max(
            self.columns[0] - column,
            column - self.columns[1],
            self.rows[0] - row,
            row - self.rows[1],
            0,
        )
        last_ring = max(
            abs(column - self.columns[0]),
            abs(column - self.columns[1]),
            abs(row - self.rows[0]),
            abs(row - self.rows[1]),
        )
        best = math.inf
        met: set[int] = set()
        for ring in range(first_ring, last_ring + 1):
            # A primitive first met in this ring lies in cells of this ring or farther out, so at least
            # (ring - 1) cells from the point, wherever it sits in its own cell.  Stopping one ring later
            # than that leaves a whole cell for float rounding of the boxes and the distances.
            if (ring - 2) * self.size > best:
                break
            for cell in _ring_cells((column, row), ring, self.columns, self.rows):
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


def _ring_cells(
    centre: tuple[int, int],
    ring: int,
    columns: tuple[int, int],
    rows: tuple[int, int],
) -> list[tuple[int, int]]:
    """The cells at Chebyshev distance ``ring`` from ``centre`` that lie in the given column and row range."""
    column, row = centre
    if ring == 0:
        return [centre]
    first_column, last_column = (
        max(columns[0], column - ring),
        min(columns[1], column + ring),
    )
    first_row, last_row = max(rows[0], row - ring + 1), min(rows[1], row + ring - 1)
    cells = [
        (c, r)
        for r in (row - ring, row + ring)
        if rows[0] <= r <= rows[1]
        for c in range(first_column, last_column + 1)
    ]
    cells.extend(
        (c, r)
        for c in (column - ring, column + ring)
        if columns[0] <= c <= columns[1]
        for r in range(first_row, last_row + 1)
    )
    return cells
