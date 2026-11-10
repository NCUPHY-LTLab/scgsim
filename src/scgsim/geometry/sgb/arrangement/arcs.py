"""Arc edges: polyline witness back-mapping and realisation (design brief section 3, decision D3).

Owns what happens to circular arcs between the source and the arrangement:

* **Witness.**  An arc carrier is noded as a chain of integer chords (sagitta at
  most g/4) whose carrier ids are ``<arc id>#<k>`` (see
  ``records.carriers.ArcCarrier.witness_points``).  Snap rounding and dissolution
  treat them as ordinary segments.
* **Merge.**  Dissolution removes a degree-2 vertex whose two edges carry
  pieces of the same arc, so a run of witness chords with no foreign hot pixel
  between them becomes one edge whose carrier set is just ``{arc id}``.
  Everything downstream sees one arc edge between two snapped vertices.
* **Realisation.**  An arc edge ``a -> b`` is the circle arc of the carrier's
  radius through the two snapped endpoints, with its centre on the same side as
  the carrier's centre (so the realised arc bulges the same way).  Its deviation
  from the exact source circle is recorded per edge.

Does not own arrangement building; it supplies the geometry the later stages consult.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction

from scgsim.geometry.sgb.arrangement.dcel import Arrangement
from scgsim.geometry.sgb.plane import Point
from scgsim.geometry.sgb.records.carriers import ArcCarrier, Lineage

SAMPLES = 33
CROSSING_EPSILON = 1e-7
# Grid units added on both sides of a segment's clipped y range when it is entered into the
# crossing cells: it covers float rounding of the clip, and only ever adds candidate cells.
CELL_PAD = 1e-6
CELL_SIZE = 256.0
"""Side of the broad-phase cells of the crossing check, in grid units."""


@dataclass(frozen=True)
class RealisedArc:
    """The arc actually emitted for an edge ``a -> b`` (grid units, floats)."""

    carrier: str  # id of the arc carrier (registry key)
    root: str  # id of the source carrier whose exact circle the deviation is measured against
    centre: tuple[float, float]
    radius: float
    ccw: bool  # direction of travel a -> b
    bulge_left: bool  # whether the arc bulges to the left of a -> b
    segment_area: float  # area between chord and arc (grid^2)
    deviation: float  # max distance of the realised arc from the exact carrier circle (grid units)
    root_centre: tuple[float, float] = (0.0, 0.0)  # exact circle of the source carrier
    root_radius: float = 0.0

    @property
    def signed_segment_twice_area(self) -> float:
        """Twice the area the arc adds to the face on its *left*: positive when it bulges right (away from it)."""
        return 2.0 * self.segment_area * (-1 if self.bulge_left else 1)


def realise(arc: ArcCarrier, root: ArcCarrier, a: Point, b: Point) -> RealisedArc:
    """Circle arc of the carrier's radius through grid points ``a`` and ``b`` (see the module docstring)."""
    radius = float(arc.radius)
    dx, dy = b[0] - a[0], b[1] - a[1]
    chord = math.hypot(dx, dy)
    half = chord / 2
    effective_radius = max(radius, half)
    sagitta_height = math.sqrt(max(0.0, effective_radius**2 - half**2))
    cx, cy = float(arc.centre[0]), float(arc.centre[1])
    side = dx * (cy - a[1]) - dy * (cx - a[0])
    centre_left = side >= 0
    nx, ny = -dy / chord, dx / chord  # left normal of a -> b
    sign = 1.0 if centre_left else -1.0
    centre = (
        (a[0] + b[0]) / 2 + sign * sagitta_height * nx,
        (a[1] + b[1]) / 2 + sign * sagitta_height * ny,
    )
    theta = 2 * math.asin(min(1.0, half / effective_radius))
    segment_area = effective_radius**2 / 2 * (theta - math.sin(theta))
    root_centre = (float(root.centre[0]), float(root.centre[1]))
    root_radius = float(root.radius)
    unmeasured = RealisedArc(
        arc.id,
        root.id,
        centre,
        effective_radius,
        centre_left,
        not centre_left,
        segment_area,
        0.0,
        root_centre,
        root_radius,
    )
    deviation = max(
        abs(math.hypot(x - root_centre[0], y - root_centre[1]) - root_radius)
        for x, y in sample_arc(unmeasured, a, b)
    )
    return RealisedArc(
        arc.id,
        root.id,
        centre,
        effective_radius,
        centre_left,
        not centre_left,
        segment_area,
        deviation,
        root_centre,
        root_radius,
    )


def sample_arc(
    arc: RealisedArc, a: Point, b: Point, count: int = SAMPLES
) -> list[tuple[float, float]]:
    """``count`` points along the realised arc from ``a`` to ``b`` (inclusive)."""
    cx, cy = arc.centre
    t0 = math.atan2(a[1] - cy, a[0] - cx)
    t1 = math.atan2(b[1] - cy, b[0] - cx)
    sweep = (t1 - t0) % (2 * math.pi) if arc.ccw else -((t0 - t1) % (2 * math.pi))
    return [
        (
            cx + arc.radius * math.cos(t0 + sweep * i / (count - 1)),
            cy + arc.radius * math.sin(t0 + sweep * i / (count - 1)),
        )
        for i in range(count)
    ]


def derived_arc(
    arc_id: str,
    root: ArcCarrier,
    realised: RealisedArc,
    ends: tuple[Point, Point],
    lineage: Lineage,
) -> ArcCarrier:
    """The realised arc re-expressed as a carrier through grid points (input of the next arrangement)."""
    return ArcCarrier(
        arc_id,
        (Fraction(realised.centre[0]), Fraction(realised.centre[1])),
        Fraction(realised.radius),
        realised.ccw,
        (Fraction(ends[0][0]), Fraction(ends[0][1])),
        (Fraction(ends[1][0]), Fraction(ends[1][1])),
        lineage,
        root.root_id,
    )


def face_exact_area(
    arr: Arrangement, curves: list[RealisedArc | None], face_index: int
) -> float:
    """Area (grid^2) of a face with its arc edges realised: chord polygon plus signed circular segments."""
    twice = float(arr.face_twice_area(face_index))
    for cycle in arr.face_cycles(face_index):
        for h in arr.cycles[cycle].half_edges:
            curve = curves[h >> 1]
            if curve is not None:
                # signed twice-area the arc adds to the face on the left of the *travelled* direction
                twice += curve.signed_segment_twice_area * (1 if h % 2 == 0 else -1)
    return twice / 2


def sampled_crossings(
    arr: Arrangement, curves: list[RealisedArc | None], step: float = 8.0
) -> list[tuple[int, int]]:
    """Pairs of edges, at least one of them an arc, whose *realised* geometry crosses.

    Arcs are sampled every ``step`` grid units.  Two straight edges are not tested: their realised
    geometry is their chord, and ``arrangements_planar`` proves in exact integer arithmetic that no
    two chords cross.  Edges of a pair that merely share an end point are not crossings.  A float
    test used for validation and by the tests; it never decides topology.
    """
    polylines = [_edge_polyline(arr, curves, ei, step) for ei in range(len(arr.edges))]
    is_arc = [curve is not None for curve in curves]
    arc_cells = _segment_cells(
        [line if arc else [] for line, arc in zip(polylines, is_arc, strict=True)],
        CELL_SIZE,
    )
    straight_cells = _straight_edges_in(arc_cells, polylines, is_arc)
    found: set[tuple[int, int]] = set()
    for cell, arc_members in arc_cells.items():
        members = arc_members + straight_cells.get(cell, [])
        for i, (edge_a, seg_a) in enumerate(arc_members):
            for edge_b, seg_b in members[i + 1 :]:
                pair = (min(edge_a, edge_b), max(edge_a, edge_b))
                if edge_a == edge_b or pair in found:
                    continue
                if _segments_cross(
                    polylines[edge_a][seg_a],
                    polylines[edge_a][seg_a + 1],
                    polylines[edge_b][seg_b],
                    polylines[edge_b][seg_b + 1],
                ):
                    found.add(pair)
    return sorted(found)


def _edge_polyline(
    arr: Arrangement, curves: list[RealisedArc | None], edge: int, step: float
) -> list[tuple[float, float]]:
    e = arr.edges[edge]
    curve = curves[edge]
    if curve is None:
        return [e.a, e.b]
    chord = math.hypot(e.b[0] - e.a[0], e.b[1] - e.a[1])
    return sample_arc(curve, e.a, e.b, max(3, int(chord / step) + 2))


def _segment_cells(
    polylines: list[list[tuple[float, float]]], size: float
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """Broad phase: every ``(edge, segment)`` listed under each ``size`` cell the segment passes through.

    Two segments that cross share the cell of their crossing point, so testing the pairs of each cell
    finds every crossing.  A segment goes only into the cells along its path, not every cell of its
    bounding box: the box of an oblique segment holds about (length / size)^2 cells, and the boxes of
    long oblique edges across a die overlap, so their pairs were tested again in every shared cell.
    """
    cells: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for edge, polyline in enumerate(polylines):
        for segment in range(len(polyline) - 1):
            for cell in _cells_on_segment(
                polyline[segment], polyline[segment + 1], size
            ):
                cells.setdefault(cell, []).append((edge, segment))
    return cells


def _straight_edges_in(
    arc_cells: dict[tuple[int, int], list[tuple[int, int]]],
    polylines: list[list[tuple[float, float]]],
    is_arc: list[bool],
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """Every straight edge (its one segment) under each of its cells that also holds an arc segment."""
    cells: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for edge, polyline in enumerate(polylines):
        if is_arc[edge]:
            continue
        for cell in _cells_on_segment(polyline[0], polyline[1], CELL_SIZE):
            if cell in arc_cells:
                cells.setdefault(cell, []).append((edge, 0))
    return cells


def _cells_on_segment(
    p: tuple[float, float], q: tuple[float, float], size: float
) -> Iterator[tuple[int, int]]:
    # Column by column: clip the segment to the column's closed x interval, then take the rows its
    # clipped y range covers.  A point of the segment lies in its column's clipped piece, so its cell
    # is listed; CELL_PAD absorbs rounding of the clipped y values (a larger range, never a smaller one).
    (x0, y0), (x1, y1) = sorted((p, q))
    first, last = int(x0 // size), int(x1 // size)
    for column in range(first, last + 1):
        if first == last:
            ya, yb = y0, y1
        else:
            slope = (y1 - y0) / (x1 - x0)
            ya = y0 + slope * (max(x0, column * size) - x0)
            yb = y0 + slope * (min(x1, (column + 1) * size) - x0)
        low, high = min(ya, yb) - CELL_PAD, max(ya, yb) + CELL_PAD
        for row in range(int(low // size), int(high // size) + 1):
            yield column, row


def _segments_cross(
    p: tuple[float, float],
    q: tuple[float, float],
    r: tuple[float, float],
    s: tuple[float, float],
) -> bool:
    d1 = (q[0] - p[0], q[1] - p[1])
    d2 = (s[0] - r[0], s[1] - r[1])
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-12:
        return False
    t = ((r[0] - p[0]) * d2[1] - (r[1] - p[1]) * d2[0]) / den
    u = ((r[0] - p[0]) * d1[1] - (r[1] - p[1]) * d1[0]) / den
    return (
        CROSSING_EPSILON < t < 1 - CROSSING_EPSILON
        and CROSSING_EPSILON < u < 1 - CROSSING_EPSILON
    )
