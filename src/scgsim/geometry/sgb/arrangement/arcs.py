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
from dataclasses import dataclass
from fractions import Fraction

from scgsim.geometry.sgb.arrangement.dcel import Arrangement
from scgsim.geometry.sgb.plane import Point
from scgsim.geometry.sgb.records.carriers import ArcCarrier, Lineage

SAMPLES = 33
CROSSING_EPSILON = 1e-7


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
    """Pairs of edges whose *realised* geometry crosses, found by sampling arcs every ``step`` grid units.

    Edges of a pair that merely share an end point are not crossings.  A float
    test used for validation and by the tests; it never decides topology.
    """
    polylines = [_edge_polyline(arr, curves, ei, step) for ei in range(len(arr.edges))]
    cells = _segment_cells(polylines, 256.0)
    found: set[tuple[int, int]] = set()
    for members in cells.values():
        for i, (edge_a, seg_a) in enumerate(members):
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
    cells: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for edge, polyline in enumerate(polylines):
        for segment in range(len(polyline) - 1):
            (x0, y0), (x1, y1) = polyline[segment], polyline[segment + 1]
            for cx in range(int(min(x0, x1) // size), int(max(x0, x1) // size) + 1):
                for cy in range(int(min(y0, y1) // size), int(max(y0, y1) // size) + 1):
                    cells.setdefault((cx, cy), []).append((edge, segment))
    return cells


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
