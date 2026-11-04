"""Arc edges: polyline witness back-mapping and realisation (brief section 3, open decision D3).

Owns what happens to circular arcs between the source and the emitted model:

* **Witness.** An arc carrier is noded as a chain of integer chords (sagitta <= g/4) whose carrier ids are
  ``<arc id>#<k>`` (see ``records.ArcCarrier.witness_points``). Snap rounding and dissolution treat them as
  ordinary segments.
* **Merge.** Dissolution removes a degree-2 vertex whose two edges carry pieces of the same arc, so a run of
  witness chords with no foreign hot pixel between them becomes one edge whose carrier set is just ``{arc id}``.
  Everything downstream sees *one arc edge between two snapped vertices*.
* **Realisation.** An arc edge ``a -> b`` is the circle arc of the carrier's radius through the two snapped
  endpoints, with its centre on the same side as the carrier's centre (so the realised arc bulges the same way).
  Its deviation from the exact carrier circle is recorded per edge.

Does not own arrangement building or emission; it supplies the geometry both consult.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from sgbproto.geometry import Point
from sgbproto.records import ArcCarrier

SAMPLES = 33


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
    """Circle arc of the carrier's radius through grid points ``a`` and ``b`` (see module docstring)."""
    r = float(arc.radius)
    dx, dy = b[0] - a[0], b[1] - a[1]
    chord = math.hypot(dx, dy)
    half = chord / 2
    r_eff = max(r, half)
    h = math.sqrt(max(0.0, r_eff * r_eff - half * half))
    cx, cy = float(arc.centre[0]), float(arc.centre[1])
    side = dx * (cy - a[1]) - dy * (cx - a[0])
    centre_left = side >= 0
    nx, ny = -dy / chord, dx / chord  # left normal of a -> b
    sign = 1.0 if centre_left else -1.0
    mx, my = (a[0] + b[0]) / 2 + sign * h * nx, (a[1] + b[1]) / 2 + sign * h * ny
    theta = 2 * math.asin(min(1.0, half / r_eff))
    segment = r_eff * r_eff / 2 * (theta - math.sin(theta))
    root_centre = (float(root.centre[0]), float(root.centre[1]))
    realised = RealisedArc(
        arc.id, root.id, (mx, my), r_eff, centre_left, not centre_left, segment, 0.0, root_centre, float(root.radius)
    )
    return RealisedArc(
        realised.carrier, realised.root, realised.centre, realised.radius, realised.ccw, realised.bulge_left,
        segment, _deviation(realised, a, b, root), root_centre, float(root.radius),
    )


def sample_arc(arc: RealisedArc, a: Point, b: Point, n: int = SAMPLES) -> list[tuple[float, float]]:
    """``n`` points along the realised arc from ``a`` to ``b`` (inclusive)."""
    cx, cy = arc.centre
    t0 = math.atan2(a[1] - cy, a[0] - cx)
    t1 = math.atan2(b[1] - cy, b[0] - cx)
    sweep = (t1 - t0) % (2 * math.pi) if arc.ccw else -((t0 - t1) % (2 * math.pi))
    return [
        (cx + arc.radius * math.cos(t0 + sweep * i / (n - 1)), cy + arc.radius * math.sin(t0 + sweep * i / (n - 1)))
        for i in range(n)
    ]


def _deviation(arc: RealisedArc, a: Point, b: Point, root: ArcCarrier) -> float:
    cx, cy, r = float(root.centre[0]), float(root.centre[1]), float(root.radius)
    return max(abs(math.hypot(x - cx, y - cy) - r) for x, y in sample_arc(arc, a, b))


def derived_arc(arc_id: str, root: ArcCarrier, realised: RealisedArc, a: Point, b: Point, lineage) -> ArcCarrier:
    """The realised arc re-expressed as a carrier through grid points (used as input of the next arrangement)."""
    from fractions import Fraction

    return ArcCarrier(
        arc_id, (Fraction(realised.centre[0]), Fraction(realised.centre[1])), Fraction(realised.radius),
        realised.ccw, (Fraction(a[0]), Fraction(a[1])), (Fraction(b[0]), Fraction(b[1])), lineage, root.root_id,
    )


def sampled_crossings(arr, curves: list, step: float = 8.0) -> list[tuple[int, int]]:
    """Pairs of edges whose *realised* geometry crosses, found by sampling arcs every ``step`` grid units.

    Edges of a pair that merely share an end point are not crossings. Pure float test used for validation and
    by the tests (with a much smaller ``step``); it never decides topology.
    """
    polylines: list[list[tuple[float, float]]] = []
    for ei, e in enumerate(arr.edges):
        curve = curves[ei]
        if curve is None:
            polylines.append([e.a, e.b])
        else:
            chord = math.hypot(e.b[0] - e.a[0], e.b[1] - e.a[1])
            polylines.append(sample_arc(curve, e.a, e.b, max(3, int(chord / step) + 2)))
    size = 256.0
    cells: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for ei, poly in enumerate(polylines):
        for si in range(len(poly) - 1):
            (x0, y0), (x1, y1) = poly[si], poly[si + 1]
            for cx in range(int(min(x0, x1) // size), int(max(x0, x1) // size) + 1):
                for cy in range(int(min(y0, y1) // size), int(max(y0, y1) // size) + 1):
                    cells.setdefault((cx, cy), []).append((ei, si))
    found: set[tuple[int, int]] = set()
    for members in cells.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                (ea, sa), (eb, sb) = members[i], members[j]
                if ea == eb or (min(ea, eb), max(ea, eb)) in found:
                    continue
                if _segments_cross(polylines[ea][sa], polylines[ea][sa + 1], polylines[eb][sb], polylines[eb][sb + 1]):
                    found.add((min(ea, eb), max(ea, eb)))
    return sorted(found)


def _segments_cross(p, q, r, s, eps: float = 1e-7) -> bool:
    d1 = (q[0] - p[0], q[1] - p[1])
    d2 = (s[0] - r[0], s[1] - r[1])
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-12:
        return False
    t = ((r[0] - p[0]) * d2[1] - (r[1] - p[1]) * d2[0]) / den
    u = ((r[0] - p[0]) * d1[1] - (r[1] - p[1]) * d1[0]) / den
    return eps < t < 1 - eps and eps < u < 1 - eps


def face_exact_area(arr, curves: list, face_index: int) -> float:
    """Area (grid^2) of a face with its arc edges realised: chord polygon plus signed circular segments."""
    twice = float(arr.face_twice_area(face_index))
    for cycle in arr.face_cycles(face_index):
        for h in arr.cycles[cycle].half_edges:
            curve = curves[h >> 1]
            if curve is not None:
                # signed twice-area the arc adds to the face on the left of the *travelled* direction
                twice += curve.signed_segment_twice_area * (1 if h % 2 == 0 else -1)
    return twice / 2


def cycle_polyline(arr, curves: list, cycle_index: int, step: float = 4.0) -> list[tuple[float, float]]:
    """A boundary cycle as a closed float polyline with every arc edge sampled (for drawing and area checks)."""
    pts: list[tuple[float, float]] = []
    for h in arr.cycles[cycle_index].half_edges:
        a, b = arr.origin(h), arr.target(h)
        curve = curves[h >> 1]
        if curve is None:
            pts.append((float(a[0]), float(a[1])))
            continue
        ea, eb = arr.edges[h >> 1].a, arr.edges[h >> 1].b
        chord = math.hypot(eb[0] - ea[0], eb[1] - ea[1])
        sampled = sample_arc(curve, ea, eb, max(3, int(chord / step) + 2))
        pts.extend(sampled[:-1] if h % 2 == 0 else reversed(sampled[1:]))
    return pts
