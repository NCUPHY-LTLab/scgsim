"""Surface-EPR inset bands (brief sections 5 and 6): the first cut-family consumer.

Owns generating, from a *support region* ``R`` (rings of exact line and arc primitives), the boundary curves of
``R_m = R (-) D(m)`` (Minkowski erosion by a disk) for margins ``m_1 < ... < m_K`` and the band label of any point.

Construction (no polygonised disk):

* every line is offset by ``m`` toward the region interior (a line);
* every convex arc (centre on the interior side) becomes the concentric arc of radius ``rho - m`` (it vanishes to
  a sharp corner when ``m >= rho``); every concave arc (e.g. a hole) becomes the concentric arc of radius ``r + m``;
* at every reflex vertex the offset curves leave a gap that is bridged by an arc of radius ``m`` centred at the
  vertex; at convex vertices the offset lines overlap and simply cross;
* these *candidate* curves are noded together by snap rounding, and an edge is kept iff its midpoint is at distance
  ``>= m`` from the region boundary (the ``dist >= m`` cells of the brief); dangling stubs are removed.

Distance origin: only the support region's own boundary primitives. Ports, contact caps, other bands and tile seams
are never origins -- they are not inputs of this module at all.

Bands: ``B_0`` = within ``m_1`` of the boundary, ``B_k`` = between ``m_k`` and ``m_{k+1}``, ``B_K`` = at least
``m_K`` away. "At least m_k from the edge" is the union of bands ``>= k``.

The offset of an oblique edge is not representable on the grid (its normal is irrational); curve end points are
snapped to the grid like every other vertex, so the carriers are exact up to the snap bound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from sgbproto.geometry import Point
from sgbproto.records import ArcCarrier, CurveFamilyRecord, LineCarrier, LineEdge, Lineage, base_carrier_id
from sgbproto.snap_rounding import Segment, snap_round

Primitive = LineEdge | ArcCarrier
EPS = 1e-9


def reverse_ring(prims: list[Primitive]) -> list[Primitive]:
    """The same boundary traversed the other way (a hole of one region is the outer ring of its complement)."""
    out: list[Primitive] = []
    for prim in reversed(prims):
        if isinstance(prim, LineEdge):
            out.append(LineEdge(prim.q, prim.p))
        else:
            out.append(ArcCarrier(prim.id, prim.centre, prim.radius, not prim.ccw, prim.end, prim.start,
                                  prim.lineage, prim.root))
    return out


# --------------------------------------------------------------------------- primitive geometry (floats)


def _start(prim: Primitive) -> tuple[float, float]:
    return (float(prim.start[0]), float(prim.start[1])) if isinstance(prim, ArcCarrier) else (float(prim.p[0]), float(prim.p[1]))


def _end(prim: Primitive) -> tuple[float, float]:
    return (float(prim.end[0]), float(prim.end[1])) if isinstance(prim, ArcCarrier) else (float(prim.q[0]), float(prim.q[1]))


def _tangent(prim: Primitive, at_end: bool) -> tuple[float, float]:
    if isinstance(prim, LineEdge):
        dx, dy = prim.q[0] - prim.p[0], prim.q[1] - prim.p[1]
        n = math.hypot(dx, dy)
        return dx / n, dy / n
    cx, cy = float(prim.centre[0]), float(prim.centre[1])
    x, y = _end(prim) if at_end else _start(prim)
    n = math.hypot(x - cx, y - cy)
    ux, uy = (x - cx) / n, (y - cy) / n
    return (-uy, ux) if prim.ccw else (uy, -ux)


def _left_normal(prim: Primitive, at_end: bool) -> tuple[float, float]:
    tx, ty = _tangent(prim, at_end)
    return -ty, tx


def _offset_point(prim: Primitive, at_end: bool, m: float) -> tuple[float, float]:
    """End point of the offset of ``prim`` by ``m`` toward the interior (single source of the rounded value)."""
    x, y = _end(prim) if at_end else _start(prim)
    nx, ny = _left_normal(prim, at_end)
    return x + m * nx, y + m * ny


def _round(p: tuple[float, float]) -> Point:
    return math.floor(p[0] + 0.5), math.floor(p[1] + 0.5)


def distance_to_primitive(prim: Primitive, x: float, y: float) -> float:
    if isinstance(prim, LineEdge):
        px, py, qx, qy = prim.p[0], prim.p[1], prim.q[0], prim.q[1]
        dx, dy = qx - px, qy - py
        t = max(0.0, min(1.0, ((x - px) * dx + (y - py) * dy) / (dx * dx + dy * dy)))
        return math.hypot(x - (px + t * dx), y - (py + t * dy))
    cx, cy, r = float(prim.centre[0]), float(prim.centre[1]), float(prim.radius)
    t0, sweep = prim.sweep()
    angle = math.atan2(y - cy, x - cx)
    delta = (angle - t0) % (2 * math.pi) if sweep > 0 else -((t0 - angle) % (2 * math.pi))
    if (sweep > 0 and 0 <= delta <= sweep) or (sweep < 0 and sweep <= delta <= 0):
        return abs(math.hypot(x - cx, y - cy) - r)
    sx, sy = _start(prim)
    ex, ey = _end(prim)
    return min(math.hypot(x - sx, y - sy), math.hypot(x - ex, y - ey))


# --------------------------------------------------------------------------- support region


@dataclass(frozen=True)
class SupportRegion:
    """A planar region bounded by rings of exact primitives (interior on the left of each ring) and its margins."""

    rings: tuple[tuple[Primitive, ...], ...]
    margins: tuple[int, ...]  # grid units, strictly increasing

    def witness_rings(self) -> list[list[Point]]:
        out = []
        for ring in self.rings:
            pts: list[Point] = []
            for prim in ring:
                if isinstance(prim, LineEdge):
                    pts.append(prim.p)
                else:
                    pts.extend(prim.witness_points()[:-1])
            out.append(pts)
        return out

    def contains(self, point: tuple[float, float]) -> bool:
        """Even-odd containment in the witness rings (exact enough: points are never near the boundary)."""
        x, y = point
        inside = False
        for pts in self._witness():
            n = len(pts)
            for i in range(n):
                (ax, ay), (bx, by) = pts[i], pts[(i + 1) % n]
                if (ay > y) != (by > y) and x < ax + (y - ay) * (bx - ax) / (by - ay):
                    inside = not inside
        return inside

    def _witness(self) -> list[list[Point]]:
        cached = self.__dict__.get("_cache")
        if cached is None:
            cached = self.witness_rings()
            object.__setattr__(self, "_cache", cached)
        return cached

    def distance(self, point: tuple[float, float]) -> float:
        return min(distance_to_primitive(p, point[0], point[1]) for ring in self.rings for p in ring)

    def band_of(self, point: tuple[float, float]) -> int | None:
        """Band index of a point inside the region (0 = closest to the boundary), ``None`` outside."""
        if not self.contains(point):
            return None
        d = self.distance(point)
        return sum(1 for m in self.margins if d >= m)

    def area(self) -> float:
        """Exact area (grid^2) of the region: line chords by the shoelace formula, arcs by their sector term."""
        twice = 0.0
        for ring in self.rings:
            for prim in ring:
                if isinstance(prim, LineEdge):
                    twice += prim.p[0] * prim.q[1] - prim.q[0] * prim.p[1]
                else:
                    cx, cy, r = float(prim.centre[0]), float(prim.centre[1]), float(prim.radius)
                    (sx, sy), (ex, ey) = _start(prim), _end(prim)
                    twice += cx * (ey - sy) - cy * (ex - sx) + r * r * prim.sweep()[1]
        return twice / 2

    def perimeter(self) -> float:
        total = 0.0
        for ring in self.rings:
            for prim in ring:
                if isinstance(prim, LineEdge):
                    total += math.hypot(prim.q[0] - prim.p[0], prim.q[1] - prim.p[1])
                else:
                    total += float(prim.radius) * abs(prim.sweep()[1])
        return total

    @property
    def band_count(self) -> int:
        return len(self.margins) + 1


# --------------------------------------------------------------------------- curve generation


def inset_family(
    family_id: str,
    region: SupportRegion,
    target_levels: tuple[Fraction, ...],
    tolerance: float = 1.5,
) -> CurveFamilyRecord:
    """The ``required`` cut family holding the boundaries of ``R_m`` for every margin of the region."""
    lines: list[LineCarrier] = []
    arcs: list[ArcCarrier] = []
    for m in region.margins:
        ls, as_ = _boundary_of_eroded(family_id, region, m, tolerance)
        lines.extend(ls)
        arcs.extend(as_)
    return CurveFamilyRecord(family_id, "inset_band", tuple(lines), target_levels, "required", tuple(arcs), region)


def _boundary_of_eroded(family_id: str, region: SupportRegion, m: int, tolerance: float):
    candidates_lines: list[tuple[str, Point, Point]] = []
    candidate_arcs: dict[str, ArcCarrier] = {}
    # A member's id is ``<support>/ring<r>/edge<k>@m<margin>``: the generating primitive plus the margin; its
    # lineage keeps the primitive alone (``<support>/ring<r>/edge<k>``), for straight members and arcs alike.
    tag = f"@m{m}"

    def add_line(name: str, p: Point, q: Point) -> None:
        if p != q:
            candidates_lines.append((f"{family_id}/{name}{tag}", p, q))

    def add_arc(name: str, centre, radius: float, ccw: bool, start, end, root: str) -> None:
        arc = ArcCarrier(f"{family_id}/{name}{tag}", (Fraction(centre[0]), Fraction(centre[1])),
                         Fraction(radius).limit_denominator(10**9),
                         ccw, (Fraction(start[0]), Fraction(start[1])), (Fraction(end[0]), Fraction(end[1])),
                         Lineage("family_member", family_id=family_id, member=f"{family_id}/{name}"), root)
        if _round(start) != _round(end):
            candidate_arcs[arc.id] = arc

    for ri, ring in enumerate(region.rings):
        n = len(ring)
        for pi, prim in enumerate(ring):
            name = f"ring{ri}/edge{pi}"
            sp, ep = _round(_offset_point(prim, False, m)), _round(_offset_point(prim, True, m))
            if isinstance(prim, LineEdge):
                add_line(name, sp, ep)
            else:
                r = float(prim.radius) + (-m if prim.ccw else m)
                if r >= 1.0:
                    add_arc(name, prim.centre, r, prim.ccw, sp, ep, "")
            nxt = ring[(pi + 1) % n]
            _corner_arcs(f"ring{ri}/corner{pi}", prim, nxt, m, add_arc)

    segments = [Segment(p, q, frozenset([cid])) for cid, p, q in candidates_lines]
    arc_pieces: dict[str, tuple[ArcCarrier, list[Point]]] = {}
    for arc in candidate_arcs.values():
        pts = arc.witness_points()
        arc_pieces[arc.id] = (arc, pts)
        segments.extend(Segment(p, q, frozenset([f"{arc.id}#{k}"])) for k, (p, q) in enumerate(zip(pts, pts[1:])))
    kept = _kept_edges(snap_round(segments).edges, region, m, tolerance)

    lines_out: list[LineCarrier] = []
    kept_pieces: dict[str, set[int]] = {}
    kept_points: dict[tuple[str, int], set[Point]] = {}  # end points of the kept (possibly trimmed) part of each chord
    pieces_of_line: dict[str, int] = {}
    for e in kept:
        for c in e.carriers:
            base, _, k = c.partition("#")
            if k:
                kept_pieces.setdefault(base, set()).add(int(k))
                kept_points.setdefault((base, int(k)), set()).update((e.a, e.b))
        if not any("#" in c for c in e.carriers):
            source = sorted(e.carriers)[0]  # the offset line (one primitive of the support) this piece lies on
            n = pieces_of_line.get(source, 0)
            pieces_of_line[source] = n + 1
            member = source.split("@", 1)[0]
            lines_out.append(LineCarrier(f"{source}.{n}", e.a, e.b,
                                         Lineage("family_member", family_id=family_id, member=member)))
    arcs_out = _arc_runs(kept_pieces, arc_pieces, family_id, "", kept_points)
    return lines_out, arcs_out


def _corner_arcs(name, prim, nxt, m, add_arc) -> None:
    """Arc of radius ``m`` about a reflex vertex joining the offsets of two consecutive primitives."""
    t0, t1 = _tangent(prim, True), _tangent(nxt, False)
    cross = t0[0] * t1[1] - t0[1] * t1[0]
    dot = t0[0] * t1[0] + t0[1] * t1[1]
    if cross >= -EPS:
        return  # straight/tangent junction or a convex corner: the offsets meet or overlap
    phi = math.atan2(-cross, dot)  # right turn angle in (0, pi)
    vx, vy = _end(prim)
    n0 = _left_normal(prim, True)
    pieces = max(1, math.ceil(phi / (math.pi / 2) - 1e-9))
    for k in range(pieces):
        a0, a1 = phi * k / pieces, phi * (k + 1) / pieces
        start = (vx + m * (n0[0] * math.cos(a0) + n0[1] * math.sin(a0)), vy + m * (-n0[0] * math.sin(a0) + n0[1] * math.cos(a0)))
        end = (vx + m * (n0[0] * math.cos(a1) + n0[1] * math.sin(a1)), vy + m * (-n0[0] * math.sin(a1) + n0[1] * math.cos(a1)))
        if k == 0:
            start = _offset_point(prim, True, m)
        if k == pieces - 1:
            end = _offset_point(nxt, False, m)
        add_arc(name if pieces == 1 else f"{name}.{k}", (round(vx), round(vy)), m, False, _round(start), _round(end), "")


def _kept_edges(edges, region: SupportRegion, m: int, tolerance: float):
    """Edges whose midpoint is at least ``m`` from the region boundary, with dangling stubs removed."""
    kept = []
    for e in edges:
        mid = ((e.a[0] + e.b[0]) / 2, (e.a[1] + e.b[1]) / 2)
        if region.contains(mid) and region.distance(mid) >= m - tolerance:
            kept.append(e)
    while True:
        degree: dict[Point, int] = {}
        for e in kept:
            degree[e.a] = degree.get(e.a, 0) + 1
            degree[e.b] = degree.get(e.b, 0) + 1
        pruned = [e for e in kept if degree[e.a] > 1 and degree[e.b] > 1]
        if len(pruned) == len(kept):
            return kept
        kept = pruned


def _along(chord: tuple[Point, Point], point: Point) -> int:
    """Position of ``point`` along the chord (dot product with its direction)."""
    (px, py), (qx, qy) = chord
    return (point[0] - px) * (qx - px) + (point[1] - py) * (qy - py)


def _arc_runs(kept_pieces, arc_pieces, family_id: str, tag: str, kept_points=None) -> list[ArcCarrier]:
    """Maximal runs of kept witness chords of each candidate arc, as sub-arcs through their end points.

    A chord that was trimmed where another carrier crosses it (the crossing lies inside the chord) ends the run at
    the crossing, not at the chord's far end; otherwise the arc would stick out past the crossing as a dangling stub.
    """
    out: list[ArcCarrier] = []
    for arc_id in sorted(kept_pieces):
        arc, pts = arc_pieces[arc_id]
        ks = sorted(kept_pieces[arc_id])
        runs, start = [], ks[0]
        for prev, cur in zip(ks, ks[1:]):
            if cur != prev + 1:
                runs.append((start, prev))
                start = cur
        runs.append((start, ks[-1]))
        for i, (k0, k1) in enumerate(runs):
            a, b = pts[k0], pts[k1 + 1]
            if kept_points:
                first = kept_points.get((arc_id, k0))
                last = kept_points.get((arc_id, k1))
                if first:
                    a = min(first, key=lambda pt: _along((pts[k0], pts[k0 + 1]), pt))
                if last:
                    b = max(last, key=lambda pt: _along((pts[k1], pts[k1 + 1]), pt))
            if a == b:
                continue
            out.append(ArcCarrier(f"{arc_id}.{i}", arc.centre, arc.radius, arc.ccw, (Fraction(a[0]), Fraction(a[1])),
                                  (Fraction(b[0]), Fraction(b[1])), arc.lineage, arc.root_id))
    return out
