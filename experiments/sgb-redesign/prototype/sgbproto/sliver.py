"""Sliver report (brief section 10): measurements only, no geometry change.

Owns *reporting* what is thin or short in a derived model so a Human can decide
about the Gate thresholds (D2). It computes, per case:

* **shortest edges** -- horizontal (arrangement) and vertical (wall column) edges;
* **narrowest faces** -- the local feature size of every face: the minimum distance between two
  non-incident boundary edges on the exact carriers (``feature_size``); ``2 A / P`` is kept as an extra field;
* **smallest angles** at arrangement vertices (the wedge between two consecutive
  edges, in degrees);
* **snap drift** -- distance from each edge end to the nearest source carrier
  (iterated snap rounding can exceed ``sqrt(2)/2 g``, see brief section 3).

Every entry carries its location (micrometres), plane and lineage (source
entities / carrier ids). Quantization, phantom and seam slivers are removed by
construction; what appears here are source facts or snap effects.
"""

from __future__ import annotations

import math
from fractions import Fraction

from sgbproto.feature_size import Prim, feature_size
from sgbproto.geometry import segment_point_distance_sq
from sgbproto.pipeline import BuildResult


def sliver_report(result: BuildResult, top: int = 10, drift: bool = True) -> dict:
    g = result.spec.grid.g
    report = {
        "grid_um": g,
        "top": top,
        "shortest_edges": _shortest_edges(result, top, g),
        "narrowest_faces": _narrowest_faces(result, top, g),
        "smallest_angles": _smallest_angles(result, top, g),
    }
    if drift:
        report["snap_drift"] = _snap_drift(result, top, g)
        report["arc_edges"] = _arc_edges(result, top, g)
    report["summary"] = {
        "shortest_edge_um": report["shortest_edges"][0]["value_um"] if report["shortest_edges"] else None,
        "narrowest_face_um": report["narrowest_faces"][0]["value_um"] if report["narrowest_faces"] else None,
        "smallest_angle_deg": report["smallest_angles"][0]["value_deg"] if report["smallest_angles"] else None,
        "max_snap_drift_grid": report.get("snap_drift", {}).get("max_grid_units"),
        "max_arc_deviation_grid": report.get("arc_edges", {}).get("max_deviation_grid"),
    }
    return report


def _entry(**fields) -> dict:
    return fields


def _shortest_edges(result: BuildResult, top: int, g: float) -> list[dict]:
    topo = result.topology
    rows = []
    for eid, e in topo.edges.items():
        a, b = topo.vertices[e.v0], topo.vertices[e.v1]
        length = math.dist((a.x * g, a.y * g, float(a.z)), (b.x * g, b.y * g, float(b.z)))
        rows.append((length, eid, e, a, b))
    rows.sort(key=lambda r: (r[0], r[1]))
    return [
        _entry(
            id=eid, kind=e.kind, value_um=round(length, 6), plane=e.plane,
            location_um=[round((a.x + b.x) / 2 * g, 4), round((a.y + b.y) / 2 * g, 4), round(float((a.z + b.z) / 2), 4)],
            sources=list(topo.lineage.get(eid, e.sources)),
        )
        for length, eid, e, a, b in rows[:top]
    ]


def _narrowest_faces(result: BuildResult, top: int, g: float) -> list[dict]:
    """Faces by local feature size (minimum distance between non-incident boundary edges, arcs exact)."""
    topo = result.topology
    rows = []
    for fid, f in topo.faces.items():
        if f.kind == "horizontal":
            prims = [_prim(topo, eid, sign) for loop in f.loops for eid, sign in loop]
            size, where, bounded = feature_size(prims)
            location = [where[0] * g, where[1] * g, float(f.z_range[0])]
            size_um = size * g
        else:
            size_um, location = _wall_feature(topo, f, g)
            bounded = False
        if not math.isfinite(size_um):
            continue
        points, perimeter = [], 0.0
        for loop in f.loops:
            pts = []
            for eid, sign in loop:
                e = topo.edges[eid]
                v = topo.vertices[e.v0 if sign > 0 else e.v1]
                pts.append((v.x * g, v.y * g, float(v.z)))
            perimeter += sum(math.dist(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts)))
            points.append(pts)
        area = abs(f.twice_xy_area) / 2 * g * g if f.kind == "horizontal" else _polygon_area_3d(points[0])
        two_a_over_p = 2 * area / perimeter if perimeter > 0 and area > 0 else None
        rows.append((size_um, fid, f, area, perimeter, two_a_over_p, location, bounded))
    rows.sort(key=lambda r: (r[0], r[1]))
    return [
        _entry(id=fid, kind=f.kind, group=f.group, value_um=round(w, 6), measure="local feature size",
               bounded=bounded, width_2a_over_p_um=None if r is None else round(r, 6), area_um2=round(a, 6),
               perimeter_um=round(p, 4), plane=f.plane, location_um=[round(c, 4) for c in loc], sources=list(f.sources))
        for w, fid, f, a, p, r, loc, bounded in rows[:top]
    ]


def _prim(topo, eid: str, sign: int) -> Prim:
    """A loop edge as an exact primitive in grid units (a segment, or the realised arc)."""
    e = topo.edges[eid]
    v0, v1 = topo.vertices[e.v0], topo.vertices[e.v1]
    a, b = ((v0.x, v0.y), (v1.x, v1.y)) if sign > 0 else ((v1.x, v1.y), (v0.x, v0.y))
    ids = (e.v0, e.v1)
    if e.curve is None:
        return Prim(a, b, ids=ids)
    c = e.curve
    ccw = c.ccw if sign > 0 else not c.ccw
    chord = math.hypot(b[0] - a[0], b[1] - a[1])
    theta = 2 * math.asin(min(1.0, chord / (2 * c.radius)))
    t0 = math.atan2(a[1] - c.centre[1], a[0] - c.centre[0])
    return Prim(a, b, c.centre, c.radius, t0, theta if ccw else -theta, ids)


def _wall_feature(topo, face, g: float) -> tuple[float, list[float]]:
    """A wall is a rectangle-like strip: min(height, distance between its two vertical edges)."""
    columns = {}
    for eid, _ in face.loops[0]:
        e = topo.edges[eid]
        if e.kind == "vertical":
            v = topo.vertices[e.v0]
            columns[(v.x, v.y)] = float(v.z)
    pts = list(columns)
    gap = min((math.dist(p, q) for i, p in enumerate(pts) for q in pts[i + 1:]), default=math.inf) * g
    height = float(face.z_range[1] - face.z_range[0])
    mid = [sum(p[0] for p in pts) / len(pts) * g, sum(p[1] for p in pts) / len(pts) * g, float(face.z_range[0]) + height / 2]
    return min(height, gap), mid


def _polygon_area_3d(pts: list[tuple[float, float, float]]) -> float:
    sx = sy = sz = 0.0
    for i in range(len(pts)):
        x0, y0, z0 = pts[i]
        x1, y1, z1 = pts[(i + 1) % len(pts)]
        sx += y0 * z1 - z0 * y1
        sy += z0 * x1 - x0 * z1
        sz += x0 * y1 - y0 * x1
    return 0.5 * math.sqrt(sx * sx + sy * sy + sz * sz)


def _smallest_angles(result: BuildResult, top: int, g: float) -> list[dict]:
    rows = []
    for layer in result.level_layers:
        arr = layer.arr
        around: dict[int, list[tuple[float, int]]] = {}
        for ei, (a, b) in enumerate(arr.edge_ends):
            pa, pb = arr.vertices[a], arr.vertices[b]
            around.setdefault(a, []).append((math.atan2(pb[1] - pa[1], pb[0] - pa[0]), ei))
            around.setdefault(b, []).append((math.atan2(pa[1] - pb[1], pa[0] - pb[0]), ei))
        for vi, items in around.items():
            if len(items) < 2:
                continue
            items.sort()
            for k in range(len(items)):
                a0, e0 = items[k]
                a1, e1 = items[(k + 1) % len(items)]
                gap = (a1 - a0) % (2 * math.pi) if len(items) > 1 else 2 * math.pi
                if gap == 0.0:
                    gap = 2 * math.pi
                rows.append((math.degrees(gap), layer, vi, e0, e1))
    rows.sort(key=lambda r: (r[0], r[1].plane, r[2], r[3], r[4]))
    out = []
    for deg, layer, vi, e0, e1 in rows[:top]:
        p = layer.arr.vertices[vi]
        out.append(_entry(
            value_deg=round(deg, 6), plane=layer.plane, vertex=layer.vertex_ids[vi],
            location_um=[round(p[0] * g, 4), round(p[1] * g, 4), round(float(layer.z), 4)],
            edges=[layer.edge_ids[e0], layer.edge_ids[e1]],
            sources=sorted(set(layer.edge_sources[e0]) | set(layer.edge_sources[e1])),
        ))
    return out


def _snap_drift(result: BuildResult, top: int, g: float) -> dict:
    carriers = {}
    for prism in result.spec.prisms:
        for c in prism.carriers():
            carriers[c.id] = (c.p, c.q)
    for fam in result.spec.families:
        for c in fam.carriers:
            carriers[c.id] = (c.p, c.q)
    rows = []
    for layer in result.slabs + result.level_layers:
        for e, eid, sources in zip(layer.arr.edges, layer.edge_ids, layer.edge_sources):
            best = None
            for s in sources:
                if s not in carriers:
                    continue
                p, q = carriers[s]
                d = max(segment_point_distance_sq(p, q, e.a), segment_point_distance_sq(p, q, e.b))
                best = d if best is None else min(best, d)
            if best is not None:
                rows.append((math.sqrt(float(best)), layer.plane, eid, e, sources))
    rows.sort(key=lambda r: (-r[0], r[1], r[2]))
    bound = math.sqrt(2) / 2
    return {
        "bound_grid_units": round(bound, 6),
        "max_grid_units": round(rows[0][0], 6) if rows else 0.0,
        "edges_over_bound": sum(1 for r in rows if r[0] > bound + 1e-9),
        "edges_measured": len(rows),
        "worst": [
            _entry(id=eid, plane=plane, value_grid_units=round(d, 6), value_um=round(d * g, 6),
                   location_um=[round(e.a[0] * g, 4), round(e.a[1] * g, 4)], sources=list(src))
            for d, plane, eid, e, src in rows[:top]
        ],
    }


def _arc_edges(result: BuildResult, top: int, g: float) -> dict:
    """Realised arc edges: deviation from the exact carrier circle (grid units), per edge, worst first."""
    rows = []
    for layer in result.slabs + result.level_layers:
        for e, eid, src, curve in zip(layer.arr.edges, layer.edge_ids, layer.edge_sources, layer.edge_curve):
            if curve is not None:
                rows.append((curve.deviation, layer.plane, eid, e, curve, src))
    rows.sort(key=lambda r: (-r[0], r[1], r[2]))
    return {
        "count": len(rows),
        "max_deviation_grid": round(rows[0][0], 6) if rows else 0.0,
        "mean_deviation_grid": round(sum(r[0] for r in rows) / len(rows), 6) if rows else 0.0,
        "worst": [
            _entry(id=eid, plane=plane, value_grid_units=round(d, 6), radius_um=round(c.radius * g, 4),
                   location_um=[round(e.a[0] * g, 4), round(e.a[1] * g, 4)], sources=list(src))
            for d, plane, eid, e, c, src in rows[:top]
        ],
    }
