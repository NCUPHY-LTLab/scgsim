"""Sliver report (design brief section 10): measurements only, never a geometry change or a failure.

Owns *reporting* what is thin or short in a derived model, so a Human can decide about
thresholds (decision D2: version 1 sets none).  Per model:

* **shortest edges** -- horizontal (arrangement) and vertical (wall column) edges;
* **narrowest faces** -- the local feature size of every face: the minimum distance between two
  non-incident boundary edges on the exact carriers (``feature_size``); ``2 A / P`` is kept as an
  extra field;
* **smallest angles** at arrangement vertices (the wedge between two consecutive edges, degrees);
* **thin slabs** -- slabs thinner than the grid, kept exactly as declared;
* **snap drift** -- distance from each edge end to the nearest source carrier (iterated snap
  rounding can exceed ``sqrt(2)/2 g``, see design brief section 3);
* **arc edges** -- the deviation of every realised arc from its exact carrier circle.

Every entry carries its location (micrometres), plane and lineage (source entities and carrier
ids).  Quantization, phantom and seam slivers are removed by construction; what appears here
are source facts or snap effects.
"""

from __future__ import annotations

import math
from typing import Any

from scgsim.geometry.sgb.topology.assembly.records import TopoFace, Topology
from scgsim.geometry.sgb.model import ConformalModel
from scgsim.geometry.sgb.exact.plane import segment_point_distance_sq
from scgsim.geometry.sgb.checks.slivers.feature_size import Prim, feature_size

Entry = dict[str, Any]


def sliver_report(model: ConformalModel, top: int = 10, *, drift: bool = True) -> Entry:
    """The sliver measurements of a model, worst first."""
    g = model.labelled.catalogue.grid.size_um
    report: Entry = {
        "grid_um": g,
        "top": top,
        "shortest_edges": _shortest_edges(model, top, g),
        "narrowest_faces": _narrowest_faces(model, top, g),
        "smallest_angles": _smallest_angles(model, top, g),
        "thin_slabs": _thin_slabs(model, g),
    }
    if drift:
        report["snap_drift"] = _snap_drift(model, top, g)
        report["arc_edges"] = _arc_edges(model, top, g)
    report["summary"] = {
        "shortest_edge_um": report["shortest_edges"][0]["value_um"]
        if report["shortest_edges"]
        else None,
        "narrowest_face_um": report["narrowest_faces"][0]["value_um"]
        if report["narrowest_faces"]
        else None,
        "smallest_angle_deg": report["smallest_angles"][0]["value_deg"]
        if report["smallest_angles"]
        else None,
        "thin_slab_count": len(report["thin_slabs"]),
        "max_snap_drift_grid": report.get("snap_drift", {}).get("max_grid_units"),
        "max_arc_deviation_grid": report.get("arc_edges", {}).get("max_deviation_grid"),
    }
    return report


def _thin_slabs(model: ConformalModel, g: float) -> list[Entry]:
    levels = model.labelled.levels.levels
    rows = [
        {
            "z_um": [float(lo), float(hi)],
            "thickness_um": float(hi - lo),
            "thickness_grid_units": float(hi - lo) / g,
        }
        for lo, hi in zip(levels, levels[1:], strict=False)
        if float(hi - lo) < g
    ]
    return sorted(rows, key=lambda r: r["thickness_um"])


def _shortest_edges(model: ConformalModel, top: int, g: float) -> list[Entry]:
    topo = model.topology
    rows = []
    for eid, e in topo.edges.items():
        a, b = topo.vertices[e.v0], topo.vertices[e.v1]
        length = math.dist(
            (a.x * g, a.y * g, float(a.z)), (b.x * g, b.y * g, float(b.z))
        )
        rows.append((length, eid, e, a, b))
    rows.sort(key=lambda r: (r[0], r[1]))
    return [
        {
            "id": eid,
            "kind": e.kind,
            "value_um": round(length, 6),
            "plane": e.plane,
            "location_um": [
                round((a.x + b.x) / 2 * g, 4),
                round((a.y + b.y) / 2 * g, 4),
                round(float((a.z + b.z) / 2), 4),
            ],
            "sources": list(topo.lineage.get(eid) or e.sources),
        }
        for length, eid, e, a, b in rows[:top]
    ]


def _narrowest_faces(model: ConformalModel, top: int, g: float) -> list[Entry]:
    """Faces by local feature size (minimum distance between non-incident boundary edges, arcs exact)."""
    topo = model.topology
    rows = []
    for fid, face in topo.faces.items():
        size_um, location, bounded = _face_feature(topo, face, g)
        if not math.isfinite(size_um):
            continue
        area, perimeter = _area_and_perimeter(topo, face, g)
        two_a_over_p = 2 * area / perimeter if perimeter > 0 and area > 0 else None
        rows.append(
            (size_um, fid, face, area, perimeter, two_a_over_p, location, bounded)
        )
    rows.sort(key=lambda r: (r[0], r[1]))
    return [
        {
            "id": fid,
            "kind": face.kind,
            "group": model.routed.face_group.get(fid),
            "value_um": round(size, 6),
            "measure": "local feature size",
            "bounded": bounded,
            "width_2a_over_p_um": None if ratio is None else round(ratio, 6),
            "area_um2": round(area, 6),
            "perimeter_um": round(perimeter, 4),
            "plane": face.plane,
            "location_um": [round(c, 4) for c in location],
            "sources": list(face.sources),
        }
        for size, fid, face, area, perimeter, ratio, location, bounded in rows[:top]
    ]


def _face_feature(
    topo: Topology, face: TopoFace, g: float
) -> tuple[float, list[float], bool]:
    if face.kind == "horizontal":
        prims = [_prim(topo, eid, sign) for loop in face.loops for eid, sign in loop]
        size, where, bounded = feature_size(prims)
        return size * g, [where[0] * g, where[1] * g, float(face.z_range[0])], bounded
    size_um, location = _wall_feature(topo, face, g)
    return size_um, location, False


def _area_and_perimeter(
    topo: Topology, face: TopoFace, g: float
) -> tuple[float, float]:
    points, perimeter = [], 0.0
    for loop in face.loops:
        ring = []
        for eid, sign in loop:
            e = topo.edges[eid]
            v = topo.vertices[e.v0 if sign > 0 else e.v1]
            ring.append((v.x * g, v.y * g, float(v.z)))
        perimeter += sum(
            math.dist(ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring))
        )
        points.append(ring)
    area = (
        abs(face.twice_xy_area) / 2 * g * g
        if face.kind == "horizontal"
        else _polygon_area_3d(points[0])
    )
    return area, perimeter


def _prim(topo: Topology, eid: str, sign: int) -> Prim:
    """A loop edge as an exact primitive in grid units (a segment, or the realised arc)."""
    e = topo.edges[eid]
    # Order the vertex ids along the loop first and read the points from them, so a reversed
    # edge keeps each end point with its own id (the altitude fallback matches them by id).
    ids = (e.v0, e.v1) if sign > 0 else (e.v1, e.v0)
    start, end = topo.vertices[ids[0]], topo.vertices[ids[1]]
    a, b = (start.x, start.y), (end.x, end.y)
    if e.curve is None:
        return Prim(a, b, ids=ids)
    c = e.curve
    ccw = c.ccw if sign > 0 else not c.ccw
    chord = math.hypot(b[0] - a[0], b[1] - a[1])
    theta = 2 * math.asin(min(1.0, chord / (2 * c.radius)))
    t0 = math.atan2(a[1] - c.centre[1], a[0] - c.centre[0])
    return Prim(a, b, c.centre, c.radius, t0, theta if ccw else -theta, ids)


def _wall_feature(
    topo: Topology, face: TopoFace, g: float
) -> tuple[float, list[float]]:
    """A wall is a rectangle-like strip: min(height, distance between its two vertical edges)."""
    columns = {}
    for eid, _ in face.loops[0]:
        e = topo.edges[eid]
        if e.kind == "vertical":
            v = topo.vertices[e.v0]
            columns[(v.x, v.y)] = float(v.z)
    points = list(columns)
    gap = (
        min(
            (math.dist(p, q) for i, p in enumerate(points) for q in points[i + 1 :]),
            default=math.inf,
        )
        * g
    )
    height = float(face.z_range[1] - face.z_range[0])
    middle = [
        sum(p[0] for p in points) / len(points) * g,
        sum(p[1] for p in points) / len(points) * g,
        float(face.z_range[0]) + height / 2,
    ]
    return min(height, gap), middle


def _polygon_area_3d(points: list[tuple[float, float, float]]) -> float:
    sx = sy = sz = 0.0
    for i in range(len(points)):
        x0, y0, z0 = points[i]
        x1, y1, z1 = points[(i + 1) % len(points)]
        sx += y0 * z1 - z0 * y1
        sy += z0 * x1 - x0 * z1
        sz += x0 * y1 - y0 * x1
    return 0.5 * math.sqrt(sx * sx + sy * sy + sz * sz)


def _smallest_angles(model: ConformalModel, top: int, g: float) -> list[Entry]:
    rows = []
    for layer in model.labelled.level_layers:
        arr = layer.arr
        around: dict[int, list[tuple[float, int]]] = {}
        for ei, (a, b) in enumerate(arr.edge_ends):
            pa, pb = arr.vertices[a], arr.vertices[b]
            around.setdefault(a, []).append(
                (math.atan2(pb[1] - pa[1], pb[0] - pa[0]), ei)
            )
            around.setdefault(b, []).append(
                (math.atan2(pa[1] - pb[1], pa[0] - pb[0]), ei)
            )
        for vi, items in around.items():
            if len(items) < 2:
                continue
            items.sort()
            for k, (a0, e0) in enumerate(items):
                a1, e1 = items[(k + 1) % len(items)]
                gap = (a1 - a0) % (2 * math.pi) or 2 * math.pi
                rows.append((math.degrees(gap), layer, vi, e0, e1))
    rows.sort(key=lambda r: (r[0], r[1].plane, r[2], r[3], r[4]))
    return [
        {
            "value_deg": round(deg, 6),
            "plane": layer.plane,
            "vertex": layer.vertex_ids[vi],
            "location_um": [
                round(layer.arr.vertices[vi][0] * g, 4),
                round(layer.arr.vertices[vi][1] * g, 4),
                round(float(layer.z), 4),
            ],
            "edges": [layer.edge_ids[e0], layer.edge_ids[e1]],
            "sources": sorted(
                set(layer.edge_sources[e0]) | set(layer.edge_sources[e1])
            ),
        }
        for deg, layer, vi, e0, e1 in rows[:top]
    ]


def _carrier_ends(
    model: ConformalModel,
) -> dict[str, tuple[tuple[int, int], tuple[int, int]]]:
    catalogue = model.labelled.catalogue
    carriers = {
        c.id: (c.p, c.q) for prism in catalogue.prisms for c in prism.carriers()
    }
    carriers.update(
        {c.id: (c.p, c.q) for family in catalogue.families for c in family.carriers}
    )
    return carriers


def _snap_drift(model: ConformalModel, top: int, g: float) -> Entry:
    carriers = _carrier_ends(model)
    rows = []
    for layer in (*model.labelled.slabs, *model.labelled.level_layers):
        for e, eid, sources in zip(
            layer.arr.edges, layer.edge_ids, layer.edge_sources, strict=True
        ):
            distances = [
                max(
                    segment_point_distance_sq(*carriers[s], e.a),
                    segment_point_distance_sq(*carriers[s], e.b),
                )
                for s in sources
                if s in carriers
            ]
            if distances:
                rows.append(
                    (math.sqrt(float(min(distances))), layer.plane, eid, e, sources)
                )
    rows.sort(key=lambda r: (-r[0], r[1], r[2]))
    bound = math.sqrt(2) / 2
    return {
        "bound_grid_units": round(bound, 6),
        "max_grid_units": round(rows[0][0], 6) if rows else 0.0,
        "edges_over_bound": sum(1 for r in rows if r[0] > bound + 1e-9),
        "edges_measured": len(rows),
        "worst": [
            {
                "id": eid,
                "plane": plane,
                "value_grid_units": round(d, 6),
                "value_um": round(d * g, 6),
                "location_um": [round(e.a[0] * g, 4), round(e.a[1] * g, 4)],
                "sources": list(sources),
            }
            for d, plane, eid, e, sources in rows[:top]
        ],
    }


def _arc_edges(model: ConformalModel, top: int, g: float) -> Entry:
    """Realised arc edges: deviation from the exact carrier circle (grid units), worst first."""
    rows = []
    for layer in (*model.labelled.slabs, *model.labelled.level_layers):
        for e, eid, sources, curve in zip(
            layer.arr.edges,
            layer.edge_ids,
            layer.edge_sources,
            layer.edge_curve,
            strict=True,
        ):
            if curve is not None:
                rows.append((curve.deviation, layer.plane, eid, e, curve, sources))
    rows.sort(key=lambda r: (-r[0], r[1], r[2]))
    return {
        "count": len(rows),
        "max_deviation_grid": round(rows[0][0], 6) if rows else 0.0,
        "mean_deviation_grid": round(sum(r[0] for r in rows) / len(rows), 6)
        if rows
        else 0.0,
        "worst": [
            {
                "id": eid,
                "plane": plane,
                "value_grid_units": round(d, 6),
                "radius_um": round(curve.radius * g, 4),
                "location_um": [round(e.a[0] * g, 4), round(e.a[1] * g, 4)],
                "sources": list(sources),
            }
            for d, plane, eid, e, curve, sources in rows[:top]
        ],
    }
