"""Figures and tables for ``docs/design/sgb/inset-imprint.qmd``.

Run (from anywhere)::

    .venv/bin/python zh-TW/docs/design/sgb/_figures/inset.py

Every geometry figure is computed from the read-only prototype (``experiments/sgb-redesign/prototype``)
by building the pad-in-ground example with Surface-EPR inset bands; nothing is copied from its HTML.
Outputs go to ``docs/design/sgb/figures/`` with the prefix ``inset-``:

* ``inset-erosion-construction``  corner rule on large margins (computed with margins 1.5, 3, 6 um)
* ``inset-bands-pad``             the z = 0 level arrangement with bands (margins 0.1, 0.5, 2 um); html + svg
* ``inset-arcs-vs-polygon``       exact offset arcs against a 512-gon erosion, radial deviation; html + svg
* ``inset-topology-change``       margins 4, 9, 30 um: a neck splits, a band vanishes; html + svg
* ``inset-mask-vs-imprint``       an element cut by a mask against a mesh aligned to the band edge, and the
                                  area error of both as the element size shrinks (2-D gmsh illustration)
* ``inset-routes-section``        conceptual cross-section of Route A (one sheet) and Route B (top and bottom faces)
* ``inset-tables.md``             markdown tables quoted by the page (band areas, KLayout cross-check, Palace check)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import inset_kit as kit  # noqa: E402  (also puts the prototype on sys.path)
from inset_kit import Label, Line, Panel, Poly, Pts  # noqa: E402

import numpy as np  # noqa: E402

from sgbproto.arcs import cycle_polyline, face_exact_area, sample_arc  # noqa: E402
from sgbproto.klayout_check import comparison_rows, eroded_klayout, exact_band_areas, region_polygons  # noqa: E402
from sgbproto.pipeline import build  # noqa: E402
from sgbproto.records import ArcCarrier, LineEdge  # noqa: E402
from sgbproto.scenes import pad_in_ground  # noqa: E402

G = kit.G_UM
MARGINS = (0.1, 0.5, 2.0)  # the running example's three margins (um)
LEVEL_Z0 = 1  # index of the z = 0 level in Route A (levels: -100, 0, 100)

VIEWS = {
    "whole chip": ((-155, 155), (-155, 155)),
    "cutout": ((-85, 85), (-65, 65)),
    "pad": ((-55, 55), (-35, 35)),
    "reflex notch": ((-16, -4), (-21, -9)),
    "rounded corner": ((-52, -40), (-32, -20)),
    "hole": ((-35, -15), (-5, 15)),
}


# --------------------------------------------------------------------------- geometry helpers


def primitive_polyline(prim, n: int = 80) -> list[tuple[float, float]]:
    """A support-region primitive (line or exact arc) as a polyline in um."""
    if isinstance(prim, LineEdge):
        return [(prim.p[0] * G, prim.p[1] * G), (prim.q[0] * G, prim.q[1] * G)]
    t0, sweep = prim.sweep()
    cx, cy, r = float(prim.centre[0]), float(prim.centre[1]), float(prim.radius)
    return [((cx + r * math.cos(t0 + sweep * i / n)) * G, (cy + r * math.sin(t0 + sweep * i / n)) * G) for i in range(n + 1)]


def region_boundary_lines(region, width: float = 2.2) -> list[Line]:
    """The support region's own boundary, which is the only distance origin."""
    out = []
    for ri, ring in enumerate(region.rings):
        pts = [p for prim in ring for p in primitive_polyline(prim)]
        out.append(Line([p[0] for p in pts], [p[1] for p in pts], kit.INK, width, name="boundary of R (distance origin)"))
    return out


def layer_edge_polylines(layer, predicate, step: float = 150.0) -> list[list[tuple[float, float]]]:
    out = []
    for e, curve, src in zip(layer.arr.edges, layer.edge_curve, layer.edge_sources):
        if not predicate(src):
            continue
        if curve is None:
            pts = [e.a, e.b]
        else:
            pts = sample_arc(curve, e.a, e.b, max(3, int(math.hypot(e.b[0] - e.a[0], e.b[1] - e.a[1]) / step) + 2))
        out.append([(x * G, y * G) for x, y in pts])
    return out


def clip_runs(points, xr, yr, pad: float = 1.0):
    """Contiguous runs of a polyline that lie inside a view window (plus ``pad`` um); keeps figure files small."""
    runs, run = [], []
    for x, y in points:
        if xr[0] - pad <= x <= xr[1] + pad and yr[0] - pad <= y <= yr[1] + pad:
            run.append((x, y))
        elif run:
            runs.append(run)
            run = []
    if run:
        runs.append(run)
    return [r for r in runs if len(r) > 1]


def face_rings_um(layer, fi: int, step: float = 150.0):
    arr = layer.arr
    return [[(x * G, y * G) for x, y in cycle_polyline(arr, layer.edge_curve, c, step)] for c in arr.face_cycles(fi)]


def face_style(label, group: str) -> tuple[str, str]:
    """(fill colour, legend text) of a level cell of the running example."""
    if label.sheet is not None and label.sheet.kind == "port":
        return kit.PORT, "port p1"
    if label.sheet is not None and label.bands:
        return kit.PAD_BANDS[label.bands[0][1]], group
    if label.sheet is not None:
        return kit.GROUND, group
    if label.bands:
        return kit.GAP_BANDS[label.bands[0][1]], group
    return kit.NEUTRAL, group


def level_polys(result, level_index: int = LEVEL_Z0) -> list[Poly]:
    layer = result.level_layers[level_index]
    topo = result.topology
    polys = []
    for fi in range(1, len(layer.arr.faces)):
        fid = layer.face_ids[fi]
        group = topo.faces[fid].group if fid in topo.faces else "(no face)"
        fill, _ = face_style(layer.labels[fi], group)
        polys.append(Poly(face_rings_um(layer, fi), fill=fill, edge="rgba(0,0,0,0.6)", edge_width=0.4, hover=f"{group}<br>{fid}"))
    return polys


# --------------------------------------------------------------------------- 1. erosion construction


def figure_construction() -> None:
    """Corner rule on margins large enough to see (1.5, 3 and 6 um on the 5 um rounded corner)."""
    margins = (1.5, 3.0, 6.0)
    spec = pad_in_ground("A", bands_um=margins, band_regions=("pad_top",), with_port=False)
    result = build(spec)
    layer = result.level_layers[LEVEL_Z0]
    region = spec.families[0].support
    colours = {1500: "#1b7837", 3000: "#e08214", 6000: "#762a83"}
    names = {1500: "m = 1.5 um", 3000: "m = 3 um", 6000: "m = 6 um"}

    def curves(m: int) -> list[Line]:
        return [Line([p[0] for p in poly], [p[1] for p in poly], colours[m], 2.0, name=f"offset curve, {names[m]}")
                for poly in layer_edge_polylines(layer, lambda src, m=m: any(s.startswith("band_pad_top/") and f"@m{m}." in s for s in src))]

    def circle(cx, cy, r, colour, name="") -> Line:
        t = np.linspace(0, 2 * math.pi, 181)
        return Line(list(cx + r * np.cos(t)), list(cy + r * np.sin(t)), colour, 0.9, "dot", name=name)

    panels = []
    boundary = region_boundary_lines(region)
    # reflex corner (the notch vertex): arcs of radius m about the vertex
    items = list(boundary) + [c for m in colours for c in curves(m)]
    items += [circle(-10, -15, m / 1000, colours[m], "disc D(m) about the vertex") for m in colours]
    items += [Pts([-10], [-15], kit.INK, 7, name="reflex vertex", symbol="x")]
    panels.append(Panel("reflex corner: offset lines leave a gap, bridged by an arc of radius m", (-19, -1), (-24, -6), items=items))
    # convex corner (chamfer): the offset lines cross and the corner stays sharp
    items = list(boundary) + [c for m in colours for c in curves(m)]
    items += [Pts([50, 38], [18, 30], kit.INK, 7, name="convex vertices", symbol="x")]
    panels.append(Panel("convex corner (chamfer): offset lines cross, corner stays sharp", (30, 53), (6, 33), items=items))
    # convex arc: rho -> rho - m, vanishes for m >= rho
    items = list(boundary) + [c for m in colours for c in curves(m)]
    items += [Pts([-45], [-25], kit.INK, 6, name="arc centre", symbol="x")]
    panels.append(Panel("convex arc: radius rho becomes rho - m, and vanishes when m >= rho", (-53, -33), (-33, -13), items=items))
    # hole
    items = list(boundary) + [c for m in colours for c in curves(m)]
    items += [Pts([-25], [5], kit.INK, 6, name="hole centre", symbol="x")]
    panels.append(Panel("hole of radius r = 8 um: concentric circle of radius r + m", (-42, -8), (-12, 22), items=items))
    kit.write_both(panels, "inset-erosion-construction", ncols=2, svg_size=(10, 8.4), html_size=(1100, 900),
                   title="Boundary of the eroded region R minus D(m) for m = 1.5, 3, 6 um (computed, prototype inset generator)", shared_legend=True)


# --------------------------------------------------------------------------- 2. bands on the running example


def figure_bands(result_a) -> None:
    layer = result_a.level_layers[LEVEL_Z0]
    polys = level_polys(result_a)
    legend = [
        Pts([-1e4], [-1e4], kit.PAD_BANDS[0], 8, name="SHEET::pad::band=0  (< 0.1 um from the edge)", symbol="square"),
        Pts([-1e4], [-1e4], kit.PAD_BANDS[1], 8, name="band 1  (0.1-0.5 um)", symbol="square"),
        Pts([-1e4], [-1e4], kit.PAD_BANDS[2], 8, name="band 2  (0.5-2 um)", symbol="square"),
        Pts([-1e4], [-1e4], kit.PAD_BANDS[3], 8, name="band 3  (>= 2 um)", symbol="square"),
        Pts([-1e4], [-1e4], kit.GAP_BANDS[0], 8, name="SA::substrate|vacuum::band=0..3 (gap ring, orange ramp)", symbol="square"),
        Pts([-1e4], [-1e4], kit.PORT, 8, name="PORT::p1", symbol="square"),
        Pts([-1e4], [-1e4], kit.GROUND, 8, name="PEC::Ground", symbol="square"),
    ]
    one = Panel("z = 0 level arrangement of pad_in_ground_A, coloured by label and band", VIEWS["pad"][0], VIEWS["pad"][1], xlabel="x (um)", ylabel="y (um)",
                items=polys + legend)
    kit.render_html([one], kit.FIGURES / "inset-bands-pad.html", size=(1000, 720), buttons=VIEWS,
                    title="Pad-in-ground, Route A: cells of the z = 0 level coloured by physical group (margins 0.1, 0.5, 2 um)")
    static = []
    for name in ("pad", "reflex notch", "hole", "rounded corner"):
        xr, yr = VIEWS[name]
        static.append(Panel(f"{name}", xr, yr, xlabel="x (um)", ylabel="y (um)", items=polys + legend[:6]))
    kit.render_svg(static, kit.FIGURES / "inset-bands-pad.svg", ncols=2, size=(10, 8),
                   suptitle="z = 0 level of pad_in_ground_A: bands of the pad sheet (blue) and of the gap ring (orange); port red", shared_legend=True)
    print("wrote inset-bands-pad")


# --------------------------------------------------------------------------- 3. exact arcs vs polygon


def klayout_rings(region, margin_um: float) -> list[list[tuple[float, float]]]:
    eroded = eroded_klayout(region_polygons(region), round(margin_um / G))
    rings = []
    for poly in eroded.each():
        rings.append([(p.x * G, p.y * G) for p in poly.each_point_hull()])
        for h in range(poly.holes()):
            rings.append([(p.x * G, p.y * G) for p in poly.each_point_hole(h)])
    return rings


def figure_arcs(spec_a, result_a) -> None:
    layer = result_a.level_layers[LEVEL_Z0]
    region = spec_a.families[0].support
    exact_curves = layer_edge_polylines(layer, lambda src: any(s.startswith("band_pad_top/") for s in src))
    rings = {m: klayout_rings(region, m) for m in MARGINS}
    features = [
        ("reflex notch corner", ((-12.5, -7.5), (-17.5, -12.5)), (-10.0, -15.0), lambda m: m),
        ("rounded corner (rho = 5 um)", ((-51, -39), (-31, -19)), (-45.0, -25.0), lambda m: 5.0 - m),
        ("hole (r = 8 um)", ((-35, -15), (-5, 15)), (-25.0, 5.0), lambda m: 8.0 + m),
    ]
    top, bottom = [], []
    for title, (xr, yr), centre, radius_of in features:
        items = [Line([p[0] for p in run], [p[1] for p in run], kit.EXACT, 2.2, name="exact carriers (arcs realised)")
                 for poly in exact_curves for run in clip_runs(poly, xr, yr)]
        for m in MARGINS:
            for ring in rings[m]:
                for run in clip_runs(ring + [ring[0]], xr, yr):
                    items.append(Line([p[0] for p in run], [p[1] for p in run], kit.OTHER, 1.0, "dot", name="KLayout erosion by a 512-gon"))
        top.append(Panel(title, xr, yr, items=items, xlabel="x (um)", ylabel="y (um)"))
        dev_items = []
        for m in MARGINS:
            r = radius_of(m)
            if r <= 0:
                continue

            def deviation(points):
                data = [(math.degrees(math.atan2(y - centre[1], x - centre[0])), (math.hypot(x - centre[0], y - centre[1]) - r) * 1000)
                        for x, y in points if abs(math.hypot(x - centre[0], y - centre[1]) - r) < 0.02]
                return data[:: max(1, len(data) // 300)]

            ex = deviation([p for poly in exact_curves for p in poly])
            kl = deviation([p for ring in rings[m] for p in ring])
            dev_items.append(Pts([d[0] for d in ex], [d[1] for d in ex], kit.EXACT, 3, name="exact carriers", hover=f"m = {m} um"))
            dev_items.append(Pts([d[0] for d in kl], [d[1] for d in kl], kit.OTHER, 3, name="512-gon (integer-rounded vertices)", hover=f"m = {m} um"))
        bottom.append(Panel("radial deviation from the exact radius (nm)", None, None, equal=False, xlabel="angle about the centre (deg)",
                            ylabel="deviation (nm)", items=dev_items))
    kit.write_both(top + bottom, "inset-arcs-vs-polygon", ncols=3, svg_size=(14, 8), html_size=(1300, 800),
                   title="Exact offset arcs against a 512-gon erosion, margins 0.1, 0.5, 2 um (computed)", shared_legend=True)


# --------------------------------------------------------------------------- 4. topology change


def figure_topology_change() -> dict:
    margins = (4.0, 9.0, 30.0)
    spec = pad_in_ground("A", bands_um=margins, band_regions=("pad_top",), with_port=False)
    result = build(spec)
    layer = result.level_layers[LEVEL_Z0]
    palette = {0: "#08306b", 1: "#4292c6", 2: "#fdae61", 3: "#d9d9d9"}
    polys = []
    for fi in range(1, len(layer.arr.faces)):
        lab = layer.labels[fi]
        if lab.sheet is None or not lab.bands:
            continue
        k = lab.bands[0][1]
        area = face_exact_area(layer.arr, layer.edge_curve, fi) * 1e-6
        polys.append(Poly(face_rings_um(layer, fi), palette[k], "rgba(0,0,0,0.6)", 0.5, hover=f"band {k}: {area:.2f} um^2"))
    legend = [Pts([-1e4], [-1e4], palette[0], 8, name="B0: closer than 4 um to the edge", symbol="square"),
              Pts([-1e4], [-1e4], palette[1], 8, name="B1: 4-9 um", symbol="square"),
              Pts([-1e4], [-1e4], palette[2], 8, name="B2: 9-30 um (three separate faces)", symbol="square"),
              Pts([-1e4], [-1e4], palette[3], 8, name="B3: at least 30 um (empty)", symbol="square")]
    panel = Panel("pad top, margins 4, 9 and 30 um", (-55, 55), (-35, 35), xlabel="x (um)", ylabel="y (um)", items=polys + legend)
    kit.write_both([panel], "inset-topology-change", svg_size=(9, 6.2), html_size=(1000, 650), shared_legend=True,
                   title="Erosion splits the pad into three islands at 9 um; nothing is 30 um from the edge")
    return json.loads((kit.PROTOTYPE_OUTPUTS / "inset" / "topology_change.json").read_text())


# --------------------------------------------------------------------------- 5. mask vs imprint (2-D gmsh illustration)

RADIUS = 0.5  # mask boundary: circle of this radius; masked region = the square minus the disc


def disc_mesh(h: float, aligned: bool):
    """Triangles of the square [-1, 1]^2 with element size ``h``; ``aligned`` makes the circle a mesh boundary."""
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("mask")
        geo = gmsh.model.geo
        corners = [geo.addPoint(x, y, 0) for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        outer = geo.addCurveLoop([geo.addLine(corners[i], corners[(i + 1) % 4]) for i in range(4)])
        inside_tag = None
        if aligned:
            c = geo.addPoint(0, 0, 0)
            q = [geo.addPoint(RADIUS * math.cos(a), RADIUS * math.sin(a), 0) for a in (0, math.pi / 2, math.pi, 3 * math.pi / 2)]
            arcs = [geo.addCircleArc(q[i], c, q[(i + 1) % 4]) for i in range(4)]
            ring = geo.addCurveLoop(arcs)
            outside_tag = geo.addPlaneSurface([outer, ring])
            inside_tag = geo.addPlaneSurface([ring])
        else:
            outside_tag = geo.addPlaneSurface([outer])
        geo.synchronize()
        for key, value in (("Mesh.MeshSizeMax", h), ("Mesh.MeshSizeMin", 0.0), ("Mesh.MeshSizeFromPoints", 0), ("Mesh.MeshSizeFromCurvature", 0),
                           ("Mesh.MeshSizeExtendFromBoundary", 0), ("Mesh.Algorithm", 6)):
            gmsh.option.setNumber(key, value)
        gmsh.model.mesh.generate(2)
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        index = {int(t): i for i, t in enumerate(node_tags)}
        xy = np.asarray(coords).reshape(-1, 3)[:, :2]

        def triangles(tag):
            _, _, nodes = gmsh.model.mesh.getElements(2, tag)
            return np.array([[index[int(n)] for n in tri] for tri in np.asarray(nodes[0]).reshape(-1, 3)]) if len(nodes) else np.zeros((0, 3), int)

        return xy, triangles(outside_tag), (triangles(inside_tag) if inside_tag else None)
    finally:
        gmsh.finalize()


def tri_areas(xy, tri):
    a, b, c = xy[tri[:, 0]], xy[tri[:, 1]], xy[tri[:, 2]]
    return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


SHIFTS = [(0.0, 0.0), (0.011, 0.007), (-0.017, 0.013), (0.023, -0.019), (-0.007, -0.029), (0.031, 0.002), (-0.013, 0.021), (0.005, -0.011)]


def sampled_mask_area(xy, tri, area, centre) -> tuple[float, float]:
    """Masked area (square minus disc about ``centre``) when the mask is sampled at 1 and at 3 points per element."""
    cx, cy = centre
    mid = (xy[tri[:, [0, 1]]].mean(axis=1), xy[tri[:, [1, 2]]].mean(axis=1), xy[tri[:, [2, 0]]].mean(axis=1))
    centroid = xy[tri].mean(axis=1)
    one = (np.hypot(centroid[:, 0] - cx, centroid[:, 1] - cy) >= RADIUS).astype(float)
    three = sum((np.hypot(m[:, 0] - cx, m[:, 1] - cy) >= RADIUS).astype(float) for m in mid) / 3.0
    return float((area * one).sum()), float((area * three).sum())


def mask_metrics(h: float) -> dict:
    """Masked area (square minus disc) from a mesh the mask cuts (several circle positions), and from a mesh aligned to the contour."""
    exact = 4.0 - math.pi * RADIUS**2
    xy, tri, _ = disc_mesh(h, aligned=False)
    area = tri_areas(xy, tri)
    radial = np.hypot(xy[:, 0], xy[:, 1])
    side = radial[tri] - RADIUS
    cut = int(np.sum((side.min(axis=1) < -1e-9) & (side.max(axis=1) > 1e-9)))
    one, three = zip(*(sampled_mask_area(xy, tri, area, shift) for shift in SHIFTS))
    err1 = [abs(v - exact) / exact for v in one]
    err3 = [abs(v - exact) / exact for v in three]
    xy2, tri_out, tri_in = disc_mesh(h, aligned=True)
    radial2 = np.hypot(xy2[:, 0], xy2[:, 1])
    side2 = radial2[tri_out] - RADIUS
    side_in = radial2[tri_in] - RADIUS
    crossing = int(np.sum((side2.min(axis=1) < -1e-6) & (side2.max(axis=1) > 1e-6))) + int(np.sum((side_in.max(axis=1) > 1e-6)))
    aligned_area = float(tri_areas(xy2, tri_out).sum())
    return {
        "h": h, "exact": exact,
        "cut_triangles": cut, "triangles_unaligned": int(len(tri)),
        "err_one_point_median": float(np.median(err1)), "err_one_point_max": max(err1), "err_one_point_min": min(err1),
        "err_three_point_median": float(np.median(err3)), "err_three_point_max": max(err3), "err_three_point_min": min(err3),
        "crossing_aligned": crossing, "triangles_aligned": int(len(tri_out) + len(tri_in)),
        "err_aligned": abs(aligned_area - exact) / exact,
    }


def figure_mask_vs_imprint() -> list[dict]:
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    h = 0.16
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.9), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    fig.patch.set_facecolor("white")
    xy, tri, _ = disc_mesh(h, aligned=False)
    radial = np.hypot(xy[:, 0], xy[:, 1]) - RADIUS
    side = radial[tri]
    is_cut = (side.min(axis=1) < 0) & (side.max(axis=1) > 0)
    centroid = xy[tri].mean(axis=1)
    kept = np.hypot(centroid[:, 0], centroid[:, 1]) >= RADIUS
    colours = np.where(is_cut, kit.ACCENT, np.where(kept, kit.PAD_BANDS[3], "#ffffff"))
    ax = axes[0]
    ax.add_collection(PolyCollection(xy[tri], facecolors=colours, edgecolors="#555", linewidths=0.4))
    t = np.linspace(0, 2 * math.pi, 400)
    ax.plot(RADIUS * np.cos(t), RADIUS * np.sin(t), color=kit.INK, lw=1.6)
    ax.set_title(f"mask evaluated on a mesh that does not know it:\n{int(is_cut.sum())} of {len(tri)} triangles straddle the contour (orange)", fontsize=9)
    xy2, tri_out, tri_in = disc_mesh(h, aligned=True)
    ax = axes[1]
    ax.add_collection(PolyCollection(xy2[tri_out], facecolors=kit.PAD_BANDS[3], edgecolors="#555", linewidths=0.4))
    ax.add_collection(PolyCollection(xy2[tri_in], facecolors="#ffffff", edgecolors="#555", linewidths=0.4))
    ax.plot(RADIUS * np.cos(t), RADIUS * np.sin(t), color=kit.INK, lw=1.6)
    ax.set_title(f"contour is a mesh boundary (an imprinted band edge):\n0 of {len(tri_out) + len(tri_in)} triangles straddle it", fontsize=9)
    for ax in axes[:2]:
        ax.set_xlim(-1.02, 1.02); ax.set_ylim(-1.02, 1.02); ax.set_aspect("equal"); ax.tick_params(labelsize=7)
    rows = [mask_metrics(x) for x in (0.4, 0.2, 0.1, 0.05, 0.025, 0.0125)]
    ax = axes[2]
    hs = np.array([r["h"] for r in rows])
    for key, colour, marker, label in (("one_point", kit.ACCENT, "o", "mask sampled at 1 point per element"),
                                       ("three_point", kit.OTHER, "s", "mask sampled at 3 points per element")):
        ax.loglog(hs, [r[f"err_{key}_median"] for r in rows], marker + "-", color=colour, label=label + " (median of 8 contour positions)")
        ax.fill_between(hs, [r[f"err_{key}_min"] for r in rows], [r[f"err_{key}_max"] for r in rows], color=colour, alpha=0.15, linewidth=0)
    ax.loglog(hs, [r["err_aligned"] for r in rows], "^-", color=kit.EXACT, lw=2, label="contour is a mesh boundary (sum over band-group faces)")
    ax.loglog(hs, 0.03 * (hs / 0.4) ** 2, color="#777", ls=":", lw=1, label="slope 2")
    ax.loglog(hs, 0.03 * (hs / 0.4), color="#777", ls="--", lw=1, label="slope 1")
    ax.set_xticks(hs); ax.set_xticklabels([f"{x:g}" for x in hs], fontsize=7); ax.minorticks_off()
    ax.set_xlabel("element size h", fontsize=8); ax.set_ylabel("relative error of the masked area", fontsize=8)
    ax.set_title("masked area of a surface against a circular contour\n(first-order elements; exact value 4 - pi r^2, r = 0.5)", fontsize=9)
    ax.legend(fontsize=6.5, loc="lower right"); ax.grid(alpha=0.3, which="major"); ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    kit.save_svg(fig, "inset-mask-vs-imprint")
    (kit.FIGURES / "inset-mask-vs-imprint.json").write_text(json.dumps(rows, indent=1))
    print("wrote inset-mask-vs-imprint")
    return rows


# --------------------------------------------------------------------------- 6. conceptual cross-section of the routes


def figure_routes_section() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_xlim(0, 10); ax.set_ylim(-2.2, 3.4); ax.axis("off")
        ax.add_patch(Rectangle((0, -2), 10, 2, facecolor="#e5e5e5", edgecolor="none"))
        ax.text(0.15, -1.2, "substrate", fontsize=8)
        ax.text(0.15, 2.7, "vacuum", fontsize=8)
    # Route A: a sheet; bands lie on the sheet; one group serves the MA and the MS row
    ax = axes[0]
    ax.set_title("Route A: zero-thickness sheet", fontsize=9)
    edges = [2, 2.5, 3.5, 6.5, 7.5, 8]
    for i in range(len(edges) - 1):
        k = min(i, 4 - i)
        ax.plot([edges[i], edges[i + 1]], [0, 0], color=kit.PAD_BANDS[k], lw=7, solid_capstyle="butt")
    ax.text(5, 0.65, "one face per band: pad::band=k", fontsize=8, ha="center")
    ax.text(5, 0.2, "(the sheet is metal-vacuum and metal-substrate at once)", fontsize=7, ha="center", color="#555")
    ax.annotate("MA row and MS row both\nread these band groups", (2.2, 0), (0.3, 1.5), fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.8))
    # Route B: a finite prism; the top face and bottom face each carry bands
    ax = axes[1]
    ax.set_title("Route B: finite metal (excluded volume), top and bottom faces banded separately", fontsize=9)
    ax.add_patch(Rectangle((2, 0), 6, 0.6, facecolor="white", edgecolor=kit.INK, lw=0.8, hatch="//"))
    for i in range(len(edges) - 1):
        k = min(i, 4 - i)
        ax.plot([edges[i], edges[i + 1]], [0.6, 0.6], color=kit.PAD_BANDS[k], lw=6, solid_capstyle="butt")
        ax.plot([edges[i], edges[i + 1]], [0.0, 0.0], color=kit.PAD_BANDS[k], lw=6, solid_capstyle="butt")
    ax.text(5, 1.2, "top face: MA::pad::band=k", fontsize=8, ha="center")
    ax.text(5, -0.55, "bottom face: MS::pad::band=k", fontsize=8, ha="center")
    ax.text(8.25, 0.3, "side walls:\nnot banded in v1", fontsize=7, ha="left", color="#555")
    fig.tight_layout()
    kit.save_svg(fig, "inset-routes-section")
    print("wrote inset-routes-section")


# --------------------------------------------------------------------------- tables for the page


def build_tables(result_a, spec_a, topology_change: dict, mask_rows: list[dict]) -> None:
    out = ["## Band areas (computed by inset.py from the Route A build, margins 0.1, 0.5, 2 um)", "",
           "| support region | band | faces | area (um^2) | R_m = bands >= k (um^2) |", "|---|---|---|---|---|"]
    layer = result_a.level_layers[LEVEL_Z0]
    for fam_id, label in (("band_pad_top", "pad top"), ("band_gap", "gap ring")):
        per_band: dict[int, list[float]] = {}
        for fi in range(1, len(layer.arr.faces)):
            for fam, k in layer.labels[fi].bands:
                if fam == fam_id:
                    per_band.setdefault(k, []).append(face_exact_area(layer.arr, layer.edge_curve, fi) * 1e-6)
        total = {k: sum(v) for k, v in per_band.items()}
        for k in sorted(total):
            out.append(f"| {label} | B{k} | {len(per_band[k])} | {total[k]:.4f} | {sum(a for b, a in total.items() if b >= k):.4f} |")
    out += ["", "## KLayout cross-check (examples/run_inset_bands.py, outputs/inset/klayout_comparison.json)", "",
            "| region | margin (um) | exact area (um^2) | KLayout area (um^2) | difference (um^2) | relative | oracle rounding bound (um^2) |", "|---|---|---|---|---|---|---|"]
    for r in json.loads((kit.PROTOTYPE_OUTPUTS / "inset" / "klayout_comparison.json").read_text()):
        out.append(f"| {r['region']} | {r['margin_units'] / 1000:g} | {r['exact_um2']:.4f} | {r['klayout_um2']:.4f} | {r['difference_um2']:+.5f} | "
                   f"{r['relative']:+.1e} | {r['oracle_rounding_bound_um2']:.3f} |")
    out += ["", "## Palace-facing check (outputs/inset/palace_check.json)", "",
            "| band group | faces | triangles | tets on each side | all triangles shared by tets | centroids inside band |", "|---|---|---|---|---|---|"]
    for name, row in json.loads((kit.PROTOTYPE_OUTPUTS / "inset" / "palace_check.json").read_text()).items():
        sides = sorted(row["tets_per_side"].values())
        out.append(f"| `{name}` | {row['faces']} | {row['triangles']} | {sides[0]} + {sides[1]} | {row['all_shared_by_tets']} | {row['centroids_in_band']} |")
    out += ["", "## Mask against imprint (inset-mask-vs-imprint.json)", "",
            "| h | triangles | straddling the contour | error, 1-point mask (median / max) | error, 3-point mask (median / max) | triangles (aligned) | straddling (aligned) | error (aligned) |",
            "|---|---|---|---|---|---|---|---|"]
    for r in mask_rows:
        out.append(f"| {r['h']} | {r['triangles_unaligned']} | {r['cut_triangles']} | {r['err_one_point_median']:.1e} / {r['err_one_point_max']:.1e} | {r['err_three_point_median']:.1e} / {r['err_three_point_max']:.1e} | "
                   f"{r['triangles_aligned']} | {r['crossing_aligned']} | {r['err_aligned']:.2e} |")
    # Route B: the stored example now bands the top face (MA) and the bottom face (MS)
    result_b = build(pad_in_ground("B", bands_um=MARGINS))
    groups = sorted(n for n in result_b.topology.groups if "::band=" in n and n.startswith(("MA::pad", "MS::pad")))
    out += ["", "## Route B (top and bottom faces banded)", "",
            "groups: " + ", ".join(groups), f"validation passed: {not result_b.report.failures}",
            f"counts: {len(result_b.topology.faces)} faces, {len(result_b.topology.edges)} edges, {len(result_b.topology.vertices)} vertices, {len(result_b.topology.groups)} groups",
            "unions: " + json.dumps(result_b.topology.unions["PEC::pad"]),
            "surface_epr_rows (Route A): " + json.dumps(result_a.topology.surface_epr_rows)]
    port = json.loads((kit.PROTOTYPE_OUTPUTS / "inset" / "port_invariance.json").read_text())
    out += ["", "## Port invariance (outputs/inset/port_invariance.json)", "", json.dumps(port)]
    out += ["", "## Topology change (outputs/inset/topology_change.json)", "", json.dumps(topology_change)]
    # sagitta of a 128-gon circle erosion for the three margins (formula m (1 - cos(pi/128)))
    out += ["", "## Sagitta of a 128-gon disc, m (1 - cos(pi / 128)), in nm", "",
            "| margin (um) | sagitta (nm) |", "|---|---|"] + [f"| {m} | {m * 1000 * (1 - math.cos(math.pi / 128)):.3f} |" for m in MARGINS]
    (kit.FIGURES / "inset-tables.md").write_text("\n".join(out) + "\n")
    print("wrote inset-tables.md")


def main() -> None:
    kit.ensure_plotly_js()
    spec_a = pad_in_ground("A", bands_um=MARGINS)
    result_a = build(spec_a)
    figure_construction()
    figure_bands(result_a)
    figure_arcs(spec_a, result_a)
    topology_change = figure_topology_change()
    mask_rows = figure_mask_vs_imprint()
    figure_routes_section()
    build_tables(result_a, spec_a, topology_change, mask_rows)


if __name__ == "__main__":
    main()
