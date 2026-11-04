"""Figures for the SGB design pages (computed from the prototype; phase 2).

Writes to ``outputs/figures/``: a self-contained Plotly HTML (``include_plotlyjs=True``) and a static SVG
(matplotlib; kaleido is not installed, so no Plotly PNG) for each of

(a) ``fig_level_arrangement_bands``   z = 0 level arrangement of pad_in_ground_A coloured by label with bands;
(b) ``fig_exact_arcs_vs_512gon``      zooms on the reflex notch corner, rounded corner and hole: exact offset arcs vs
                                       the KLayout 512-gon erosion, plus their radial deviation in nm;
(c) ``fig_dissolution_before_after``  the same level before and after dissolution (witness chords -> arc edges);
(d) ``fig_exploded_volumes_B``        3-D exploded view of pad_in_ground_B volumes and walls.
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plotly.subplots import make_subplots

from _common import OUTPUTS

from sgbproto.arcs import cycle_polyline, sample_arc
from sgbproto.emit import surface_triangles
from sgbproto.klayout_check import eroded_klayout, region_polygons
from sgbproto.pipeline import build
from sgbproto.scenes import pad_in_ground

FIG = OUTPUTS / "figures"
G = 0.001  # um per grid unit
MARGINS = (0.1, 0.5, 2.0)
BAND_COLOURS = {
    "pad": ["#08306b", "#2171b5", "#6baed6", "#c6dbef"],  # band 0 (edge) darkest
    "gap": ["#7f2704", "#d94801", "#fd8d3c", "#fdd0a2"],
}


def face_colour(label) -> tuple[str, str]:
    """(colour, text) of a level-arrangement face label."""
    if label.sheet is not None and label.sheet.kind == "port":
        return "#d62728", "port p1"
    if label.sheet is not None and label.bands:
        return BAND_COLOURS["pad"][label.bands[0][1]], f"SHEET::pad band {label.bands[0][1]}"
    if label.sheet is not None:
        return "#969696", f"PEC::{label.sheet.name}"
    if label.bands:
        return BAND_COLOURS["gap"][label.bands[0][1]], f"SA gap band {label.bands[0][1]}"
    return "#e8e8e8", "substrate|vacuum (no band)"


def rings_um(layer, fi, step=20.0):
    arr = layer.arr
    return [[(x * G, y * G) for x, y in cycle_polyline(arr, layer.edge_curve, c, step)] for c in arr.face_cycles(fi)]


def svg_path(rings) -> str:
    return " ".join("M " + " L ".join(f"{x:.4f},{y:.4f}" for x, y in ring) + " Z" for ring in rings)


def mpl_path(rings) -> MplPath:
    verts, codes = [], []
    for ring in rings:
        verts += list(ring) + [ring[0]]
        codes += [MplPath.MOVETO] + [MplPath.LINETO] * (len(ring) - 1) + [MplPath.CLOSEPOLY]
    return MplPath(verts, codes)


def save(fig: go.Figure, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.write_html(FIG / f"{name}.html", include_plotlyjs=True)


def save_svg(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


VIEWS = {
    "whole chip": ((-155, 155), (-155, 155)),
    "cutout": ((-85, 85), (-65, 65)),
    "pad": ((-55, 55), (-35, 35)),
    "reflex notch corner": ((-12, -8), (-17, -13)),
    "rounded corner": ((-52, -40), (-32, -20)),
    "hole": ((-35, -15), (-5, 15)),
}


def figure_a(result) -> None:
    layer = result.level_layers[1]
    fig = go.Figure()
    shapes, hover_x, hover_y, hover_t = [], [], [], []
    for fi in range(1, len(layer.arr.faces)):
        colour, text = face_colour(layer.labels[fi])
        rings = rings_um(layer, fi)
        shapes.append(dict(type="path", path=svg_path(rings), fillcolor=colour, fillrule="evenodd",
                           line=dict(color="rgba(0,0,0,0.55)", width=0.4), layer="below"))
        x, y = rings[0][0]
        hover_x.append(x); hover_y.append(y); hover_t.append(f"{text}<br>{layer.face_ids[fi]}")
    fig.add_trace(go.Scatter(x=hover_x, y=hover_y, mode="markers", marker=dict(size=4, color="black"), text=hover_t,
                             hoverinfo="text", name="faces"))
    fig.update_layout(shapes=shapes, title="z = 0 level arrangement of pad_in_ground_A by label and inset band "
                      "(blue: SHEET::pad bands 0-3, orange: SA gap bands 0-3, red: port, grey: ground)",
                      xaxis=dict(range=VIEWS["pad"][0], title="x (um)", constrain="domain"),
                      yaxis=dict(range=VIEWS["pad"][1], title="y (um)", scaleanchor="x"), width=1000, height=720,
                      updatemenus=[dict(type="buttons", direction="right", x=0, y=1.12,
                                        buttons=[dict(label=k, method="relayout", args=[{"xaxis.range": list(v[0]), "yaxis.range": list(v[1])}])
                                                 for k, v in VIEWS.items()])])
    save(fig, "fig_level_arrangement_bands")
    for suffix, view in (("pad", VIEWS["pad"]), ("notch", VIEWS["reflex notch corner"])):
        f, ax = plt.subplots(figsize=(9, 6))
        for fi in range(1, len(layer.arr.faces)):
            colour, _ = face_colour(layer.labels[fi])
            ax.add_patch(PathPatch(mpl_path(rings_um(layer, fi)), facecolor=colour, edgecolor="#222", lw=0.4))
        ax.set_xlim(*view[0]); ax.set_ylim(*view[1]); ax.set_aspect("equal")
        ax.set_title(f"pad_in_ground_A, z = 0 level arrangement by label and band ({suffix} view)")
        ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
        save_svg(f, f"fig_level_arrangement_bands_{suffix}")


def edge_polylines(layer, predicate, step=10.0):
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


def klayout_boundaries(region, margin):
    eroded = eroded_klayout(region_polygons(region), margin)
    rings = []
    for poly in eroded.each():
        rings.append([(p.x * G, p.y * G) for p in poly.each_point_hull()])
        for h in range(poly.holes()):
            rings.append([(p.x * G, p.y * G) for p in poly.each_point_hole(h)])
    return rings


def figure_b(spec, result) -> None:
    layer = result.level_layers[1]
    fam = spec.families[0]
    features = [
        ("reflex notch corner (-10, -15)", ((-12.5, -7.5), (-17.5, -12.5)), (-10.0, -15.0), lambda m: m),
        ("rounded corner (r = 5 um)", ((-51, -39), (-31, -19)), (-45.0, -25.0), lambda m: 5.0 - m),
        ("hole (r = 8 um)", ((-35, -15), (-5, 15)), (-25.0, 5.0), lambda m: 8.0 + m),
    ]
    band_edges = edge_polylines(layer, lambda src: any(s.startswith("band_pad_top/") for s in src))
    kl = {m: klayout_boundaries(fam.support, round(m * 1000)) for m in MARGINS}
    fig = make_subplots(rows=2, cols=3, subplot_titles=[f[0] for f in features] + ["radial deviation (nm)"] * 3,
                        vertical_spacing=0.14)
    fmatplot, axes = plt.subplots(2, 3, figsize=(15, 8))
    for c, (title, (xr, yr), centre, radius_of) in enumerate(features, start=1):
        for poly in band_edges:
            fig.add_trace(go.Scatter(x=[p[0] for p in poly], y=[p[1] for p in poly], mode="lines", line=dict(color="#1f77b4", width=2),
                                     name="exact carriers (arcs realised)", legendgroup="exact", showlegend=c == 1 and poly is band_edges[0]), row=1, col=c)
            axes[0][c - 1].plot([p[0] for p in poly], [p[1] for p in poly], color="#1f77b4", lw=1.6)
        for m, rings in kl.items():
            for ring in rings:
                xs, ys = [p[0] for p in ring] + [ring[0][0]], [p[1] for p in ring] + [ring[0][1]]
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#d62728", width=1, dash="dot"),
                                         name="KLayout, complement dilated by a 512-gon", legendgroup="kl",
                                         showlegend=c == 1 and ring is rings[0] and m == MARGINS[0]), row=1, col=c)
                axes[0][c - 1].plot(xs, ys, color="#d62728", lw=0.8, ls=":")
        for ax_range, axis in ((xr, "x"), (yr, "y")):
            fig.update_xaxes(range=list(xr), row=1, col=c); fig.update_yaxes(range=list(yr), row=1, col=c)
        axes[0][c - 1].set_xlim(*xr); axes[0][c - 1].set_ylim(*yr); axes[0][c - 1].set_aspect("equal"); axes[0][c - 1].set_title(title)
        # radial deviation of both representations from the exact circle of radius radius_of(m) about the centre
        for m in MARGINS:
            r = radius_of(m)
            if r <= 0:
                continue
            exact = [(math.degrees(math.atan2(y - centre[1], x - centre[0])), (math.hypot(x - centre[0], y - centre[1]) - r) * 1000)
                     for poly in band_edges for x, y in poly if abs(math.hypot(x - centre[0], y - centre[1]) - r) < 0.02]
            poly_pts = [(math.degrees(math.atan2(y - centre[1], x - centre[0])), (math.hypot(x - centre[0], y - centre[1]) - r) * 1000)
                        for ring in kl[m] for x, y in ring if abs(math.hypot(x - centre[0], y - centre[1]) - r) < 0.02]
            for data, colour, name in ((exact, "#1f77b4", "exact"), (poly_pts, "#d62728", "KLayout 512-gon")):
                data = data[:: max(1, len(data) // 400)]  # keep the files small
                if data:
                    fig.add_trace(go.Scatter(x=[d[0] for d in data], y=[d[1] for d in data], mode="markers", marker=dict(size=3, color=colour),
                                             name=name, showlegend=False, hovertext=f"m = {m} um"), row=2, col=c)
                    axes[1][c - 1].plot([d[0] for d in data], [d[1] for d in data], ".", ms=2, color=colour)
        axes[1][c - 1].set_xlabel("angle about the centre (deg)"); axes[1][c - 1].set_ylabel("radial deviation (nm)")
        axes[1][c - 1].set_title("deviation from the exact radius")
    fig.update_layout(title="Exact offset carriers vs the polygonal (512-gon) erosion, margins 0.1, 0.5, 2.0 um",
                      width=1500, height=900)
    save(fig, "fig_exact_arcs_vs_512gon")
    fmatplot.suptitle("Exact offset arcs (blue) vs KLayout 512-gon erosion (red dotted); margins 0.1, 0.5, 2.0 um")
    save_svg(fmatplot, "fig_exact_arcs_vs_512gon")


def figure_c(spec) -> None:
    result = build(spec, keep_raw=True)
    layer = result.level_layers[1]
    raw, labels = layer.raw["arrangement"], layer.raw["labels"]
    view = VIEWS["pad"]
    before = [[(p[0] * G, p[1] * G) for p in (e.a, e.b)] for e in raw.edges]
    after = edge_polylines(layer, lambda src: True)
    families = edge_polylines(layer, lambda src: any(s.startswith("band_") for s in src))
    stats = layer.stats
    fig = make_subplots(rows=1, cols=2, shared_xaxes=True, shared_yaxes=True,
                        subplot_titles=(f"before: {stats['edges_before']} noded edges, {stats['faces_before']} cells "
                                        "(arcs as witness chords)", f"after: {stats['edges_after']} edges, {stats['faces_after']} cells"))
    for col, polys, colour, width in ((1, before, "#555", 0.6), (2, after, "#111", 1.0)):
        for poly in polys:
            fig.add_trace(go.Scatter(x=[p[0] for p in poly], y=[p[1] for p in poly], mode="lines", line=dict(color=colour, width=width),
                                     showlegend=False, hoverinfo="skip"), row=1, col=col)
    for fi in range(1, len(layer.arr.faces)):
        colour, text = face_colour(layer.labels[fi])
        rings = rings_um(layer, fi)
        fig.add_shape(type="path", path=svg_path(rings), fillcolor=colour, opacity=0.55, fillrule="evenodd", line_width=0, layer="below", row=1, col=2)
    for col in (1, 2):
        fig.update_xaxes(range=list(view[0]), row=1, col=col)
        fig.update_yaxes(range=list(view[1]), row=1, col=col, scaleanchor=f"x{'' if col == 1 else 2}")
    fig.update_layout(title="z = 0 level of pad_in_ground_A (with bands) before and after dissolution", width=1500, height=600)
    save(fig, "fig_dissolution_before_after")
    f, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
    for ax, polys, lw in ((axes[0], before, 0.4), (axes[1], after, 0.9)):
        ax.add_collection(LineCollection(polys, colors="#222", linewidths=lw))
        ax.set_xlim(*view[0]); ax.set_ylim(*view[1]); ax.set_aspect("equal")
    for fi in range(1, len(layer.arr.faces)):
        colour, _ = face_colour(layer.labels[fi])
        axes[1].add_patch(PathPatch(mpl_path(rings_um(layer, fi)), facecolor=colour, alpha=0.5, edgecolor="none"))
    axes[0].set_title(f"before dissolution: {stats['edges_before']} edges, {stats['faces_before']} cells")
    axes[1].set_title(f"after: {stats['edges_after']} edges, {stats['faces_after']} cells")
    save_svg(f, "fig_dissolution_before_after")


def stretch_metal(z: np.ndarray, factor: float = 200.0) -> np.ndarray:
    """Display-only z map: the 0.2 um metal layer is stretched ``factor`` times so its walls are visible."""
    top = 0.2 * factor
    return np.where(z <= 0, z, np.where(z <= 0.2, z * factor, z + top - 0.2))


def figure_d() -> None:
    result = build(pad_in_ground("B"))
    topo = result.topology
    tris = surface_triangles(topo, mesh_size=25.0, mesh_size_min=2.0)
    shift = {"substrate": -60.0, "vacuum": 60.0}
    palette = {"SA": "#9ecae1", "MA": "#2171b5", "MS": "#6a51a3", "EXT": "#bdbdbd", "PORT": "#d62728"}
    meshes, polys, colours = [], [], []
    fig = go.Figure()
    for v in topo.volumes.values():
        dz = shift.get(v.domain.name, 0.0)
        label = f"{v.domain.name} ({'emitted' if v.emitted else 'excluded PEC'})"
        for shell in v.shells:
            for fid, _ in shell.faces:
                if fid not in tris:
                    continue
                pts, tri = tris[fid]
                group = topo.faces[fid].group.split("::")[0]
                colour = palette.get(group, "#999")
                offset = np.array([0, 0, dz])
                zs = stretch_metal(pts[:, 2])
                pts = np.column_stack([pts[:, 0], pts[:, 1], zs])
                fig.add_trace(go.Mesh3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2] + dz, i=tri[:, 0], j=tri[:, 1], k=tri[:, 2],
                                        color=colour, opacity=0.55, flatshading=True, name=label, legendgroup=label,
                                        showlegend=False, hovertext=f"{label}: {topo.faces[fid].group}"))
                for t in tri:
                    polys.append(pts[t] + offset)
                    colours.append(colour)
    fig.update_layout(title="pad_in_ground_B exploded: substrate down, vacuum up, excluded Ground/pad in place; "
                      "the 0.2 um metal layer is drawn 200x thicker; each face is drawn once per adjacent volume", width=1100, height=800,
                      scene=dict(aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.6), zaxis=dict(title="z (um, metal layer stretched 200x)", range=[-200, 200])))
    save(fig, "fig_exploded_volumes_B")
    f = plt.figure(figsize=(11, 8))
    ax = f.add_subplot(111, projection="3d")
    ax.add_collection3d(Poly3DCollection(polys, facecolors=colours, edgecolors="none", alpha=0.5))
    ax.set_xlim(-150, 150); ax.set_ylim(-150, 150); ax.set_zlim(-200, 200)
    ax.set_box_aspect((1, 1, 0.6)); ax.view_init(elev=22, azim=-55)
    ax.set_title("pad_in_ground_B exploded volumes and walls (metal layer drawn 200x thicker)")
    save_svg(f, "fig_exploded_volumes_B")


def main() -> None:
    spec = pad_in_ground("A", bands_um=MARGINS)
    result = build(spec)
    figure_a(result)
    figure_b(spec, result)
    figure_c(spec)
    figure_d()
    print("figures:", sorted(p.name for p in FIG.iterdir()))


if __name__ == "__main__":
    main()
