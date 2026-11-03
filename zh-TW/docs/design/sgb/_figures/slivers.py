"""Figures and numbers of ``slivers.qmd``: one computed example per sliver source of brief section 10.

Each example is small, built from the prototype's own pipeline, and drawn at the scale of the 1 nm grid (pixel grid lines
visible). Writes into ``docs/design/sgb/figures/``:

* ``slivers-gallery.svg/.html``  four rows, "as the source geometry would give it" next to "what SGB builds":
  1. quantization sliver (three nearly concurrent curves)        -> snap rounding,
  2. phantom sliver (edges of two unrelated layers 4 nm apart)    -> local arrangements,
  3. seam sliver (two same-Net prisms overlapping by 2 nm)        -> dissolution,
  4. source near-coincidence (5 nm gap ring; a derived band curve 4 nm from a source curve) -> reported, not removed;
* ``slivers-numbers.json``      the measurements, including the report entries the page quotes.

"As the source would give it" is the exact arrangement (1), the old global projection emulated by a ``required`` family
that carries every layer's boundary on every level (2), and the noded cells before dissolution (3). Example 4b's "after" is
an *emulation* of derived-to-source snapping (the prototype has no such policy): the source curve is moved onto the
derived curve by 4 nm, which gives the same topology the policy would.

Run (from the package root):  .venv/bin/python docs/design/sgb/_figures/slivers.py
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

from labels_common import PALETTE, go, np, plt, save_plotly, save_svg, write_numbers
from matplotlib.patches import PathPatch, Patch
from plotly.subplots import make_subplots

from sgbproto.arcs import cycle_polyline
from sgbproto.errors import ContactError
from sgbproto.arrangement import build_arrangement
from sgbproto.levels import build_levels
from sgbproto.pipeline import build
from sgbproto.records import CurveFamilyRecord, Footprint, LineCarrier, Lineage, ModelSpec, PrismRecord, z_value
from sgbproto.scenes import _chip_prisms, pad_in_ground, rect_um, to_grid
from sgbproto.sliver import sliver_report
from sgbproto.snap_rounding import Segment, snap_round

from labels_common import mpl_path

RED = PALETTE["bad"]


# --------------------------------------------------------------------------- shared drawing (grid units)


def rings_grid(layer, fi: int, arr=None, curves=None, window=None):
    """Boundary rings of a face in grid units; with ``window = (xlim, ylim)`` the face is clipped to it (keeps the SVG small)."""
    arr = arr if arr is not None else layer.arr
    curves = curves if curves is not None else layer.edge_curve
    rings = [cycle_polyline(arr, curves, c, 4.0) for c in arr.face_cycles(fi)]
    if window is None:
        return rings
    from shapely.geometry import Polygon, box

    (x0, x1), (y0, y1) = window
    clipped = Polygon(rings[0], rings[1:]).buffer(0).intersection(box(x0 - 1, y0 - 1, x1 + 1, y1 + 1))
    out = []
    for part in getattr(clipped, "geoms", [clipped]):
        if part.is_empty or part.geom_type != "Polygon":
            continue
        out.append(list(part.exterior.coords)[:-1])
        out.extend(list(r.coords)[:-1] for r in part.interiors)
    return out


def face_width(arr, fi: int) -> float:
    """Width 2A/P of a face in grid units (holes count in the perimeter)."""
    area = arr.face_twice_area(fi) / 2
    per = sum(math.dist(p, q) for a, b in [(0, 0)] for p, q in arr.face_segments(fi))
    return 2 * area / per if per else 0.0


def pixel_grid(ax, xlim, ylim) -> None:
    for x in range(math.ceil(xlim[0]), math.floor(xlim[1]) + 1):
        ax.axvline(x - 0.5, color="#d0d0d0", lw=0.5, zorder=0)
    for y in range(math.ceil(ylim[0]), math.floor(ylim[1]) + 1):
        ax.axhline(y - 0.5, color="#d0d0d0", lw=0.5, zorder=0)


def draw_cells(ax, layer, colour_of, xlim, ylim, narrow=10.0, arr=None, curves=None, lw=1.0) -> list[float]:
    arr = arr if arr is not None else layer.arr
    labels = layer.labels if arr is layer.arr else layer.raw["labels"]
    widths = []
    for fi in range(1, len(arr.faces)):
        w = face_width(arr, fi)
        widths.append(w)
        sliver = w < narrow
        rings = rings_grid(layer, fi, arr, curves, (xlim, ylim))
        if not rings:  # the face lies outside the window
            continue
        ax.add_patch(PathPatch(mpl_path(rings), facecolor=colour_of(labels[fi], fi), edgecolor=PALETTE["ink"],
                               lw=lw, hatch="////" if sliver else None, zorder=2))
    pixel_grid(ax, xlim, ylim)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks(np.linspace(xlim[0], xlim[1], 5).round())
    ax.ticklabel_format(useOffset=False, style="plain")
    return widths


def pair_colour(label, fi=None) -> str:
    below = getattr(label, "below", None)
    above = getattr(label, "above", None)
    names = (getattr(below, "name", "-"), getattr(above, "name", "-"))
    if label.sheet is not None:
        return "#b9b9b9" if label.sheet.kind == "conductor" else "#f3b4ae"
    if names[0] == "substrate" and names[1] == "vacuum":
        return "#a9cbe8"
    if names[0] == "substrate":
        return "#7b7f85"
    return "#fbf6df"


def domain_fill(domain, fi=None) -> str:
    if domain is None:
        return "#f2f2f2"
    return {"conductor": "#7aa3cf", "vacuum": "#fbf6df", "dielectric": "#cfe0f1"}[domain.kind]


# --------------------------------------------------------------------------- 1. quantization sliver


def cross(s, t):
    (x1, y1), (x2, y2) = s
    (x3, y3), (x4, y4) = t
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    a, b = x1 * y2 - y1 * x2, x3 * y4 - y3 * x4
    return (Fraction(a * (x3 - x4) - (x1 - x2) * b, d), Fraction(a * (y3 - y4) - (y1 - y2) * b, d))


def quantization(numbers: dict):
    l1, l2, l3 = ((0, 0), (1000, 1000)), ((0, 1000), (1000, 0)), ((0, 495), (995, 505))
    exact = [cross(l1, l2), cross(l1, l3), cross(l2, l3)]
    sides = [math.dist(map(float, p), map(float, q)) for p, q in itertools.combinations(exact, 2)]
    area = abs(float((exact[1][0] - exact[0][0]) * (exact[2][1] - exact[0][1]) - (exact[2][0] - exact[0][0]) * (exact[1][1] - exact[0][1]))) / 2
    snapped = snap_round([Segment(s[0], s[1], frozenset([n])) for s, n in zip((l1, l2, l3), ("L1", "L2", "L3"))])
    arr = build_arrangement(snapped.edges)
    lengths = sorted(math.dist(e.a, e.b) for e in arr.edges)
    numbers["quantization"] = {
        "curves": "L1 (0,0)-(1000,1000), L2 (0,1000)-(1000,0), L3 (0,495)-(995,505), grid units",
        "exact_triangle_vertices": [[float(x), float(y)] for x, y in exact], "exact_triangle_sides_grid": sides,
        "exact_triangle_area_grid2": area, "exact_edges_shorter_than_g": sum(s < 1 for s in sides),
        "snapped_bounded_faces": len(arr.faces) - 1, "snapped_edges": len(arr.edges), "snapped_shortest_edge_grid": lengths[0],
        "snapped_hot_pixels": snapped.hot_pixels, "snapped_rounds": snapped.rounds,
        "snapped_max_vertex_drift_grid": 0.7071,
    }
    return l1, l2, l3, exact, arr


def draw_quantization(ax_left, ax_right, data) -> None:
    l1, l2, l3, exact, arr = data
    for line, colour in ((l1, "#1f77b4"), (l2, "#2ca02c"), (l3, "#d62728")):
        (x0, y0), (x1, y1) = line
        xs = np.array([499.9, 500.1])
        ys = y0 + (y1 - y0) * (xs - x0) / (x1 - x0)
        ax_left.plot(xs, ys, color=colour, lw=1.6)
        xr = np.array([496.0, 504.0])
        ax_right.plot(xr, y0 + (y1 - y0) * (xr - x0) / (x1 - x0), color=colour, lw=0.7, ls=":", zorder=1)
    tri = [(float(x), float(y)) for x, y in exact]
    ax_left.fill([p[0] for p in tri], [p[1] for p in tri], color=RED, alpha=0.35, hatch="////", zorder=3)
    ax_left.plot([p[0] for p in tri], [p[1] for p in tri], "o", color=RED, ms=3)
    ax_left.ticklabel_format(useOffset=False, style="plain")
    ax_left.set_xlim(499.93, 500.07)
    ax_left.set_ylim(499.93, 500.07)
    ax_left.set_aspect("equal")
    ax_left.set_title("1a. Exact arrangement of three curves, zoom 0.14 g wide:\na face with three edges shorter than g", fontsize=8.5)
    for e in arr.edges:
        ax_right.plot([e.a[0], e.b[0]], [e.a[1], e.b[1]], color=PALETTE["ink"], lw=1.6, zorder=3)
    verts = {v for e in arr.edges for v in (e.a, e.b)}
    ax_right.plot([v[0] for v in verts], [v[1] for v in verts], "o", color=PALETTE["ink"], ms=4, zorder=4)
    pixel_grid(ax_right, (496, 504), (496, 504))
    ax_right.set_xlim(496, 504)
    ax_right.set_ylim(496, 504)
    ax_right.set_aspect("equal")
    ax_right.set_title("1b. Snap rounding, window 8 g wide (dotted: the exact curves):\none vertex, no face, no edge shorter than g", fontsize=8.5)


# --------------------------------------------------------------------------- 2. phantom sliver


def phantom_spec(global_projection: bool) -> ModelSpec:
    base = pad_in_ground("B")
    grid = base.grid
    prisms = _chip_prisms(grid, 150.0)
    f1 = Footprint.from_rings(to_grid(grid, rect_um(-100, -50, 100, 50)))
    f2 = Footprint.from_rings(to_grid(grid, rect_um(-99.996, -50, 100.004, 50)))
    m1 = PrismRecord("m1", f1, z_value(0), z_value("0.2"), "Al", "conductor", net_id="m1")
    m2 = PrismRecord("m2", f2, z_value(3), z_value("3.3"), "Al", "conductor", net_id="m2")
    prisms += [m1, m2]
    families = ()
    if global_projection:
        levels = build_levels(tuple(prisms)).levels
        carriers = tuple(c for p in (m1, m2) for c in p.carriers())
        families = (CurveFamilyRecord("global_projection", "projection", tuple(
            LineCarrier(f"gp/{c.id}", c.p, c.q, Lineage("family_member", family_id="gp", member=c.id)) for c in carriers), tuple(levels), "required"),)
    return ModelSpec(grid, tuple(prisms), families, name="phantom", meta={"route": "B"})


def phantom(numbers: dict):
    local, glob = build(phantom_spec(False)), build(phantom_spec(True))
    out = {}
    for name, r in (("local", local), ("global_emulated", glob)):
        rep = sliver_report(r, top=200, drift=False)
        thin = [f for f in rep["narrowest_faces"] if f["value_um"] < 0.01]
        out[name] = {"faces_narrower_than_10nm": len(thin), "planes_of_those_faces": sorted({f["plane"] for f in thin}),
                     "model_faces": r.counts["faces"], "model_edges": r.counts["edges"], "level_cells": r.counts["level_cells"],
                     "narrowest_face": rep["narrowest_faces"][0], "shortest_edge_um": rep["summary"]["shortest_edge_um"],
                     "narrowest_face_um": rep["summary"]["narrowest_face_um"], "validation_passed": not r.report.failures}
    numbers["phantom"] = out
    return local, glob


def draw_phantom(ax_left, ax_right, local, glob) -> None:
    xlim, ylim = (99996, 100008), (-6, 6)
    for ax, result, title in ((ax_left, glob, "2a. Old global projection (emulated): level z = 0 also carries the upper layer's\nedge, 4 nm from the lower layer's: a 4 nm substrate-vacuum face"),
                              (ax_right, local, "2b. Local arrangement: level z = 0 holds only curves that meet there")):
        idx = result.levels.levels.index(Fraction(0))
        layer = result.level_layers[idx]
        draw_cells(ax, layer, pair_colour, xlim, ylim)
        ax.set_title(title, fontsize=8.5)


# --------------------------------------------------------------------------- 3. seam sliver


def seam_spec() -> ModelSpec:
    base = pad_in_ground("B")
    grid = base.grid
    prisms = _chip_prisms(grid, 150.0)
    a = Footprint.from_rings(to_grid(grid, rect_um(-100, -20, 0.002, 20)))
    b = Footprint.from_rings(to_grid(grid, rect_um(0.0, -20, 100, 20)))
    prisms += [PrismRecord("a", a, z_value(0), z_value("0.2"), "Al", "conductor", net_id="n"),
               PrismRecord("b", b, z_value(0), z_value("0.2"), "Al", "conductor", net_id="n")]
    return ModelSpec(grid, tuple(prisms), name="seam", meta={"route": "B"})


def seam(numbers: dict):
    result = build(seam_spec(), keep_raw=True)
    layer = next(s for s in result.slabs if s.plane == "slab[0,1/5]")
    raw, raw_labels = layer.raw["arrangement"], layer.raw["labels"]
    raw_widths = sorted(face_width(raw, fi) for fi in range(1, len(raw.faces)))
    dis_widths = sorted(face_width(layer.arr, fi) for fi in range(1, len(layer.arr.faces)))
    numbers["seam"] = {"slab": layer.plane, "raw_cells": len(raw.faces) - 1, "cells_after": len(layer.arr.faces) - 1,
                       "raw_edges": len(raw.edges), "edges_after": len(layer.arr.edges),
                       "raw_narrowest_cell_width_grid": raw_widths[0], "after_narrowest_cell_width_grid": dis_widths[0],
                       "overlap_um": 0.002}
    return result, layer


def draw_seam(ax_left, ax_right, layer) -> None:
    raw, raw_labels = layer.raw["arrangement"], layer.raw["labels"]
    xlim, ylim = (-4, 6), (-5, 5)
    no_curves = [None] * len(raw.edges)
    for fi in range(1, len(raw.faces)):
        rings = rings_grid(layer, fi, raw, no_curves, (xlim, ylim))
        if rings:
            ax_left.add_patch(PathPatch(mpl_path(rings), facecolor=domain_fill(raw_labels[fi]), edgecolor="none",
                                        hatch="////" if face_width(raw, fi) < 10 else None, zorder=2))
    for ei, e in enumerate(raw.edges):
        left, right = raw_labels[raw.he_face[2 * ei]], raw_labels[raw.he_face[2 * ei + 1]]
        ax_left.plot([e.a[0], e.b[0]], [e.a[1], e.b[1]], color=RED if left == right else PALETTE["ink"], lw=2.0 if left == right else 1.2, zorder=3)
    pixel_grid(ax_left, xlim, ylim)
    ax_left.set_xlim(*xlim)
    ax_left.set_ylim(*ylim)
    ax_left.set_aspect("equal")
    ax_left.set_title("3a. Noded cells before dissolution: the 2 nm overlap of two same-Net\nprisms is a cell; its two edges (red) have equal labels on both sides", fontsize=8.5)
    draw_cells(ax_right, layer, domain_fill, xlim, ylim)
    ax_right.set_title("3b. After dissolution: one conductor cell, no seam", fontsize=8.5)


# --------------------------------------------------------------------------- 4. source near-coincidence


def ring_spec() -> ModelSpec:
    base = pad_in_ground("A", with_port=False)
    grid = base.grid
    prisms = _chip_prisms(grid, 150.0)
    ground = Footprint.from_rings(to_grid(grid, rect_um(-150, -150, 150, 150)), [to_grid(grid, rect_um(-50.005, -30.005, 50.005, 30.005))])
    pad = Footprint.from_rings(to_grid(grid, rect_um(-50, -30, 50, 30)))
    prisms += [PrismRecord("ground", ground, z_value(0), z_value(0), "Al", "conductor", net_id="Ground"),
               PrismRecord("pad", pad, z_value(0), z_value(0), "Al", "conductor", net_id="pad")]
    return ModelSpec(grid, tuple(prisms), name="ring_gap", meta={"route": "A"})


def source_gap(numbers: dict):
    result = build(ring_spec())
    rep = sliver_report(result, top=3, drift=False)
    numbers["source_gap"] = {"gap_um": 0.005, "narrowest_faces": rep["narrowest_faces"][:2], "shortest_edges": rep["shortest_edges"][:1],
                             "summary": rep["summary"], "model_faces": result.counts["faces"], "validation_passed": not result.report.failures}
    return result


def coarse_grid_stand_in(numbers: dict) -> None:
    """Stand-in for a coarser snap grid: the 4 nm gap ring rebuilt on a whole-model 10 nm grid (the prototype has one grid per model).

    The gap closes, the two Nets touch, and the build must stop with a contact error."""
    from sgbproto.records import Grid

    out = {}
    for label, g in (("grid_1nm", 0.001), ("grid_10nm", 0.01)):
        grid = Grid(g)
        prisms = _chip_prisms(grid, 150.0)
        gap = 0.004
        ground = Footprint.from_rings(to_grid(grid, rect_um(-150, -150, 150, 150)),
                                      [to_grid(grid, rect_um(-50 - gap, -30 - gap, 50 + gap, 30 + gap))])
        pad = Footprint.from_rings(to_grid(grid, rect_um(-50, -30, 50, 30)))
        prisms += [PrismRecord("ground", ground, z_value(0), z_value(0), "Al", "conductor", net_id="Ground"),
                   PrismRecord("pad", pad, z_value(0), z_value(0), "Al", "conductor", net_id="pad")]
        try:
            r = build(ModelSpec(grid, tuple(prisms), name="coarse", meta={"route": "A"}))
        except ContactError as e:  # the gap closes on the coarse grid: different-Net contact stops the build
            out[label] = {"gap_nm": 4, "stopped": "ContactError", "message": str(e)}
            continue
        out[label] = {"gap_nm": 4, "model_faces": r.counts["faces"], "gap_faces": sum(f.group == "SA::substrate|vacuum" for f in r.topology.faces.values()),
                      "validation_passed": not r.report.failures}
    numbers["coarser_grid_stand_in"] = out


def derived_spec(edge_um: float) -> ModelSpec:
    from sgbproto.records import CurveFamilyRecord as CFR

    base = pad_in_ground("A", bands_um=(0.1, 0.5, 2.0), band_regions=("pad_top",), with_port=False)
    pts = [(-60.0, -20.0), (edge_um, -20.0), (edge_um, 20.0), (-60.0, 20.0)]
    ip = [(round(x * 1000), round(y * 1000)) for x, y in pts]
    carriers = tuple(LineCarrier(f"window/{i}", ip[i], ip[(i + 1) % 4], Lineage("family_member", family_id="window", member=f"window/{i}")) for i in range(4))
    window = CFR("window", "partition", carriers, (Fraction(0),), "required")
    return ModelSpec(base.grid, base.prisms, base.families + (window,), name="derived_vs_source", meta=base.meta)


def derived(numbers: dict):
    near, snapped = build(derived_spec(-47.996)), build(derived_spec(-48.0))
    out = {}
    for name, r in (("before", near), ("after_emulated_snap", snapped)):
        rep = sliver_report(r, top=3, drift=False)
        layer = r.level_layers[1]
        thin = [f for f in sliver_report(r, top=200, drift=False)["narrowest_faces"] if f["value_um"] < 0.01]
        out[name] = {"level_cells": len(layer.arr.faces) - 1, "model_faces": r.counts["faces"], "narrowest_face": rep["narrowest_faces"][0],
                     "faces_narrower_than_10nm": thin,
                     "narrowest_face_um": rep["summary"]["narrowest_face_um"], "validation_passed": not r.report.failures}
    numbers["derived_vs_source"] = out
    return near, snapped


def band_fill(label, fi=None) -> str:
    if label.sheet is not None and label.sheet.kind == "port":
        return "#f3b4ae"
    if label.sheet is not None and label.bands:
        return ["#08306b", "#2171b5", "#6baed6", "#c6dbef"][label.bands[0][1]]
    return "#dddddd"


def draw_source(ax_gap, ax_a, ax_b, ring, near, snapped) -> None:
    xlim, ylim = (49996, 50010), (-6, 6)
    layer = ring.level_layers[1]

    def ring_fill(label, fi=None):
        if label.sheet is None:
            return "#a9cbe8"
        return "#7aa3cf" if label.sheet.name == "pad" else "#9a9a9a"

    draw_cells(ax_gap, layer, ring_fill, xlim, ylim)
    ax_gap.set_title("4a. A 5 nm gap between pad and ground (source fact): the gap cell\nis 5 nm wide along its whole length; kept and reported", fontsize=8.5)
    xlim2, ylim2 = (-48012, -47988), (-8, 8)
    for ax, result, title in ((ax_a, near, "4b. A band curve (derived, 2 µm from the pad edge) and a source curve\n4 nm apart: a 4 nm cell, kept and reported"),
                              (ax_b, snapped, "4c. Emulated derived-to-source snapping: the two curves coincide,\nthe cell is gone")):
        draw_cells(ax, result.level_layers[1], band_fill, xlim2, ylim2, lw=0.9)
        ax.set_title(title, fontsize=8.5)


# --------------------------------------------------------------------------- orchestration


def main() -> None:
    numbers: dict = {}
    q = quantization(numbers)
    local, glob = phantom(numbers)
    seam_result, seam_layer = seam(numbers)
    ring = source_gap(numbers)
    near, snapped = derived(numbers)
    coarse_grid_stand_in(numbers)

    fig, axes = plt.subplots(4, 3, figsize=(14.5, 15.5), gridspec_kw={"width_ratios": [1, 1, 1]})
    for ax in axes.flat:
        ax.tick_params(labelsize=7)
    draw_quantization(axes[0][0], axes[0][1], q)
    axes[0][2].axis("off")
    axes[0][2].text(0, 0.5, "Eliminated by construction:\nsnap rounding\n\nexact crossing triangle with\nthree edges of 0.04 to 0.05 g\nbecomes one vertex", fontsize=9, va="center")
    draw_phantom(axes[1][0], axes[1][1], local, glob)
    axes[1][2].axis("off")
    axes[1][2].text(0, 0.5, "Eliminated by construction:\nlocal arrangements\n\nlevel z = 0 never meets the\nz = 3 layer, so its curve is\nnot on this level", fontsize=9, va="center")
    draw_seam(axes[2][0], axes[2][1], seam_layer)
    axes[2][2].axis("off")
    axes[2][2].text(0, 0.5, "Eliminated by construction:\ndissolution\n\nthe overlap cell and its two\nseam edges disappear", fontsize=9, va="center")
    draw_source(axes[3][0], axes[3][1], axes[3][2], ring, near, snapped)
    for ax, label in zip(axes[3], ("(gap ring)", "(derived vs source)", "(emulated policy)")):
        pass
    for ax in axes.flat:
        if ax.axison:
            ax.set_xlabel("x (grid units, 1 nm)", fontsize=7)
            ax.set_ylabel("y (grid units)", fontsize=7)
    fig.legend(handles=[Patch(facecolor="white", edgecolor="#222", hatch="////", label="cell narrower than 10 g (a sliver)"),
                        plt.Line2D([0], [0], color=RED, lw=2, label="seam edge (equal labels on both sides)")],
               loc="lower center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save_svg(fig, "slivers-gallery")

    # interactive: the same four comparisons as tables of numbers (hover) over a small bar chart of narrowest widths (nm)
    rows = [
        ("1 quantization: shortest exact edge (g)", min(numbers["quantization"]["exact_triangle_sides_grid"]), numbers["quantization"]["snapped_shortest_edge_grid"]),
        ("2 phantom: narrowest face (nm)", numbers["phantom"]["global_emulated"]["narrowest_face_um"] * 1000, numbers["phantom"]["local"]["narrowest_face_um"] * 1000),
        ("3 seam: narrowest cell (g)", numbers["seam"]["raw_narrowest_cell_width_grid"], numbers["seam"]["after_narrowest_cell_width_grid"]),
        ("4b derived vs source: narrowest face (nm)", numbers["derived_vs_source"]["before"]["narrowest_face_um"] * 1000,
         numbers["derived_vs_source"]["after_emulated_snap"]["narrowest_face_um"] * 1000),
    ]
    pcs = go.Figure()
    pcs.add_trace(go.Bar(y=[r[0] for r in rows], x=[r[1] for r in rows], orientation="h", name="without the rule (or before)", marker_color=RED))
    pcs.add_trace(go.Bar(y=[r[0] for r in rows], x=[r[2] for r in rows], orientation="h", name="with SGB (or after)", marker_color=PALETTE["good"]))
    pcs.update_xaxes(type="log", title="measured width or length (log scale; units as labelled)")
    pcs.update_layout(barmode="group", title="Smallest measured feature per sliver source, without and with the rule (computed by slivers.py)",
                      width=1100, height=520, margin=dict(l=320))
    save_plotly(pcs, "slivers-gallery")
    numbers["gallery_bars"] = [{"label": r[0], "without": r[1], "with": r[2]} for r in rows]
    write_numbers("slivers", numbers)
    print("slivers: done")


if __name__ == "__main__":
    main()
