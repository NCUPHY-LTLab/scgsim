"""Figures and numbers of ``labels-and-dissolution.qmd`` (all computed from the prototype).

Writes into ``docs/design/sgb/figures/``:

* ``labels-slab-cells.svg``      slab labels of the Route B running example: which prisms cover each cell and what wins;
* ``labels-level-links.svg/.html`` level labels of the same model obtained by linking, with the "needs a face" rule;
* ``labels-dissolution.svg/.html`` one slab before labelling, with its seams marked, and after dissolution;
* ``labels-numbers.json``        the counts, area memberships and error messages the page quotes.

The dissolution example is the running example plus one extra same-Net prism (``pad_tab``) that overlaps the pad, so
that there is a same-material seam to remove. Everything else is the unchanged ``pad_in_ground("B")`` scene.

Run (from the package root):  .venv/bin/python docs/design/sgb/_figures/labels_and_dissolution.py
"""

from __future__ import annotations

from collections import Counter

from labels_common import (
    G_UM, PALETTE, cell_rings_um, domain_colour, domain_name, edge_polyline_um, finish_plan, go, level_label_text,
    mpl_path, plt, plotly_cells, save_plotly, save_svg, write_numbers,
)
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Patch
from plotly.subplots import make_subplots

from sgbproto.arcs import face_exact_area
from sgbproto.errors import ContactError, OverlapError
from sgbproto.labels import label_slab
from sgbproto.pipeline import build
from sgbproto.records import Footprint, ModelSpec, PrismRecord, z_value
from sgbproto.scenes import pad_in_ground, rect_um, to_grid

VIEW = ((-56.0, 56.0), (-36.0, 52.0))  # pad region, micrometres
SLAB = 1  # slab[0,1/10] of Route B: ground, pad and the vacuum above the substrate are active


def with_tab() -> ModelSpec:
    """Running example (Route B) plus a same-Net tab that overlaps the pad's top edge and sticks out of it."""
    base = pad_in_ground("B")
    footprint = Footprint.from_rings(to_grid(base.grid, rect_um(-20, 20, 20, 45)))
    tab = PrismRecord("pad_tab", footprint, z_value(0), z_value("0.2"), "Al", "conductor", net_id="pad")
    return ModelSpec(base.grid, base.prisms + (tab,), name="pad_in_ground_B_tab", meta=base.meta)


def hit_class(hits: list[int], prisms) -> str:
    names = sorted(prisms[k].entity_id for k in hits if prisms[k].entity_id not in ("vacuum", "substrate"))
    return "+".join(names) if names else "vacuum only"


# --------------------------------------------------------------------------- figure 1: slab labels


def figure_slab_cells(result, numbers: dict) -> None:
    layer = result.slabs[SLAB]
    arr = layer.arr
    labels, hits = label_slab(arr, layer.cells)
    prisms = result.spec.prisms
    rows = []
    fig, (ax, ax_tab) = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
    for fi in range(1, len(arr.faces)):
        ax.add_patch(PathPatch(mpl_path(cell_rings_um(layer, fi)), facecolor=domain_colour(labels[fi]), edgecolor=PALETTE["ink"], lw=0.8))
        covering = sorted(prisms[k].entity_id for k in hits[fi])
        rows.append({"cell": fi, "prisms_covering": covering, "label": domain_name(labels[fi]),
                     "kind": labels[fi].kind, "area_um2": face_exact_area(arr, layer.edge_curve, fi) * G_UM**2})
    # one annotation per cell, anchored at its representative point
    from sgbproto.arrangement import interior_point

    for fi in range(1, len(arr.faces)):
        p = interior_point(arr, fi)
        x, y = p[0] / p[2] * G_UM, p[1] / p[2] * G_UM
        if abs(x) > 85 or abs(y) > 65:  # the cell's representative point lies outside the cropped view
            x, y = -86.0, 64.0
        ax.annotate(f"cell {fi}", (x, y), fontsize=8, ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1))
    finish_plan(ax, (-90, 90), (-70, 70), f"{layer.plane}: cells of the running example (Route B, view cropped to the cutout)")
    ax.set_xlabel("x (µm)")
    ax_tab.axis("off")
    cell_text = [[str(r["cell"]), ", ".join(r["prisms_covering"]), r["label"], r["kind"]] for r in rows]
    table = ax_tab.table(cellText=cell_text, colLabels=["cell", "prisms covering the cell", "slab label", "kind"],
                         loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width([0, 1, 2, 3])
    table.scale(1, 1.6)
    ax_tab.set_title("Domain resolution per cell", pad=14)
    save_svg(fig, "labels-slab-cells")
    numbers["slab_cells"] = {"plane": layer.plane, "active_prisms": [prisms[k].entity_id for k in result.levels.slab_active[SLAB]],
                             "cells": rows}


# --------------------------------------------------------------------------- figure 2: level labels by linking


PAIR_COLOURS = {
    ("substrate", "Ground"): "#7b7f85", ("substrate", "pad"): "#2e5c8a", ("substrate", "vacuum"): "#a9cbe8",
    ("Ground", "Ground"): "#e3e4e6", ("pad", "pad"): "#cfdced", ("vacuum", "vacuum"): "#fbf6df",
    ("Ground", "vacuum"): "#5b5f64", ("pad", "vacuum"): "#1f4e79",
}


def pair_of(label) -> tuple[str, str]:
    return domain_name(label.below), domain_name(label.above)


def level_colour(label) -> str:
    if label.sheet is not None and label.sheet.kind == "port":
        return PALETTE["port"]
    return PAIR_COLOURS.get(pair_of(label), "#cccccc")


def figure_level_links(result, numbers: dict) -> None:
    indices = [1, 2, 3]  # z = 0, 0.1, 0.2
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    pcs = make_subplots(rows=1, cols=3, subplot_titles=[result.level_layers[i].plane for i in indices], horizontal_spacing=0.04)
    summary = []
    for col, (ax, li) in enumerate(zip(axes, indices), start=1):
        layer = result.level_layers[li]
        faces = 0
        for fi in range(1, len(layer.arr.faces)):
            lab = layer.labels[fi]
            needs = lab.needs_face()
            faces += needs
            ax.add_patch(PathPatch(mpl_path(cell_rings_um(layer, fi)), facecolor=level_colour(lab),
                                   alpha=1.0 if needs else 0.55, edgecolor=PALETTE["ink"], lw=1.4 if needs else 0.5,
                                   hatch=None if needs else "///"))
        finish_plan(ax, (-90, 90), (-70, 70), f"{layer.plane}: {faces} of {len(layer.arr.faces) - 1} cells need a face")
        plotly_cells(pcs, layer, lambda lay, fi: level_colour(lay.labels[fi]), lambda lay, fi: level_label_text(lay.labels[fi]) +
                     f"<br>needs a face: {lay.labels[fi].needs_face()}", 1, col, (-90, 90), (-70, 70))
        summary.append({"plane": layer.plane, "cells": len(layer.arr.faces) - 1, "cells_needing_face": faces,
                        "labels": sorted({level_label_text(layer.labels[fi]) for fi in range(1, len(layer.arr.faces))})})
    legend = [Patch(facecolor=c, edgecolor="#222", label=f"{a} below, {b} above") for (a, b), c in PAIR_COLOURS.items()]
    legend.append(Patch(facecolor=PALETTE["port"], edgecolor="#222", label="port sheet p1 (owner on the level)"))
    legend.append(Patch(facecolor="white", edgecolor="#222", hatch="///", label="hatched: same domain both sides, no sheet: no face"))
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout()
    save_svg(fig, "labels-level-links")
    pcs.update_layout(title="Level labels (below | above | sheet) of pad_in_ground B, obtained by linking to the slab cells",
                      width=1500, height=520)
    for i in range(1, 4):
        pcs.update_yaxes(scaleanchor=f"x{'' if i == 1 else i}", row=1, col=i)
    save_plotly(pcs, "labels-level-links")
    numbers["level_labels"] = summary


# --------------------------------------------------------------------------- figure 3: dissolution


def classify_raw_edges(raw, raw_labels) -> tuple[list[int], list[int]]:
    """Raw edge indices whose two sides differ (required by label) and whose two sides are equal (seams)."""
    required, seams = [], []
    for ei in range(len(raw.edges)):
        left, right = raw_labels[raw.he_face[2 * ei]], raw_labels[raw.he_face[2 * ei + 1]]
        (required if left != right else seams).append(ei)
    return required, seams


def figure_dissolution(result, numbers: dict, tab_result=None) -> None:
    layer = result.slabs[SLAB]
    raw, raw_labels = layer.raw["arrangement"], layer.raw["labels"]
    no_curves = [None] * len(raw.edges)
    _, hits = label_slab(raw, layer.cells)
    prisms = result.spec.prisms
    required, seams = classify_raw_edges(raw, raw_labels)
    stats = layer.stats

    cls_colour = {"vacuum only": PALETTE["vacuum"], "ground": PALETTE["ground"], "pad": "#7aa3cf", "pad_tab": "#e7b24a",
                  "pad+pad_tab": "#c46a28"}
    pad_cells = [fi for fi in range(1, len(raw.faces)) if "pad" in hit_class(hits[fi], prisms) or "tab" in hit_class(hits[fi], prisms)]

    def raw_class(fi):
        return hit_class(hits[fi], prisms)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))
    pcs = make_subplots(rows=1, cols=3, horizontal_spacing=0.03, subplot_titles=(
        f"A. noded cells: {stats['raw_faces']} cells, {stats['edges_before']} edges",
        "B. slab labels: equal labels across red seams",
        f"C. dissolved: {stats['faces_after']} cells, {stats['edges_after']} edges"))
    # A: raw cells by covering prisms, seams red
    for fi in range(1, len(raw.faces)):
        cls = raw_class(fi)
        rings = cell_rings_um(layer, fi, arr=raw, curves=no_curves)
        axes[0].add_patch(PathPatch(mpl_path(rings), facecolor=cls_colour.get(cls, "#ddd"), edgecolor="none"))
        pcs.add_shape(type="path", path=_svg(rings), fillcolor=cls_colour.get(cls, "#ddd"), fillrule="evenodd", line_width=0,
                      layer="below", row=1, col=1)
    # B: labels
    for fi in range(1, len(raw.faces)):
        lab = raw_labels[fi]
        rings = cell_rings_um(layer, fi, arr=raw, curves=no_curves)
        axes[1].add_patch(PathPatch(mpl_path(rings), facecolor=domain_colour(lab), edgecolor="none"))
        pcs.add_shape(type="path", path=_svg(rings), fillcolor=domain_colour(lab), fillrule="evenodd", line_width=0,
                      layer="below", row=1, col=2)
    for col, ax in ((1, axes[0]), (2, axes[1])):
        for group, colour, width in ((required, PALETTE["ink"], 0.7), (seams, PALETTE["seam"], 2.0)):
            for ei in group:
                pts = edge_polyline_um(raw, no_curves, ei)
                ax.plot(*zip(*pts), color=colour, lw=width, solid_capstyle="round")
        pcs.add_trace(go.Scatter(x=[p for ei in required for p in (edge_polyline_um(raw, no_curves, ei)[0][0], edge_polyline_um(raw, no_curves, ei)[-1][0], None)],
                                 y=[p for ei in required for p in (edge_polyline_um(raw, no_curves, ei)[0][1], edge_polyline_um(raw, no_curves, ei)[-1][1], None)],
                                 mode="lines", line=dict(color=PALETTE["ink"], width=1), name="labels differ", hoverinfo="skip"), row=1, col=col)
        pcs.add_trace(go.Scatter(x=[p for ei in seams for p in (edge_polyline_um(raw, no_curves, ei)[0][0], edge_polyline_um(raw, no_curves, ei)[-1][0], None)],
                                 y=[p for ei in seams for p in (edge_polyline_um(raw, no_curves, ei)[0][1], edge_polyline_um(raw, no_curves, ei)[-1][1], None)],
                                 mode="lines", line=dict(color=PALETTE["seam"], width=3), name="seam: same label both sides"), row=1, col=col)
    # C: after
    for fi in range(1, len(layer.arr.faces)):
        rings = cell_rings_um(layer, fi)
        axes[2].add_patch(PathPatch(mpl_path(rings), facecolor=domain_colour(layer.labels[fi]), edgecolor="none"))
        pcs.add_shape(type="path", path=_svg(rings), fillcolor=domain_colour(layer.labels[fi]), fillrule="evenodd", line_width=0,
                      layer="below", row=1, col=3)
    for ei in range(len(layer.arr.edges)):
        pts = edge_polyline_um(layer.arr, layer.edge_curve, ei)
        axes[2].plot(*zip(*pts), color=PALETTE["ink"], lw=1.4)
        pcs.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="lines", line=dict(color=PALETTE["ink"], width=2),
                                 showlegend=False, hoverinfo="skip"), row=1, col=3)
    titles = [f"A. noded cells: {stats['raw_faces']} cells, {stats['edges_before']} edges; colour = prisms covering the cell",
              "B. slab labels: the overlap cell and its neighbours carry the same label",
              f"C. after dissolution: {stats['faces_after']} cells, {stats['edges_after']} edges"]
    for ax, title in zip(axes, titles):
        finish_plan(ax, *VIEW)
        ax.set_title(title, fontsize=8.5)
    handles = [Patch(facecolor=cls_colour[k], edgecolor="#555", label=k) for k in ("vacuum only", "pad", "pad_tab", "pad+pad_tab")]
    handles += [Line2D([0], [0], color=PALETTE["seam"], lw=2.5, label="seam (dissolved)"),
                Line2D([0], [0], color=PALETTE["ink"], lw=1.2, label="required (labels differ)")]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout()
    save_svg(fig, "labels-dissolution")
    for col in (1, 2, 3):
        pcs.update_xaxes(range=list(VIEW[0]), row=1, col=col)
        pcs.update_yaxes(range=list(VIEW[1]), row=1, col=col, scaleanchor=f"x{'' if col == 1 else col}")
    pcs.update_layout(title=f"{layer.plane} of pad_in_ground B with a same-Net tab, before and after dissolution", width=1500, height=560)
    save_plotly(pcs, "labels-dissolution")

    # numbers: edge classes, merges, area membership of the merged pad cell
    kept_by_label = len(required)
    pad_face = next(fi for fi in range(1, len(layer.arr.faces)) if layer.labels[fi].kind == "conductor" and layer.labels[fi].net == "pad")
    covered: Counter = Counter()
    for fi in range(1, len(raw.faces)):
        area = raw.face_twice_area(fi) / 2 * G_UM**2
        for k in hits[fi]:
            if prisms[k].entity_id in ("pad", "pad_tab"):
                covered[prisms[k].entity_id] += area
    face_area = face_exact_area(layer.arr, layer.edge_curve, pad_face) * G_UM**2  # arcs included
    # the area-weighted membership the prototype records on the emitted top face of the merged pad (brief section 14)
    top = next(f for f in tab_result.topology.faces.values() if f.group == "MA::pad" and f.kind == "horizontal")
    wall = next(f for f in tab_result.topology.faces.values() if f.group == "MA::pad" and f.kind == "wall" and "pad_tab" in f.membership)
    numbers["membership_record"] = {"top_face_group": top.group, "top_face_membership": dict(top.membership),
                                    "a_wall_of_the_tab": {"group": wall.group, "membership": dict(wall.membership)}}
    numbers["dissolution"] = {
        "plane": layer.plane, "raw_cells": stats["raw_faces"], "cells_after": stats["faces_after"],
        "raw_edges": stats["edges_before"], "edges_after": stats["edges_after"],
        "raw_edges_labels_differ": kept_by_label, "raw_edges_seam": len(seams),
        "edges_merged_by_degree_two_removal": kept_by_label - stats["edges_after"],
        "merged_pad_cell_area_um2": face_area,
        "membership_um2": dict(covered), "membership_fraction": {k: v / face_area for k, v in covered.items()},
    }


def _svg(rings) -> str:
    from labels_common import svg_path

    return svg_path(rings)


# --------------------------------------------------------------------------- errors and the running example without the tab


def error_messages(numbers: dict) -> None:
    base = pad_in_ground("B")
    box = Footprint.from_rings(to_grid(base.grid, rect_um(-60, -40, -40, -20)))  # overlaps the pad's left end
    other = PrismRecord("other_net", box, z_value(0), z_value("0.2"), "Al", "conductor", net_id="other")
    out = {}
    try:
        build(ModelSpec(base.grid, base.prisms + (other,), name="bad", meta=base.meta))
    except OverlapError as e:
        out["different_net_overlap"] = str(e)
    ambiguous = PrismRecord("alumina", box, z_value(-100), z_value(0), "alumina", "dielectric")
    try:
        build(ModelSpec(base.grid, base.prisms + (ambiguous,), name="bad2", meta=base.meta))
    except OverlapError as e:
        out["equal_priority_dielectrics"] = str(e)
    touching = PrismRecord("touching_net", Footprint.from_rings(to_grid(base.grid, rect_um(-60, -20, -50, 20))), z_value(0), z_value("0.2"),
                           "Al", "conductor", net_id="other")  # shares the pad's left edge x = -50 um
    try:
        build(ModelSpec(base.grid, base.prisms + (touching,), name="bad3", meta=base.meta))
    except ContactError as e:
        out["different_net_contact"] = str(e)
    prioritised = PrismRecord("alumina", box, z_value(-100), z_value(0), "alumina", "dielectric", priority=1)
    ok = build(ModelSpec(base.grid, base.prisms + (prioritised,), name="ok", meta=base.meta))
    out["priority_resolves"] = {"volumes": sorted(v.domain.name for v in ok.topology.volumes.values()),
                                "validation_passed": not ok.report.failures}
    numbers["errors"] = out


def main() -> None:
    plain = build(pad_in_ground("B"), keep_raw=True)
    tab = build(with_tab(), keep_raw=True)
    numbers: dict = {"scene": "pad_in_ground('B')", "tab_scene": "pad_in_ground('B') + pad_tab (same Net, z 0..0.2, x -20..20, y 20..45 um)"}
    figure_slab_cells(plain, numbers)
    figure_level_links(plain, numbers)
    figure_dissolution(tab, numbers, tab)
    error_messages(numbers)
    numbers["plain_dissolution_stats"] = {s.plane: {"raw_cells": s.stats["raw_faces"], "cells_after": s.stats["faces_after"],
                                                    "raw_edges": s.stats["edges_before"], "edges_after": s.stats["edges_after"]}
                                          for s in plain.slabs + plain.level_layers}
    write_numbers("labels", numbers)
    print("labels: done")


if __name__ == "__main__":
    main()
