"""Figures and numbers of ``labels-and-dissolution.qmd`` (computed with the production package ``scgsim.geometry.sgb``).

Writes into ``docs/design/sgb/figures/``:

* ``labels-slab-cells.svg``      slab labels of the Route B running example: which prisms cover each cell and what wins;
* ``labels-level-links.svg/.html`` level labels of the same model obtained by linking, with the "needs a face" rule;
* ``labels-dissolution.svg/.html`` one slab of the running example plus a second polygon of the pad entity, before and
  after dissolution;
* ``labels-numbers.json``        the counts, area memberships, group table and error messages the page quotes.

The running example is the production real input ``gds_pad_in_ground`` (``tests/sgb_inputs.py``) under Route B: the metal is
a 0.2 um film on the substrate, the lumped port lies on the support plane z = 0, so the metal layer has the levels 0 and
0.2 only.  The dissolution example adds one polygon (``polygon=1``) to the catalogue's ``pad`` entity, so that the two
polygons of one entity overlap and leave a seam to remove.  Two further builds are labelled as such in the numbers:
the same tab as a separate entity (production keeps the seam between entities) and the stage-level scene
``pad_in_ground("B")`` of ``tests/sgb_scenes.py`` (true arcs, mid-plane port), used only to count the chord edges that
dissolution merges back into arcs.

Run:  SGB_CORE=<sgb-core checkout> <sgb-core>/.artifacts/env/bin/python docs/design/sgb/_figures/labels_and_dissolution.py
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from fractions import Fraction

from labels_common import (
    G_UM, PALETTE, build_model, cell_rings_um, domain_colour, domain_name, edge_polyline_um, finish_plan, go,
    level_label_text, mpl_path, plt, plotly_cells, real_catalogue, save_plotly, save_svg, svg_path, write_numbers,
)
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Patch
from plotly.subplots import make_subplots

from scgsim.geometry.sgb.arrangement.arcs import face_exact_area
from scgsim.geometry.sgb.arrangement.dcel import interior_point
from scgsim.geometry.sgb.errors import ContactError, OverlapError
from scgsim.geometry.sgb.records.grid import exact_z
from scgsim.geometry.sgb.records.prisms import PrismRecord

from sgb_scenes import footprint, pad_in_ground as stage_pad_in_ground, rect_um

PLAN = ((-24.0, 124.0), (-24.0, 124.0))  # the whole model, micrometres
VIEW = ((26.0, 74.0), (26.0, 74.0))  # the cutout, micrometres
METAL_SLAB = "slab[0,1/5]"  # the only slab in which ground and pad are active


def slab_index(model, plane: str) -> int:
    return next(k for k, s in enumerate(model.labelled.slabs) if s.plane == plane)


def level_index(model, z: Fraction) -> int:
    return model.labelled.levels.levels.index(z)


def pad_prism(catalogue) -> PrismRecord:
    return next(p for p in catalogue.prisms if p.entity_id == "pad")


def with_tab(catalogue, *, same_entity: bool = True, priority: int = 0):
    """The catalogue plus a tab that overlaps the pad's top edge and sticks out of it into the gap.

    ``same_entity`` adds it as a second polygon of the entity ``pad``; otherwise as the entity ``pad_tab`` of the same Net.
    """
    pad = pad_prism(catalogue)
    shape = footprint(rect_um(44.0, 50.0, 56.0, 66.0))
    tab = replace(pad, footprint=shape, polygon=1, arcs=())
    if not same_entity:
        tab = replace(tab, entity_id="pad_tab", polygon=0, priority=priority)
    return replace(catalogue, prisms=(*catalogue.prisms, tab))


def covering(layer, prisms, arr, face_index: int) -> list[int]:
    return layer.cells.hit(interior_point(arr, face_index))


def cover_name(prism: PrismRecord) -> str:
    return prism.entity_id if prism.polygon == 0 else f"{prism.entity_id} (polygon {prism.polygon})"


# --------------------------------------------------------------------------- figure 1: slab labels


def figure_slab_cells(model, numbers: dict) -> None:
    k = slab_index(model, METAL_SLAB)
    layer = model.labelled.slabs[k]
    arr = layer.arr
    prisms = model.labelled.catalogue.prisms
    rows = []
    fig, (ax, ax_tab) = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1.1, 1.1]})
    for fi in range(1, len(arr.faces)):
        ax.add_patch(PathPatch(mpl_path(cell_rings_um(layer, fi)), facecolor=domain_colour(layer.labels[fi]),
                               edgecolor=PALETTE["ink"], lw=0.8))
        hits = covering(layer, prisms, arr, fi)
        rows.append({"cell": fi, "prisms_covering": sorted(prisms[h].entity_id for h in hits), "label": domain_name(layer.labels[fi]),
                     "kind": layer.labels[fi].kind, "area_um2": face_exact_area(arr, layer.edge_curve, fi) * G_UM**2})
    # one annotation per cell, anchored at its exact interior point
    for fi in range(1, len(arr.faces)):
        px, py, den = interior_point(arr, fi)
        ax.annotate(f"cell {fi}", (px / den * G_UM, py / den * G_UM), fontsize=8, ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1))
    finish_plan(ax, *PLAN, f"{layer.plane}: the cells of the running example (Route B)")
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
    numbers["slab_cells"] = {"plane": layer.plane, "active_prisms": [prisms[j].entity_id for j in model.labelled.levels.slab_active[k]],
                             "cells": rows}


# --------------------------------------------------------------------------- figure 2: level labels by linking


def pair_of(label) -> tuple[str, str]:
    return domain_name(label.below), domain_name(label.above)


PAIR_COLOURS = {
    ("substrate", "ground_0000"): "#7b7f85", ("substrate", "pad"): "#2e5c8a", ("substrate", "VACUUM_REGION"): "#a9cbe8",
    ("VACUUM_REGION", "VACUUM_REGION"): "#fbf6df", ("ground_0000", "VACUUM_REGION"): "#5b5f64", ("pad", "VACUUM_REGION"): "#1f4e79",
}


def level_colour(label) -> str:
    if label.sheet is not None and label.sheet.kind == "port":
        return PALETTE["port"]
    return PAIR_COLOURS.get(pair_of(label), "#cccccc")


def figure_level_links(model, numbers: dict) -> None:
    indices = [level_index(model, Fraction(0)), level_index(model, Fraction(1, 5))]
    layers = model.labelled.level_layers
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0))
    pcs = make_subplots(rows=1, cols=2, subplot_titles=[layers[i].plane for i in indices], horizontal_spacing=0.05)
    summary = []
    for col, (ax, li) in enumerate(zip(axes, indices), start=1):
        layer = layers[li]
        faces = 0
        for fi in range(1, len(layer.arr.faces)):
            lab = layer.labels[fi]
            needs = lab.needs_face()
            faces += needs
            ax.add_patch(PathPatch(mpl_path(cell_rings_um(layer, fi)), facecolor=level_colour(lab),
                                   alpha=1.0 if needs else 0.55, edgecolor=PALETTE["ink"], lw=1.4 if needs else 0.5,
                                   hatch=None if needs else "///"))
        finish_plan(ax, *PLAN, f"{layer.plane}: {faces} of {len(layer.arr.faces) - 1} cells need a face")
        plotly_cells(pcs, layer, lambda lay, fi: level_colour(lay.labels[fi]), lambda lay, fi: level_label_text(lay.labels[fi]) +
                     f"<br>needs a face: {lay.labels[fi].needs_face()}", 1, col, *PLAN)
        summary.append({"plane": layer.plane, "cells": len(layer.arr.faces) - 1, "cells_needing_face": faces,
                        "labels": sorted({level_label_text(layer.labels[fi]) for fi in range(1, len(layer.arr.faces))})})
    legend = [Patch(facecolor=c, edgecolor="#222", label=f"{a} below, {b} above") for (a, b), c in PAIR_COLOURS.items()]
    legend.append(Patch(facecolor=PALETTE["port"], edgecolor="#222", label="port sheet (owner on the level)"))
    legend.append(Patch(facecolor="white", edgecolor="#222", hatch="///", label="hatched: same domain both sides, no sheet: no face"))
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save_svg(fig, "labels-level-links")
    pcs.update_layout(title="Level labels (below | above | sheet) of the running example, obtained by linking to the slab cells",
                      width=1300, height=640)
    for i in range(1, 3):
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


def raw_class(layer, prisms, raw, fi: int) -> str:
    hits = covering(layer, prisms, raw, fi)
    names = []
    for h in hits:
        p = prisms[h]
        if p.entity_id == "pad":
            names.append("pad" if p.polygon == 0 else "tab")
        elif p.entity_id == "ground_0000":
            names.append("ground")
    names = sorted(set(names))
    return "+".join(names) if names else "vacuum only"


def figure_dissolution(model, numbers: dict) -> None:
    k = slab_index(model, METAL_SLAB)
    layer = model.labelled.slabs[k]
    raw, raw_labels = layer.raw_arrangement, layer.raw_labels
    no_curves = [None] * len(raw.edges)
    prisms = model.labelled.catalogue.prisms
    required, seams = classify_raw_edges(raw, raw_labels)
    stats = layer.stats

    cls_colour = {"vacuum only": PALETTE["vacuum"], "ground": PALETTE["ground"], "pad": "#7aa3cf", "tab": "#e7b24a", "pad+tab": "#c46a28"}

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
    pcs = make_subplots(rows=1, cols=3, horizontal_spacing=0.03, subplot_titles=(
        f"A. noded cells: {stats['raw_faces']} cells, {stats['edges_before']} edges",
        "B. slab labels: equal labels across red seams",
        f"C. dissolved: {stats['faces_after']} cells, {stats['edges_after']} edges"))
    for fi in range(1, len(raw.faces)):
        cls = raw_class(layer, prisms, raw, fi)
        rings = cell_rings_um(layer, fi, arr=raw, curves=no_curves)
        axes[0].add_patch(PathPatch(mpl_path(rings), facecolor=cls_colour.get(cls, "#ddd"), edgecolor="none"))
        pcs.add_shape(type="path", path=svg_path(rings), fillcolor=cls_colour.get(cls, "#ddd"), fillrule="evenodd", line_width=0,
                      layer="below", row=1, col=1)
        lab = raw_labels[fi]
        axes[1].add_patch(PathPatch(mpl_path(rings), facecolor=domain_colour(lab), edgecolor="none"))
        pcs.add_shape(type="path", path=svg_path(rings), fillcolor=domain_colour(lab), fillrule="evenodd", line_width=0,
                      layer="below", row=1, col=2)
    for col, ax in ((1, axes[0]), (2, axes[1])):
        for group, colour, width, name in ((required, PALETTE["ink"], 0.9, "labels differ"), (seams, PALETTE["seam"], 2.4, "seam: same label both sides")):
            xs, ys = [], []
            for ei in group:
                pts = edge_polyline_um(raw, no_curves, ei)
                ax.plot(*zip(*pts), color=colour, lw=width, solid_capstyle="round")
                xs += [p[0] for p in pts] + [None]
                ys += [p[1] for p in pts] + [None]
            pcs.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=colour, width=max(1, width * 1.3)), name=name,
                                     showlegend=col == 1, hoverinfo="skip"), row=1, col=col)
    for fi in range(1, len(layer.arr.faces)):
        rings = cell_rings_um(layer, fi)
        axes[2].add_patch(PathPatch(mpl_path(rings), facecolor=domain_colour(layer.labels[fi]), edgecolor="none"))
        pcs.add_shape(type="path", path=svg_path(rings), fillcolor=domain_colour(layer.labels[fi]), fillrule="evenodd", line_width=0,
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
    handles = [Patch(facecolor=cls_colour[n], edgecolor="#555", label=n) for n in ("vacuum only", "ground", "pad", "tab", "pad+tab")]
    handles += [Line2D([0], [0], color=PALETTE["seam"], lw=2.5, label="seam (dissolved)"),
                Line2D([0], [0], color=PALETTE["ink"], lw=1.2, label="required (labels differ)")]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    save_svg(fig, "labels-dissolution")
    for col in (1, 2, 3):
        pcs.update_xaxes(range=list(VIEW[0]), row=1, col=col)
        pcs.update_yaxes(range=list(VIEW[1]), row=1, col=col, scaleanchor=f"x{'' if col == 1 else col}")
    pcs.update_layout(title=f"{layer.plane} of the running example with a second polygon of the pad, before and after dissolution",
                      width=1500, height=560)
    save_plotly(pcs, "labels-dissolution")

    covered: Counter = Counter()
    for fi in range(1, len(raw.faces)):
        area = raw.face_twice_area(fi) / 2 * G_UM**2
        for h in covering(layer, prisms, raw, fi):
            if prisms[h].entity_id == "pad":
                covered[cover_name(prisms[h])] += area
    pad_face = next(fi for fi in range(1, len(layer.arr.faces)) if layer.labels[fi].kind == "conductor" and layer.labels[fi].name == "pad")
    face_area = face_exact_area(layer.arr, layer.edge_curve, pad_face) * G_UM**2
    numbers["dissolution"] = {
        "plane": layer.plane, "raw_cells": stats["raw_faces"], "cells_after": stats["faces_after"],
        "raw_edges": stats["edges_before"], "edges_after": stats["edges_after"],
        "raw_edges_labels_differ": len(required), "raw_edges_seam": len(seams),
        "edges_merged_by_degree_two_removal": len(required) - stats["edges_after"],
        "merged_pad_cell_area_um2": face_area,
        "polygon_areas_um2": dict(covered), "polygon_fraction_of_merged_cell": {n: a / face_area for n, a in covered.items()},
        "membership_of_merged_cell": dict(layer.membership[pad_face]),
    }


# --------------------------------------------------------------------------- the other builds the page quotes


def per_entity_tab(catalogue, numbers: dict) -> None:
    """The same tab as its own entity of the same Net: production keeps the seam and the higher priority owns the overlap."""
    model = build_model(with_tab(catalogue, same_entity=False, priority=1), "B")
    layer = model.labelled.slabs[slab_index(model, METAL_SLAB)]
    top = next(f for f in model.topology.faces.values() if model.routed.face_group[f.id] == "MA::pad::top")
    tab_top = next(f for f in model.topology.faces.values() if model.routed.face_group[f.id] == "MA::pad_tab::top")
    numbers["tab_as_second_entity"] = {
        "plane": layer.plane, "cells_after": layer.stats["faces_after"], "edges_after": layer.stats["edges_after"],
        "labels": sorted(domain_name(l) for l in layer.labels[1:]),
        "pad_top_membership": dict(top.membership), "pad_tab_top_membership": dict(tab_top.membership),
        "groups": sorted(g for g in model.routed.groups if "::pad" in g),
    }


def stage_scene_chords(numbers: dict) -> None:
    """The stage-level scene with true arcs: how many noded chord edges dissolution merges back (mid-plane port scene)."""
    model = build_model(stage_pad_in_ground("B"), "B")
    out = {}
    for layer in (*model.labelled.slabs, *model.labelled.level_layers):
        if layer.plane in ("slab[0,1/10]", "level[1/10]"):
            out[layer.plane] = {"edges_noded": layer.stats["edges_before"], "edges_after": layer.stats["edges_after"]}
    numbers["stage_scene_arcs"] = out


def group_table(model, numbers: dict) -> None:
    faces = {n: len(g.ids) for n, g in model.routed.groups.items() if g.dim == 2}
    numbers["route_b_groups"] = {"faces_per_group": faces, "volumes": sorted(n for n, g in model.routed.groups.items() if g.dim == 3)}
    pad_top = next(f for f in model.topology.faces.values() if model.routed.face_group[f.id] == "MA::pad::top")
    pad_walls = [f for f in model.topology.faces.values() if model.routed.face_group[f.id] == "MA::pad::sidewall"]
    numbers["membership_record"] = {
        "top_face_group": "MA::pad::top", "top_face_membership": dict(pad_top.membership),
        "pad_sidewalls": sorted([dict(w.membership) for w in pad_walls], key=lambda m: sorted(m.items())),
    }


def error_messages(catalogue, numbers: dict) -> None:
    box = footprint(rect_um(35.0, 45.0, 45.0, 55.0))  # overlaps the pad's left end and the gap
    out = {}

    def attempt(key, prism_records, exception):
        try:
            build_model(replace(catalogue, prisms=(*catalogue.prisms, *prism_records)), "B")
            out[key] = "built"
        except exception as e:
            out[key] = str(e)

    attempt("different_net_overlap", [PrismRecord("other_net", box, exact_z(0), exact_z("0.2"), "Al", "conductor", net_id="other")], OverlapError)
    attempt("equal_priority_dielectrics", [PrismRecord("alumina", box, exact_z(-10), exact_z(0), "alumina", "dielectric")], OverlapError)
    attempt("different_net_contact", [PrismRecord("touching_net", footprint(rect_um(34.0, 45.0, 40.0, 55.0)), exact_z(0), exact_z("0.2"),
                                                  "Al", "conductor", net_id="other")], ContactError)
    pad = pad_prism(catalogue)
    attempt("same_net_equal_priority", [replace(pad, entity_id="pad_tab", footprint=footprint(rect_um(44.0, 50.0, 56.0, 66.0)), polygon=0)], OverlapError)
    ok = build_model(replace(catalogue, prisms=(*catalogue.prisms,
                     PrismRecord("alumina", box, exact_z(-10), exact_z(0), "alumina", "dielectric", priority=1))), "B")
    out["priority_resolves"] = {"volumes": sorted(v.domain.name for v in ok.topology.volumes.values()), "validation_passed": not ok.report.failures}
    numbers["errors"] = out


def main() -> None:
    catalogue = real_catalogue()
    model = build_model(catalogue, "B")
    tab_model = build_model(with_tab(catalogue), "B", keep_raw=True)
    numbers: dict = {"scene": "gds_pad_in_ground (production real input), Route B", "tab_scene": "the same catalogue + a second polygon of the entity pad (x 44..56, y 50..66 um, z 0..0.2)"}
    figure_slab_cells(model, numbers)
    figure_level_links(model, numbers)
    figure_dissolution(tab_model, numbers)
    per_entity_tab(catalogue, numbers)
    stage_scene_chords(numbers)
    group_table(model, numbers)
    error_messages(catalogue, numbers)
    numbers["levels"] = [str(z) for z in model.labelled.levels.levels]
    numbers["plain_dissolution_stats"] = {s.plane: {"raw_cells": s.stats["raw_faces"], "cells_after": s.stats["faces_after"],
                                                    "raw_edges": s.stats["edges_before"], "edges_after": s.stats["edges_after"]}
                                          for s in (*model.labelled.slabs, *model.labelled.level_layers)}
    write_numbers("labels", numbers)
    print("labels: done")


if __name__ == "__main__":
    main()
