"""Figures and numbers of ``cut-families.qmd``: several curve families on one level of the running example.

The scene is ``pad_in_ground("A", bands_um=(0.1, 0.5, 2.0))`` (Route A, Surface-EPR inset bands on the pad top and on the gap)
plus two families written for this page and added without touching any pipeline code:

* ``window``  -- a closed rectangle in the gap, edge policy ``required`` (a user partition: the cell is split, the group is not);
* ``refine``  -- one open line across gap and pad, edge policy ``refinement_only`` (kept as an embedded curve, no label change).

Writes into ``docs/design/sgb/figures/``:

* ``families-level.svg/.html``  level z = 0 with every edge coloured by the family that cut it, cells by primary group;
* ``families-numbers.json``     ablation counts (what each family adds), band areas with and without the window, order
  independence of the build identity, and the family record fields.

Run (from the package root):  .venv/bin/python docs/design/sgb/_figures/families.py
"""

from __future__ import annotations

from fractions import Fraction

from labels_common import (
    G_UM, PALETTE, cell_rings_um, edge_polyline_um, finish_plan, go, mpl_path, plt, save_plotly, save_svg, svg_path,
    write_numbers,
)
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Patch
from plotly.subplots import make_subplots

from sgbproto.arcs import face_exact_area
from sgbproto.pipeline import build, ids_digest
from sgbproto.records import CurveFamilyRecord, LineCarrier, Lineage, ModelSpec
from sgbproto.scenes import pad_in_ground

MARGINS = (0.1, 0.5, 2.0)
VIEW = ((-86.0, 86.0), (-66.0, 66.0))


def um(x: float) -> int:
    return round(x * 1000)


def line_family(family_id: str, role: str, points_um, closed: bool, policy: str) -> CurveFamilyRecord:
    pts = [(um(x), um(y)) for x, y in points_um]
    pairs = list(zip(pts, pts[1:])) + ([(pts[-1], pts[0])] if closed else [])
    carriers = tuple(
        LineCarrier(f"{family_id}/{i}", p, q, Lineage("family_member", family_id=family_id, member=f"{family_id}/{i}"))
        for i, (p, q) in enumerate(pairs)
    )
    return CurveFamilyRecord(family_id, role, carriers, (Fraction(0),), policy)


def window_family() -> CurveFamilyRecord:
    return line_family("window", "partition", [(-79.0, -20.0), (-60.0, -20.0), (-60.0, 20.0), (-79.0, 20.0)], True, "required")


def refine_family() -> CurveFamilyRecord:
    return line_family("refine", "refinement", [(-70.0, -8.0), (70.0, -8.0)], False, "refinement_only")


def scene(window: bool, refine: bool, bands: bool, reverse: bool = False) -> ModelSpec:
    base = pad_in_ground("A", bands_um=MARGINS if bands else None)
    fams = list(base.families)
    if window:
        fams.append(window_family())
    if refine:
        fams.append(refine_family())
    if reverse:
        fams.reverse()
    return ModelSpec(base.grid, base.prisms, tuple(fams), name=base.name, meta=base.meta)


def family_of(sources) -> str:
    """Name of the family (or source entity) an edge's carriers belong to."""
    names = {s.split("/")[0] for s in sources}
    for tag in ("window", "band_pad_top", "band_gap", "refine"):
        if tag in names:
            return tag
    if "p1" in names:
        return "port"
    return "physical boundary"


EDGE_STYLE = {  # family -> (colour, width, label)
    "physical boundary": (PALETTE["ink"], 1.6, "physical boundary (prism footprints)"),
    "port": (PALETTE["port"], 2.0, "port footprint (sheet of the port prism)"),
    "band_pad_top": (PALETTE["band"], 0.9, "inset bands of the pad top"),
    "band_gap": ("#e6550d", 0.9, "inset bands of the gap"),
    "window": (PALETTE["cap"], 2.2, "window family (required)"),
    "refine": (PALETTE["refine"], 1.6, "refinement curve (refinement_only, embedded)"),
}
GROUP_FILL = {"PEC::Ground": "#9a9a9a", "PORT::p1": "#f3b4ae"}


def cell_colour(result, layer, fi: int) -> str:
    face = result.topology.faces.get(layer.face_ids[fi])
    group = face.group if face else None
    if group is None:
        return "#f5f5f5"
    if group in GROUP_FILL:
        return GROUP_FILL[group]
    if "::band=" in group:
        k = int(group.rsplit("=", 1)[1])
        ramp = ["#08306b", "#2171b5", "#6baed6", "#c6dbef"] if group.startswith("SHEET") else ["#7f2704", "#d94801", "#fd8d3c", "#fdd0a2"]
        return ramp[k]
    return "#dddddd"


def band_areas(result) -> dict:
    """Exact area (um^2) per (family, band) over the cells of the level z = 0."""
    layer = result.level_layers[1]
    out: dict[str, float] = {}
    for fi in range(1, len(layer.arr.faces)):
        for fam, k in layer.labels[fi].bands:
            key = f"{fam}:band={k}"
            out[key] = out.get(key, 0.0) + face_exact_area(layer.arr, layer.edge_curve, fi) * G_UM**2
    return out


def figure_level(result, numbers: dict) -> None:
    layer = result.level_layers[1]
    arr = layer.arr
    fig, ax = plt.subplots(figsize=(10.5, 7.4))
    pcs = make_subplots(rows=1, cols=1)
    for fi in range(1, len(arr.faces)):
        rings = cell_rings_um(layer, fi)
        colour = cell_colour(result, layer, fi)
        ax.add_patch(PathPatch(mpl_path(rings), facecolor=colour, edgecolor="none"))
        pcs.add_shape(type="path", path=svg_path(rings), fillcolor=colour, fillrule="evenodd", line_width=0, layer="below")
    used = {}
    by_family: dict[str, list] = {}
    for ei in range(len(arr.edges)):
        fam = family_of(layer.edge_sources[ei])
        by_family.setdefault(fam, []).append(edge_polyline_um(arr, layer.edge_curve, ei))
    for fam, polys in by_family.items():
        colour, width, label = EDGE_STYLE[fam]
        used[fam] = (colour, label)
        xs, ys = [], []
        for poly in polys:
            ax.plot(*zip(*poly), color=colour, lw=width, solid_capstyle="round")
            xs += [p[0] for p in poly] + [None]
            ys += [p[1] for p in poly] + [None]
        pcs.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=colour, width=max(1, width * 1.5)), name=label))
    refinement = layer.dissolved.refinement
    xs, ys = [], []
    for edge, _face in refinement:
        ax.plot([edge.a[0] * G_UM, edge.b[0] * G_UM], [edge.a[1] * G_UM, edge.b[1] * G_UM], color=EDGE_STYLE["refine"][0], lw=1.8, ls=(0, (5, 3)))
        xs += [edge.a[0] * G_UM, edge.b[0] * G_UM, None]
        ys += [edge.a[1] * G_UM, edge.b[1] * G_UM, None]
    used["refine"] = (EDGE_STYLE["refine"][0], EDGE_STYLE["refine"][2])
    pcs.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=EDGE_STYLE["refine"][0], width=3, dash="dash"), name=EDGE_STYLE["refine"][2]))
    finish_plan(ax, *VIEW, "Level z = 0 of pad_in_ground A with ports, bands, a required window and a refinement curve")
    handles = [Line2D([0], [0], color=c, lw=2.5, ls="--" if k == "refine" else "-", label=l) for k, (c, l) in used.items()]
    handles += [Patch(facecolor="#6baed6", label="cells: pad bands (blue ramp, dark = edge)"),
                Patch(facecolor="#fd8d3c", label="cells: gap bands (orange ramp)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=8, frameon=False)
    save_svg(fig, "families-level")
    pcs.update_xaxes(range=list(VIEW[0]), title="x (µm)")
    pcs.update_yaxes(range=list(VIEW[1]), scaleanchor="x", title="y (µm)")
    pcs.update_layout(title="Level z = 0: every edge coloured by the family that cut it", width=1100, height=800)
    save_plotly(pcs, "families-level")
    numbers["level_edges_by_family"] = {fam: len(polys) for fam, polys in by_family.items()}
    numbers["level_refinement_pieces"] = len(refinement)


def counts(result) -> dict:
    layer = result.level_layers[1]
    topo = result.topology
    groups = {n: len(g["ids"]) for n, g in topo.groups.items() if g["dim"] == 2}
    return {
        "level_cells": len(layer.arr.faces) - 1, "level_edges": len(layer.arr.edges),
        "level_noded_edges": layer.stats["edges_before"], "level_hot_pixels": layer.stats["hot_pixels"],
        "refinement_pieces": len(layer.dissolved.refinement),
        "topology": {k: result.counts[k] for k in ("vertices", "edges", "faces")},
        "face_groups": len(groups), "faces_in_SA_groups": sum(n for g, n in groups.items() if g.startswith("SA")),
        "faces_in_pad_sheet_band_groups": sum(n for g, n in groups.items() if g.startswith("SHEET::pad")),
        "validation_passed": not result.report.failures,
    }


def refinement_without_face(numbers: dict) -> None:
    """A refinement curve on a level where its cell has no face stops the build (Route B, port plane z = 0.1)."""
    from sgbproto.errors import SgbError
    from sgbproto.scenes import pad_in_ground as scene_b

    base = scene_b("B")
    carrier = LineCarrier("r/0", (-70000, -8000), (-60000, -8000), Lineage("family_member", family_id="r", member="r/0"))
    family = CurveFamilyRecord("r", "refinement", (carrier,), (Fraction(1, 10),), "refinement_only")
    try:
        build(ModelSpec(base.grid, base.prisms, (family,), name="refine_no_face", meta=base.meta))
        numbers["refinement_in_faceless_cell"] = "built"
    except SgbError as e:
        numbers["refinement_in_faceless_cell"] = {"error": type(e).__name__, "message": str(e)}
    try:
        CurveFamilyRecord("t", "tile_seam", (carrier,), (Fraction(0),), "dissolve")
        numbers["dissolve_policy"] = "accepted"
    except Exception as e:  # the policy was removed: tile seams are not a family
        numbers["dissolve_policy"] = {"error": type(e).__name__, "message": str(e)}


def main() -> None:
    numbers: dict = {"scene": "pad_in_ground('A', bands (0.1, 0.5, 2.0) um) + window + refine"}
    full = build(scene(True, True, True))
    figure_level(full, numbers)
    numbers["ablation"] = {
        "port_and_physical_boundaries_only": counts(build(scene(False, False, False))),
        "plus_bands": counts(build(scene(False, False, True))),
        "plus_bands_and_window": counts(build(scene(True, False, True))),
        "plus_bands_and_refine": counts(build(scene(False, True, True))),
        "all_families": counts(full),
    }
    # distance-origin rule: the window and the refinement curve never move a band boundary
    with_all, bands_only = band_areas(full), band_areas(build(scene(False, False, True)))
    numbers["band_areas_um2"] = {k: {"bands_only": bands_only[k], "with_window_and_refine": with_all[k]} for k in sorted(bands_only)}
    numbers["max_band_area_difference_um2"] = max(abs(with_all[k] - bands_only[k]) for k in bands_only)
    # order independence: reversing the family list gives the same identity
    numbers["identity_digest_equal_when_family_order_reversed"] = ids_digest(full) == ids_digest(build(scene(True, True, True, reverse=True)))
    numbers["family_record_fields"] = ["family_id", "role", "carriers", "target_levels", "edge_policy", "arcs", "support"]
    numbers["families_of_scene"] = [
        {"family_id": f.family_id, "role": f.role, "edge_policy": f.edge_policy, "target_levels": [str(z) for z in f.target_levels],
         "carriers": len(f.carriers), "arcs": len(f.arcs), "has_support_region": f.support is not None}
        for f in full.spec.families
    ]
    refinement_without_face(numbers)
    write_numbers("families", numbers)
    print("families: done")


if __name__ == "__main__":
    main()
