"""Figures and numbers of ``cut-families.qmd``: several curve families on one level of the running example (production).

The scene is the production real input ``gds_pad_in_ground`` interpreted with the ``thin_film`` profile (Route A: ground and
pad are sheets at z = 0, the port lies on the same plane) with the typed requests

* ``InsetRequest`` bands of margins 0.1, 0.5 and 2 um on the pad sheet and on the substrate-vacuum gap,
* ``RefinementRequest`` one open line across gap and pad, edge policy ``refinement_only`` (kept as an embedded curve),

plus one family written for this page and added without touching any pipeline code (the request API has no such family):

* ``window``  -- a closed rectangle in the gap, edge policy ``required`` (a user partition: the cell is split, the group is not).

Writes into ``docs/design/sgb/figures/``:

* ``families-level.svg/.html``  level z = 0 with every edge coloured by the family that cut it, cells by primary group;
* ``families-numbers.json``     ablation counts (what each family adds), band areas with and without the extra families, order
  independence of the build identity, the family record fields, and the two errors the page quotes.

Run:  SGB_CORE=<sgb-core checkout> <sgb-core>/.artifacts/env/bin/python docs/design/sgb/_figures/families.py
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

from labels_common import (
    G_UM,
    PALETTE,
    build_model,
    cell_rings_um,
    edge_polyline_um,
    finish_plan,
    go,
    mpl_path,
    plt,
    real_catalogue,
    save_plotly,
    save_svg,
    svg_path,
    with_extra_families,
    write_numbers,
)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch
from plotly.subplots import make_subplots

from scgsim.geometry.sgb.arrangement.arcs import face_exact_area
from scgsim.geometry.sgb.cutfamilies.generate import with_cut_families
from scgsim.geometry.sgb.cutfamilies.requests import (
    ConductorSurface,
    InsetRequest,
    RefinementRequest,
    SolutionInterface,
)
from scgsim.geometry.sgb.errors import SgbError
from scgsim.geometry.sgb.model import build_model_from_catalogue
from scgsim.geometry.sgb.records.carriers import Lineage, LineCarrier
from scgsim.geometry.sgb.records.families import CurveFamilyRecord

MARGINS = (0.1, 0.5, 2.0)
VIEW = ((26.0, 74.0), (26.0, 74.0))
PAD_SHEET = "pad"
GAP = "SA::substrate|VACUUM_REGION::interface"


def _um(x: float) -> int:
    return round(x * 1000)


def _line_family(
    family_id: str, role: str, points_um, closed: bool, policy: str
) -> CurveFamilyRecord:
    pts = [(_um(x), _um(y)) for x, y in points_um]
    pairs = list(zip(pts, pts[1:], strict=False)) + (
        [(pts[-1], pts[0])] if closed else []
    )
    carriers = tuple(
        LineCarrier(
            f"{family_id}/{i}",
            p,
            q,
            Lineage("family_member", family_id=family_id, member=f"{family_id}/{i}"),
        )
        for i, (p, q) in enumerate(pairs)
    )
    return CurveFamilyRecord(family_id, role, carriers, (Fraction(0),), policy)


def _window_family() -> CurveFamilyRecord:
    return _line_family(
        "window",
        "partition",
        [(31.0, 55.0), (37.0, 55.0), (37.0, 65.0), (31.0, 65.0)],
        True,
        "required",
    )


def _requests(bands: bool, refine: bool):
    insets = (
        (
            InsetRequest("band_pad", ConductorSurface("pad", "sheet"), MARGINS),
            InsetRequest(
                "band_gap",
                SolutionInterface("substrate", "VACUUM_REGION", z_um=0.0),
                MARGINS,
            ),
        )
        if bands
        else ()
    )
    refinements = (
        (RefinementRequest("refine", 0.0, ((32.0, 45.0), (50.0, 45.0))),)
        if refine
        else ()
    )
    return insets, refinements


def scene(catalogue, window: bool, refine: bool, bands: bool, reverse: bool = False):
    """The Route A model with the chosen families; ``reverse`` lists the families in the opposite order."""
    insets, refinements = _requests(bands, refine)
    cat = with_extra_families(catalogue, _window_family()) if window else catalogue
    if not reverse:
        return build_model(cat, "A", insets=insets, refinements=refinements)
    full = with_cut_families(cat, insets, refinements)
    return build_model_from_catalogue(
        replace(full, families=tuple(reversed(full.families))), "A"
    )


def _level0(model):
    return model.labelled.level_layers[model.labelled.levels.levels.index(Fraction(0))]


def family_of(sources) -> str:
    """Name of the family (or source entity) an edge's carriers belong to."""
    names = {s.split("/")[0] for s in sources}
    for tag in ("window", "band_pad", "band_gap", "refine"):
        if tag in names:
            return tag
    if any(n.startswith("PORT_SHEET") for n in names):
        return "port"
    return "physical boundary"


EDGE_STYLE = {  # family -> (colour, width, label)
    "physical boundary": (PALETTE["ink"], 1.6, "physical boundary (prism footprints)"),
    "port": (PALETTE["port"], 2.0, "port footprint (sheet of the port prism)"),
    "band_pad": (PALETTE["band"], 0.9, "inset bands of the pad sheet"),
    "band_gap": ("#e6550d", 0.9, "inset bands of the gap"),
    "window": (PALETTE["cap"], 2.2, "window family (required)"),
    "refine": (PALETTE["refine"], 1.6, "refinement curve (refinement_only, embedded)"),
}
GROUP_FILL = {"ground_0000": "#9a9a9a", "PORT::PORT_SHEET__P1__0000": "#f3b4ae"}


def _cell_colour(model, layer, fi: int) -> str:
    face = model.topology.faces.get(layer.face_ids[fi])
    group = model.routed.face_group[face.id] if face else None
    if group is None:
        return "#f5f5f5"
    if group in GROUP_FILL:
        return GROUP_FILL[group]
    if "::band=" in group:
        k = int(group.rsplit("=", 1)[1])
        ramp = (
            ["#08306b", "#2171b5", "#6baed6", "#c6dbef"]
            if group.startswith(PAD_SHEET)
            else ["#7f2704", "#d94801", "#fd8d3c", "#fdd0a2"]
        )
        return ramp[k]
    return "#dddddd"


def band_areas(model) -> dict:
    """Exact area (um^2) per (family, band) over the cells of the level z = 0 that the band groups carry (a port face is no band's surface)."""
    layer = _level0(model)
    out: dict[str, float] = {}
    for fi in range(1, len(layer.arr.faces)):
        sheet = layer.labels[fi].sheet
        if sheet is not None and sheet.kind == "port":
            continue
        for fam, k in layer.labels[fi].bands:
            key = f"{fam}:band={k}"
            out[key] = (
                out.get(key, 0.0)
                + face_exact_area(layer.arr, layer.edge_curve, fi) * G_UM**2
            )
    return out


def _figure_level(model, numbers: dict) -> None:
    layer = _level0(model)
    arr = layer.arr
    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    pcs = make_subplots(rows=1, cols=1)
    for fi in range(1, len(arr.faces)):
        rings = cell_rings_um(layer, fi)
        colour = _cell_colour(model, layer, fi)
        ax.add_patch(PathPatch(mpl_path(rings), facecolor=colour, edgecolor="none"))
        pcs.add_shape(
            type="path",
            path=svg_path(rings),
            fillcolor=colour,
            fillrule="evenodd",
            line_width=0,
            layer="below",
        )
    used = {}
    by_family: dict[str, list] = {}
    for ei in range(len(arr.edges)):
        by_family.setdefault(family_of(layer.edge_sources[ei]), []).append(
            edge_polyline_um(arr, layer.edge_curve, ei)
        )
    for fam, polys in by_family.items():
        colour, width, label = EDGE_STYLE[fam]
        used[fam] = (colour, label)
        xs, ys = [], []
        for poly in polys:
            ax.plot(
                *zip(*poly, strict=True),
                color=colour,
                lw=width,
                solid_capstyle="round",
            )
            xs += [p[0] for p in poly] + [None]
            ys += [p[1] for p in poly] + [None]
        pcs.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color=colour, width=max(1, width * 1.5)),
                name=label,
            )
        )
    refinement = layer.dissolved.refinement
    xs, ys = [], []
    for edge, _face in refinement:
        ax.plot(
            [edge.a[0] * G_UM, edge.b[0] * G_UM],
            [edge.a[1] * G_UM, edge.b[1] * G_UM],
            color=EDGE_STYLE["refine"][0],
            lw=1.8,
            ls=(0, (5, 3)),
        )
        xs += [edge.a[0] * G_UM, edge.b[0] * G_UM, None]
        ys += [edge.a[1] * G_UM, edge.b[1] * G_UM, None]
    used["refine"] = (EDGE_STYLE["refine"][0], EDGE_STYLE["refine"][2])
    pcs.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color=EDGE_STYLE["refine"][0], width=3, dash="dash"),
            name=EDGE_STYLE["refine"][2],
        )
    )
    finish_plan(
        ax,
        *VIEW,
        "Level z = 0 of the running example (Route A) with port, bands, a required window and a refinement curve",
    )
    handles = [
        Line2D([0], [0], color=c, lw=2.5, ls="--" if k == "refine" else "-", label=name)
        for k, (c, name) in used.items()
    ]
    handles += [
        Patch(facecolor="#6baed6", label="cells: pad bands (blue ramp, dark = edge)"),
        Patch(facecolor="#fd8d3c", label="cells: gap bands (orange ramp)"),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.09),
        ncol=2,
        fontsize=8,
        frameon=False,
    )
    save_svg(fig, "families-level")
    pcs.update_xaxes(range=list(VIEW[0]), title="x (µm)")
    pcs.update_yaxes(range=list(VIEW[1]), scaleanchor="x", title="y (µm)")
    pcs.update_layout(
        title="Level z = 0: every edge coloured by the family that cut it",
        width=1100,
        height=800,
    )
    save_plotly(pcs, "families-level")
    numbers["level_edges_by_family"] = {
        fam: len(polys) for fam, polys in by_family.items()
    }
    numbers["level_refinement_pieces"] = len(refinement)


def counts(model) -> dict:
    """Counts of the level-0 arrangement, the face groups and the topology of a built model."""
    layer = _level0(model)
    groups = {n: len(g.ids) for n, g in model.routed.groups.items() if g.dim == 2}
    return {
        "level_cells": len(layer.arr.faces) - 1,
        "level_edges": len(layer.arr.edges),
        "level_noded_edges": layer.stats["edges_before"],
        "level_hot_pixels": layer.stats["hot_pixels"],
        "refinement_pieces": len(layer.dissolved.refinement),
        "topology": {k: model.counts[k] for k in ("vertices", "edges", "faces")},
        "face_groups": len(groups),
        "faces_in_SA_groups": sum(n for g, n in groups.items() if g.startswith("SA")),
        "faces_in_pad_sheet_band_groups": sum(
            n for g, n in groups.items() if g.startswith(f"{PAD_SHEET}::band=")
        ),
        "validation_passed": not model.report.failures,
    }


def port_invariance(numbers: dict) -> None:
    """The port moves no band but takes its footprint out of the gap bands: the exact area of every band label with and without the port."""
    insets, _ = _requests(True, False)
    with_port = band_areas(build_model(real_catalogue(film=True), "A", insets=insets))
    without = band_areas(
        build_model(real_catalogue(film=True, with_port=False), "A", insets=insets)
    )
    gap = [k for k in without if k.startswith("band_gap:")]
    numbers["port_invariance"] = {
        "band_areas_um2_with_port": with_port,
        "band_areas_um2_without_port": without,
        "largest_band_area_increase_um2": max(
            with_port[k] - without[k] for k in without
        ),
        "gap_area_removed_um2": sum(without[k] - with_port[k] for k in gap),
        "other_bands_max_abs_difference_um2": max(
            (abs(with_port[k] - without[k]) for k in without if k not in gap),
            default=0.0,
        ),
    }


def errors(numbers: dict) -> None:
    """A refinement curve on a level where its cell has no face stops the build (Route B, metal top z = 0.2); no ``dissolve`` policy."""
    catalogue = real_catalogue()
    request = RefinementRequest("r", 0.2, ((32.0, 55.0), (38.0, 55.0)))
    try:
        build_model(catalogue, "B", refinements=(request,))
        numbers["refinement_in_faceless_cell"] = "built"
    except SgbError as e:
        numbers["refinement_in_faceless_cell"] = {
            "error": type(e).__name__,
            "message": str(e),
        }
    carrier = LineCarrier(
        "r/0",
        (-70000, -8000),
        (-60000, -8000),
        Lineage("family_member", family_id="r", member="r/0"),
    )
    try:
        CurveFamilyRecord("t", "tile_seam", (carrier,), (Fraction(0),), "dissolve")
        numbers["dissolve_policy"] = "accepted"
    except ValueError as e:  # the policy was removed: tile seams are not a family
        numbers["dissolve_policy"] = {"error": type(e).__name__, "message": str(e)}


def main() -> None:
    """Write the cut-family figure and its numbers."""
    catalogue = real_catalogue(film=True)
    numbers: dict = {
        "scene": "gds_pad_in_ground with the thin_film profile (Route A) + bands (0.1, 0.5, 2.0 um) + window + refine"
    }
    full = scene(catalogue, True, True, True)
    _figure_level(full, numbers)
    bands_only = scene(catalogue, False, False, True)
    numbers["ablation"] = {
        "port_and_physical_boundaries_only": counts(
            scene(catalogue, False, False, False)
        ),
        "plus_bands": counts(bands_only),
        "plus_bands_and_window": counts(scene(catalogue, True, False, True)),
        "plus_bands_and_refine": counts(scene(catalogue, False, True, True)),
        "all_families": counts(full),
    }
    # distance-origin rule: the window and the refinement curve never move a band boundary
    with_all, only = band_areas(full), band_areas(bands_only)
    numbers["band_areas_um2"] = {
        k: {"bands_only": only[k], "with_window_and_refine": with_all[k]}
        for k in sorted(only)
    }
    numbers["max_band_area_difference_um2"] = max(
        abs(with_all[k] - only[k]) for k in only
    )
    # order independence: reversing the family list gives the same identity
    numbers["identity_digest_equal_when_family_order_reversed"] = (
        full.identity_digest()
        == scene(catalogue, True, True, True, reverse=True).identity_digest()
    )
    numbers["family_record_fields"] = [
        "family_id",
        "role",
        "carriers",
        "target_levels",
        "edge_policy",
        "arcs",
        "support",
    ]
    numbers["families_of_scene"] = [
        {
            "family_id": f.family_id,
            "role": f.role,
            "edge_policy": f.edge_policy,
            "target_levels": [str(z) for z in f.target_levels],
            "carriers": len(f.carriers),
            "arcs": len(f.arcs),
            "has_support_region": f.support is not None,
        }
        for f in full.labelled.catalogue.families
    ]
    numbers["groups_with_all_families"] = {
        n: len(g.ids) for n, g in full.routed.groups.items() if g.dim == 2
    }
    numbers["groups_bands_only"] = {
        n: len(g.ids) for n, g in bands_only.routed.groups.items() if g.dim == 2
    }
    port_invariance(numbers)
    errors(numbers)
    write_numbers("families", numbers)
    print("families: done")


if __name__ == "__main__":
    main()
