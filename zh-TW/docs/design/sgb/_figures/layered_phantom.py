"""Figure of layered-model.qmd: a phantom sliver from a global projection, absent from the local arrangements.

A two-metal model with unrelated layers: M1 at z = 0 to 0.2 um and M2 at z = 3 to 3.2 um, separated by vacuum. The
right edge of M2 lies 3 nm beside the right edge of M1 when both are projected on the XY plane. The script builds

* the *local* arrangements with the production pipeline (``build_model_from_catalogue``), where every level arrangement
  only receives the curves of the slabs that touch it, and
* a *global* arrangement: one snap-rounded arrangement of every prism footprint curve (what a single projection
  of all layers would give), which every level would then have to use.

Writes ``figures/layered-phantom.html`` and ``.svg`` and ``figures/layered-phantom-numbers.json``.
"""

from __future__ import annotations

import json
import math

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from intro_common import COLOURS, G_UM, mpl_path, save_plotly, save_svg, write_numbers
from matplotlib.patches import PathPatch
from sgb_scenes import GRID, chip_prisms, conductor, rect_um

from scgsim.geometry.sgb.arrangement.dcel import build_arrangement
from scgsim.geometry.sgb.model import build_model_from_catalogue
from scgsim.geometry.sgb.arrangement.noding.snap_rounding import Segment, snap_round
from scgsim.geometry.sgb.exact.prisms import PrismCatalogue

GAP_NM = 3  # lateral distance of the two edges: 3 grid units = 3 nm


def _two_layer_catalogue() -> PrismCatalogue:
    half = 80.0
    x_m2 = 40.0 + GAP_NM * GRID.size_um
    prisms = (
        *chip_prisms(GRID, half),
        conductor("m1", rect_um(-40, -30, 40, 30), 0, "0.2", "m1"),
        conductor("m2", rect_um(-20, -10, x_m2, 20), 3, "3.2", "m2"),
    )
    return PrismCatalogue(GRID, prisms, name="two_layer_phantom")


def global_arrangement(catalogue: PrismCatalogue):
    """One arrangement of every footprint curve of every prism, as a single projection of all layers would give."""
    segments = [
        Segment(c.p, c.q, frozenset([c.id]))
        for p in catalogue.prisms
        for c in p.carriers()
    ]
    snapped = snap_round(segments)
    return build_arrangement(snapped.edges), snapped


def face_widths(arr) -> list[tuple[int, float, float]]:
    """(face index, width, area) with width = 2 * area / perimeter in micrometres (the sliver report's definition)."""
    out = []
    for fi in range(1, len(arr.faces)):
        area = arr.face_twice_area(fi) / 2
        perimeter = sum(
            math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in arr.face_segments(fi)
        )
        out.append((fi, 2 * area / perimeter * G_UM, area * G_UM * G_UM))
    return out


def _rings_of(arr, fi):
    return [
        [(x * G_UM, y * G_UM) for x, y in arr.cycle_points(c)]
        for c in arr.face_cycles(fi)
    ]


def _draw_arr(ax, arr, highlight: set[int], view) -> None:
    for fi in range(1, len(arr.faces)):
        hot = fi in highlight
        ax.add_patch(
            PathPatch(
                mpl_path(_rings_of(arr, fi)),
                facecolor=COLOURS["bad"] if hot else "#e9eef5",
                edgecolor="#222",
                lw=0.8 if not hot else 0.3,
            )
        )
    ax.set_xlim(*view[0])
    ax.set_ylim(*view[1])
    ax.set_aspect("equal")


def main() -> None:
    """Write the phantom-sliver figure and its numbers."""
    spec = _two_layer_catalogue()
    result = build_model_from_catalogue(spec, "B").labelled
    arr, snapped = global_arrangement(spec)
    widths = face_widths(arr)
    narrowest = min(widths, key=lambda w: w[1])
    sliver = {fi for fi, w, _ in widths if w < 0.01}
    local, levels, local_cells = _local_level_stats(result)
    _phantom_svg(spec, result, arr, narrowest, sliver, levels, local_cells)
    _phantom_html(arr, widths, sliver)
    numbers = {
        "setup": {
            "edge_gap_nm": GAP_NM,
            "grid_nm": 1,
            "m1_um": "[-40, 40] x [-30, 30], z 0 to 0.2",
            "m2_um": "[-20, 40.003] x [-10, 20], z 3 to 3.2",
        },
        "global": {
            "cells": len(arr.faces) - 1,
            "edges": len(arr.edges),
            "hot_pixels": snapped.hot_pixels,
            "narrowest_width_nm": round(narrowest[1] * 1000, 3),
            "narrowest_area_um2": round(narrowest[2], 4),
            "sliver_cells": len(sliver),
            "sliver_length_um": 30.0,
        },
        "local": {
            "levels": levels,
            "cells_total": local_cells,
            "per_level": local,
            "narrowest_width_um": min(
                level["narrowest_width_um"]
                for level in local
                if level["narrowest_width_um"] is not None
            ),
        },
        "cells_if_every_level_used_the_global_arrangement": (len(arr.faces) - 1)
        * levels,
        "build_validated": True,  # build_model_from_catalogue raises on a failed invariant
    }
    write_numbers("layered-phantom", numbers)
    print(json.dumps(numbers, indent=1))


def _local_level_stats(result):
    """The cells and the narrowest cell of each level's own arrangement."""
    local = []
    for layer in result.level_layers:
        lw = face_widths(layer.arr) if len(layer.arr.faces) > 1 else []
        local.append(
            {
                "plane": f"z = {float(layer.level_z):g}",
                "cells": len(layer.arr.faces) - 1,
                "edges": len(layer.arr.edges),
                "narrowest_width_um": round(min(w for _, w, _ in lw), 4)
                if lw
                else None,
            }
        )
    local_cells = sum(level["cells"] for level in local)
    levels = len(result.level_layers)
    return local, levels, local_cells


def _phantom_svg(spec, result, arr, narrowest, sliver, levels, local_cells):  # noqa: PLR0913 -- the source, both arrangements, the sliver and the cell counts of one figure
    """The static figure: the source, the global projection with its sliver, the zoom and the cell counts."""
    fig = plt.figure(figsize=(13, 7.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
    a = fig.add_subplot(gs[0, 0])
    for p, colour in (
        (spec.prisms[2], COLOURS["pad"]),
        (spec.prisms[3], COLOURS["accent"]),
    ):
        pts = [(x * G_UM, y * G_UM) for x, y in p.footprint.outer]
        a.add_patch(
            plt.Polygon(pts, facecolor=colour, alpha=0.45, edgecolor="#222", lw=0.8)
        )
    a.annotate("M1: z = 0 to 0.2 µm", (-38, -26), fontsize=8)
    a.annotate("M2: z = 3 to 3.2 µm", (-18, 12), fontsize=8)
    a.add_patch(
        plt.Rectangle((38, -10), 5, 30, fill=False, edgecolor=COLOURS["bad"], lw=1.2)
    )
    a.annotate(
        f"right edges {GAP_NM} nm apart\n(unrelated layers)",
        (43, 4),
        xytext=(48, -30),
        fontsize=8,
        color=COLOURS["bad"],
        arrowprops=dict(arrowstyle="-", color=COLOURS["bad"]),
    )
    a.set_xlim(-60, 80)
    a.set_ylim(-45, 60)
    a.set_aspect("equal")
    a.set_title("Source: two prisms on different levels")
    a.set_xlabel("x (µm)")
    a.set_ylabel("y (µm)")

    b = fig.add_subplot(gs[0, 1])
    _draw_arr(b, arr, sliver, ((-60, 80), (-45, 60)))
    b.annotate(
        "sliver: too thin to see here",
        (40.0, 5),
        xytext=(46, -30),
        fontsize=8,
        color=COLOURS["bad"],
        arrowprops=dict(arrowstyle="->", color=COLOURS["bad"]),
    )
    b.set_title(f"Global projection: {len(arr.faces) - 1} cells; sliver in red")
    b.set_xlabel("x (µm)")

    c = fig.add_subplot(gs[0, 2])
    _draw_arr(c, arr, sliver, ((39.995, 40.008), (-12, 22)))
    c.set_aspect("auto")
    c.set_xticks([40.0, 40.003])
    c.set_xticklabels(["40.000", "40.003"])
    c.set_title(f"Zoom: a {narrowest[1] * 1000:.1f} nm wide cell along 30 µm")
    c.set_xlabel("x (µm), note the 13 nm window")
    c.set_ylabel("y (µm)")

    for col, (k, title) in enumerate(
        (
            (2, "Local level z = 0.2 µm (M1 and its slabs only)"),
            (3, "Local level z = 3 µm (M2 and its slabs only)"),
        )
    ):
        ax = fig.add_subplot(gs[1, col])
        layer = result.level_layers[k]
        _draw_arr(ax, layer.arr, set(), ((-60, 80), (-45, 60)))
        ax.set_title(f"{title}: {len(layer.arr.faces) - 1} cells")
        ax.set_xlabel("x (µm)")
    d = fig.add_subplot(gs[1, 2])
    per_level = len(arr.faces) - 1
    d.bar(
        ["global projection\non every level", "local arrangements,\nall levels"],
        [per_level * levels, local_cells],
        color=[COLOURS["bad"], COLOURS["good"]],
    )
    d.text(
        0,
        per_level * levels,
        f"{per_level} per level × {levels} levels = {per_level * levels}",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    d.text(
        1, local_cells, f"{local_cells} in total", ha="center", va="bottom", fontsize=8
    )
    d.set_ylim(0, per_level * levels * 1.2)
    d.set_ylabel("cells")
    d.set_title("Cells to label and mesh-constrain")
    fig.tight_layout()
    save_svg(fig, "layered-phantom")


def _phantom_html(arr, widths, sliver):
    """The interactive copy of the global projection and the local ones."""
    # interactive: the global arrangement with a zoom button
    fi_ = go.Figure()
    shapes = []
    hover_x, hover_y, hover_t = [], [], []
    for fi in range(1, len(arr.faces)):
        rings = _rings_of(arr, fi)
        path = " ".join(
            "M " + " L ".join(f"{x:.4f},{y:.4f}" for x, y in r) + " Z" for r in rings
        )
        shapes.append(
            dict(
                type="path",
                path=path,
                fillcolor="#c0392b" if fi in sliver else "#e9eef5",
                line=dict(color="#222", width=0.6),
                layer="below",
                fillrule="evenodd",
            )
        )
        x, y = rings[0][0]
        w = next(w for f, w, _ in widths if f == fi)
        hover_x.append(x)
        hover_y.append(y)
        hover_t.append(f"cell {fi}: width {w * 1000:.2f} nm")
    fi_.add_trace(
        go.Scatter(
            x=hover_x,
            y=hover_y,
            mode="markers",
            marker=dict(size=5, color="#222"),
            text=hover_t,
            hoverinfo="text",
            name="cells",
        )
    )
    views = {
        "whole projection": ((-60, 80), (-45, 60)),
        "zoom on the sliver": ((39.995, 40.008), (-12, 22)),
        "zoom on the lower end": ((39.9995, 40.0045), (-11, -9)),
    }
    fi_.update_layout(
        shapes=shapes,
        title="Global projection of the two layers; the red cell is 3 nm wide",
        xaxis=dict(range=list(views["whole projection"][0]), title="x (µm)"),
        yaxis=dict(
            range=list(views["whole projection"][1]), title="y (µm)", scaleanchor="x"
        ),
        height=620,
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0,
                y=1.12,
                buttons=[
                    dict(
                        label=k,
                        method="relayout",
                        args=[
                            {
                                "xaxis.range": list(v[0]),
                                "yaxis.range": list(v[1]),
                                "yaxis.scaleanchor": "x"
                                if k == "whole projection"
                                else None,
                            }
                        ],
                    )
                    for k, v in views.items()
                ],
            )
        ],
    )
    save_plotly(fi_, "layered-phantom")


if __name__ == "__main__":
    main()
