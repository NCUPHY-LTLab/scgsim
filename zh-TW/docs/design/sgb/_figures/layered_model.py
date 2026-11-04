"""Figures of layered-model.qmd: prisms, levels, slabs, slab versus level arrangements, wall coalescing.

Computed with the production package ``scgsim.geometry.sgb``: the running example (``intro_common.running_example``,
Route B, port on the support plane z = 0) and the extended example ``sgb_scenes.airbridge()``.  The wall-coalescing
figure also builds the stage scene whose port lies at the metal mid-plane z = 0.1: the compiler never places a port
there, but in that scene a level does cut the metal walls, which the running example no longer shows.

Writes ``figures/layered-stack.svg`` (x-z sections with levels, slabs and active sets),
``figures/layered-exploded.html`` and ``.svg`` (3-D view prisms -> levels -> slabs of the airbridge),
``figures/layered-slab-vs-level.svg`` (plan views of A_1, S_1, A_2 of the running example),
``figures/layered-walls.svg`` (wall faces with and without coalescing) and ``figures/layered-numbers.json``.
"""

from __future__ import annotations

import json
import math

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from intro_common import (
    COLOURS,
    G_UM,
    build_example,
    cell_rings_um,
    domain_colour,
    domain_name,
    draw_section,
    mpl_path,
    running_example,
    save_plotly,
    save_svg,
    write_numbers,
)
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sgb_scenes import airbridge
from shapely.geometry import Polygon, box
from shapely.ops import polylabel

from scgsim.geometry.sgb.model import build_model_from_catalogue
from scgsim.geometry.sgb.prisms.levels import build_levels

# --------------------------------------------------------------------------- stack of levels and slabs


def _active_names(spec, structure, k: int) -> list[str]:
    return [spec.prisms[i].entity_id for i in structure.slab_active[k]]


def _stack_figure(specs: dict) -> dict:
    fig, axes = plt.subplots(
        1, 2, figsize=(12.5, 5.2), gridspec_kw={"width_ratios": [1, 1]}
    )
    facts = {}
    for ax, (name, y, spec) in zip(axes, specs.values(), strict=True):
        structure = build_levels(spec.prisms, spec.families)
        draw_section(ax, spec, structure.levels, y, label_slabs=False)
        for k in range(structure.slab_count):
            ax.text(
                -152,
                k + 0.5,
                f"slab {k}: " + ", ".join(_active_names(spec, structure, k)),
                va="center",
                ha="left",
                fontsize=7,
                zorder=10,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1),
            )
        ax.set_title(f"{name}: x-z section at y = {y:g} µm")
        facts[name] = {
            "levels_um": [float(z) for z in structure.levels],
            "active_per_slab": [
                _active_names(spec, structure, k) for k in range(structure.slab_count)
            ],
            "sheets_per_level": [
                [spec.prisms[i].entity_id for i in s] for s in structure.level_sheets
            ],
        }
    axes[0].legend(
        handles=[
            Line2D([0], [0], color=c, lw=6, label=n)
            for n, c in (
                ("substrate", COLOURS["substrate"]),
                ("vacuum", COLOURS["vacuum"]),
                ("ground", COLOURS["ground"]),
                ("pad", COLOURS["pad"]),
                ("port sheet", COLOURS["port"]),
            )
        ],
        loc="upper center",
        ncol=5,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.1),
    )
    fig.tight_layout()
    save_svg(fig, "layered-stack")
    return facts


# --------------------------------------------------------------------------- slab versus level arrangement


def _level_key(label) -> str:
    if label.sheet is not None:
        return f"sheet: {label.sheet.name}"
    return f"{domain_name(label.below)} | {domain_name(label.above)}"


def _draw_layer(ax, layer, fill, view) -> None:
    window = box(view[0][0], view[1][0], view[0][1], view[1][1])
    for fi in range(1, len(layer.arr.faces)):
        colour, text = fill(layer.labels[fi])
        rings = cell_rings_um(layer, fi)
        ax.add_patch(
            PathPatch(mpl_path(rings), facecolor=colour, edgecolor="#222", lw=0.7)
        )
        inside = Polygon(rings[0], rings[1:]).buffer(0).intersection(window)
        if not inside.is_empty:
            biggest = (
                max(inside.geoms, key=lambda g: g.area)
                if hasattr(inside, "geoms")
                else inside
            )
            p = polylabel(biggest, tolerance=0.5)
            ax.text(
                p.x,
                p.y,
                text.replace(" | ", "\n| "),
                fontsize=6.5,
                ha="center",
                va="center",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.2),
            )
    ax.set_xlim(*view[0])
    ax.set_ylim(*view[1])
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)")


def slab_vs_level_figure(model) -> dict:
    """The arrangements around the metal: A_1 (z = 0, the port sheet lies here), the metal slab S_1 and A_2 (z = 0.2)."""
    lab = model.labelled
    a1, s1, a2 = lab.level_layers[1], lab.slabs[1], lab.level_layers[2]
    palette = {
        "substrate | ground": "#d9d9d9",
        "substrate | pad": "#c6dbef",
        "substrate | vacuum": "#e5f2e5",
        "ground | vacuum": "#bdbdbd",
        "pad | vacuum": "#9ecae1",
        "vacuum | vacuum": "#fbf6df",
        "sheet: p1": COLOURS["port"],
    }

    def level_fill(label):
        key = _level_key(label)
        return palette[key], key

    def slab_fill(domain):
        return domain_colour(domain), domain_name(domain)

    view = ((-100, 100), (-75, 75))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    z1 = float(lab.levels.levels[2])
    panels = (
        (
            axes[0],
            a1,
            level_fill,
            f"A_1 at z = 0: {len(a1.arr.faces) - 1} cells, {len(a1.arr.edges)} edges",
        ),
        (
            axes[1],
            s1,
            slab_fill,
            f"S_1, slab 0 to {z1:g} µm: {len(s1.arr.faces) - 1} cells, {len(s1.arr.edges)} edges",
        ),
        (
            axes[2],
            a2,
            level_fill,
            f"A_2 at z = {z1:g} µm: {len(a2.arr.faces) - 1} cells, {len(a2.arr.edges)} edges",
        ),
    )
    for ax, layer, fill, title in panels:
        _draw_layer(ax, layer, fill, view)
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("y (µm)")
    fig.text(
        0.5,
        -0.02,
        "Level cells are labelled (below | above); slab cells carry one occupying domain.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout()
    save_svg(fig, "layered-slab-vs-level")
    per_layer = {}
    for name, layer in (("A_1", a1), ("S_1", s1), ("A_2", a2)):
        per_layer[name] = {
            "cells": len(layer.arr.faces) - 1,
            "edges": len(layer.arr.edges),
            "labels": sorted(
                {
                    _level_key(label) if hasattr(label, "below") else domain_name(label)
                    for label in layer.labels[1:]
                }
            ),
        }
    return per_layer


# --------------------------------------------------------------------------- wall coalescing


def _wall_facts(model) -> dict:
    topo = model.topology
    levels = list(topo.levels)
    metal_slab = (
        levels[1],
        levels[-2],
    )  # the metal layer of the running example: the slabs between its two faces
    walls = []
    for f in topo.faces.values():
        if f.kind != "wall":
            continue
        pts = []
        length = 0.0
        for eid, _ in f.loops[0]:
            e = topo.edges[eid]
            if e.kind != "horizontal":
                continue
            a, b = topo.vertices[e.v0], topo.vertices[e.v1]
            pts += [(a.x, a.y), (b.x, b.y)]
            length += math.hypot(b.x - a.x, b.y - a.y)
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        mid = (sum(xs) / len(xs) * G_UM, sum(ys) / len(ys) * G_UM)
        z0, z1 = f.z_range
        slabs = sum(
            1
            for lo, hi in zip(levels, levels[1:], strict=False)
            if lo >= z0 and hi <= z1
        )
        walls.append(
            {
                "id": f.id,
                "group": model.routed.face_group.get(f.id) or "",
                "z": (float(z0), float(z1)),
                "slabs": slabs,
                "mid": mid,
                "length_um": length * G_UM / 2,
                "curved": f.curved,
                "cut": (z0, z1) != metal_slab,
            }
        )
    metal = [w for w in walls if w["group"].startswith("MA::")]
    return {
        "walls": walls,
        "coalesced": len(walls),
        "uncoalesced": sum(w["slabs"] for w in walls),
        "metal_coalesced": len(metal),
        "metal_uncoalesced": sum(w["slabs"] for w in metal),
        "metal_walls_cut_inside_the_metal_layer": sum(1 for w in metal if w["cut"]),
    }


def walls_figure(support, mid) -> dict:
    """Left: metal wall faces of the stage scene with the port at z = 0.1.  Right: counts for both port planes."""
    facts, facts_support = _wall_facts(mid), _wall_facts(support)
    walls = facts["walls"]
    fig, axes = plt.subplots(
        1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [1.5, 1]}
    )
    _draw_wall_pieces(walls, axes)
    _draw_wall_counts(facts, facts_support, axes)
    fig.tight_layout()
    save_svg(fig, "layered-walls")

    def totals(model, f) -> dict:
        return {
            "coalesced": f["coalesced"],
            "uncoalesced": f["uncoalesced"],
            "metal_coalesced": f["metal_coalesced"],
            "metal_uncoalesced": f["metal_uncoalesced"],
            "total_faces": len(model.topology.faces),
            "horizontal_faces": sum(
                1 for x in model.topology.faces.values() if x.kind == "horizontal"
            ),
            "levels_um": [float(z) for z in model.topology.levels],
            "metal_walls_cut_inside_the_metal_layer": f[
                "metal_walls_cut_inside_the_metal_layer"
            ],
        }

    return {
        "support_plane_port": totals(support, facts_support),
        "mid_plane_port_stage_scene": {
            **totals(mid, facts),
            "split_walls": [
                {
                    "group": w["group"],
                    "z": w["z"],
                    "mid_um": [round(v, 1) for v in w["mid"]],
                }
                for w in walls
                if w["cut"] and w["group"].startswith("MA::")
            ],
        },
    }


def _draw_wall_pieces(walls, axes):
    """The metal wall faces of the stage scene with the port at z = 0.1, drawn piece by piece with their coalescing."""
    ax = axes[0]
    rows = (
        (
            "pad outline",
            lambda w: (
                w["group"] == "MA::pad::sidewall"
                and math.hypot(w["mid"][0] + 25, w["mid"][1] - 5) > 12
            ),
            (0.0, 0.0),
        ),
        (
            "ground cutout",
            lambda w: (
                w["group"] == "MA::ground::sidewall"
                and abs(w["mid"][0]) < 85
                and abs(w["mid"][1]) < 65
            ),
            (0.0, 0.0),
        ),
        (
            "pad hole (r = 8 µm)",
            lambda w: (
                w["group"] == "MA::pad::sidewall"
                and math.hypot(w["mid"][0] + 25, w["mid"][1] - 5) <= 12
            ),
            (-25.0, 5.0),
        ),
    )
    x_offset = 0.0
    for title, pick, centre in rows:
        columns: dict[tuple, list] = {}
        for w in (w for w in walls if pick(w)):
            columns.setdefault(
                (round(w["mid"][0], 1), round(w["mid"][1], 1)), []
            ).append(w)
        sel = sorted(
            columns.values(),
            key=lambda c: math.atan2(
                c[0]["mid"][1] - centre[1], c[0]["mid"][0] - centre[0]
            ),
        )
        start = x_offset
        for column in sel:
            width = max(column[0]["length_um"], 4.0)
            for w in column:
                cut = len(column) > 1
                ax.add_patch(
                    plt.Rectangle(
                        (x_offset, w["z"][0]),
                        width * 0.97,
                        w["z"][1] - w["z"][0],
                        facecolor=COLOURS["bad"] if cut else COLOURS["pad"],
                        edgecolor="white",
                        lw=1.5,
                        alpha=0.9,
                    )
                )
            x_offset += width
        ax.text(start, 0.212, title, ha="left", fontsize=8, fontweight="bold")
        x_offset += 12
    ax.axhline(0.1, color="#222", ls="--", lw=1.0, zorder=10)
    ax.set_xlim(-4, x_offset)
    ax.set_ylim(-0.01, 0.24)
    ax.set_xlabel("wall pieces in boundary order, width = plan length (µm)")
    ax.set_ylabel("z (µm)")
    ax.set_title(
        "Metal wall faces of the stage scene with the port at z = 0.1 µm (one bar per face)"
    )
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=COLOURS["pad"],
                lw=8,
                label="one face spans both slabs (coalesced)",
            ),
            Line2D(
                [0],
                [0],
                color=COLOURS["bad"],
                lw=8,
                label="cut at z = 0.1 µm where the port touches",
            ),
            Line2D(
                [0],
                [0],
                color="#222",
                lw=1.2,
                ls="--",
                label="level z = 0.1 µm (port sheet plane)",
            ),
        ],
        fontsize=7,
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0, -0.38),
        ncol=3,
    )


def _draw_wall_counts(facts, facts_support, axes):
    """The metal wall faces of both builds, as built and with coalescing disabled."""
    b = axes[1]
    labels = ["port at z = 0\n(running example)", "port at z = 0.1\n(stage scene)"]
    per_slab = [facts_support["metal_uncoalesced"], facts["metal_uncoalesced"]]
    merged = [facts_support["metal_coalesced"], facts["metal_coalesced"]]
    width = 0.38
    for offset, values, colour, name in (
        (-width / 2, per_slab, COLOURS["ground"], "one wall per slab (no coalescing)"),
        (width / 2, merged, COLOURS["pad"], "coalesced"),
    ):
        bars = b.bar(
            [i + offset for i in range(2)], values, width, color=colour, label=name
        )
        for bar in bars:
            b.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f"{int(bar.get_height())}",
                ha="center",
                fontsize=9,
            )
    b.set_xticks(range(2))
    b.set_xticklabels(labels, fontsize=8)
    b.set_ylabel("metal wall faces")
    b.set_title("Metal wall faces of the pad-in-ground, Route B")
    b.legend(fontsize=7, frameon=False, loc="upper left")
    b.set_ylim(0, max(per_slab) * 1.18)


# --------------------------------------------------------------------------- 3-D prisms, levels, slabs


def _box_mesh(x0, x1, y0, y1, z0, z1):  # noqa: PLR0913 -- the two corners of an axis-aligned box
    xs = [x0, x1, x1, x0, x0, x1, x1, x0]
    ys = [y0, y0, y1, y1, y0, y0, y1, y1]
    zs = [z0, z0, z0, z0, z1, z1, z1, z1]
    tri = [
        (0, 1, 2),
        (0, 2, 3),
        (4, 6, 5),
        (4, 7, 6),
        (0, 4, 5),
        (0, 5, 1),
        (1, 5, 6),
        (1, 6, 2),
        (2, 6, 7),
        (2, 7, 3),
        (3, 7, 4),
        (3, 4, 0),
    ]
    return xs, ys, zs, tri


def _box_quads(x0, x1, y0, y1, z0, z1):  # noqa: PLR0913 -- the two corners of an axis-aligned box
    c = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    return [
        [c[i] for i in q]
        for q in (
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        )
    ]


def box_outline(x0, x1, y0, y1, z0, z1):  # noqa: PLR0913 -- the two corners of an axis-aligned box
    """The 12 edges of a box as one polyline with gaps (for a Scatter3d trace)."""
    c = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    xs, ys, zs = [], [], []
    for a, b in (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ):
        xs += [c[a][0], c[b][0], None]
        ys += [c[a][1], c[b][1], None]
        zs += [c[a][2], c[b][2], None]
    return xs, ys, zs


SPACING = 30.0  # display height of one slab (levels are equally spaced)
GAP = 40.0  # extra display gap between slabs in the exploded view


def prism_boxes(spec, structure):
    """Metal prisms of the airbridge as (name, bbox in um, level index of z0, level index of z1)."""
    index = {z: k for k, z in enumerate(structure.levels)}
    out = []
    for p in spec.prisms:
        if p.material_kind != "conductor":
            continue
        pts = [q for ring in p.footprint.rings() for q in ring]
        xs, ys = [q[0] * G_UM for q in pts], [q[1] * G_UM for q in pts]
        out.append(
            (
                p.entity_id,
                (min(xs), max(xs), min(ys), max(ys)),
                index[p.z0],
                index[p.z1],
            )
        )
    return out


def _exploded_data(spec):
    structure = build_levels(spec.prisms, spec.families)
    boxes = prism_boxes(spec, structure)
    views = {"prisms": [], "levels": [], "slabs": []}
    for name, (x0, x1, y0, y1), k0, k1 in boxes:
        views["prisms"].append(
            (
                name,
                (x0, x1, y0, y1),
                k0 * SPACING,
                k1 * SPACING,
                f"prism {name}: levels {k0} to {k1}",
            )
        )
        for k in range(k0, k1):
            views["slabs"].append(
                (
                    name,
                    (x0, x1, y0, y1),
                    k * (SPACING + GAP),
                    k * (SPACING + GAP) + SPACING,
                    f"{name} in slab {k}",
                )
            )
    return structure, boxes, views


def _exploded_figures(spec) -> dict:
    structure, boxes, views = _exploded_data(spec)
    colour = {
        "ground_left": COLOURS["ground"],
        "ground_right": COLOURS["ground"],
        "post_left": COLOURS["accent"],
        "post_right": COLOURS["accent"],
        "deck": COLOURS["pad"],
    }
    levels_z = [k * SPACING for k in range(len(structure.levels))]
    _exploded_html(structure, views, colour, levels_z)
    _exploded_svg(views, colour, levels_z)
    return {
        "metal_prisms": len(boxes),
        "slab_pieces": len(views["slabs"]),
        "levels": [float(z) for z in structure.levels],
    }


def _exploded_html(structure, views, colour, levels_z):
    """The interactive exploded view with buttons for the prisms, levels and slabs."""
    fig = go.Figure()
    member: list[set[str]] = []  # which views show each trace

    def add_boxes(items, shown_in):
        for name, (x0, x1, y0, y1), z0, z1, hover in items:
            xs, ys, zs, tri = _box_mesh(x0, x1, y0, y1, z0, z1)
            fig.add_trace(
                go.Mesh3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    i=[t[0] for t in tri],
                    j=[t[1] for t in tri],
                    k=[t[2] for t in tri],
                    color=colour[name],
                    opacity=0.9,
                    flatshading=True,
                    name=name,
                    showlegend=False,
                    hovertext=hover,
                    hoverinfo="text",
                )
            )
            member.append(shown_in)
            ox, oy, oz = box_outline(x0, x1, y0, y1, z0, z1)
            fig.add_trace(
                go.Scatter3d(
                    x=ox,
                    y=oy,
                    z=oz,
                    mode="lines",
                    line=dict(color="#222", width=2),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            member.append(shown_in)

    add_boxes(views["prisms"], {"prisms", "levels"})
    for k, z in enumerate(levels_z):
        text = f"level {k}: z = {float(structure.levels[k]):g} µm"
        fig.add_trace(
            go.Mesh3d(
                x=[-150, 150, 150, -150],
                y=[-150, -150, 150, 150],
                z=[z] * 4,
                i=[0, 0],
                j=[1, 2],
                k=[2, 3],
                color="#999",
                opacity=0.18,
                name=text,
                showlegend=False,
                hovertext=text,
                hoverinfo="text",
            )
        )
        member.append({"levels"})
    add_boxes(views["slabs"], {"slabs"})
    for trace, views_of in zip(fig.data, member, strict=True):
        trace.visible = "prisms" in views_of
    buttons = [
        dict(
            label=label,
            method="update",
            args=[
                {
                    "visible": [
                        "" if False else (view in views_of) for views_of in member
                    ]
                }
            ],
        )
        for view, label in (
            ("prisms", "1  prisms"),
            ("levels", "2  prisms and levels"),
            ("slabs", "3  prisms cut into slabs (exploded)"),
        )
    ]
    fig.update_layout(
        updatemenus=[
            dict(type="buttons", direction="right", x=0, y=1.1, buttons=buttons)
        ],
        title="Airbridge: the same seven prisms as prisms, with their levels, and cut into slabs",
        height=640,
        scene=dict(
            aspectmode="data",
            xaxis_title="x (µm)",
            yaxis_title="y (µm)",
            zaxis=dict(title="z (levels equally spaced)"),
            camera=dict(eye=dict(x=1.6, y=-1.7, z=0.9)),
        ),
    )
    save_plotly(fig, "layered-exploded")


def _exploded_svg(views, colour, levels_z):
    """The static panels of the prisms, levels and slabs."""
    f = plt.figure(figsize=(13, 4.8))
    titles = (
        "1  prisms: footprint × [z0, z1]",
        "2  levels: planes at every z0 and z1",
        "3  slabs: prisms cut between levels (exploded)",
    )
    for n, (title, items) in enumerate(
        zip(titles, (views["prisms"], views["prisms"], views["slabs"]), strict=True),
        start=1,
    ):
        ax = f.add_subplot(1, 3, n, projection="3d")
        for name, (x0, x1, y0, y1), z0, z1, _ in items:
            ax.add_collection3d(
                Poly3DCollection(
                    _box_quads(x0, x1, y0, y1, z0, z1),
                    facecolors=colour[name],
                    edgecolors="#222",
                    linewidths=0.6,
                    alpha=0.92,
                )
            )
        if n == 2:
            for z in levels_z:
                ax.add_collection3d(
                    Poly3DCollection(
                        [
                            [
                                (-150, -150, z),
                                (150, -150, z),
                                (150, 150, z),
                                (-150, 150, z),
                            ]
                        ],
                        facecolors="#999",
                        alpha=0.12,
                        edgecolors="#777",
                        linewidths=0.4,
                    )
                )
        ax.set_xlim(-150, 150)
        ax.set_ylim(-150, 150)
        ax.set_zlim(0, (len(levels_z) - 1) * (SPACING + (GAP if n == 3 else 0)))
        ax.set_box_aspect((1, 1, 0.8))
        ax.view_init(elev=34, azim=-42)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
        ax.set_zticks([])
    f.tight_layout()
    save_svg(f, "layered-exploded")


def main() -> None:
    """Write the layered-model figures and their numbers."""
    pad = running_example("B")
    bridge = airbridge()
    support = build_example("B")
    mid = build_example("B", port_plane="mid")
    bridge_model = build_model_from_catalogue(bridge, "B")
    numbers = {
        "stacks": _stack_figure(
            {
                "running": ("pad_in_ground_B", 0.0, pad),
                "extended": ("airbridge_B", 0.0, bridge),
            }
        ),
        "layers": slab_vs_level_figure(support),
        "walls": walls_figure(support, mid),
        "exploded": _exploded_figures(bridge),
        "counts_pad_in_ground_B": support.counts,
        "counts_pad_in_ground_B_midplane_port_stage_scene": mid.counts,
        "airbridge_counts": bridge_model.counts,
    }
    write_numbers("layered", numbers)
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
