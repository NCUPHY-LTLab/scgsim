"""Figures and numbers of ``routes.qmd``: Route A and Route B of the running example on the same cells.

Writes into ``docs/design/sgb/figures/``:

* ``routes-cells.svg/.html``  the same five cells of the sheet plane in both routes (Route A level z = 0, Route B level
  z = 0.1) plus the other two metal-layer levels of Route B, each cell coloured by the primary physical group of its face;
* ``routes-3d.svg/.html``     both models as exploded volumes coloured by primary group;
* ``routes-numbers.json``     side-by-side counts (vertices, edges, faces, groups, volumes, emission) of the two routes.

Everything comes from ``build(pad_in_ground("A"))`` and ``build(pad_in_ground("B"))`` plus the stored emission summaries
(``outputs/pad_in_ground_{A,B}/stage_summary.json``).

Run (from the package root):  .venv/bin/python docs/design/sgb/_figures/routes.py
"""

from __future__ import annotations

import json
from collections import Counter

from labels_common import (
    GROUP_COLOURS, OUTPUTS, PALETTE, cell_rings_um, compact_triangles, finish_plan, go, mesh_trace, mpl_path, np, plt,
    plotly_cells, save_plotly, save_svg, stretch_z, svg_path, write_numbers,
)
from matplotlib.colors import to_rgba
from matplotlib.patches import PathPatch, Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plotly.subplots import make_subplots

from sgbproto.pipeline import build
from sgbproto.scenes import pad_in_ground

GROUP_FILL = {
    "PEC::Ground": "#7f7f7f", "PEC::pad": "#3a6ea5", "MS::Ground": "#a9a9a9", "MS::pad": "#8f7bc2",
    "MA::Ground": "#5b5f64", "MA::pad": "#2171b5", "SA::substrate|vacuum": "#9ecae1", "PORT::p1": "#d62728",
    None: "#f5f5f5",
}
VIEW = ((-90, 90), (-70, 70))


def cell_group(result, layer, fi: int):
    face = result.topology.faces.get(layer.face_ids[fi])
    return face.group if face is not None else None


def figure_cells(a, b, numbers: dict) -> None:
    panels = [
        ("Route A, level z = 0 (sheet plane)", a, a.level_layers[1]),
        ("Route B, level z = 0.1 (port plane): the same five cells", b, b.level_layers[2]),
        ("Route B, level z = 0 (metal bottom)", b, b.level_layers[1]),
        ("Route B, level z = 0.2 (metal top)", b, b.level_layers[3]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6))
    pcs = make_subplots(rows=2, cols=2, subplot_titles=[p[0] for p in panels], horizontal_spacing=0.05, vertical_spacing=0.1)
    used = set()
    summary = []
    for ax, (title, result, layer), (r, c) in zip(axes.flat, panels, [(1, 1), (1, 2), (2, 1), (2, 2)]):
        groups = []
        for fi in range(1, len(layer.arr.faces)):
            group = cell_group(result, layer, fi)
            groups.append(group)
            used.add(group)
            ax.add_patch(PathPatch(mpl_path(cell_rings_um(layer, fi)), facecolor=GROUP_FILL[group], edgecolor=PALETTE["ink"],
                                   lw=1.2 if group else 0.5, hatch=None if group else "///"))
        finish_plan(ax, *VIEW, title)
        plotly_cells(pcs, layer, lambda lay, fi, _r=result: GROUP_FILL[cell_group(_r, lay, fi)],
                     lambda lay, fi, _r=result: f"group: {cell_group(_r, lay, fi)}<br>" + str(lay.labels[fi].sheet or ""), r, c, *VIEW)
        summary.append({"panel": title, "plane": layer.plane, "cells": len(layer.arr.faces) - 1, "edges": len(layer.arr.edges),
                        "groups": dict(Counter(str(g) for g in groups))})
    handles = [Patch(facecolor=GROUP_FILL[g], edgecolor="#222", label=str(g) if g else "no face (same domain both sides)")
               for g in GROUP_FILL if g in used]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_svg(fig, "routes-cells")
    for r in (1, 2):
        for c in (1, 2):
            pcs.update_yaxes(scaleanchor=f"x{(r - 1) * 2 + c if (r, c) != (1, 1) else ''}", row=r, col=c)
    pcs.update_layout(title="Cells of the metal-layer levels coloured by the primary physical group of their face", width=1250, height=950)
    save_plotly(pcs, "routes-cells")
    # same cells: identical edge geometry of Route A z = 0 and Route B z = 0.1
    ea = {(e.a, e.b) for e in a.level_layers[1].arr.edges}
    eb = {(e.a, e.b) for e in b.level_layers[2].arr.edges}
    numbers["cells"] = {"panels": summary, "route_A_level0_and_route_B_level_0.1_have_identical_edges": ea == eb,
                        "shared_edge_count": len(ea & eb)}


# --------------------------------------------------------------------------- 3D

SHIFT = {"substrate": -60.0, "vacuum": 60.0}


def face_batches(result, tris):
    """(points, triangles, group, hover, volume) per drawn face: each face once per adjacent emitted volume."""
    topo = result.topology
    batches = []
    for vol in topo.volumes.values():
        if not vol.emitted:
            continue
        for shell in vol.shells:
            for fid, _ in shell.faces:
                if fid in tris:
                    pts, tri = tris[fid]
                    batches.append((pts, tri, topo.faces[fid].group, fid, vol.domain.name))
    return batches


def figure_3d(a, b) -> None:
    fig = plt.figure(figsize=(13, 6.6))
    pcs = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                        subplot_titles=("Route A: sheets", "Route B: finite metal, conductor volumes excluded"))
    used = {}
    for k, (result, label) in enumerate(((a, "Route A (sheet metal)"), (b, "Route B (excluded finite metal)")), start=1):
        tris = compact_triangles(result.topology, mesh_size=60.0, mesh_size_min=4.0)
        ax = fig.add_subplot(1, 2, k, projection="3d")
        polys, colours = [], []
        for pts, tri, group, fid, vol in face_batches(result, tris):
            prefix = group.split("::")[0]
            colour = GROUP_COLOURS.get(prefix, "#999")
            used[prefix] = colour
            z = stretch_z(pts[:, 2]) + SHIFT.get(vol, 0.0)
            xyz = np.column_stack([pts[:, 0], pts[:, 1], z])
            alpha = 0.12 if prefix == "EXT" else 0.75  # the outer chip faces would hide everything else
            pcs.add_trace(mesh_trace(xyz, tri, colour, group, f"{group}<br>{vol} side<br>{fid}", opacity=alpha), row=1, col=k)
            polys.extend(xyz[t] for t in tri)
            colours.extend([to_rgba(colour, alpha)] * len(tri))
        ax.add_collection3d(Poly3DCollection(polys, facecolors=colours, edgecolors="none"))
        ax.set_xlim(-150, 150)
        ax.set_ylim(-150, 150)
        ax.set_zlim(-200, 200)
        ax.set_box_aspect((1, 1, 0.6))
        ax.view_init(elev=22, azim=-55)
        ax.set_title(label)
    fig.legend(handles=[Patch(facecolor=c, label=f"{n} group") for n, c in sorted(used.items())], loc="lower center", ncol=5,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))
    save_svg(fig, "routes-3d")
    for k in (1, 2):
        pcs.update_layout(**{f"scene{'' if k == 1 else k}": dict(aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.6),
                                                                  zaxis=dict(range=[-200, 200], title="z (display)"))})
    pcs.update_layout(title="Exploded volumes (substrate down, vacuum up; metal layer drawn 200x thicker); each face once per adjacent emitted volume",
                      width=1500, height=720)
    save_plotly(pcs, "routes-3d")


# --------------------------------------------------------------------------- numbers


def route_numbers(result, name: str) -> dict:
    topo = result.topology
    summary = json.loads((OUTPUTS / name / "stage_summary.json").read_text())["stages"]
    return {
        "counts": result.counts,
        "faces_by_kind": dict(Counter(f.kind for f in topo.faces.values())),
        "edges_by_kind": dict(Counter(e.kind for e in topo.edges.values())),
        "level_cells_total": result.counts["level_cells"],
        "volumes": {v.domain.name: {"emitted": v.emitted, "shells": len(v.shells), "embedded_faces": len(v.embedded_faces)}
                    for v in topo.volumes.values()},
        "primary_groups_faces": {n: len(g["ids"]) for n, g in topo.groups.items() if g["dim"] == 2},
        "primary_groups_volumes": {n: len(g["ids"]) for n, g in topo.groups.items() if g["dim"] == 3},
        "unions": topo.unions,
        "emission_native_entities_by_dim": summary["S10_emission"]["native_entities_by_dim"],
        "mesh_nodes": summary["S10_emission"]["mesh"]["nodes"],
        "mesh_conformal": summary["S10_emission"]["mesh"]["conformal"],
        "validation_checks_passed": summary["S9_validation"],
    }


def figure_abc_section() -> None:
    """Conceptual cross-section (not computed): the same pad edge region in Route A, B and C."""
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), sharey=True)
    titles = ("Route A: a sheet", "Route B: conductor volume excluded", "Route C: conductor volume kept")
    for ax, title in zip(axes, titles):
        ax.set_xlim(0, 10)
        ax.set_ylim(-3, 4.2)
        ax.add_patch(plt.Rectangle((0, -3), 10, 3, facecolor="#cfe0f1", edgecolor="none"))
        ax.add_patch(plt.Rectangle((0, 0), 10, 4.2, facecolor="#fbf6df", edgecolor="none"))
        ax.text(0.2, -2.6, "substrate", fontsize=8)
        ax.text(0.2, 3.7, "vacuum", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    # A: one sheet
    axes[0].plot([3, 10], [0, 0], color="#7f7f7f", lw=4)
    axes[0].text(6.5, 0.35, "PEC face\n(one face, two sides)", ha="center", fontsize=8)
    # B: excluded metal: boundary faces drawn, interior hatched
    axes[1].add_patch(plt.Rectangle((3, 0), 7, 1.2, facecolor="white", edgecolor="#555", hatch="////", lw=0))
    axes[1].plot([3, 10], [1.2, 1.2], color="#2171b5", lw=3)
    axes[1].plot([3, 10], [0, 0], color="#6a51a3", lw=3)
    axes[1].plot([3, 3], [0, 1.2], color="#2171b5", lw=3)
    axes[1].text(6.5, 1.5, "MA face (vacuum side)", ha="center", fontsize=8, color="#2171b5")
    axes[1].text(6.5, -0.55, "MS face (substrate side)", ha="center", fontsize=8, color="#6a51a3")
    axes[1].text(6.5, 0.45, "excluded: not meshed", ha="center", fontsize=8, bbox=dict(facecolor="white", edgecolor="none", pad=1))
    # C: kept metal, meshed inside, shared faces
    axes[2].add_patch(plt.Rectangle((3, 0), 7, 1.2, facecolor="#bfc3c7", edgecolor="none"))
    for x in np.linspace(3, 10, 9):
        axes[2].plot([x, x], [0, 1.2], color="#888", lw=0.5)
    axes[2].plot([3, 10], [1.2, 1.2], color=PALETTE["ink"], lw=3)
    axes[2].plot([3, 10], [0, 0], color=PALETTE["ink"], lw=3)
    axes[2].plot([3, 3], [0, 1.2], color=PALETTE["ink"], lw=3)
    axes[2].text(6.5, 1.5, "shared face with vacuum", ha="center", fontsize=8)
    axes[2].text(6.5, -0.55, "shared face with substrate", ha="center", fontsize=8)
    axes[2].text(6.5, 0.45, "conductor interior: solution domain", ha="center", fontsize=8, bbox=dict(facecolor="white", edgecolor="none", pad=1))
    fig.suptitle("Conceptual cross-section (not computed): the same edge of a pad under the three routes", fontsize=9, y=1.02)
    fig.tight_layout()
    save_svg(fig, "routes-abc-section")


def main() -> None:
    a = build(pad_in_ground("A"), keep_raw=True)
    b = build(pad_in_ground("B"), keep_raw=True)
    numbers = {"route_A": route_numbers(a, "pad_in_ground_A"), "route_B": route_numbers(b, "pad_in_ground_B")}
    figure_cells(a, b, numbers)
    figure_3d(a, b)
    figure_abc_section()
    write_numbers("routes", numbers)
    print("routes: done")


if __name__ == "__main__":
    main()
