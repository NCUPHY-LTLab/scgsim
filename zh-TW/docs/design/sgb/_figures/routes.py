"""Figures and numbers of ``routes.qmd``: Route A, B and C of the running example (computed with the production package).

Writes into ``docs/design/sgb/figures/``:

* ``routes-cells.svg/.html``  the cells of the metal-layer levels of the running example, each coloured by the primary
  physical group of its face: Route A at z = 0 (sheet plane), Route B at z = 0 (metal bottom, where the port lies) and at
  z = 0.2 (metal top);
* ``routes-3d.svg/.html``     Route A and Route B as exploded volumes coloured by primary group;
* ``routes-abc-section.svg``  a conceptual cross-section (not computed);
* ``routes-numbers.json``     side-by-side counts of the three routes, plus the two stage-level examples of the page.

The running example is the production real input ``gds_pad_in_ground`` (``tests/sgb_inputs.py``); the port lies on the
support plane z = 0.  Route A needs zero-thickness conductors, so Route A is the same GDS and stack interpreted with the
``thin_film`` profile (``labels_common.real_input(film=True)``); Route B and C use the ``solid`` profile.  Two examples
are not the real input and say so in the numbers: the stage-level scene ``pad_in_ground("B")`` of ``tests/sgb_scenes.py``
(port in the metal's mid-plane, a sheet inside one volume) and the real input with one dangling refinement curve.

Run:  SGB_CORE=<sgb-core checkout> <sgb-core>/.artifacts/env/bin/python docs/design/sgb/_figures/routes.py
"""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from fractions import Fraction
from pathlib import Path

from labels_common import (
    GROUP_COLOURS,
    PALETTE,
    build_model,
    cell_rings_um,
    compact_triangles,
    finish_plan,
    mesh_trace,
    mpl_path,
    np,
    plotly_cells,
    plt,
    real_catalogue,
    save_plotly,
    save_svg,
    stretch_z,
    write_numbers,
)
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch, PathPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plotly.subplots import make_subplots
from scgsim.geometry.sgb.build import emit_model
from scgsim.geometry.sgb.cutfamilies.requests import (
    ConductorSurface,
    InsetRequest,
    RefinementRequest,
    SolutionInterface,
)
from sgb_scenes import pad_in_ground as stage_pad_in_ground

GROUP_FILL = {
    "ground_0000": "#7f7f7f",
    "pad": "#3a6ea5",
    "MS::ground_0000::bottom": "#a9a9a9",
    "MS::pad::bottom": "#8f7bc2",
    "MA::ground_0000::top": "#5b5f64",
    "MA::pad::top": "#2171b5",
    "SA::substrate|VACUUM_REGION::interface": "#9ecae1",
    "PORT::PORT_SHEET__P1__0000": "#d62728",
    None: "#f5f5f5",
}
VIEW = ((-24, 124), (-24, 124))


def cell_group(model, layer, fi: int):
    face = model.topology.faces.get(layer.face_ids[fi])
    return model.routed.face_group[face.id] if face is not None else None


def level_of(model, z: Fraction):
    return model.labelled.level_layers[model.labelled.levels.levels.index(z)]


def figure_cells(a, b, numbers: dict) -> None:
    panels = [
        ("Route A, level z = 0 (sheet plane)", a, level_of(a, Fraction(0))),
        (
            "Route B, level z = 0 (metal bottom, port plane)",
            b,
            level_of(b, Fraction(0)),
        ),
        ("Route B, level z = 0.2 (metal top)", b, level_of(b, Fraction(1, 5))),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6))
    pcs = make_subplots(
        rows=1, cols=3, subplot_titles=[p[0] for p in panels], horizontal_spacing=0.04
    )
    used = set()
    summary = []
    for ax, (title, model, layer), col in zip(axes, panels, (1, 2, 3)):
        groups = []
        for fi in range(1, len(layer.arr.faces)):
            group = cell_group(model, layer, fi)
            groups.append(group)
            used.add(group)
            ax.add_patch(
                PathPatch(
                    mpl_path(cell_rings_um(layer, fi)),
                    facecolor=GROUP_FILL[group],
                    edgecolor=PALETTE["ink"],
                    lw=1.2 if group else 0.5,
                    hatch=None if group else "///",
                )
            )
        finish_plan(ax, *VIEW, title)
        ax.set_title(title, fontsize=9)
        plotly_cells(
            pcs,
            layer,
            lambda lay, fi, _m=model: GROUP_FILL[cell_group(_m, lay, fi)],
            lambda lay, fi, _m=model: (
                f"group: {cell_group(_m, lay, fi)}<br>"
                + str(lay.labels[fi].sheet or "")
            ),
            1,
            col,
            *VIEW,
        )
        summary.append(
            {
                "panel": title,
                "plane": layer.plane,
                "cells": len(layer.arr.faces) - 1,
                "edges": len(layer.arr.edges),
                "groups": dict(Counter(str(g) for g in groups)),
            }
        )
    handles = [
        Patch(
            facecolor=GROUP_FILL[g],
            edgecolor="#222",
            label=str(g) if g else "no face (same domain both sides)",
        )
        for g in GROUP_FILL
        if g in used
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save_svg(fig, "routes-cells")
    for col in (1, 2, 3):
        pcs.update_yaxes(scaleanchor=f"x{col if col > 1 else ''}", row=1, col=col)
    pcs.update_layout(
        title="Cells of the metal-layer levels coloured by the primary physical group of their face",
        width=1500,
        height=640,
    )
    save_plotly(pcs, "routes-cells")
    # same cells: identical edge geometry of Route A z = 0 and Route B z = 0
    ea = {(e.a, e.b) for e in panels[0][2].arr.edges}
    eb = {(e.a, e.b) for e in panels[1][2].arr.edges}
    numbers["cells"] = {
        "panels": summary,
        "route_A_level0_and_route_B_level0_have_identical_edges": ea == eb,
        "shared_edge_count": len(ea & eb),
    }


# --------------------------------------------------------------------------- 3D

SHIFT = {"substrate": -60.0, "VACUUM_REGION": 60.0}
METAL_STRETCH = 100.0


def face_batches(model, tris):
    """(points, triangles, group, face id, volume) per drawn face: each face once per adjacent emitted volume."""
    topo = model.topology
    batches = []
    for vol in topo.volumes.values():
        if vol.id not in model.routed.emitted_volumes:
            continue
        for shell in vol.shells:
            for fid, _ in shell.faces:
                if fid in tris:
                    pts, tri = tris[fid]
                    batches.append(
                        (pts, tri, model.routed.face_group[fid], fid, vol.domain.name)
                    )
    return batches


def figure_3d(a, b) -> None:
    fig = plt.figure(figsize=(13, 6.6))
    pcs = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(
            "Route A: sheets",
            "Route B: finite metal, conductor volumes excluded",
        ),
    )
    used = {}
    for k, (model, label, metal) in enumerate(
        (
            (a, "Route A (sheet metal)", 1.0),
            (b, "Route B (excluded finite metal)", METAL_STRETCH),
        ),
        start=1,
    ):
        tris = compact_triangles(model, mesh_size=60.0, mesh_size_min=4.0)
        ax = fig.add_subplot(1, 2, k, projection="3d")
        polys, colours = [], []
        for pts, tri, group, fid, vol in face_batches(model, tris):
            prefix = group.split("::")[0] if "::" in group else "SHEET"
            colour = GROUP_COLOURS.get(prefix, "#999")
            used[prefix] = colour
            z = stretch_z(pts[:, 2], metal=metal) + SHIFT.get(vol, 0.0)
            xyz = np.column_stack([pts[:, 0], pts[:, 1], z])
            alpha = (
                0.12 if prefix == "EXT" else 0.75
            )  # the outer faces would hide everything else
            pcs.add_trace(
                mesh_trace(
                    xyz,
                    tri,
                    colour,
                    group,
                    f"{group}<br>{vol} side<br>{fid}",
                    opacity=alpha,
                ),
                row=1,
                col=k,
            )
            polys.extend(xyz[t] for t in tri)
            colours.extend([to_rgba(colour, alpha)] * len(tri))
        ax.add_collection3d(
            Poly3DCollection(polys, facecolors=colours, edgecolors="none")
        )
        ax.set_xlim(-30, 130)
        ax.set_ylim(-30, 130)
        ax.set_zlim(-100, 130)
        ax.set_zticks([])
        ax.set_zlabel("z (exploded, not to scale)")
        ax.set_box_aspect((1, 1, 0.8))
        ax.view_init(elev=22, azim=-55)
        ax.set_title(label)
    fig.legend(
        handles=[
            Patch(facecolor=c, label=f"{n} group") for n, c in sorted(used.items())
        ],
        loc="lower center",
        ncol=5,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )
    save_svg(fig, "routes-3d")
    for k in (1, 2):
        pcs.update_layout(
            **{
                f"scene{'' if k == 1 else k}": dict(
                    aspectmode="manual",
                    aspectratio=dict(x=1, y=1, z=0.8),
                    zaxis=dict(range=[-100, 130], title="z (display)"),
                )
            }
        )
    pcs.update_layout(
        title=f"Exploded volumes (substrate down, VACUUM_REGION up; Route B metal drawn {METAL_STRETCH:g}x thicker); "
        "each face once per adjacent emitted volume",
        width=1500,
        height=720,
    )
    save_plotly(pcs, "routes-3d")


# --------------------------------------------------------------------------- numbers


def emitted(model) -> dict:
    """The model's counts, the emitted counts of its BREP plan, and the sidecar facts the page quotes."""
    with tempfile.TemporaryDirectory() as tmp:
        artifacts = emit_model(model, Path(tmp))
        sidecar = json.loads(artifacts.sidecar.read_text())
    return {
        "counts": artifacts.counts,
        "embedded_faces": sidecar["embedded_faces"],
        "refinement_edges": sidecar["refinement_edges"],
        "ports": [
            {k: p[k] for k in ("group", "host_volumes", "adjacent_volumes", "z_um")}
            for p in sidecar["ports"]
        ],
    }


def route_numbers(model) -> dict:
    topo = model.topology
    out = emitted(model)
    free = next(
        (c for c in model.report.checks if c.name == "every_edge_shared_by_two_faces"),
        None,
    )
    return {
        "model_counts": model.counts,
        "faces_by_kind": dict(Counter(f.kind for f in topo.faces.values())),
        "edges_by_kind": dict(Counter(e.kind for e in topo.edges.values())),
        "level_cells_total": model.counts["level_cells"],
        "volumes": {
            v.domain.name: {
                "emitted": v.id in model.routed.emitted_volumes,
                "shells": len(v.shells),
                "embedded_faces": len(v.embedded_faces),
            }
            for v in topo.volumes.values()
        },
        "primary_groups_faces": {
            n: len(g.ids) for n, g in model.routed.groups.items() if g.dim == 2
        },
        "primary_groups_volumes": {
            n: len(g.ids) for n, g in model.routed.groups.items() if g.dim == 3
        },
        "unions": model.routed.unions,
        "validation": {
            "checks": len(model.report.checks),
            "passed": not model.report.failures,
            "warnings": [c.name for c in model.report.checks if c.warning],
            "every_edge_shared_by_two_faces": free.detail if free else None,
        },
        **out,
    }


def midplane_scene(numbers: dict) -> None:
    """Stage-level scene: a port sheet in the middle of the metal, inside one volume (the only embedded-face case)."""
    model = build_model(stage_pad_in_ground("B"), "B")
    numbers["stage_scene_midplane_port"] = {
        "scene": "tests/sgb_scenes.py pad_in_ground('B'): port at z = 0.1 um",
        **route_numbers(model),
    }


def refinement_on_real_input(b_catalogue, numbers: dict) -> None:
    """The real input with one dangling refinement curve in the gap: what production really embeds."""
    request = RefinementRequest("ref", 0.0, ((32.0, 55.0), (38.0, 55.0)))
    model = build_model(b_catalogue, "B", refinements=(request,))
    numbers["real_input_with_refinement_curve"] = {
        "request": "refinement curve (32, 55) -> (38, 55) um at z = 0",
        **route_numbers(model),
    }


def bands_on_real_input(catalogue, numbers: dict) -> None:
    """Route B with Surface-EPR bands (margins 0.1, 0.5, 2 um) on the pad top, the pad bottom and the substrate-vacuum gap."""
    margins = (0.1, 0.5, 2.0)
    requests = (
        InsetRequest("band_pad_top", ConductorSurface("pad", "top"), margins),
        InsetRequest("band_pad_bottom", ConductorSurface("pad", "bottom"), margins),
        InsetRequest(
            "band_gap",
            SolutionInterface("substrate", "VACUUM_REGION", z_um=0.0),
            margins,
        ),
    )
    model = build_model(catalogue, "B", insets=requests)
    numbers["route_B_bands"] = {
        "requests": "bands (0.1, 0.5, 2) um on pad top, pad bottom and the substrate-vacuum gap at z = 0",
        **route_numbers(model),
    }


def figure_abc_section() -> None:
    """Conceptual cross-section (not computed): the same pad edge region in Route A, B and C."""
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), sharey=True)
    titles = (
        "Route A: a sheet",
        "Route B: conductor volume excluded",
        "Route C: conductor volume kept",
    )
    for ax, title in zip(axes, titles):
        ax.set_xlim(0, 10)
        ax.set_ylim(-3, 4.2)
        ax.add_patch(
            plt.Rectangle((0, -3), 10, 3, facecolor="#cfe0f1", edgecolor="none")
        )
        ax.add_patch(
            plt.Rectangle((0, 0), 10, 4.2, facecolor="#fbf6df", edgecolor="none")
        )
        ax.text(0.2, -2.6, "substrate", fontsize=8)
        ax.text(0.2, 3.7, "vacuum", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    # A: one sheet
    axes[0].plot([3, 10], [0, 0], color="#7f7f7f", lw=4)
    axes[0].text(
        6.5, 0.35, "sheet face\n(one face, two sides)", ha="center", fontsize=8
    )
    # B: excluded metal: boundary faces drawn, interior hatched
    axes[1].add_patch(
        plt.Rectangle(
            (3, 0), 7, 1.2, facecolor="white", edgecolor="#555", hatch="////", lw=0
        )
    )
    axes[1].plot([3, 10], [1.2, 1.2], color="#2171b5", lw=3)
    axes[1].plot([3, 10], [0, 0], color="#6a51a3", lw=3)
    axes[1].plot([3, 3], [0, 1.2], color="#2171b5", lw=3)
    axes[1].text(
        6.5, 1.5, "MA face (vacuum side)", ha="center", fontsize=8, color="#2171b5"
    )
    axes[1].text(
        6.5, -0.55, "MS face (substrate side)", ha="center", fontsize=8, color="#6a51a3"
    )
    axes[1].text(
        6.5,
        0.45,
        "excluded: not meshed",
        ha="center",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="none", pad=1),
    )
    # C: kept metal, meshed inside, shared faces
    axes[2].add_patch(
        plt.Rectangle((3, 0), 7, 1.2, facecolor="#bfc3c7", edgecolor="none")
    )
    for x in np.linspace(3, 10, 9):
        axes[2].plot([x, x], [0, 1.2], color="#888", lw=0.5)
    axes[2].plot([3, 10], [1.2, 1.2], color=PALETTE["ink"], lw=3)
    axes[2].plot([3, 10], [0, 0], color=PALETTE["ink"], lw=3)
    axes[2].plot([3, 3], [0, 1.2], color=PALETTE["ink"], lw=3)
    axes[2].text(6.5, 1.5, "shared face with vacuum", ha="center", fontsize=8)
    axes[2].text(6.5, -0.55, "shared face with substrate", ha="center", fontsize=8)
    axes[2].text(
        6.5,
        0.45,
        "conductor interior: solution domain",
        ha="center",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="none", pad=1),
    )
    fig.suptitle(
        "Conceptual cross-section (not computed): the same edge of a pad under the three routes",
        fontsize=9,
        y=1.02,
    )
    fig.tight_layout()
    save_svg(fig, "routes-abc-section")


def main() -> None:
    catalogue_b = real_catalogue()
    a = build_model(real_catalogue(film=True), "A", keep_raw=True)
    b = build_model(catalogue_b, "B", keep_raw=True)
    c = build_model(catalogue_b, "C")
    numbers = {
        "scene": "gds_pad_in_ground (production real input); Route A = the same GDS and stack with the thin_film profile",
        "route_A": route_numbers(a),
        "route_B": route_numbers(b),
        "route_C": route_numbers(c),
    }
    figure_cells(a, b, numbers)
    figure_3d(a, b)
    figure_abc_section()
    midplane_scene(numbers)
    refinement_on_real_input(catalogue_b, numbers)
    bands_on_real_input(catalogue_b, numbers)
    write_numbers("routes", numbers)
    print("routes: done")


if __name__ == "__main__":
    main()
