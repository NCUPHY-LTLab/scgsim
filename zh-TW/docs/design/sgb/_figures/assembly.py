"""Figures and numbers of ``assembly.qmd`` (computed with the production package ``scgsim.geometry.sgb``).

Writes into ``docs/design/sgb/figures/``:

* ``assembly-wall-anatomy.svg``  the walls along three consecutive pad sidewalls of Route B, as built (coalesced) and as
  they would be if every level cut every wall; vertical edges, chains and shared vertices are drawn from the topology;
* ``assembly-shells.svg/.html``  the shell of the vacuum volume of the airbridge model, every face pulled slightly apart
  so shared edges show as gaps; faces are coloured by role and the coalesced walls are highlighted;
* ``assembly-inner-void.svg``    a floating metal island inside the vacuum of the running example: the vacuum volume gets
  a second, inner shell;
* ``assembly-curved-split.svg``  a conceptual drawing (not computed);
* ``assembly-numbers.json``      the counts the page quotes.

Which model is which.  The running example is the production real input ``gds_pad_in_ground`` (the port lies on the
support plane z = 0).  A level cuts a wall only where something at that level touches it, and the real input has no
level between the metal's bottom and top; the walls figure therefore uses the stage-level scene ``pad_in_ground("B")`` of
``tests/sgb_scenes.py`` (port in the metal's mid-plane z = 0.1, true arcs), and so do the airbridge counts
(``airbridge()`` of the same module).  Both say so in the numbers.

The "every level cuts every wall" counts are computed by replacing ``assembly.walls._level_cuts_wall`` with a constant
``True`` while building (the script restores it); nothing else changes.

Run:  SGB_CORE=<sgb-core checkout> <sgb-core>/.artifacts/env/bin/python docs/design/sgb/_figures/assembly.py
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import replace

from labels_common import (
    PALETTE,
    build_model,
    compact_triangles,
    go,
    mesh_trace,
    np,
    plt,
    real_catalogue,
    save_plotly,
    save_svg,
    write_numbers,
)
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch, Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sgb_scenes import airbridge, footprint, rect_um
from sgb_scenes import pad_in_ground as stage_pad_in_ground

from scgsim.geometry.sgb.topology.assembly import walls as walls_module
from scgsim.geometry.sgb.exact.grid import exact_z
from scgsim.geometry.sgb.exact.prisms import PrismRecord

G = 0.001


def build_uncoalesced(catalogue, route: str):
    """Build with the coalescing test replaced by 'every level cuts every wall' (restored afterwards)."""
    original = walls_module._level_cuts_wall  # noqa: SLF001 -- the counterfactual replaces the package's coalescing test; no public switch exists
    walls_module._level_cuts_wall = lambda *args: True  # noqa: SLF001 -- same counterfactual
    try:
        return build_model(catalogue, route)
    finally:
        walls_module._level_cuts_wall = original  # noqa: SLF001 -- same counterfactual: restore the package's test


# --------------------------------------------------------------------------- wall anatomy

SEGMENTS = [  # the pad outline as unrolled: ((x0, y0), (x1, y1)), s offset (um)
    ((10.0, -30.0), (50.0, -30.0), 0.0),
    ((50.0, -30.0), (50.0, 18.0), 40.0),
    ((50.0, 18.0), (38.0, 30.0), 88.0),
]
LEVEL_INDEX = {0.0: 0, 0.1: 1, 0.2: 2}


def unrolled_walls(model):
    """(face, polygon in (s, level index), edge list) of the pad walls on SEGMENTS."""
    topo = model.topology
    out = []
    for f in topo.faces.values():
        if f.kind != "wall" or model.routed.face_group[f.id] != "MA::pad__sidewall":
            continue
        verts = []
        for eid, sign in f.loops[0]:
            e = topo.edges[eid]
            v = topo.vertices[e.v0 if sign > 0 else e.v1]
            verts.append((v.x * G, v.y * G, float(v.z), e.kind, eid))
        for (ax, ay), (bx, by), s0 in SEGMENTS:
            length = math.hypot(bx - ax, by - ay)
            ux, uy = (bx - ax) / length, (by - ay) / length
            on = [
                abs((x - ax) * uy - (y - ay) * ux) < 1e-6
                and -1e-6 <= (x - ax) * ux + (y - ay) * uy <= length + 1e-6
                for x, y, _, _, _ in verts
            ]
            if all(on):
                poly = [
                    (s0 + (x - ax) * ux + (y - ay) * uy, LEVEL_INDEX[round(z, 3)])
                    for x, y, z, _, _ in verts
                ]
                edges = []
                for k, (eid, _) in enumerate(f.loops[0]):
                    a, b = poly[k], poly[(k + 1) % len(poly)]
                    edges.append((a, b, topo.edges[eid].kind, eid))
                out.append((f, poly, edges))
                break
    return out


def _draw_walls(ax, walls, title: str) -> dict:
    cmap = ["#cfe0f1", "#e8d9a8", "#cfe8d4", "#f1cfcf", "#d9cfe8", "#e0e0e0"]
    vertical_segments = 0
    for k, (f, poly, edges) in enumerate(
        sorted(walls, key=lambda w: (min(p[0] for p in w[1]), min(p[1] for p in w[1])))
    ):
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        # a small inset keeps neighbouring faces apart so that shared edges are visible as two lines
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        shrink = [(cx + (x - cx) * 0.985, cy + (y - cy) * 0.94) for x, y in poly]
        ax.add_patch(
            Polygon(
                shrink,
                closed=True,
                facecolor=cmap[k % len(cmap)],
                edgecolor="none",
                alpha=0.9,
            )
        )
        for a, b, kind, _ in edges:
            colour, width = (
                (PALETTE["bad"], 2.4) if kind == "vertical" else (PALETTE["ink"], 1.4)
            )
            vertical_segments += kind == "vertical"
            sa = (cx + (a[0] - cx) * 0.985, cy + (a[1] - cy) * 0.94)
            sb = (cx + (b[0] - cx) * 0.985, cy + (b[1] - cy) * 0.94)
            ax.plot(
                [sa[0], sb[0]],
                [sa[1], sb[1]],
                color=colour,
                lw=width,
                solid_capstyle="round",
            )
        for x, y in shrink:
            ax.plot(x, y, "o", ms=3.2, color=PALETTE["ink"], zorder=5)
        horiz_bottom = sum(
            1
            for a, b, kind, _ in edges
            if kind == "horizontal" and a[1] == b[1] == min(ys)
        )
        horiz_top = sum(
            1
            for a, b, kind, _ in edges
            if kind == "horizontal" and a[1] == b[1] == max(ys)
        )
        ax.text(
            cx,
            cy,
            f"wall {f.id[-4:]}\nbottom {horiz_bottom} edge{'s' * (horiz_bottom != 1)}\ntop {horiz_top} edge{'s' * (horiz_top != 1)}",
            ha="center",
            va="center",
            fontsize=7,
        )
    ax.set_xlim(-3, 108)
    ax.set_ylim(-0.25, 2.35)
    ax.set_yticks([0, 1, 2], ["z = 0", "z = 0.1 (port plane)", "z = 0.2"])
    ax.set_xlabel(
        "distance along the pad outline from (10, -30) µm: bottom edge, x = 50 sidewall, chamfer"
    )
    ax.set_title(title, fontsize=9)
    for s in (40, 88):
        ax.axvline(s, color="#999", lw=0.6, ls=":")
    return {"faces": len(walls), "vertical_edge_segments_in_loops": vertical_segments}


def _figure_wall_anatomy(numbers: dict) -> None:
    stage = stage_pad_in_ground("B")
    built = build_model(stage, "B")
    cut = build_uncoalesced(stage, "B")
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.4), sharex=True)
    a = _draw_walls(
        axes[0],
        unrolled_walls(built),
        "As built: a level cuts a wall only where something at that level touches it",
    )
    b = _draw_walls(
        axes[1],
        unrolled_walls(cut),
        "If every level cut every wall (same model, coalescing test disabled)",
    )
    fig.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                color=PALETTE["ink"],
                lw=1.4,
                label="horizontal edge (chain of the level arrangement)",
            ),
            plt.Line2D(
                [0],
                [0],
                color=PALETTE["bad"],
                lw=2.4,
                label="vertical edge segment of a wall column",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=PALETTE["ink"],
                label="vertex",
            ),
        ],
        loc="lower center",
        ncol=3,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_svg(fig, "assembly-wall-anatomy")
    numbers["wall_anatomy"] = {
        "scene": "tests/sgb_scenes.py pad_in_ground('B'): stage-level scene, port at z = 0.1 um",
        "walls_drawn_as_built": a,
        "walls_drawn_without_coalescing": b,
        "model_faces": {
            "as_built": built.counts["faces"],
            "every_level_cuts": cut.counts["faces"],
        },
        "model_edges": {
            "as_built": built.counts["edges"],
            "every_level_cuts": cut.counts["edges"],
        },
        "model_vertices": {
            "as_built": built.counts["vertices"],
            "every_level_cuts": cut.counts["vertices"],
        },
        "faces_by_kind_as_built": dict(
            Counter(f.kind for f in built.topology.faces.values())
        ),
        "every_level_cuts_validates": not cut.report.failures,
    }
    counts = {
        "pad_in_ground_B": {
            "scene": "stage-level scene pad_in_ground('B')",
            "faces_as_built": built.counts["faces"],
            "faces_every_level_cuts": cut.counts["faces"],
        }
    }
    scenes = {
        "pad_in_ground_B_polygon_hole32": (
            "stage-level scene pad_in_ground('B', hole_sides=32, rounded_corner=False)",
            lambda: stage_pad_in_ground("B", hole_sides=32, rounded_corner=False),
            "B",
        ),
        "airbridge_B": ("stage-level scene airbridge()", airbridge, "B"),
        "real_input_B": (
            "production real input gds_pad_in_ground",
            real_catalogue,
            "B",
        ),
        "real_input_C": (
            "production real input gds_pad_in_ground",
            real_catalogue,
            "C",
        ),
    }
    for name, (what, make, route) in scenes.items():
        catalogue = make()
        as_built, every = (
            build_model(catalogue, route),
            build_uncoalesced(catalogue, route),
        )
        counts[name] = {
            "scene": what,
            "faces_as_built": as_built.counts["faces"],
            "faces_every_level_cuts": every.counts["faces"],
            "edges_as_built": as_built.counts["edges"],
            "edges_every_level_cuts": every.counts["edges"],
            "vertices_as_built": as_built.counts["vertices"],
            "vertices_every_level_cuts": every.counts["vertices"],
        }
    numbers["coalescing_counts"] = counts


# --------------------------------------------------------------------------- shells of the airbridge vacuum volume

ZMAP = (
    (0.0, 0.0),
    (3.3, 99.0),
    (100.0, 130.0),
)  # display-only piecewise-linear z: the 3.3 um bridge drawn 30x taller


def _display_z(z):
    return np.interp(z, [p[0] for p in ZMAP], [p[1] for p in ZMAP])


def _shrunk(points: np.ndarray, tri: np.ndarray, factor: float = 0.9):
    centre = points[np.unique(tri)].mean(axis=0)
    return centre + (points - centre) * factor


def _spans_of(topo, levels, fid: str) -> int:
    f = topo.faces[fid]
    if f.kind != "wall":
        return 0
    return sum(
        1
        for k in range(len(levels) - 1)
        if f.z_range[0] <= levels[k] and levels[k + 1] <= f.z_range[1]
    )


def _figure_shells(numbers: dict) -> None:
    model = build_model(airbridge(), "B")
    topo = model.topology
    levels = model.labelled.levels.levels
    tris = compact_triangles(model, mesh_size=80.0)
    vacuum = next(v for v in topo.volumes.values() if v.domain.name == "vacuum")
    shell = vacuum.shells[0]
    spans = {fid: _spans_of(topo, levels, fid) for fid, _ in shell.faces}

    def role(fid: str) -> tuple[str, str]:
        f = topo.faces[fid]
        if f.kind == "horizontal":
            return ("horizontal face (cell of a level arrangement)", "#9ecae1")
        if spans[fid] > 1:
            return ("wall coalesced over more than one slab", PALETTE["accent"])
        if model.routed.faces[fid].interface_type == "EXT":
            return ("wall on the outer chip boundary", "#bdbdbd")
        return ("wall of one slab", "#3a6ea5")

    polys, colours, traces = [], [], []
    legend_names = {}
    for fid, _sign in shell.faces:
        points, tri = tris[fid]
        pts = np.column_stack([points[:, 0], points[:, 1], _display_z(points[:, 2])])
        pts = _shrunk(pts, tri, 0.88)
        name, colour = role(fid)
        legend_names[name] = colour
        f = topo.faces[fid]
        hover = f"{name}<br>{model.routed.face_group[fid]}<br>z {float(f.z_range[0]):g} to {float(f.z_range[1]):g} um<br>{fid}"
        alpha = (
            0.45 if spans[fid] > 1 else 0.9
        )  # the tall coalesced walls would hide the bridge behind them
        traces.append(mesh_trace(pts, tri, colour, name, hover, opacity=alpha))
        for t in tri:
            polys.append(pts[t])
            colours.append(to_rgba(colour, alpha))
    fig = go.Figure(traces)
    fig.update_layout(
        title=f"Shell of the vacuum volume (airbridge model, Route B): {len(shell.faces)} faces, each pulled toward its own centre; "
        "z 0 to 3.3 um drawn 30x taller",
        width=1100,
        height=760,
        scene=dict(
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.6),
            xaxis_title="x (µm)",
            yaxis_title="y (µm)",
            zaxis_title="z (display)",
        ),
    )
    save_plotly(fig, "assembly-shells")
    fig2 = plt.figure(figsize=(10.5, 7.2))
    ax = fig2.add_subplot(111, projection="3d")
    ax.add_collection3d(Poly3DCollection(polys, facecolors=colours, edgecolors="none"))
    ax.set_xlim(-150, 150)
    ax.set_ylim(-150, 150)
    ax.set_zlim(0, 130)
    ax.set_box_aspect((1, 1, 0.6))
    ax.view_init(elev=24, azim=-58)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_zticks([])
    ax.set_zlabel("z (display: 0 to 3.3 µm drawn 30x taller)")
    ax.set_title(
        "Vacuum volume of the airbridge model: its shell faces pulled apart (Route B)"
    )
    ax.legend(
        handles=[Patch(facecolor=c, label=n) for n, c in legend_names.items()],
        loc="upper left",
        fontsize=8,
        frameon=False,
    )
    save_svg(fig2, "assembly-shells")
    counts = Counter(role(fid)[0] for fid, _ in shell.faces)
    numbers["airbridge_vacuum_shell"] = {
        "scene": "tests/sgb_scenes.py airbridge(), Route B: stage-level scene",
        "faces": len(shell.faces),
        "by_role": dict(counts),
        "shells": len(vacuum.shells),
        "signed_volume_um3": float(shell.signed_volume) * G**2,
        "levels": [str(z) for z in levels],
        "model_counts": model.counts,
        "volumes": sorted(v.domain.name for v in topo.volumes.values()),
        "coalesced_wall_z_ranges": [
            [str(topo.faces[fid].z_range[0]), str(topo.faces[fid].z_range[1])]
            for fid, _ in shell.faces
            if spans[fid] > 1
        ],
        "coalesced_wall_groups": sorted(
            {model.routed.face_group[fid] for fid, _ in shell.faces if spans[fid] > 1}
        ),
    }


# --------------------------------------------------------------------------- inner void

ISLAND = (
    105.0,
    105.0,
    115.0,
    115.0,
    10.0,
    11.0,
)  # x0, y0, x1, y1, z0, z1 in um: 10 x 10 x 1 um inside the vacuum region


def _island_catalogue():
    base = real_catalogue()
    x0, y0, x1, y1, z0, z1 = ISLAND
    island = PrismRecord(
        "island",
        footprint(rect_um(x0, y0, x1, y1)),
        exact_z(z0),
        exact_z(z1),
        "Al",
        "conductor",
        net_id="float",
    )
    return replace(base, prisms=(*base.prisms, island))


def _figure_inner_void(numbers: dict) -> None:
    model = build_model(_island_catalogue(), "B")
    topo = model.topology
    by_name = {v.domain.name: v for v in topo.volumes.values()}
    vac = by_name["VACUUM_REGION"]
    tris = compact_triangles(model, mesh_size=150.0)
    inner = next(s for s in vac.shells if s.role == "inner")
    fig = plt.figure(figsize=(8.6, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    polys = []
    for fid, _ in inner.faces:
        points, tri = tris[fid]
        polys.extend(points[t] for t in tri)
    ax.add_collection3d(
        Poly3DCollection(
            polys,
            facecolors="#e8a09a",
            edgecolors=PALETTE["bad"],
            linewidths=0.6,
            alpha=0.9,
        )
    )
    ax.set_xlim(100, 120)
    ax.set_ylim(100, 120)
    ax.set_zlim(7, 14)
    ax.set_box_aspect((1, 1, 0.45))
    ax.view_init(elev=24, azim=-52)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_zlabel("z (µm)")
    ax.set_title(
        "Inner shell of the vacuum volume: the six faces of a floating metal island (10 x 10 x 1 µm)",
        fontsize=9,
    )
    save_svg(fig, "assembly-inner-void")
    numbers["inner_void"] = {
        "scene": "gds_pad_in_ground real input + island (Net float, x 105..115, y 105..115, z 10..11 um), Route B",
        "volumes": {
            v.domain.name: {
                "emitted": v.id in model.routed.emitted_volumes,
                "shells": [
                    {
                        "role": s.role,
                        "faces": len(s.faces),
                        "signed_volume_um3": float(s.signed_volume) * G**2,
                    }
                    for s in v.shells
                ],
            }
            for v in topo.volumes.values()
        },
        "validation_passed": not model.report.failures,
        "shell_checks": [
            c.detail
            for c in model.report.checks
            if c.name in ("shells_closed_and_oriented", "shell_euler_characteristic")
        ],
        "counts": model.counts,
    }


def figure_curved_wall_split() -> None:
    """Conceptual: a cylinder wall whose arc carries a foreign vertex, and the planned fix (split by a vertical edge)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.9), sharey=True)
    # developed (unrolled) wall: u = arc length, z = height; the foreign vertex splits the arc at u = 0.4
    for ax in (ax1, ax2):
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.25, 1.3)
        ax.set_xticks([])
        ax.set_yticks([0, 1], ["z bottom", "z top"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    ax1.add_patch(
        Polygon(
            [(0, 0), (1, 0), (1, 1), (0, 1)],
            closed=True,
            facecolor="#cfe0f1",
            edgecolor="none",
        )
    )
    ax1.plot([0, 1], [0, 0], color=PALETTE["ink"], lw=2)
    ax1.plot([0, 0.4], [1, 1], color=PALETTE["ink"], lw=2)
    ax1.plot([0.4, 1], [1, 1], color=PALETTE["accent"], lw=2)
    ax1.plot([0, 0], [0, 1], color=PALETTE["bad"], lw=2.4)
    ax1.plot([1, 1], [0, 1], color=PALETTE["bad"], lw=2.4)
    ax1.plot([0.4], [1], "o", color=PALETTE["bad"], ms=6)
    ax1.annotate(
        "a foreign vertex on the top arc\n(e.g. a port edge ending on a round pad)",
        (0.4, 1),
        (0.1, 1.17),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", lw=0.8),
    )
    ax1.text(
        0.5,
        0.5,
        "5 boundary edges:\n1 arc below, 2 arcs above,\n2 vertical edges",
        ha="center",
        va="center",
        fontsize=9,
    )
    ax1.set_title(
        "Without a split: the two top arcs lie on\nslightly different circles, so the wall is\nnot one cylinder patch",
        fontsize=9,
    )
    ax2.add_patch(
        Polygon(
            [(0, 0), (0.4, 0), (0.4, 1), (0, 1)],
            closed=True,
            facecolor="#cfe0f1",
            edgecolor="none",
        )
    )
    ax2.add_patch(
        Polygon(
            [(0.4, 0), (1, 0), (1, 1), (0.4, 1)],
            closed=True,
            facecolor="#e8d9a8",
            edgecolor="none",
        )
    )
    ax2.plot([0, 1], [0, 0], color=PALETTE["ink"], lw=2)
    ax2.plot([0, 0.4], [1, 1], color=PALETTE["ink"], lw=2)
    ax2.plot([0.4, 1], [1, 1], color=PALETTE["accent"], lw=2)
    for x in (0, 0.4, 1):
        ax2.plot([x, x], [0, 1], color=PALETTE["bad"], lw=2.4)
    ax2.plot([0.4], [0], "o", color=PALETTE["bad"], ms=6)
    ax2.plot([0.4], [1], "o", color=PALETTE["bad"], ms=6)
    ax2.text(0.2, 0.5, "piece 1\n4 edges", ha="center", va="center", fontsize=9)
    ax2.text(0.7, 0.5, "piece 2\n4 edges", ha="center", va="center", fontsize=9)
    ax2.set_title(
        "With the planned split: a vertical edge at the\nforeign vertex (bottom arc split there too);\neach piece is an exact cylinder on its own circle",
        fontsize=9,
    )
    fig.suptitle(
        "Conceptual drawing (not computed): a curved wall developed flat; red = vertical edges, black and orange = arcs",
        fontsize=9,
        y=1.02,
    )
    fig.tight_layout()
    save_svg(fig, "assembly-curved-split")


def real_example(numbers: dict) -> None:
    """The running example itself: its faces by kind, and the walls that span more than one slab."""
    model = build_model(real_catalogue(), "B")
    topo = model.topology
    levels = model.labelled.levels.levels
    walls = [f for f in topo.faces.values() if f.kind == "wall"]
    per_group = Counter(model.routed.face_group[f.id] for f in walls)
    multi = [f for f in walls if _spans_of(topo, levels, f.id) > 1]
    numbers["real_input_B"] = {
        "levels": [str(z) for z in levels],
        "model_counts": model.counts,
        "faces_by_kind": dict(Counter(f.kind for f in topo.faces.values())),
        "walls_by_group": dict(per_group),
        "walls_spanning_more_than_one_slab": {
            "count": len(multi),
            "groups": sorted({model.routed.face_group[f.id] for f in multi}),
            "z_ranges": sorted({f"{f.z_range[0]}..{f.z_range[1]}" for f in multi}),
        },
        "volumes": {
            v.domain.name: {
                "emitted": v.id in model.routed.emitted_volumes,
                "shells": [(s.role, len(s.faces)) for s in v.shells],
            }
            for v in topo.volumes.values()
        },
        "edges_by_kind": dict(Counter(e.kind for e in topo.edges.values())),
    }


def main() -> None:
    """Write the assembly page's wall, shell and inner-void figures and their numbers."""
    numbers: dict = {}
    real_example(numbers)
    _figure_wall_anatomy(numbers)
    _figure_shells(numbers)
    _figure_inner_void(numbers)
    figure_curved_wall_split()
    write_numbers("assembly", numbers)
    print("assembly: done")


if __name__ == "__main__":
    main()
