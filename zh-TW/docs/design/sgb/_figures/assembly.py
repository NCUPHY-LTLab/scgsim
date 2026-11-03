"""Figures and numbers of ``assembly.qmd`` (computed from the prototype's planned topology).

Writes into ``docs/design/sgb/figures/``:

* ``assembly-wall-anatomy.svg``  the walls along three consecutive pad sidewalls of Route B, as built (coalesced) and as
  they would be if every level cut every wall; vertical edges, chains and shared vertices are drawn from the topology;
* ``assembly-shells.svg/.html``  the shell of the Ground volume of the airbridge model, every face pulled slightly apart
  so shared edges show as gaps; faces are coloured by role and the coalesced walls are highlighted;
* ``assembly-inner-void.svg``    a floating metal island inside the vacuum: the vacuum volume gets a second, inner shell;
* ``assembly-numbers.json``      the counts the page quotes.

The "every level cuts every wall" counts are computed by replacing the prototype's ``_level_cuts_wall`` test with a
constant ``True`` while building (the script restores it); nothing else changes.

Run (from the package root):  .venv/bin/python docs/design/sgb/_figures/assembly.py
"""

from __future__ import annotations

import math
from collections import Counter

from labels_common import (
    PALETTE, compact_triangles, go, mesh_trace, np, plt, save_plotly, save_svg, write_numbers,
)
from matplotlib.patches import Patch, Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from sgbproto import assembly as assembly_module
from sgbproto.pipeline import build
from sgbproto.records import Footprint, ModelSpec, PrismRecord, z_value
from sgbproto.scenes import airbridge, pad_in_ground, rect_um, to_grid

G = 0.001


def build_uncoalesced(spec):
    """Build with the coalescing test replaced by 'every level cuts every wall' (restored afterwards)."""
    original = assembly_module._level_cuts_wall
    assembly_module._level_cuts_wall = lambda *args: True
    try:
        return build(spec)
    finally:
        assembly_module._level_cuts_wall = original


# --------------------------------------------------------------------------- wall anatomy

SEGMENTS = [  # the pad outline as unrolled: ((x0, y0), (x1, y1)), s offset (um)
    ((10.0, -30.0), (50.0, -30.0), 0.0),
    ((50.0, -30.0), (50.0, 18.0), 40.0),
    ((50.0, 18.0), (38.0, 30.0), 88.0),
]
LEVEL_INDEX = {0.0: 0, 0.1: 1, 0.2: 2}


def unrolled_walls(topo):
    """(face, polygon in (s, level index), vertical edge ids, edge list) of the pad walls on SEGMENTS."""
    out = []
    for f in topo.faces.values():
        if f.kind != "wall" or f.group != "MA::pad":
            continue
        verts = []
        for eid, sign in f.loops[0]:
            e = topo.edges[eid]
            v = topo.vertices[e.v0 if sign > 0 else e.v1]
            verts.append((v.x * G, v.y * G, float(v.z), e.kind, eid))
        for (ax, ay), (bx, by), s0 in SEGMENTS:
            length = math.hypot(bx - ax, by - ay)
            ux, uy = (bx - ax) / length, (by - ay) / length
            on = [abs((x - ax) * uy - (y - ay) * ux) < 1e-6 and -1e-6 <= (x - ax) * ux + (y - ay) * uy <= length + 1e-6
                  for x, y, _, _, _ in verts]
            if all(on):
                poly = [(s0 + (x - ax) * ux + (y - ay) * uy, LEVEL_INDEX[round(z, 3)]) for x, y, z, _, _ in verts]
                edges = []
                for k, (eid, sign) in enumerate(f.loops[0]):
                    a, b = poly[k], poly[(k + 1) % len(poly)]
                    edges.append((a, b, topo.edges[eid].kind, eid))
                out.append((f, poly, edges))
                break
    return out


def draw_walls(ax, walls, title: str) -> dict:
    cmap = ["#cfe0f1", "#e8d9a8", "#cfe8d4", "#f1cfcf", "#d9cfe8", "#e0e0e0"]
    vertical_segments = 0
    for k, (f, poly, edges) in enumerate(sorted(walls, key=lambda w: (min(p[0] for p in w[1]), min(p[1] for p in w[1])))):
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        # a small inset keeps neighbouring faces apart so that shared edges are visible as two lines
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        shrink = [(cx + (x - cx) * 0.985, cy + (y - cy) * 0.94) for x, y in poly]
        ax.add_patch(Polygon(shrink, closed=True, facecolor=cmap[k % len(cmap)], edgecolor="none", alpha=0.9))
        for a, b, kind, eid in edges:
            colour, width = (PALETTE["bad"], 2.4) if kind == "vertical" else (PALETTE["ink"], 1.4)
            vertical_segments += kind == "vertical"
            sa = (cx + (a[0] - cx) * 0.985, cy + (a[1] - cy) * 0.94)
            sb = (cx + (b[0] - cx) * 0.985, cy + (b[1] - cy) * 0.94)
            ax.plot([sa[0], sb[0]], [sa[1], sb[1]], color=colour, lw=width, solid_capstyle="round")
        for x, y in shrink:
            ax.plot(x, y, "o", ms=3.2, color=PALETTE["ink"], zorder=5)
        horiz_bottom = sum(1 for a, b, kind, _ in edges if kind == "horizontal" and a[1] == b[1] == min(ys))
        horiz_top = sum(1 for a, b, kind, _ in edges if kind == "horizontal" and a[1] == b[1] == max(ys))
        ax.text(cx, cy, f"wall {f.id[-4:]}\nbottom {horiz_bottom} edge{'s' * (horiz_bottom != 1)}\ntop {horiz_top} edge{'s' * (horiz_top != 1)}",
                ha="center", va="center", fontsize=7)
    ax.set_xlim(-3, 108)
    ax.set_ylim(-0.25, 2.35)
    ax.set_yticks([0, 1, 2], ["z = 0", "z = 0.1 (port plane)", "z = 0.2"])
    ax.set_xlabel("distance along the pad outline from (10, -30) µm: bottom edge, x = 50 sidewall, chamfer")
    ax.set_title(title, fontsize=9)
    for s in (40, 88):
        ax.axvline(s, color="#999", lw=0.6, ls=":")
    return {"faces": len(walls), "vertical_edge_segments_in_loops": vertical_segments}


def figure_wall_anatomy(numbers: dict) -> None:
    spec = pad_in_ground("B")
    built = build(spec)
    cut = build_uncoalesced(spec)
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.4), sharex=True)
    a = draw_walls(axes[0], unrolled_walls(built.topology), "As built: a level cuts a wall only where something at that level touches it")
    b = draw_walls(axes[1], unrolled_walls(cut.topology), "If every level cut every wall (same model, coalescing test disabled)")
    fig.legend(handles=[plt.Line2D([0], [0], color=PALETTE["ink"], lw=1.4, label="horizontal edge (chain of the level arrangement)"),
                        plt.Line2D([0], [0], color=PALETTE["bad"], lw=2.4, label="vertical edge segment of a wall column"),
                        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["ink"], label="vertex")],
               loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_svg(fig, "assembly-wall-anatomy")
    numbers["wall_anatomy"] = {
        "walls_drawn_as_built": a, "walls_drawn_without_coalescing": b,
        "model_faces": {"as_built": built.counts["faces"], "every_level_cuts": cut.counts["faces"]},
        "model_edges": {"as_built": built.counts["edges"], "every_level_cuts": cut.counts["edges"]},
        "model_vertices": {"as_built": built.counts["vertices"], "every_level_cuts": cut.counts["vertices"]},
        "every_level_cuts_validates": not cut.report.failures,
    }
    for name, make in (("pad_in_ground_B_polygon_hole32", lambda: pad_in_ground("B", hole_sides=32, rounded_corner=False)),
                       ("airbridge_B", airbridge)):
        s = make()
        numbers.setdefault("coalescing_counts", {})[name] = {
            "faces_as_built": build(s).counts["faces"], "faces_every_level_cuts": build_uncoalesced(s).counts["faces"]}
    numbers["coalescing_counts"]["pad_in_ground_B"] = {"faces_as_built": built.counts["faces"],
                                                       "faces_every_level_cuts": cut.counts["faces"]}


def triangles_of_all_volumes(topo, mesh_size: float):
    """Triangles of every shell face, also those of the excluded metal volumes (flag restored afterwards)."""
    saved = {vid: v.emitted for vid, v in topo.volumes.items()}
    for v in topo.volumes.values():
        v.emitted = True
    try:
        return compact_triangles(topo, mesh_size)
    finally:
        for vid, flag in saved.items():
            topo.volumes[vid].emitted = flag


# --------------------------------------------------------------------------- shells of the airbridge Ground volume

ZS = 30.0  # display-only z stretch of the airbridge figure


def shrunk(points: np.ndarray, tri: np.ndarray, factor: float = 0.9):
    centre = points[np.unique(tri)].mean(axis=0)
    return centre + (points - centre) * factor


def figure_shells(numbers: dict) -> None:
    result = build(airbridge())
    topo = result.topology
    tris = triangles_of_all_volumes(topo, mesh_size=80.0)
    ground = next(v for v in topo.volumes.values() if v.domain.name == "Ground")
    shell = ground.shells[0]
    spans_slabs = {}
    levels = result.levels.levels
    for fid, sign in shell.faces:
        f = topo.faces[fid]
        n = sum(1 for k in range(len(levels) - 1) if f.z_range[0] <= levels[k] and levels[k + 1] <= f.z_range[1]) if f.kind == "wall" else 0
        spans_slabs[fid] = n

    def role(fid: str) -> tuple[str, str]:
        f = topo.faces[fid]
        if f.kind == "horizontal":
            return ("horizontal face (cell of a level arrangement)", "#9ecae1")
        if spans_slabs[fid] > 1:
            return ("wall coalesced over more than one slab", PALETTE["accent"])
        if f.group.startswith("EXT"):
            return ("wall on the outer chip boundary", "#bdbdbd")
        return ("wall of one slab", "#3a6ea5")

    polys, colours, traces = [], [], []
    legend_names = {}
    for fid, sign in shell.faces:
        points, tri = tris[fid]
        pts = np.column_stack([points[:, 0], points[:, 1], points[:, 2] * ZS])
        pts = shrunk(pts, tri, 0.88)
        name, colour = role(fid)
        legend_names[name] = colour
        f = topo.faces[fid]
        hover = f"{name}<br>{f.group}<br>z {float(f.z_range[0]):g} to {float(f.z_range[1]):g} um<br>{fid}"
        traces.append(mesh_trace(pts, tri, colour, name, hover, opacity=0.85))
        for t in tri:
            polys.append(pts[t])
            colours.append(colour)
    fig = go.Figure(traces)
    fig.update_layout(title="Shell of the Ground volume (airbridge model, Route B): 24 faces, each pulled toward its own centre; "
                      f"z drawn {ZS:g}x taller", width=1100, height=760,
                      scene=dict(aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.5),
                                 xaxis_title="x (µm)", yaxis_title="y (µm)", zaxis_title=f"z (µm x {ZS:g})"))
    save_plotly(fig, "assembly-shells")
    fig2 = plt.figure(figsize=(10.5, 7.2))
    ax = fig2.add_subplot(111, projection="3d")
    ax.add_collection3d(Poly3DCollection(polys, facecolors=colours, edgecolors="none", alpha=0.9))
    ax.set_xlim(-150, 150)
    ax.set_ylim(-150, 150)
    ax.set_zlim(0, 3.3 * ZS)
    ax.set_box_aspect((1, 1, 0.5))
    ax.view_init(elev=24, azim=-58)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_zlabel(f"z (µm x {ZS:g})")
    ax.set_title("Ground volume of the airbridge model: its 24 shell faces pulled apart (Route B)")
    ax.legend(handles=[Patch(facecolor=c, label=n) for n, c in legend_names.items()], loc="upper left", fontsize=8, frameon=False)
    save_svg(fig2, "assembly-shells")
    counts = Counter(role(fid)[0] for fid, _ in shell.faces)
    numbers["airbridge_ground_shell"] = {
        "faces": len(shell.faces), "by_role": dict(counts), "signed_volume_um3": float(shell.signed_volume) * G**2,
        "levels": [str(z) for z in levels], "model_counts": result.counts,
        "coalesced_wall_z_ranges": [[str(topo.faces[fid].z_range[0]), str(topo.faces[fid].z_range[1])]
                                    for fid, _ in shell.faces if role(fid)[0].startswith("wall coalesced")],
    }


# --------------------------------------------------------------------------- inner void


def island_spec() -> ModelSpec:
    base = pad_in_ground("B")
    footprint = Footprint.from_rings(to_grid(base.grid, rect_um(100, 100, 120, 120)))
    island = PrismRecord("island", footprint, z_value(50), z_value(51), "Al", "conductor", net_id="float")
    return ModelSpec(base.grid, base.prisms + (island,), name="pad_in_ground_B_island", meta=base.meta)


def figure_inner_void(numbers: dict) -> None:
    result = build(island_spec())
    topo = result.topology
    by_name = {v.domain.name: v for v in topo.volumes.values()}
    vac = by_name["vacuum"]
    tris = triangles_of_all_volumes(topo, mesh_size=150.0)
    inner = next(s for s in vac.shells if s.role == "inner")
    fig = plt.figure(figsize=(8.6, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    polys = []
    for fid, _ in inner.faces:
        points, tri = tris[fid]
        polys.extend(points[t] for t in tri)
    ax.add_collection3d(Poly3DCollection(polys, facecolors="#e8a09a", edgecolors=PALETTE["bad"], linewidths=0.6, alpha=0.9))
    ax.set_xlim(95, 125)
    ax.set_ylim(95, 125)
    ax.set_zlim(46, 55)
    ax.set_box_aspect((1, 1, 0.45))
    ax.view_init(elev=24, azim=-52)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_zlabel("z (µm)")
    ax.set_title("Inner shell of the vacuum volume: the six faces of a floating metal island (20 x 20 x 1 µm)", fontsize=9)
    save_svg(fig, "assembly-inner-void")
    numbers["inner_void"] = {
        "scene": "pad_in_ground('B') + island (Net float, x 100..120, y 100..120, z 50..51 um)",
        "volumes": {v.domain.name: {"emitted": v.emitted, "shells": [{"role": s.role, "faces": len(s.faces), "signed_volume_um3": float(s.signed_volume) * G**2}
                                                                  for s in v.shells]} for v in topo.volumes.values()},
        "validation_passed": not result.report.failures,
        "shell_checks": [c.detail for c in result.report.checks if c.name in ("shells_closed_and_oriented", "shell_euler_characteristic")],
        "counts": result.counts,
    }


def main() -> None:
    numbers: dict = {}
    figure_wall_anatomy(numbers)
    figure_shells(numbers)
    figure_inner_void(numbers)
    write_numbers("assembly", numbers)
    print("assembly: done")


if __name__ == "__main__":
    main()
