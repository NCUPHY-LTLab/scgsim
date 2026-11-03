"""Figures of the overview page: the pad-in-ground running example as input and as derived solids.

Writes ``figures/index-example-input.svg`` (plan view of the source prisms and an x-z section with levels),
``figures/index-example-3d.html`` plus ``.svg`` (exploded volumes, computed from the planned topology) and
``figures/index-numbers.json`` (the counts the page quotes).

Run:  .venv/bin/python docs/design/sgb/_figures/index_example.py   (from the package root)
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from _intro_common import (
    COLOURS, OUTPUTS, draw_footprint, draw_section, save_plotly, save_svg, stretch_z, write_numbers,
)

from sgbproto.emit import surface_triangles
from sgbproto.levels import build_levels
from sgbproto.pipeline import build
from sgbproto.scenes import airbridge, pad_in_ground

GROUP_COLOURS = {"SA": "#9ecae1", "MA": "#2171b5", "MS": "#6a51a3", "EXT": "#bdbdbd", "PORT": "#d62728"}
GROUP_NAMES = {
    "SA": "SA  substrate|vacuum gap surface",
    "MA": "MA  metal-vacuum surface",
    "MS": "MS  metal-substrate surface",
    "EXT": "EXT  outer chip boundary",
    "PORT": "PORT  lumped-port face",
}
SHIFT = {"substrate": -60.0, "vacuum": 60.0}  # exploded view offsets (display units)


def input_figure(spec) -> None:
    levels = build_levels(spec.prisms, spec.families).levels
    fig, (ax_plan, ax_sec) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    for prism in spec.prisms:
        if prism.entity_id in ("substrate", "vacuum"):
            continue
        colour = {"ground": COLOURS["ground"], "pad": COLOURS["pad"], "p1": COLOURS["port"]}[prism.entity_id]
        draw_footprint(ax_plan, prism.footprint, facecolor=colour, edgecolor="#222", lw=0.6, alpha=0.9 if prism.entity_id != "p1" else 1.0)
    ax_plan.add_patch(plt.Rectangle((-150, -150), 300, 300, fill=False, edgecolor="#555", lw=0.8, ls="--"))
    ax_plan.annotate("ground plane (cutout 160 × 120 µm)", (-140, 120), fontsize=8, ha="left", bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax_plan.annotate("pad: chamfer, reflex notch,\nrounded corner, circular hole", (-5, 28), xytext=(-120, 75), fontsize=8, ha="left",
                     arrowprops=dict(arrowstyle="-", color="#222", lw=0.6), bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax_plan.annotate("lumped port p1", (65, 0), xytext=(40, -45), fontsize=8, ha="left", color=COLOURS["port"],
                     arrowprops=dict(arrowstyle="-", color=COLOURS["port"], lw=0.8), bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax_plan.annotate("substrate (z < 0) and vacuum (z > 0)\nboth span the 300 × 300 µm chip", (0, -140), fontsize=8, ha="center", bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax_plan.set_xlim(-160, 160)
    ax_plan.set_ylim(-160, 160)
    ax_plan.set_aspect("equal")
    ax_plan.set_xlabel("x (µm)")
    ax_plan.set_ylabel("y (µm)")
    ax_plan.set_title("Source prisms, plan view (footprints)")
    ax_plan.axhline(0, color="#e08a1e", lw=0.8, ls=":")
    ax_plan.text(-158, 3, "section y = 0", color="#e08a1e", fontsize=8)
    draw_section(ax_sec, spec, levels, 0.0)
    ax_sec.set_title("x-z section at y = 0 (levels equally spaced, not to scale)")
    ax_sec.legend(handles=[Line2D([0], [0], color=c, lw=6, label=n) for n, c in
                           (("substrate", COLOURS["substrate"]), ("vacuum", COLOURS["vacuum"]), ("ground", COLOURS["ground"]),
                            ("pad", COLOURS["pad"]), ("port sheet", COLOURS["port"]))],
                  loc="upper center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save_svg(fig, "index-example-input")


def outline_segments(pts, tris, z):
    """Boundary edges of one face's triangulation (edges used by exactly one triangle) as 3-D point pairs."""
    count: dict[tuple[int, int], int] = {}
    for a, b, c in tris:
        for u, v in ((a, b), (b, c), (c, a)):
            key = (min(u, v), max(u, v))
            count[key] = count.get(key, 0) + 1
    return [((float(pts[u, 0]), float(pts[u, 1]), float(z[u])), (float(pts[v, 0]), float(pts[v, 1]), float(z[v])))
            for (u, v), n in count.items() if n == 1]


def exploded_traces(topo, tri):
    """Face triangles grouped by (volume, group) with the exploded offset and the stretched metal layer applied."""
    batches: dict[tuple[str, str], dict] = {}
    for vol in topo.volumes.values():
        dz = SHIFT.get(vol.domain.name, 0.0)
        status = "emitted" if vol.emitted else "excluded PEC volume"
        for shell in vol.shells:
            for fid, _ in shell.faces:
                if fid not in tri:
                    continue
                xyz, tris = tri[fid]
                used, local = np.unique(tris, return_inverse=True)  # keep only the nodes of this face
                pts, tris = xyz[used], local.reshape(-1, 3)
                group = topo.faces[fid].group.split("::")[0]
                z = np.array([stretch_z(v) for v in pts[:, 2]]) + dz
                batch = batches.setdefault((f"{vol.domain.name} ({status})", group), {"x": [], "y": [], "z": [], "i": [], "j": [], "k": []})
                base = len(batch["x"])
                batch["x"] += [round(float(v), 2) for v in pts[:, 0]]
                batch["y"] += [round(float(v), 2) for v in pts[:, 1]]
                batch["z"] += [round(float(v), 2) for v in z]
                batch["i"] += [int(v) for v in tris[:, 0] + base]
                batch["j"] += [int(v) for v in tris[:, 1] + base]
                batch["k"] += [int(v) for v in tris[:, 2] + base]
                batch.setdefault("outline", []).extend(outline_segments(pts, tris, z))
    return batches


def exploded_figures(result) -> int:
    topo = result.topology
    tri = surface_triangles(topo, mesh_size=60.0, mesh_size_min=4.0)
    batches = exploded_traces(topo, tri)
    fig = go.Figure()
    shown = set()
    for (volume, group), b in sorted(batches.items()):
        fig.add_trace(go.Mesh3d(x=b["x"], y=b["y"], z=b["z"], i=b["i"], j=b["j"], k=b["k"], color=GROUP_COLOURS[group], opacity=0.6,
                                flatshading=True, name=GROUP_NAMES[group], legendgroup=group, showlegend=group not in shown,
                                hovertext=f"{volume}: {group} faces", hoverinfo="text"))
        shown.add(group)
    fig.update_layout(
        title="pad_in_ground_B, exploded: substrate moved down, vacuum up, the two excluded metal volumes in place",
        height=640, legend=dict(orientation="h", y=-0.02),
        scene=dict(aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.6), xaxis_title="x (µm)", yaxis_title="y (µm)",
                   zaxis=dict(title="z (µm; metal layer drawn 200× thicker)", range=[-200, 200]),
                   camera=dict(eye=dict(x=1.5, y=-1.6, z=0.9))),
    )
    xs, ys, zs = [], [], []
    for b in batches.values():
        for p, q in b["outline"]:
            xs += [round(p[0], 2), round(q[0], 2), None]; ys += [round(p[1], 2), round(q[1], 2), None]; zs += [round(p[2], 2), round(q[2], 2), None]
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color="#222", width=2), name="face boundaries", hoverinfo="skip"))
    save_plotly(fig, "index-example-3d")

    polys, colours = [], []
    for (volume, group), b in batches.items():
        pts = np.column_stack([b["x"], b["y"], b["z"]])
        for i, j, k in zip(b["i"], b["j"], b["k"]):
            polys.append(pts[[i, j, k]])
            colours.append(GROUP_COLOURS[group])
    f = plt.figure(figsize=(8.5, 6.2))
    ax = f.add_subplot(111, projection="3d")
    ax.add_collection3d(Poly3DCollection(polys, facecolors=colours, edgecolors="none", alpha=0.35))
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    ax.add_collection3d(Line3DCollection([seg for b in batches.values() for seg in b["outline"]], colors="#222", linewidths=0.6))
    ax.set_xlim(-150, 150); ax.set_ylim(-150, 150); ax.set_zlim(-200, 200)
    ax.set_box_aspect((1, 1, 0.6)); ax.view_init(elev=22, azim=-55)
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)"); ax.set_zlabel("z (display)")
    ax.legend(handles=[Line2D([0], [0], color=c, lw=6, label=GROUP_NAMES[g]) for g, c in GROUP_COLOURS.items()],
              loc="upper left", fontsize=7, frameon=False)
    ax.set_title("pad_in_ground_B, exploded (metal layer drawn 200× thicker)")
    save_svg(f, "index-example-3d")
    return sum(len(t[1]) for t in tri.values())


def centre_vertices() -> dict:
    """Planned vertices that only carry an arc's centre (rebuilt from the prototype; not written by the direct writer)."""
    specs = {"pad_in_ground_A": pad_in_ground("A"), "pad_in_ground_B": pad_in_ground("B"),
             "pad_in_ground_B_bands": pad_in_ground("B", bands_um=(0.1, 0.5, 2.0)), "airbridge_B": airbridge()}
    out = {}
    for case, spec in specs.items():
        topo = build(spec).topology
        centres = {e.centre_vertex for e in topo.edges.values() if e.centre_vertex}
        volumes = {v.id: v for v in topo.volumes.values()}
        unwritten = [f for f in topo.faces.values() if not any(volumes[v].emitted for v in (f.neg_volume, f.pos_volume) if v)]
        used = {e for f in topo.faces.values() if f not in unwritten for loop in f.loops for e, _ in loop}
        out[case] = {"planned_vertices": len(topo.vertices), "arc_centre_vertices": len(centres),
                     "arc_edges": sum(1 for e in topo.edges.values() if e.curve is not None),
                     "faces_bounding_no_emitted_volume": len(unwritten), "their_groups": sorted({f.group for f in unwritten}),
                     "edges_only_on_those_faces": sum(1 for e in topo.edges.values() if e.id not in used and e.kind != "refinement")}
    return out


def counts() -> dict:
    """Planned counts (stage summaries) and what the direct writer wrote and Gmsh loaded back (e3_brep validation)."""
    out = {}
    for case in ("pad_in_ground_A", "pad_in_ground_B", "pad_in_ground_B_bands", "airbridge_B"):
        s4 = json.loads((OUTPUTS / case / "stage_summary.json").read_text())["stages"]
        written = json.loads((OUTPUTS / "e3_brep" / case / "validation.json").read_text())
        out[case] = {
            "prisms": s4["S2_prism_catalogue"]["prisms"],
            "levels": s4["S2_prism_catalogue"]["levels"],
            "slabs": s4["S2_prism_catalogue"]["slabs"],
            "topology": s4["S6_topology"],
            "groups": s4["S6_topology"].get("groups", len(s4["S8_groups"])),
            "validation_checks": s4["S9_validation"],
            "written_by_direct_writer": written["verify_brep"]["counts"]["native"],
            "gmsh_boundary_mismatches": written["verify_brep"]["boundaries_mismatching"],
            "gmsh_loaded_counts_equal_planned": written["verify_brep"]["counts"]["native"] == written["verify_brep"]["counts"]["planned"],
        }
    return out


def main() -> None:
    spec = pad_in_ground("B")
    result = build(spec)
    input_figure(spec)
    triangles = exploded_figures(result)
    numbers = counts()
    for case, extra in centre_vertices().items():
        numbers[case]["arc_centre_vertices"] = extra
        assert numbers[case]["written_by_direct_writer"]["0"] == extra["planned_vertices"] - extra["arc_centre_vertices"]
    numbers["exploded_view"] = {"surface_triangles_drawn": triangles, "mesh_size_um": 60.0, "mesh_size_min_um": 4.0,
                                "volume_domains": sorted(f"{v.domain.name} emitted={v.emitted}" for v in result.topology.volumes.values())}
    write_numbers("index", numbers)
    print(json.dumps(numbers, indent=1)[:1200])


if __name__ == "__main__":
    main()
