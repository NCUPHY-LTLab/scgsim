"""Figures of the overview page: the pad-in-ground running example as input and as derived solids, computed with
the production package ``scgsim.geometry.sgb``.

The running example is the test scene ``sgb_scenes.pad_in_ground`` (a stage-level prism catalogue with the pad's
chamfer, reflex notch, rounded corner and circular hole as exact arcs) with its lumped port on the hosts' support
plane z = 0, where ``prisms.ports`` puts it for real input; see ``intro_common.running_example``.

Writes ``figures/index-example-input.svg`` (plan view of the source prisms and an x-z section with levels),
``figures/index-example-3d.html`` plus ``.svg`` (the written XAO of Route B, every face drawn once for each volume it
bounds, pulled apart) and ``figures/index-numbers.json`` (the counts the page quotes).

Run (from the repository root; see ``intro_common.py`` for ``SGB_CORE``)::

    $SGB_CORE/.artifacts/env/bin/python docs/design/sgb/_figures/index_example.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import gmsh
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from intro_common import (
    COLOURS,
    build_example,
    draw_footprint,
    draw_section,
    running_example,
    save_plotly,
    save_svg,
    stretch_z,
    write_numbers,
)
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from sgb_catalogue_report import measure
from sgb_emit_cases import CASES, pad_requests
from sgb_gmsh_oracle import gmsh_session, mesh_quality, mesh_with_cheap_recipe
from sgb_scenes import airbridge

from scgsim.contracts.sgb_sidecar import read_sidecar
from scgsim.geometry.sgb.build import emit_model
from scgsim.geometry.sgb.checks.verification.loaded import apply_sidecar_embeds
from scgsim.geometry.sgb.model import build_model_from_catalogue
from scgsim.geometry.sgb.prisms.levels import build_levels

GROUP_COLOURS = {
    "SA": "#9ecae1",
    "MA": "#2171b5",
    "MS": "#6a51a3",
    "EXT": "#bdbdbd",
    "PORT": "#d62728",
}
GROUP_NAMES = {
    "SA": "SA  substrate|vacuum gap surface",
    "MA": "MA  metal-vacuum surface",
    "MS": "MS  metal-substrate surface",
    "EXT": "EXT  outer chip boundary",
    "PORT": "PORT  lumped-port face",
}
SHIFT = {"substrate": -60.0, "vacuum": 60.0}  # exploded view offsets (display units)


def _input_figure(catalogue) -> None:
    levels = build_levels(catalogue.prisms, catalogue.families).levels
    fig, (ax_plan, ax_sec) = plt.subplots(
        1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    for prism in catalogue.prisms:
        if prism.entity_id in ("substrate", "vacuum"):
            continue
        colour = {
            "ground": COLOURS["ground"],
            "pad": COLOURS["pad"],
            "p1": COLOURS["port"],
        }[prism.entity_id]
        draw_footprint(
            ax_plan,
            prism.footprint,
            facecolor=colour,
            edgecolor="#222",
            lw=0.6,
            alpha=0.9 if prism.entity_id != "p1" else 1.0,
        )
    ax_plan.add_patch(
        plt.Rectangle(
            (-150, -150), 300, 300, fill=False, edgecolor="#555", lw=0.8, ls="--"
        )
    )
    ax_plan.annotate(
        "ground plane (cutout 160 × 120 µm)",
        (-140, 120),
        fontsize=8,
        ha="left",
        bbox=dict(facecolor="white", edgecolor="none", pad=1),
    )
    ax_plan.annotate(
        "pad: chamfer, reflex notch,\nrounded corner, circular hole",
        (-5, 28),
        xytext=(-120, 75),
        fontsize=8,
        ha="left",
        arrowprops=dict(arrowstyle="-", color="#222", lw=0.6),
        bbox=dict(facecolor="white", edgecolor="none", pad=1),
    )
    ax_plan.annotate(
        "lumped port p1",
        (65, 0),
        xytext=(40, -45),
        fontsize=8,
        ha="left",
        color=COLOURS["port"],
        arrowprops=dict(arrowstyle="-", color=COLOURS["port"], lw=0.8),
        bbox=dict(facecolor="white", edgecolor="none", pad=1),
    )
    ax_plan.annotate(
        "substrate (z < 0) and vacuum (z > 0)\nboth span the 300 × 300 µm chip",
        (0, -140),
        fontsize=8,
        ha="center",
        bbox=dict(facecolor="white", edgecolor="none", pad=1),
    )
    ax_plan.set_xlim(-160, 160)
    ax_plan.set_ylim(-160, 160)
    ax_plan.set_aspect("equal")
    ax_plan.set_xlabel("x (µm)")
    ax_plan.set_ylabel("y (µm)")
    ax_plan.set_title("Source prisms, plan view (footprints)")
    ax_plan.axhline(0, color="#e08a1e", lw=0.8, ls=":")
    ax_plan.text(-158, 3, "section y = 0", color="#e08a1e", fontsize=8)
    draw_section(ax_sec, catalogue, levels, 0.0)
    ax_sec.set_title("x-z section at y = 0 (levels equally spaced, not to scale)")
    ax_sec.legend(
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
        ncol=3,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.12),
    )
    fig.tight_layout()
    save_svg(fig, "index-example-input")


def outline_segments(pts, tris, z):
    """Boundary edges of one face's triangulation (edges used by exactly one triangle) as 3-D point pairs."""
    count: dict[tuple[int, int], int] = {}
    for a, b, c in tris:
        for u, v in ((a, b), (b, c), (c, a)):
            key = (min(u, v), max(u, v))
            count[key] = count.get(key, 0) + 1
    return [
        (
            (float(pts[u, 0]), float(pts[u, 1]), float(z[u])),
            (float(pts[v, 0]), float(pts[v, 1]), float(z[v])),
        )
        for (u, v), n in count.items()
        if n == 1
    ]


def surface_triangles(
    artifacts, sidecar, mesh_size: float = 60.0, mesh_size_min: float = 4.0
) -> dict:
    """Triangles of every written face, by planned face id: the written XAO opened in Gmsh and meshed in 2-D (display only)."""

    triangles = {}
    with gmsh_session(artifacts.xao):
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size_min)
        gmsh.model.mesh.generate(2)
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        xyz = dict(
            zip(node_tags.tolist(), np.asarray(coords).reshape(-1, 3), strict=True)
        )
        for tag, planned_id in enumerate(
            sidecar["binding"]["ids_by_tag"]["2"], start=1
        ):
            kinds, _, nodes = gmsh.model.mesh.getElements(2, tag)
            flat = [
                n
                for kind, ns in zip(kinds, nodes, strict=True)
                if kind == 2
                for n in ns.tolist()
            ]
            used = sorted(set(flat))
            index = {n: i for i, n in enumerate(used)}
            triangles[planned_id] = (
                np.array([xyz[n] for n in used]),
                np.array([index[n] for n in flat]).reshape(-1, 3),
            )
    return triangles


def _group_kind(model, face_id: str) -> str:
    return (model.routed.face_group.get(face_id) or "other").split("::")[0]


def exploded_traces(model, tri):
    """Face triangles grouped by (volume, group) with the exploded offset and the stretched metal layer applied."""
    batches: dict[tuple[str, str], dict] = {}
    for volume_id, volume in model.topology.volumes.items():
        dz = SHIFT.get(volume.domain.name, 0.0)
        status = (
            "written"
            if volume_id in model.routed.emitted_volumes
            else "excluded metal volume"
        )
        for shell in volume.shells:
            for face_id, _ in shell.faces:
                if face_id not in tri:
                    continue
                xyz, tris = tri[face_id]
                group = _group_kind(model, face_id)
                z = np.array([stretch_z(v) for v in xyz[:, 2]]) + dz
                batch = batches.setdefault(
                    (f"{volume.domain.name} ({status})", group),
                    {
                        "x": [],
                        "y": [],
                        "z": [],
                        "i": [],
                        "j": [],
                        "k": [],
                        "outline": [],
                    },
                )
                base = len(batch["x"])
                batch["x"] += [round(float(v), 2) for v in xyz[:, 0]]
                batch["y"] += [round(float(v), 2) for v in xyz[:, 1]]
                batch["z"] += [round(float(v), 2) for v in z]
                batch["i"] += [int(v) for v in tris[:, 0] + base]
                batch["j"] += [int(v) for v in tris[:, 1] + base]
                batch["k"] += [int(v) for v in tris[:, 2] + base]
                batch["outline"].extend(outline_segments(xyz, tris, z))
    return batches


def _exploded_figures(model, artifacts, sidecar) -> dict:
    tri = surface_triangles(artifacts, sidecar)
    batches = exploded_traces(model, tri)
    fig = go.Figure()
    shown = set()
    for (volume, group), b in sorted(batches.items()):
        fig.add_trace(
            go.Mesh3d(
                x=b["x"],
                y=b["y"],
                z=b["z"],
                i=b["i"],
                j=b["j"],
                k=b["k"],
                color=GROUP_COLOURS[group],
                opacity=0.6,
                flatshading=True,
                name=GROUP_NAMES[group],
                legendgroup=group,
                showlegend=group not in shown,
                hovertext=f"{volume}: {group} faces",
                hoverinfo="text",
            )
        )
        shown.add(group)
    fig.update_layout(
        title="pad_in_ground (Route B), exploded: substrate moved down, vacuum up, the pad and ground volumes in place",
        height=640,
        legend=dict(orientation="h", y=-0.02),
        scene=dict(
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.6),
            xaxis_title="x (µm)",
            yaxis_title="y (µm)",
            zaxis=dict(
                title="z (µm; metal layer drawn 200× thicker)", range=[-200, 200]
            ),
            camera=dict(eye=dict(x=1.5, y=-1.6, z=0.9)),
        ),
    )
    xs, ys, zs = [], [], []
    for b in batches.values():
        for p, q in b["outline"]:
            xs += [round(p[0], 2), round(q[0], 2), None]
            ys += [round(p[1], 2), round(q[1], 2), None]
            zs += [round(p[2], 2), round(q[2], 2), None]
    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(color="#222", width=2),
            name="face boundaries",
            hoverinfo="skip",
        )
    )
    save_plotly(fig, "index-example-3d")

    polys, colours = [], []
    for (_, group), b in batches.items():
        pts = np.column_stack([b["x"], b["y"], b["z"]])
        for i, j, k in zip(b["i"], b["j"], b["k"], strict=True):
            polys.append(pts[[i, j, k]])
            colours.append(GROUP_COLOURS[group])
    f = plt.figure(figsize=(8.5, 6.2))
    ax = f.add_subplot(111, projection="3d")
    ax.add_collection3d(
        Poly3DCollection(polys, facecolors=colours, edgecolors="none", alpha=0.35)
    )
    ax.add_collection3d(
        Line3DCollection(
            [seg for b in batches.values() for seg in b["outline"]],
            colors="#222",
            linewidths=0.6,
        )
    )
    ax.set_xlim(-150, 150)
    ax.set_ylim(-150, 150)
    ax.set_zlim(-200, 200)
    ax.set_box_aspect((1, 1, 0.6))
    ax.view_init(elev=22, azim=-55)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_zlabel("z (display)")
    ax.legend(
        handles=[
            Line2D([0], [0], color=c, lw=6, label=GROUP_NAMES[g])
            for g, c in GROUP_COLOURS.items()
        ],
        loc="upper left",
        fontsize=7,
        frameon=False,
    )
    ax.set_title("pad_in_ground (Route B), exploded (metal layer drawn 200× thicker)")
    save_svg(f, "index-example-3d")
    drawn_faces = sorted(
        {
            fid
            for v in model.topology.volumes.values()
            for s in v.shells
            for fid, _ in s.faces
            if fid in tri
        }
    )
    return {
        "surface_triangles_drawn": sum(len(t[1]) for t in tri.values()),
        "mesh_size_um": 60.0,
        "mesh_size_min_um": 4.0,
        "faces_drawn": len(drawn_faces),
        "faces_planned": len(model.topology.faces),
        "volumes": sorted(
            f"{v.domain.name} written={vid in model.routed.emitted_volumes}"
            for vid, v in model.topology.volumes.items()
        ),
        "face_groups_drawn": sorted({_group_kind(model, f) for f in drawn_faces}),
    }


# --------------------------------------------------------------------------- the table of the supplement


def example_models() -> dict:
    """The five columns of the supplement table: the running example under Routes A, B, C, with bands, and the airbridge."""

    return {
        "pad_in_ground_B": lambda: build_example("B"),
        "pad_in_ground_A": lambda: build_example("A"),
        "pad_in_ground_C": lambda: build_example("C"),
        "pad_in_ground_B_bands": lambda: build_example("B", insets=pad_requests("B")),
        "airbridge_B": lambda: build_model_from_catalogue(airbridge(), "B"),
    }


def unwritten(model) -> dict:
    """Faces that bound no written volume, the edges only they use, and the planned-only arc-centre vertices."""
    topo = model.topology
    emitted = model.routed.emitted_volumes
    dead = [
        f
        for f in topo.faces.values()
        if not any(v in emitted for v in (f.neg_volume, f.pos_volume) if v)
    ]
    used = {
        e
        for f in topo.faces.values()
        if f not in dead
        for loop in f.loops
        for e, _ in loop
    }
    return {
        "faces_bounding_no_written_volume": len(dead),
        "their_groups": sorted(
            {model.routed.face_group.get(f.id) or "(no group)" for f in dead}
        ),
        "edges_only_on_those_faces": sum(
            1
            for e in topo.edges.values()
            if e.id not in used and e.kind != "refinement"
        ),
        "arc_centre_vertices": len(
            {e.centre_vertex for e in topo.edges.values() if e.centre_vertex}
        ),
    }


def counts() -> dict:
    """Planned counts (the model) and what the direct writer wrote (the sidecar), for the supplement table."""
    out = {}
    for case, build in example_models().items():
        model = build()
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = emit_model(model, Path(tmp))
            sidecar = read_sidecar(artifacts.sidecar)
            gap = unwritten(model)
        written = sidecar["summary"]["counts"]
        assert (
            written["vertices"]
            == model.counts["vertices"] - gap["arc_centre_vertices"]
            == model.counts["vertices"] - written["arcs"]
        )
        checks = model.report.checks
        out[case] = {
            "prisms": model.counts["prisms"],
            "levels": model.counts["levels"],
            "slabs": model.counts["slabs"],
            "planned": {
                k: model.counts[k] for k in ("vertices", "edges", "faces", "volumes")
            },
            "written": {
                "vertices": written["vertices"],
                "edges": written["edges"],
                "faces": written["faces"],
                "volumes": written["solids"],
                "arcs": written["arcs"],
                "cylinders": written["cylinders"],
            },
            "groups": model.counts["groups"],
            "validation_checks": {
                "total": len(checks),
                "passed": sum(1 for c in checks if c.passed),
                "warnings": sum(1 for c in checks if c.warning),
            },
            "not_written": gap,
        }
    return out


def example_checks() -> dict:
    """The load, OCCT and coarse-mesh results of the seven running-example models (60 µm elements), as the catalogue report measures them."""

    models = {
        "pad_a": lambda: build_example("A"),
        "pad_b": lambda: build_example("B"),
        "pad_c": lambda: build_example("C"),
        "bridge_b": lambda: build_model_from_catalogue(airbridge(), "B"),
        "pad_a_bands": lambda: build_example("A", insets=pad_requests("A")),
        "pad_b_bands": lambda: build_example("B", insets=pad_requests("B")),
        "pad_c_bands": lambda: build_example("C", insets=pad_requests("C")),
    }
    keep = (
        "faces",
        "load_check",
        "occt",
        "mesh",
        "tets",
        "minSICN_min",
        "minSICN_p1",
        "gamma_min",
        "gamma_p1",
        "max_snap_drift_grid",
        "max_arc_deviation_grid",
        "narrowest_face_um",
    )
    out = {}
    for name, build in models.items():
        with tempfile.TemporaryDirectory() as tmp:
            row = measure(build(), Path(tmp), mesh_size=60.0)
        out[name] = {k: row.get(k) for k in keep}
    return out


def cheap_recipe_mesh_checks() -> dict:
    """The banded models meshed again with the cheap recipe: HXT, ``MeshSizeMax`` = bounding-box diagonal / 20, no size field (emission.qmd#mesh-recipe)."""

    models = {
        "pad_a_bands": lambda: build_example("A", insets=pad_requests("A")),
        "pad_b_bands": lambda: build_example("B", insets=pad_requests("B")),
        "pad_c_bands": lambda: build_example("C", insets=pad_requests("C")),
        "stage_pad_b_bands": CASES[
            "pad_b_bands"
        ],  # the stage-level scene with its Route B port at the metal mid-plane
    }
    out: dict = {"max_size": "bounding-box diagonal / 20", "algorithm3d": "HXT (10)"}
    for name, build in models.items():
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = emit_model(build(), Path(tmp))
            sidecar = read_sidecar(artifacts.sidecar)
            with gmsh_session(artifacts.xao):
                apply_sidecar_embeds(sidecar)
                algorithm = mesh_with_cheap_recipe(artifacts.xao, sidecar)
                quality = mesh_quality()
        out[name] = {
            "algorithm3d_used": algorithm,
            "tets": quality["tets"],
            "minSICN_min": quality["minSICN_min"],
            "gamma_min": quality["gamma_min"],
        }
    return out


def main() -> None:
    """Write the running-example figures and the numbers of the overview page."""
    catalogue = running_example("B")
    model = build_example("B")
    with tempfile.TemporaryDirectory() as tmp:
        artifacts = emit_model(model, Path(tmp))
        sidecar = read_sidecar(artifacts.sidecar)
        _input_figure(catalogue)
        view = _exploded_figures(model, artifacts, sidecar)
    numbers = counts()
    numbers["exploded_view"] = view
    numbers["running_example_checks"] = example_checks()
    numbers["cheap_recipe_mesh_checks"] = cheap_recipe_mesh_checks()
    numbers["port"] = {
        "plane_z_um": 0.0,
        "levels_um": [float(z) for z in build_levels(catalogue.prisms).levels],
    }
    write_numbers("index", numbers)
    print(json.dumps(numbers, indent=1)[:3000])


if __name__ == "__main__":
    main()
