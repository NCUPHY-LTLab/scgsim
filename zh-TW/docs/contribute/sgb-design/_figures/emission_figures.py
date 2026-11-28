"""Figures of the emission pages: the shared face, the reopened XAO by group, the timing figures and the five-edge wall."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gmsh
import inset_kit as kit  # noqa: E402  (also puts the production core's tests on sys.path)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from emission_common import (
    FIGURES,
    GROUP_COLOURS,
    GROUP_OPACITY,
    INK,
    STAGE_COLOURS,
    face_reference,
)
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

MESH_SIZE_MAX, MESH_SIZE_MIN = (
    40.0,
    2.0,
)  # um; coarse on purpose: the view only needs the surfaces


# --------------------------------------------------------------------------- shared face figure


_SIGN = {1: "+", -1: "-"}


@dataclass(frozen=True)
class _SharedFace:
    """The shared face the figure draws, and how the shell of each solid uses it."""

    face_id: str
    reference: int
    group: str
    domain: dict[int, str]  # shell index -> name of its solid's domain
    signs: dict[
        int, int
    ]  # shell index -> +1 or -1, the sense in which the shell uses the face
    upper: int  # the shell that uses the face with "-" lies above it
    lower: int
    shared: list[int]  # 1-based numbers of every face both shells use
    both_shared: list[int]  # the references of those faces, sorted

    def use(self, shell: int) -> str:
        """The sign and reference with which the shell's record lists the face."""
        return f"{_SIGN[self.signs[shell]]}{self.reference}"


def _find_shared_face(topo, plan) -> _SharedFace:
    group_of_face = {i: g.name for g in plan.groups if g.dim == 2 for i in g.elements}
    shared = [
        abs(r)
        for r in plan.found.shells[0]
        if abs(r) in {abs(x) for x in plan.found.shells[1]}
    ]
    face_no = next(
        f for f in shared if group_of_face[f - 1] == "SA::substrate|vacuum__interface"
    )  # 1-based face number (index + 1)
    signs = {
        k: next(1 if r > 0 else -1 for r in shell if abs(r) == face_no)
        for k, shell in enumerate(plan.found.shells)
    }
    upper = next(
        k for k in range(2) if signs[k] < 0
    )  # the shell that uses the face with "-" lies above it
    return _SharedFace(
        face_id=plan.face_ids[face_no - 1],
        reference=face_reference(plan, face_no - 1),
        group=group_of_face[face_no - 1],
        domain={k: topo.volumes[plan.solid_ids[k]].domain.name for k in range(2)},
        signs=signs,
        upper=upper,
        lower=1 - upper,
        shared=shared,
        both_shared=sorted(face_reference(plan, f - 1) for f in shared),
    )


def shared_face_figure(topo, plan) -> dict:
    """Schematic cross-section (conceptual) with the real reference numbers of the running example."""
    face = _find_shared_face(topo, plan)
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    _draw_cross_section(ax, face)
    _draw_shell_records(ax, plan, face)
    fig.savefig(
        FIGURES / "emission-shared-face.svg", facecolor="white", bbox_inches="tight"
    )
    plt.close(fig)
    print("wrote emission-shared-face.svg")
    print(
        f"  shared face: planned id {face.face_id}, reference {face.reference}, group {face.group}, upper {face.domain[face.upper]} ({_SIGN[face.signs[face.upper]]}), lower {face.domain[face.lower]} ({_SIGN[face.signs[face.lower]]})"
    )
    return {
        "face_id": face.face_id,
        "reference": face.reference,
        "group": face.group,
        "upper": face.domain[face.upper],
        "lower": face.domain[face.lower],
        "upper_sign": _SIGN[face.signs[face.upper]],
        "lower_sign": _SIGN[face.signs[face.lower]],
        "shared_references": face.both_shared,
    }


def _draw_cross_section(ax, face: _SharedFace) -> None:
    """The two solids above and below the shared face, with the face normal and each solid's outward direction."""
    ax.add_patch(
        Rectangle(
            (0.2, 2.1), 4.8, 1.7, facecolor="#f4dcea", edgecolor="#999", linewidth=1
        )
    )
    ax.add_patch(
        Rectangle(
            (0.2, 0.4), 4.8, 1.7, facecolor="#d6efe5", edgecolor="#999", linewidth=1
        )
    )
    ax.plot([0.2, 5.0], [2.1, 2.1], color="#E69F00", linewidth=4, solid_capstyle="butt")
    ax.text(
        2.6,
        3.35,
        f"solid {face.domain[face.upper]}\nits shell uses the face as {face.use(face.upper)}",
        ha="center",
        va="center",
        fontsize=10,
        color=INK,
    )
    ax.text(
        2.6,
        0.95,
        f"solid {face.domain[face.lower]}\nits shell uses the face as {face.use(face.lower)}",
        ha="center",
        va="center",
        fontsize=10,
        color=INK,
    )
    ax.text(
        3.3,
        2.2,
        f"one face, written once ({face.group})",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#7a4a00",
    )
    for y1, text, label_y in (
        (2.9, "outward\nfrom upper\nsolid", 2.78),
        (1.3, "outward\nfrom lower\nsolid", 1.65),
    ):
        ax.add_patch(
            FancyArrowPatch(
                (0.8, 2.1),
                (0.8, y1),
                arrowstyle="-|>",
                mutation_scale=14,
                color="#444",
                linewidth=1.4,
            )
        )
        ax.text(0.95, label_y, text, fontsize=7.5, va="center", color="#444")
    ax.add_patch(
        FancyArrowPatch(
            (4.6, 2.1),
            (4.6, 2.9),
            arrowstyle="-|>",
            mutation_scale=14,
            color="#b00020",
            linewidth=1.8,
        )
    )
    ax.text(
        4.45,
        2.6,
        "face normal\n(plane +z)",
        fontsize=7.5,
        ha="right",
        va="center",
        color="#b00020",
    )
    ax.text(
        0.2,
        4.0,
        "Conceptual cross-section at the substrate / vacuum interface",
        fontsize=9,
        color="#555",
    )


def _shell_record(plan, shell_index: int, shared: list[int]) -> str:
    """The shell record with the shared references in brackets; the rest is elided after the first eight."""
    refs = plan.found.shells[shell_index]
    items = [f"{'+' if r > 0 else '-'}{face_reference(plan, abs(r) - 1)}" for r in refs]
    keep = [i for i, r in enumerate(refs) if i < 8 or abs(r) in shared]
    out, last = [], -1
    for i in keep:
        if i != last + 1:
            out.append("...")
        out.append(f"[{items[i]}]" if abs(refs[i]) in shared else items[i])
        last = i
    if last != len(refs) - 1:
        out.append("...")
    return "Sh  " + " ".join(out) + f"   ({len(refs)} faces)"


def _draw_shell_records(ax, plan, face: _SharedFace) -> None:
    """The real shell records of the written BREP, with the shared references in brackets."""
    ax.text(
        5.4,
        3.85,
        "Shell records in semantic_geometry_route_b.brep (real)",
        fontsize=9,
        color="#555",
        va="top",
    )
    ax.text(
        5.4,
        3.35,
        f"shell of the {face.domain[face.upper]} solid:",
        fontsize=9,
        va="top",
        color=INK,
    )
    ax.text(
        5.4,
        3.05,
        _shell_record(plan, face.upper, face.shared),
        fontsize=7.5,
        va="top",
        family="monospace",
        color=INK,
    )
    ax.text(
        5.4,
        2.5,
        f"shell of the {face.domain[face.lower]} solid:",
        fontsize=9,
        va="top",
        color=INK,
    )
    ax.text(
        5.4,
        2.2,
        _shell_record(plan, face.lower, face.shared),
        fontsize=7.5,
        va="top",
        family="monospace",
        color=INK,
    )
    ax.text(
        5.4,
        1.5,
        f"Brackets mark the {len(face.both_shared)} faces both shells use (references {', '.join(map(str, face.both_shared))});\n"
        f"each appears once with + and once with -. The face drawn is {face.face_id}.",
        fontsize=8,
        va="top",
        color="#444",
    )


# --------------------------------------------------------------------------- reopened XAO, coloured by group


def _group_triangles(plan, xao: Path):
    """Mesh the surfaces of the XAO and return per face entity: (group name, tag, points, triangles)."""

    out = []
    gmsh.initialize(["-noenv"])
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Geometry.OCCAutoFix", 0)
        gmsh.open(str(xao))
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", MESH_SIZE_MAX)
        gmsh.option.setNumber("Mesh.MeshSizeMin", MESH_SIZE_MIN)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 24)
        gmsh.model.mesh.generate(2)
        group_of = {}
        for dim, ptag in gmsh.model.getPhysicalGroups(2):
            name = gmsh.model.getPhysicalName(dim, ptag)
            for entity in gmsh.model.getEntitiesForPhysicalGroup(dim, ptag):
                group_of[int(entity)] = name
        for _, tag in gmsh.model.getEntities(2):
            node_tags, coords, _ = gmsh.model.mesh.getNodes(
                2, tag, includeBoundary=True, returnParametricCoord=False
            )
            types, _, element_nodes = gmsh.model.mesh.getElements(2, tag)
            triangles = (
                np.concatenate(
                    [
                        np.asarray(e).reshape(-1, 3)
                        for t, e in zip(types, element_nodes, strict=False)
                        if t == 2
                    ]
                )
                if len(types)
                else np.zeros((0, 3), int)
            )
            lookup = {int(n): i for i, n in enumerate(node_tags)}
            points = np.asarray(coords).reshape(-1, 3)
            index = np.vectorize(lookup.get)(triangles) if len(triangles) else triangles
            out.append((group_of.get(tag, "(none)"), int(tag), points, index))
    finally:
        gmsh.finalize()
    return out


def groups_3d_figures(plan, xao: Path) -> dict:
    """The reopened XAO meshed on its surfaces and coloured by physical group, as Plotly HTML and static SVG."""
    entities = _group_triangles(plan, xao)
    total = sum(len(t) for _, _, _, t in entities)
    print(
        f"  surface mesh of the reopened XAO: {len(entities)} faces, {total} triangles"
    )
    fig = go.Figure()
    shown: set[str] = set()
    for name, tag, points, tri in entities:
        if not len(tri):
            continue
        fig.add_trace(
            go.Mesh3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                i=tri[:, 0],
                j=tri[:, 1],
                k=tri[:, 2],
                color=GROUP_COLOURS[name],
                opacity=GROUP_OPACITY.get(name, 1.0),
                name=name,
                legendgroup=name,
                showlegend=name not in shown,
                flatshading=True,
                hovertemplate=f"{name}<br>Gmsh face tag {tag}<br>planned id {plan.face_ids[tag - 1]}<extra></extra>",
            )
        )
        shown.add(name)
    fig.update_layout(
        width=980,
        height=620,
        paper_bgcolor="white",
        font=dict(color=INK, size=12),
        scene=dict(
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.45),
            xaxis_title="x (µm)",
            yaxis_title="y (µm)",
            zaxis_title="z (µm)",
            camera=dict(eye=dict(x=1.5, y=-1.6, z=1.0)),
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(itemsizing="constant"),
    )
    fig.write_html(
        FIGURES / "emission-groups-3d.html",
        include_plotlyjs="directory",
        config={"displaylogo": False},
    )
    print("wrote emission-groups-3d.html")

    sfig = plt.figure(figsize=(10.5, 4.6))
    ax = sfig.add_subplot(1, 2, 1, projection="3d")
    for name, _, points, tri in entities:
        if not len(tri):
            continue
        ax.add_collection3d(
            Poly3DCollection(
                points[tri],
                facecolor=GROUP_COLOURS[name],
                alpha=GROUP_OPACITY.get(name, 1.0),
                edgecolor="none",
                rasterized=True,
            )
        )
    ax.set_xlim(-150, 150)
    ax.set_ylim(-150, 150)
    ax.set_zlim(-100, 100)
    ax.set_box_aspect((1, 1, 0.45))
    ax.view_init(elev=24, azim=-55)
    ax.set_xlabel("x (µm)", fontsize=8)
    ax.set_ylabel("y (µm)", fontsize=8)
    ax.set_zlabel("z (µm)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title("whole model (z stretched)", fontsize=9)

    plan_ax = sfig.add_subplot(
        1, 2, 2
    )  # top view of the horizontal faces, lowest first so the metal top paints last
    horizontal = [
        (float(points[tri][:, :, 2].mean()), name, points[tri][:, :, :2])
        for name, _, points, tri in entities
        if len(tri)
        and not name.startswith("EXT")
        and np.ptp(points[tri][:, :, 2]) < 1e-6
    ]
    for _, name, xy in sorted(horizontal, key=lambda item: item[0]):
        plan_ax.add_collection(
            PolyCollection(
                xy,
                facecolor=GROUP_COLOURS[name],
                edgecolor="white",
                linewidth=0.2,
                rasterized=True,
            )
        )
    plan_ax.set_xlim(-95, 95)
    plan_ax.set_ylim(-72, 72)
    plan_ax.set_aspect("equal")
    plan_ax.set_xlabel("x (µm)", fontsize=8)
    plan_ax.set_ylabel("y (µm)", fontsize=8)
    plan_ax.tick_params(labelsize=7)
    plan_ax.set_title(
        "top view of the cutout (horizontal faces; walls and outer box not drawn)",
        fontsize=9,
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c, alpha=max(GROUP_OPACITY.get(n, 1.0), 0.35))
        for n, c in GROUP_COLOURS.items()
    ]
    sfig.legend(
        handles,
        GROUP_COLOURS.keys(),
        fontsize=7,
        loc="lower center",
        ncol=4,
        frameon=False,
    )
    sfig.tight_layout(rect=(0, 0.1, 1, 1))
    sfig.savefig(FIGURES / "emission-groups-3d.svg", facecolor="white")
    plt.close(sfig)
    print("wrote emission-groups-3d.svg")
    return {"faces": len(entities), "triangles": total}


def timing_figures(head: list[dict]) -> None:
    """Write and load cost of the Gmsh-API oracle against the direct writer, as Plotly HTML and static SVG."""
    sides = []
    for r in head:
        api, d, ld = r["api"], r["direct"], r["load"]
        sides.append(
            {
                "n": r["n"],
                "api": {
                    "OCC build (addPoint ... addVolume)": api["occ_build_s"],
                    "synchronize": api["synchronize_s"],
                    "physical groups": api["physical_groups_s"],
                    "write XAO": api["write_xao_s"],
                    "load into Gmsh": ld["api_xao_s"],
                },
                "direct": {
                    "plan (serial)": d["plan_s"],
                    "render": d["render_brep_s"] + d["render_xao_s"],
                    "load into Gmsh": ld["direct_xao_s"],
                },
            }
        )
    labels = ["Gmsh-API emitter", "direct writer"]
    fig = make_subplots(
        rows=1,
        cols=len(sides),
        subplot_titles=[f"N = {r['n']:,} features" for r in sides],
        horizontal_spacing=0.07,
    )
    seen: set[str] = set()
    for col, r in enumerate(sides, start=1):
        for side, label in zip(("api", "direct"), labels, strict=False):
            for stage, value in r[side].items():
                fig.add_trace(
                    go.Bar(
                        x=[label],
                        y=[value],
                        name=stage,
                        marker_color=STAGE_COLOURS[stage],
                        legendgroup=stage,
                        showlegend=stage not in seen,
                        hovertemplate=f"{stage}: %{{y:.3f}} s<extra>{label}, N = {r['n']:,}</extra>",
                    ),
                    row=1,
                    col=col,
                )
                seen.add(stage)
        for label, side in zip(labels, ("api", "direct"), strict=False):
            fig.add_annotation(
                x=label,
                y=sum(r[side].values()),
                text=f"{sum(r[side].values()):.1f} s",
                showarrow=False,
                yshift=10,
                row=1,
                col=col,
                font=dict(size=11, color=INK),
            )
    fig.update_layout(
        barmode="stack",
        width=1000,
        height=470,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color=INK, size=12),
        legend=dict(orientation="h", y=-0.12),
        margin=dict(l=55, r=15, t=45, b=60),
    )
    fig.update_yaxes(
        title_text="seconds (write + load)", row=1, col=1, gridcolor="#e5e5e5"
    )
    for col in range(2, len(sides) + 1):
        fig.update_yaxes(gridcolor="#e5e5e5", row=1, col=col)
    kit.ensure_plotly_js()
    fig.write_html(
        FIGURES / "emission-timing.html",
        include_plotlyjs="directory",
        config={"displaylogo": False},
    )
    print("wrote emission-timing.html")

    sfig, axes = plt.subplots(1, len(sides), figsize=(10.5, 4.4))
    for ax, r in zip(axes, sides, strict=False):
        for i, side in enumerate(("api", "direct")):
            bottom = 0.0
            for stage, value in r[side].items():
                ax.bar(
                    i,
                    value,
                    bottom=bottom,
                    color=STAGE_COLOURS[stage],
                    width=0.7,
                    edgecolor="white",
                    linewidth=0.5,
                    label=stage,
                )
                bottom += value
            ax.text(i, bottom, f"{bottom:.1f} s", ha="center", va="bottom", fontsize=8)
        ax.set_xticks([0, 1], labels, fontsize=8)
        ax.set_title(f"N = {r['n']:,} features", fontsize=9)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("seconds (write + load)", fontsize=8)
    handles, names = axes[-1].get_legend_handles_labels()
    unique = dict(zip(names, handles, strict=False))
    sfig.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        ncol=4,
        fontsize=7.5,
        frameon=False,
    )
    sfig.tight_layout(rect=(0, 0.1, 1, 1))
    sfig.savefig(FIGURES / "emission-timing.svg", facecolor="white")
    plt.close(sfig)
    print("wrote emission-timing.svg")


# --------------------------------------------------------------------------- conceptual figure


def five_edge_wall_figure() -> None:
    """Conceptual: the cylinder wall of a round pad unrolled (angle across, z up), before and after splitting at a foreign vertex."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, split in zip(axes, (False, True), strict=False):
        ax.set_xlim(-0.6, 5.6)
        ax.set_ylim(-0.7, 2.7)
        ax.axis("off")
        face = "#fde9d6"
        if not split:
            ax.add_patch(Rectangle((0, 0), 5, 2, facecolor=face, edgecolor="none"))
            ax.plot([0, 5, 5, 0, 0], [0, 0, 2, 2, 0], color="#1b1b1b", linewidth=1.6)
            ax.plot([3], [0], "o", color="#D55E00", markersize=9)
            ax.text(
                3, -0.35, "foreign vertex", ha="center", fontsize=8, color="#D55E00"
            )
            ax.text(1.5, -0.35, "bottom arc 1", ha="center", fontsize=8)
            ax.text(4.1, -0.35, "arc 2", ha="center", fontsize=8)
            ax.text(
                2.5,
                1.0,
                "one wall, 5 boundary edges\n(2 bottom arcs, 1 top arc, 2 vertical)",
                ha="center",
                fontsize=9,
            )
            ax.set_title("before the split: one wall, five boundary edges", fontsize=9)
        else:
            for x0, x1 in ((0, 3), (3, 5)):
                ax.add_patch(
                    Rectangle((x0, 0), x1 - x0, 2, facecolor=face, edgecolor="none")
                )
            ax.plot([0, 5, 5, 0, 0], [0, 0, 2, 2, 0], color="#1b1b1b", linewidth=1.6)
            ax.plot([3, 3], [0, 2], color="#0072B2", linewidth=2.4)
            ax.plot([3], [0], "o", color="#D55E00", markersize=9)
            ax.plot([3], [2], "o", color="#0072B2", markersize=7)
            ax.text(
                3.1,
                2.2,
                "new vertical edge, top arc split at the same place",
                ha="center",
                fontsize=8,
                color="#0072B2",
            )
            ax.text(1.5, 1.0, "piece 1\n4 edges", ha="center", fontsize=9)
            ax.text(4.0, 1.0, "piece 2\n4 edges", ha="center", fontsize=9)
            ax.set_title("after: two exact cylinders on their own circles", fontsize=9)
    fig.suptitle(
        "Conceptual: a curved wall with a foreign vertex on its arc, unrolled and split there",
        fontsize=10,
    )
    fig.savefig(
        FIGURES / "emission-five-edge-wall.svg", facecolor="white", bbox_inches="tight"
    )
    plt.close(fig)
    print("wrote emission-five-edge-wall.svg")
