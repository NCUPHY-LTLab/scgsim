"""Figures and tables for ``docs/design/sgb/emission.qmd``.

Run, with the Python of the checkout that contains the production SGB core (``scgsim.geometry.sgb``)::

    SGB_CORE=<sgb-core> <sgb-core>/.artifacts/env/bin/python docs/design/sgb/_figures/emission.py            # figures and tables
    SGB_CORE=<sgb-core> <sgb-core>/.artifacts/env/bin/python docs/design/sgb/_figures/emission.py --measure  # also rerun the timings
    ... emission.py --measure-only                                                                           # only the timings
    ... emission.py --occt-only                                                                              # only the OCCT and catalogue tables

Everything is computed by the production package.  The running example is the pad-in-ground scene of
``tests/sgb_scenes.py`` under Route B, its lumped port on the support plane z = 0 (see ``inset_kit.pad_catalogue``): it is built,
written with ``emit_model`` into a scratch folder, numbered by ``plan_brep`` (whose numbering is checked against the written
sidecar's ``ids_by_tag`` and the written BREP's shell records) and reopened in Gmsh.  The timings are measured by this script
(``--measure``, one subprocess per N so the peak memory is honest) and stored in ``figures/emission-timing-data.json``; the other
runs read that file.  Only what ``figures/why-scaling-measured.json`` (``why_scaling.py``) does not hold is measured here: the Gmsh-API
oracle stage by stage next to the direct writer at N = 500, 1000 and 2000, and the time of every check at N = 1000, 5000 and 20000.
The direct writer's tail (plan, render with 1 and 16 workers, reports, sidecar, Gmsh load, ``verify_loaded``, file sizes) is read from
that shared file.  The OCCT checks use OCP (the ``occt`` group of the SGB core).  Outputs go to ``docs/design/sgb/figures/``
with the prefix ``emission-``:

* ``emission-timing``         write and load cost of the Gmsh-API oracle emitter against the direct writer (html + svg)
* ``emission-shared-face``    one shared face referenced + by a shell and - by the other (svg; schematic plus real references)
* ``emission-groups-3d``      the reopened XAO meshed on its surfaces, coloured by physical group (html + svg)
* ``emission-*-table.md``     markdown tables included by the page (counts, timing, sidecar, tags, groups, validation, OCCT, bands, ...)
* ``emission-example-numbers.json``  the counts of the running example that the page quotes
* ``emission-step-table.md``  is NOT written here: it is the frozen STEP round-trip table of the AEDT experiment, included by ``aedt-handoff.qmd``
"""

from __future__ import annotations

import json
import math
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import inset_kit as kit  # noqa: E402  (also puts the production core's tests on sys.path)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import sgb_occt_mutations as mutations  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402
from scgsim.geometry.sgb.build import emit_model  # noqa: E402
from scgsim.geometry.sgb.emission.plan import plan_brep  # noqa: E402
from scgsim.geometry.sgb.emission.writer import write_brep_file, write_xao_file  # noqa: E402
from scgsim.geometry.sgb.model import build_model_from_catalogue  # noqa: E402
from scgsim.geometry.sgb.sidecar.files import read_sidecar  # noqa: E402
from scgsim.geometry.sgb.verification.loaded import apply_sidecar_embeds, verify_loaded  # noqa: E402
from scgsim.geometry.sgb.verification.occt import check_file, check_shape, read_brep  # noqa: E402
from sgb_extreme_cases import build_model
from sgb_extreme_cases import cases as extreme_cases  # noqa: E402
from sgb_gmsh_oracle import (  # noqa: E402
    OracleResult,
    _build_entities,
    band_mesh_report,
    check_mesh_conformity,
    emit_with_gmsh_api,
    gmsh_session,
    mesh_coarsely,
    refine_near,
)
from sgb_scenes import airbridge, random_layout  # noqa: E402

FIGURES = kit.FIGURES
DATA = FIGURES / "emission-timing-data.json"
INK = "#1b1b1b"
SHARED = (
    FIGURES / "why-scaling-measured.json"
)  # production stage times, written by why_scaling.py; reused here, not re-measured
HEAD_SIZES = (
    500,
    1000,
    2000,
)  # the Gmsh-API oracle is superlinear in the number of physical groups; it did not finish N = 5000 in 10 minutes
VALIDATOR_SIZES = (1000, 5000, 20000)
RANDOM_SEED = 2  # random_layout(n, seed=2, route="B"): seed 1 at N = 20000 legitimately fails (two shapes of one Net pinch, NonManifoldError)
STAGE_COLOURS = {  # one colour per named stage, shared by the plotly and matplotlib versions
    "OCC build (addPoint ... addVolume)": "#D55E00",
    "synchronize": "#E69F00",
    "physical groups": "#F0E442",
    "write XAO": "#CC79A7",
    "plan (serial)": "#0072B2",
    "render (16 workers)": "#56B4E9",
    "load into Gmsh": "#444444",
}
GROUP_COLOURS = {
    "MA::pad::top": "#0072B2",
    "MA::pad::sidewall": "#7fb8e0",
    "MS::pad::bottom": "#56B4E9",
    "MA::ground::top": "#8C8C8C",
    "MA::ground::sidewall": "#b0b0b0",
    "MS::ground::bottom": "#CFCFCF",
    "SA::substrate|vacuum::interface": "#E69F00",
    "PORT::p1": "#D55E00",
    "EXT::substrate::bottom": "#009E73",
    "EXT::substrate::sidewall": "#4fc3a1",
    "EXT::vacuum::top": "#CC79A7",
    "EXT::vacuum::sidewall": "#e3a8c9",
}
GROUP_OPACITY = {name: 0.12 for name in GROUP_COLOURS if name.startswith("EXT")}
MESH_SIZE_MAX, MESH_SIZE_MIN = (
    40.0,
    2.0,
)  # um; coarse on purpose: the view only needs the surfaces
LOAD_SCRIPT = (
    "import sys, time, gmsh\n"
    "gmsh.initialize(['-noenv']); gmsh.option.setNumber('General.Terminal', 0); gmsh.option.setNumber('Geometry.OCCAutoFix', 0)\n"
    "t = time.perf_counter()\n"
    "(gmsh.open if sys.argv[1].endswith('.xao') else gmsh.merge)(sys.argv[1])\n"
    "gmsh.model.occ.synchronize()\n"
    "print(time.perf_counter() - t)\n"
)


# --------------------------------------------------------------------------- helpers


def _sec(x: float) -> str:
    return f"{x:.1f}" if x >= 10 else f"{x:.2f}" if x >= 1 else f"{x:.3f}"


def _table(header: list[str], rows: list[list[str]], align: str | None = None) -> str:
    align = align or "l" + "r" * (len(header) - 1)
    rule = ["---:" if a == "r" else ":---" for a in align]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(rule) + "|"]
    lines += [
        "| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |"
        for row in rows
    ]
    return "\n".join(lines) + "\n"


def _write(name: str, text: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES / name).write_text(text)
    print("wrote", name)


def _case_models() -> dict:
    """The running examples of the production tests (``CASES``), with the pad's lumped port on the support plane."""
    return {
        "pad_in_ground_A": lambda: kit.pad_model("A", None),
        "pad_in_ground_B": lambda: kit.pad_model("B", None),
        "pad_in_ground_C": lambda: build_model_from_catalogue(
            kit.pad_catalogue("B"), "C"
        ),
        "airbridge_B": lambda: build_model_from_catalogue(airbridge(), "B"),
        "pad_in_ground_A_bands": lambda: kit.pad_model("A", kit.MARGINS),
        "pad_in_ground_B_bands": lambda: kit.pad_model("B", kit.MARGINS),
        "pad_in_ground_C_bands": lambda: build_model_from_catalogue(
            kit.pad_catalogue("B"), "C", insets=kit.pad_requests("B")
        ),
    }


def _bundle(model, folder: Path):
    """Write a model and return (artifacts, plan, sidecar)."""
    artifacts = emit_model(model, folder)
    return (
        artifacts,
        plan_brep(model.topology, model.routed),
        read_sidecar(artifacts.sidecar),
    )


# --------------------------------------------------------------------------- the running example


def _running_example(folder: Path):
    """The pad-in-ground example (Route B), its files and plan, and the check that the plan's numbering equals the files'."""
    model = kit.pad_model("B", None)
    artifacts, plan, sidecar = _bundle(model, folder)
    recorded = {
        pid: (int(dim), i + 1)
        for dim, ids in sidecar["binding"]["ids_by_tag"].items()
        for i, pid in enumerate(ids)
    }
    if recorded != plan.tags():
        raise SystemExit("the plan's numbering is not the written sidecar's ids_by_tag")
    return model, plan, sidecar, artifacts


def _face_reference(plan, face_index: int) -> int:
    """The number a shell uses for this face: its position counted from the end of the TShapes section."""
    return plan.section_bases()["F"] - face_index


def _check_shell_references(plan, brep: Path) -> None:
    """The shell records of the written BREP are the plan's shells with the references this script derives."""
    lines = brep.read_text().split("\n")
    written = []
    for i, line in enumerate(lines):
        if line == "Sh" and lines[i + 1] == "":
            tokens = lines[i + 3].split()
            written.append([int(tokens[k]) for k in range(0, len(tokens) - 1, 2)])
    expected = [
        [(1 if r > 0 else -1) * _face_reference(plan, abs(r) - 1) for r in shell]
        for shell in plan.found.shells
    ]
    if written != expected:
        raise SystemExit(
            "the shell records of the BREP file are not what the plan's references say"
        )


def _example_numbers(model, plan, sidecar) -> dict:
    uses: dict[int, int] = {}
    for shell in plan.found.shells:
        for ref in shell:
            uses[abs(ref)] = uses.get(abs(ref), 0) + 1
    numbers = {
        "planned": model.topology.counts(),
        "emitted": plan.counts,
        "section_bases": plan.section_bases(),
        "faces_used_by_two_shells": sum(1 for n in uses.values() if n == 2),
        "faces_used_by_one_shell": sum(1 for n in uses.values() if n == 1),
        "groups": {
            name: {"dim": g["dim"], "tag": g["tag"], "members": len(g["ids"])}
            for name, g in sidecar["physical_groups"].items()
        },
        "ids_by_tag": {
            dim: len(ids) for dim, ids in sidecar["binding"]["ids_by_tag"].items()
        },
        "sidecar_bytes": None,
        "embedded_faces": len(sidecar["embedded_faces"]),
        "free_faces": plan.counts["free_faces"],
    }
    return numbers


def _counts_table(model, plan) -> None:
    c = plan.counts
    rows = [
        ["vertex", str(c["vertices"]), "`Ve`", "edge"],
        ["edge", str(c["edges"]), "`Ed`", "wire"],
        ["wire（迴圈）", str(c["wires"]), "`Wi`", "face"],
        [
            "face",
            f"{c['faces']}（其中 {c['free_faces']} 個為 free face）",
            "`Fa`",
            "shell",
        ],
        ["shell", str(c["shells"]), "`Sh`", "solid"],
        ["solid", str(c["solids"]), "`So`", "compound"],
        ["compound", "1", "`Co`", "無（根節點）"],
    ]
    _write(
        "emission-counts-table.md",
        _table(
            ["元素", "pad-in-ground（Route B）中的數量", "record", "被誰引用"],
            rows,
            "lrll",
        ),
    )


def _tag_rows(topo, plan, sidecar) -> list[list[str]]:
    """A hand-picked but fully computed selection of planned ids with their native tags and XAO references."""
    tags = plan.tags()
    found = plan.found
    group_of = {}
    for g in plan.groups:
        for i in g.elements:
            group_of[(g.dim, i)] = (g.name, g.tag)
    refs = {
        3: found.solid_ref,
        2: found.face_ref,
        1: found.edge_ref,
        0: found.vertex_ref,
    }
    picks = []  # (what, planned id, dim, index)
    for i, vid in enumerate(plan.solid_ids):
        picks.append((f"solid，{topo.volumes[vid].domain.name}", vid, 3, i))
    first_face_of = {}
    for i in range(len(plan.face_ids)):
        first_face_of.setdefault(group_of[(2, i)][0], i)
    for name in (
        "SA::substrate|vacuum::interface",
        "MS::pad::bottom",
        "MA::pad::top",
        "PORT::p1",
        "MA::ground::top",
    ):
        picks.append(
            (
                f"face，{name} 的第一個",
                plan.face_ids[first_face_of[name]],
                2,
                first_face_of[name],
            )
        )
    curved = next(
        i
        for i, fid in enumerate(plan.face_ids)
        if topo.faces[fid].curved and found.face_flipped[i]
    )
    picks.append(
        ("face，curved wall (cylinder)，wire 已反向", plan.face_ids[curved], 2, curved)
    )
    picks.append(("edge，第一個寫出", plan.edge_ids[0], 1, 0))
    picks.append(("vertex，第一個寫出", plan.vertex_ids[0], 0, 0))
    rows = []
    for what, pid, dim, index in picks:
        assert tags[pid] == (dim, index + 1)
        group = group_of.get((dim, index)) if dim >= 2 else None
        rows.append(
            [
                f"`{pid}`",
                what,
                f"{group[0]} (tag {group[1]})" if group else "無",
                f"({dim}, {index + 1})",
                str(index),
                str(refs[dim][index]),
            ]
        )
    return rows


def _tags_and_groups_tables(topo, plan, sidecar) -> None:
    _write(
        "emission-tags-table.md",
        _table(
            [
                "planned id",
                "元素",
                "primary physical group",
                "Gmsh (dim, tag)",
                "XAO index",
                "XAO reference",
            ],
            _tag_rows(topo, plan, sidecar),
            "llllrr",
        ),
    )
    rows, covered = [], 0
    for g in plan.groups:
        kind = "face" if g.dim == 2 else "solid"
        rows.append([g.name, kind, str(g.tag), str(len(g.elements))])
        covered += len(g.elements) if kind == "face" else 0
    assert covered == len(plan.face_ids), (
        "physical groups do not partition the emitted faces"
    )
    assert len({i for g in plan.groups if g.dim == 2 for i in g.elements}) == covered
    _write(
        "emission-groups-table.md",
        _table(["primary group", "維度", "physical tag", "成員數"], rows, "llrr"),
    )
    print(
        f"  face groups cover {covered} of {len(plan.face_ids)} emitted faces, each exactly once"
    )


# --------------------------------------------------------------------------- shared face figure


def _shared_face_figure(topo, plan) -> dict:
    """Schematic cross-section (conceptual) with the real reference numbers of the running example."""
    group_of_face = {i: g.name for g in plan.groups if g.dim == 2 for i in g.elements}
    shared = [
        abs(r)
        for r in plan.found.shells[0]
        if abs(r) in {abs(x) for x in plan.found.shells[1]}
    ]
    face_no = next(
        f for f in shared if group_of_face[f - 1] == "SA::substrate|vacuum::interface"
    )  # 1-based face number (index + 1)
    ref = _face_reference(plan, face_no - 1)
    face_id = plan.face_ids[face_no - 1]
    group = group_of_face[face_no - 1]
    signs = {
        k: next(1 if r > 0 else -1 for r in shell if abs(r) == face_no)
        for k, shell in enumerate(plan.found.shells)
    }
    domain = {k: topo.volumes[plan.solid_ids[k]].domain.name for k in range(2)}
    upper = next(
        k for k in range(2) if signs[k] < 0
    )  # the shell that uses the face with "-" lies above it
    lower = 1 - upper
    sym = {1: "+", -1: "-"}

    both_shared = sorted(_face_reference(plan, f - 1) for f in shared)
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
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
        f"solid {domain[upper]}\nits shell uses the face as {sym[signs[upper]]}{ref}",
        ha="center",
        va="center",
        fontsize=10,
        color=INK,
    )
    ax.text(
        2.6,
        0.95,
        f"solid {domain[lower]}\nits shell uses the face as {sym[signs[lower]]}{ref}",
        ha="center",
        va="center",
        fontsize=10,
        color=INK,
    )
    ax.text(
        3.3,
        2.2,
        f"one face, written once ({group})",
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

    def record(shell_index: int) -> str:
        """The shell record with the shared references in brackets; the rest is elided after the first eight."""
        refs = plan.found.shells[shell_index]
        items = [
            f"{'+' if r > 0 else '-'}{_face_reference(plan, abs(r) - 1)}" for r in refs
        ]
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
        f"shell of the {domain[upper]} solid:",
        fontsize=9,
        va="top",
        color=INK,
    )
    ax.text(
        5.4, 3.05, record(upper), fontsize=7.5, va="top", family="monospace", color=INK
    )
    ax.text(
        5.4,
        2.5,
        f"shell of the {domain[lower]} solid:",
        fontsize=9,
        va="top",
        color=INK,
    )
    ax.text(
        5.4, 2.2, record(lower), fontsize=7.5, va="top", family="monospace", color=INK
    )
    ax.text(
        5.4,
        1.5,
        f"Brackets mark the {len(both_shared)} faces both shells use (references {', '.join(map(str, both_shared))});\n"
        f"each appears once with + and once with -. The face drawn is {face_id}.",
        fontsize=8,
        va="top",
        color="#444",
    )

    fig.savefig(
        FIGURES / "emission-shared-face.svg", facecolor="white", bbox_inches="tight"
    )
    plt.close(fig)
    print("wrote emission-shared-face.svg")
    print(
        f"  shared face: planned id {face_id}, reference {ref}, group {group}, upper {domain[upper]} ({sym[signs[upper]]}), lower {domain[lower]} ({sym[signs[lower]]})"
    )
    return {
        "face_id": face_id,
        "reference": ref,
        "group": group,
        "upper": domain[upper],
        "lower": domain[lower],
        "upper_sign": sym[signs[upper]],
        "lower_sign": sym[signs[lower]],
        "shared_references": both_shared,
    }


# --------------------------------------------------------------------------- reopened XAO, coloured by group


def _group_triangles(plan, xao: Path):
    """Mesh the surfaces of the XAO and return per face entity: (group name, tag, points, triangles)."""
    import gmsh

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
                        for t, e in zip(types, element_nodes)
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


def _groups_3d_figures(plan, xao: Path) -> dict:
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


# --------------------------------------------------------------------------- timings


def _load_seconds(path: Path) -> float:
    """Seconds Gmsh needs to open (XAO) or merge (BREP) a file and synchronize, in a fresh process."""
    out = subprocess.run(
        [sys.executable, "-c", LOAD_SCRIPT, str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(out.stdout.strip().splitlines()[-1])


def head_one(n: int) -> dict:
    """One random Route B layout through the Gmsh-API oracle, stage by stage, and through the direct writer (16 workers), then both loads."""
    import gmsh

    model = build_model_from_catalogue(
        random_layout(n, seed=RANDOM_SEED, route="B"), "B"
    )
    row: dict = {"n": n, "counts": model.counts, "groups": len(model.routed.groups)}
    with tempfile.TemporaryDirectory(prefix=f"emission-head-{n}-") as scratch:
        folder = Path(scratch)
        started = time.perf_counter()
        plan = plan_brep(model.topology, model.routed)
        plan_s = time.perf_counter() - started
        brep, xao = folder / "direct.brep", folder / "direct.xao"
        render_brep = write_brep_file(plan, brep, workers=16)
        render_xao = write_xao_file(plan, xao, name="direct", workers=16, brep=brep)
        row["direct"] = {
            "plan_s": plan_s,
            "render_brep_s": render_brep,
            "render_xao_s": render_xao,
            "plan_counts": plan.counts,
            "xao_bytes": xao.stat().st_size,
            "brep_bytes": brep.stat().st_size,
        }
        with gmsh_session():
            result = OracleResult()
            t0 = time.perf_counter()
            _build_entities(model, result)
            t1 = time.perf_counter()
            gmsh.model.occ.synchronize()
            t2 = time.perf_counter()
            for name, group in model.routed.groups.items():
                tags = [result.tags[i][1] for i in group.ids if i in result.tags]
                if tags:
                    gmsh.model.addPhysicalGroup(group.dim, tags, name=name)
            t3 = time.perf_counter()
            api_xao = folder / "api.xao"
            gmsh.write(str(api_xao))
            t4 = time.perf_counter()
        row["api"] = {
            "occ_build_s": t1 - t0,
            "synchronize_s": t2 - t1,
            "physical_groups_s": t3 - t2,
            "write_xao_s": t4 - t3,
            "total_s": t4 - t0,
            "xao_bytes": api_xao.stat().st_size,
        }
        row["load"] = {
            "api_xao_s": _load_seconds(api_xao),
            "direct_xao_s": _load_seconds(xao),
            "direct_brep_s": _load_seconds(brep),
        }
    return row


def validator_one(n: int) -> dict:
    """Build one layout, write all its files, open the XAO and run ``verify_loaded``: the time of every check, structural and load-side."""
    import gmsh

    started = time.perf_counter()
    model = build_model_from_catalogue(
        random_layout(n, seed=RANDOM_SEED, route="B"), "B"
    )
    build_s = time.perf_counter() - started
    with tempfile.TemporaryDirectory(prefix=f"emission-validator-{n}-") as scratch:
        artifacts = emit_model(model, Path(scratch), workers=16)
        sidecar = read_sidecar(artifacts.sidecar)
        with gmsh_session():
            started = time.perf_counter()
            gmsh.open(str(artifacts.xao))
            gmsh.model.occ.synchronize()
            load_s = time.perf_counter() - started
            started = time.perf_counter()
            report = verify_loaded(sidecar)
            verify_s = time.perf_counter() - started
        assert report.ok, report.problems[:3]
    return {
        "n": n,
        "counts": model.counts,
        "written": {
            k: artifacts.counts[k] for k in ("vertices", "edges", "faces", "solids")
        },
        "build_s": build_s,
        "stage_s": model.timings,
        "validator_s": model.report.timings,
        "load_s": load_s,
        "verify_s": verify_s,
        "verify_phases_s": report.timings_s,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }


def measure() -> None:
    """Run the two measurements of this page, one subprocess per N (a clean peak memory), and store them."""
    out: dict = {
        "layout": f"random_layout(n, seed={RANDOM_SEED}, route='B')",
        "cpu_count": os.cpu_count(),
        "load_average_at_start": os.getloadavg(),
        "head": [],
        "validator": [],
    }
    for key, flag, sizes in (
        ("head", "--head-one", HEAD_SIZES),
        ("validator", "--validator-one", VALIDATOR_SIZES),
    ):
        for n in sizes:
            started = time.perf_counter()
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), flag, str(n)],
                capture_output=True,
                text=True,
                check=True,
            )
            out[key].append(json.loads(proc.stdout.strip().splitlines()[-1]))
            print(f"{flag} N = {n}: {time.perf_counter() - started:.0f} s", flush=True)
            DATA.write_text(json.dumps(out, indent=1))
    out["load_average_at_end"] = os.getloadavg()
    DATA.write_text(json.dumps(out, indent=1))


def _shared_rows() -> dict[int, dict]:
    """The one-tile, one-worker runs of ``figures/why-scaling-measured.json`` (production build, files, Gmsh load and ``verify_loaded``), by N."""
    rows = json.loads(SHARED.read_text())["rows"]
    return {
        r["n_features"]: r
        for r in rows
        if r["tiles_per_side"] == 1 and r["workers"] == 1
    }


def _timing_figures(head: list[dict]) -> None:
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
                    "render (16 workers)": d["render_brep_s"] + d["render_xao_s"],
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
        for side, label in zip(("api", "direct"), labels):
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
        for label, side in zip(labels, ("api", "direct")):
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
    for ax, r in zip(axes, sides):
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
    unique = dict(zip(names, handles))
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


def _timing_tables(head: list[dict], validator: list[dict]) -> dict:
    """The head-to-head tables (this script's run) and the direct-writer tail (``why-scaling-measured.json``)."""
    shared = _shared_rows()
    api_rows, load_rows, direct_rows, sidecar_rows = [], [], [], []
    quoted: dict = {"head": {}, "tail": {}}
    for r in head:
        api, d, ld = r["api"], r["direct"], r["load"]
        direct_total = d["plan_s"] + d["render_brep_s"] + d["render_xao_s"]
        api_rows.append(
            [
                f"{r['n']:,}",
                f"{d['plan_counts']['faces']:,}",
                f"{r['groups']:,}",
                _sec(api["occ_build_s"]),
                _sec(api["synchronize_s"]),
                _sec(api["physical_groups_s"]),
                _sec(api["write_xao_s"]),
                _sec(api["total_s"]),
                _sec(direct_total),
                f"{api['total_s'] / direct_total:.1f}x",
            ]
        )
        load_rows.append(
            [
                f"{r['n']:,}",
                f"{api['xao_bytes'] / 1e6:.1f} / {d['xao_bytes'] / 1e6:.1f} / {d['brep_bytes'] / 1e6:.1f}",
                _sec(ld["api_xao_s"]),
                _sec(ld["direct_xao_s"]),
                _sec(ld["direct_brep_s"]),
            ]
        )
        quoted["head"][r["n"]] = {
            "api_total_s": api["total_s"],
            "direct_total_s": direct_total,
            "ratio": api["total_s"] / direct_total,
            "groups_share": api["physical_groups_s"] / api["total_s"],
            "groups": r["groups"],
            "faces": d["plan_counts"]["faces"],
            "load_api_s": ld["api_xao_s"],
            "load_direct_xao_s": ld["direct_xao_s"],
            "load_direct_brep_s": ld["direct_brep_s"],
        }
    for n in (5000, 20000):
        e = shared[n]["emission"]
        load_rows.append(
            [
                f"{n:,}",
                f"— / {e['file_bytes']['xao'] / 1e6:.1f} / {e['file_bytes']['brep'] / 1e6:.1f}",
                "未執行",
                _sec(e["gmsh_load_s"]),
                "未量測",
            ]
        )
    for n, row in sorted(shared.items()):
        e = row["emission"]
        one, wide = e["direct_w1"], e["direct_w16"]
        total1 = one["plan"] + one["write_brep"] + one["write_xao"]
        total16 = wide["plan"] + wide["write_brep"] + wide["write_xao"]
        entities = (
            e["written_counts"]["vertices"]
            + e["written_counts"]["edges"]
            + e["written_counts"]["faces"]
        )
        direct_rows.append(
            [
                f"{n:,}",
                f"{e['written_counts']['faces']:,}",
                _sec(one["plan"]),
                _sec(one["write_brep"] + one["write_xao"]),
                _sec(total1),
                _sec(wide["plan"]),
                _sec(wide["write_brep"] + wide["write_xao"]),
                _sec(total16),
                f"{total1 / total16:.2f}x",
            ]
        )
        sidecar_rows.append(
            [
                f"{n:,}",
                _sec(one["reports"]),
                _sec(one["sidecar"]),
                f"{e['file_bytes']['sidecar'] / 1e6:.1f}",
                f"{e['file_bytes']['xao'] / 1e6:.1f}",
                f"{e['file_bytes']['brep'] / 1e6:.1f}",
                _sec(e["gmsh_load_s"]),
            ]
        )
        quoted["tail"][n] = {
            "write_w1_s": total1,
            "write_w16_s": total16,
            "write_speedup": total1 / total16,
            "plan_w1_s": one["plan"],
            "plan_share_w16": wide["plan"] / total16,
            "render_w1_s": one["write_brep"] + one["write_xao"],
            "render_w16_s": wide["write_brep"] + wide["write_xao"],
            "reports_s": one["reports"],
            "sidecar_s": one["sidecar"],
            "gmsh_load_s": e["gmsh_load_s"],
            "verify_loaded_s": e["verify_loaded_s"],
            "file_mb": {k: v / 1e6 for k, v in e["file_bytes"].items()},
            "build_s": shared[n]["build_wall_s"],
            "per_entity_us_w16": total16 / entities * 1e6,
            "written_counts": e["written_counts"],
        }
    _write(
        "emission-timing-api-table.md",
        _table(
            [
                "N",
                "face 數",
                "physical group 數",
                "OCC build (s)",
                "synchronize (s)",
                "physical groups (s)",
                "寫出 XAO (s)",
                "Gmsh-API 合計 (s)",
                "direct writer：plan 加算繪 (s)",
                "Gmsh-API 是 direct 的幾倍",
            ],
            api_rows,
        ),
    )
    _write(
        "emission-timing-load-table.md",
        _table(
            [
                "N",
                "檔案大小：API XAO / 直接 XAO / 直接 BREP (MB)",
                "載入 API XAO (s)",
                "載入直接 XAO (s)",
                "載入直接 BREP (s)",
            ],
            load_rows,
        ),
    )
    _write(
        "emission-timing-direct-table.md",
        _table(
            [
                "N",
                "寫出的 face 數",
                "plan (s)",
                "算繪，1 worker (s)",
                "合計，1 worker (s)",
                "plan，16 worker 的那次 (s)",
                "算繪，16 worker (s)",
                "合計，16 worker (s)",
                "1 worker 對 16 worker",
            ],
            direct_rows,
        ),
    )
    _write(
        "emission-sidecar-table.md",
        _table(
            [
                "N",
                "sliver 與 validation report (s)",
                "sidecar (s)",
                "sidecar (MB)",
                "XAO (MB)",
                "BREP (MB)",
                "Gmsh 載入 XAO (s)",
            ],
            sidecar_rows,
        ),
    )
    return quoted


VALIDATOR_ABOUT = {  # what each timed check or load-side phase loops over (checks of validation/ and verification/loaded.py)
    "face_areas": "載入側：每個 face 一次 gmsh mass 呼叫，O(F)",
    "planarity": "S9：各 arrangement 的 edge，用 noding kernel（逐 tile 平行）",
    "shells": "S9：每個 shell 的 face（逐 shell 平行）",
    "boundaries": "載入側：每個 edge、face、solid 一次 getBoundary，O(E + F)",
    "volumes": "載入側：每個 solid 一次 mass 積分，O(solid)",
    "counts_and_vertex_placement": "載入側：每個 vertex 一次 gmsh 呼叫，O(V)",
    "physical_groups": "載入側：每個 entity 問一次 group，O(entity)",
    "edges_in_faces": "S9：每個 face 的 edge 與 vertical edge，O(E + F)",
    "atomic_ids": "S9：各 arrangement 的 vertex、edge、face id 的唯一性，O(V + E + F)",
    "groups": "S9：每個 face 與 volume 的 group 歸屬，O(F)",
    "grid_and_lengths": "S9：每條 edge 的長度與 vertex 的 grid 位置，O(E)",
    "face_sides": "S9：每個 face 的兩側，O(F)",
    "areas": "S9：各 arrangement 的面積守恆，O(E)",
    "prism_areas": "S9：prism 面積在 cell 上守恆，O(E)",
    "arcs": "S9：arc edge 互不相交與偏差，O(arc edge)",
    "cylinders": "S9：curved wall 是單一 cylinder，O(wall)",
    "bands": "S9：band 面積單調，O(含 band 的 level 的 face)",
}


def _validator_tables(validator: list[dict]) -> dict:
    raw = {r["n"]: r for r in validator}
    ns = sorted(raw)
    faces = {n: raw[n]["counts"]["faces"] for n in ns}
    seconds: dict[str, dict[int, float]] = {}
    for n in ns:
        for name, value in raw[n]["validator_s"].items():
            seconds.setdefault(name, {})[n] = value
        for name, value in raw[n]["verify_phases_s"].items():
            seconds.setdefault(name, {})[n] = value
        for name in ("contact_check", "manifold_check"):
            seconds.setdefault(name, {})[n] = raw[n]["stage_s"][name]
    about = {
        **VALIDATOR_ABOUT,
        "contact_check": "S5：different-Net 接觸，每個 vertex 與 face，O(V + F)",
        "manifold_check": "S5：non-manifold pinch，O(E)",
    }
    missing = sorted(set(seconds) - set(about))
    assert not missing, f"checks without a description: {missing}"
    order = sorted(seconds, key=lambda k: -seconds[k][ns[-1]])
    table, slopes = [], {}
    for name in order:
        t = seconds[name]
        per = [t[n] / faces[n] * 1e6 for n in ns]
        slope = (
            math.log(t[ns[-1]] / t[ns[0]]) / math.log(faces[ns[-1]] / faces[ns[0]])
            if t[ns[0]] > 0 and t[ns[-1]] > 0
            else float("nan")
        )
        slopes[name] = slope
        table.append(
            [
                f"`{name}`",
                about[name],
                *[f"{t[n]:.3f} s" for n in ns],
                f"{per[0]:.2f} -> {per[-1]:.2f}",
                f"{slope:.2f}" if slope == slope else "n/a",
            ]
        )
    header = [
        "檢查",
        "做什麼",
        *[f"N = {n:,} (F = {faces[n]:,})" for n in ns],
        "每個 face 的 µs（首項到末項）",
        "對 F 的 slope",
    ]
    _write("emission-validator-table.md", _table(header, table, "llrrrrr"))
    run_rows = []
    for n in ns:
        r = raw[n]
        structural = (
            sum(r["validator_s"].values())
            + r["stage_s"]["contact_check"]
            + r["stage_s"]["manifold_check"]
        )
        run_rows.append(
            [
                f"{n:,}",
                f"{r['counts']['faces']:,}",
                f"{r['written']['faces']:,}",
                f"{structural:.2f} s",
                f"{r['verify_s']:.2f} s",
                f"{r['peak_rss_mb']:,.0f} MB",
            ]
        )
    _write(
        "emission-validator-run-table.md",
        _table(
            [
                "N",
                "規劃的 face 數",
                "寫出的 face 數",
                "各項結構檢查的時間相加（S5 與 S9）",
                "`verify_loaded`",
                "整個 process 的 peak RSS",
            ],
            run_rows,
        ),
    )
    big = ns[-1]
    heavy = [
        k for k in order if seconds[k][big] >= 0.1 and slopes[k] == slopes[k]
    ]  # the checks that cost 0.1 s or more at the largest N
    return {
        "slope_range_of_checks_over_0.1s": [
            min(slopes[k] for k in heavy),
            max(slopes[k] for k in heavy),
        ],
        "per_check_at_largest": {k: seconds[k][big] for k in order},
        "verify_total_s": {n: raw[n]["verify_s"] for n in ns},
        "structural_total_s": {
            n: sum(raw[n]["validator_s"].values())
            + raw[n]["stage_s"]["contact_check"]
            + raw[n]["stage_s"]["manifold_check"]
            for n in ns
        },
        "slopes": slopes,
    }


# --------------------------------------------------------------------------- loaded checks of the examples


def _analytic_volumes(loaded: dict[str, float]) -> str:
    """The pad's closed-form volumes: a 300 x 300 x 100 um substrate and the vacuum box minus the 0.2 um metal."""
    pad_area = (
        100 * 60 - 20 * 15 - 12 * 12 / 2 - 25 * (1 - math.pi / 4) - math.pi * 64
    )  # the scene's pad: notch, chamfer, rounded corner, hole
    ground_area = 300 * 300 - 160 * 120
    expected = {
        "substrate": 300 * 300 * 100.0,
        "vacuum": 300 * 300 * 100.0 - 0.2 * (ground_area + pad_area),
    }
    for name, want in expected.items():
        assert abs(loaded[name] - want) <= 1e-12 * want, (name, loaded[name], want)
    return "；".join(
        f"{k} {loaded[k]:.9g}，解析值 {v:.9g} µm³" for k, v in expected.items()
    )


def _oracle_agrees(model, plan, sidecar) -> dict:
    """The Gmsh-API oracle (an independent bottom-up replay) against the plan: entity counts, group members and the face areas.

    The oracle builds a curved wall with Gmsh's own surface construction, not as the exact cylinder the writer prints (the test side allows 1e-3
    for it), so the area error of the curved walls is reported separately from the planar faces.
    """
    import gmsh

    with gmsh_session():
        oracle = emit_with_gmsh_api(model)
        counts = {d: len(gmsh.model.getEntities(d)) for d in range(4)}
        inverse = {tag: pid for pid, tag in oracle.tags.items()}
        groups = {
            name: sorted(inverse[(dim, t)] for t in tags)
            for name, (dim, tags) in oracle.groups.items()
        }
        areas = {
            pid: gmsh.model.occ.getMass(2, tag)
            for pid, (dim, tag) in oracle.tags.items()
            if dim == 2
        }
    planned = plan.counts
    same_counts = counts == {
        0: planned["vertices"],
        1: planned["edges"],
        2: planned["faces"],
        3: planned["solids"],
    }
    same_groups = groups == {
        name: sorted(g.ids) for name, g in model.routed.groups.items() if name in groups
    } and set(groups) == set(model.routed.groups)
    want = {
        pid: sidecar["expected"]["face_area_um2"][i]
        for i, pid in enumerate(sidecar["binding"]["ids_by_tag"]["2"])
    }
    worst = {"planar": 0.0, "curved": 0.0}
    for pid, area in want.items():
        kind = "curved" if model.topology.faces[pid].curved else "planar"
        worst[kind] = max(worst[kind], abs(areas[pid] - area) / area)
    return {
        "entities_and_groups": same_counts and same_groups,
        "planar_area_error": worst["planar"],
        "curved_area_error": worst["curved"],
        "curved_faces": sum(1 for pid in want if model.topology.faces[pid].curved),
    }


def _loaded_check(model, folder: Path, mesh_size: float) -> dict:
    """Write a model, load it in a fresh Gmsh session and run the mesher's checks and a coarse mesh."""
    import gmsh

    artifacts, plan, sidecar = _bundle(model, folder)
    with gmsh_session(artifacts.xao):
        report = verify_loaded(sidecar)
        masses = {
            model.topology.volumes[vid].domain.name: gmsh.model.occ.getMass(3, i + 1)
            for i, vid in enumerate(plan.solid_ids)
        }
        apply_sidecar_embeds(sidecar)
        if model.insets:
            refine_near(
                model,
                plan.tags(),
                ("band_pad_top", "band_pad_bottom", "band_gap"),
                mesh_size,
                1.0,
                (1.0, 20.0),
            )
        mesh_coarsely(mesh_size)
        conformity = check_mesh_conformity(model, plan.tags())
        bands = (
            band_mesh_report(
                model,
                plan.tags(),
                {
                    f.family_id: f.support
                    for f in model.labelled.catalogue.families
                    if f.support
                },
            )
            if model.insets
            else {}
        )
    return {
        "report": report,
        "masses": masses,
        "conformity": conformity,
        "bands": bands,
        "plan": plan,
        "sidecar": sidecar,
    }


def _validation_table() -> None:
    cases = ("pad_in_ground_A", "pad_in_ground_B", "airbridge_B")
    sizes = {"pad_in_ground_A": 40.0, "pad_in_ground_B": 60.0, "airbridge_B": 60.0}
    builders = _case_models()
    results = {}
    for case in cases:
        model = builders[case]()
        with tempfile.TemporaryDirectory(prefix="emission-validation-") as scratch:
            checked = _loaded_check(model, Path(scratch), sizes[case])
            results[case] = {
                **checked,
                "oracle": _oracle_agrees(model, checked["plan"], checked["sidecar"]),
            }

    def counts(case: str) -> str:
        n = results[case]["report"].counts["native"]
        return " / ".join(str(n[str(d)]) for d in range(4))

    def mismatches(case: str) -> str:
        m = results[case]["report"].boundaries_mismatching
        return f"{m['edges']} / {m['faces']} / {m['solids']}"

    def groups(case: str) -> str:
        g = results[case]["report"].groups
        return f"{g['loaded']} of {g['planned']}"

    def mesh(case: str) -> str:
        c = results[case]["conformity"]
        return f"{c['nodes']} nodes，{c['tets']} tets，{c['duplicate_nodes']} 個重複 node，conformal：{'是' if c['conformal'] else '否'}"

    analytic = _analytic_volumes(results["pad_in_ground_B"]["masses"])
    rows = [
        [
            "Gmsh 中的元素數（vertex / edge / face / solid），與規劃相符",
            *(counts(c) for c in cases),
        ],
        [
            "以 `getBoundary` 檢查的邊界不一致數（edge / face / solid）",
            *(mismatches(c) for c in cases),
        ],
        [
            "vertex 與規劃位置的最大偏差 (µm)",
            *(f"{results[c]['report'].max_vertex_offset_um:g}" for c in cases),
        ],
        [
            "從 XAO 載入的 physical group 數（BREP 不含 group）",
            *(groups(c) for c in cases),
        ],
        [
            "volume 對規劃的最大相對誤差（XAO）",
            *(f"{results[c]['report'].max_relative_volume_error:.2g}" for c in cases),
        ],
        ["載入後 XAO 的粗略 3-D mesh", *(mesh(c) for c in cases)],
        [
            "face 面積對規劃的最大相對誤差",
            *(
                f"{results[c]['report'].max_relative_face_area_error:.2g}"
                for c in cases
            ),
        ],
        [
            "與 Gmsh-API oracle 的 entity 數與 group 成員相同",
            *(
                "是" if results[c]["oracle"]["entities_and_groups"] else "否"
                for c in cases
            ),
        ],
        [
            "與 oracle 的 face 面積最大相對差（平面 face；curved wall）",
            *(
                f"{results[c]['oracle']['planar_area_error']:.2g}；{results[c]['oracle']['curved_area_error']:.2g}（{results[c]['oracle']['curved_faces']} 面）"
                if results[c]["oracle"]["curved_faces"]
                else f"{results[c]['oracle']['planar_area_error']:.2g}；無 curved wall"
                for c in cases
            ),
        ],
        ["解析 volume（封閉形式）", "未計算", analytic, "未計算"],
        [
            "`verify_loaded` 是否通過",
            *("是" if results[c]["report"].ok else "否" for c in cases),
        ],
    ]
    _write(
        "emission-validation-table.md", _table(["載入後的檢查", *cases], rows, "llll")
    )
    _bands_table()


def _bands_table() -> None:
    rows = []
    builders = _case_models()
    for case, size in (
        ("pad_in_ground_A_bands", 40.0),
        ("pad_in_ground_B_bands", 60.0),
        ("pad_in_ground_C_bands", 60.0),
    ):
        model = builders[case]()
        with tempfile.TemporaryDirectory(prefix="emission-bands-") as scratch:
            r = _loaded_check(model, Path(scratch), size)
        report, conf, bands = r["report"], r["conformity"], r["bands"]
        passing = sum(
            1
            for b in bands.values()
            if b["all_shared_by_tets"] and b["centroids_in_band"]
        )
        rows.append(
            [
                f"`{case}`",
                f"{report.groups['loaded']} of {report.groups['planned']}",
                str(len(bands)),
                f"{report.max_relative_face_area_error:.2g}",
                f"{report.max_relative_volume_error:.2g}",
                f"{conf['nodes']:,} nodes，{conf['tets']:,} tets，conformal：{'是' if conf['conformal'] else '否'}",
                f"{passing} of {len(bands)}",
            ]
        )
    _write(
        "emission-bands-table.md",
        _table(
            [
                "case",
                "已載入 groups",
                "band group 數",
                "face 面積最大誤差",
                "volume 最大誤差",
                "粗略 3-D mesh",
                "triangle 全為 tet face 且落在 band 內的 band group 數",
            ],
            rows,
            "lrrrrll",
        ),
    )


# --------------------------------------------------------------------------- OCCT checks and the catalogue


def _occt_tables() -> dict:
    """Official OCCT checks over the seven running-example models (BREP and XAO), and the negative controls."""
    rows = []
    seen = {
        "brep_check_ms": [],
        "bop_ms": [],
        "total_ms": [],
        "reports": 0,
        "all_ok": True,
        "max_face_error": 0.0,
        "max_volume_error": 0.0,
        "excluded_free_faces": 0,
    }
    for case, build in _case_models().items():
        model = build()
        with tempfile.TemporaryDirectory(prefix="emission-occt-") as scratch:
            artifacts, plan, _ = _bundle(model, Path(scratch))
            for fmt, path in (("brep", artifacts.brep), ("xao", artifacts.xao)):
                r = check_file(path, model.topology, plan).to_dict()
                shape, props = r["shape_analysis"], r["properties"]
                seen["brep_check_ms"].append(r["timings_s"]["brep_check"] * 1000)
                seen["bop_ms"].append(r["timings_s"]["bop_argument_analyzer"] * 1000)
                seen["total_ms"].append(sum(r["timings_s"].values()) * 1000)
                seen["reports"] += 1
                seen["all_ok"] &= r["ok"]
                seen["max_face_error"] = max(
                    seen["max_face_error"], props["faces"]["max_relative_error"]
                )
                seen["max_volume_error"] = max(
                    seen["max_volume_error"],
                    max(x["relative_error"] for x in props["solids"]),
                )
                seen["excluded_free_faces"] += r["bop_argument_analyzer"][
                    "excluded_free_faces"
                ]
                rows.append(
                    [
                        f"`{case}`",
                        fmt.upper(),
                        "是" if r["ok"] else "否",
                        "是" if r["brepcheck"]["valid"] else "否",
                        str(r["brepcheck"]["faulty_count"]),
                        "有" if r["bop_argument_analyzer"]["faulty"] else "無",
                        str(r["bop_argument_analyzer"]["result_count"]),
                        str(r["bop_argument_analyzer"]["excluded_free_faces"]),
                        f"{shape['shells_with_free_edges']} / {shape['shells_with_bad_edges']}",
                        f"{shape['shortest_edge_um']:.3g}",
                        f"{props['faces']['max_relative_error']:.2g}",
                        f"{max(x['relative_error'] for x in props['solids']):.2g}",
                    ]
                )
    _write(
        "emission-occt-table.md",
        _table(
            [
                "case",
                "格式",
                "ok",
                "BRepCheck valid",
                "faulty 元素數",
                "BOP 偵測到 fault",
                "BOP result 數",
                "BOP 排除的 free face 數",
                "有 free／bad edge 的 shell 數",
                "最短 edge (µm)",
                "face 面積最大相對誤差",
                "solid volume 最大相對誤差",
            ],
            rows,
            "llllrlrrlrrr",
        ),
    )
    model = kit.pad_model("B", None)
    with tempfile.TemporaryDirectory(prefix="emission-controls-") as scratch:
        artifacts, plan, _ = _bundle(model, Path(scratch))
        rows = []
        for defect, mutate in mutations.CONTROLS.items():
            started = time.perf_counter()
            report = check_shape(
                mutate(read_brep(artifacts.brep)), model.topology, plan, source=defect
            )
            millis = (time.perf_counter() - started) * 1000
            d = report.to_dict()
            statuses = sorted(d["brepcheck"]["histogram"])
            bop = sorted({x["status"] for x in d["bop_argument_analyzer"]["results"]})
            seen_shape = bool(
                d["shape_analysis"]["shells_with_free_edges"]
                or d["shape_analysis"]["shells_with_bad_edges"]
            )

            def caught(flag: bool, labels: list[str]) -> str:
                return (
                    ("抓到（" + "，".join(labels) + "）")
                    if flag and labels
                    else ("抓到" if flag else "未抓到")
                )

            rows.append(
                [
                    f"`{defect}`",
                    caught(not d["brepcheck"]["valid"], statuses),
                    caught(d["bop_argument_analyzer"]["faulty"], bop),
                    caught(seen_shape, []),
                    caught(bool(d["properties"]["mismatches"]), []),
                    f"{millis:.1f}",
                ]
            )
    _write(
        "emission-occt-controls-table.md",
        _table(
            [
                "注入的缺陷",
                "BRepCheck_Analyzer",
                "BOPAlgo_ArgumentAnalyzer",
                "ShapeAnalysis_Shell",
                "plan 面積／volume 比對",
                "全部檢查耗時 (ms)",
            ],
            rows,
            "lllllr",
        ),
    )
    return {
        key: ([min(v), max(v)] if isinstance(v, list) else v) for key, v in seen.items()
    }


def _extreme_tables() -> dict:
    """The extreme-case catalogue under every route: build, write, load, coarse mesh and the OCCT checks, with timings."""

    results = []
    for fixture, route in extreme_cases():
        row = {"fixture": fixture.name, "route": route}
        expected = fixture.expect_error.get(route)
        started = time.perf_counter()
        try:
            model = build_model(fixture, route)
        except Exception as error:  # noqa: BLE001 -- a refusal is a measurement
            row["refused"] = type(error).__name__
            row["refusal_ms"] = (time.perf_counter() - started) * 1000
            row["expected"] = expected is not None and isinstance(error, expected)
            results.append(row)
            continue
        assert expected is None, f"{fixture.name} {route} was expected to be refused"
        with tempfile.TemporaryDirectory(prefix="emission-extreme-") as scratch:
            artifacts, plan, sidecar = _bundle(model, Path(scratch))
            with gmsh_session(artifacts.xao):
                row["load_ok"] = verify_loaded(sidecar).ok
                apply_sidecar_embeds(sidecar)
                mesh_coarsely(4.0)
                row["mesh_conformal"] = check_mesh_conformity(model, plan.tags())[
                    "conformal"
                ]
            report = check_file(
                artifacts.xao, model.topology, plan, self_interference=True
            )
            t = report.timings
            row.update(
                {
                    "occt_ok": report.ok,
                    "brep_check_ms": t["brep_check"] * 1000,
                    "bop_ms": t["bop_argument_analyzer"] * 1000,
                    "shape_plan_ms": (
                        t["shape_analysis"] + t.get("plan_properties", 0.0)
                    )
                    * 1000,
                    "occt_total_ms": sum(t.values()) * 1000,
                }
            )
        results.append(row)
    fixtures = {r["fixture"] for r in results}
    passed = [
        r
        for r in results
        if "refused" not in r and r["load_ok"] and r["mesh_conformal"] and r["occt_ok"]
    ]
    built = [r for r in results if "refused" not in r]
    refused = [r for r in results if "refused" in r]
    assert len(passed) == len(built) and all(r["expected"] for r in refused)
    count = lambda kind: sum(r["refused"] == kind for r in refused)  # noqa: E731
    rows = [
        [
            "通過全部檢查",
            str(len(passed)),
            "build 與結構檢查、寫出 BREP 與 XAO、Gmsh 載入與 `verify_loaded`、conformal mesh、`BRepCheck_Analyzer`、`BOPAlgo_ArgumentAnalyzer`（全部模式，含 self-interference）、`ShapeAnalysis`、與 plan 的比對",
        ],
        [
            "預期的拒絕：build 階段",
            str(len(refused)),
            "、".join(
                f"`{k}` {count(k)} 個" for k in sorted({r["refused"] for r in refused})
            )
            + "；在 build 開始後數毫秒內丟出",
        ],
    ]
    _write(
        "emission-extreme-table.md",
        _table(
            [
                "結果",
                f"組合數（共 {len(results)} 個，{len(fixtures)} 個 fixture）",
                "內容",
            ],
            rows,
            "lrl",
        ),
    )
    spans = [
        ("`BRepCheck_Analyzer`", "brep_check_ms"),
        ("`BOPAlgo_ArgumentAnalyzer`（全部模式）", "bop_ms"),
        ("`ShapeAnalysis` 加 plan 比對", "shape_plan_ms"),
        ("OCCT 檢查合計（含讀檔）", "occt_total_ms"),
    ]
    trows = [
        [
            name,
            f"{min(r[key] for r in passed):.1f}",
            f"{max(r[key] for r in passed):.1f}",
        ]
        for name, key in spans
    ]
    trows.append(
        [
            "build 階段的拒絕（丟出錯誤所需時間）",
            f"{min(r['refusal_ms'] for r in refused):.1f}",
            f"{max(r['refusal_ms'] for r in refused):.1f}",
        ]
    )
    _write(
        "emission-extreme-timing-table.md",
        _table(["檢查", "最短 (ms)", "最長 (ms)"], trows),
    )
    return {
        "combinations": len(results),
        "fixtures": len(fixtures),
        "passed": len(passed),
        "refused": {k: count(k) for k in sorted({r["refused"] for r in refused})},
        "occt_ms": {
            key: [min(r[key] for r in passed), max(r[key] for r in passed)]
            for _, key in spans
        },
        "refusal_ms": [
            min(r["refusal_ms"] for r in refused),
            max(r["refusal_ms"] for r in refused),
        ],
    }


# --------------------------------------------------------------------------- conceptual figure


def _five_edge_wall_figure() -> None:
    """Conceptual: the cylinder wall of a round pad unrolled (angle across, z up), before and after splitting at a foreign vertex."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, split in zip(axes, (False, True)):
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


# --------------------------------------------------------------------------- main


def main() -> None:
    for flag, function in (
        ("--head-one", head_one),
        ("--validator-one", validator_one),
    ):
        if flag in sys.argv:
            print(json.dumps(function(int(sys.argv[sys.argv.index(flag) + 1]))))
            return
    if "--measure-only" in sys.argv:
        measure()
        return
    if "--occt-only" in sys.argv:
        print(json.dumps({"examples": _occt_tables(), "catalogue": _extreme_tables()}))
        return
    kit.ensure_plotly_js()
    if "--measure" in sys.argv or not DATA.exists():
        measure()
    numbers: dict = {}
    with tempfile.TemporaryDirectory(prefix="emission-example-") as scratch:
        model, plan, sidecar, artifacts = _running_example(Path(scratch))
        _check_shell_references(plan, artifacts.brep)
        numbers = _example_numbers(model, plan, sidecar)
        numbers["sidecar_bytes"] = artifacts.sidecar.stat().st_size
        numbers["bytes"] = {
            "brep": artifacts.brep.stat().st_size,
            "xao": artifacts.xao.stat().st_size,
        }
        _counts_table(model, plan)
        _tags_and_groups_tables(model.topology, plan, sidecar)
        numbers["shared_face"] = _shared_face_figure(model.topology, plan)
        numbers["groups_3d"] = _groups_3d_figures(plan, artifacts.xao)
    data = json.loads(DATA.read_text())
    _timing_figures(data["head"])
    numbers["timing"] = _timing_tables(data["head"], data["validator"])
    numbers["validator"] = _validator_tables(data["validator"])
    _validation_table()
    numbers["occt"] = _occt_tables()
    numbers["catalogue"] = _extreme_tables()
    _five_edge_wall_figure()
    (FIGURES / "emission-example-numbers.json").write_text(
        json.dumps(numbers, indent=1, default=str)
    )
    print("wrote emission-example-numbers.json")


if __name__ == "__main__":
    main()
