"""Figures and numbers of ``validation.qmd``: what each invariant catches, shown by breaking a valid model on purpose.

The prototype's own test removes one face from a shell and expects the shell check to fail. This script does the same on the
running example (Route B) and, in addition, applies eight other deliberate defects, each to a fresh build, and records which
named checks fail. It also records the two errors that stop a build before validation (overlapping Nets, a wrong route).

Writes into ``docs/design/sgb/figures/``:

* ``validation-shell.svg/.html``                the vacuum shell with its top face removed; the edges the shell now uses once in red;
* ``validation-tampered-shell-report.json``     the whole validation report of that tampered model (20 checks, 2 failing);
* ``validation-numbers.json``                   the table of defects and failing checks, the errors, and the check inventory.

Run (from the package root):  .venv/bin/python docs/design/sgb/_figures/validation.py
"""

from __future__ import annotations

import json
from collections import Counter

from labels_common import FIGURES, GROUP_COLOURS, PALETTE, compact_triangles, go, mesh_trace, np, plt, save_plotly, save_svg, write_numbers
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from sgbproto.errors import ContactError, OverlapError, RouteError, SgbError, ValidationError
from sgbproto.pipeline import build
from sgbproto.records import Footprint, ModelSpec, PrismRecord, z_value
from sgbproto.scenes import pad_in_ground, rect_um, to_grid
from sgbproto.validate import validate

G = 0.001


def revalidate(result):
    return validate(result.spec, result.levels, result.slabs, result.level_layers, result.topology, check_planarity=True)


def fresh():
    return build(pad_in_ground("B"))


# --------------------------------------------------------------------------- deliberate defects (each on a fresh build)


def vacuum_volume(topo):
    return next(v for v in topo.volumes.values() if v.emitted and v.domain.name == "vacuum")


def top_face_id(topo) -> str:
    vol = vacuum_volume(topo)
    return next(fid for fid, _ in vol.shells[0].faces if topo.faces[fid].kind == "horizontal" and topo.faces[fid].group == "EXT::vacuum")


def drop_top_face(result):
    vol = vacuum_volume(result.topology)
    top = top_face_id(result.topology)
    vol.shells[0].faces = tuple(e for e in vol.shells[0].faces if e[0] != top)


def flip_one_face(result):
    vol = vacuum_volume(result.topology)
    (fid, sign), *rest = vol.shells[0].faces
    vol.shells[0].faces = ((fid, -sign), *rest)


def face_in_two_groups(result):
    groups = result.topology.groups
    groups["MS::pad"]["ids"] = groups["MS::pad"]["ids"] + [groups["MA::pad"]["ids"][0]]


def empty_group(result):
    result.topology.groups["PORT::p1"]["ids"] = []


def face_without_group(result):
    result.topology.groups["MA::Ground"]["ids"] = result.topology.groups["MA::Ground"]["ids"][1:]


def union_unknown_member(result):
    result.topology.unions["PEC::pad"] = result.topology.unions["PEC::pad"] + ["MA::pad_typo"]


def interior_face_one_sided(result):
    topo = result.topology
    face = next(f for f in topo.faces.values() if f.group == "SA::substrate|vacuum")
    face.pos_volume = None


def banded():
    return build(pad_in_ground("A", bands_um=(0.1, 0.5, 2.0)))


def band_loses_a_cell(result):
    """Remove one cell of the deepest band: the bands no longer partition the support region."""
    from sgbproto.records import LevelLabel

    layer = result.level_layers[1]
    victim = next(fi for fi in range(1, len(layer.arr.faces)) if layer.labels[fi].bands and layer.labels[fi].bands[0][1] == 3)
    saved = layer.labels[victim]
    layer.dissolved.labels[victim] = LevelLabel(saved.below, saved.above, saved.sheet, ())


def area_not_conserved(result):
    result.slabs[1].raw["total_twice_area"] += 2


DEFECTS = [
    ("remove the top face from the vacuum shell", drop_top_face, "the shell is open"),
    ("flip the orientation of one shell face", flip_one_face, "faces disagree about the inside"),
    ("put one face into two physical groups", face_in_two_groups, "a face is double-booked"),
    ("empty a physical group", empty_group, "a group names nothing"),
    ("drop a face from its physical group", face_without_group, "an emitted face has no group"),
    ("name a group that does not exist in a union", union_unknown_member, "a union has a typo"),
    ("remove the volume on one side of an interface face", interior_face_one_sided, "an interface face with one side"),
    ("change the recorded area of a slab arrangement", area_not_conserved, "area lost or gained by an arrangement"),
    ("remove one cell of the deepest inset band (banded model)", band_loses_a_cell, "bands no longer partition the support region"),
]
BUILDERS = {band_loses_a_cell: banded}


def run_defects(numbers: dict) -> None:
    rows = []
    for title, apply, meaning in DEFECTS:
        result = BUILDERS.get(apply, fresh)()
        apply(result)
        report = revalidate(result)
        rows.append({"defect": title, "meaning": meaning, "failing_checks": [c.name for c in report.failures],
                     "details": {c.name: c.detail[:200] for c in report.failures}})
    numbers["defects"] = rows


# --------------------------------------------------------------------------- the example failure report and its figure


def tampered_report(numbers: dict):
    result = fresh()
    assert not result.report.failures
    topo = result.topology
    vol = vacuum_volume(topo)
    shell_before = vol.shells[0]
    top = top_face_id(topo)
    removed = topo.faces[top]
    faces_before = len(shell_before.faces)
    drop_top_face(result)
    report = revalidate(result)
    try:
        report.raise_if_failed()
        message = None
    except ValidationError as e:
        message = str(e)
    (FIGURES / "validation-tampered-shell-report.json").write_text(json.dumps(
        {"model": "pad_in_ground B, vacuum shell without its top face (face " + top + ")", **report.to_dict(), "raised": message}, indent=1) + "\n")
    # edges the tampered shell uses once
    uses = Counter()
    for fid, _ in vol.shells[0].faces:
        for lp in topo.faces[fid].loops:
            for eid, _s in lp:
                uses[eid] += 1
    free = [eid for eid, n in uses.items() if n == 1]
    numbers["tampered_shell"] = {
        "removed_face": top, "removed_face_group": removed.group, "shell_faces_before": faces_before,
        "shell_faces_after": len(vol.shells[0].faces), "edges_used_once": len(free),
        "checks_total": len(report.checks), "checks_failed": [c.name for c in report.failures],
        "checks_passed": sum(c.passed for c in report.checks), "warnings": [c.name for c in report.checks if c.warning],
        "raised_message": message,
        "failing_details": {c.name: c.detail for c in report.failures},
    }
    return result, vol, free, top


def figure_shell(result, vol, free, top) -> None:
    topo = result.topology
    tris = compact_triangles(topo, mesh_size=60.0, mesh_size_min=4.0)
    fig = plt.figure(figsize=(9.5, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    polys, colours, traces = [], [], []
    for fid, _ in vol.shells[0].faces:
        if fid not in tris:
            continue
        pts, tri = tris[fid]
        group = topo.faces[fid].group
        colour = GROUP_COLOURS.get(group.split("::")[0], "#999")
        alpha = 0.1 if group.startswith("EXT") else 0.5
        traces.append(mesh_trace(pts, tri, colour, group, f"{group}<br>{fid}", opacity=alpha))
        polys.extend(pts[t] for t in tri)
        from matplotlib.colors import to_rgba

        colours.extend([to_rgba(colour, alpha)] * len(tri))
    ax.add_collection3d(Poly3DCollection(polys, facecolors=colours, edgecolors="none"))
    lines = []
    for eid in free:
        e = topo.edges[eid]
        a, b = topo.vertices[e.v0], topo.vertices[e.v1]
        lines.append(((a.x * G, a.y * G, float(a.z)), (b.x * G, b.y * G, float(b.z))))
        ax.plot([lines[-1][0][0], lines[-1][1][0]], [lines[-1][0][1], lines[-1][1][1]], [lines[-1][0][2], lines[-1][1][2]], color=PALETTE["bad"], lw=3.0)
    ax.set_xlim(-150, 150)
    ax.set_ylim(-150, 150)
    ax.set_zlim(0, 100)
    ax.set_box_aspect((1, 1, 0.45))
    ax.view_init(elev=24, azim=-56)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_zlabel("z (µm)")
    ax.set_title("Vacuum shell with its top face removed: the four edges now used once (red)", fontsize=10)
    ax.legend(handles=[plt.Line2D([0], [0], color=PALETTE["bad"], lw=3, label=f"{len(free)} edges used once by the shell"),
                       Patch(facecolor="#9ecae1", label="remaining shell faces (outer chip faces drawn faint)")], loc="upper left", fontsize=8, frameon=False)
    save_svg(fig, "validation-shell")
    fig2 = go.Figure(traces)
    for (a, b) in lines:
        fig2.add_trace(go.Scatter3d(x=[a[0], b[0]], y=[a[1], b[1]], z=[a[2], b[2]], mode="lines", line=dict(color=PALETTE["bad"], width=8),
                                    name="edge used once", showlegend=False, hoverinfo="name"))
    fig2.update_layout(title="Vacuum shell without its top face: edges used once (red)", width=1000, height=720,
                       scene=dict(aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.45)))
    save_plotly(fig2, "validation-shell")


# --------------------------------------------------------------------------- errors before validation, check inventory


def early_errors(numbers: dict) -> None:
    base = pad_in_ground("B")
    box = Footprint.from_rings(to_grid(base.grid, rect_um(-60, -40, -40, -20)))
    other = PrismRecord("other_net", box, z_value(0), z_value("0.2"), "Al", "conductor", net_id="other")
    out = {}
    try:
        build(ModelSpec(base.grid, base.prisms + (other,), name="bad", meta=base.meta))
    except OverlapError as e:
        out["OverlapError (labelling, S5)"] = str(e)
    try:
        build(pad_in_ground("B"), route="A")
    except RouteError as e:
        out["RouteError (route check, before S2)"] = str(e)
    # different-Net contact (shared edge) and a same-Net pinch (finite metal meeting only at a corner)
    touching = PrismRecord("touching_net", Footprint.from_rings(to_grid(base.grid, rect_um(-60, -20, -50, 20))), z_value(0), z_value("0.2"),
                           "Al", "conductor", net_id="other")
    try:
        build(ModelSpec(base.grid, base.prisms + (touching,), name="bad", meta=base.meta))
    except ContactError as e:
        out["ContactError (contact check, after the arrangements)"] = str(e)
    pinch = PrismRecord("pinch", Footprint.from_rings(to_grid(base.grid, rect_um(50, 18, 70, 40))), z_value(0), z_value("0.2"),
                        "Al", "conductor", net_id="pad")  # same Net, touches the pad only at the corner (50, 18)
    try:
        build(ModelSpec(base.grid, base.prisms + (pinch,), name="pinch", meta=base.meta))
    except ValidationError as e:
        out["ValidationError (same-Net pinch at a corner, finite metal)"] = str(e)
    from fractions import Fraction

    from sgbproto.records import CurveFamilyRecord, LineCarrier, Lineage

    carrier = LineCarrier("r/0", (-70000, -8000), (-60000, -8000), Lineage("family_member", family_id="r", member="r/0"))
    family = CurveFamilyRecord("r", "refinement", (carrier,), (Fraction(1, 10),), "refinement_only")
    try:
        build(ModelSpec(base.grid, base.prisms, (family,), name="refine", meta=base.meta))
    except SgbError as e:
        out["SgbError (refinement curve in a cell with no face, assembly)"] = str(e)
    numbers["early_errors"] = out


def inventory(numbers: dict) -> None:
    result = fresh()
    numbers["checks_of_a_valid_build"] = [{"name": c.name, "passed": c.passed, "warning": c.warning, "detail": c.detail} for c in result.report.checks]


def main() -> None:
    numbers: dict = {}
    run_defects(numbers)
    result, vol, free, top = tampered_report(numbers)
    figure_shell(result, vol, free, top)
    early_errors(numbers)
    inventory(numbers)
    write_numbers("validation", numbers)
    print("validation: done")


if __name__ == "__main__":
    main()
