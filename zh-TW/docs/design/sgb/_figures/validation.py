"""Figures and numbers of ``validation.qmd``: what each invariant catches, shown by breaking a valid model on purpose.

The production test suite removes a face from a model and expects a check to fail.  This script does the same on the running
example (the real input ``gds_pad_in_ground`` under Route B) and applies nine other deliberate defects, each to a fresh build,
and records which named checks fail (it re-runs ``validate_model`` on the tampered model, the way ``tests/test_sgb_validation.py``
does).  It also records the errors that stop a build before validation.

Writes into ``docs/design/sgb/figures/``:

* ``validation-shell.svg/.html``                the outer shell of the vacuum volume with its top face removed; the edges the shell
  now uses once in red;
* ``validation-tampered-shell-report.json``     the whole validation report of that tampered model (31 checks, 2 failing);
* ``validation-numbers.json``                   the table of defects and failing checks, the errors, and the check inventory.

The defect "a top arc moved to another circle" needs a curved wall; the real input has none, so that one row is built on the
stage-level scene ``pad_in_ground("B")`` of ``tests/sgb_scenes.py`` and says so in the numbers.

Run:  SGB_CORE=<sgb-core checkout> <sgb-core>/.artifacts/env/bin/python docs/design/sgb/_figures/validation.py
"""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path

from labels_common import (
    FIGURES,
    GROUP_COLOURS,
    PALETTE,
    build_model,
    compact_triangles,
    go,
    mesh_trace,
    plt,
    real_catalogue,
    save_plotly,
    save_svg,
    write_numbers,
)
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scgsim.geometry.sgb.cutfamilies.requests import (
    ConductorSurface,
    InsetRequest,
    RefinementRequest,
    SolutionInterface,
)
from scgsim.geometry.sgb.emission.plan import plan_brep
from scgsim.geometry.sgb.emission.writer import write_brep_file, write_xao_file
from scgsim.geometry.sgb.errors import (
    ContactError,
    NonManifoldError,
    OverlapError,
    RouteError,
    SgbError,
    ValidationError,
)
from scgsim.geometry.sgb.parallel import Runtime
from scgsim.geometry.sgb.records.grid import exact_z
from scgsim.geometry.sgb.records.labels import LevelLabel
from scgsim.geometry.sgb.records.prisms import PrismRecord
from scgsim.geometry.sgb.validation.validate import validate_model
from scgsim.geometry.sgb.verification.occt import check_file
from sgb_scenes import airbridge, footprint, rect_um
from sgb_scenes import pad_in_ground as stage_pad_in_ground

G = 0.001
MARGINS = (0.1, 0.5, 2.0)


def _revalidate(model):
    return validate_model(model.labelled, model.topology, model.routed, Runtime())


def _fresh():
    return build_model(real_catalogue(), "B")


# --------------------------------------------------------------------------- deliberate defects (each on a fresh build)


def _vacuum_volume(model):
    return next(
        v
        for v in model.topology.volumes.values()
        if v.id in model.routed.emitted_volumes and v.domain.name == "VACUUM_REGION"
    )


def _top_face_id(model) -> str:
    return next(
        fid
        for fid, _ in _vacuum_volume(model).shells[0].faces
        if model.routed.face_group[fid] == "EXT::VACUUM_REGION::top"
    )


def _drop_top_face(model):
    shell = _vacuum_volume(model).shells[0]
    top = _top_face_id(model)
    shell.faces = tuple(e for e in shell.faces if e[0] != top)


def _flip_one_face(model):
    shell = _vacuum_volume(model).shells[0]
    (fid, sign), *rest = shell.faces
    shell.faces = ((fid, -sign), *rest)


def _face_in_two_groups(model):
    groups = model.routed.groups
    groups["MS::pad::bottom"].ids = groups["MS::pad::bottom"].ids + [
        groups["MA::pad::top"].ids[0]
    ]


def _empty_group(model):
    model.routed.groups["PORT::PORT_SHEET__P1__0000"].ids = []


def _face_without_group(model):
    model.routed.groups["MA::pad::sidewall"].ids = model.routed.groups[
        "MA::pad::sidewall"
    ].ids[1:]


def _union_unknown_member(model):
    model.routed.unions["NET::P"] = model.routed.unions["NET::P"] + ["MA::pad_typo"]


def _interior_face_one_sided(model):
    face = next(
        f
        for f in model.topology.faces.values()
        if model.routed.face_group[f.id] == "SA::substrate|VACUUM_REGION::interface"
    )
    face.pos_volume = None


def _banded():
    insets = (
        InsetRequest("band_pad_top", ConductorSurface("pad", "top"), MARGINS),
        InsetRequest("band_pad_bottom", ConductorSurface("pad", "bottom"), MARGINS),
        InsetRequest(
            "band_gap",
            SolutionInterface("substrate", "VACUUM_REGION", z_um=0.0),
            MARGINS,
        ),
    )
    return build_model(real_catalogue(), "B", insets=insets)


def band_loses_a_cell(model):
    """Remove one cell of the deepest band of the pad top: the bands no longer partition the support region."""
    layer = next(
        candidate
        for candidate in model.labelled.level_layers
        if candidate.plane == "level[1/5]"
    )
    victim = next(
        fi
        for fi in range(1, len(layer.arr.faces))
        if any(f == "band_pad_top" and k == 3 for f, k in layer.labels[fi].bands)
    )
    saved = layer.labels[victim]
    layer.dissolved.labels[victim] = LevelLabel(
        saved.below, saved.above, saved.sheet, ()
    )


def _area_not_conserved(model):
    next(
        candidate
        for candidate in model.labelled.level_layers
        if candidate.plane == "level[0]"
    ).raw.total_twice_area += 2


def _staged():
    return build_model(stage_pad_in_ground("B"), "B")


def _arc_on_another_circle(model):
    wall = next(f for f in model.topology.faces.values() if f.curved)
    top = model.topology.edges[wall.cylinder[1]]
    top.curve = replace(top.curve, radius=top.curve.radius + 1.0)


DEFECTS = [
    (
        "remove the top face from the outer shell of the vacuum volume",
        _drop_top_face,
        "the shell is open",
    ),
    (
        "flip the orientation of one shell face",
        _flip_one_face,
        "faces disagree about the inside",
    ),
    (
        "put one face into two physical groups",
        _face_in_two_groups,
        "a face is double-booked",
    ),
    ("empty a physical group", _empty_group, "a group names nothing"),
    (
        "drop a face from its physical group",
        _face_without_group,
        "an emitted face has no group",
    ),
    (
        "name a group that does not exist in a union",
        _union_unknown_member,
        "a union has a typo",
    ),
    (
        "remove the volume on one side of an interface face",
        _interior_face_one_sided,
        "an interface face with one side",
    ),
    (
        "change the recorded area of a level arrangement",
        _area_not_conserved,
        "area lost or gained by an arrangement",
    ),
    (
        "remove one cell of the deepest band of the pad top (banded model)",
        band_loses_a_cell,
        "bands no longer partition the support region",
    ),
    (
        "move the top arc of a curved wall to another circle (stage-level scene with arcs)",
        _arc_on_another_circle,
        "a curved wall is not one cylinder",
    ),
]
BUILDERS = {band_loses_a_cell: _banded, _arc_on_another_circle: _staged}


def _run_defects(numbers: dict) -> None:
    rows = []
    for title, apply, meaning in DEFECTS:
        model = BUILDERS.get(apply, _fresh)()
        apply(model)
        report = _revalidate(model)
        rows.append(
            {
                "defect": title,
                "meaning": meaning,
                "failing_checks": [c.name for c in report.failures],
                "details": {c.name: c.detail[:200] for c in report.failures},
            }
        )
    numbers["defects"] = rows


# --------------------------------------------------------------------------- the example failure report and its figure


def _tampered_report(numbers: dict):
    model = _fresh()
    assert not model.report.failures
    topo = model.topology
    vol = _vacuum_volume(model)
    top = _top_face_id(model)
    faces_before = len(vol.shells[0].faces)
    _drop_top_face(model)
    report = _revalidate(model)
    try:
        report.raise_if_failed()
        message = None
    except ValidationError as e:
        message = str(e)
    body = {
        k: v for k, v in report.to_dict().items() if k != "timings_s"
    }  # wall times are not part of any file
    (FIGURES / "validation-tampered-shell-report.json").write_text(
        json.dumps(
            {
                "model": "running example (real input), Route B; outer shell of the vacuum volume without its top face (face "
                + top
                + ")",
                **body,
                "raised": message,
            },
            indent=1,
        )
        + "\n"
    )
    # edges the tampered shell uses once
    uses = Counter()
    for fid, _ in vol.shells[0].faces:
        for lp in topo.faces[fid].loops:
            for eid, _s in lp:
                uses[eid] += 1
    free = [eid for eid, n in uses.items() if n == 1]
    numbers["tampered_shell"] = {
        "removed_face": top,
        "removed_face_group": model.routed.face_group[top],
        "shell_faces_before": faces_before,
        "shell_faces_after": len(vol.shells[0].faces),
        "edges_used_once": len(free),
        "checks_total": len(report.checks),
        "checks_failed": [c.name for c in report.failures],
        "checks_passed": sum(c.passed for c in report.checks),
        "warnings": [c.name for c in report.checks if c.warning],
        "raised_message": message,
        "failing_details": {c.name: c.detail for c in report.failures},
    }
    return model, vol, free


def _figure_shell(model, vol, free) -> None:
    topo = model.topology
    clean = _fresh()  # the written model needs an intact shell: draw the tampered shell's faces from an untouched build
    tris = compact_triangles(clean, mesh_size=60.0, mesh_size_min=4.0)
    ids = {f.id for f in clean.topology.faces.values()}
    assert {
        fid for fid, _ in vol.shells[0].faces
    } <= ids  # content-addressed ids: both builds name the faces alike
    fig = plt.figure(figsize=(9.5, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    polys, colours, traces = [], [], []
    for fid, _ in vol.shells[0].faces:
        pts, tri = tris[fid]
        group = model.routed.face_group[fid]
        colour = GROUP_COLOURS.get(group.split("::")[0], "#999")
        traces.append(
            mesh_trace(pts, tri, colour, group, f"{group}<br>{fid}", opacity=0.35)
        )
        polys.extend(pts[t] for t in tri)
        colours.extend([to_rgba(colour, 0.35)] * len(tri))
    ax.add_collection3d(Poly3DCollection(polys, facecolors=colours, edgecolors="none"))
    lines = []
    for eid in free:
        e = topo.edges[eid]
        a, b = topo.vertices[e.v0], topo.vertices[e.v1]
        lines.append(((a.x * G, a.y * G, float(a.z)), (b.x * G, b.y * G, float(b.z))))
        ax.plot(
            [lines[-1][0][0], lines[-1][1][0]],
            [lines[-1][0][1], lines[-1][1][1]],
            [lines[-1][0][2], lines[-1][1][2]],
            color=PALETTE["bad"],
            lw=3.0,
        )
    ax.set_xlim(-25, 125)
    ax.set_ylim(-25, 125)
    ax.set_zlim(-30, 21)
    ax.set_box_aspect((1, 1, 0.45))
    ax.view_init(elev=24, azim=-56)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_zlabel("z (µm)")
    ax.set_title(
        "Outer shell of the vacuum volume with its top face removed: the four edges now used once (red)",
        fontsize=10,
    )
    ax.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                color=PALETTE["bad"],
                lw=3,
                label=f"{len(free)} edges used once by the shell",
            ),
            Patch(
                facecolor="#bdbdbd", label="the five remaining faces of the shell (EXT)"
            ),
        ],
        loc="upper left",
        fontsize=8,
        frameon=False,
    )
    save_svg(fig, "validation-shell")
    fig2 = go.Figure(traces)
    for a, b in lines:
        fig2.add_trace(
            go.Scatter3d(
                x=[a[0], b[0]],
                y=[a[1], b[1]],
                z=[a[2], b[2]],
                mode="lines",
                line=dict(color=PALETTE["bad"], width=8),
                name="edge used once",
                showlegend=False,
                hoverinfo="name",
            )
        )
    fig2.update_layout(
        title="Outer shell of the vacuum volume without its top face: edges used once (red)",
        width=1000,
        height=720,
        scene=dict(aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.45)),
    )
    save_plotly(fig2, "validation-shell")


# --------------------------------------------------------------------------- errors before validation, check inventory


def _early_errors(numbers: dict) -> None:
    catalogue = real_catalogue()
    out = {}

    def attempt(key, extra, exception, route="B", **kw):
        try:
            build_model(
                replace(catalogue, prisms=(*catalogue.prisms, *extra)), route, **kw
            )
            out[key] = "built"
        except exception as e:
            out[key] = str(e)

    box = footprint(rect_um(35.0, 45.0, 45.0, 55.0))
    attempt(
        "OverlapError (labelling, S5)",
        [
            PrismRecord(
                "other_net",
                box,
                exact_z(0),
                exact_z("0.2"),
                "Al",
                "conductor",
                net_id="other",
            )
        ],
        OverlapError,
    )
    try:
        build_model(catalogue, "A")
    except RouteError as e:
        out["RouteError (route check, before S2)"] = str(e)
    # different-Net contact (shared edge) and a same-Net pinch (finite metal meeting only at a corner)
    attempt(
        "ContactError (contact check, after the arrangements)",
        [
            PrismRecord(
                "touching_net",
                footprint(rect_um(34.0, 45.0, 40.0, 55.0)),
                exact_z(0),
                exact_z("0.2"),
                "Al",
                "conductor",
                net_id="other",
            )
        ],
        ContactError,
    )
    attempt(
        "NonManifoldError (same-Net pinch at a corner, finite metal; before assembly)",
        [
            PrismRecord(
                "pinch",
                footprint(rect_um(60.0, 60.0, 65.0, 65.0)),
                exact_z(0),
                exact_z("0.2"),
                "Al",
                "conductor",
                net_id="P",
            )
        ],
        NonManifoldError,
    )
    try:
        build_model(
            catalogue,
            "B",
            refinements=(RefinementRequest("r", 0.2, ((32.0, 55.0), (38.0, 55.0))),),
        )
    except SgbError as e:
        out["SgbError (refinement curve in a cell with no face, assembly)"] = str(e)
    numbers["early_errors"] = out


def _inventory(numbers: dict) -> None:
    model = _fresh()
    numbers["checks_of_a_valid_build"] = [
        {"name": c.name, "passed": c.passed, "warning": c.warning, "detail": c.detail}
        for c in model.report.checks
    ]


def occt_examples(numbers: dict) -> None:
    """Tier 1: the official OCCT checks (OCP) on the written BREP and XAO of the running examples, one run each.

    The six real-input models are Route A, B and C with and without Surface-EPR bands; the stage-level airbridge and the stage-level
    pad with true arcs (cylinder walls) are the other two.
    Needs OCP (the ``occt`` dependency group of the package); the wall times are one run on this machine.
    """

    real, film = real_catalogue(), real_catalogue(film=True)
    bands_a = (
        InsetRequest("band_pad", ConductorSurface("pad", "sheet"), MARGINS),
        InsetRequest(
            "band_gap",
            SolutionInterface("substrate", "VACUUM_REGION", z_um=0.0),
            MARGINS,
        ),
    )
    bands_b = (
        InsetRequest("band_pad_top", ConductorSurface("pad", "top"), MARGINS),
        InsetRequest("band_pad_bottom", ConductorSurface("pad", "bottom"), MARGINS),
        InsetRequest(
            "band_gap",
            SolutionInterface("substrate", "VACUUM_REGION", z_um=0.0),
            MARGINS,
        ),
    )
    models = {
        "real input, Route A": build_model(film, "A"),
        "real input, Route B": build_model(real, "B"),
        "real input, Route C": build_model(real, "C"),
        "real input, Route A with bands": build_model(film, "A", insets=bands_a),
        "real input, Route B with bands": build_model(real, "B", insets=bands_b),
        "real input, Route C with bands": build_model(real, "C", insets=bands_b),
        "airbridge, Route B (stage-level scene)": build_model(airbridge(), "B"),
        # the only example with true arcs: cylinder walls, where the plan comparison shows its largest error
        "pad_in_ground, Route B (stage-level scene with arcs)": build_model(
            stage_pad_in_ground("B"), "B"
        ),
    }
    reports, rows = {}, []
    for name, model in models.items():
        plan = plan_brep(model.topology, model.routed)
        with tempfile.TemporaryDirectory() as tmp:
            brep, xao = Path(tmp) / "m.brep", Path(tmp) / "m.xao"
            write_brep_file(plan, brep)
            write_xao_file(plan, xao, brep=brep)
            for path in (brep, xao):
                report = check_file(path, model.topology, plan, self_interference=True)
                rows.append(
                    {
                        "model": name,
                        "file": path.suffix[1:],
                        "ok": report.ok,
                        "problems": report.problems,
                        "brepcheck_valid": report.brepcheck["valid"],
                        "brepcheck_faulty": report.brepcheck["faulty_count"],
                        "argument_analyzer_faulty": report.arguments["faulty"],
                        "shells_with_free_edges": report.shape_analysis[
                            "shells_with_free_edges"
                        ],
                        "face_area_max_relative_error": report.properties["faces"][
                            "max_relative_error"
                        ],
                        "solid_volume_max_relative_error": max(
                            (s["relative_error"] for s in report.properties["solids"]),
                            default=0.0,
                        ),
                        "timings_s": {
                            k: round(v, 5) for k, v in report.timings.items()
                        },
                    }
                )
    reports["rows"] = rows
    reports["reports"] = len(rows)
    reports["all_ok"] = all(r["ok"] for r in rows)
    reports["max_face_area_relative_error"] = max(
        r["face_area_max_relative_error"] for r in rows
    )
    reports["max_solid_volume_relative_error"] = max(
        r["solid_volume_max_relative_error"] for r in rows
    )
    timing_keys = sorted({k for r in rows for k in r["timings_s"]})
    reports["timing_ranges_ms"] = {
        k: [
            round(min(r["timings_s"].get(k, 0.0) for r in rows) * 1000, 2),
            round(max(r["timings_s"].get(k, 0.0) for r in rows) * 1000, 2),
        ]
        for k in timing_keys
    }
    numbers["occt_examples"] = reports


def main() -> None:
    """Write the validation figure, the tampered-model report and the numbers of the defect table."""
    numbers: dict = {"scene": "gds_pad_in_ground (production real input), Route B"}
    _run_defects(numbers)
    model, vol, free = _tampered_report(numbers)
    _figure_shell(model, vol, free)
    _early_errors(numbers)
    _inventory(numbers)
    occt_examples(numbers)
    write_numbers("validation", numbers)
    print("validation: done")


if __name__ == "__main__":
    main()
