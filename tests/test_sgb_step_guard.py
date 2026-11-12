"""The durable import guard of the AEDT STEP: face vertices and curvature, file binding, and free-face loops.

The importer reads the guard again before every assignment, so each test checks one thing the importer
relies on against the model the STEP was written from (the sidecar's ``expected`` block and face records),
not against the guard itself.  Skipped where OCP (the ``occt`` dependency group) is not installed.
"""

from __future__ import annotations

import json
import math
import shutil

import pytest

pytest.importorskip("OCP")

from sgb_extreme_cases import (  # noqa: E402
    FIXTURES,
    build_model,
    catalogue_of,
    chip,
    disk,
    port,
)

from scgsim.contracts.sgb_sidecar import views  # noqa: E402
from scgsim.geometry.sgb.emission.plan import plan_brep  # noqa: E402
from scgsim.geometry.sgb.emission.writer import write_brep_file  # noqa: E402
from scgsim.geometry.sgb.exchange.guard import (  # noqa: E402
    GUARD_SCHEMA,
    StepGuardError,
    file_sha256,
    read_step_guard,
    write_step_guard,
)
from scgsim.geometry.sgb.exchange.step import export_aedt_step  # noqa: E402
from scgsim.geometry.sgb.model import build_model_from_catalogue  # noqa: E402
from scgsim.geometry.sgb.records.grid import exact_z  # noqa: E402
from scgsim.geometry.sgb.sidecar.build import SidecarContext, build_sidecar  # noqa: E402
from scgsim.geometry.sgb.sidecar.files import write_sidecar  # noqa: E402
from scgsim.geometry.sgb.slivers.report import sliver_report  # noqa: E402

pytestmark = pytest.mark.occt


def export(model, folder):
    """Write BREP and sidecar of ``model`` into ``folder`` and export the STEP; the sidecar and the export."""
    folder.mkdir(parents=True, exist_ok=True)
    plan = plan_brep(model.topology, model.routed)
    write_brep_file(plan, folder / "m.brep")
    sidecar = build_sidecar(
        model, plan, SidecarContext(sliver_report(model)["summary"])
    )
    write_sidecar(folder / "m.sidecar.json", sidecar)
    return sidecar, export_aedt_step(
        folder / "m.brep", folder / "m.sidecar.json", folder / "m.step"
    )


@pytest.fixture(scope="module")
def banded(tmp_path_factory, emit_models):
    """Route B pad with bands: rounded corners give curved walls, and its port sheet is a free face."""
    folder = tmp_path_factory.mktemp("guard")
    sidecar, exported = export(emit_models["pad_b_bands"], folder)
    return folder, sidecar, exported


def copy_of(sidecar):
    """A deep copy of a sidecar, to change."""
    return json.loads(json.dumps(sidecar))


def planned_vertices(sidecar, face_id):
    """The planned positions of a face's vertices, from the sidecar's load-side view (tags are 1-based)."""
    expected = views.expected(sidecar)
    tag = sidecar["binding"]["ids_by_tag"]["2"].index(face_id)
    vertex_tags = {
        v for e in expected["face_edges"][tag] for v in expected["edge_vertices"][e - 1]
    }
    return sorted(tuple(expected["vertices_um"][v - 1]) for v in vertex_tags)


def test_the_guard_is_written_next_to_the_step_and_reads_back_unchanged(banded):
    """``m.step.guard.json`` sits beside ``m.step``; reading it checks the files and returns what was written."""
    folder, _, exported = banded
    assert exported.guard_path == folder / "m.step.guard.json"
    guard = read_step_guard(exported.guard_path)
    assert guard == exported.guard
    assert guard["schema"] == GUARD_SCHEMA and guard["unit"] == "UM"


def test_the_guard_binds_the_step_the_geometry_and_the_sidecar_by_hash(banded):
    """Each bound file is named relative to the guard and carries the SHA-256 of its bytes."""
    folder, sidecar, exported = banded
    files = exported.guard["files"]
    assert {role: record["path"] for role, record in files.items()} == {
        "step": "m.step",
        "geometry": "m.brep",
        "sidecar": "m.sidecar.json",
    }
    for record in files.values():
        assert record["sha256"] == file_sha256(folder / record["path"])
    assert exported.guard["model"] == {
        "name": sidecar["model"]["name"],
        "identity_digest": sidecar["model"]["identity_digest"],
    }
    assert exported.guard["route"] == sidecar["model"]["route"] == "B"


def test_every_guard_face_lists_the_planned_vertices_of_its_face(banded):
    """The vertices of each face, as written, are the planned vertex positions of that face."""
    _, sidecar, exported = banded
    for solid in exported.solids:
        for face in solid["faces"]:
            assert sorted(map(tuple, face["vertices_um"])) == pytest.approx(
                planned_vertices(sidecar, face["face"]), abs=1e-9
            )


def test_curved_faces_are_flagged_and_keep_the_vertex_centroid_apart(banded):
    """A face is curved exactly when SGB planned it as a curved wall; its two centres are recorded separately."""
    _, sidecar, exported = banded
    curved = set(views.curved_wall_faces(sidecar))
    faces = [face for solid in exported.solids for face in solid["faces"]]
    assert curved and {f["face"] for f in faces if f["curved"]} == curved
    for face in faces:
        mean = [
            sum(v[k] for v in face["vertices_um"]) / len(face["vertices_um"])
            for k in range(3)
        ]
        assert face["vertex_centroid_um"] == pytest.approx(mean, abs=1e-9)
    # on a curved wall the centre of mass lies off the chord through its vertices
    assert any(
        math.dist(f["centre_um"], f["vertex_centroid_um"]) > 1e-3
        for f in faces
        if f["curved"]
    )


@pytest.mark.parametrize("role", ["step", "geometry", "sidecar"])
def test_a_changed_bound_file_is_refused(tmp_path, banded, role):
    """One changed byte in any bound file and the guard no longer reads."""
    folder, _, exported = banded
    copy = tmp_path / "copy"
    shutil.copytree(folder, copy)
    target = copy / exported.guard["files"][role]["path"]
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(StepGuardError, match=f"the {role} file .* changed"):
        read_step_guard(copy / "m.step.guard.json")


def test_a_missing_bound_file_is_refused(tmp_path, banded):
    """A guard without its STEP is refused, naming the missing file."""
    folder, _, _ = banded
    copy = tmp_path / "copy"
    shutil.copytree(folder, copy)
    (copy / "m.step").unlink()
    with pytest.raises(StepGuardError, match="the step file m.step is missing"):
        read_step_guard(copy / "m.step.guard.json")


def test_a_sidecar_of_another_model_is_refused_even_with_its_own_hash(tmp_path, banded):
    """Re-hashing a swapped sidecar does not help: its model identity is not the guarded one."""
    folder, sidecar, _ = banded
    copy = tmp_path / "copy"
    shutil.copytree(folder, copy)
    other = {
        **sidecar,
        "model": {**sidecar["model"], "identity_digest": "model:" + "0" * 16},
    }
    write_sidecar(copy / "m.sidecar.json", other)
    guard = json.loads((copy / "m.step.guard.json").read_text())
    guard["files"]["sidecar"]["sha256"] = file_sha256(copy / "m.sidecar.json")
    write_step_guard(copy / "m.step.guard.json", guard)
    with pytest.raises(StepGuardError, match="is not the guarded"):
        read_step_guard(copy / "m.step.guard.json")


@pytest.mark.parametrize(
    "change",
    [{"schema": "other"}, {"unit": "MM"}],
    ids=["schema", "unit"],
)
def test_a_guard_of_another_schema_or_unit_is_refused(tmp_path, banded, change):
    """The reader accepts only the schema and the micrometre unit it was written with."""
    folder, _, _ = banded
    guard = {**json.loads((folder / "m.step.guard.json").read_text()), **change}
    write_step_guard(tmp_path / "m.step.guard.json", guard)
    with pytest.raises(StepGuardError):
        read_step_guard(tmp_path / "m.step.guard.json")


def test_a_stp_and_a_step_of_one_stem_keep_separate_guards(tmp_path, emit_models):
    """The guard keeps the STEP's full file name, so ``m.step`` and ``m.stp`` in one folder do not share one."""
    _, first = export(emit_models["pad_b_bands"], tmp_path)
    second = export_aedt_step(
        tmp_path / "m.brep", tmp_path / "m.sidecar.json", tmp_path / "m.stp"
    )
    assert second.guard_path == tmp_path / "m.stp.guard.json"
    assert read_step_guard(first.guard_path)["files"]["step"]["path"] == "m.step"
    assert read_step_guard(second.guard_path)["files"]["step"]["path"] == "m.stp"


def edit_guard(tmp_path, banded, change):
    """A copy of the example's files whose guard ``change`` edited in place; the file hashes stay valid."""
    folder, _, _ = banded
    copy = tmp_path / "copy"
    shutil.copytree(folder, copy)
    guard = json.loads((copy / "m.step.guard.json").read_text())
    change(guard)
    write_step_guard(copy / "m.step.guard.json", guard)
    return copy / "m.step.guard.json"


def regroup_free_face(guard):
    """Give the free port sheet another port's group."""
    guard["free_faces"][0]["group"] = "PORT::other"


def renumber_face(guard):
    """Give a face of the first solid the planned id of a face of the last solid."""
    faces = guard["solids"][0]["faces"]
    faces[0]["face"] = guard["solids"][-1]["faces"][-1]["face"]


def rename_solid(guard):
    """Give the first solid a name the sidecar does not give it."""
    guard["solids"][0]["aedt_name"] = guard["solids"][0]["aedt_name"] + "_x"


def drop_face(guard):
    """Leave the last face out of the first solid."""
    del guard["solids"][0]["faces"][-1]


def drop_free_face(guard):
    """Leave the free port sheet out."""
    del guard["free_faces"][0]


def flip_curved(guard):
    """Flip the curved flag of one face."""
    face = guard["solids"][0]["faces"][0]
    face["curved"] = not face["curved"]


def swap_solid_ids(guard):
    """Swap the planned ids of the first two solids, keeping their faces."""
    first, second = guard["solids"][0], guard["solids"][1]
    first["solid"], second["solid"] = second["solid"], first["solid"]


def swap_planar_face_ids(guard):
    """Swap the planned ids of two planar faces of one solid, keeping every measurement in place."""
    for solid in guard["solids"]:
        planar = [f for f in solid["faces"] if not f["curved"]]
        if len(planar) >= 2:
            planar[0]["face"], planar[1]["face"] = planar[1]["face"], planar[0]["face"]
            return
    raise AssertionError("the example has no solid with two planar faces")


def disconnect_loop(guard):
    """Move the end of one free-face loop edge, keeping every start point (the vertices) and the area."""
    guard["free_faces"][0]["loops"][0][0]["points_um"][-1][0] += 1.0


def move_vertex_centroid(guard):
    """Move one face's vertex centroid off the mean of its vertices."""
    guard["solids"][0]["faces"][0]["vertex_centroid_um"][0] += 1.0


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (regroup_free_face, "free face .* has group 'PORT::other'"),
        (renumber_face, "lists faces .* the sidecar plans"),
        (rename_solid, "is named .* the sidecar names it"),
        (drop_face, "lists faces .* the sidecar plans"),
        (drop_free_face, "the free faces are"),
        (flip_curved, "flags curved wrongly"),
        (swap_solid_ids, "not distinct planned solids in tag order"),
        (swap_planar_face_ids, "the recorded vertices are not those of planned face"),
        (move_vertex_centroid, "the vertex centroid is not the mean"),
        (disconnect_loop, "loop 0 edge 0 ends at .* the next edge starts at"),
    ],
    ids=lambda v: getattr(v, "__name__", None),
)
def test_a_guard_edited_against_its_sidecar_is_refused(
    tmp_path, banded, change, message
):
    """Each edit leaves every file hash valid; the cross-check with the bound sidecar still refuses it."""
    with pytest.raises(
        StepGuardError, match=f"does not match its sidecar: .*{message}"
    ):
        read_step_guard(edit_guard(tmp_path, banded, change))


def drop_key(guard):
    """Remove the vertices of one face."""
    del guard["solids"][0]["faces"][0]["vertices_um"]


def add_key(guard):
    """Add a key the schema does not have to a free face."""
    guard["free_faces"][0]["note"] = "extra"


def bad_point(guard):
    """Give one face a two-coordinate centre."""
    guard["solids"][0]["faces"][0]["centre_um"] = [0.0, 0.0]


def bad_arc(guard):
    """Replace a loop edge by an arc with two points."""
    guard["free_faces"][0]["loops"][0][0] = {
        "kind": "arc",
        "points_um": [[0.0, 0.0, 0.0]] * 2,
    }


def no_solids(guard):
    """List no solid."""
    guard["solids"] = []


@pytest.mark.parametrize(
    "change",
    [drop_key, add_key, bad_point, bad_arc, no_solids],
    ids=lambda v: v.__name__,
)
def test_a_malformed_guard_is_refused(tmp_path, banded, change):
    """A missing or extra key, or a value of the wrong shape, is refused before any file is read."""
    with pytest.raises(StepGuardError, match="is malformed"):
        read_step_guard(edit_guard(tmp_path, banded, change))


def test_swapped_ids_of_two_free_faces_are_refused(tmp_path):
    """Two free faces of one model swap ids, loops and areas unchanged: each record no longer fits its id.

    The Route C port across a round pad leaves two port pieces off the support plane, both free.
    """
    folder = tmp_path / "model"
    export(build_model(FIXTURES["port_crossing_round_pad"], "C"), folder)
    guard = json.loads((folder / "m.step.guard.json").read_text())
    first, second = guard["free_faces"][:2]
    # the ids stay in tag order (a plain swap breaks that and is refused for it); the records swap under them
    first["face"], second["face"] = second["face"], first["face"]
    guard["free_faces"][:2] = [second, first]
    write_step_guard(folder / "m.step.guard.json", guard)
    with pytest.raises(
        StepGuardError, match="free face .*: the recorded vertices are not those"
    ):
        read_step_guard(folder / "m.step.guard.json")


def diameter_split_round_pad():
    """Route C round pad whose top a port sheet covers up to a diameter: two mirror-image half-disks in one solid."""
    lumped = port(
        "p1", [(-6.0, 0.0), (6.0, 0.0), (6.0, 6.0), (-6.0, 6.0)], exact_z("0.2")
    )
    pad = disk("disk", "N", (0, 0), 5000, "C")
    catalogue = catalogue_of("diameter_split_round_pad", "C", [*chip(), pad, lumped])
    return build_model_from_catalogue(catalogue, "C")


def test_the_halves_of_a_round_pad_split_along_a_diameter_keep_their_ids(tmp_path):
    """Two congruent half-disks of one solid, equal in area and sharing the diameter's ends, are told apart.

    The disk is four quarter arcs, so each half also has the quarter point on its own side as a vertex: the
    export reads back, and swapping the halves' ids (measurements in place) is refused.
    """
    sidecar, exported = export(diameter_split_round_pad(), tmp_path)
    assert read_step_guard(exported.guard_path) == exported.guard
    (index, pad), *_ = [
        (i, s)
        for i, s in enumerate(exported.solids)
        if sidecar["volume"][s["solid"]]["kind"] == "conductor"
    ]
    halves = [
        f
        for f in pad["faces"]
        if sidecar["face"][f["face"]]["face_kind"] in ("top", "port")
    ]
    assert len(halves) == 2
    assert halves[0]["area_um2"] == pytest.approx(halves[1]["area_um2"], rel=1e-9)
    guard = json.loads(exported.guard_path.read_text())
    first, second = (
        f
        for f in guard["solids"][index]["faces"]
        if f["face"] in {h["face"] for h in halves}
    )
    first["face"], second["face"] = second["face"], first["face"]
    write_step_guard(exported.guard_path, guard)
    with pytest.raises(
        StepGuardError, match="the recorded vertices are not those of planned face"
    ):
        read_step_guard(exported.guard_path)


def test_faces_the_sidecar_cannot_tell_apart_are_refused(tmp_path, banded):
    """If two faces of one solid had the same planned vertices and area, their ids could not be bound."""
    folder, sidecar, exported = banded
    copy = tmp_path / "copy"
    shutil.copytree(folder, copy)
    a, b = (f["face"] for f in exported.solids[0]["faces"][:2])
    twin = copy_of(sidecar)
    twin["face"][b]["loops"] = twin["face"][a]["loops"]
    twin["face"][b]["area_um2"] = twin["face"][a]["area_um2"]
    write_sidecar(copy / "m.sidecar.json", twin)
    guard = json.loads((copy / "m.step.guard.json").read_text())
    guard["files"]["sidecar"]["sha256"] = file_sha256(copy / "m.sidecar.json")
    write_step_guard(copy / "m.step.guard.json", guard)
    with pytest.raises(StepGuardError, match="cannot bind their ids"):
        read_step_guard(copy / "m.step.guard.json")


def loop_points(loop, samples=256):
    """The loop as a dense polygon: line ends, and each arc sampled on the circle through its three points."""
    points = []
    for segment in loop:
        if segment["kind"] == "line":
            points.append(segment["points_um"][0][:2])
        else:
            points.extend(arc_points(*segment["points_um"], samples)[:-1])
    return points


def arc_points(start, mid, end, samples):
    """Points from ``start`` through ``mid`` to ``end`` on the circle through the three (in the xy plane)."""
    (ax, ay), (bx, by), (cx, cy) = start[:2], mid[:2], end[:2]
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = (
        (ax**2 + ay**2) * (by - cy)
        + (bx**2 + by**2) * (cy - ay)
        + (cx**2 + cy**2) * (ay - by)
    ) / d
    uy = (
        (ax**2 + ay**2) * (cx - bx)
        + (bx**2 + by**2) * (ax - cx)
        + (cx**2 + cy**2) * (bx - ax)
    ) / d
    radius = math.dist((ux, uy), (ax, ay))
    t0, tm, t1 = (math.atan2(y - uy, x - ux) for x, y in ((ax, ay), (bx, by), (cx, cy)))
    sweep = (t1 - t0) % (2 * math.pi)
    if (tm - t0) % (2 * math.pi) > sweep:  # the arc runs clockwise through mid
        sweep -= 2 * math.pi
    return [
        (
            ux + radius * math.cos(t0 + sweep * i / samples),
            uy + radius * math.sin(t0 + sweep * i / samples),
        )
        for i in range(samples + 1)
    ]


def shoelace(points):
    """The signed area of a closed polygon."""
    return 0.5 * sum(
        x0 * y1 - x1 * y0
        for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1], strict=True)
    )


def check_loops(face):
    """Each loop is closed and connected, and outer minus holes encloses the face's area."""
    for loop in face["loops"]:
        for here, after in zip(loop, loop[1:] + loop[:1], strict=True):
            assert here["points_um"][-1] == pytest.approx(
                after["points_um"][0], abs=1e-9
            )
    areas = [abs(shoelace(loop_points(loop))) for loop in face["loops"]]
    assert areas[0] - sum(areas[1:]) == pytest.approx(face["area_um2"], rel=1e-6)


def test_a_free_port_sheet_is_recorded_with_its_loop_and_plane(banded):
    """The port sheet of the example bounds no solid: the guard keeps its group, normal and closed loop."""
    _, sidecar, exported = banded
    (free,) = exported.guard["free_faces"]
    assert free["group"] == "PORT::p1"
    (port,) = sidecar["port"].values()
    assert free["face"] in port["faces"]
    assert free["normal"] == [0.0, 0.0, 1.0]
    assert all(s["kind"] == "line" for loop in free["loops"] for s in loop)
    check_loops(free)
    assert free["face"] not in {
        f["face"] for solid in exported.solids for f in solid["faces"]
    }


def test_a_free_face_with_an_arc_keeps_the_arc_as_three_points(tmp_path):
    """A port across a round pad has an arc edge; the arc's three points rebuild the face's area."""
    _, exported = export(
        build_model(FIXTURES["port_crossing_round_pad"], "B"), tmp_path
    )
    (free,) = exported.guard["free_faces"]
    assert read_step_guard(exported.guard_path) == exported.guard
    kinds = [s["kind"] for loop in free["loops"] for s in loop]
    assert "arc" in kinds
    assert all(
        len(s["points_um"]) == (3 if s["kind"] == "arc" else 2)
        for loop in free["loops"]
        for s in loop
    )
    check_loops(free)


def test_a_route_a_floating_sheet_is_a_free_face(tmp_path):
    """A Route A sheet with the same volume on both sides bounds no solid and is kept for rebuilding."""
    sidecar, exported = export(
        build_model(FIXTURES["floating_conductor"], "A"), tmp_path
    )
    assert read_step_guard(exported.guard_path) == exported.guard
    embedded = {face for face, _ in views.embedded_faces(sidecar)}
    assert {f["face"] for f in exported.guard["free_faces"]} == embedded
    for free in exported.guard["free_faces"]:
        assert free["group"] == sidecar["face"][free["face"]]["group"]
        check_loops(free)
