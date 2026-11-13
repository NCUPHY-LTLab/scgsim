"""Direct BREP / XAO serialisation: format, sharing, orientation, determinism and the round trip through Gmsh."""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import gmsh
import numpy as np
import pytest
from sgb_emit_cases import CASES, MESH_SIZE, support_regions
from sgb_gmsh_oracle import (
    band_mesh_report,
    check_mesh_conformity,
    emit_with_gmsh_api,
    gmsh_session,
    mesh_coarsely,
    mesh_quality,
    mesh_with_cheap_recipe,
    refine_near,
)
from sgb_scenes import random_layout

from scgsim.contracts.sgb_sidecar import read_sidecar, views
from scgsim.geometry.sgb.build import emit_model
from scgsim.geometry.sgb.output.emission.numbers import angle_of
from scgsim.geometry.sgb.output.emission.plan import check_one_tag_per_id, plan_brep
from scgsim.geometry.sgb.output.emission.writer import write_brep_file, write_xao_file
from scgsim.geometry.sgb.errors import SgbError
from scgsim.geometry.sgb.model import build_model_from_catalogue
from scgsim.geometry.sgb.output.sidecar.build import SidecarContext, build_sidecar
from scgsim.geometry.sgb.checks.slivers.report import sliver_report
from scgsim.geometry.sgb.checks.verification.loaded import (
    apply_sidecar_embeds,
    verify_loaded,
    verify_xao,
)


def sidecar_of(model, plan):
    """The sidecar of a model and its plan."""
    return build_sidecar(model, plan, SidecarContext(sliver_report(model)["summary"]))


def written(model, tmp_path: Path, suffix: str = "xao", **options):
    """Plan and write a model; returns the plan and the path."""
    plan = plan_brep(model.topology, model.routed)
    path = tmp_path / f"m.{suffix}"
    (write_xao_file if suffix == "xao" else write_brep_file)(plan, path, **options)
    return plan, path


# --------------------------------------------------------------------------- number policy


def test_the_angle_without_libm_agrees_with_atan2_to_a_few_ulps_in_every_quadrant():
    """``angle_of`` is built from basic operations only, yet matches the C library to 1e-15."""
    rng = np.random.default_rng(7)
    theta = rng.uniform(-math.pi, math.pi, 20000)
    radius = rng.uniform(0.1, 1e4, 20000)
    got = angle_of(radius * np.sin(theta), radius * np.cos(theta))
    want = np.arctan2(radius * np.sin(theta), radius * np.cos(theta))
    assert np.abs(got - want).max() < 1e-15
    axes = angle_of(np.array([0.0, 1.0, 0.0, -1.0]), np.array([1.0, 0.0, -1.0, 0.0]))
    assert list(axes) == [0.0, math.pi / 2, math.pi, -math.pi / 2]


# --------------------------------------------------------------------------- format


def test_file_layout_matches_the_documented_sections(tmp_path, emit_models):
    """File layout matches the documented sections."""
    plan, path = written(emit_models["pad_a"], tmp_path, "brep")
    lines = path.read_text().splitlines()
    assert lines[:3] == [
        "DBRep_DrawableShape",
        "",
        "CASCADE Topology V1, (c) Matra-Datavision",
    ]
    wanted = {
        "Locations",
        "Curve2ds",
        "Curves",
        "Polygon3D",
        "PolygonOnTriangulations",
        "Surfaces",
        "Triangulations",
        "TShapes",
    }
    assert [ln.split()[0] for ln in lines if ln.split() and ln.split()[0] in wanted] == [
        "Locations", "Curve2ds", "Curves", "Polygon3D", "PolygonOnTriangulations", "Surfaces", "Triangulations", "TShapes"
    ]  # fmt: skip
    c = plan.counts
    assert (
        f"TShapes {1 + c['vertices'] + c['edges'] + c['wires'] + c['faces'] + c['shells'] + c['solids']}"
        in lines
    )
    kinds = [ln for ln in lines if ln in {"Ve", "Ed", "Wi", "Fa", "Sh", "So", "Co"}]
    assert [kinds.count(k) for k in ("Ve", "Ed", "Wi", "Fa", "Sh", "So", "Co")] == [
        c["vertices"], c["edges"], c["wires"], c["faces"], c["shells"], c["solids"], 1
    ]  # fmt: skip
    assert lines[-1] == "+1 0 "  # the root: the compound


def test_tolerance_is_written_on_every_vertex_edge_and_face(tmp_path, emit_models):
    """Tolerance is written on every vertex edge and face."""
    model = emit_models["pad_a"]
    plan = plan_brep(model.topology, model.routed, 2.5e-7)
    write_brep_file(plan, tmp_path / "m.brep")
    text = (tmp_path / "m.brep").read_text()
    assert text.count("\n2.5e-07\n") == plan.counts["vertices"]
    assert " 2.5e-07 1 1 0\n" in text and "0  2.5e-07 " in text


def test_the_brep_inside_the_xao_is_the_brep_file_whether_copied_or_rendered(
    tmp_path, emit_models
):
    """Copying the body of the BREP file into the XAO saves a second rendering and changes no byte."""
    plan, brep = written(emit_models["pad_b"], tmp_path, "brep")
    write_xao_file(plan, tmp_path / "copied.xao", name="m", brep=brep)
    write_xao_file(plan, tmp_path / "rendered.xao", name="m")
    assert (tmp_path / "copied.xao").read_bytes() == (
        tmp_path / "rendered.xao"
    ).read_bytes()
    inner = ET.parse(tmp_path / "copied.xao").getroot().find("geometry/shape").text
    assert inner.strip("\n") == brep.read_text().removeprefix(
        "DBRep_DrawableShape\n\n"
    ).strip("\n")


# --------------------------------------------------------------------------- sharing and tag prediction


@pytest.mark.parametrize("name", ["pad_a", "pad_b", "pad_c", "bridge_b"])
def test_shared_faces_are_written_once_and_used_with_both_orientations(
    emit_models, name
):
    """Shared faces are written once and used with both orientations."""
    model = emit_models[name]
    plan = plan_brep(model.topology, model.routed)
    uses: dict[int, int] = {}
    for shell in plan.found.shells:
        for ref in shell:
            uses[abs(ref)] = uses.get(abs(ref), 0) + 1
    interface = [f for f, n in uses.items() if n == 2]
    assert interface, "the emitted volumes share interface faces"
    for f in interface:  # an interface face is referenced once with each orientation
        assert sorted(
            ref for shell in plan.found.shells for ref in shell if abs(ref) == f
        ) == [-f, f]


@pytest.mark.parametrize("name", sorted(CASES))
def test_round_trip_counts_boundaries_groups_volumes_areas_and_a_conformal_mesh(
    tmp_path, emit_models, name
):
    """Open the XAO in Gmsh: verify_loaded passes on the sidecar alone, and a coarse mesh is conformal."""
    model = emit_models[name]
    plan, path = written(model, tmp_path, "xao")
    sidecar = sidecar_of(model, plan)
    with gmsh_session(path):
        report = verify_loaded(sidecar)
        assert report.ok, report.problems
        assert report.max_vertex_offset_um < 1e-9
        assert report.boundaries_mismatching == {"edges": 0, "faces": 0, "solids": 0}
        embedded = apply_sidecar_embeds(sidecar)
        assert embedded["faces"] == sum(
            len(v.embedded_faces)
            for v in model.topology.volumes.values()
            if v.id in model.routed.emitted_volumes
        )
        mesh_coarsely(MESH_SIZE[name])
        conformity = check_mesh_conformity(model, plan.tags())
    assert conformity["conformal"], conformity["problems"][:3]


def test_the_xao_is_well_formed_and_lists_the_planned_groups(tmp_path, emit_models):
    """The xao is well formed and lists the planned groups."""
    model = emit_models["pad_b"]
    plan, path = written(model, tmp_path, "xao")
    root = ET.parse(path).getroot()
    assert {g.get("name") for g in root.find("groups")} == set(model.routed.groups)
    assert int(root.find("groups").get("count")) == len(model.routed.groups)
    assert int(root.find("geometry/topology/faces").get("count")) == len(plan.face_ids)
    assert root.find("geometry/shape").get("format") == "BREP"
    refs = [int(f.get("reference")) for f in root.find("geometry/topology/faces")]
    assert refs == sorted(set(refs))  # strictly increasing pre-order indices


def test_every_group_element_is_a_planned_entity_of_the_right_dimension(emit_models):
    """Every group element is a planned entity of the right dimension."""
    model = emit_models["pad_b"]
    plan = plan_brep(model.topology, model.routed)
    for group in plan.groups:
        ids = plan.face_ids if group.dim == 2 else plan.solid_ids
        assert {ids[i] for i in group.elements} <= set(
            model.routed.groups[group.name].ids
        )
        assert len(group.elements) == len(set(group.elements))


# --------------------------------------------------------------------------- orientation and exact curved walls


def test_reversed_shells_give_negative_volume_so_the_check_sees_orientation(
    tmp_path, emit_models
):
    """Reversed shells give negative volume so the check sees orientation."""
    model = emit_models["pad_a"]
    plan = plan_brep(model.topology, model.routed)
    plan.found.shells = [[-ref for ref in shell] for shell in plan.found.shells]
    write_brep_file(plan, tmp_path / "bad.brep")
    with gmsh_session(tmp_path / "bad.brep"):
        assert all(
            gmsh.model.occ.getMass(3, t) < 0 for _, t in gmsh.model.getEntities(3)
        )
    report_problems = _verify(plan, model, tmp_path / "bad.brep")
    assert any("non-positive" in p for p in report_problems)


def _verify(plan, model, path):
    with gmsh_session(path):
        return verify_loaded(sidecar_of(model, plan), groups=False).problems


def test_pad_b_volumes_equal_the_closed_form_including_cylinder_walls(
    tmp_path, emit_models
):
    """The substrate and the vacuum volume agree with an analytic area to 1e-12: the walls are exact cylinders."""
    model = emit_models["pad_b"]
    plan, path = written(model, tmp_path, "brep")
    polygon = [
        (-45, -30),
        (-10, -30),
        (-10, -15),
        (10, -15),
        (10, -30),
        (50, -30),
        (50, 18),
        (38, 30),
        (-50, 30),
        (-50, -25),
    ]
    area = (
        sum(
            polygon[i][0] * polygon[(i + 1) % 10][1]
            - polygon[(i + 1) % 10][0] * polygon[i][1]
            for i in range(10)
        )
        / 2
    )
    area += 25 * (math.pi / 2 - 1) / 2 - math.pi * 64
    expected = {"substrate": 9e6, "vacuum": 9e6 - (90000 - 19200) * 0.2 - area * 0.2}
    assert plan.counts["cylinders"] == 5 and plan.counts["arcs"] == 10
    with gmsh_session(path):
        for i, vid in enumerate(plan.solid_ids):
            mass = gmsh.model.occ.getMass(3, i + 1)
            assert mass == pytest.approx(
                expected[model.topology.volumes[vid].domain.name], rel=1e-12
            )


def test_every_curved_wall_is_one_cylinder_written_exactly(emit_models):
    """Every curved wall is one cylinder written exactly."""
    for name in ("pad_b", "pad_c", "bridge_b"):
        model = emit_models[name]
        plan = plan_brep(model.topology, model.routed)
        curved = [
            f
            for f in model.topology.faces.values()
            if f.curved and f.id in set(plan.face_ids)
        ]
        assert plan.counts["cylinders"] == len(curved)


# --------------------------------------------------------------------------- pcurves, parsed back from the text


def _numbers(line: str) -> list[float]:
    return [float(x) for x in line.split()]


def _sections(path: Path) -> dict[str, list[str]]:
    lines = path.read_text().splitlines()
    out: dict[str, list[str]] = {}
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if parts and parts[0] in {"Curve2ds", "Curves", "Surfaces"}:
            n = int(parts[1])
            out[parts[0]] = lines[i + 1 : i + 1 + n]
            i += n
        elif parts and parts[0] == "TShapes":
            out["TShapes"] = lines[i + 1 :]
            break
        i += 1
    return out


def _curve(rec: list[float], t: float) -> np.ndarray:
    if rec[0] == 1:
        return np.array(rec[1:4]) + t * np.array(rec[4:7])
    c, x, y, r = np.array(rec[1:4]), np.array(rec[7:10]), np.array(rec[10:13]), rec[13]
    return c + r * (math.cos(t) * x + math.sin(t) * y)


def _surface(rec: list[float], u: float, v: float) -> np.ndarray:
    o, n, x, y = (np.array(rec[1 + 3 * k : 4 + 3 * k]) for k in range(4))
    if rec[0] == 1:
        return o + u * x + v * y
    return o + rec[13] * (math.cos(u) * x + math.sin(u) * y) + v * n


def _pcurve(rec: list[float], t: float) -> tuple[float, float]:
    if rec[0] == 1:
        return rec[1] + t * rec[3], rec[2] + t * rec[4]
    c, x, y, r = np.array(rec[1:3]), np.array(rec[3:5]), np.array(rec[5:7]), rec[7]
    p = c + r * (math.cos(t) * x + math.sin(t) * y)
    return p[0], p[1]


@pytest.mark.parametrize("name", ["pad_b", "bridge_b", "pad_c"])
def test_every_pcurve_lifts_onto_its_3d_curve(tmp_path, emit_models, name):
    """Parse the written file: for each edge representation, surface(pcurve(t)) == curve(t) at both ends and the middle."""
    _, path = written(emit_models[name], tmp_path, "brep")
    sections = _sections(path)
    curves = [_numbers(x) for x in sections["Curves"]]
    pcurves = [_numbers(x) for x in sections["Curve2ds"]]
    surfaces = [_numbers(x) for x in sections["Surfaces"]]
    tshapes, checked, worst = sections["TShapes"], 0, 0.0
    for i, line in enumerate(tshapes):
        if line != "Ed":
            continue
        j, reps = i + 2, []
        while tshapes[j] != "0":
            reps.append(tshapes[j].split())
            j += 1
        curve_index, last = int(reps[0][1]), float(reps[0][4])
        for rep in reps[1:]:
            pc, surf = pcurves[int(rep[1]) - 1], surfaces[int(rep[2]) - 1]
            assert float(rep[5]) == last
            for t in (0.0, last / 2, last):
                u, v = _pcurve(pc, t)
                worst = max(
                    worst,
                    float(
                        np.linalg.norm(
                            _surface(surf, u, v) - _curve(curves[curve_index - 1], t)
                        )
                    ),
                )
            checked += 1
    assert checked > 0 and worst < 1e-9


# --------------------------------------------------------------------------- determinism


def test_output_is_byte_identical_for_every_chunk_size(tmp_path, emit_models):
    """Many tiny chunks and one large one; the bytes never change."""
    model = emit_models["pad_b"]
    plan = plan_brep(model.topology, model.routed)
    blobs = set()
    for chunk in (40_000, 7, 3):
        base = tmp_path / f"c{chunk}"
        write_brep_file(plan, base.with_suffix(".brep"), chunk_items=chunk)
        write_xao_file(plan, base.with_suffix(".xao"), name="m", chunk_items=chunk)
        blobs.add(
            (
                base.with_suffix(".brep").read_bytes(),
                base.with_suffix(".xao").read_bytes(),
            )
        )
    assert len(blobs) == 1


# Digests of the written bytes: any change in the format, the numbering or the number policy shows up here, and
# a platform whose arithmetic differs would too.  Regenerate with the snippet in the failure message.
PINNED_DIGESTS = {
    "pad_a": "2143195455a977d510a9e0583b275c0e07fc95232893c94864d10f74a69e6ce3",
    "pad_b": "a785ef6c04ab5eddbb3c56647368fbc0e831f6fe15aaf6d63d60aca7b9a5f1fa",
}


@pytest.mark.parametrize("name", sorted(PINNED_DIGESTS))
def test_the_written_bytes_are_pinned_by_a_digest(tmp_path, emit_models, name):
    """The written bytes are pinned by a digest."""
    _, path = written(emit_models[name], tmp_path, "brep")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == PINNED_DIGESTS[name], f"{name}: sha256 of the BREP is now {digest}"


# --------------------------------------------------------------------------- refusals


def test_a_curved_face_that_is_not_a_cylinder_is_refused_not_written_wrongly():
    """A curved face that is not a cylinder is refused not written wrongly."""
    model = CASES["pad_a"]()  # its own model: the test damages it
    face = next(iter(model.topology.faces.values()))
    face.planar = False
    with pytest.raises(SgbError, match="non-planar"):
        plan_brep(model.topology, model.routed)


def test_an_edge_that_leaves_its_plane_is_refused():
    """An edge that leaves its plane is refused."""
    model = CASES["pad_b"]()
    topology = model.topology
    vertical = next(e for e in topology.edges.values() if e.kind == "vertical")
    topology.vertices[vertical.v1].x += 5  # drag one end of a wall column sideways
    with pytest.raises(SgbError, match="leaves its surface"):
        plan_brep(topology, model.routed)


def test_a_planned_id_numbered_twice_is_refused_in_every_dimension(emit_models):
    """Every planned id maps to exactly one native tag, so a repeated id is an error, not a quiet merge."""
    plan = plan_brep(
        emit_models["pad_b_bands"].topology, emit_models["pad_b_bands"].routed
    )
    check_one_tag_per_id(plan.found)
    assert len(plan.tags()) == sum(
        len(ids)
        for ids in (plan.vertex_ids, plan.edge_ids, plan.face_ids, plan.solid_ids)
    )
    for ids in (plan.vertex_ids, plan.edge_ids, plan.face_ids, plan.solid_ids):
        ids.append(ids[0])
        with pytest.raises(SgbError, match="more than one native tag"):
            check_one_tag_per_id(plan.found)
        ids.pop()


# --------------------------------------------------------------------------- paths the examples do not all exercise


def test_inner_void_shell_is_a_second_shell_with_cavity_orientation(
    tmp_path, emit_models
):
    """Inner void shell is a second shell with cavity orientation."""
    model = emit_models["void_shell_b"]
    plan, path = written(model, tmp_path, "brep")
    vacuum = next(
        i
        for i, vid in enumerate(plan.solid_ids)
        if model.topology.volumes[vid].domain.name == "vacuum"
    )
    assert plan.found.solid_shell_count[vacuum] == 2
    assert [s.role for s in model.topology.volumes[plan.solid_ids[vacuum]].shells] == [
        "outer",
        "inner",
    ]
    with gmsh_session(path):
        mass = gmsh.model.occ.getMass(3, vacuum + 1)
        assert verify_loaded(sidecar_of(model, plan), groups=False).ok
    # the chip is 40 x 40 x 10 um above z = 0; the floating conductor is 12 x 8 um and 0.2 um thick
    assert mass == pytest.approx(40 * 40 * 10 - 12 * 8 * 0.2, rel=1e-12)


def test_a_void_shell_with_the_outward_orientation_is_visibly_wrong(
    tmp_path, emit_models
):
    """Negative control: flipping only the inner shell adds the cavity volume instead of removing it."""
    model = emit_models["void_shell_b"]
    plan = plan_brep(model.topology, model.routed)
    vacuum = next(
        i
        for i, vid in enumerate(plan.solid_ids)
        if model.topology.volumes[vid].domain.name == "vacuum"
    )
    inner = plan.found.solid_first_shell[vacuum] + 1
    plan.found.shells[inner] = [-ref for ref in plan.found.shells[inner]]
    write_brep_file(plan, tmp_path / "bad.brep")
    with gmsh_session(tmp_path / "bad.brep"):
        assert gmsh.model.occ.getMass(3, vacuum + 1) == pytest.approx(
            40 * 40 * 10 + 12 * 8 * 0.2, rel=1e-12
        )
    assert any("volume" in p for p in _verify(plan, model, tmp_path / "bad.brep"))


def test_a_pinched_hole_is_written_as_two_wires_that_share_the_pinch_vertex(tmp_path):
    """A pinched hole is written as two wires that share the pinch vertex."""
    from sgb_extreme_cases import FIXTURES, build_model  # noqa: PLC0415 -- used by this test only

    model = build_model(FIXTURES["kissing_corners_same_net"], "A")
    pinched = [
        f
        for f in model.topology.faces.values()
        if len(f.loops) == 2 and len(f.loops[1]) == 8
    ]
    assert (
        len(pinched) == 1
    )  # the hole boundary of two squares touching at a corner: one cycle, eight edges
    plan, path = written(model, tmp_path, "xao")
    assert max(plan.found.face_wire_count) == 3  # outer wire + one wire per lobe
    with gmsh_session(path):
        assert verify_loaded(sidecar_of(model, plan)).ok


def test_refinement_only_curve_is_a_free_edge_and_is_reembedded_from_the_sidecar(
    tmp_path, emit_models
):
    """Refinement only curve is a free edge and is reembedded from the sidecar."""
    model = emit_models["pad_a_refinement"]
    host = [f for f in model.topology.faces.values() if f.refinement_edges]
    assert len(host) == 1 and len(host[0].refinement_edges) == 1
    plan, path = written(model, tmp_path, "xao")
    sidecar = sidecar_of(model, plan)
    assert plan.counts["free_edges"] == 1
    edge_tag = plan.tags()[host[0].refinement_edges[0]][1]
    with gmsh_session(path):
        assert verify_loaded(sidecar).ok
        assert (
            gmsh.model.getAdjacencies(1, edge_tag)[0].size == 0
        )  # free: no face owns it in the file
        assert apply_sidecar_embeds(sidecar) == {"faces": 0, "edges": 1}
        mesh_coarsely(40.0)
        assert (
            gmsh.model.mesh.getElements(1, edge_tag)[1][0].size > 1
        )  # meshed, embedded in the host face
    edge = host[0].refinement_edges[0]
    assert views.embedded_edges(sidecar) == [(edge, host[0].id)]
    assert sidecar["edge"][edge]["written_as_free_edge"]
    assert {
        sidecar["carrier"][c]["request"] for c in sidecar["edge"][edge]["carriers"]
    } == {"ref"}


# --------------------------------------------------------------------------- inset bands through the direct writer


@pytest.mark.parametrize("name", ["pad_a_bands", "pad_b_bands", "pad_c_bands"])
def test_band_faces_through_the_direct_writer(tmp_path, emit_models, name):
    """Per-group areas equal the plan and no triangle or tet straddles a band boundary."""
    model = emit_models[name]
    band_groups = [n for n in model.routed.groups if "::band=" in n]
    assert len(band_groups) == (8 if name.startswith("pad_a") else 12)
    plan, path = written(model, tmp_path, "xao")
    sidecar = sidecar_of(model, plan)
    with gmsh_session(path):
        assert verify_loaded(sidecar).ok
        for group in band_groups:
            native = views.native_tags(sidecar, 2, views.group_members(sidecar)[group])
            loaded = sum(gmsh.model.occ.getMass(2, t) for t in native)
            assert loaded == pytest.approx(
                sidecar["group"][group]["area_um2"], rel=1e-9
            )
        tags = plan.tags()
        refine_near(
            model,
            tags,
            ("band_pad_top", "band_pad_bottom", "band_gap"),
            MESH_SIZE[name],
            1.0,
            (1.0, 20.0),
        )
        mesh_coarsely(MESH_SIZE[name])
        bands = band_mesh_report(model, tags, support_regions(model))
    assert set(bands) == set(band_groups)
    for group, row in bands.items():
        assert (
            row["triangles"] > 0
            and row["all_shared_by_tets"]
            and row["centroids_in_band"]
        ), group
        assert (
            len(set(row["tets_per_side"].values())) == 1
        )  # the same triangles are tet faces on both sides


def test_pad_b_bands_meshed_with_the_cheap_recipe_has_no_flat_tetrahedron(
    tmp_path, emit_models
):
    """The 0.1 um band strips make a uniform 60 um mesh put a zero-volume tet on them; the cheap recipe does not.

    With ``pad_b_bands`` at a uniform 60 um and Delaunay, four parallel band-boundary edges symmetric about
    x = 0 each get a node at x = 0, and Delaunay joins those four collinear nodes into a tetrahedron of
    volume zero (``minSICN`` 0).  The geometry is sound (OCCT checks pass, the Gmsh-API emission of the same
    topology has the same flat tets, a 5 um margin set or a slightly asymmetric cutout has none), so the
    remedy is the mesher's: a maximum size of a twentieth of the bounding-box diagonal with HXT (the default
    algorithm when HXT aborts), no size field and no use of the band features.
    """
    model = emit_models["pad_b_bands"]
    plan, path = written(model, tmp_path, "xao")
    sidecar = sidecar_of(model, plan)
    with gmsh_session(path):
        apply_sidecar_embeds(sidecar)
        mesh_with_cheap_recipe(path, sidecar)
        quality = mesh_quality()
        conformity = check_mesh_conformity(model, plan.tags())
    assert quality["minSICN_min"] > 1e-4
    assert conformity["conformal"], conformity["problems"]


# --------------------------------------------------------------------------- the Gmsh-API oracle


@pytest.mark.parametrize(
    "name", ["pad_a", "pad_b", "pad_a_bands", "pad_a_refinement", "void_shell_b"]
)
def test_the_gmsh_api_oracle_agrees_on_entities_groups_and_face_areas(
    tmp_path, emit_models, name
):
    """An independent bottom-up replay through Gmsh's API gives the same entities, groups and face areas."""
    model = emit_models[name]
    plan, path = written(model, tmp_path, "xao")
    with gmsh_session():
        oracle = emit_with_gmsh_api(model)
        oracle_area = {
            fid: gmsh.model.occ.getMass(2, tag)
            for fid, (dim, tag) in oracle.tags.items()
            if dim == 2
        }
        oracle_groups = dict(oracle.groups)
        counts = {d: len(gmsh.model.getEntities(d)) for d in range(4)}
    assert counts == {
        0: plan.counts["vertices"],
        1: plan.counts["edges"],
        2: plan.counts["faces"],
        3: plan.counts["solids"],
    }
    assert {oid for oid, (dim, _) in oracle.tags.items() if dim == 2} == set(
        plan.face_ids
    )
    assert set(oracle_groups) == {g.name for g in plan.groups}
    with gmsh_session(path):
        for i, fid in enumerate(plan.face_ids, start=1):
            mine = gmsh.model.occ.getMass(2, i)
            tolerance = (
                1e-3 if model.topology.faces[fid].curved else 1e-9
            )  # a 4-edge wall is exact, longer ones are filled
            assert mine == pytest.approx(oracle_area[fid], rel=tolerance), fid


# --------------------------------------------------------------------------- bytes across interpreters and at scale

SCRIPT = (
    f"import sys; sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})\n"
    "import hashlib, pathlib, tempfile\n"
    "from sgb_emit_cases import pad_model\n"
    "from scgsim.geometry.sgb.output.emission.plan import plan_brep\n"
    "from scgsim.geometry.sgb.output.emission.writer import write_xao_file\n"
    "digests = []\n"
    "for route in 'ABC':\n"
    "    model = pad_model(route, bands=True, refine=route != 'C')\n"
    "    path = pathlib.Path(tempfile.mkdtemp()) / 'm.xao'\n"
    "    write_xao_file(plan_brep(model.topology, model.routed), path, name='m', chunk_items=50)\n"
    "    digests.append(hashlib.sha256(path.read_bytes()).hexdigest())\n"
    "print(*digests)\n"
)


def test_the_xao_bytes_do_not_depend_on_python_hash_randomisation():
    """Three hash seeds and tiny chunks: one digest per route."""
    outputs = set()
    for hash_seed in ("0", "7", "31337"):
        result = subprocess.run(
            [sys.executable, "-c", SCRIPT],
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.add(result.stdout.strip())
    assert len(outputs) == 1


@pytest.mark.slow
@pytest.mark.parametrize("route", ["A", "B", "C"])
def test_a_random_layout_round_trips(tmp_path, route):
    """150 random features: written, loaded in Gmsh, checked against the sidecar alone."""
    catalogue = random_layout(150, seed=2, route="A" if route == "A" else "B")
    model = build_model_from_catalogue(catalogue, route)
    artifacts = emit_model(model, tmp_path)
    assert verify_xao(artifacts.xao, read_sidecar(artifacts.sidecar)).ok
