"""Bottom-up emission, XAO + sidecar, and mesh conformity (brief section 12)."""

from __future__ import annotations

import json
import math

import gmsh
import pytest

from sgbproto.emit import emit
from sgbproto.pipeline import build
from sgbproto.scenes import airbridge, pad_in_ground

CASES = [
    ("pad_a", lambda: pad_in_ground("A"), 40.0, 0.0),
    ("pad_b", lambda: pad_in_ground("B"), 60.0, 2.0),
    ("bridge", airbridge, 60.0, 2.0),
]


@pytest.mark.parametrize("name,make,size,size_min", CASES)
def test_emission_is_conformal_and_every_native_entity_has_one_planned_id(tmp_path, name, make, size, size_min):
    result = build(make())
    out = emit(result.topology, tmp_path, name, mesh_size=size, mesh_size_min=size_min)
    assert out.mesh["conformal"], out.mesh["problems"]
    assert out.mesh["duplicate_nodes"] == 0
    sidecar = json.loads(out.sidecar_path.read_text())
    order = sidecar["construction_order"]
    assert len({(e["dim"], e["tag"]) for e in order}) == len(order)  # one tag per planned id
    assert [e["order"] for e in order] == list(range(len(order)))
    assert {e["id"] for e in order} == set(out.tags)


def test_xao_reopens_with_the_planned_groups_and_volume_count(tmp_path):
    result = build(pad_in_ground("B"))
    out = emit(result.topology, tmp_path, "model")
    gmsh.initialize(["-noenv"])
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(out.xao_path))
        names = {gmsh.model.getPhysicalName(d, t) for d, t in gmsh.model.getPhysicalGroups()}
        assert names == set(result.topology.groups)
        assert len(gmsh.model.getEntities(3)) == 2
        assert len(gmsh.model.getEntities(2)) == len([i for i in out.tags.values() if i[0] == 2])
    finally:
        gmsh.finalize()


def test_emission_uses_no_boolean_and_faces_are_shared_not_duplicated(tmp_path):
    result = build(pad_in_ground("A"))
    out = emit(result.topology, tmp_path, "model")
    planned_faces = [i for i, (d, _) in out.tags.items() if d == 2]
    # 15 planned faces for 2 volumes; the interface faces are emitted once and shared
    assert len(planned_faces) == len(set(planned_faces)) == 15


def test_duplicate_node_check_detects_a_non_conformal_stack():
    """Negative control: two boxes with separate (duplicated) interface surfaces mesh with coincident nodes."""
    from sgbproto.emit import duplicate_node_count

    gmsh.initialize(["-noenv"])
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("stack")
        gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        gmsh.model.occ.addBox(0, 0, 1, 1, 1, 1)  # touching, deliberately not fragmented
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.5)
        gmsh.model.mesh.generate(3)
        _, duplicates = duplicate_node_count()
        assert duplicates > 0
    finally:
        gmsh.finalize()


def test_curved_walls_are_exact_cylinders_and_the_mesh_stays_conformal(tmp_path):
    """Route B pad: the circular hole (4 quarter arcs) and the rounded corner are cylinder walls through 4 shared edges."""
    result = build(pad_in_ground("B"))
    out = emit(result.topology, tmp_path, "model", mesh_size=60.0, mesh_size_min=2.0)
    assert out.mesh["conformal"], out.mesh["problems"]
    assert len(out.wall_deviation) == 5  # 4 hole quarter arcs + 1 rounded corner
    assert max(w["vs_exact_cylinder_grid"] for w in out.wall_deviation) < 1e-6  # addBSplineFilling is exact here
    assert all(w["vs_realised_cylinder_grid"] < 1e-6 for w in out.wall_deviation)
    sidecar = json.loads(out.sidecar_path.read_text())
    assert len(sidecar["curved_wall_faces"]) == 5 and len(sidecar["arc_centre_points"]) >= 5


def test_arc_centre_points_are_construction_aids_not_native_entities(tmp_path):
    result = build(pad_in_ground("A"))
    out = emit(result.topology, tmp_path, "model")
    centres = {v for v in result.topology.vertices if v.startswith("vc:")}
    assert centres and not centres & set(out.tags)  # no free point is left in the model to be meshed


# --------------------------------------------------------------------------- OCC masses against the plan

MASS_CASES = {
    "pad_a": lambda: pad_in_ground("A"),
    "pad_b": lambda: pad_in_ground("B"),
    "bridge": airbridge,
    "pad_a_bands": lambda: pad_in_ground("A", bands_um=(0.1, 0.5, 2.0)),
    "pad_b_bands": lambda: pad_in_ground("B", bands_um=(0.1, 0.5, 2.0)),
    "pad_a_polygon": lambda: pad_in_ground("A", hole_sides=32, rounded_corner=False),
}


@pytest.mark.parametrize("name", sorted(MASS_CASES))
def test_every_emitted_face_area_and_solid_mass_equals_the_plan(tmp_path, name):
    """A hole wire with the wrong OCC sense doubles-counts the hole (area = outer + hole), and a solid built from
    a wrongly oriented shell reports a negative mass; neither shows in the mesh, so compare ``getMass`` to the plan."""
    result = build(MASS_CASES[name]())
    out = emit(result.topology, tmp_path, "m", write_xao=False, measure_masses=True)
    faces, volumes = out.masses["faces"], out.masses["volumes"]
    assert faces and volumes
    assert json.loads(out.sidecar_path.read_text())["inside_out_solids"] == []
    for fid, row in faces.items():
        assert row["occ"] == pytest.approx(row["planned"], rel=1e-7), (row["group"], fid)
    for vid, row in volumes.items():
        assert row["planned"] > 0
        assert row["occ"] == pytest.approx(row["planned"], rel=1e-9), (row["group"], vid)


def test_pad_b_solid_masses_match_the_analytic_volumes(tmp_path):
    pad_area = 100 * 60 - 20 * 15 - 12 * 12 / 2 - 25 * (1 - math.pi / 4) - math.pi * 64  # um^2, analytic
    ground_area = 300 * 300 - 160 * 120
    out = emit(build(pad_in_ground("B")).topology, tmp_path, "m", write_xao=False, measure_masses=True)
    mass = {row["group"]: row["occ"] for row in out.masses["volumes"].values()}
    assert mass["VOL::substrate"] == pytest.approx(300 * 300 * 100, rel=1e-9)
    assert mass["VOL::vacuum"] == pytest.approx(300 * 300 * 100 - (ground_area + pad_area) * 0.2, rel=1e-9)


# --------------------------------------------------------------------------- D8: gmsh-API model vs BREP-written model


def group_masses_from_xao(path) -> dict[str, float]:
    """Sum of OCC masses per physical group of a model loaded from an XAO file."""
    gmsh.initialize(["-noenv"])
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Geometry.OCCAutoFix", 0)
        gmsh.open(str(path))
        gmsh.model.occ.synchronize()
        totals: dict[str, float] = {}
        for dim, tag in gmsh.model.getPhysicalGroups():
            name = gmsh.model.getPhysicalName(dim, tag)
            totals[name] = sum(gmsh.model.occ.getMass(dim, e) for e in gmsh.model.getEntitiesForPhysicalGroup(dim, tag))
        return totals
    finally:
        gmsh.finalize()


@pytest.mark.parametrize("name", sorted(MASS_CASES))
def test_gmsh_api_model_and_brep_written_model_have_equal_group_areas_and_volumes(tmp_path, name):
    """``emit`` is the oracle for ``brep_writer`` (design decision D8): same planned groups, same areas and volumes."""
    from sgbproto.brep_writer import write_xao

    topo = build(MASS_CASES[name]()).topology
    oracle = emit(topo, tmp_path, "api", write_xao=False, measure_masses=True)
    expected: dict[str, float] = {}
    for rows in oracle.masses.values():
        for row in rows.values():
            expected[row["group"]] = expected.get(row["group"], 0.0) + row["occ"]
    write_xao(topo, tmp_path / "brep.xao")
    written = group_masses_from_xao(tmp_path / "brep.xao")
    assert set(written) == set(expected) == set(topo.groups)
    for group, value in expected.items():
        assert written[group] == pytest.approx(value, rel=1e-7), group
