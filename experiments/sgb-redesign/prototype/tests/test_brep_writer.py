"""Direct BREP / XAO serialisation (experiment E3): format, sharing, orientation, round trip through Gmsh."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from contextlib import contextmanager

import gmsh
import numpy as np
import pytest

from sgbproto import brep_writer
from sgbproto.brep_writer import plan_brep, planned_counts, verify_loaded, write_brep, write_xao
from sgbproto.errors import SgbError
from sgbproto.pipeline import build
from sgbproto.scenes import airbridge, pad_in_ground, random_layout

CASES = {
    "pad_a": (lambda: pad_in_ground("A"), 40.0, 0.0),
    "pad_b": (lambda: pad_in_ground("B"), 60.0, 2.0),  # arcs, cylinder walls, an embedded port face
    "bridge": (airbridge, 60.0, 2.0),
}


@pytest.fixture(scope="module")
def topologies():
    return {name: build(make()).topology for name, (make, _, _) in CASES.items()}


@contextmanager
def loaded(path, autofix=0):
    gmsh.initialize(["-noenv"])
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Geometry.OCCAutoFix", autofix)
        (gmsh.open if str(path).endswith(".xao") else gmsh.merge)(str(path))
        gmsh.model.occ.synchronize()
        yield
    finally:
        gmsh.finalize()


# --------------------------------------------------------------------------- format


def test_file_layout_matches_the_documented_sections(tmp_path, topologies):
    res = write_brep(topologies["pad_a"], tmp_path / "m.brep")
    lines = (tmp_path / "m.brep").read_text().splitlines()
    assert lines[:3] == ["DBRep_DrawableShape", "", "CASCADE Topology V1, (c) Matra-Datavision"]
    headers = [ln.split()[0] for ln in lines if ln.split() and ln.split()[0] in
               {"Locations", "Curve2ds", "Curves", "Polygon3D", "PolygonOnTriangulations", "Surfaces", "Triangulations", "TShapes"}]
    assert headers == ["Locations", "Curve2ds", "Curves", "Polygon3D", "PolygonOnTriangulations", "Surfaces", "Triangulations", "TShapes"]
    c = res.plan.counts
    total = 1 + c["vertices"] + c["edges"] + c["wires"] + c["faces"] + c["shells"] + c["solids"]
    assert f"TShapes {total}" in lines
    kinds = [ln for ln in lines if ln in {"Ve", "Ed", "Wi", "Fa", "Sh", "So", "Co"}]
    assert [kinds.count(k) for k in ("Ve", "Ed", "Wi", "Fa", "Sh", "So", "Co")] == [
        c["vertices"], c["edges"], c["wires"], c["faces"], c["shells"], c["solids"], 1]
    assert lines[-1] == "+1 0 "  # the root: the compound


def test_tolerance_is_written_on_every_vertex_edge_and_face(tmp_path, topologies):
    write_brep(topologies["pad_a"], tmp_path / "m.brep", tolerance=2.5e-7)
    text = (tmp_path / "m.brep").read_text()
    assert text.count("\n2.5e-07\n") == plan_brep(topologies["pad_a"]).counts["vertices"]  # vertices
    assert " 2.5e-07 1 1 0\n" in text and "0  2.5e-07 " in text  # edges, faces


# --------------------------------------------------------------------------- sharing and tag prediction


@pytest.mark.parametrize("name", CASES)
def test_shared_faces_edges_and_vertices_are_written_once(topologies, name):
    topo = topologies[name]
    plan = plan_brep(topo)
    planned = planned_counts(topo)
    assert (len(plan.vertex_ids), len(plan.edge_ids), len(plan.face_ids), len(plan.solid_ids)) == (
        planned[0], planned[1], planned[2], planned[3])
    uses: dict[int, int] = {}
    for shell in plan.shells:
        for ref in shell:
            uses[abs(ref)] = uses.get(abs(ref), 0) + 1
    interface = [f for f, n in uses.items() if n == 2]
    assert interface, "the two emitted volumes share interface faces"
    # an interface face is referenced once with each orientation
    for f in interface:
        assert sorted(ref for shell in plan.shells for ref in shell if abs(ref) == f) == [-f, f]


@pytest.mark.parametrize("name", CASES)
def test_round_trip_counts_boundaries_groups_volumes_and_conformal_mesh(tmp_path, topologies, name):
    topo = topologies[name]
    _, size, size_min = CASES[name]
    for writer, suffix in ((write_brep, "brep"), (write_xao, "xao")):
        res = writer(topo, tmp_path / f"m.{suffix}")
        with loaded(res.path):
            report = verify_loaded(topo, res.plan, mesh_size=size, mesh_size_min=size_min, groups=suffix == "xao")
        assert report["ok"], report["problems"]
        assert report["max_vertex_offset_um"] < 1e-9
        assert report["boundaries_mismatching"] == {"edges": 0, "faces": 0, "solids": 0}
        assert report["mesh"]["conformal"] and report["mesh"]["duplicate_nodes"] == 0


def test_xao_is_well_formed_and_lists_the_planned_groups(tmp_path, topologies):
    topo = topologies["pad_b"]
    res = write_xao(topo, tmp_path / "m.xao")
    root = ET.parse(tmp_path / "m.xao").getroot()
    names = {g.get("name") for g in root.find("groups")}
    assert names == set(topo.groups) and int(root.find("groups").get("count")) == len(topo.groups)
    assert int(root.find("geometry/topology/faces").get("count")) == len(res.plan.face_ids)
    assert root.find("geometry/shape").get("format") == "BREP"
    refs = [int(f.get("reference")) for f in root.find("geometry/topology/faces")]
    assert refs == sorted(set(refs))  # strictly increasing pre-order indices


def test_every_group_element_is_a_planned_entity_of_the_right_dimension(topologies):
    topo = topologies["pad_b"]
    plan = plan_brep(topo)
    for name, dim, tag, elements in plan.groups:
        ids = plan.face_ids if dim == "face" else plan.solid_ids
        assert {ids[i] for i in elements} <= set(topo.groups[name]["ids"])
        assert len(elements) == len(set(elements))


# --------------------------------------------------------------------------- orientation


def test_reversed_shells_give_negative_volume_so_the_mass_check_sees_orientation(tmp_path, topologies):
    topo = topologies["pad_a"]
    plan = plan_brep(topo)
    plan.shells = [[-ref for ref in shell] for shell in plan.shells]
    res = write_brep(topo, tmp_path / "bad.brep", plan=plan)
    with loaded(res.path):
        masses = [gmsh.model.occ.getMass(3, t) for _, t in gmsh.model.getEntities(3)]
        assert all(m < 0 for m in masses)


def test_pad_b_volumes_equal_the_closed_form_including_cylinder_walls(tmp_path, topologies):
    topo = topologies["pad_b"]
    res = write_brep(topo, tmp_path / "m.brep")
    polygon = [(-45, -30), (-10, -30), (-10, -15), (10, -15), (10, -30), (50, -30), (50, 18), (38, 30), (-50, 30), (-50, -25)]
    area = sum(polygon[i][0] * polygon[(i + 1) % 10][1] - polygon[(i + 1) % 10][0] * polygon[i][1] for i in range(10)) / 2
    area += 25 * (math.pi / 2 - 1) / 2 - math.pi * 64
    expected = {"substrate": 9e6, "vacuum": 9e6 - (90000 - 19200) * 0.2 - area * 0.2}
    assert res.plan.counts["cylinders"] == 5 and res.plan.counts["arcs"] == 10
    with loaded(res.path):
        for i, vid in enumerate(res.plan.solid_ids):
            mass = gmsh.model.occ.getMass(3, i + 1)
            assert mass == pytest.approx(expected[topo.volumes[vid].domain.name], rel=1e-12)


# --------------------------------------------------------------------------- pcurves, parsed back from the text


def _parse_numbers(line: str) -> list[float]:
    return [float(x) for x in line.split()]


def _read_sections(path):
    lines = path.read_text().splitlines()
    sections: dict[str, list[str]] = {}
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if parts and parts[0] in {"Curve2ds", "Curves", "Surfaces"}:
            n = int(parts[1])
            sections[parts[0]] = lines[i + 1:i + 1 + n]
            i += n
        elif parts and parts[0] == "TShapes":
            sections["TShapes"] = lines[i + 1:]
            break
        i += 1
    return sections


def _eval_curve(rec: list[float], t: float) -> np.ndarray:
    if rec[0] == 1:
        return np.array(rec[1:4]) + t * np.array(rec[4:7])
    c, n, x, y, r = (np.array(rec[1:4]), np.array(rec[4:7]), np.array(rec[7:10]), np.array(rec[10:13]), rec[13])
    return c + r * (math.cos(t) * x + math.sin(t) * y)


def _eval_surface(rec: list[float], u: float, v: float) -> np.ndarray:
    o, n, x, y = (np.array(rec[1 + 3 * k:4 + 3 * k]) for k in range(4))
    if rec[0] == 1:
        return o + u * x + v * y
    return o + rec[13] * (math.cos(u) * x + math.sin(u) * y) + v * n


def _eval_pcurve(rec: list[float], t: float) -> tuple[float, float]:
    if rec[0] == 1:
        return rec[1] + t * rec[3], rec[2] + t * rec[4]
    c, x, y, r = np.array(rec[1:3]), np.array(rec[3:5]), np.array(rec[5:7]), rec[7]
    p = c + r * (math.cos(t) * x + math.sin(t) * y)
    return p[0], p[1]


@pytest.mark.parametrize("name", ["pad_b", "bridge"])
def test_every_pcurve_lifts_onto_its_3d_curve(tmp_path, topologies, name):
    """Parse the written file: for each edge representation, surface(pcurve(t)) == curve(t) at both ends and the middle."""
    write_brep(topologies[name], tmp_path / "m.brep")
    sections = _read_sections(tmp_path / "m.brep")
    curves = [_parse_numbers(x) for x in sections["Curves"]]
    pcurves = [_parse_numbers(x) for x in sections["Curve2ds"]]
    surfaces = [_parse_numbers(x) for x in sections["Surfaces"]]
    tshapes, checked, worst = sections["TShapes"], 0, 0.0
    for i, line in enumerate(tshapes):
        if line != "Ed":
            continue
        j, reps = i + 2, []
        while tshapes[j] != "0":
            reps.append(tshapes[j].split())
            j += 1
        _, c_idx, _, _, last = reps[0][0], int(reps[0][1]), reps[0][2], reps[0][3], float(reps[0][4])
        for rep in reps[1:]:
            pc, surf, last_pc = pcurves[int(rep[1]) - 1], surfaces[int(rep[2]) - 1], float(rep[5])
            assert last_pc == last
            for t in (0.0, last / 2, last):
                u, v = _eval_pcurve(pc, t)
                worst = max(worst, float(np.linalg.norm(_eval_surface(surf, u, v) - _eval_curve(curves[c_idx - 1], t))))
            checked += 1
    assert checked > 0 and worst < 1e-9


def test_pcurves_are_not_needed_for_planes_but_are_for_cylinders(tmp_path, topologies):
    """Measured on OCCT 7.8.1 / Gmsh 4.15.2: without pcurves planar models load and mesh, a cylinder wall fails at merge."""
    for name, expect_ok in (("bridge", True), ("pad_b", False)):
        res = write_brep(topologies[name], tmp_path / f"{name}.brep", pcurves=False)
        gmsh.initialize(["-noenv"])
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("Geometry.OCCAutoFix", 0)
            try:
                gmsh.merge(str(res.path))
                gmsh.model.occ.synchronize()
                gmsh.option.setNumber("Mesh.MeshSizeMax", 60)
                gmsh.model.mesh.generate(3)
                ok = True
            except Exception:
                ok = False
        finally:
            gmsh.finalize()
        assert ok is expect_ok, name


# --------------------------------------------------------------------------- parallel text generation


def test_output_is_byte_identical_for_every_worker_count(tmp_path, topologies, monkeypatch):
    monkeypatch.setattr(brep_writer, "CHUNK_ITEMS", 7)  # many tiny chunks so the pool really interleaves
    topo = topologies["pad_b"]
    digests = set()
    for w in (1, 2, 4):
        write_xao(topo, tmp_path / f"w{w}.xao", workers=w, name="m")
        write_brep(topo, tmp_path / f"w{w}.brep", workers=w)
        digests.add((tmp_path / f"w{w}.xao").read_bytes())
    assert len(digests) == 1
    assert len({(tmp_path / f"w{w}.brep").read_bytes() for w in (1, 2, 4)}) == 1


def test_random_layout_round_trip_with_workers(tmp_path):
    topo = build(random_layout(40, seed=3, route="B")).topology
    res = write_xao(topo, tmp_path / "r.xao", workers=2)
    with loaded(res.path):
        report = verify_loaded(topo, res.plan)
    assert report["ok"], report["problems"]


# --------------------------------------------------------------------------- refusals


def test_unsupported_faces_are_refused_not_written_wrongly(topologies):
    topo = build(pad_in_ground("A")).topology
    wall = next(f for f in topo.faces.values() if f.kind == "wall") if any(f.kind == "wall" for f in topo.faces.values()) else None
    face = wall or next(iter(topo.faces.values()))
    face.planar = False
    with pytest.raises(SgbError):
        plan_brep(topo)


def test_an_edge_that_leaves_its_plane_is_refused(topologies):
    topo = build(pad_in_ground("B")).topology
    vertical = next(e for e in topo.edges.values() if e.kind == "vertical")
    topo.vertices[vertical.v1].x += 5  # drag one end of a wall column sideways
    with pytest.raises(SgbError):
        plan_brep(topo)


# --------------------------------------------------------------------------- follow-up: paths E3 did not exercise


def _floating_conductor_spec():
    """Route B: a finite conductor fully enclosed by vacuum -> the vacuum volume gets an inner (void) shell."""
    from fractions import Fraction

    from sgbproto.records import Grid, ModelSpec, PrismRecord, z_value
    from sgbproto.scenes import _chip_prisms, _footprint, rect_um

    grid = Grid()
    prisms = _chip_prisms(grid, 150.0)
    prisms.append(PrismRecord("float", _footprint(grid, rect_um(-20, -15, 20, 15)), z_value(20), z_value(21), "Al",
                              "conductor", net_id="floating"))
    return ModelSpec(grid, tuple(prisms), name="floating_B", meta={"route": "B"}), Fraction(40 * 30 * 1)


def test_inner_void_shell_is_written_as_a_second_forward_shell_with_cavity_orientation(tmp_path):
    spec, void_volume = _floating_conductor_spec()
    topo = build(spec).topology
    plan = plan_brep(topo)
    vacuum = next(i for i, vid in enumerate(plan.solid_ids) if topo.volumes[vid].domain.name == "vacuum")
    assert plan.solid_nshells[vacuum] == 2 and [r.role for r in topo.volumes[plan.solid_ids[vacuum]].shells] == ["outer", "inner"]
    res = write_brep(topo, tmp_path / "m.brep", plan=plan)
    text = (tmp_path / "m.brep").read_text()
    lines = text.splitlines()
    solid_children = [lines[i + 3] for i, ln in enumerate(lines) if ln == "So"]  # So, blank, flags, children
    assert sorted((len(ln.split()) - 1) // 2 for ln in solid_children) == [1, 2]  # substrate: one shell, vacuum: two
    with loaded(res.path):
        report = verify_loaded(topo, res.plan, mesh_size=60.0, mesh_size_min=2.0, groups=False)
        masses = {topo.volumes[vid].domain.name: gmsh.model.occ.getMass(3, i + 1) for i, vid in enumerate(res.plan.solid_ids)}
    assert report["ok"], report["problems"]
    assert masses["vacuum"] == pytest.approx(300 * 300 * 100 - float(void_volume), rel=1e-12)  # the cavity is subtracted
    assert report["mesh"]["conformal"]


def test_a_void_shell_with_the_outward_orientation_is_visibly_wrong(tmp_path):
    """Negative control: flipping only the inner shell adds the cavity volume instead of removing it."""
    spec, void_volume = _floating_conductor_spec()
    topo = build(spec).topology
    plan = plan_brep(topo)
    vacuum = next(i for i, vid in enumerate(plan.solid_ids) if topo.volumes[vid].domain.name == "vacuum")
    inner = plan.solid_shell0[vacuum] + 1
    plan.shells[inner] = [-ref for ref in plan.shells[inner]]
    res = write_brep(topo, tmp_path / "bad.brep", plan=plan)
    with loaded(res.path):
        mass = gmsh.model.occ.getMass(3, vacuum + 1)
    assert mass == pytest.approx(300 * 300 * 100 + float(void_volume), rel=1e-12)


def _refinement_spec():
    from fractions import Fraction

    from sgbproto.records import CurveFamilyRecord, LineCarrier, Lineage, ModelSpec

    base = pad_in_ground("A")
    carrier = LineCarrier("ref0", (-40_000, 55_000), (40_000, 55_000), Lineage("family_member", family_id="ref", member="ref0"))
    family = CurveFamilyRecord("ref", "refinement", (carrier,), (Fraction(0),), "refinement_only")
    return ModelSpec(base.grid, base.prisms, (family,), name="with_refinement", meta=base.meta)


def test_refinement_only_curve_is_a_free_edge_of_the_compound_and_is_reembedded_by_the_mesher(tmp_path):
    topo = build(_refinement_spec()).topology
    host = [f for f in topo.faces.values() if f.refinement_edges]
    assert len(host) == 1 and len(host[0].refinement_edges) == 1
    res = write_xao(topo, tmp_path / "m.xao")
    assert res.plan.counts["free_edges"] == 1
    with loaded(res.path):
        report = verify_loaded(topo, res.plan, mesh_size=40.0)
        edge_tag = res.plan.tags()[host[0].refinement_edges[0]][1]
        assert gmsh.model.getAdjacencies(1, edge_tag)[0].size == 0  # free: no face owns it in the file
        assert gmsh.model.mesh.getElements(1, edge_tag)[1][0].size > 1  # meshed, embedded in the host face
    assert report["ok"], report["problems"]
    side = json.loads((tmp_path / "m.sidecar.json").read_text())
    assert side["refinement_edges"] == [{"edge": host[0].refinement_edges[0], "host_face": host[0].id, "written_as_free_edge": True}]


# --------------------------------------------------------------------------- inset bands


MARGINS = (0.1, 0.5, 2.0)
REFINE = {"refine_families": ("band_pad_top", "band_pad_bottom", "band_gap"), "refine_size": 1.0, "refine_distance": (1.0, 20.0)}


# MeshSizeMin stays 0 here: a minimum size of 2 um cannot resolve the 0.1 um band (the gmsh-API path fails the same way)
@pytest.mark.parametrize("route,size,size_min", [("A", 40.0, 0.0), ("B", 60.0, 0.0)])
def test_band_faces_through_the_direct_writer(tmp_path, route, size, size_min):
    spec = pad_in_ground(route, bands_um=MARGINS)
    topo = build(spec).topology
    supports = {f.family_id: f.support for f in spec.families}
    band_groups = [n for n in topo.groups if "::band=" in n]
    assert len(band_groups) == (8 if route == "A" else 12)  # Route B bands the pad's top (MA) and bottom (MS) face
    res = write_xao(topo, tmp_path / "m.xao")
    with loaded(res.path):
        report = verify_loaded(topo, res.plan, mesh_size=size, mesh_size_min=size_min, supports=supports, **REFINE)
    assert report["ok"], report["problems"]
    assert report["boundaries_mismatching"] == {"edges": 0, "faces": 0, "solids": 0}
    assert report["max_relative_face_area_error"] < 1e-9
    for name in band_groups:  # per-group area equals the plan
        row = report["group_areas_um2"][name]
        assert row["loaded"] == pytest.approx(row["planned"], rel=1e-9)
    bands = report["mesh"]["bands"]
    assert set(bands) == set(band_groups)
    for name, row in bands.items():
        assert row["triangles"] > 0 and row["all_shared_by_tets"] and row["centroids_in_band"], name
        assert len(set(row["tets_per_side"].values())) == 1  # the same triangles are tet faces on both sides


# --------------------------------------------------------------------------- production sidecar


SIDECAR_CASES = {**{k: v for k, v in CASES.items()}, "bands_A": (lambda: pad_in_ground("A", bands_um=MARGINS), 40.0, 0.0),
                 "bridge_refine": (_refinement_spec, 40.0, 0.0)}


@pytest.mark.parametrize("name", SIDECAR_CASES)
def test_sidecar_agrees_with_the_gmsh_api_oracle_on_groups_unions_and_embeds(tmp_path, name):
    from sgbproto.emit import emit

    topo = build(SIDECAR_CASES[name][0]()).topology
    write_xao(topo, tmp_path / "direct.xao")
    mine = json.loads((tmp_path / "direct.sidecar.json").read_text())
    oracle = json.loads(emit(topo, tmp_path / "api", "model").sidecar_path.read_text())
    strip = lambda groups: {n: (g["dim"], g["physical_tag"], g["ids"]) for n, g in groups.items()}
    assert strip(mine["physical_groups"]) == strip(oracle["physical_groups"])
    assert mine["unions"] == oracle["unions"]
    assert mine["embedded"] == oracle["embedded"]
    assert mine["surface_epr_rows"] == oracle["surface_epr_rows"]
    # curve-in-face embeddings are in both sidecars: (edge, host face) pairs and (face, host volume) pairs
    pair = lambda rows, a, b: sorted((r[a], r[b]) for r in rows)
    assert pair(mine["refinement_edges"], "edge", "host_face") == pair(oracle["refinement_edges"], "edge", "host_face")
    assert pair(mine["embedded_faces"], "face", "host_volume") == pair(oracle["embedded_faces"], "face", "host_volume")
    assert set(mine["curved_wall_faces"]) == set(oracle["curved_wall_faces"])
    assert mine["route"] == oracle["route"] and mine["grid_um"] == oracle["grid_um"]
    # same planned entities, one native tag each, per dimension
    oracle_ids = {dim: {e["id"] for e in oracle["construction_order"] if e["dim"] == int(dim)} for dim in "0123"}
    for dim, ids in mine["ids_by_tag"].items():
        assert set(ids) == oracle_ids[dim] and len(ids) == len(set(ids))
    # pairs: every embedded face names its host volume, every refinement edge its host face
    for pair in mine["embedded_faces"]:
        assert pair["face"] in oracle["embedded"][pair["host_volume"]]
    assert len(mine["refinement_edges"]) == sum(len(f.refinement_edges) for f in topo.faces.values()
                                                if f.id in set(mine["ids_by_tag"]["2"]))


def test_sidecar_tags_match_the_loaded_model_and_the_groups_resolve_to_the_planned_faces(tmp_path, topologies):
    from sgbproto.brep_writer import read_sidecar_tags

    topo = topologies["pad_b"]
    res = write_xao(topo, tmp_path / "m.xao")
    tags = read_sidecar_tags(tmp_path / "m.sidecar.json")
    assert tags == res.plan.tags()
    side = json.loads((tmp_path / "m.sidecar.json").read_text())
    with loaded(res.path):
        for name, group in side["physical_groups"].items():
            native = sorted(int(t) for t in gmsh.model.getEntitiesForPhysicalGroup(group["dim"], group["physical_tag"]))
            assert native == sorted(tags[i][1] for i in group["ids"]) == sorted(group["native_tags"]), name
            assert gmsh.model.getPhysicalName(group["dim"], group["physical_tag"]) == name
    assert side["lineage"]["faces"].keys() == set(side["ids_by_tag"]["2"])
    assert all(isinstance(c, str) for srcs in side["lineage"]["edges"].values() for c in srcs)


def test_sidecar_is_byte_identical_for_every_worker_count(tmp_path, topologies, monkeypatch):
    monkeypatch.setattr(brep_writer, "CHUNK_ITEMS", 7)
    topo = topologies["pad_b"]
    blobs = set()
    for w in (1, 2, 4):
        write_xao(topo, tmp_path / f"w{w}.xao", workers=w, name="m", sidecar=tmp_path / f"side{w}.json")
        blobs.add((tmp_path / f"side{w}.json").read_bytes())
    assert len(blobs) == 1


def test_sidecar_can_be_switched_off_and_lineage_left_out(tmp_path, topologies):
    write_brep(topologies["pad_a"], tmp_path / "a.brep", sidecar=False)
    assert not (tmp_path / "a.sidecar.json").exists()
    write_brep(topologies["pad_a"], tmp_path / "b.brep", lineage=False)
    assert "lineage" not in json.loads((tmp_path / "b.sidecar.json").read_text())
