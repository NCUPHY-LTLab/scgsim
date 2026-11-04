"""Official OCCT validity checks (OCP: BRepCheck_Analyzer, BOPAlgo_ArgumentAnalyzer, ShapeAnalysis) on written models.

Positive fixtures must be clean; negative controls (deliberate defects) must be caught by the official checks.
Skipped when OCP (the dev extra) is not installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("OCP")

import occt_mutations as mutations  # noqa: E402
from test_brep_writer import _floating_conductor_spec, _refinement_spec  # noqa: E402

from sgbproto import occt_check  # noqa: E402
from sgbproto.brep_writer import plan_brep, write_brep, write_xao  # noqa: E402
from sgbproto.errors import SgbError  # noqa: E402
from sgbproto.occt_check import OcctCheckError, OcctUnavailable, check_file, check_shape, read_brep  # noqa: E402
from sgbproto.pipeline import build  # noqa: E402
from sgbproto.scenes import airbridge, pad_in_ground, random_layout  # noqa: E402

FIXTURES = {
    "pad_a": lambda: pad_in_ground("A"),
    "pad_b": lambda: pad_in_ground("B"),
    "bridge": airbridge,
    "pad_a_bands": lambda: pad_in_ground("A", bands_um=(0.1, 0.5, 2.0)),
    "pad_b_bands": lambda: pad_in_ground("B", bands_um=(0.1, 0.5, 2.0)),
    "void_shell": lambda: _floating_conductor_spec()[0],
    "refinement": _refinement_spec,
    "pad_a_polygon": lambda: pad_in_ground("A", hole_sides=32, rounded_corner=False),
    **{f"random10_{r}_{s}": (lambda r=r, s=s: random_layout(10, seed=s, route=r)) for r in "AB" for s in (0, 1, 2)},
    **{f"random100_{r}_{s}": (lambda r=r, s=s: random_layout(100, seed=s, route=r)) for r in "AB" for s in (0, 1)},
    "random1000_B_1": lambda: random_layout(1000, seed=1, route="B"),
}


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_written_brep_and_the_brep_inside_the_xao_pass_every_official_check(tmp_path, name):
    topo = build(FIXTURES[name]()).topology
    brep = write_brep(topo, tmp_path / "m.brep", sidecar=False)
    xao = write_xao(topo, tmp_path / "m.xao", sidecar=False)
    for result in (brep, xao):
        report = check_file(result.path, topo, result.plan, self_interference=len(topo.faces) < 5000)
        assert report.ok, report.problems
        assert report.brepcheck["valid"] and report.brepcheck["faulty_count"] == 0
        assert not report.arguments["faulty"] and report.arguments["flags"]["SmallEdgeMode"]
        assert report.shape_analysis["shells_with_free_edges"] == 0 and report.shape_analysis["edges_below_10x_tolerance"] == 0
        assert report.counts["solid"] == len(result.plan.solid_ids) and report.counts["face"] == len(result.plan.face_ids)
        assert report.properties["faces"]["max_relative_error"] < 1e-7
        assert all(s["relative_error"] < 1e-7 for s in report.properties["solids"])


# --------------------------------------------------------------------------- negative controls


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    path = tmp_path_factory.mktemp("occt") / "m.brep"
    topo = build(pad_in_ground("B")).topology  # arcs, cylinder walls, holes, an embedded face
    result = write_brep(topo, path, sidecar=False)
    return topo, result.plan, path


def detections(loaded, control):
    topo, plan, path = loaded
    report = check_shape(mutations.CONTROLS[control](read_brep(path)), topo, plan, source=control)
    return report, {
        "BRepCheck": not report.brepcheck["valid"],
        "BOPAlgo_ArgumentAnalyzer": report.arguments["faulty"],
        "ShapeAnalysis_Shell": bool(report.shape_analysis["shells_with_free_edges"]
                                    or report.shape_analysis["shells_with_bad_edges"]),
        "plan_cross_check": bool(report.properties["mismatches"]),
    }


@pytest.mark.parametrize("control", sorted(mutations.CONTROLS))
def test_every_negative_control_is_caught_by_brepcheck_and_the_report_fails(loaded, control):
    report, caught = detections(loaded, control)
    assert caught["BRepCheck"], f"BRepCheck_Analyzer missed {control}"
    assert not report.ok
    with pytest.raises(OcctCheckError):
        report.raise_if_failed()


# What else sees each defect (recorded, not a requirement of the checkers): documents which of our own invariants
# are indispensable. BRepCheck alone reports a status list only for some; the others are flagged through contexts.
EXPECTED_ALSO_CAUGHT_BY = {
    "reversed_inner_wire": {"ShapeAnalysis_Shell", "plan_cross_check"},
    "flipped_shared_face_in_one_shell": {"BOPAlgo_ArgumentAnalyzer", "ShapeAnalysis_Shell", "plan_cross_check"},
    "vertex_moved_1e-4_um_off_its_curve": {"BOPAlgo_ArgumentAnalyzer"},  # areas/volumes move by < 1e-7: plan check is blind
    "missing_face_open_shell": {"ShapeAnalysis_Shell", "plan_cross_check"},
    "wire_with_a_gap": {"BOPAlgo_ArgumentAnalyzer", "ShapeAnalysis_Shell"},
}


@pytest.mark.parametrize("control", sorted(mutations.CONTROLS))
def test_which_other_checks_also_see_each_defect(loaded, control):
    _, caught = detections(loaded, control)
    assert {k for k, v in caught.items() if v and k != "BRepCheck"} == EXPECTED_ALSO_CAUGHT_BY[control]


def test_a_vertex_move_inside_the_tolerance_is_not_flagged(loaded):
    topo, plan, path = loaded
    shape = mutations.vertex_off_curve(read_brep(path), distance_um=1e-9)  # 1e-9 um << tolerance 1e-7 um
    assert check_shape(shape, topo, plan).brepcheck["valid"]


def test_the_brepcheck_report_lists_per_subshape_statuses_not_only_a_boolean(loaded):
    topo, plan, path = loaded
    report = check_shape(mutations.reversed_inner_wire(read_brep(path)), topo, plan)
    assert report.brepcheck["histogram"] == {"face:BadOrientationOfSubshape": 1, "shell:BadOrientationOfSubshape": 1}
    assert {r["kind"] for r in report.brepcheck["faulty"]} >= {"face", "shell"}


# --------------------------------------------------------------------------- strict mode hook


def test_strict_write_checks_the_written_file_and_returns_the_report(tmp_path):
    topo = build(pad_in_ground("B")).topology
    result = write_xao(topo, tmp_path / "m.xao", strict=True)
    assert result.occt is not None and result.occt.ok
    assert write_xao(topo, tmp_path / "plain.xao").occt is None


def test_strict_without_ocp_raises_a_clear_error_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(occt_check, "OCP_AVAILABLE", False)
    topo = build(pad_in_ground("A")).topology
    for writer, name in ((write_xao, "m.xao"), (write_brep, "m.brep")):
        with pytest.raises(OcctUnavailable, match="dev extra"):
            writer(topo, tmp_path / name, strict=True)
        assert not (tmp_path / name).exists()
    assert isinstance(OcctUnavailable("x"), SgbError)


def test_strict_raises_on_a_broken_plan(tmp_path):
    topo = build(pad_in_ground("B")).topology
    plan = plan_brep(topo)
    face = next(i for i, w in enumerate(plan.face_nwires) if w > 1)
    first_wire = plan.face_wire0[face]
    plan.wires[first_wire + 1] = [-e for e in reversed(plan.wires[first_wire + 1])]  # inner wire in the outer's sense
    with pytest.raises(OcctCheckError):
        write_brep(topo, tmp_path / "bad.brep", plan=plan, sidecar=False, strict=True)


def test_refinement_curve_is_excluded_from_the_argument_analyzer_but_not_from_brepcheck(tmp_path):
    topo = build(_refinement_spec()).topology
    result = write_brep(topo, tmp_path / "m.brep", sidecar=False)
    report = check_file(result.path, topo, result.plan)
    assert report.arguments["excluded_free_edges"] == 1  # an embedded curve interferes with its host face by design
    assert report.ok and report.brepcheck["checked"]["edge"] == len(result.plan.edge_ids)


def test_strict_options_are_passed_to_the_checker(tmp_path):
    topo = build(pad_in_ground("A")).topology
    result = write_brep(topo, tmp_path / "m.brep", sidecar=False, strict={"self_interference": False})
    assert result.occt.ok and result.occt.arguments["flags"]["SelfInterMode"] is False


def test_strict_mode_skips_the_expensive_self_interference_check_unless_asked(tmp_path):
    topo = build(pad_in_ground("A")).topology
    cheap = write_brep(topo, tmp_path / "a.brep", sidecar=False, strict=True)
    full = write_brep(topo, tmp_path / "b.brep", sidecar=False, strict={"self_interference": True})
    assert cheap.occt.arguments["flags"]["SelfInterMode"] is False
    assert full.occt.arguments["flags"]["SelfInterMode"] is True and full.occt.ok


def test_embedded_free_faces_are_excluded_from_the_solid_analysis_and_analysed_on_their_own(tmp_path):
    from sgbproto.extreme_cases import FIXTURES

    topo = build(FIXTURES["floating_conductor"].build("A")).topology
    result = write_brep(topo, tmp_path / "m.brep", sidecar=False)
    report = check_file(result.path, topo, result.plan)
    assert report.ok and report.arguments["excluded_free_faces"] == 1
