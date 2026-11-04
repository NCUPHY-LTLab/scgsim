"""Official OCCT validity checks (OCP) on written models, with negative controls, plus the AEDT STEP export.

Positive fixtures must be clean; deliberate defects -- in a loaded shape and in the BREP *file* -- must be caught by
the official checks.  The whole module is skipped where OCP (the ``occt`` dependency group) is not installed.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("OCP")

import sgb_occt_mutations as mutations  # noqa: E402
from OCP.BRepGProp import BRepGProp  # noqa: E402
from OCP.GProp import GProp_GProps  # noqa: E402
from OCP.IFSelect import IFSelect_RetDone  # noqa: E402
from OCP.Interface import Interface_Static  # noqa: E402
from OCP.STEPControl import STEPControl_Reader  # noqa: E402
from OCP.TopAbs import TopAbs_SOLID  # noqa: E402
from OCP.TopExp import TopExp  # noqa: E402
from OCP.TopTools import TopTools_IndexedMapOfShape  # noqa: E402
from sgb_emit_cases import CASES  # noqa: E402
from sgb_scenes import random_layout, slab_model  # noqa: E402

from scgsim.geometry.sgb.emission.plan import plan_brep  # noqa: E402
from scgsim.geometry.sgb.emission.writer import write_brep_file, write_xao_file  # noqa: E402
from scgsim.geometry.sgb.exchange.step import StepExportError, export_aedt_step  # noqa: E402
from scgsim.geometry.sgb.model import build_model_from_catalogue  # noqa: E402
from scgsim.geometry.sgb.records.grid import Grid  # noqa: E402
from scgsim.geometry.sgb.sidecar.build import SidecarContext, build_sidecar  # noqa: E402
from scgsim.geometry.sgb.slivers.report import sliver_report  # noqa: E402
from scgsim.geometry.sgb.verification import ocp  # noqa: E402
from scgsim.geometry.sgb.verification.occt import (
    OcctCheckError,
    check_file,
    check_shape,
    read_brep,
    read_shape,
    sidecar_mismatches,
)  # noqa: E402

pytestmark = pytest.mark.occt


@pytest.fixture(scope="module")
def models():
    """Every named case, built once."""
    return {name: build() for name, build in CASES.items()}


@pytest.mark.parametrize("name", sorted(CASES))
def test_written_brep_and_the_brep_inside_the_xao_pass_every_official_check(
    tmp_path, models, name
):
    """Written brep and the brep inside the xao pass every official check."""
    model = models[name]
    plan = plan_brep(model.topology, model.routed)
    write_brep_file(plan, tmp_path / "m.brep")
    write_xao_file(plan, tmp_path / "m.xao", brep=tmp_path / "m.brep")
    for path in (tmp_path / "m.brep", tmp_path / "m.xao"):
        report = check_file(
            path, model.topology, plan, self_interference=model.counts["faces"] < 5000
        )
        assert report.ok, report.problems
        assert report.brepcheck["valid"] and report.brepcheck["faulty_count"] == 0
        assert (
            not report.arguments["faulty"]
            and report.arguments["flags"]["SmallEdgeMode"]
        )
        assert report.shape_analysis["shells_with_free_edges"] == 0
        assert report.shape_analysis["edges_below_10x_tolerance"] == 0
        assert report.counts["solid"] == len(plan.solid_ids) and report.counts[
            "face"
        ] == len(plan.face_ids)
        assert report.properties["faces"]["max_relative_error"] < 1e-7
        assert all(s["relative_error"] < 1e-7 for s in report.properties["solids"])


# --------------------------------------------------------------------------- negative controls on a loaded shape


@pytest.fixture(scope="module")
def pad(tmp_path_factory, models):
    """Route B pad: arcs, cylinder walls, holes and an embedded face -- written once."""
    model = models["pad_b"]
    plan = plan_brep(model.topology, model.routed)
    path = tmp_path_factory.mktemp("occt") / "m.brep"
    write_brep_file(plan, path)
    return model.topology, plan, path


def detections(pad, control):
    """The report for a defective shape and which checkers saw the defect."""
    topology, plan, path = pad
    report = check_shape(
        mutations.CONTROLS[control](read_brep(path)), topology, plan, source=control
    )
    return report, {
        "BRepCheck": not report.brepcheck["valid"],
        "BOPAlgo_ArgumentAnalyzer": report.arguments["faulty"],
        "ShapeAnalysis_Shell": bool(
            report.shape_analysis["shells_with_free_edges"]
            or report.shape_analysis["shells_with_bad_edges"]
        ),
        "plan_cross_check": bool(report.properties["mismatches"]),
    }


@pytest.mark.parametrize("control", sorted(mutations.CONTROLS))
def test_every_negative_control_is_caught_by_brepcheck_and_the_report_fails(
    pad, control
):
    """Every negative control is caught by brepcheck and the report fails."""
    report, caught = detections(pad, control)
    assert caught["BRepCheck"], f"BRepCheck_Analyzer missed {control}"
    assert not report.ok
    with pytest.raises(OcctCheckError):
        report.raise_if_failed()


# What else is guaranteed to see each defect (a lower bound; which face is chosen decides whether more checkers also
# do): it documents which of our own invariants are indispensable.
ALSO_CAUGHT_BY = {
    "reversed_inner_wire": {"ShapeAnalysis_Shell", "plan_cross_check"},
    "flipped_shared_face_in_one_shell": {"ShapeAnalysis_Shell", "plan_cross_check"},
    "vertex_moved_1e-4_um_off_its_curve": {
        "BOPAlgo_ArgumentAnalyzer"
    },  # areas and volumes move by < 1e-7: the plan check is blind
    "missing_face_open_shell": {"ShapeAnalysis_Shell", "plan_cross_check"},
    "wire_with_a_gap": {"BOPAlgo_ArgumentAnalyzer", "ShapeAnalysis_Shell"},
}


@pytest.mark.parametrize("control", sorted(mutations.CONTROLS))
def test_which_other_checks_also_see_each_defect(pad, control):
    """Which other checks also see each defect."""
    _, caught = detections(pad, control)
    assert {k for k, v in caught.items() if v and k != "BRepCheck"} >= ALSO_CAUGHT_BY[
        control
    ]


def test_a_vertex_move_inside_the_tolerance_is_not_flagged(pad):
    """A vertex move inside the tolerance is not flagged."""
    topology, plan, path = pad
    shape = mutations.vertex_off_curve(
        read_brep(path), distance_um=1e-9
    )  # 1e-9 um << tolerance 1e-7 um
    assert check_shape(shape, topology, plan).brepcheck["valid"]


def test_the_brepcheck_report_lists_per_subshape_statuses_not_only_a_boolean(pad):
    """The brepcheck report lists per subshape statuses not only a boolean."""
    topology, plan, path = pad
    report = check_shape(mutations.reversed_inner_wire(read_brep(path)), topology, plan)
    assert report.brepcheck["histogram"] == {
        "face:BadOrientationOfSubshape": 1,
        "shell:BadOrientationOfSubshape": 1,
    }
    assert {r["kind"] for r in report.brepcheck["faulty"]} >= {"face", "shell"}


# --------------------------------------------------------------------------- negative controls on the BREP file itself


def corrupt(source: Path, target: Path, how: str) -> Path:
    """A copy of a BREP file with one deliberate defect written into its text."""
    text = source.read_text()
    if how == "flip_a_face_in_a_shell":
        text = re.sub(r"(Sh\n\n[01]+\n)\+", r"\1-", text, count=1)
    elif how == "drop_a_face_from_a_shell":
        text = re.sub(r"(Sh\n\n[01]+\n)\+\d+ 0 ", r"\1", text, count=1)
    elif how == "flip_an_edge_in_a_wire":
        text = re.sub(r"(Wi\n\n[01]+\n[+-]\d+ 0 )\+", r"\1-", text, count=1)
    elif how == "move_a_vertex":
        text = re.sub(
            r"(Ve\n[0-9.e+-]+\n)([0-9.e+-]+) ",
            lambda m: m.group(1) + repr(float(m.group(2)) + 1e-4) + " ",
            text,
            count=1,
        )
    elif how == "shorten_a_line_edge_range":
        text = re.sub(
            r"(Ed\n [0-9.e+-]+ 1 1 0\n1  \d+ 0 0 )([0-9.e+-]+)",
            lambda m: m.group(1) + repr(float(m.group(2)) * 0.5),
            text,
            count=1,
        )
    elif how == "truncate":
        text = text[: len(text) // 2]
    else:  # pragma: no cover - guards the parametrisation
        raise AssertionError(how)
    target.write_text(text)
    return target


@pytest.mark.parametrize(
    "how",
    [
        "flip_a_face_in_a_shell",
        "drop_a_face_from_a_shell",
        "flip_an_edge_in_a_wire",
        "move_a_vertex",
        "shorten_a_line_edge_range",
        "truncate",
    ],
)
def test_a_corrupted_brep_file_is_caught(pad, tmp_path, how):
    """A corrupted brep file is caught."""
    topology, plan, path = pad
    bad = corrupt(path, tmp_path / "bad.brep", how)
    try:
        report = check_file(bad, topology, plan, self_interference=False)
    except OcctCheckError:
        return  # the file could not even be read
    assert not report.ok, f"{how} went unnoticed"
    assert report.problems


def test_the_uncorrupted_file_is_clean_so_the_controls_are_meaningful(pad):
    """The uncorrupted file is clean so the controls are meaningful."""
    topology, plan, path = pad
    assert check_file(path, topology, plan, self_interference=False).ok


# --------------------------------------------------------------------------- what the argument analyzer must not see


def test_refinement_curve_is_excluded_from_the_argument_analyzer_but_not_from_brepcheck(
    tmp_path, models
):
    """Refinement curve is excluded from the argument analyzer but not from brepcheck."""
    model = models["pad_a_refinement"]
    plan = plan_brep(model.topology, model.routed)
    write_brep_file(plan, tmp_path / "m.brep")
    report = check_file(tmp_path / "m.brep", model.topology, plan)
    assert (
        report.arguments["excluded_free_edges"] == 1
    )  # an embedded curve interferes with its host face by design
    assert report.ok and report.brepcheck["checked"]["edge"] == len(plan.edge_ids)


def test_embedded_free_faces_are_excluded_from_the_solid_analysis_and_analysed_on_their_own(
    tmp_path, models
):
    """Embedded free faces are excluded from the solid analysis and analysed on their own."""
    model = models["floating_sheet_a"]
    plan = plan_brep(model.topology, model.routed)
    write_brep_file(plan, tmp_path / "m.brep")
    report = check_file(tmp_path / "m.brep", model.topology, plan)
    assert report.ok and report.arguments["excluded_free_faces"] == 1


def test_self_interference_is_a_switch_and_the_default_report_is_complete(
    tmp_path, models
):
    """Self interference is a switch and the default report is complete."""
    model = models["pad_a"]
    plan = plan_brep(model.topology, model.routed)
    write_brep_file(plan, tmp_path / "m.brep")
    cheap = check_file(
        tmp_path / "m.brep", model.topology, plan, self_interference=False
    )
    full = check_file(tmp_path / "m.brep", model.topology, plan, self_interference=True)
    assert cheap.arguments["flags"]["SelfInterMode"] is False
    assert (
        full.arguments["flags"]["SelfInterMode"] is True
        and full.ok
        and full.to_dict()["ok"]
    )


def test_without_ocp_the_checks_name_the_dependency_group(monkeypatch):
    """Without ocp the checks name the dependency group."""
    monkeypatch.setattr(ocp, "ocp_available", lambda: False)
    with pytest.raises(ocp.OcctUnavailable, match=r"occt` dependency group"):
        ocp.require_ocp()


# --------------------------------------------------------------------------- the AEDT STEP export


@pytest.fixture(scope="module")
def exported(tmp_path_factory, models):
    """Route B pad with bands: BREP, sidecar and the STEP export."""
    model = models["pad_b_bands"]
    plan = plan_brep(model.topology, model.routed)
    folder = tmp_path_factory.mktemp("step")
    write_brep_file(plan, folder / "m.brep")
    sidecar = build_sidecar(
        model, plan, SidecarContext(sliver_report(model)["summary"])
    )
    return (
        model,
        plan,
        sidecar,
        folder,
        export_aedt_step(folder / "m.brep", sidecar, folder / "m.step"),
    )


def test_the_step_is_an_assembly_of_named_solids_in_micrometres(exported):
    """The step is an assembly of named solids in micrometres."""
    _, _, sidecar, folder, export = exported
    text = (folder / "m.step").read_text()
    products = re.findall(r"PRODUCT\('([^']*)'", text)
    expected = {v["aedt_name"] for v in sidecar["volumes"].values() if v["emitted"]}
    assert export.object_names == expected == {"substrate", "vacuum"}
    assert set(products) == expected | {"SGB_MODEL"}
    assert export.unit == "UM" and "SI_UNIT(.MICRO.,.METRE.)" in text
    assert "SI_UNIT(.MILLI.,.METRE.)" not in text


def test_the_step_numbers_are_the_model_numbers_not_millimetres(exported):
    """The OCCT pitfall of E4: a wrong recipe divides the numbers by 1000 while keeping the micrometre label."""
    model, _, _, folder, _ = exported
    text = (folder / "m.step").read_text()
    xs = [
        float(x)
        for x, _, _ in re.findall(
            r"CARTESIAN_POINT\('[^']*',\(([-0-9.E+]+),([-0-9.E+]+),([-0-9.E+]+)\)", text
        )
    ]
    half = 150.0  # the die of the pad example is 300 um wide
    assert max(xs) == pytest.approx(half) and min(xs) == pytest.approx(-half)
    assert model.topology.grid_um == 0.001


def test_the_step_reads_back_with_the_planned_volumes(exported):
    """Re-read the STEP with OCP: two solids whose volumes equal the planned ones."""

    _, _, sidecar, folder, _ = exported
    Interface_Static.SetCVal_s(
        "xstep.cascade.unit", "UM"
    )  # read the micrometre numbers as micrometres
    reader = STEPControl_Reader()
    assert reader.ReadFile(str(folder / "m.step")) == IFSelect_RetDone
    reader.TransferRoots()
    solids = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(reader.OneShape(), TopAbs_SOLID, solids)
    volumes = sorted(
        (
            BRepGProp.VolumeProperties_s(solids.FindKey(i), props := GProp_GProps()),
            props.Mass(),
        )[1]
        for i in range(1, solids.Extent() + 1)
    )
    planned = sorted(
        v["volume_um3"] for v in sidecar["volumes"].values() if v["emitted"]
    )
    assert volumes == pytest.approx(planned, rel=1e-7)


def test_the_guard_lists_every_face_of_each_solid_in_occ_order_with_area_and_centre(
    exported,
):
    """The guard lists every face of each solid in occ order with area and centre."""
    model, plan, sidecar, _, export = exported
    ids = {s.solid_id for s in export.solids}
    assert ids == {v for v, rec in sidecar["volumes"].items() if rec["emitted"]}
    for solid in export.solids:
        planned_faces = {
            fid
            for fid, f in sidecar["faces"].items()
            if solid.solid_id in (f["negative_volume"], f["positive_volume"])
            and f["negative_volume"] != f["positive_volume"]
        }  # an embedded sheet has the same volume on both sides and bounds no shell
        assert {f["face"] for f in solid.faces} == planned_faces
        assert all(f["area_um2"] > 0 and len(f["centre_um"]) == 3 for f in solid.faces)
    assert sum(len(s.faces) for s in export.solids) == sum(
        len(plan.found.shells[i]) for i in range(len(plan.found.shells))
    )


def written(model, folder):
    """BREP and sidecar of a model, written into ``folder``."""
    plan = plan_brep(model.topology, model.routed)
    write_brep_file(plan, folder / "m.brep")
    sidecar = build_sidecar(
        model, plan, SidecarContext(sliver_report(model)["summary"])
    )
    return folder / "m.brep", sidecar


def same_lattice_models(grid_sizes_um):
    """One lattice at several physical grid sizes: the same ids and counts, different lengths."""
    catalogue = slab_model("B", [("pad", "P", (-4, -4, 4, 4))], half=10)
    return [
        build_model_from_catalogue(replace(catalogue, grid=Grid(size)), "B")
        for size in grid_sizes_um
    ]


def test_a_brep_of_the_same_topology_at_another_scale_is_refused(tmp_path):
    """Same counts, different size: the sidecar's expected positions, areas and volumes name the mismatch."""
    fine, coarse = same_lattice_models((0.001, 0.002))
    (tmp_path / "fine").mkdir()
    (tmp_path / "coarse").mkdir()
    brep, sidecar = written(fine, tmp_path / "fine")
    _, other_sidecar = written(coarse, tmp_path / "coarse")
    assert sidecar["binding"]["ids_by_tag"] == other_sidecar["binding"]["ids_by_tag"]
    assert export_aedt_step(brep, sidecar, tmp_path / "ok.step").path.is_file()
    with pytest.raises(
        StepExportError, match="does not belong to the sidecar"
    ) as refused:
        export_aedt_step(brep, other_sidecar, tmp_path / "x.step")
    for what in (
        "vertices not where",
        "face areas differing",
        "solid volumes differing",
    ):
        assert what in str(refused.value)
    assert not (tmp_path / "x.step").exists()


def test_a_brep_with_other_counts_than_the_sidecar_is_refused(
    exported, models, tmp_path
):
    """A sidecar of another model (other entity counts) is refused, naming what differs."""
    _, _, sidecar, folder, _ = exported
    plan = plan_brep(models["pad_c"].topology, models["pad_c"].routed)
    other = build_sidecar(
        models["pad_c"], plan, SidecarContext(sliver_report(models["pad_c"])["summary"])
    )
    with pytest.raises(
        StepExportError,
        match=r"does not belong to the sidecar.*(faces|solids): the shape has",
    ):
        export_aedt_step(folder / "m.brep", other, tmp_path / "x.step")
    assert not (tmp_path / "x.step").exists()
    assert sidecar["binding"]["ids_by_tag"][
        "3"
    ]  # the matching pair is what ``exported`` already exports


@pytest.mark.parametrize("name", sorted(CASES))
def test_every_case_belongs_to_its_own_sidecar_within_the_load_check_tolerances(
    tmp_path, models, name
):
    """The BREP and the XAO of every named case match the expected block: the tolerances are loose enough for arcs and cylinders."""
    model = models[name]
    brep, sidecar = written(model, tmp_path)
    plan = plan_brep(model.topology, model.routed)
    write_xao_file(plan, tmp_path / "m.xao", brep=brep)
    for path in (brep, tmp_path / "m.xao"):
        assert sidecar_mismatches(read_shape(path), sidecar) == []


def test_the_export_works_from_an_xao_too(exported, tmp_path):
    """The export works from an xao too."""
    model, plan, sidecar, folder, export = exported
    write_xao_file(plan, tmp_path / "m.xao", brep=folder / "m.brep")
    again = export_aedt_step(tmp_path / "m.xao", sidecar, tmp_path / "m.step")
    assert again.object_names == export.object_names
    shutil.rmtree(tmp_path, ignore_errors=True)


# --------------------------------------------------------------------------- larger random layouts


@pytest.mark.parametrize(("features", "route"), [(100, "A"), (100, "B"), (100, "C")])
def test_random_layouts_of_a_hundred_features_pass_every_official_check(
    tmp_path, features, route
):
    """Arcs, holes, ports and stacked metal in bulk: BRepCheck and the argument analyzer, self-interference included."""
    catalogue = random_layout(features, seed=1, route="A" if route == "A" else "B")
    model = build_model_from_catalogue(catalogue, route)
    plan = plan_brep(model.topology, model.routed)
    write_brep_file(plan, tmp_path / "m.brep")
    report = check_file(
        tmp_path / "m.brep", model.topology, plan, self_interference=True
    )
    assert report.ok, report.problems


@pytest.mark.slow
def test_a_random_layout_of_a_thousand_features_passes_the_cheap_official_checks(
    tmp_path,
):
    """The cheap checks (self-interference off, as the strict mode runs them) at N = 1000."""
    model = build_model_from_catalogue(
        random_layout(1000, seed=1, route="B"), "B", workers=2, tiles=2
    )
    plan = plan_brep(model.topology, model.routed)
    write_brep_file(plan, tmp_path / "m.brep", workers=2)
    report = check_file(
        tmp_path / "m.brep", model.topology, plan, self_interference=False
    )
    assert report.ok, report.problems
    assert (
        report.counts["solid"] == len(plan.solid_ids) and report.counts["face"] > 10_000
    )
