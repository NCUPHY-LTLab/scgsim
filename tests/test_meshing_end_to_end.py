"""The meshing module on a real SGB build: attribution, a clean mesh per recipe, determinism, a failing repair and a second-order mesh.

Runs Gmsh in worker processes (each case takes seconds), so the whole module is marked slow.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import gdstk
import pytest
from sgb_inputs import gds_pad_in_ground
from test_sgb_band_labels import PORT_CASES, all_interface_insets

pytest.importorskip("gmsh", reason="the meshing module needs Gmsh (extra: palace)")

from scgsim.geometry.sgb.build import build_sgb_geometry  # noqa: E402
from scgsim.meshing import (  # noqa: E402
    HXT,
    Budgets,
    FeatureRecipe,
    GlobalSizeRecipe,
    MesherOptions,
    MeshRepairError,
    MeshResult,
    RepairPolicy,
    UniformBoundaryRecipe,
    driver,  # noqa: E402
    mesh_sgb,
    repair,
)
from scgsim.meshing.driver import measure_bounding_box, run_round  # noqa: E402
from scgsim.meshing.features import load_features  # noqa: E402
from scgsim.meshing.job import MeshJob  # noqa: E402

pytestmark = pytest.mark.slow

THRESHOLD = 0.01


@pytest.fixture(scope="module")
def pad_in_ground_b(tmp_path_factory):
    """The 100 um pad in ground, Route B, with bands on every surface: XAO and v2 sidecar on disk."""
    directory = tmp_path_factory.mktemp("meshing_case")
    effective = gds_pad_in_ground(directory)
    insets = all_interface_insets(effective, "B", PORT_CASES["B"][1])
    return build_sgb_geometry(
        effective.build_input,
        effective.representation,
        route="B",
        insets=insets,
        output_dir=directory,
    )


def physical_names(msh: Path) -> set[str]:
    """Names in the ``$PhysicalNames`` section of an MSH 4.1 file."""
    text = msh.read_text(encoding="utf-8")
    section = text.split("$PhysicalNames")[1].split("$EndPhysicalNames")[0]
    return set(re.findall(r'^\d+ \d+ "(.*)"$', section, flags=re.MULTILINE))


def drop_timing(report: dict) -> dict:
    """The report without its timing, which differs between runs."""
    return {key: value for key, value in report.items() if key != "timing"}


def test_the_module_meshes_the_pad_in_ground_clean_and_keeps_the_xao_groups(
    pad_in_ground_b, tmp_path
):
    """The module meshes the pad in ground clean and keeps the XAO groups."""
    result = mesh_sgb(
        pad_in_ground_b.xao, pad_in_ground_b.sidecar, tmp_path, threshold=THRESHOLD
    )
    sidecar = json.loads(pad_in_ground_b.sidecar.read_text(encoding="utf-8"))
    assert result.mesh == tmp_path / f"{pad_in_ground_b.xao.stem}.msh"
    assert result.report["stop"] == "clean"
    assert result.report["mesh"]["bad_count"] == 0
    assert result.report["mesh"]["min_quality"] >= THRESHOLD
    assert result.report["recipe"]["version"] == FeatureRecipe.version
    assert (
        result.report["mesh"]["sized_curves"] > 0
    )  # the 0.2 um film walls are sized a priori
    assert result.report["mesh"]["local_sizes"] == 0
    assert physical_names(result.mesh) >= set(sidecar["physical_groups"])
    assert json.loads(result.report_path.read_text(encoding="utf-8")) == result.report
    assert "clean after 0 refinement rounds" in result.summary_path.read_text(
        encoding="utf-8"
    )
    assert not list(tmp_path.glob(".meshing-work*"))


def test_the_sidecar_may_be_passed_decoded(pad_in_ground_b, tmp_path):
    """The sidecar may be passed decoded."""
    sidecar = json.loads(pad_in_ground_b.sidecar.read_text(encoding="utf-8"))
    result = mesh_sgb(pad_in_ground_b.xao, sidecar, tmp_path, threshold=THRESHOLD)
    assert result.report["stop"] == "clean"


def test_bad_elements_are_attributed_to_planned_ids_groups_and_feature_kinds(
    pad_in_ground_b, tmp_path
):
    # A threshold this high makes many ordinary elements "bad", which is enough to exercise the attribution.
    """Bad elements are attributed to planned ids groups and feature kinds."""
    features = load_features(pad_in_ground_b.sidecar)
    box = measure_bounding_box(pad_in_ground_b.xao, tmp_path)
    job = MeshJob(
        xao=pad_in_ground_b.xao,
        sidecar=pad_in_ground_b.sidecar,
        msh=tmp_path / "attribution.msh",
        sizes=FeatureRecipe().size_requests(features, box),
        threshold=0.5,
    )
    audit = run_round(job, tmp_path).audit
    assert audit is not None and audit.bad_count == len(audit.bad) > 0
    assert audit.bad == tuple(sorted(audit.bad, key=lambda e: e.element))
    on_curves = [e for e in audit.bad if e.location in ("point", "curve")]
    assert on_curves
    film_tags = {
        t for f in features.features if f.kind == "film_sidewall" for t in f.native_tags
    }
    for element in audit.bad:
        assert element.quality < 0.5
        assert element.edge_min <= element.edge_max
        assert len(element.sgb_ids) == len(element.points) + len(element.curves) + len(
            element.surfaces
        ) + len(element.volumes)
        assert all(
            features.binding.id_of(1, c) in element.sgb_ids for c in element.curves
        )
    for element in on_curves:
        assert element.candidates, (
            "an element on a curve or a point can be repaired through some curve"
        )
        assert element.groups, (
            "a curve or a point inherits the groups of the faces around it"
        )
    wall_elements = [e for e in on_curves if film_tags & {c.tag for c in e.candidates}]
    assert wall_elements
    assert all("film_sidewall" in e.feature_kinds for e in wall_elements)
    assert any(any(g.endswith("__sidewall") for g in e.groups) for e in wall_elements)


def test_the_conductor_outline_rule_meshes_clean_with_more_elements_near_the_conductors(
    pad_in_ground_b, tmp_path
):
    """The optional rule adds a 5 um field on the conductors' outline edges and the mesh stays clean."""
    plain = mesh_sgb(
        pad_in_ground_b.xao,
        pad_in_ground_b.sidecar,
        tmp_path / "plain",
        threshold=THRESHOLD,
    )
    outlined = mesh_sgb(
        pad_in_ground_b.xao,
        pad_in_ground_b.sidecar,
        tmp_path / "outlined",
        threshold=THRESHOLD,
        recipe=FeatureRecipe(conductor_outline=True),
    )
    assert outlined.report["stop"] == "clean"
    assert outlined.report["recipe"]["parameters"]["conductor_outline"] is True
    assert outlined.report["mesh"]["tetrahedra"] != plain.report["mesh"]["tetrahedra"]


def test_two_runs_give_the_same_mesh_and_the_same_report(pad_in_ground_b, tmp_path):
    """Two runs give the same mesh and the same report."""
    first = mesh_sgb(
        pad_in_ground_b.xao,
        pad_in_ground_b.sidecar,
        tmp_path / "a",
        threshold=THRESHOLD,
    )
    second = mesh_sgb(
        pad_in_ground_b.xao,
        pad_in_ground_b.sidecar,
        tmp_path / "b",
        threshold=THRESHOLD,
    )
    assert first.mesh.read_bytes() == second.mesh.read_bytes()
    assert drop_timing(first.report) == drop_timing(second.report)


def test_a_repair_that_cannot_reach_the_threshold_fails_with_the_report_and_keeps_the_last_mesh(
    pad_in_ground_b, tmp_path
):
    """A repair that cannot reach the threshold fails with the report and keeps the last mesh."""
    with pytest.raises(MeshRepairError) as failure:
        mesh_sgb(
            pad_in_ground_b.xao,
            pad_in_ground_b.sidecar,
            tmp_path,
            threshold=0.3,  # far above what one repair round brings this model to
            budgets=Budgets(max_rounds=1),
        )
    report = failure.value.report
    assert report["stop"] == "rounds"
    assert report["mesh"]["bad_count"] > 0
    assert report["remaining_bad"]["count"] == report["mesh"]["bad_count"]
    assert report["remaining_bad"]["listed"][0]["groups"]
    assert (
        json.loads((tmp_path / f"{pad_in_ground_b.xao.stem}.quality.json").read_text())
        == report
    )
    assert (tmp_path / f"{pad_in_ground_b.xao.stem}.rejected.msh").is_file()
    assert not (tmp_path / f"{pad_in_ground_b.xao.stem}.msh").exists()


def test_a_rerun_leaves_only_the_mesh_its_report_describes(pad_in_ground_b, tmp_path):
    """A failed rerun removes the earlier accepted mesh, and a clean rerun removes the rejected one."""
    stem = pad_in_ground_b.xao.stem
    mesh_sgb(
        pad_in_ground_b.xao, pad_in_ground_b.sidecar, tmp_path, threshold=THRESHOLD
    )
    with pytest.raises(MeshRepairError):
        mesh_sgb(
            pad_in_ground_b.xao,
            pad_in_ground_b.sidecar,
            tmp_path,
            threshold=0.3,
            budgets=Budgets(max_rounds=1),
        )
    assert not (tmp_path / f"{stem}.msh").exists()
    assert (tmp_path / f"{stem}.rejected.msh").is_file()
    mesh_sgb(
        pad_in_ground_b.xao, pad_in_ground_b.sidecar, tmp_path, threshold=THRESHOLD
    )
    assert (tmp_path / f"{stem}.msh").is_file()
    assert not (tmp_path / f"{stem}.rejected.msh").exists()


def test_overlapping_calls_sharing_an_output_directory_keep_their_own_files(
    pad_in_ground_b, tmp_path, monkeypatch
):
    """A call that runs entirely after another's first round leaves the other's round mesh intact."""
    models = tmp_path / "models"
    models.mkdir()
    a, b = models / "a.xao", models / "b.xao"
    shutil.copyfile(pad_in_ground_b.xao, a)
    shutil.copyfile(pad_in_ground_b.xao, b)
    output = tmp_path / "out"
    real_round = repair.run_round
    inner: list[MeshResult] = []
    started_inner = []

    def round_then_another_call(job, work_dir, timeout_seconds=None):
        outcome = real_round(job, work_dir, timeout_seconds)
        # The first call runs the second one to completion after its round.
        if not started_inner:
            started_inner.append(True)
            inner.append(
                mesh_sgb(b, pad_in_ground_b.sidecar, output, threshold=THRESHOLD)
            )
        return outcome

    monkeypatch.setattr(repair, "run_round", round_then_another_call)
    outer = mesh_sgb(a, pad_in_ground_b.sidecar, output, threshold=THRESHOLD)
    assert (outer.mesh.name, inner[0].mesh.name) == ("a.msh", "b.msh")
    assert outer.mesh.read_bytes() == inner[0].mesh.read_bytes()
    assert not list(output.glob(".meshing-work*"))


def test_the_report_records_the_recipe_the_policies_and_what_the_quality_was_measured_on(
    pad_in_ground_b, tmp_path
):
    """Recipe identity, repair policy, mesher options and budgets are in the report, and so is the measured mesh."""
    report = mesh_sgb(
        pad_in_ground_b.xao,
        pad_in_ground_b.sidecar,
        tmp_path,
        threshold=THRESHOLD,
        repair=RepairPolicy(edge_fraction=0.4),
        budgets=Budgets(max_rounds=3),
        mesher=MesherOptions(algorithm_2d=5),
    ).report
    assert report["report_version"] == "scgsim.meshing.report/4"
    assert report["recipe"]["name"] == "features"
    assert report["recipe"]["parameters"]["max_size_divisor"] == 20.0
    assert report["repair"] == {"edge_fraction": 0.4}
    assert report["mesher"] == {
        "algorithm_2d": 5,
        "algorithm_3d": 1,
        "geometry_order": 1,
        "high_order_optimize": True,
    }
    assert report["budgets"] == {
        "max_rounds": 3,
        "max_tetrahedra": 2_000_000,
        "time_limit_seconds": 180.0,
        "time_target_seconds": 60.0,
        "time_reference_area_mm2": 4.0,
    }
    # The 100 um die, vacuum padding included, is far below 4 mm2: the limits are not scaled.
    assert report["time_budget"]["footprint_mm2"] < 4.0
    assert (
        report["time_budget"]["limit_seconds"],
        report["time_budget"]["target_seconds"],
    ) == (180.0, 60.0)
    assert report["timing"]["target_met"] is True
    assert report["mesh"]["measured_on"] == "linear mesh before elevation"
    assert report["mesh"]["geometry_order"] == 1


@pytest.mark.parametrize(
    ("recipe", "version"),
    [
        (GlobalSizeRecipe(), "scgsim.meshing.global-size/1"),
        (UniformBoundaryRecipe(), "scgsim.meshing.uniform-boundary/1"),
    ],
)
def test_a_reference_recipe_meshes_once_without_repair_and_is_judged_by_the_threshold(
    pad_in_ground_b, tmp_path, recipe, version
):
    """Both references are clean on the 100 um chip; a higher threshold rejects the same mesh with its report."""
    accepted = mesh_sgb(
        pad_in_ground_b.xao,
        pad_in_ground_b.sidecar,
        tmp_path / "accepted",
        threshold=THRESHOLD,
        recipe=recipe,
        budgets=Budgets(max_rounds=0),
    )
    assert accepted.report["recipe"]["version"] == version
    assert (accepted.report["stop"], accepted.report["rounds"]) == ("clean", 0)
    assert accepted.report["mesh"]["min_quality"] >= THRESHOLD
    with pytest.raises(MeshRepairError) as failure:
        mesh_sgb(
            pad_in_ground_b.xao,
            pad_in_ground_b.sidecar,
            tmp_path / "rejected",
            threshold=0.9,
            recipe=recipe,
            budgets=Budgets(max_rounds=0),
        )
    report = failure.value.report
    assert (report["stop"], report["rounds"], len(report["history"])) == (
        "rounds",
        0,
        1,
    )
    assert report["mesh"]["tetrahedra"] == accepted.report["mesh"]["tetrahedra"]
    assert report["mesh"]["bad_count"] > 0
    assert (
        tmp_path / "rejected" / f"{pad_in_ground_b.xao.stem}.rejected.msh"
    ).is_file()


def test_the_uniform_boundary_recipe_refines_every_curve_the_binding_names(
    pad_in_ground_b, tmp_path
):
    """The curve count comes from the sidecar's binding, so every curve of the XAO is sized."""
    result = mesh_sgb(
        pad_in_ground_b.xao,
        pad_in_ground_b.sidecar,
        tmp_path,
        threshold=THRESHOLD,
        recipe=UniformBoundaryRecipe(),
        budgets=Budgets(max_rounds=0),
    )
    sidecar = json.loads(pad_in_ground_b.sidecar.read_text(encoding="utf-8"))
    assert result.report["mesh"]["sized_curves"] == len(
        sidecar["binding"]["ids_by_tag"]["1"]
    )


ELEMENT_TYPES_OF_FILE = """
import json, sys
import gmsh
gmsh.initialize(["-noenv"])
gmsh.option.setNumber("General.Terminal", 0)
gmsh.open(sys.argv[1])
types = {dim: sorted(int(t) for t in gmsh.model.mesh.getElementTypes(dim)) for dim in (2, 3)}
names = [gmsh.model.getPhysicalName(d, t) for d, t in gmsh.model.getPhysicalGroups()]
print(json.dumps({"types": types, "names": names}))
"""


def read_element_types(msh: Path) -> dict:
    """Element types per dimension and physical-group names of a mesh file, read by Gmsh in its own process."""
    done = subprocess.run(
        [sys.executable, "-c", ELEMENT_TYPES_OF_FILE, str(msh)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(done.stdout)


QUADRATIC_TRIANGLE, QUADRATIC_TETRAHEDRON = 9, 11
LINEAR_TRIANGLE, LINEAR_TETRAHEDRON = 2, 4


@pytest.mark.parametrize("optimize", [True, False])
def test_a_second_order_mesh_is_elevated_after_the_linear_audit_and_keeps_its_groups(
    pad_in_ground_b, tmp_path, optimize
):
    """The delivered file has second-order tetrahedra and the XAO's groups; the report is of the linear mesh."""
    linear = mesh_sgb(
        pad_in_ground_b.xao,
        pad_in_ground_b.sidecar,
        tmp_path / "linear",
        threshold=THRESHOLD,
    )
    curved = mesh_sgb(
        pad_in_ground_b.xao,
        pad_in_ground_b.sidecar,
        tmp_path / "curved",
        threshold=THRESHOLD,
        mesher=MesherOptions(geometry_order=2, high_order_optimize=optimize),
    )
    sidecar = json.loads(pad_in_ground_b.sidecar.read_text(encoding="utf-8"))
    delivered = read_element_types(curved.mesh)
    assert delivered["types"]["3"] == [QUADRATIC_TETRAHEDRON]
    assert QUADRATIC_TRIANGLE in delivered["types"]["2"]
    assert set(delivered["names"]) >= set(sidecar["physical_groups"])
    assert read_element_types(linear.mesh)["types"]["3"] == [LINEAR_TETRAHEDRON]
    # The audit and the threshold are those of the linear mesh, which the elevation does not change.
    assert curved.report["mesh"]["measured_on"] == "linear mesh before elevation"
    assert curved.report["mesh"]["geometry_order"] == 2
    assert curved.report["mesher"]["geometry_order"] == 2
    assert curved.report["mesher"]["high_order_optimize"] is optimize
    for key in ("tetrahedra", "nodes", "min_quality", "bad_count"):
        assert curved.report["mesh"][key] == linear.report["mesh"][key]
    assert not list(tmp_path.rglob("*.elevated.msh"))
    assert not list(tmp_path.rglob(".meshing-work*"))


def test_a_rejected_mesh_is_kept_linear_whatever_order_was_asked(
    pad_in_ground_b, tmp_path
):
    """A mesh the loop rejects is never elevated: the kept file is the linear one and the report says order 1."""
    with pytest.raises(MeshRepairError) as failure:
        mesh_sgb(
            pad_in_ground_b.xao,
            pad_in_ground_b.sidecar,
            tmp_path,
            threshold=0.9,
            budgets=Budgets(max_rounds=0),
            mesher=MesherOptions(geometry_order=2),
        )
    rejected = tmp_path / f"{pad_in_ground_b.xao.stem}.rejected.msh"
    assert read_element_types(rejected)["types"]["3"] == [LINEAR_TETRAHEDRON]
    assert failure.value.report["mesh"]["geometry_order"] == 1
    assert failure.value.report["mesher"]["geometry_order"] == 2


def test_hxt_can_be_chosen_through_the_mesher_options(pad_in_ground_b, tmp_path):
    """The 3D algorithm is a mesher option: the report names it and the mesh is clean."""
    result = mesh_sgb(
        pad_in_ground_b.xao,
        pad_in_ground_b.sidecar,
        tmp_path,
        threshold=THRESHOLD,
        mesher=MesherOptions(algorithm_3d=HXT),
    )
    assert result.report["stop"] == "clean"
    assert result.report["mesh"]["algorithm_3d"] == HXT


# The benchmark vacuum of issue 124: nothing beside the die, 1000 um above and below it.
TALL_VACUUM = {"z_minus_um": 1000.0, "z_plus_um": 1000.0}


def built_case(directory: Path, **inputs):
    """``gds_pad_in_ground`` with ``inputs``, Route B, bands on every surface: the SGB artifacts."""
    effective = gds_pad_in_ground(directory, **inputs)
    insets = all_interface_insets(effective, "B", PORT_CASES["B"][1])
    return build_sgb_geometry(
        effective.build_input,
        effective.representation,
        route="B",
        insets=insets,
        output_dir=directory,
    )


@pytest.fixture(scope="module")
def one_millimetre_tall_vacuum(tmp_path_factory):
    """The 1 mm die under the benchmark vacuum: with only a global size, its film walls leave bad elements."""
    return built_case(
        tmp_path_factory.mktemp("one_mm"), die_um=1000.0, vacuum_padding=TALL_VACUUM
    )


@pytest.fixture(scope="module")
def pad_with_a_spike(tmp_path_factory):
    """The 100 um pad with a spike one grid step wide and 1.4 um long on its right edge.

    It is the shape the PDK's transmon pads had under simulation sampling (orcapen-sc-pdk issue 47): a
    face one nanometre wide at its base.
    """
    centre = 50.0
    pad = gdstk.Polygon(
        [
            (centre - 10, centre - 10),
            (centre + 10, centre - 10),
            (centre + 10, centre),
            (centre + 11.3, centre + 0.5),
            (centre + 10, centre + 0.001),
            (centre + 10, centre + 10),
            (centre - 10, centre + 10),
        ]
    )
    return built_case(tmp_path_factory.mktemp("spike"), pad_polygon=pad)


def test_local_sizes_repair_a_real_mesh_whose_recipe_sized_no_curve(
    one_millimetre_tall_vacuum, tmp_path
):
    """With only the global size the first mesh has bad elements at the film walls; local sizes clean it."""
    case = one_millimetre_tall_vacuum
    result = mesh_sgb(
        case.xao, case.sidecar, tmp_path, threshold=THRESHOLD, recipe=GlobalSizeRecipe()
    )
    report = result.report
    assert report["stop"] == "clean" and report["rounds"] >= 1
    assert report["history"][0]["bad_count"] > 0
    assert report["mesh"]["sized_curves"] == 0
    assert report["mesh"]["local_sizes"] > 0
    assert report["mesh"]["min_quality"] >= THRESHOLD
    again = mesh_sgb(
        case.xao,
        case.sidecar,
        tmp_path / "again",
        threshold=THRESHOLD,
        recipe=GlobalSizeRecipe(),
    )
    assert again.mesh.read_bytes() == result.mesh.read_bytes()
    assert drop_timing(again.report) == drop_timing(report)


def test_geometry_the_mesher_cannot_repair_stops_on_a_stall_with_its_report(
    pad_with_a_spike, tmp_path
):
    """A spike one grid step wide: the bad count does not fall for two rounds, so the loop stops there."""
    case = pad_with_a_spike
    with pytest.raises(MeshRepairError) as failure:
        mesh_sgb(
            case.xao,
            case.sidecar,
            tmp_path,
            threshold=THRESHOLD,
            recipe=GlobalSizeRecipe(),
        )
    report = failure.value.report
    counts = [record["bad_count"] for record in report["history"]]
    assert report["stop"] == "stalled"
    assert min(counts[-2:]) >= min(counts[:-2])
    assert report["rounds"] < Budgets().max_rounds
    # Every remaining element is at the spike, which runs from x = 60 to 61.3 um at y = 50 to 50.5 um.
    for element in report["remaining_bad"]["listed"]:
        x, y, _ = element["centroid"]
        assert 59.0 < x < 62.5 and 49.0 < y < 51.5
    assert (tmp_path / f"{case.xao.stem}.rejected.msh").is_file()


def test_a_probe_that_overruns_the_limit_is_stopped_and_fails_with_the_budget(
    pad_in_ground_b, tmp_path, monkeypatch
):
    """The extent probe is bounded too: a real probe takes about 0.13 s, so one given 0.03 s is stopped."""
    probes = []

    def recorded(arguments, timeout_seconds=None):
        finished = real_launch(arguments, timeout_seconds)
        probes.append((arguments[0], timeout_seconds, finished.timed_out))
        return finished

    real_launch = driver.launch_worker
    monkeypatch.setattr(driver, "launch_worker", recorded)
    started = time.perf_counter()
    with pytest.raises(MeshRepairError, match="time_limit") as failure:
        mesh_sgb(
            pad_in_ground_b.xao,
            pad_in_ground_b.sidecar,
            tmp_path,
            threshold=THRESHOLD,
            budgets=Budgets(time_limit_seconds=0.03),
        )
    elapsed = time.perf_counter() - started
    [(command, limit, timed_out)] = probes
    assert command == "probe" and 0 < limit <= 0.03 and timed_out
    assert elapsed < 5.0
    report = failure.value.report
    assert report["stop"] == "time_limit"
    # The footprint was never read, so the budget is the unscaled one.
    assert report["time_budget"] == {
        "footprint_mm2": None,
        "limit_seconds": 0.03,
        "target_seconds": 60.0,
    }
    assert (report["history"], report["mesh"]) == ([], None)
    assert not list(tmp_path.glob("*.msh"))
    summary = (tmp_path / f"{pad_in_ground_b.xao.stem}.quality.txt").read_text()
    assert "time_limit" in summary and "an unread footprint" in summary


def test_a_worker_still_running_at_its_time_limit_is_killed(pad_in_ground_b, tmp_path):
    """A real worker given a tenth of a second cannot start Gmsh, open the model and mesh it: it is stopped."""
    features = load_features(pad_in_ground_b.sidecar)
    job = MeshJob(
        xao=pad_in_ground_b.xao,
        sidecar=pad_in_ground_b.sidecar,
        msh=tmp_path / "cut.msh",
        sizes=FeatureRecipe().size_requests(
            features, measure_bounding_box(pad_in_ground_b.xao, tmp_path)
        ),
        threshold=THRESHOLD,
    )
    outcome = run_round(job, tmp_path, timeout_seconds=0.1)
    assert outcome.timed_out and outcome.audit is None and outcome.fallback is None
    assert outcome.wall_seconds < 5.0
    assert not job.msh.exists()
