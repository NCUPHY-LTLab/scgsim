"""The meshing module on a real SGB build: attribution, a clean mesh, determinism and a failing repair.

Runs Gmsh in worker processes (each case takes seconds), so the whole module is marked slow.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest
from sgb_inputs import gds_pad_in_ground
from test_sgb_band_labels import PORT_CASES, all_interface_insets

pytest.importorskip("gmsh", reason="the meshing module needs Gmsh (extra: palace)")

from scgsim.geometry.sgb.build import build_sgb_geometry  # noqa: E402
from scgsim.meshing import (  # noqa: E402
    Budgets,
    MeshRepairError,
    MeshResult,
    RuleSet,
    mesh_sgb,
    repair,
)
from scgsim.meshing.driver import measure_bounding_box, run_round  # noqa: E402
from scgsim.meshing.features import load_features  # noqa: E402
from scgsim.meshing.job import MeshJob  # noqa: E402
from scgsim.meshing.policy import size_requests  # noqa: E402

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
    assert result.report["rules"]["version"] == "scgsim.meshing.rules/1"
    assert (
        result.report["mesh"]["refined_curves"] > 0
    )  # the 0.2 um film walls are sized a priori
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
        sizes=size_requests(features, box, RuleSet()),
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
    assert any(any(g.endswith("::sidewall") for g in e.groups) for e in wall_elements)


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
        ruleset=RuleSet(conductor_outline=True),
    )
    assert outlined.report["stop"] == "clean"
    assert outlined.report["rules"]["parameters"]["conductor_outline"] is True
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
            threshold=0.9,  # beyond what any tetrahedral mesh of this model reaches
            budgets=Budgets(max_rounds=1),
        )
    report = failure.value.report
    assert report["stop"] in ("rounds", "unattributed")
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
            threshold=0.9,
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

    def round_then_another_call(job, work_dir):
        outcome = real_round(job, work_dir)
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
