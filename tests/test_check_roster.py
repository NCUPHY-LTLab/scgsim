"""The nightly roster check fails when any roster layout fails a stage, and only then (no PDK needed); and the
Driven case builds and passes ``simulation_drc`` as the nightly check runs it (PDK)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# The roster lives in scripts/; CI's pdk-tests job runs `python -I`, which puts neither it nor the checkout on
# sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import check_roster  # noqa: E402
from check_roster import STAGES, failed_cases  # noqa: E402
from roster_cases import (  # noqa: E402
    ROSTER,
    configured_simulation,
    geometry_plan,
    prepare_plan,
    simulation_class,
    surface_epr_specs,
)

from scgsim.palace import DrivenSim  # noqa: E402


def _record(case: str, **results: str) -> check_roster.CaseRecord:
    stages: dict[str, check_roster.StageRecord] = {
        stage: {"result": result, "seconds": 1.0} for stage, result in results.items()
    }
    return {
        "case": case,
        "simulation": "electrostatic",
        "stages": stages,
        "seconds": 3.0,
    }


PASSED = {stage: "pass" for stage in STAGES}


def test_a_layout_passes_only_when_every_stage_passed():
    """A failed stage, a stage left running by a dead process, or a stage that never ran fails the layout."""
    records = [
        _record("simple_pad", **PASSED),
        _record("circular_pad", build_layout="pass", simulation_drc="error"),
        _record(
            "airbridge_coupon", build_layout="pass", simulation_drc="pass", mesh="error"
        ),
        _record(
            "flip_chip_pad", build_layout="pass", simulation_drc="pass", mesh="running"
        ),
        _record("disk_qubit_default", build_layout="pass", simulation_drc="pass"),
    ]

    assert failed_cases(records) == [
        "circular_pad",
        "airbridge_coupon",
        "flip_chip_pad",
        "disk_qubit_default",
    ]
    assert failed_cases(records[:1]) == []


@pytest.mark.parametrize(
    ("mesh_result", "status"),
    [("pass", 0), ("error", 1)],
    ids=["all-pass", "one-fails"],
)
def test_the_run_fails_when_a_layout_fails_after_every_layout_ran(
    monkeypatch, tmp_path, capsys, mesh_result, status
):
    """The exit status is 1 when one layout failed, and every layout still ran and was reported."""
    ran = []

    def check_case(name, path, material_records):
        ran.append(name)
        results = {**PASSED, "mesh": mesh_result} if name == "circular_pad" else PASSED
        return _record(name, **results)

    monkeypatch.setattr(check_roster, "_activate_pdk", dict)
    monkeypatch.setattr(check_roster, "check_case", check_case)

    assert check_roster.main([str(tmp_path / "records")]) == status
    assert ran == list(check_roster.ROSTER)
    passed = len(ran) - (mesh_result != "pass")
    assert (
        capsys.readouterr().out.splitlines()[-1]
        == f"roster: {passed}/{len(ran)} passed"
    )


@pytest.mark.pdk
def test_the_driven_case_passes_simulation_drc_with_its_two_ports_as_the_nightly_check_runs_it():
    """The launcher-to-launcher case, built with its bands, passes both stages with its sheets as ports 1 and 2.

    It goes through the roster's own calls, as ``check_roster`` makes them; the nightly check also meshes it.
    """
    pytest.importorskip("gdsfactory", reason="the launcher layout needs gdsfactory")
    orcapen_sc_pdk = pytest.importorskip(
        "orcapen_sc_pdk", reason="public OrcaPen SC PDK is not installed"
    )
    orcapen_sc_pdk.activate()
    material_records = orcapen_sc_pdk.get_material_records()
    specs = surface_epr_specs(bands=True)
    layout = ROSTER["launcher_to_launcher"]()

    report = simulation_class(layout).simulation_drc(
        geometry_plan(layout, material_records, specs)
    )

    assert report.geometry.passed
    assert report.meshing.passed
    assert [line.split(":")[0] for line in report.lumped_elements] == [
        "P1 chip/o_lumped_left",
        "P2 chip/o_lumped_right",
    ]
    simulation = configured_simulation(
        layout, prepare_plan(layout, material_records, specs)
    )
    assert isinstance(simulation, DrivenSim)
