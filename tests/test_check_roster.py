"""The nightly roster check fails when any roster layout fails a stage, and only then (no PDK needed)."""

from __future__ import annotations

import pytest

from scripts import check_roster
from scripts.check_roster import STAGES, failed_cases


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
