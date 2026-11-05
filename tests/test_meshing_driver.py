"""The process boundary and the HXT fallback, with the worker process replaced by a scripted stand-in."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import MappingProxyType

import pytest
from meshing_support import audit_of, bad_element

from scgsim.meshing import driver
from scgsim.meshing.driver import WorkerExit, measure_bounding_box, run_round
from scgsim.meshing.errors import MeshingError
from scgsim.meshing.job import DELAUNAY, HXT, MeshJob
from scgsim.meshing.policy import SizeRequests


def job_in(tmp_path: Path, **overrides) -> MeshJob:
    """A mesh job for the scripted worker, with optional field overrides."""
    fields = {
        "xao": tmp_path / "case.xao",
        "sidecar": tmp_path / "case.sidecar.json",
        "msh": tmp_path / "round.msh",
        "sizes": SizeRequests(10.0, 1.0, MappingProxyType({3: 2.5, 4: 2.5, 9: 1.25})),
        "threshold": 0.01,
    }
    return MeshJob(**(fields | overrides))


class ScriptedWorker:
    """Stands in for ``launch_worker``: each call pops one behaviour and records the request it was given."""

    def __init__(self, *behaviours: Callable[[Path], WorkerExit]) -> None:
        self.behaviours = list(behaviours)
        self.requests: list[dict] = []

    def __call__(self, arguments) -> WorkerExit:
        """Run the next scripted behaviour, recording the request of a mesh command."""
        command, source, result = arguments
        if command == "mesh":
            self.requests.append(json.loads(Path(source).read_text(encoding="utf-8")))
        return self.behaviours.pop(0)(Path(result))


def succeeds(audit=None) -> Callable[[Path], WorkerExit]:
    """A worker behaviour that writes a successful result around ``audit``."""

    def behaviour(result: Path) -> WorkerExit:
        result.write_text(
            json.dumps({"ok": True, "audit": (audit or audit_of()).to_json()})
        )
        return WorkerExit(0, "")

    return behaviour


def killed_by(signal: int) -> Callable[[Path], WorkerExit]:
    """A worker behaviour that dies from ``signal`` before writing a result."""
    return lambda result: WorkerExit(-signal, "munmap_chunk(): invalid pointer")


def reports_error(message: str) -> Callable[[Path], WorkerExit]:
    """A worker behaviour that exits normally with Gmsh's error in the result."""

    def behaviour(result: Path) -> WorkerExit:
        result.write_text(json.dumps({"ok": False, "error": message}))
        return WorkerExit(0, "")

    return behaviour


def test_a_round_that_succeeds_returns_the_audit_without_a_fallback(
    tmp_path, monkeypatch
):
    """A round that succeeds returns the audit without a fallback."""
    worker = ScriptedWorker(succeeds(audit_of((bad_element(5, [(3, 8.0, 100.0)]),))))
    monkeypatch.setattr(driver, "launch_worker", worker)
    outcome = run_round(job_in(tmp_path), tmp_path)
    assert outcome.audit is not None and outcome.audit.bad[0].candidates[0].tag == 3
    assert (outcome.algorithm_3d, outcome.fallback, outcome.failure) == (
        HXT,
        None,
        None,
    )
    assert len(worker.requests) == 1


def test_the_request_carries_the_sizes_and_the_options(tmp_path, monkeypatch):
    """The request carries the sizes and the options."""
    worker = ScriptedWorker(succeeds())
    monkeypatch.setattr(driver, "launch_worker", worker)
    run_round(job_in(tmp_path, attribute=False), tmp_path)
    (request,) = worker.requests
    assert request["algorithm_3d"] == HXT and request["algorithm_2d"] is None
    assert request["max_size"] == 10.0 and request["gradation"] == 1.0
    assert request["curve_sizes"] == {"3": 2.5, "4": 2.5, "9": 1.25}
    assert request["threshold"] == 0.01 and request["attribute"] is False
    assert MeshJob.from_json(request) == job_in(tmp_path, attribute=False)


def test_hxt_killed_by_a_signal_is_retried_once_with_delaunay_and_the_fallback_is_recorded(
    tmp_path, monkeypatch
):
    """HXT killed by a signal is retried once with delaunay and the fallback is recorded."""
    worker = ScriptedWorker(killed_by(6), succeeds())
    monkeypatch.setattr(driver, "launch_worker", worker)
    outcome = run_round(job_in(tmp_path), tmp_path)
    assert outcome.audit is not None and outcome.failure is None
    assert outcome.algorithm_3d == DELAUNAY
    assert outcome.fallback is not None and "signal 6" in outcome.fallback
    assert "invalid pointer" in outcome.fallback
    assert [r["algorithm_3d"] for r in worker.requests] == [HXT, DELAUNAY]
    assert worker.requests[1]["curve_sizes"] == worker.requests[0]["curve_sizes"]


def test_an_hxt_error_reported_by_the_worker_also_falls_back(tmp_path, monkeypatch):
    """An HXT error reported by the worker also falls back."""
    worker = ScriptedWorker(reports_error("HXT: tetrahedralization failed"), succeeds())
    monkeypatch.setattr(driver, "launch_worker", worker)
    outcome = run_round(job_in(tmp_path), tmp_path)
    assert outcome.algorithm_3d == DELAUNAY
    assert (
        outcome.fallback is not None and "tetrahedralization failed" in outcome.fallback
    )


def test_a_nonzero_exit_without_a_result_falls_back(tmp_path, monkeypatch):
    """A nonzero exit without a result falls back."""
    worker = ScriptedWorker(lambda result: WorkerExit(1, "boom"), succeeds())
    monkeypatch.setattr(driver, "launch_worker", worker)
    outcome = run_round(job_in(tmp_path), tmp_path)
    assert outcome.fallback is not None and "status 1" in outcome.fallback


def test_when_the_fallback_fails_too_the_round_has_no_mesh_and_both_reasons(
    tmp_path, monkeypatch
):
    """When the fallback fails too the round has no mesh and both reasons."""
    worker = ScriptedWorker(killed_by(6), killed_by(11))
    monkeypatch.setattr(driver, "launch_worker", worker)
    outcome = run_round(job_in(tmp_path), tmp_path)
    assert outcome.audit is None
    assert outcome.fallback is not None and "signal 6" in outcome.fallback
    assert outcome.failure is not None and "signal 11" in outcome.failure
    assert len(worker.requests) == 2  # one retry, never more


def test_a_delaunay_round_is_not_retried(tmp_path, monkeypatch):
    """A delaunay round is not retried."""
    worker = ScriptedWorker(killed_by(6))
    monkeypatch.setattr(driver, "launch_worker", worker)
    outcome = run_round(job_in(tmp_path, algorithm_3d=DELAUNAY), tmp_path)
    assert outcome.audit is None and outcome.fallback is None
    assert len(worker.requests) == 1


def test_a_stale_result_from_an_earlier_attempt_is_not_mistaken_for_the_new_one(
    tmp_path, monkeypatch
):
    """A stale result from an earlier attempt is not mistaken for the new one."""
    (tmp_path / "result.json").write_text(
        json.dumps({"ok": True, "audit": audit_of().to_json()})
    )
    worker = ScriptedWorker(killed_by(6), killed_by(6))
    monkeypatch.setattr(driver, "launch_worker", worker)
    assert run_round(job_in(tmp_path), tmp_path).audit is None


def test_the_bounding_box_comes_back_from_a_probe_worker(tmp_path, monkeypatch):
    """The bounding box comes back from a probe worker."""

    def probe(arguments):
        command, xao, result = arguments
        assert command == "probe" and xao.endswith("case.xao")
        Path(result).write_text(
            json.dumps({"minimum": [0, 0, -10], "maximum": [3, 4, 12]})
        )
        return WorkerExit(0, "")

    monkeypatch.setattr(driver, "launch_worker", probe)
    box = measure_bounding_box(tmp_path / "case.xao", tmp_path)
    assert box.diagonal == pytest.approx((9 + 16 + 484) ** 0.5)


def test_a_probe_that_fails_names_the_file(tmp_path, monkeypatch):
    """A probe that fails names the file."""
    monkeypatch.setattr(
        driver, "launch_worker", lambda arguments: WorkerExit(2, "no such file")
    )
    with pytest.raises(MeshingError, match=r"case\.xao.*status 2"):
        measure_bounding_box(tmp_path / "case.xao", tmp_path)
