"""The process boundary and the HXT fallback, with the worker process replaced by a scripted stand-in."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import MappingProxyType

import pytest
from meshing_support import audit_of, bad_element

from scgsim.meshing import driver
from scgsim.meshing.driver import (
    ProbeTimeLimitError,
    WorkerExit,
    launch_worker,
    measure_bounding_box,
    run_round,
)
from scgsim.meshing.errors import MeshingError
from scgsim.meshing.job import Elevation, MeshJob
from scgsim.meshing.mesher import DELAUNAY, HXT, MesherOptions
from scgsim.meshing.sizing import LocalSize, SizeField, SizeRequests


def job_in(tmp_path: Path, **overrides) -> MeshJob:
    """A mesh job for the scripted worker, with optional field overrides."""
    fields = {
        "xao": tmp_path / "case.xao",
        "sidecar": tmp_path / "case.sidecar.json",
        "msh": tmp_path / "round.msh",
        "sizes": SizeRequests(
            10.0,
            1.0,
            MappingProxyType({3: 2.5, 4: 2.5, 9: 1.25}),
            (SizeField((3, 5), 5.0, 300.0),),
        ),
        "threshold": 0.01,
        "mesher": MesherOptions(algorithm_3d=HXT),
    }
    return MeshJob(**(fields | overrides))


class ScriptedWorker:
    """Stands in for ``launch_worker``: each call pops one behaviour and records the request it was given."""

    def __init__(self, *behaviours: Callable[[Path], WorkerExit]) -> None:
        self.behaviours = list(behaviours)
        self.requests: list[dict] = []
        self.timeouts: list[float | None] = []

    def __call__(self, arguments, timeout_seconds=None) -> WorkerExit:
        """Run the next scripted behaviour, recording the request of a mesh command and its time limit."""
        command, source, result = arguments
        if command == "mesh":
            self.requests.append(json.loads(Path(source).read_text(encoding="utf-8")))
            self.timeouts.append(timeout_seconds)
        return self.behaviours.pop(0)(Path(result))


def succeeds(
    audit=None, *, elevated=False, elevation_error=None
) -> Callable[[Path], WorkerExit]:
    """A worker behaviour that writes a successful result around ``audit``."""

    def behaviour(result: Path) -> WorkerExit:
        result.write_text(
            json.dumps(
                {
                    "ok": True,
                    "audit": (audit or audit_of()).to_json(),
                    "elevated": elevated,
                    "elevation_error": elevation_error,
                }
            )
        )
        return WorkerExit(0, "")

    return behaviour


def killed_by(signal: int) -> Callable[[Path], WorkerExit]:
    """A worker behaviour that dies from ``signal`` before writing a result."""
    return lambda result: WorkerExit(-signal, "munmap_chunk(): invalid pointer")


def runs_out_of_time(result: Path) -> WorkerExit:
    """A worker behaviour that was killed at its time limit."""
    return WorkerExit(-9, "", timed_out=True)


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
    assert request["mesher"] == {
        "algorithm_2d": None,
        "algorithm_3d": HXT,
        "geometry_order": 1,
        "high_order_optimize": True,
    }
    assert request["elevation"] is None
    assert request["max_size"] == 10.0 and request["gradation"] == 1.0
    assert request["curve_sizes"] == {"3": 2.5, "4": 2.5, "9": 1.25}
    assert request["fields"] == [{"curves": [3, 5], "size": 5.0, "distance": 300.0}]
    assert request["local_sizes"] == []
    assert request["threshold"] == 0.01 and request["attribute"] is False
    assert MeshJob.from_json(request) == job_in(tmp_path, attribute=False)


def test_the_request_carries_the_two_dimensional_algorithm_and_the_order(
    tmp_path, monkeypatch
):
    """The 2D algorithm, the order, the optimisation switch and the elevation target travel with the job."""
    worker = ScriptedWorker(succeeds(elevated=True))
    monkeypatch.setattr(driver, "launch_worker", worker)
    job = job_in(
        tmp_path,
        mesher=MesherOptions(
            algorithm_2d=5,
            algorithm_3d=DELAUNAY,
            geometry_order=2,
            high_order_optimize=False,
        ),
        elevation=Elevation(tmp_path / "round.elevated.msh", 1234),
    )
    run_round(job, tmp_path)
    (request,) = worker.requests
    assert request["mesher"]["algorithm_2d"] == 5
    assert request["mesher"]["geometry_order"] == 2
    assert request["mesher"]["high_order_optimize"] is False
    assert request["elevation"] == {
        "msh": str(tmp_path / "round.elevated.msh"),
        "max_tetrahedra": 1234,
        "enforce_threshold": True,
    }
    assert MeshJob.from_json(request) == job


def test_the_outcome_says_whether_the_worker_elevated_the_mesh(tmp_path, monkeypatch):
    """An elevated mesh and an elevation error both come back with the audit of the linear mesh."""
    worker = ScriptedWorker(
        succeeds(elevated=True), succeeds(elevation_error="setOrder failed")
    )
    monkeypatch.setattr(driver, "launch_worker", worker)
    elevated = run_round(job_in(tmp_path), tmp_path)
    failed = run_round(job_in(tmp_path), tmp_path)
    assert (elevated.elevated, elevated.elevation_error) == (True, None)
    assert (failed.elevated, failed.elevation_error) == (False, "setOrder failed")
    assert failed.audit is not None and failed.fallback is None


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
    assert [r["mesher"]["algorithm_3d"] for r in worker.requests] == [HXT, DELAUNAY]
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


def test_a_delaunay_round_that_fails_is_retried_once_with_hxt(tmp_path, monkeypatch):
    """The retry goes to the other algorithm, whichever was first."""
    worker = ScriptedWorker(killed_by(11), succeeds())
    monkeypatch.setattr(driver, "launch_worker", worker)
    outcome = run_round(
        job_in(tmp_path, mesher=MesherOptions(algorithm_3d=DELAUNAY)), tmp_path
    )
    assert outcome.audit is not None and outcome.algorithm_3d == HXT
    assert outcome.fallback is not None and "algorithm 1" in outcome.fallback
    assert [r["mesher"]["algorithm_3d"] for r in worker.requests] == [DELAUNAY, HXT]


def test_another_algorithm_is_not_retried(tmp_path, monkeypatch):
    """Only the two algorithms the module uses have a fallback."""
    worker = ScriptedWorker(killed_by(6))
    monkeypatch.setattr(driver, "launch_worker", worker)
    outcome = run_round(
        job_in(tmp_path, mesher=MesherOptions(algorithm_3d=4)), tmp_path
    )
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

    limits = []

    def probe(arguments, timeout_seconds=None):
        command, xao, result = arguments
        assert command == "probe" and xao.endswith("case.xao")
        limits.append(timeout_seconds)
        Path(result).write_text(
            json.dumps({"minimum": [0, 0, -10], "maximum": [3, 4, 12]})
        )
        return WorkerExit(0, "")

    monkeypatch.setattr(driver, "launch_worker", probe)
    box = measure_bounding_box(tmp_path / "case.xao", tmp_path, timeout_seconds=9.0)
    assert box.diagonal == pytest.approx((9 + 16 + 484) ** 0.5)
    assert limits == [9.0]


def test_a_probe_that_fails_names_the_file(tmp_path, monkeypatch):
    """A probe that fails names the file."""
    monkeypatch.setattr(
        driver,
        "launch_worker",
        lambda arguments, timeout_seconds=None: WorkerExit(2, "no such file"),
    )
    with pytest.raises(MeshingError, match=r"case\.xao.*status 2"):
        measure_bounding_box(tmp_path / "case.xao", tmp_path)


def test_a_probe_stopped_at_its_time_limit_raises_the_time_limit_error(
    tmp_path, monkeypatch
):
    """A probe killed at its limit, or given no time at all, raises ``ProbeTimeLimitError``."""
    started = []

    def overrun(arguments, timeout_seconds=None):
        started.append(timeout_seconds)
        return WorkerExit(-9, "", timed_out=True)

    monkeypatch.setattr(driver, "launch_worker", overrun)
    with pytest.raises(ProbeTimeLimitError, match=r"case\.xao.*0\.5 s"):
        measure_bounding_box(tmp_path / "case.xao", tmp_path, timeout_seconds=0.5)
    with pytest.raises(ProbeTimeLimitError, match="no time left"):
        measure_bounding_box(tmp_path / "case.xao", tmp_path, timeout_seconds=0.0)
    assert started == [0.5]


def test_the_local_sizes_travel_with_the_job(tmp_path, monkeypatch):
    """The repair loop's local sizes are part of the request and come back unchanged."""
    worker = ScriptedWorker(succeeds())
    monkeypatch.setattr(driver, "launch_worker", worker)
    sizes = SizeRequests(
        10.0,
        1.0,
        MappingProxyType({}),
        (),
        (LocalSize((1.0, 2.0, 3.0), 0.625), LocalSize((4.0, 5.0, 6.0), 1.25, 3.0)),
    )
    job = job_in(tmp_path, sizes=sizes)
    run_round(job, tmp_path)
    (request,) = worker.requests
    assert request["local_sizes"] == [
        {"centre": [1.0, 2.0, 3.0], "size": 0.625, "radius": None},
        {"centre": [4.0, 5.0, 6.0], "size": 1.25, "radius": 3.0},
    ]
    assert MeshJob.from_json(request) == job


def test_a_round_gets_its_time_limit_and_a_worker_killed_at_it_is_not_retried(
    tmp_path, monkeypatch
):
    """The worker is given the time left; one stopped at the limit ends the round without the fallback."""
    worker = ScriptedWorker(runs_out_of_time)
    monkeypatch.setattr(driver, "launch_worker", worker)
    outcome = run_round(job_in(tmp_path), tmp_path, timeout_seconds=42.0)
    assert worker.timeouts == [42.0]
    assert (outcome.audit, outcome.fallback, outcome.timed_out) == (None, None, True)
    assert outcome.failure is not None and "time limit" in outcome.failure


def test_the_fallback_gets_only_the_time_that_is_left(tmp_path, monkeypatch):
    """The retry with the other algorithm shares the round's limit; without a limit neither attempt has one."""
    worker = ScriptedWorker(killed_by(6), succeeds())
    monkeypatch.setattr(driver, "launch_worker", worker)
    outcome = run_round(job_in(tmp_path), tmp_path, timeout_seconds=42.0)
    assert outcome.audit is not None and not outcome.timed_out
    first, second = worker.timeouts
    assert first == 42.0 and second is not None and 0 < second <= 42.0
    unlimited = ScriptedWorker(killed_by(6), succeeds())
    monkeypatch.setattr(driver, "launch_worker", unlimited)
    run_round(job_in(tmp_path), tmp_path)
    assert unlimited.timeouts == [None, None]


def test_a_round_with_no_time_left_starts_no_worker(tmp_path, monkeypatch):
    """A limit already spent is a timed-out round, and no process is started for it."""
    worker = ScriptedWorker()
    monkeypatch.setattr(driver, "launch_worker", worker)
    outcome = run_round(job_in(tmp_path), tmp_path, timeout_seconds=0.0)
    assert outcome.timed_out and outcome.audit is None
    assert worker.requests == []


def test_launch_worker_reports_a_process_stopped_at_its_time_limit(monkeypatch):
    """``subprocess.run`` kills the worker at the limit; the exit says it timed out and keeps its stderr."""
    seen = {}

    def expire(command, **options):
        seen.update(options)
        raise subprocess.TimeoutExpired(
            command, options["timeout"], stderr="meshing 3D..."
        )

    monkeypatch.setattr(driver.subprocess, "run", expire)
    finished = launch_worker(["mesh", "request.json", "result.json"], 7.5)
    assert seen["timeout"] == 7.5
    assert finished.timed_out and finished.returncode < 0
    assert finished.stderr_tail == "meshing 3D..."
