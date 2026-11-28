"""Runs one meshing round in a fresh worker process, retrying once with the other 3D algorithm if it fails.

Owns the process boundary: it writes the request, launches ``scgsim.meshing.worker`` and reads the result.
It imports no Gmsh.  A round that returns a mesh is a :class:`RoundOutcome` with the audit; a round that
cannot is one with ``audit=None`` and the reason.  Whether that ends the repair loop is the loop's decision.
A round may be given a time limit: the worker still running when it expires is killed, and the round says
so (``timed_out``).
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from scgsim.refusal import internal, refused

from .errors import MeshingError
from .findings import MeshAudit
from .job import MeshJob
from .mesher import DELAUNAY, HXT
from .sizing import BoundingBox

_WHERE_BUDGETS = "the Budgets of the meshing call"

_STDERR_TAIL_CHARS = 400
_OTHER_ALGORITHM = {HXT: DELAUNAY, DELAUNAY: HXT}


@dataclass(frozen=True)
class WorkerExit:
    """How a worker process ended: its exit status (negative for a signal) and the end of its stderr."""

    returncode: int
    stderr_tail: str
    timed_out: bool = False  # killed because the round's time limit expired


@dataclass(frozen=True)
class RoundOutcome:
    """The result of one round, after the fallback if it was needed."""

    audit: MeshAudit | None  # None when no algorithm produced a mesh
    algorithm_3d: int  # the algorithm that produced the mesh, or the last one tried
    fallback: str | None  # why HXT was abandoned for Delaunay; None when it was not
    failure: str | None  # why no mesh exists; None when the round produced one
    wall_seconds: float
    elevated: bool = False  # the worker also wrote the elevated mesh the job asked for
    elevation_error: str | None = (
        None  # why it could not, when the linear mesh was acceptable
    )
    timed_out: bool = False  # the time limit expired before a worker finished


class ProbeTimeLimitError(MeshingError):
    """Reading the model's extent did not finish within the time it was given."""


def measure_bounding_box(
    xao: Path, work_dir: Path, timeout_seconds: float | None = None
) -> BoundingBox:
    """Bounding box of the model in ``xao``, read in a worker process so this process never loads Gmsh.

    A probe still running after ``timeout_seconds`` is killed and :class:`ProbeTimeLimitError` raised.
    """
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ProbeTimeLimitError(
            refused(
                f"the model extent of {xao}",
                "cannot be read: no time is left in the time budget",
                _WHERE_BUDGETS,
            )
        )
    work_dir.mkdir(parents=True, exist_ok=True)
    result_path = work_dir / "probe.json"
    result_path.unlink(missing_ok=True)
    finished = launch_worker(["probe", str(xao), str(result_path)], timeout_seconds)
    if finished.timed_out:
        raise ProbeTimeLimitError(
            refused(
                f"reading the model extent of {xao}",
                f"did not finish within {timeout_seconds:.3g} s",
                _WHERE_BUDGETS,
            )
        )
    if finished.returncode != 0 or not result_path.is_file():
        raise MeshingError(
            internal(f"could not read the model extent of {xao}: {_describe(finished)}")
        )
    raw = json.loads(result_path.read_text(encoding="utf-8"))
    return BoundingBox(tuple(raw["minimum"]), tuple(raw["maximum"]))


def run_round(
    job: MeshJob, work_dir: Path, timeout_seconds: float | None = None
) -> RoundOutcome:
    """Mesh ``job`` once; if the worker exits on a signal or reports an error, retry once with the other algorithm.

    HXT is the one seen to abort the whole process (heap corruption), Delaunay is the one that answers
    when it does; a Delaunay failure is retried with HXT.  The fallback is recorded in the outcome, so a
    report can say which algorithm made the final mesh.  ``timeout_seconds`` bounds the whole round, the
    retry included; a worker still running then is killed and the round has no mesh.
    """
    started = time.perf_counter()
    attempt = _attempt(job, work_dir, timeout_seconds)
    fallback = None
    if (
        attempt.audit is None
        and not attempt.timed_out
        and job.mesher.algorithm_3d in _OTHER_ALGORITHM
    ):
        fallback = attempt.failure
        other = _OTHER_ALGORITHM[job.mesher.algorithm_3d]
        job = replace(job, mesher=replace(job.mesher, algorithm_3d=other))
        attempt = _attempt(
            job, work_dir, _remaining(timeout_seconds, time.perf_counter() - started)
        )
    return RoundOutcome(
        audit=attempt.audit,
        algorithm_3d=job.mesher.algorithm_3d,
        fallback=fallback,
        failure=attempt.failure,
        wall_seconds=time.perf_counter() - started,
        elevated=attempt.elevated,
        elevation_error=attempt.elevation_error,
        timed_out=attempt.timed_out,
    )


def launch_worker(
    arguments: Sequence[str], timeout_seconds: float | None = None
) -> WorkerExit:
    """Start ``python -m scgsim.meshing.worker`` with ``arguments`` and wait for it, at most ``timeout_seconds``.

    A separate function so a test can stand in for the process (a crash is otherwise hard to provoke).
    """
    command = [sys.executable, "-m", "scgsim.meshing.worker", *arguments]
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as expired:
        # subprocess.run has already killed the worker and waited for it.
        stderr = expired.stderr or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        return WorkerExit(-signal.SIGKILL, stderr[-_STDERR_TAIL_CHARS:], timed_out=True)
    return WorkerExit(completed.returncode, completed.stderr[-_STDERR_TAIL_CHARS:])


@dataclass(frozen=True)
class _Attempt:
    """What one worker process came back with: an audit, or the reason there is none."""

    audit: MeshAudit | None
    failure: str | None = None
    elevated: bool = False
    elevation_error: str | None = None
    timed_out: bool = False


def _attempt(job: MeshJob, work_dir: Path, timeout_seconds: float | None) -> _Attempt:
    """One worker process: the audit of its mesh, or the reason it made none."""
    algorithm = job.mesher.algorithm_3d
    if timeout_seconds is not None and timeout_seconds <= 0:
        return _Attempt(None, f"algorithm {algorithm}: no time left", timed_out=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    request_path = work_dir / "request.json"
    result_path = work_dir / "result.json"
    result_path.unlink(missing_ok=True)
    request_path.write_text(json.dumps(job.to_json()), encoding="utf-8")
    finished = launch_worker(
        ["mesh", str(request_path), str(result_path)], timeout_seconds
    )
    if finished.timed_out:
        return _Attempt(
            None,
            f"algorithm {algorithm}: stopped at the time limit ({timeout_seconds:.0f} s left for the round)",
            timed_out=True,
        )
    if finished.returncode != 0 or not result_path.is_file():
        return _Attempt(None, f"algorithm {algorithm}: {_describe(finished)}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not result["ok"]:
        return _Attempt(None, f"algorithm {algorithm}: {result['error']}")
    return _Attempt(
        MeshAudit.from_json(result["audit"]),
        elevated=result["elevated"],
        elevation_error=result["elevation_error"],
    )


def _remaining(timeout_seconds: float | None, spent: float) -> float | None:
    return None if timeout_seconds is None else timeout_seconds - spent


def _describe(finished: WorkerExit) -> str:
    if finished.returncode < 0:
        how = f"worker killed by signal {-finished.returncode}"
    elif finished.returncode > 0:
        how = f"worker exited with status {finished.returncode}"
    else:
        how = "worker wrote no result"
    return (
        f"{how}: {finished.stderr_tail.strip()}"
        if finished.stderr_tail.strip()
        else how
    )
