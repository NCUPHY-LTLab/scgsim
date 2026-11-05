"""Runs one meshing round in a fresh worker process, retrying once with the other 3D algorithm if it fails.

Owns the process boundary: it writes the request, launches ``scgsim.meshing.worker`` and reads the result.
It imports no Gmsh.  A round that returns a mesh is a :class:`RoundOutcome` with the audit; a round that
cannot is one with ``audit=None`` and the reason.  Whether that ends the repair loop is the loop's decision.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from .errors import MeshingError
from .findings import MeshAudit
from .job import MeshJob
from .mesher import DELAUNAY, HXT
from .sizing import BoundingBox

_STDERR_TAIL_CHARS = 400
_OTHER_ALGORITHM = {HXT: DELAUNAY, DELAUNAY: HXT}


@dataclass(frozen=True)
class WorkerExit:
    """How a worker process ended: its exit status (negative for a signal) and the end of its stderr."""

    returncode: int
    stderr_tail: str


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


def measure_bounding_box(xao: Path, work_dir: Path) -> BoundingBox:
    """Bounding box of the model in ``xao``, read in a worker process so this process never loads Gmsh."""
    work_dir.mkdir(parents=True, exist_ok=True)
    result_path = work_dir / "probe.json"
    result_path.unlink(missing_ok=True)
    finished = launch_worker(["probe", str(xao), str(result_path)])
    if finished.returncode != 0 or not result_path.is_file():
        raise MeshingError(
            f"could not read the model extent of {xao}: {_describe(finished)}"
        )
    raw = json.loads(result_path.read_text(encoding="utf-8"))
    return BoundingBox(tuple(raw["minimum"]), tuple(raw["maximum"]))


def run_round(job: MeshJob, work_dir: Path) -> RoundOutcome:
    """Mesh ``job`` once; if the worker exits on a signal or reports an error, retry once with the other algorithm.

    HXT is the one seen to abort the whole process (heap corruption), Delaunay is the one that answers
    when it does; a Delaunay failure is retried with HXT.  The fallback is recorded in the outcome, so a
    report can say which algorithm made the final mesh.
    """
    started = time.perf_counter()
    attempt = _attempt(job, work_dir)
    fallback = None
    if attempt.audit is None and job.mesher.algorithm_3d in _OTHER_ALGORITHM:
        fallback = attempt.failure
        other = _OTHER_ALGORITHM[job.mesher.algorithm_3d]
        job = replace(job, mesher=replace(job.mesher, algorithm_3d=other))
        attempt = _attempt(job, work_dir)
    return RoundOutcome(
        audit=attempt.audit,
        algorithm_3d=job.mesher.algorithm_3d,
        fallback=fallback,
        failure=attempt.failure,
        wall_seconds=time.perf_counter() - started,
        elevated=attempt.elevated,
        elevation_error=attempt.elevation_error,
    )


def launch_worker(arguments: Sequence[str]) -> WorkerExit:
    """Start ``python -m scgsim.meshing.worker`` with ``arguments`` and wait for it.

    A separate function so a test can stand in for the process (a crash is otherwise hard to provoke).
    """
    completed = subprocess.run(
        [sys.executable, "-m", "scgsim.meshing.worker", *arguments],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    return WorkerExit(completed.returncode, completed.stderr[-_STDERR_TAIL_CHARS:])


@dataclass(frozen=True)
class _Attempt:
    """What one worker process came back with: an audit, or the reason there is none."""

    audit: MeshAudit | None
    failure: str | None = None
    elevated: bool = False
    elevation_error: str | None = None


def _attempt(job: MeshJob, work_dir: Path) -> _Attempt:
    """One worker process: the audit of its mesh, or the reason it made none."""
    work_dir.mkdir(parents=True, exist_ok=True)
    request_path = work_dir / "request.json"
    result_path = work_dir / "result.json"
    result_path.unlink(missing_ok=True)
    request_path.write_text(json.dumps(job.to_json()), encoding="utf-8")
    finished = launch_worker(["mesh", str(request_path), str(result_path)])
    algorithm = job.mesher.algorithm_3d
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
