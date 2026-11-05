"""Runs one meshing round in a fresh worker process, retrying once with Delaunay if HXT fails.

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
from .job import DELAUNAY, HXT, MeshJob
from .policy import BoundingBox

_STDERR_TAIL_CHARS = 400


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
    """Mesh ``job`` once; if HXT exits on a signal or reports an error, retry once with Delaunay.

    The fallback is recorded in the outcome, so a report can say which algorithm made the final mesh.
    """
    started = time.perf_counter()
    audit, failure = _attempt(job, work_dir)
    fallback = None
    if audit is None and job.algorithm_3d == HXT:
        fallback = failure
        job = replace(job, algorithm_3d=DELAUNAY)
        audit, failure = _attempt(job, work_dir)
    return RoundOutcome(
        audit=audit,
        algorithm_3d=job.algorithm_3d,
        fallback=fallback,
        failure=failure,
        wall_seconds=time.perf_counter() - started,
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


def _attempt(job: MeshJob, work_dir: Path) -> tuple[MeshAudit | None, str | None]:
    """One worker process: the audit of its mesh, or the reason it made none."""
    work_dir.mkdir(parents=True, exist_ok=True)
    request_path = work_dir / "request.json"
    result_path = work_dir / "result.json"
    result_path.unlink(missing_ok=True)
    request_path.write_text(json.dumps(job.to_json()), encoding="utf-8")
    finished = launch_worker(["mesh", str(request_path), str(result_path)])
    if finished.returncode != 0 or not result_path.is_file():
        return None, f"algorithm {job.algorithm_3d}: {_describe(finished)}"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not result["ok"]:
        return None, f"algorithm {job.algorithm_3d}: {result['error']}"
    return MeshAudit.from_json(result["audit"]), None


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
