"""Run Palace's Gmsh lowering across a private JSON process boundary; no geometry construction."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from scgsim.contracts.palace_run_folder import LOGS_DIR, json_text
from scgsim.diagnostics import diagnostic_stage, worker_config

from .physical_groups import Groups


def lower_mesh_in_worker(
    *, source_mesh: Path, mesh_path: Path, groups: Groups, run_dir: Path
) -> dict[str, Any]:
    """Lower the module's mesh to solver attributes in one fresh Gmsh process and return its observations.

    The fresh boundary keeps Gmsh state out of the calling process.  The request carries the group dict (which
    already holds the attribute plan); the paths are relative to ``run_dir``.  Returns the worker's mapping
    with ``gmsh_version`` and ``statistics`` of the written ``palace.msh``.
    """
    request: dict[str, Any] = {
        "source_mesh": source_mesh.relative_to(run_dir).as_posix(),
        "mesh_path": mesh_path.relative_to(run_dir).as_posix(),
        "groups": groups,
    }
    diagnostic_config = worker_config()
    if diagnostic_config is not None:
        request["preparation_diagnostics"] = diagnostic_config
    result = _run_gmsh_worker(request=request, run_dir=run_dir)
    if not mesh_path.is_file():
        raise FileNotFoundError("fresh Gmsh mesh worker did not write the MSH file.")
    if not isinstance(result.get("statistics"), dict):
        raise TypeError("fresh Gmsh mesh worker result lacks mesh statistics.")
    return dict(result)


@diagnostic_stage("palace.mesh.worker_wait")
def _run_gmsh_worker(*, request: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    """Run the attribute lowering in a fresh process."""
    with TemporaryDirectory(prefix="scgsim-gmsh-worker-", dir=run_dir) as temporary:
        temporary_path = Path(temporary)
        temporary_path.chmod(0o700)
        request_path = temporary_path / "request.json"
        result_path = temporary_path / "result.json"
        request_path.write_text(json.dumps(dict(request)), encoding="utf-8")
        _run_private_worker(
            arguments=(
                sys.executable,
                "-m",
                "scgsim.palace.mesh.worker",
                str(request_path),
                str(result_path),
            ),
            run_dir=run_dir,
            worker_name="gmsh_mesh_worker",
        )
        if not result_path.is_file():
            raise FileNotFoundError("fresh Gmsh mesh worker did not write a result.")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(result, Mapping):
            raise TypeError("fresh Gmsh mesh worker result must be a mapping.")
        return result


def _run_private_worker(
    *, arguments: Sequence[str], run_dir: Path, worker_name: str
) -> None:
    """Execute one private IPC worker and retain only its diagnostic logs."""
    logs = run_dir / LOGS_DIR
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / f"{worker_name}.stdout.log"
    stderr_path = logs / f"{worker_name}.stderr.log"
    worker_environment = _private_worker_environment()
    (logs / "private_worker_environment_keys.json").write_text(
        json_text(sorted(worker_environment)), encoding="utf-8"
    )
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        completed = subprocess.run(
            list(arguments),
            check=False,
            env=worker_environment,
            shell=False,
            start_new_session=True,
            cwd=run_dir,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
    if completed.returncode == 0:
        return
    if completed.returncode < 0:
        detail = f"terminated by signal {-completed.returncode}"
    else:
        detail = f"exited with status {completed.returncode}"
    raise RuntimeError(
        f"private {worker_name} {detail}; diagnostics are in "
        f"{LOGS_DIR}/{stdout_path.name} and {LOGS_DIR}/{stderr_path.name}"
    )


def _private_worker_environment() -> dict[str, str]:
    """Return a deterministic native-library environment for private workers."""
    return {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    }
