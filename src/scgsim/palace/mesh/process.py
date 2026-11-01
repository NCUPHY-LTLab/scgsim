"""Run Gmsh lowering across a private JSON process boundary; no geometry construction."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from scgsim.diagnostics import _stage, _worker_config
from scgsim.preparation import _native_preparation

from .controls import MeshSizing


def mesh_from_route_xao(  # noqa: PLR0913 -- process request preserves explicit source identities and controls
    *,
    xao_path: Path,
    route: str,
    records: Sequence[Any],
    stack: Mapping[str, Any],
    mesh_path: Path,
    refined_mesh_size: float | None,
    max_mesh_size: float | None,
    mesh_controls: Mapping[str, Any],
) -> tuple[Path, dict[str, dict[str, Any]], dict[str, Any]]:
    """Mesh an SGB XAO in one fresh Gmsh process.

    SGB has already written the exact XAO and structured sidecar in the caller.
    The fresh boundary contains only Gmsh's native XAO lowering. It changes
    process lifetime only: the XAO, structured records, exact physical mapping,
    refinement, Algorithm3D and validations remain the same mesh path for both
    routes.
    """
    sizing = MeshSizing(mesh_controls["size_policy"], refined_mesh_size, max_mesh_size)
    request = {
        "xao_path": xao_path.relative_to(mesh_path.parent).as_posix(),
        "route": route,
        "records": list(records),
        "stack": dict(stack),
        "mesh_path": mesh_path.relative_to(mesh_path.parent).as_posix(),
        **sizing.to_payload(),
        "mesh_controls": {
            key: value for key, value in mesh_controls.items() if key != "size_policy"
        },
    }
    diagnostic_config = _worker_config()
    if diagnostic_config is not None:
        request["preparation_diagnostics"] = diagnostic_config
    with _native_preparation(mesh_controls):
        result = _run_gmsh_worker(request=request, run_dir=mesh_path.parent)
    groups = result.get("groups")
    if not isinstance(groups, dict):
        raise TypeError("fresh Gmsh mesh worker result lacks structured groups.")
    if not mesh_path.is_file():
        raise FileNotFoundError("fresh Gmsh mesh worker did not write the MSH file.")
    meshing = result["meshing"]
    if not isinstance(meshing, dict):
        raise TypeError("fresh Gmsh mesh worker result lacks meshing observations.")
    return mesh_path, groups, meshing


@_stage("palace.mesh.worker_wait")
def _run_gmsh_worker(*, request: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    """Run unchanged gsim-derived XAO lowering in a fresh process."""
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
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / f"{worker_name}.stdout.log"
    stderr_path = logs / f"{worker_name}.stderr.log"
    worker_environment = _private_worker_environment()
    (logs / "private_worker_environment_keys.json").write_text(
        json.dumps(sorted(worker_environment), indent=2) + "\n", encoding="utf-8"
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
        f"logs/{worker_name}.stdout.log and logs/{worker_name}.stderr.log"
    )


def _private_worker_environment() -> dict[str, str]:
    """Return a deterministic native-library environment for private workers."""
    return {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    }
