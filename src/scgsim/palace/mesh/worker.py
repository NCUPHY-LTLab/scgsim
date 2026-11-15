"""The fresh process ``process.lower_mesh_in_worker`` starts: it reads a JSON request and writes the lowering's result as JSON."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scgsim.diagnostics import worker_diagnostics
from scgsim.palace.mesh.native import lower_mesh_in_current_process
from scgsim.refusal import found_text, internal


def main(arguments: list[str] | None = None) -> int:
    """Read one explicit request and write its JSON-serializable mesh result."""
    values = sys.argv[1:] if arguments is None else arguments
    if len(values) != 2:
        raise SystemExit("usage: python -m scgsim.palace.mesh.worker REQUEST RESULT")
    request_path, result_path = (Path(value) for value in values)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(request, Mapping):
        raise TypeError(
            internal(
                "Gmsh worker request must be a mapping"
                + ": "
                + found_text(type(request).__name__)
            )
        )
    diagnostic_config = request.get("preparation_diagnostics")
    if diagnostic_config is not None and not isinstance(diagnostic_config, Mapping):
        raise TypeError(
            internal(
                "Worker diagnostic configuration must be a mapping"
                + ": "
                + found_text(type(diagnostic_config).__name__)
            )
        )
    with worker_diagnostics(diagnostic_config):
        return _run_request(request, result_path)


def _run_request(request: Mapping[str, Any], result_path: Path) -> int:
    """Validate and execute one decoded worker request."""
    result = lower_mesh_in_current_process(
        source_mesh=Path(_required(request, "source_mesh", str)),
        mesh_path=Path(_required(request, "mesh_path", str)),
        groups=_required(request, "groups", dict),
    )
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


def _required(payload: Mapping[str, Any], key: str, expected: type[Any]) -> Any:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, expected):
        raise TypeError(
            internal(f"Gmsh worker request field {key!r} has an invalid type")
        )
    return value


if __name__ == "__main__":
    raise SystemExit(main())
