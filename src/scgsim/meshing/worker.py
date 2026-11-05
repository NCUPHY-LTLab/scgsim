"""The Gmsh worker: one request in, one JSON result out, in a process of its own.

Run as ``python -m scgsim.meshing.worker mesh REQUEST RESULT`` or ``... probe XAO RESULT``.  A native crash
(HXT has been seen to abort with ``munmap_chunk(): invalid pointer``) kills only this process; the driver
sees the exit status and a missing result file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .features import load_features
from .job import MeshJob
from .native import bounding_box_of, mesh_in_current_process

_USAGE = (
    "usage: python -m scgsim.meshing.worker mesh REQUEST RESULT\n"
    "       python -m scgsim.meshing.worker probe XAO RESULT"
)


def main(arguments: list[str] | None = None) -> int:
    """Run the command named by the first argument and write its result file."""
    values = sys.argv[1:] if arguments is None else arguments
    if len(values) != 3 or values[0] not in ("mesh", "probe"):
        raise SystemExit(_USAGE)
    command, source, result_path = values[0], Path(values[1]), Path(values[2])
    result = _mesh(source) if command == "mesh" else _probe(source)
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return 0


def _mesh(request_path: Path) -> dict[str, object]:
    job = MeshJob.from_json(json.loads(request_path.read_text(encoding="utf-8")))
    features = load_features(job.sidecar)
    try:
        audit = mesh_in_current_process(job, features)
    except Exception as error:  # noqa: BLE001 -- Gmsh raises a bare Exception; the driver decides what a failure means
        return {"ok": False, "error": str(error)}
    return {"ok": True, "audit": audit.to_json()}


def _probe(xao: Path) -> dict[str, object]:
    box = bounding_box_of(xao)
    return {"minimum": list(box.minimum), "maximum": list(box.maximum)}


if __name__ == "__main__":
    raise SystemExit(main())
