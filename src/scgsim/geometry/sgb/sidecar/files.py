"""Reading and writing the sidecar file, and its JSON Schema.

Owns the byte format: UTF-8 JSON, keys sorted, one top-level key per line and compact values, so a
diff of two sidecars points at the section that changed.  The same dictionary always gives the same
bytes (no timestamps, no absolute paths).  The schema ships as package data next to this module.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

SCHEMA_FILE = "sidecar.schema.json"


def write_sidecar(path: Path, sidecar: dict[str, Any]) -> int:
    """Write the sidecar and return its size in bytes."""
    lines = [
        f"{json.dumps(key)}:{_compact(value)}" for key, value in sorted(sidecar.items())
    ]
    text = "{\n" + ",\n".join(lines) + "\n}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    path.write_bytes(data)  # bytes, not text: no newline translation on any platform
    return len(data)


def _compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read_sidecar(path: Path) -> dict[str, Any]:
    """The sidecar dictionary stored in ``path``."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sidecar_schema() -> dict[str, Any]:
    """The JSON Schema (draft 2020-12) of ``scgsim.sgb.sidecar.v1``."""
    text = (
        resources.files("scgsim.geometry.sgb.sidecar")
        .joinpath(SCHEMA_FILE)
        .read_text(encoding="utf-8")
    )
    return json.loads(text)
