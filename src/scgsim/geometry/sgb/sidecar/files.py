"""Writing the sidecar file.

Owns the byte format: UTF-8 JSON, keys sorted, compact values, one line per top-level field and one line per
record of each table, so a diff of two sidecars names the record that changed.  The same dictionary always
gives the same bytes (no timestamps, no absolute paths).  The schema is derived from the contract
(``scgsim.contracts.sgb_sidecar``), which also holds ``read_sidecar`` to read a sidecar back.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from scgsim.contracts.sgb_sidecar.records import Sidecar

# The fields that are a record or a value, written on one line; every other field is a table.
_SINGLE_LINE = frozenset({"schema", "model", "binding", "summary"})


def write_sidecar(path: Path, sidecar: Sidecar) -> int:
    """Write the sidecar and return its size in bytes."""
    lines = [
        f"{json.dumps(key)}:{_compact(value) if key in _SINGLE_LINE else _table(cast(Mapping[str, Any], value))}"
        for key, value in sorted(sidecar.items())
    ]
    text = "{\n" + ",\n".join(lines) + "\n}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    path.write_bytes(data)  # bytes, not text: no newline translation on any platform
    return len(data)


def _table(records: Mapping[str, Any]) -> str:
    if not records:
        return "{}"
    lines = [
        f"{json.dumps(key)}:{_compact(value)}" for key, value in sorted(records.items())
    ]
    return "{\n" + ",\n".join(lines) + "\n}"


def _compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
