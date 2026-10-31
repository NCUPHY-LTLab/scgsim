"""Small file helpers shared by AEDT hand-off, run and result code.

Reading, writing and hashing files only. What the JSON or CSV means belongs to the
caller; this module imports nothing from AEDT.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, read in 1 MiB chunks so large AEDT files are not loaded whole."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write indented JSON with a trailing newline, creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(
    path: Path, rows: list[Mapping[str, Any]], *, fieldnames: list[str]
) -> None:
    """Write rows under `fieldnames` only; a missing field becomes an empty cell."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
