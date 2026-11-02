"""Hash AEDT files and serialize detached JSON and CSV records."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    """Hash the exact file bytes without loading the whole file in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a detached record with the established JSON layout."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON record."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(
    path: Path, rows: list[Mapping[str, Any]], *, fieldnames: list[str]
) -> None:
    """Write rows in the caller's declared column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
