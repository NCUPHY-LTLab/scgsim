"""Read, hash and locate files inside one returned run directory.

Hashes come in two shapes in the metadata: a ``{path, bytes, sha256}`` record, and the
``(bytes, sha256)`` pair this module turns it into for comparison. Missing files hash
to ``(None, None)``, so a comparison against a declared record fails instead of
raising.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, TypedDict

from ._expect import _expect_int, _expect_scalar

# (size in bytes, SHA-256 hex digest), or (None, None) for a file that is absent.
HashEntry = tuple[int | None, str | None]


class HashRecord(TypedDict):
    """One ``hashes`` / ``input_hashes`` / ``output_files`` entry in run metadata."""

    path: str
    bytes: int
    sha256: str


def _read_json(path: Path) -> dict[str, Any]:
    payload = path.read_text(encoding="utf-8")
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return parsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Paths named inside metadata are untrusted text. Resolving before the containment
# test also rejects symlinks that point out of the run directory.
def _confined_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("artifact paths must be non-empty relative paths.")
    candidate = (root / relative).resolve()
    root = root.resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("artifact path escapes the run directory.")
    return candidate


def _extract_hash_map(
    entries: list[HashRecord],
    label: str = "hash entries",
    *,
    require_optional_values: bool = False,
) -> dict[str, HashEntry]:
    result: dict[str, HashEntry] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError(f"{label} must be mappings.")
        path = _expect_scalar(entry, "path", str)
        if path in result:
            raise ValueError(f"{label} contains duplicate path {path!r}.")
        if require_optional_values:
            bytes_value = _expect_int(entry.get("bytes"), optional=True)
            sha_value = _expect_scalar(entry, "sha256", str, fallback=None)
        else:
            bytes_value = _expect_int(entry.get("bytes"))
            sha_value = _expect_scalar(entry, "sha256", str)
        result[path] = (
            bytes_value,
            sha_value,
        )
    return result


def _compute_hash_entry(path: Path) -> HashEntry:
    if not path.is_file():
        return None, None
    return path.stat().st_size, _sha256(path)


def _compute_input_hash_entry(root: Path, rel: str) -> HashEntry:
    path = _confined_path(root, rel)
    return _compute_hash_entry(path)
