"""Shared file identity and JSON operations for backend preview loaders."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scgsim.geometry.models.refusals import Declared

_WHERE_PREVIEW_RUN = "the run folder the preview reads"


AEDT_RUN_FOLDER = "the AEDT run folder: fetch it again, or run the AEDT handoff again"
"""Where a refused record of an AEDT run folder is set right."""


def read_json_record(path: Path) -> dict[str, Any]:
    """Read a JSON object from one preview input file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(
            Declared(f"{path.name}", _WHERE_PREVIEW_RUN).refusal(
                "must contain a JSON object"
            )
        )
    return payload


def write_json_atomically(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one JSON record through a sibling temporary file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def confined_artifact_path(root: Path, relative: str) -> Path:
    """Resolve one relative artifact path within its run folder."""
    path = Path(relative)
    if path.is_absolute():
        raise ValueError(
            Declared("preview artifact identity", _WHERE_PREVIEW_RUN).refusal(
                f"must be relative to the run folder {root}; got {relative!r}"
            )
        )
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(
            Declared(
                f"preview artifact identity {relative!r}", _WHERE_PREVIEW_RUN
            ).refusal(f"escapes the run folder {root}")
        )
    return resolved


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def receipt_authority_sha256(path: Path) -> str:
    """Hash receipt authority while excluding its preview annotation."""
    payload = read_json_record(path)
    payload.pop("geometry_preview", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def artifact_matches(root: Path, artifact: Any) -> bool:
    """Check whether a recorded preview artifact still matches its file."""
    if not isinstance(artifact, Mapping):
        return False
    relative = artifact.get("path")
    expected = artifact.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected, str):
        return False
    path = confined_artifact_path(root, relative)
    return path.is_file() and file_sha256(path) == expected
