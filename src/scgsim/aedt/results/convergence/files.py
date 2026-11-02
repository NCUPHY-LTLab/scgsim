"""Read and identify contained native convergence artifacts without solver interpretation."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from scgsim.aedt.io import file_sha256


class EvidenceSource(TypedDict):
    """The file identity serialized with native convergence evidence."""

    path: str
    bytes: int
    sha256: str


def evidence_source(path: Path, root: Path) -> EvidenceSource:
    """Bind a native evidence file to its relative path, exact bytes and content hash."""
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def contained_evidence_path(root: Path, relative: str, solver: str) -> Path:
    """Resolve one native evidence path within its declared result directory."""
    if not relative or Path(relative).is_absolute():
        raise RuntimeError(f"{solver} native evidence path is invalid")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise RuntimeError(f"{solver} native evidence escapes the run directory")
    return path


def read_evidence_text(path: Path, solver: str) -> str:
    """Require a nonempty UTF-8 native convergence artifact."""
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"{solver} native convergence evidence is missing: {path}")
    try:
        return path.read_text(encoding="utf-8", errors="strict")
    except UnicodeError as exc:
        raise RuntimeError(
            f"{solver} native convergence evidence is not UTF-8: {path}"
        ) from exc
