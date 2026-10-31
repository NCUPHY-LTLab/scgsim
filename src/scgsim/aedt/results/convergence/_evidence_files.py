"""Reading native convergence evidence files, shared by the HFSS and Q2D/Q3D readers.

Owns path containment, strict UTF-8 reads and the source record. It does not own
parsing of the evidence text; each solver reader does that.
"""

from __future__ import annotations

from pathlib import Path

from scgsim.aedt._io import file_sha256


def source_record(path: Path, root: Path) -> dict[str, str | int]:
    """Describe an evidence file by run-relative path, size and SHA-256."""
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def contained_evidence_path(root: Path, relative: str, solver: str) -> Path:
    """Resolve ``relative`` under ``root``, refusing absolute paths and escapes."""
    if not relative or Path(relative).is_absolute():
        raise RuntimeError(f"{solver} native evidence path is invalid")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise RuntimeError(f"{solver} native evidence escapes the run directory")
    return path


def read_evidence_text(path: Path, solver: str) -> str:
    """Read a non-empty UTF-8 evidence file or raise with the solver named."""
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"{solver} native convergence evidence is missing: {path}")
    try:
        return path.read_text(encoding="utf-8", errors="strict")
    except UnicodeError as exc:
        raise RuntimeError(
            f"{solver} native convergence evidence is not UTF-8: {path}"
        ) from exc
