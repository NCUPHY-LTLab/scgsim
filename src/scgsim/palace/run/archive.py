"""Sibling `.tar.gz` layout for a prepared Palace run folder."""

from __future__ import annotations

from pathlib import Path

from scgsim.geometry.models.refusals import Declared
from scgsim.palace.results.artifacts import RETURNED_RUN_FOLDER


def sibling_run_archive_path(run_dir: Path) -> Path:
    """Return `{run_dir.name}.tar.gz` beside the run folder."""
    run_dir = Path(run_dir)
    return run_dir.parent / f"{run_dir.name}.tar.gz"


def sibling_run_archive_relative(run_dir: Path) -> str:
    """Return the run-folder-relative path of the sibling archive."""
    return f"../{Path(run_dir).name}.tar.gz"


def run_archive_arcname(run_dir: Path, path: Path) -> str:
    """Name a tar member so extracting the sibling archive recreates the folder."""
    return f"{Path(run_dir).name}/{path.relative_to(run_dir).as_posix()}"


def logical_tar_member_name(member_name: str, folder: str) -> str | None:
    """Map a tar member name onto a path inside the run folder.

    Every member is prefixed with the run folder name; the top-level folder member has no logical path,
    and a member outside the folder is refused.
    """
    name = member_name.replace("\\", "/").rstrip("/")
    prefix = f"{folder}/"
    if name == folder:
        return None
    if not name.startswith(prefix):
        raise ValueError(
            Declared(f"archive member {member_name!r}", RETURNED_RUN_FOLDER).refusal(
                "is outside the run folder"
            )
        )
    return name[len(prefix) :]


def resolve_run_archive_path(root: Path, relative: str) -> Path:
    """Resolve a resource-record archive path, which names the archive beside the run folder.

    The archive is not inside the run folder, so the run-folder confinement (``confined_path``) does not
    apply: the one path accepted is the sibling named after the folder.
    """
    root = Path(root).resolve()
    expected = sibling_run_archive_relative(root)
    if relative != expected:
        raise ValueError(
            Declared("handoff archive", RETURNED_RUN_FOLDER).refusal(
                f"must be named like the run folder and sit beside it (expected {expected!r}, the resource record names {relative!r})"
            )
        )
    candidate = sibling_run_archive_path(root).resolve()
    if candidate.parent != root.parent:
        raise ValueError(
            Declared(f"handoff archive {candidate}", RETURNED_RUN_FOLDER).refusal(
                f"resolves outside the folder {root.parent} that holds the run folder; replace the link with the archive itself"
            )
        )
    return candidate
