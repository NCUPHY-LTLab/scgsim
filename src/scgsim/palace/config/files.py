"""Write ``config.json`` and the metadata files beside it into the run folder, config last."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from scgsim.contracts.palace_run_folder import json_text


class MetadataFile(NamedTuple):
    """One metadata file written with ``config.json``: its path in the run folder and its JSON payload."""

    relative: str
    payload: Mapping[str, Any]


def persist_problem_files(
    *,
    run_folder: Path,
    metadata_files: Sequence[MetadataFile],
    config_path: Path,
    config: Mapping[str, Any],
    removed: Sequence[str] = (),
) -> Path:
    """Remove each ``removed`` file, replace each metadata file in order, then ``config.json`` last.

    ``removed`` are run-folder files an earlier config wrote and this one does not, such as a convergence plan
    after the declaration was removed.  Each written file goes through a temporary file; writing the config
    last means a run folder whose ``config.json`` is new has every metadata file it needs and no stale one.
    """
    for relative in removed:
        (run_folder / relative).unlink(missing_ok=True)
    for metadata in metadata_files:
        _atomic_json(run_folder / metadata.relative, metadata.payload)
    _atomic_json(config_path, config)
    return config_path


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json_text(payload), encoding="utf-8")
    temporary.replace(path)


__all__ = ["MetadataFile", "persist_problem_files"]
