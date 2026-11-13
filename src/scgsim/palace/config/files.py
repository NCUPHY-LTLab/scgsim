"""Write ``config.json`` and the metadata files beside it into the run folder, config last."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def persist_problem_files(
    *,
    metadata_files: Sequence[tuple[Path, Mapping[str, Any]]],
    config_path: Path,
    config: Mapping[str, Any],
) -> Path:
    """Replace each metadata file in order, then ``config.json`` last, each through a temporary file.

    Writing the config last means a run folder whose ``config.json`` is new has every metadata file it needs.
    """
    for path, payload in metadata_files:
        _atomic_json(path, payload)
    _atomic_json(config_path, config)
    return config_path


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


__all__ = ["persist_problem_files"]
