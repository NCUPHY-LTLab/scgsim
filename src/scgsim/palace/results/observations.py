"""Read optional Palace report observations without inventing missing values."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from scgsim.palace.results.artifacts import read_json


def mapping_max(payload: Any) -> float | None:
    """Read a recorded Max observation or scalar without inferring missing measurements."""
    if isinstance(payload, dict):
        value = payload.get("Max")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None
    if isinstance(payload, (int, float)) and not isinstance(payload, bool):
        return float(payload)
    return None


def read_optional_json(path: Path) -> dict[str, Any] | None:
    """Read a present observation file as JSON; return None only when absent."""
    if not path.is_file():
        return None
    return read_json(path)


def as_int(value: Any) -> int | None:
    """Project a finite recorded numerical observation to an integer, rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)):
        return None
    return int(value)
