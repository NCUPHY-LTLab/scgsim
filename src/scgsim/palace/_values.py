"""Small value coercions shared across Palace mesh and preparation code."""

from __future__ import annotations

from typing import Any


def optional_stripped_string(value: Any) -> str | None:
    """Trimmed ``str(value)``, or ``None`` when ``value`` is ``None`` or blank.

    Unlike ``results/report_data.py`` this never raises: non-string values are
    stringified rather than rejected.
    """
    if value is None:
        return None
    text = str(value).strip()
    return text or None
