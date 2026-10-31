"""Text validation shared by the AEDT runtime modules.

Owns the one non-empty-text check used on handoff and spec fields at run time.
It does not strip: unlike ``scgsim.aedt._data._stripped_text`` it returns the
value exactly as given, so whitespace-only text passes.
"""

from __future__ import annotations

from typing import Any


def unstripped_nonempty_text(value: Any, field: str) -> str:
    """Return ``value`` unchanged if it is a non-empty string, else raise ValueError."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be non-empty text")
    return value
