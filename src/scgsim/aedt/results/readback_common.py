"""Shared strict offline checks for native readback identities."""

from __future__ import annotations

from typing import Any


def positive_unique_ids(value: Any, count: int) -> bool:
    """Check a native identity list for the expected unique positive IDs."""
    return (
        isinstance(value, list)
        and len(value) == count
        and all(
            isinstance(item, int) and not isinstance(item, bool) and item > 0
            for item in value
        )
        and len(set(value)) == count
    )
