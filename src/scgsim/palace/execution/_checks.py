"""Input checks shared by the handoff and its run script."""

from __future__ import annotations

from typing import Any


def positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field} must be a positive integer.")
    return value


def single_line(value: str, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\n" in value
        or "\r" in value
    ):
        raise ValueError(f"{field} must be non-empty single-line text.")
    return value
