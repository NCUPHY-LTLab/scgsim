"""Typed reads from run JSON that has not been checked yet.

Every field of a returned run directory is untrusted until proven otherwise, so the
resolver reads it through these helpers. They fix which exception a bad field raises:
``TypeError`` for a value of the wrong type, ``ValueError`` for an empty list that must
not be empty (``_expect_list``). Callers and tests rely on that split; it is not
an accident of implementation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _expect_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object.")
    return value


def _expect_scalar(
    payload: Mapping[str, Any],
    field: str,
    _type: type,
    *,
    fallback: Any = None,
) -> Any:
    value = payload.get(field) if isinstance(payload, dict) else fallback
    if value is None:
        return fallback
    if (_type is int and isinstance(value, bool)) or not isinstance(value, _type):
        raise TypeError(f"field {field!r} must be {_type.__name__}.")
    return value


def _expect_list(payload: Any, label: str, *, allow_empty: bool = False) -> list[Any]:
    if not isinstance(payload, list):
        raise TypeError(f"{label} must be a list.")
    if not allow_empty and not payload:
        raise ValueError(f"{label} must not be empty.")
    return payload


def _expect_int(value: Any, *, optional: bool = False) -> int | None:
    if value is None:
        if optional:
            return None
        raise TypeError("value must be integer-compatible.")
    if isinstance(value, bool):
        raise TypeError("value must be integer-compatible.")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise TypeError("value must be integer-compatible.")


def _optional_dict(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _to_float(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("value must be float-compatible.")
    return float(value)


def _as_float_map(payload: Any, keys: tuple[str, ...]) -> dict[str, float | None]:
    if not isinstance(payload, dict):
        return {key: None for key in keys}
    return {
        key: _to_float(payload.get(key))
        if isinstance(payload.get(key), (int, float))
        else None
        for key in keys
    }
