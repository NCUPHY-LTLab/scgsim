"""Read sealed JSON, hashes and confined paths without interpreting solver physics."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, overload


def expect_mapping(value: Any, name: str) -> dict[str, Any]:
    """Require a JSON object before reading its named artifact fields."""
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object.")
    return value


def extract_hash_map(
    entries: list[dict[str, Any]],
    label: str = "hash entries",
    *,
    require_optional_values: bool = False,
) -> dict[str, tuple[int | None, str | None]]:
    """Index recorded file sizes and SHA-256 values, rejecting duplicate paths."""
    result: dict[str, tuple[int | None, str | None]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError(f"{label} must be mappings.")
        path = expect_scalar(entry, "path", str)
        if path in result:
            raise ValueError(f"{label} contains duplicate path {path!r}.")
        if require_optional_values:
            bytes_value = expect_int(entry.get("bytes"), optional=True)
            sha_value = expect_scalar(entry, "sha256", str, fallback=None)
        else:
            bytes_value = expect_int(entry.get("bytes"))
            sha_value = expect_scalar(entry, "sha256", str)
        result[path] = (
            bytes_value,
            sha_value,
        )
    return result


def compute_hash_entry(path: Path) -> tuple[int | None, str | None]:
    """Return the current file size and SHA-256, or the absent-file pair."""
    if not path.is_file():
        return None, None
    return path.stat().st_size, sha256(path)


def compute_input_hash_entry(root: Path, rel: str) -> tuple[int | None, str | None]:
    """Resolve one confined input path before measuring its recorded identity."""
    path = confined_path(root, rel)
    return compute_hash_entry(path)


def validate_timestamp(value: str, label: str) -> None:
    """Require a timestamp accepted by the recorded ISO-8601 contract."""
    try:
        parsed = value
        if value.endswith("Z"):
            parsed = value.replace("Z", "+00:00")
        __import__("datetime").datetime.fromisoformat(parsed)
    except Exception as exc:
        raise ValueError(f"{label} must be an ISO-8601 UTC timestamp.") from exc


def validate_contract_schema(payload: dict[str, Any], label: str) -> None:
    """Require version 1 of a handoff metadata document."""
    if expect_scalar(payload, "schema_version", int, fallback=0) != 1:
        raise ValueError(f"{label} schema_version must be integer 1.")


def read_json(path: Path) -> dict[str, Any]:
    """Load one artifact as a JSON object, retaining parse and filesystem errors."""
    payload = path.read_text(encoding="utf-8")
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return parsed


def expect_scalar(
    payload: Mapping[str, Any],
    field: str,
    _type: type,
    *,
    fallback: Any = None,
) -> Any:
    """Read a typed field with its existing absent-value fallback semantics."""
    value = payload.get(field) if isinstance(payload, dict) else fallback
    if value is None:
        return fallback
    if (_type is int and isinstance(value, bool)) or not isinstance(value, _type):
        raise TypeError(f"field {field!r} must be {_type.__name__}.")
    return value


def expect_list(payload: Any, label: str, *, allow_empty: bool = False) -> list[Any]:
    """Require an artifact array and enforce its caller-selected empty-list rule."""
    if not isinstance(payload, list):
        raise TypeError(f"{label} must be a list.")
    if not allow_empty and not payload:
        raise ValueError(f"{label} must not be empty.")
    return payload


@overload
def expect_int(value: Any, *, optional: Literal[False] = False) -> int: ...


@overload
def expect_int(value: Any, *, optional: Literal[True]) -> int | None: ...


def expect_int(value: Any, *, optional: bool = False) -> int | None:
    """Read integer-compatible JSON values while keeping booleans distinct."""
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


def to_float(value: Any) -> float:
    """Convert a recorded real number without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("value must be float-compatible.")
    return float(value)


def as_float_map(payload: Any, keys: tuple[str, ...]) -> dict[str, float | None]:
    """Project selected numeric observations, retaining missing values as None."""
    if not isinstance(payload, dict):
        return {key: None for key in keys}
    return {
        key: to_float(payload.get(key))
        if isinstance(payload.get(key), (int, float))
        else None
        for key in keys
    }


def optional_dict(value: Any) -> dict[str, Any] | None:
    """Return a JSON object unchanged; represent other shapes as unavailable."""
    return value if isinstance(value, dict) else None


def sha256(path: Path) -> str:
    """Hash the exact file bytes used to bind a Palace artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def confined_path(root: Path, relative: str) -> Path:
    """Resolve an artifact beneath the run directory and reject path escape."""
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("artifact paths must be non-empty relative paths.")
    candidate = (root / relative).resolve()
    root = root.resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("artifact path escapes the run directory.")
    return candidate
