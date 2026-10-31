"""Validation, freezing and hashing helpers for JSON-like request data.

Standard library only, so every spec module can use it without import cycles. It
owns three jobs: small field validators, turning data into a frozen read-only copy
(`_freeze_str_keys_only`) or a plain mutable one (`detached`), and the canonical
SHA-256 that identifies a payload (`canonical_sha256`).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any


def _stripped_text(value: Any, name: str) -> str:
    """Return `value` stripped of surrounding whitespace; it must be non-empty text."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


def _number_at_least(value: Any, name: str, *, minimum: float = 0.0) -> float:
    """Return `value` as a float that is finite and at least `minimum`; bools are refused."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum:g}")
    return result


def _freeze_str_keys_only(value: Any, *, verified: Any = None) -> Any:
    """Freeze JSON-like data: mappings become read-only views, sequences tuples.

    Mapping keys must be strings, and leaves must be None, str, int, float or bool.
    `verified` is an earlier frozen copy; any part of `value` that is the very same
    object as the matching part of `verified` is returned as is, not copied.
    """
    if verified is not None and value is verified:
        return verified
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("EPR mapping keys must be strings")
        prior = verified if isinstance(verified, Mapping) else {}
        return MappingProxyType(
            {
                key: _freeze_str_keys_only(item, verified=prior.get(key))
                for key, item in value.items()
            }
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        prior = (
            verified
            if isinstance(verified, Sequence) and not isinstance(verified, (str, bytes))
            else ()
        )
        return tuple(
            _freeze_str_keys_only(
                item, verified=prior[index] if index < len(prior) else None
            )
            for index, item in enumerate(value)
        )
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"EPR data contains unsupported {type(value).__name__}")


def detached(value: Any) -> Any:
    """Return recursively detached plain containers."""
    if isinstance(value, Mapping):
        return {key: detached(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [detached(item) for item in value]
    if isinstance(value, list):
        return [detached(item) for item in value]
    return value


def canonical_sha256(value: Any) -> str:
    """SHA-256 of the sorted, compact, ASCII JSON of `value`.

    Key order and tuple-versus-list do not change the digest, so the same data
    hashes the same however it was assembled.
    """
    encoded = json.dumps(
        detached(value), ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
