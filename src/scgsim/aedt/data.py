"""Detach immutable request data and compute canonical identities using stdlib."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any


def text_value(value: Any, name: str) -> str:
    """Read a stripped, nonempty text value."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


def number_value(value: Any, name: str, *, minimum: float = 0.0) -> float:
    """Read a finite numeric request value at its declared lower bound."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum:g}")
    return result


def freeze(value: Any, *, verified: Any = None) -> Any:
    """Recursively freeze plain data, retaining already verified values."""
    if verified is not None and value is verified:
        return verified
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("EPR mapping keys must be strings")
        prior = verified if isinstance(verified, Mapping) else {}
        return MappingProxyType(
            {key: freeze(item, verified=prior.get(key)) for key, item in value.items()}
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        prior = (
            verified
            if isinstance(verified, Sequence) and not isinstance(verified, (str, bytes))
            else ()
        )
        return tuple(
            freeze(item, verified=prior[index] if index < len(prior) else None)
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
    """Hash detached values with the canonical JSON codec."""
    encoded = json.dumps(
        detached(value), ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
