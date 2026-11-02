"""Recognize material-kind labels accepted by shared semantic projections."""

from __future__ import annotations

from typing import Final

MATERIAL_KINDS: Final = frozenset({"vacuum", "dielectric", "conductor"})


def is_supported_material_kind(value: object) -> bool:
    """Return whether a value is one of the three supported material labels."""
    return isinstance(value, str) and value in MATERIAL_KINDS


def is_vacuum_material_kind(value: object) -> bool:
    """Return whether a material label identifies vacuum."""
    return value == "vacuum"
