"""Source geometry primitives retained independently of profile interpretation.

Neutral physical declarations and selected-profile arithmetic belong to
geometry.models.physical and geometry.planning.physical. This module does not
classify source bodies from route roles or author approximation semantics.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

_Z_TOLERANCE_UM = 1e-9


def record_geometry(record: Mapping[str, Any], context: str) -> dict[str, Any]:
    raw = record.get("geometry")
    if raw is None:
        geometry = dict(record)
    elif isinstance(raw, Mapping):
        geometry = dict(raw)
    else:
        raise TypeError(f"{context} geometry must be a mapping.")
    for first, second in (("z_min_um", "z_max_um"), ("z_um", "thickness_um")):
        if first in record or second in record:
            if first not in record or second not in record:
                raise ValueError(f"{context} must define both {first} and {second}.")
            geometry.setdefault(first, record[first])
            geometry.setdefault(second, record[second])
    return geometry

def geometry_z_range(geometry: Any, context: str) -> tuple[float, float]:
    if not isinstance(geometry, Mapping):
        raise TypeError(f"{context} geometry must be a mapping.")
    if "z_min_um" in geometry or "z_max_um" in geometry:
        z_min = _finite_z(geometry.get("z_min_um"), f"{context} z_min_um")
        z_max = _finite_z(geometry.get("z_max_um"), f"{context} z_max_um")
    else:
        z_min = _finite_z(geometry.get("z_um"), f"{context} z_um")
        thickness = _finite_z(geometry.get("thickness_um"), f"{context} thickness_um")
        z_max = z_min + thickness
    if z_max < z_min:
        raise ValueError(f"{context} has a negative Z extent.")
    return z_min, z_max

def same_z(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=_Z_TOLERANCE_UM)

def _finite_z(value: Any, field: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field} must be a finite number.")
    return float(value)
