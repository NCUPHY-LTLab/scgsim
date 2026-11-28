"""The energy of a lossy surface film from its field integrals, and the unit-bearing values they are read from."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

# Vacuum permittivity is shared by surface and bulk electric-energy primitives.
VACUUM_PERMITTIVITY_F_PER_M = 8.8541878128e-12


def finite_value(value: Any, name: str, *, nonnegative: bool = True) -> float:
    """Read a finite numeric value and optionally require it to be nonnegative."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "finite non-negative" if nonnegative else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def exact_mapping(value: Any, name: str) -> Mapping[str, Any]:
    """Require a mapping with string keys while retaining the original mapping."""
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping")
    return value


def quantity(value: Any, name: str, unit: str, *, nonnegative: bool = True) -> float:
    """Validate a canonical unit-bearing quantity and return its numeric value."""
    item = exact_mapping(value, name)
    if set(item) != {"value", "unit"} or item.get("unit") != unit:
        raise ValueError(f"{name} must use canonical unit {unit!r}")
    return finite_value(item.get("value"), name, nonnegative=nonnegative)


def surface_energy_j(
    interface_kind: str,
    thickness_m: float,
    film_epsilon: float,
    substrate_epsilon: float,
    integrals: tuple[float, float],
) -> float:
    """Compute one surface energy from its normal and tangential integrals."""
    normal, tangential = integrals
    if interface_kind == "MA":
        return VACUUM_PERMITTIVITY_F_PER_M * thickness_m * normal / (2.0 * film_epsilon)
    if interface_kind == "MS":
        return (
            VACUUM_PERMITTIVITY_F_PER_M
            * thickness_m
            * substrate_epsilon**2
            * normal
            / (2.0 * film_epsilon)
        )
    if interface_kind == "SA":
        return (
            VACUUM_PERMITTIVITY_F_PER_M
            * thickness_m
            * (film_epsilon * tangential + normal / film_epsilon)
            / 2.0
        )
    raise ValueError(f"unsupported surface interface {interface_kind!r}")


def surface_energy_coefficients(
    interface_kind: str,
    thickness_m: float,
    film_epsilon: float,
    substrate_epsilon: float,
) -> tuple[float, float]:
    """Return the normal and tangential peak-phasor energy factors of one interface."""
    if interface_kind == "MA":
        return VACUUM_PERMITTIVITY_F_PER_M * thickness_m / (2.0 * film_epsilon), 0.0
    if interface_kind == "MS":
        return (
            VACUUM_PERMITTIVITY_F_PER_M
            * thickness_m
            * substrate_epsilon**2
            / (2.0 * film_epsilon),
            0.0,
        )
    if interface_kind == "SA":
        return (
            VACUUM_PERMITTIVITY_F_PER_M * thickness_m / (2.0 * film_epsilon),
            VACUUM_PERMITTIVITY_F_PER_M * thickness_m * film_epsilon / 2.0,
        )
    raise ValueError(f"unsupported surface interface {interface_kind!r}")
