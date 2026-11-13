"""Shared staged simulation primitives for Palace candidates."""

from __future__ import annotations

import math
from typing import Any, Literal, cast

RouteAThinFilm = Literal["substrate_face", "metal_gap_equivalent"]


def validate_nonempty_string(value: Any, field: str) -> str:
    """Return a non-empty trimmed string."""
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string.")
    text = value.strip()
    if not text:
        raise ValueError(f"{field} must be non-empty.")
    return text


def validate_positive_number(value: Any, field: str) -> float:
    """Validate finite positive real values used by the mesh controls."""
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field} must be a finite number.")
    if float(value) <= 0.0:
        raise ValueError(f"{field} must be > 0.")
    return float(value)


def validate_non_negative_int(value: Any, field: str) -> int:
    """Validate a non-negative integer control."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field} must be an int.")
    if value < 0:
        raise ValueError(f"{field} must be >= 0.")
    return int(value)


def normalize_route_a_thin_film(route: str, value: str | None) -> RouteAThinFilm | None:
    """Validate the explicit Route-A thin-film lowering choice."""
    if route == "B":
        if value is not None:
            raise ValueError("Route B does not accept route_a_thin_film.")
        return None
    if route != "A":
        raise ValueError("route must be either 'A' or 'B'.")
    normalized = value if value in {"substrate_face", "metal_gap_equivalent"} else None
    if normalized is not None:
        return cast(RouteAThinFilm, normalized)
    raise ValueError(
        "Route A requires route_a_thin_film='substrate_face' or 'metal_gap_equivalent'."
    )
