"""Data records and scalar readers shared by Palace config preparation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any

MODEL_L0_M = 1e-6


@dataclass(frozen=True)
class TerminalBinding:
    """Exact net mapping record for a conductor terminal."""

    name: str
    net_id: str


@dataclass(frozen=True)
class ConfigBuildResult:
    """Electrostatic config and the semantic records derived from it."""

    config: dict[str, Any]
    domain_names: list[str]
    terminal_index_map: list[dict[str, Any]]
    surface_epr_index_map: list[dict[str, Any]]
    domain_energy_index_map: list[dict[str, Any]]
    material_resolution: list[dict[str, Any]]
    ground_boundary_resolution: dict[str, Any]


@dataclass(frozen=True)
class LayoutPortBinding:
    """One authored GDSFactory junction port resolved for SGB lowering."""

    index: int
    name: str
    target_layer: str
    source_layer: str
    direction: tuple[float, float, float]
    inductance: float


@dataclass(frozen=True)
class EigenmodeConfigBuildResult:
    """Eigenmode config and the semantic records derived from it."""

    config: dict[str, Any]
    port_information: list[dict[str, Any]]
    index_entries: list[dict[str, Any]]
    material_resolution: list[dict[str, Any]]


def positive_finite(value: Any) -> bool:
    """Return whether a non-boolean real number is finite and positive."""
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def physical_group_values(value: Any) -> tuple[int, ...]:
    """Read one or more integer Gmsh physical attributes."""
    raw = value if isinstance(value, (list, tuple, set)) else (value,)
    values = [
        int(item)
        for item in raw
        if isinstance(item, int) and not isinstance(item, bool)
    ]
    if not values:
        raise ValueError("Physical group must have at least one integer attribute.")
    return tuple(values)


def optional_string(value: Any) -> str | None:
    """Normalize an optional textual identifier to non-empty text or None."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "MODEL_L0_M",
    "ConfigBuildResult",
    "EigenmodeConfigBuildResult",
    "LayoutPortBinding",
    "TerminalBinding",
    "optional_string",
    "physical_group_values",
    "positive_finite",
]
