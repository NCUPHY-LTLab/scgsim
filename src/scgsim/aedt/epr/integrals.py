"""Interpret persisted field-integral bindings and canonical scalar units."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from scgsim.aedt.data import (
    detached,
    freeze,
)

# Each surface integral is of one sidecar Surface-EPR row of an SGB model (issue #67).
SURFACE_GRANULARITY = "sgb_surface_epr_row.v1"

IntegralKind = Literal[
    "effective_volume",
    "electric_volume",
    "magnetic_energy",
    "electric_normal",
    "electric_tangential",
    "masked_area",
    "junction_voltage_real",
    "junction_voltage_imag",
]

_INTEGRAL_KINDS: tuple[IntegralKind, ...] = (
    "effective_volume",
    "electric_volume",
    "magnetic_energy",
    "electric_normal",
    "electric_tangential",
    "masked_area",
    "junction_voltage_real",
    "junction_voltage_imag",
)


@dataclass(frozen=True, slots=True)
class IntegralTarget:
    """Immutable interpretation of one persisted expression binding."""

    kind: IntegralKind
    scope_id: str
    selection: Mapping[str, Any]


def integral_target(purpose: str, selection: Mapping[str, Any]) -> IntegralTarget:
    """Decode the kind and scope id of a persisted purpose once; they are never stored beside it."""

    kind = next(
        (item for item in _INTEGRAL_KINDS if purpose.startswith(f"{item}_")),
        None,
    )
    if kind is None:
        raise RuntimeError(f"unknown EPR expression purpose {purpose!r}")
    if not isinstance(selection, Mapping):
        raise TypeError("EPR expression selection must be a mapping")
    if kind in {"effective_volume", "electric_volume", "magnetic_energy"}:
        selection_kind, key = "volume", "semantic_id"
    elif kind in {"electric_normal", "electric_tangential", "masked_area"}:
        selection_kind, key = "surface_row", "row"
    else:
        selection_kind, key = "junction_sheet", "junction_id"
    scope_id = selection.get(key)
    if (
        selection.get("kind") != selection_kind
        or not isinstance(scope_id, str)
        or not scope_id
    ):
        raise RuntimeError(f"{kind} requires a {selection_kind} selection with {key}")
    if kind == "magnetic_energy":
        permeability = selection.get("relative_permeability")
        if (
            isinstance(permeability, bool)
            or not isinstance(permeability, (int, float))
            or not math.isfinite(permeability)
            or permeability <= 0
        ):
            raise RuntimeError(
                "magnetic energy requires positive relative permeability"
            )
    return IntegralTarget(cast(IntegralKind, kind), scope_id, freeze(selection))


def canonical_integral(
    scalar: dict[str, Any], *, target: IntegralTarget, purpose: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize only the dimensions of the six owned CLC integral recipes."""

    expected_unit = purpose_unit(target.kind)
    native_unit = scalar["unit"]
    suffix = "" if native_unit is None else str(native_unit).strip()
    number = float(scalar["value"])
    if not math.isfinite(number):
        raise RuntimeError("native integral value is nonfinite")
    # A blank suffix is not native unit verification: the scale is the SI
    # dimension of this exact E/H/geometry integral recipe, never a generic
    # assumption about arbitrary calculator values.
    if suffix and re.sub(r"\s+", "", suffix) != expected_unit:
        raise RuntimeError(
            f"native integral unit differs: expected {expected_unit!r}, got {native_unit!r}"
        )
    canonical = {"value": number, "unit": expected_unit}
    evidence = {
        "raw": scalar.get("raw"),
        "native_value": number,
        "native_unit": native_unit,
        "canonical_unit": expected_unit,
        "scale_factor": 1.0,
        "scale_basis": (
            f"{purpose}:field_integral_operations_or_"
            "junction_voltage_operations_SI_dimensional_recipe"
        ),
        "unit_binding": (
            "native_suffix" if suffix else "recipe_basis_native_suffix_absent"
        ),
    }
    return canonical, evidence


def store_integral(
    raw: dict[str, Any],
    *,
    target: IntegralTarget,
    value: dict[str, Any],
    expression_identity: Mapping[str, Any] | None = None,
) -> None:
    """Store a canonical integral with its expression and selection provenance."""
    if target.kind == "effective_volume":
        raw["effective_domain_volumes_m3"][target.scope_id] = value
    elif target.kind == "electric_volume":
        raw["electric_domain_integrals_v2_m"][target.scope_id] = value
    elif target.kind == "magnetic_energy":
        raw["magnetic_domain_integrals_a2_m"][target.scope_id] = value
        raw["relative_permeability"][target.scope_id] = {
            "value": target.selection["relative_permeability"],
            "unit": "1",
        }
    elif target.kind in {"electric_normal", "electric_tangential"}:
        _record_surface_group(raw, target)
        key = target.scope_id
        record = raw["surface_integrals_v2"].setdefault(key, {})
        component = "normal" if target.kind == "electric_normal" else "tangential"
        record[component] = value
    elif target.kind == "masked_area":
        _record_surface_group(raw, target)
        key = target.scope_id
        raw["masked_areas_m2"][key] = value
    elif target.kind in {"junction_voltage_real", "junction_voltage_imag"}:
        record = raw["junction_integrals_v_m"].setdefault(target.scope_id, {})
        component = "real" if target.kind == "junction_voltage_real" else "imag"
        record[component] = value
    else:
        raise AssertionError(f"unsupported integral kind {target.kind!r}")


def _record_surface_group(raw: dict[str, Any], target: IntegralTarget) -> None:
    key = target.scope_id
    selection = detached(target.selection)
    if (
        key in raw["surface_group_provenance"]
        and raw["surface_group_provenance"][key] != selection
    ):
        raise RuntimeError("surface group expression provenance differs")
    raw["surface_group_provenance"][key] = selection


def purpose_unit(kind: IntegralKind) -> str:
    """Return the canonical SI unit for one persisted integral kind."""
    if kind == "effective_volume":
        return "m^3"
    if kind == "electric_volume":
        return "V^2*m"
    if kind == "magnetic_energy":
        return "A^2*m"
    if kind in {"electric_normal", "electric_tangential"}:
        return "V^2"
    if kind == "masked_area":
        return "m^2"
    if kind in {"junction_voltage_real", "junction_voltage_imag"}:
        return "V*m"
    raise AssertionError(f"unsupported integral kind {kind!r}")


def empty_raw_integrals() -> dict[str, Any]:
    """Return the canonical recombination payload before native values are added."""

    return {
        "effective_domain_volumes_m3": {},
        "electric_domain_integrals_v2_m": {},
        "magnetic_domain_integrals_a2_m": {},
        "relative_permeability": {},
        "surface_integrals_v2": {},
        "masked_areas_m2": {},
        "surface_granularity": SURFACE_GRANULARITY,
        "surface_group_provenance": {},
        "junction_integrals_v_m": {},
    }
