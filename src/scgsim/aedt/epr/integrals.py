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
from scgsim.aedt.runtime.native.common import detached_data

SURFACE_ANALYSIS_SCOPE = {
    "included_field_sides": ["top", "bottom"],
    "excluded_field_sides": ["sidewall"],
    "excluded_sidewall_energy": "not_evaluated_not_zero",
    "q2d_sidewall_correction": "not_applied",
}

IntegralKind = Literal[
    "effective_volume",
    "electric_volume",
    "magnetic_energy",
    "electric_normal",
    "electric_tangential",
    "masked_area",
    "native_surface_reference_normal",
    "native_surface_reference_tangential",
    "native_surface_reference_area",
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
    "native_surface_reference_normal",
    "native_surface_reference_tangential",
    "native_surface_reference_area",
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
    """Decode the legacy identity once; never add these fields to its wire form."""

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
        selection_kind, key = "surface_group", "group_id"
    elif kind in {
        "native_surface_reference_normal",
        "native_surface_reference_tangential",
        "native_surface_reference_area",
    }:
        selection_kind, key = "native_surface_reference", "reference_id"
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
    elif target.kind in {
        "native_surface_reference_normal",
        "native_surface_reference_tangential",
        "native_surface_reference_area",
    }:
        key = target.scope_id
        selection = detached(target.selection)
        references = raw["native_surface_references"]
        record = references.setdefault(
            key, {"selection": selection, "expression_identities": {}}
        )
        if record["selection"] != selection:
            raise RuntimeError("native surface reference provenance differs")
        component = {
            "native_surface_reference_normal": "normal",
            "native_surface_reference_tangential": "tangential",
            "native_surface_reference_area": "area",
        }[target.kind]
        if component in record:
            raise RuntimeError("native surface reference integral repeats")
        if expression_identity is None:
            raise RuntimeError("native surface reference lacks expression identity")
        record[component] = value
        record["expression_identities"][component] = detached(expression_identity)
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


def attach_saved_native_surface_membership(
    raw: dict[str, Any],
    result_row: dict[str, Any],
    references: list[dict[str, Any]],
) -> None:
    """Attach saved-project membership beside, not inside, expression data."""

    membership_by_id: dict[str, dict[str, Any]] = {}
    for reference in references:
        reference_id = reference["reference_id"]
        face_list = reference["face_list"]
        saved_membership = face_list.get("saved_membership")
        if not isinstance(saved_membership, Mapping):
            raise RuntimeError(
                f"native surface reference {reference_id!r} lacks saved membership"
            )
        membership_by_id[reference_id] = detached_data(saved_membership)

    raw_references = raw.get("native_surface_references", {})
    if not isinstance(raw_references, Mapping):
        raise TypeError("raw native surface references are invalid")
    for reference_id, record in raw_references.items():
        membership = membership_by_id.get(reference_id)
        if membership is None:
            raise RuntimeError(
                f"raw native surface reference {reference_id!r} was not saved"
            )
        record["saved_membership"] = detached_data(membership)

    result_references = result_row.get("native_surface_references", ())
    for reference in result_references:
        reference_id = reference["reference_id"]
        membership = membership_by_id.get(reference_id)
        if membership is None:
            raise RuntimeError(
                f"result native surface reference {reference_id!r} was not saved"
            )
        reference["saved_membership"] = detached_data(membership)


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
    if kind in {
        "native_surface_reference_normal",
        "native_surface_reference_tangential",
    }:
        return "V^2"
    if kind == "native_surface_reference_area":
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
        "surface_granularity": "owner_interface_margin.v1",
        "surface_group_provenance": {},
        "native_surface_references": {},
        "junction_integrals_v_m": {},
    }
