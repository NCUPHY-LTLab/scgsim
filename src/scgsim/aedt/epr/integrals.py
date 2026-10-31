"""Typed native field integrals and their raw-result payload.

Owns: decoding a persisted expression purpose into an `IntegralTarget`, checking
the unit of one native integral, and storing it into the raw dictionary that
`combine_epr_mode` consumes.
Does not own: authoring or caching the expressions (`cache.py`), reading their
traces (`history.py`), or the energy arithmetic (`analysis.py`).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from scgsim.aedt._data import _freeze_str_keys_only, detached
from scgsim.aedt.runtime.native.common import detached_data

# Which thin-film field sides the surface EPR (energy participation ratio) integrates.
# Surfaces are the MA (metal-air), MS (metal-substrate) and SA (substrate-air)
# interfaces of docs/eigenmode-epr.qmd; the sidewall is left out of the integral.
_SURFACE_ANALYSIS_SCOPE = {
    "included_field_sides": ["top", "bottom"],
    "excluded_field_sides": ["sidewall"],
    "excluded_sidewall_energy": "not_evaluated_not_zero",
    "q2d_sidewall_correction": "not_applied",
}

# Each persisted expression purpose starts with one of these kinds, followed by
# the id of the volume, surface group, native face reference or junction it covers.
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
    """Private, immutable interpretation of one persisted expression binding."""

    kind: IntegralKind
    scope_id: str
    selection: Mapping[str, Any]


def _integral_target(purpose: str, selection: Mapping[str, Any]) -> IntegralTarget:
    """Decode a persisted `<kind>_<id>` purpose and validate its selection.

    The kind is parsed from the purpose string only; it is never written back into
    the persisted expression identity.
    """

    kind = next(
        (item for item in _INTEGRAL_KINDS if purpose.startswith(f"{item}_")),
        None,
    )
    if kind is None:
        raise RuntimeError(f"unknown EPR expression purpose {purpose!r}")
    if not isinstance(selection, Mapping):
        raise TypeError("EPR expression selection must be a mapping")
    # Each kind is bound to one selection shape and the key naming its owner.
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
        # Magnetic energy is weighted by 1/mu_r later, so mu_r must be usable.
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
    return IntegralTarget(kind, scope_id, _freeze_str_keys_only(selection))


def _canonical_integral(
    scalar: dict[str, Any], *, target: IntegralTarget, purpose: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Check one native scalar against the SI unit its recipe produces.

    CLC is the AEDT fields-calculator language. The scale factor is always 1: the
    recipes already integrate in SI, so only the unit label is verified.
    """

    expected_unit = _purpose_unit(target.kind)
    native_unit = scalar["unit"]
    suffix = "" if native_unit is None else str(native_unit).strip()
    number = float(scalar["value"])
    if not math.isfinite(number):
        raise RuntimeError("native integral value is nonfinite")
    # A blank suffix is not native unit verification: the scale is the SI
    # dimension of this exact E/H/geometry integral recipe, never a generic
    # assumption about arbitrary calculator values. The evidence records which
    # of the two bindings applied.
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


def _store_integral(
    raw: dict[str, Any],
    *,
    target: IntegralTarget,
    value: dict[str, Any],
    expression_identity: Mapping[str, Any] | None = None,
) -> None:
    """File one canonical integral into the raw payload slot for its kind."""

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
        _record_surface_group_provenance(raw, target)
        record = raw["surface_integrals_v2"].setdefault(target.scope_id, {})
        component = "normal" if target.kind == "electric_normal" else "tangential"
        record[component] = value
    elif target.kind == "masked_area":
        _record_surface_group_provenance(raw, target)
        raw["masked_areas_m2"][target.scope_id] = value
    elif target.kind in _NATIVE_SURFACE_COMPONENT:
        _store_native_surface_integral(raw, target, value, expression_identity)
    elif target.kind in {"junction_voltage_real", "junction_voltage_imag"}:
        record = raw["junction_integrals_v_m"].setdefault(target.scope_id, {})
        component = "real" if target.kind == "junction_voltage_real" else "imag"
        record[component] = value
    else:
        raise AssertionError(f"unsupported integral kind {target.kind!r}")


def _record_surface_group_provenance(
    raw: dict[str, Any], target: IntegralTarget
) -> None:
    """Keep one selection per surface group; the three integrals must agree."""

    key = target.scope_id
    selection = detached(target.selection)
    if (
        key in raw["surface_group_provenance"]
        and raw["surface_group_provenance"][key] != selection
    ):
        raise RuntimeError("surface group expression provenance differs")
    raw["surface_group_provenance"][key] = selection


_NATIVE_SURFACE_COMPONENT: dict[IntegralKind, str] = {
    "native_surface_reference_normal": "normal",
    "native_surface_reference_tangential": "tangential",
    "native_surface_reference_area": "area",
}


def _store_native_surface_integral(
    raw: dict[str, Any],
    target: IntegralTarget,
    value: dict[str, Any],
    expression_identity: Mapping[str, Any] | None,
) -> None:
    """Store one component of a native-face-reference surface (Solid modeling)."""

    key = target.scope_id
    selection = detached(target.selection)
    references = raw["native_surface_references"]
    record = references.setdefault(
        key, {"selection": selection, "expression_identities": {}}
    )
    if record["selection"] != selection:
        raise RuntimeError("native surface reference provenance differs")
    component = _NATIVE_SURFACE_COMPONENT[target.kind]
    if component in record:
        raise RuntimeError("native surface reference integral repeats")
    if expression_identity is None:
        raise RuntimeError("native surface reference lacks expression identity")
    record[component] = value
    record["expression_identities"][component] = detached(expression_identity)


def _attach_saved_native_surface_membership(
    raw: dict[str, Any],
    result_row: dict[str, Any],
    references: list[dict[str, Any]],
) -> None:
    """Attach saved-project face membership beside, not inside, expression data.

    Both the raw payload and the result row must name only references that were
    saved, so a missing reference is an error rather than a skipped entry.
    """

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


def _purpose_unit(kind: IntegralKind) -> str:
    """SI unit of the integral that one recipe kind evaluates."""

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


def _empty_raw_integrals() -> dict[str, Any]:
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
