"""Explicit Surface-EPR lowering for structured SGB interface records.

Owns: validating the configured surface-film specs and turning matching
structured boundary groups into Palace ``Boundaries.Postprocessing.Dielectric``
rows plus their index-map entries.  Does not own: building the structured groups
(geometry) or reading results back (``palace.results``).

Surface-EPR means the electric-field energy participation of a thin lossy film.
Interface types: MA is metal-air, MS is metal-substrate, SA is substrate-air.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any, NamedTuple

from scgsim.semantics.epr import film_assumptions

_INTERFACE_TYPES = frozenset({"MA", "MS", "SA"})
_SPEC_FIELDS = frozenset(
    {
        "thickness",
        "film_thickness_m",
        "permittivity",
        "material_id",
        "loss_tangent",
        "source",
        "preset",
        "face_kinds",
        "required",
        "inset_margins_um",
    }
)
_DIELECTRIC_SECTION = "Boundaries.Postprocessing.Dielectric"


def normalize_surface_epr_specs(
    specs: Mapping[str, Mapping[str, Any]],
    *,
    materials: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Normalize configured MA/MS/SA layers from numeric or explicit material data."""
    if not isinstance(specs, Mapping):
        raise TypeError("surface_epr specs must be a mapping.")
    unknown_types = set(specs).difference(_INTERFACE_TYPES)
    if unknown_types:
        raise ValueError(
            f"surface_epr has unsupported interface types: {sorted(unknown_types)}."
        )
    return {
        interface_type: _normalize_spec(interface_type, raw, materials)
        for interface_type, raw in sorted(specs.items())
    }


def _normalize_spec(
    interface_type: str,
    raw: Any,
    materials: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate one interface spec; checks run in the order users see errors."""
    if not isinstance(raw, Mapping):
        raise TypeError(f"Surface EPR {interface_type} spec must be a mapping.")
    unknown = set(raw).difference(_SPEC_FIELDS)
    if unknown:
        raise ValueError(
            f"Surface EPR {interface_type} has unknown fields: {sorted(unknown)}."
        )
    thickness_field = _thickness_field(interface_type, raw)
    numeric, material_id, default_loss = _resolve_permittivity(
        interface_type, raw, materials
    )
    resolved: dict[str, Any] = {}
    if material_id is not None:
        resolved["material_id"] = material_id
    if not _positive(numeric):
        raise ValueError(
            f"Surface EPR {interface_type} requires positive resolved permittivity."
        )
    loss_tangent = raw.get("loss_tangent", default_loss)
    film_assumptions({"loss_tangent": loss_tangent}, partial=True)
    source = raw.get("source", "material" if material_id is not None else "explicit")
    preset = raw.get("preset")
    film_assumptions({"source": source, "preset": preset}, partial=True)
    if raw.get("face_kinds") is not None:
        resolved["face_kinds"] = _normalize_face_kinds(
            interface_type, raw["face_kinds"]
        )
    required = raw.get("required", True)
    if not isinstance(required, bool):
        raise TypeError(f"Surface EPR {interface_type} required must be a bool.")
    if "inset_margins_um" in raw:
        resolved["inset_margins_um"] = _normalize_inset_margins(
            interface_type, raw["inset_margins_um"]
        )
    return {
        **thickness_field,
        "permittivity": float(numeric),
        "loss_tangent": None if loss_tangent is None else float(loss_tangent),
        "source": source,
        "preset": preset,
        "required": required,
        **resolved,
    }


def _thickness_field(interface_type: str, raw: Mapping[str, Any]) -> dict[str, float]:
    """Return the single thickness entry: native units or metres, never both."""
    thickness = raw.get("thickness")
    thickness_m = raw.get("film_thickness_m")
    if (thickness is None) == (thickness_m is None):
        raise ValueError(
            f"Surface EPR {interface_type} requires exactly one native thickness or film_thickness_m."
        )
    if thickness is not None and not _positive(thickness):
        raise ValueError(f"Surface EPR {interface_type} requires positive thickness.")
    if thickness_m is not None:
        film_assumptions({"film_thickness_m": thickness_m}, partial=True)
        return {"film_thickness_m": float(thickness_m)}
    return {"thickness": float(thickness)}


def _resolve_permittivity(
    interface_type: str,
    raw: Mapping[str, Any],
    materials: Mapping[str, Mapping[str, Any]],
) -> tuple[Any, str | None, Any]:
    """Return ``(permittivity, material_id, material default loss tangent)``."""
    numeric = raw.get("permittivity")
    material_id = raw.get("material_id")
    if (numeric is None) == (material_id is None):
        raise ValueError(
            f"Surface EPR {interface_type} requires exactly one of permittivity or material_id."
        )
    if material_id is None:
        return numeric, None, None
    if not isinstance(material_id, str) or not material_id.strip():
        raise ValueError(f"Surface EPR {interface_type} material_id must be non-empty.")
    material = materials.get(material_id)
    if not isinstance(material, Mapping) or material.get("kind") != "dielectric":
        raise ValueError(
            f"Surface EPR {interface_type} material_id must resolve an explicit dielectric material."
        )
    return material.get("permittivity"), material_id, material.get("loss_tangent")


def _normalize_face_kinds(interface_type: str, face_kinds: Any) -> list[str]:
    if not isinstance(face_kinds, (list, tuple)) or not face_kinds:
        raise ValueError(
            f"Surface EPR {interface_type} face_kinds must be a non-empty sequence."
        )
    normalized = sorted({str(value).strip().lower() for value in face_kinds})
    if not all(value for value in normalized):
        raise ValueError(f"Surface EPR {interface_type} face_kinds must be non-empty.")
    return normalized


def _normalize_inset_margins(interface_type: str, margins: Any) -> list[float]:
    if isinstance(margins, (str, bytes)) or not isinstance(margins, (list, tuple)):
        raise TypeError(
            f"Surface EPR {interface_type} inset_margins_um must be an ordered non-empty sequence."
        )
    if not margins:
        raise ValueError(
            f"Surface EPR {interface_type} inset_margins_um must be non-empty."
        )
    normalized = []
    for margin in margins:
        if not _nonnegative(margin):
            raise ValueError(
                f"Surface EPR {interface_type} inset_margins_um values must be finite non-negative reals."
            )
        normalized.append(float(margin))
    return normalized


class _MaskRequest(NamedTuple):
    """One inset-mask row, deferred until every baseline row has its index."""

    baseline_index: int
    name: str
    info: Mapping[str, Any]
    snapshot: Mapping[str, Any]
    attrs: list[int]
    margin_index: int
    margin_um: float


def build_surface_epr_postprocessing(
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]],
    specs: Mapping[str, Mapping[str, Any]],
    *,
    model_l0_m: float = 1e-6,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Emit Palace rows using only exact structured surface identity fields.

    Baseline (unmasked) rows come first, in interface order.  Inset-mask rows
    follow, so a baseline row's index never depends on how many masks exist.
    """
    if not _positive(model_l0_m):
        raise ValueError("Model.L0 must be a finite positive length in metres.")
    rows: list[dict[str, Any]] = []
    index_map: list[dict[str, Any]] = []
    mask_requests: list[_MaskRequest] = []
    masks_requested = any("inset_margins_um" in preset for preset in specs.values())
    for interface_type, preset in sorted(specs.items()):
        matches = _matching_groups(groups, interface_type, preset)
        if not matches and preset["required"]:
            raise ValueError(
                f"Required Surface EPR {interface_type} has no matching structured group."
            )
        snapshot = _film_snapshot(preset, model_l0_m)
        for name, info in matches:
            index = len(rows) + 1
            attrs = _attributes(info.get("phys_group"))
            rows.append(_dielectric_row(index, attrs, interface_type, snapshot))
            baseline_entry = _index_entry(index, name, name, info, attrs, snapshot)
            if masks_requested:
                baseline_entry["baseline_index"] = index
            index_map.append(baseline_entry)
            for margin_index, margin_um in enumerate(
                preset.get("inset_margins_um", ())
            ):
                _validate_native_mask_support(
                    name=name,
                    info=info,
                    margin_um=float(margin_um),
                )
                mask_requests.append(
                    _MaskRequest(
                        index,
                        name,
                        info,
                        snapshot,
                        attrs,
                        margin_index,
                        float(margin_um),
                    )
                )

    for request in mask_requests:
        index = len(rows) + 1
        # The input file coordinates are Model.L0 units, despite Palace's
        # historical MaskData::margin comment claiming metres.
        native_margin = request.margin_um * 1e-6 / float(model_l0_m)
        rows.append(
            {
                **_dielectric_row(
                    index,
                    request.attrs,
                    request.info["interface_type"],
                    request.snapshot,
                ),
                "Mask": {"Type": "Inset", "Margin": native_margin},
            }
        )
        index_map.append(_mask_entry(index, request, native_margin, model_l0_m))
    return rows, index_map


def _matching_groups(
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]],
    interface_type: str,
    preset: Mapping[str, Any],
) -> list[tuple[str, Mapping[str, Any]]]:
    matches = []
    face_filter = preset.get("face_kinds")
    for name, info in sorted(groups.get("boundary_surfaces", {}).items()):
        if not isinstance(info, Mapping) or not info.get("surface_epr"):
            continue
        if info.get("interface_type") != interface_type:
            continue
        if face_filter is not None and info.get("face_kind") not in face_filter:
            continue
        matches.append((name, info))
    return matches


def _film_snapshot(preset: Mapping[str, Any], model_l0_m: float) -> dict[str, Any]:
    """Preset with thickness in both native units and metres, for the index map."""
    native_thickness = (
        preset["film_thickness_m"] / float(model_l0_m)
        if "film_thickness_m" in preset
        else preset["thickness"]
    )
    return {
        **preset,
        "thickness": native_thickness,
        "film_thickness_m": native_thickness * float(model_l0_m),
        "model_l0_m": float(model_l0_m),
        "schema_version": "scgsim.palace.surface-film.v2",
    }


def _dielectric_row(
    index: int,
    attrs: list[int],
    interface_type: str,
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "Index": index,
        "Attributes": attrs,
        "Type": interface_type,
        "Thickness": snapshot["thickness"],
        "Permittivity": snapshot["permittivity"],
        # Palace needs a numeric value; the index snapshot keeps None distinct.
        "LossTan": 0.0
        if snapshot["loss_tangent"] is None
        else snapshot["loss_tangent"],
    }


def _index_entry(  # noqa: PLR0913 - one argument per index-map field that differs between rows
    index: int,
    entry_name: str,
    name: str,
    info: Mapping[str, Any],
    attrs: list[int],
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "section": _DIELECTRIC_SECTION,
        "index": index,
        "entry_name": entry_name,
        "role": "surface_epr",
        "attributes": attrs,
        "physical_names": [name],
        "metadata": _structured_metadata(info),
        "epr_spec": dict(snapshot),
    }


def _mask_entry(
    index: int, request: _MaskRequest, native_margin: float, model_l0_m: float
) -> dict[str, Any]:
    return {
        **_index_entry(
            index,
            f"{request.name}__inset_{request.margin_index}",
            request.name,
            request.info,
            request.attrs,
            request.snapshot,
        ),
        "baseline_index": request.baseline_index,
        "mask": {
            "type": "Inset",
            "margin_um": request.margin_um,
            "margin_index": request.margin_index,
            "native_margin": native_margin,
            "model_l0_m": float(model_l0_m),
        },
    }


def _validate_native_mask_support(
    *,
    name: str,
    info: Mapping[str, Any],
    margin_um: float,
) -> None:
    """Reject only positive masks that need unavailable side-resolved geometry."""
    if margin_um == 0.0:
        return
    provenance = info.get("source_provenance")
    unsupported = tuple(
        support
        for support in _surface_mask_support_records(provenance)
        if support.get("positive_inset_supported") is False
    )
    if not unsupported:
        return
    details = tuple(
        {
            "reason": support.get("unsupported_reason"),
            "sides": tuple(support.get("sides", ())),
        }
        for support in unsupported
    )
    raise ValueError(
        f"Surface EPR group {name!r} cannot apply positive inset margin "
        f"{margin_um} um to aggregated side contributions: {details!r}."
    )


def _surface_mask_support_records(value: Any) -> tuple[Mapping[str, Any], ...]:
    """Collect ``mask_support`` records from a provenance tree, depth first."""
    if not isinstance(value, Mapping):
        return ()
    records: list[Mapping[str, Any]] = []
    support = value.get("mask_support")
    if isinstance(support, Mapping):
        records.append(support)
    sources = value.get("sources", ())
    if isinstance(sources, (list, tuple)):
        for source in sources:
            records.extend(_surface_mask_support_records(source))
    return tuple(records)


def _structured_metadata(info: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: info[key]
        for key in (
            "surface_id",
            "interface_type",
            "face_kind",
            "owner_semantic_ids",
            "adjacent_solution_volume_ids",
            "conductor_component_id",
            "net_id",
            "equipotential_id",
            "source_provenance",
            "physical_attribute",
            "representation",
            "route",
        )
        if key in info
    }


def _attributes(raw: Any) -> list[int]:
    values = raw if isinstance(raw, (list, tuple, set)) else (raw,)
    result = [
        int(value)
        for value in values
        if isinstance(value, int) and not isinstance(value, bool)
    ]
    if not result:
        raise ValueError("Surface EPR group requires physical attributes.")
    return sorted(set(result))


def _positive(value: Any) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _nonnegative(value: Any) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )
