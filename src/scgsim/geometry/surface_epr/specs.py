"""Surface-EPR specs: the ``MA``/``MS``/``SA`` film parameters and the inset margins SGB cuts as bands.

``MA`` is the metal-air interface, ``MS`` metal-substrate and ``SA`` substrate-air.  A spec gives the film
(thickness, permittivity or a dielectric material, loss tangent) and, optionally, the inset margins of its
bands: ``inset_margins_um`` for every face of its type, ``inset_margins_by_entity_um`` for chosen conductors.
Both are backend-neutral: Palace and AEDT read the same normalized specs, and the margins are geometry that
:mod:`scgsim.geometry.surface_epr.build` turns into SGB inset requests.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from numbers import Real
from typing import Any, cast

from scgsim.semantics.epr import film_assumptions

from .insets import check_bandable_face_kinds


def normalize_surface_epr_specs(
    specs: Mapping[str, Mapping[str, Any]],
    *,
    materials: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Normalize configured MA/MS/SA layers from numeric or explicit material data."""
    if not isinstance(specs, Mapping):
        raise TypeError("surface_epr specs must be a mapping.")
    unknown_types = set(specs).difference({"MA", "MS", "SA"})
    if unknown_types:
        raise ValueError(
            f"surface_epr has unsupported interface types: {sorted(unknown_types)}."
        )
    result: dict[str, dict[str, Any]] = {}
    for interface_type, raw in sorted(specs.items()):
        result[interface_type] = _normalize_interface_spec(
            interface_type, raw, materials
        )
    return result


def _normalize_interface_spec(
    interface_type: str,
    raw: Mapping[str, Any],
    materials: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Normalize one interface without changing validation or material precedence."""
    if not isinstance(raw, Mapping):
        raise TypeError(f"Surface EPR {interface_type} spec must be a mapping.")
    unknown = set(raw).difference(
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
            "inset_margins_by_entity_um",
        }
    )
    if unknown:
        raise ValueError(
            f"Surface EPR {interface_type} has unknown fields: {sorted(unknown)}."
        )
    thickness = raw.get("thickness")
    thickness_m = raw.get("film_thickness_m")
    if (thickness is None) == (thickness_m is None):
        raise ValueError(
            f"Surface EPR {interface_type} requires exactly one native thickness or film_thickness_m."
        )
    if thickness is not None and not positive_real(thickness):
        raise ValueError(f"Surface EPR {interface_type} requires positive thickness.")
    if thickness_m is not None:
        film_assumptions({"film_thickness_m": thickness_m}, partial=True)
    numeric = raw.get("permittivity")
    material_id = raw.get("material_id")
    if (numeric is None) == (material_id is None):
        raise ValueError(
            f"Surface EPR {interface_type} requires exactly one of permittivity or material_id."
        )
    numeric, default_loss, resolved = _resolve_film_material(
        interface_type, material_id, numeric, materials
    )
    if not positive_real(numeric):
        raise ValueError(
            f"Surface EPR {interface_type} requires positive resolved permittivity."
        )
    loss_tangent = raw.get("loss_tangent", default_loss)
    film_assumptions({"loss_tangent": loss_tangent}, partial=True)
    source = raw.get("source", "material" if material_id is not None else "explicit")
    preset = raw.get("preset")
    film_assumptions({"source": source, "preset": preset}, partial=True)
    face_kinds = raw.get("face_kinds")
    if face_kinds is not None:
        if not isinstance(face_kinds, (list, tuple)) or not face_kinds:
            raise ValueError(
                f"Surface EPR {interface_type} face_kinds must be a non-empty sequence."
            )
        normalized_faces = sorted({str(value).strip().lower() for value in face_kinds})
        if not all(value for value in normalized_faces):
            raise ValueError(
                f"Surface EPR {interface_type} face_kinds must be non-empty."
            )
        resolved["face_kinds"] = normalized_faces
    required = raw.get("required", True)
    if not isinstance(required, bool):
        raise TypeError(f"Surface EPR {interface_type} required must be a bool.")
    _set_inset_margins(interface_type, raw, resolved)
    if "face_kinds" in resolved and (
        "inset_margins_um" in resolved or "inset_margins_by_entity_um" in resolved
    ):
        check_bandable_face_kinds(interface_type, resolved["face_kinds"])
    return {
        **(
            {"thickness": float(thickness)}
            if thickness is not None
            else {"film_thickness_m": float(cast(Any, thickness_m))}
        ),
        "permittivity": float(cast(Any, numeric)),
        "loss_tangent": None if loss_tangent is None else float(loss_tangent),
        "source": source,
        "preset": preset,
        "required": required,
        **resolved,
    }


def _resolve_film_material(
    interface_type: str,
    material_id: Any,
    numeric: Any,
    materials: Mapping[str, Mapping[str, Any]],
) -> tuple[Any, Any, dict[str, Any]]:
    """Resolve explicit dielectric identity and its optional default loss."""
    if material_id is not None:
        if not isinstance(material_id, str) or not material_id.strip():
            raise ValueError(
                f"Surface EPR {interface_type} material_id must be non-empty."
            )
        material = materials.get(material_id)
        if not isinstance(material, Mapping) or material.get("kind") != "dielectric":
            raise ValueError(
                f"Surface EPR {interface_type} material_id must resolve an explicit dielectric material."
            )
        numeric = material.get("permittivity")
        default_loss = material.get("loss_tangent")
        resolved = {"material_id": material_id}
    else:
        default_loss = None
        resolved = {}
    return numeric, default_loss, resolved


def _set_inset_margins(
    interface_type: str, raw: Mapping[str, Any], resolved: dict[str, Any]
) -> None:
    """Retain the margins SGB cuts into the model as bands: the spec's, and those of customised conductors."""
    if "inset_margins_um" in raw:
        resolved["inset_margins_um"] = _validated_margins(
            raw["inset_margins_um"], f"Surface EPR {interface_type} inset_margins_um"
        )
    if "inset_margins_by_entity_um" in raw:
        resolved["inset_margins_by_entity_um"] = _by_entity_margins(
            interface_type, raw["inset_margins_by_entity_um"]
        )


def _by_entity_margins(interface_type: str, raw: Any) -> dict[str, list[float]]:
    """Margins per customised conductor, checked like ``inset_margins_um``."""
    label = f"Surface EPR {interface_type} inset_margins_by_entity_um"
    if interface_type == "SA":
        raise ValueError(
            f"{label} customises conductors, and an SA interface lies between solution regions; "
            "use inset_margins_um for SA."
        )
    if not isinstance(raw, Mapping) or not raw:
        raise TypeError(f"{label} must be a non-empty mapping of entity id to margins.")
    result: dict[str, list[float]] = {}
    for entity, margins in sorted(raw.items(), key=lambda item: str(item[0])):
        if not isinstance(entity, str) or not entity:
            raise ValueError(f"{label} keys must be non-empty entity ids.")
        result[entity] = _validated_margins(margins, f"{label}[{entity!r}]")
    return result


def _validated_margins(margins: Any, label: str) -> list[float]:
    """The ordered, strictly increasing positive margins of one sequence."""
    if isinstance(margins, (str, bytes)) or not isinstance(margins, (list, tuple)):
        raise TypeError(f"{label} must be an ordered non-empty sequence.")
    if not margins:
        raise ValueError(f"{label} must be non-empty.")
    normalized = []
    for margin in margins:
        if not positive_real(margin):
            raise ValueError(
                f"{label} values must be finite positive reals "
                "(the whole surface is the baseline row)."
            )
        normalized.append(float(margin))
    if any(b <= a for a, b in zip(normalized, normalized[1:], strict=False)):
        raise ValueError(f"{label} must be strictly increasing.")
    return normalized


def margins_of(spec: Mapping[str, Any], entities: Iterable[str]) -> list[float]:
    """The inset margins (µm) of a face of ``entities`` under ``spec``.

    ``inset_margins_by_entity_um`` customises conductors one by one; every other face takes the spec's
    ``inset_margins_um``.  A spec with neither has no margins.
    """
    by_entity = spec.get("inset_margins_by_entity_um") or {}
    for entity in entities:
        if entity in by_entity:
            return list(by_entity[entity])
    return list(spec.get("inset_margins_um", ()))


def positive_real(value: Any) -> bool:
    """Whether ``value`` is a finite positive real number (a bool is not a number here)."""
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )
