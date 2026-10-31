"""Native AEDT material definition and readback for source-backed materials.

Owns turning resolved source scalars into project-local AEDT materials and
reading a native material back as plain numbers. It does not own which
materials a request needs or how conductors map to PEC."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from scgsim.aedt._data import detached


def source_material_properties(record: Mapping[str, Any]) -> dict[str, float]:
    """Project only declared scalar source properties to pinned AEDT names.

    Unrecognized material metadata is not a physical default. Known properties
    with unsupported shapes must fail instead of silently using a library value.
    """
    properties: dict[str, float] = {}
    for source_key, native_key in (
        ("permittivity", "permittivity"),
        ("permeability", "permeability"),
        ("conductivity", "conductivity"),
        ("loss_tangent", "dielectric_loss_tangent"),
    ):
        if source_key not in record:
            continue
        value = record[source_key]
        if source_key == "conductivity":
            raise TypeError(
                "source conductivity has no established scalar unit contract; "
                "the raw declared value is retained but cannot be applied"
            )
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"source material {source_key!r} requires a scalar number; "
                "unit expressions and tensor/nonlinear shapes are unsupported"
            )
        number = float(value)
        # These are the existing prepared permittivity/loss contracts. Other
        # declared scalar properties are forwarded without a new physics policy.
        if source_key == "permittivity" and (
            not math.isfinite(number) or number <= 0.0
        ):
            raise ValueError("prepared material requires positive permittivity")
        if source_key == "loss_tangent" and (not math.isfinite(number) or number < 0.0):
            raise ValueError("prepared material has invalid loss tangent")
        properties[native_key] = number
    return properties


def read_native_material(app: Any, native_name: str) -> dict[str, Any]:
    """Read the independent project material definition, never cached values."""
    from ansys.aedt.core.generic.data_handlers import _arg2dict

    raw = list(app.materials.omaterial_manager.GetData(native_name))
    parsed: dict[str, Any] = {}
    _arg2dict(raw, parsed)
    definition = parsed.get(native_name)
    if definition is None and len(parsed) == 1:
        # AEDT owns the spelling/case in its named GetData block.
        definition = next(iter(parsed.values()))
    if not isinstance(definition, Mapping):
        raise RuntimeError(
            f"native material definition is unavailable: {native_name!r}"
        )
    observed = {}
    unavailable = {}
    for key in (
        "permittivity",
        "permeability",
        "conductivity",
        "dielectric_loss_tangent",
    ):
        if key in definition:
            observed[key] = detached(definition[key])
        else:
            unavailable[key] = "Property absent from native GetData definition"
    result: dict[str, Any] = {
        "native_name": native_name,
        "observed": observed,
        "unavailable": unavailable,
        "native_definition": detached(definition),
        "native_GetData_raw": detached(raw),
        "readback_basis": "native MaterialManager.GetData, independent of cached MatProperty.value",
    }
    for native_key, historical_key in (
        ("permittivity", "relative_permittivity"),
        ("permeability", "relative_permeability"),
    ):
        value = observed.get(native_key)
        try:
            result[historical_key] = float(value)
        except (TypeError, ValueError):
            result[historical_key] = None
            unavailable[historical_key] = {
                "reason": "Native scalar value unavailable; raw shape retained",
                "raw": value,
            }
    return result


def realize_source_material(
    app: Any,
    material_id: str,
    record: Mapping[str, Any],
    *,
    native_name: str | None = None,
) -> dict[str, Any]:
    """Apply resolved source scalars to this project, including existing names.

    Lookup/template names do not override authored values. This function edits
    only the owned project's DefinitionManager; source/library files are untouched.
    Callers retain conductor/PEC translation and identity-only legacy ownership.
    """
    name = material_id if native_name is None else native_name
    properties = source_material_properties(record)
    material = app.materials.exists_material(name)
    if not material:
        template_name = record.get("aedt_library_name")
        if template_name is not None and template_name.casefold() != name.casefold():
            template = app.materials.exists_material(template_name)
            if not template:
                raise RuntimeError(
                    f"AEDT material template is unavailable: {template_name!r}"
                )
            material = app.materials.duplicate_material(template.name, name)
        else:
            material = app.materials.add_material(name, properties)
    if material is False or material is None:
        raise RuntimeError(f"failed to define native material {material_id!r}")
    # Pinned Material uses this flag for batch property editing. Commit all
    # requested values through one explicit project AddMaterial/EditMaterial call.
    auto_update = material._material_update  # noqa: SLF001 - pinned PyAEDT batch-edit flag
    material._material_update = False  # noqa: SLF001
    try:
        for key, value in properties.items():
            setattr(material, key, value)
        updated = material.update()
    finally:
        material._material_update = auto_update  # noqa: SLF001
    if updated is not True:
        raise RuntimeError(f"native material update failed: {name!r}")
    result = read_native_material(app, material.name)
    result.update(
        {
            "material_id": material_id,
            "requested": detached(dict(record)),
            "applied": properties,
            "application_basis": "project-local Material.update returned True; native readback retained separately",
        }
    )
    return result
