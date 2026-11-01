"""Narrow shared PyAEDT binding helpers; Desktop lifecycle stays in transaction."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from scgsim.aedt.data import detached
from scgsim.aedt.specs.common import LOCKED_PYAEDT
from scgsim.aedt.specs.hfss import HfssDrivenSpec, HfssEigenmodeSpec
from scgsim.aedt.specs.parse import AedtSpec, parse_aedt_spec
from scgsim.aedt.specs.q3d import Q3dSpec
from scgsim.aedt.versions import AedtVersionProfile


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
    from ansys.aedt.core.generic.data_handlers import (  # noqa: PLC0415 -- optional SDK import at native boundary.  # private-import: pinned SDK decoder owns native GetData.
        _arg2dict,  # private-import: pinned SDK decoder owns the native GetData representation.
    )

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
        value: Any = observed.get(native_key)
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
    auto_update = material._material_update  # noqa: SLF001 -- pinned Material batches requested properties before one explicit update.
    material._material_update = False  # noqa: SLF001 -- pinned Material batches requested properties before one explicit update.
    try:
        for key, value in properties.items():
            setattr(material, key, value)
        updated = material.update()
    finally:
        material._material_update = auto_update  # noqa: SLF001 -- pinned Material batches requested properties before one explicit update.
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


def _positive_identity(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"{label} is invalid")
    return value


def _native_desktop_process_id(desktop: Any, label: str) -> int:
    try:
        value = desktop.odesktop.GetProcessID()
    except (AttributeError, TypeError) as exc:
        raise RuntimeError(f"{label} is unavailable") from exc
    return _positive_identity(value, label)


def owned_application_constructor(
    factory: Callable[..., Any], desktop: Any, *, profile: AedtVersionProfile
) -> Callable[..., Any]:
    """Pin one family application constructor to the transaction-owned Desktop."""

    expected_pid = _positive_identity(
        getattr(desktop, "aedt_process_id", None),
        "owned AEDT Desktop process identity",
    )
    expected_port = _positive_identity(
        getattr(desktop, "port", None), "owned AEDT Desktop endpoint"
    )
    native_pid = _native_desktop_process_id(
        desktop, "owned AEDT Desktop native process identity"
    )
    if native_pid != expected_pid:
        raise RuntimeError("owned AEDT Desktop native process identity is inconsistent")

    def construct(*args: Any, **kwargs: Any) -> Any:
        if any(name in kwargs for name in ("aedt_process_id", "port", "version")):
            raise TypeError(
                "family application cannot override the owned AEDT identity or version"
            )
        app = factory(
            *args,
            version=profile.requested_version,
            aedt_process_id=expected_pid,
            port=expected_port,
            **kwargs,
        )
        app_desktop = getattr(app, "desktop_class", None)
        actual_pid = _positive_identity(
            getattr(app_desktop, "aedt_process_id", None),
            "family application Desktop process identity",
        )
        actual_port = _positive_identity(
            getattr(app_desktop, "port", None),
            "family application Desktop endpoint",
        )
        if actual_pid != expected_pid or actual_port != expected_port:
            raise RuntimeError("family application did not bind the owned AEDT Desktop")
        actual_native_pid = _native_desktop_process_id(
            app_desktop, "family application native process identity"
        )
        if actual_native_pid != expected_pid:
            raise RuntimeError("family application native process identity changed")
        return app

    return construct


def _freeze_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("bound AEDT payload keys must be strings")
        return MappingProxyType(
            {key: _freeze_payload(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_payload(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"bound AEDT payload contains unsupported {type(value).__name__}")


def _thaw_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_payload(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_payload(item) for item in value]
    return value


@dataclass(frozen=True)
class BoundAedtRequest:
    """One detached spec payload bound to one absolute execution workspace."""

    workspace: Path
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        workspace = Path(self.workspace).expanduser().resolve()
        payload = _freeze_payload(self.payload)
        object.__setattr__(self, "workspace", workspace)
        object.__setattr__(self, "payload", payload)
        parse_aedt_spec(_thaw_payload(payload), base_dir=workspace)

    @classmethod
    def bind(cls, run_dir: str | Path, spec: AedtSpec) -> BoundAedtRequest:
        """Detach an already validated DTO without retaining caller containers."""
        return cls(Path(run_dir), spec.to_payload())

    def parse(self) -> AedtSpec:
        """Parse a fresh DTO using only the retained payload and bound root."""
        return parse_aedt_spec(_thaw_payload(self.payload), base_dir=self.workspace)

    def payload_copy(self) -> dict[str, Any]:
        """Return detached plain data for diagnostics and hashing."""
        return _thaw_payload(self.payload)


def detached_data(value: Any) -> Any:
    """Copy nested plain readback data without traversing native app handles."""
    if isinstance(value, Mapping):
        return {key: detached_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [detached_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(detached_data(item) for item in value)
    return value


def native_object_evidence(hfss: Any, object_name: str) -> dict[str, Any]:
    """Read object identity, shape kind, and faces directly from the editor."""
    editor = hfss.modeler.oeditor
    object_id = int(editor.GetObjectIDByName(object_name))
    memberships = [
        object_type
        for object_type, group in (
            ("Solid", "Solids"),
            ("Sheet", "Sheets"),
            ("Line", "Lines"),
            ("Unclassified", "Unclassified"),
        )
        if object_name in {str(name) for name in editor.GetObjectsInGroup(group)}
    ]
    if len(memberships) != 1:
        raise RuntimeError(
            f"native object type is ambiguous for {object_name!r}: {memberships!r}"
        )
    face_ids = [int(face_id) for face_id in editor.GetFaceIDs(object_name)]
    if len(face_ids) != len(set(face_ids)) or not face_ids:
        raise RuntimeError(f"native object faces are invalid for {object_name!r}")
    return {
        "native_object_id": object_id,
        "native_object_type": memberships[0],
        "native_face_ids": face_ids,
    }


def native_boundary_type(hfss: Any, name: str) -> str:
    """Read the unique boundary type from the live native boundary inventory."""
    raw = [str(value) for value in hfss.oboundary.GetBoundaries()]
    if len(raw) % 2:
        raise RuntimeError("native HFSS boundary list is invalid")
    matches = [raw[index + 1] for index in range(0, len(raw), 2) if raw[index] == name]
    if len(matches) != 1:
        raise RuntimeError(f"native boundary type is unavailable for {name!r}")
    return matches[0]


def resolve_native_assignment(
    hfss: Any, raw_ids: list[int]
) -> tuple[list[dict[str, Any]], set[int], set[str]]:
    """Resolve AEDT's face-or-object boundary IDs without guessing their kind."""
    by_object_id: dict[int, tuple[str, list[int]]] = {}
    by_face_id: dict[int, str] = {}
    for object_name in [str(name) for name in hfss.modeler.object_names]:
        native = native_object_evidence(hfss, object_name)
        object_id = native["native_object_id"]
        face_ids = native["native_face_ids"]
        if object_id in by_object_id:
            raise RuntimeError(f"duplicate native object ID: {object_id}")
        by_object_id[object_id] = (object_name, face_ids)
        for face_id in face_ids:
            if face_id in by_face_id:
                raise RuntimeError(f"ambiguous native face ID: {face_id}")
            by_face_id[face_id] = object_name

    resolved: list[dict[str, Any]] = []
    covered_faces: set[int] = set()
    covered_objects: set[str] = set()
    for raw_id in raw_ids:
        object_match = by_object_id.get(raw_id)
        face_match = by_face_id.get(raw_id)
        if object_match is not None and face_match is not None:
            raise RuntimeError(f"ambiguous PEC native assignment ID: {raw_id}")
        if object_match is not None:
            object_name, face_ids = object_match
            resolved.append(
                {
                    "raw_id": raw_id,
                    "native_kind": "object",
                    "object_name": object_name,
                    "face_ids": list(face_ids),
                }
            )
            covered_faces.update(face_ids)
            covered_objects.add(object_name)
        elif face_match is not None:
            resolved.append(
                {
                    "raw_id": raw_id,
                    "native_kind": "face",
                    "object_name": face_match,
                    "face_ids": [raw_id],
                }
            )
            covered_faces.add(raw_id)
            covered_objects.add(face_match)
        else:
            raise RuntimeError(f"unknown PEC native assignment ID: {raw_id}")
    return resolved, covered_faces, covered_objects


def create_region(
    hfss: Any, spec: HfssDrivenSpec | HfssEigenmodeSpec | Q3dSpec
) -> dict[str, Any]:
    """Create and verify the requested GDS-family solution Region."""
    if hfss.modeler.get_object_from_name("Region") is not None:
        raise RuntimeError("new V1 design unexpectedly already has Region")
    envelope = spec.effective_region
    keys = (
        "x_plus_um",
        "x_minus_um",
        "y_plus_um",
        "y_minus_um",
        "z_plus_um",
        "z_minus_um",
    )
    effective_padding = (
        tuple(envelope["effective_padding_um"][key] for key in keys)
        if envelope is not None
        else spec.region_padding_um
    )
    region = hfss.modeler.create_region(
        pad_value=list(effective_padding),
        pad_type="Absolute Offset",
        name="Region",
    )
    vacuum = spec.materials[spec.vacuum_material_id]
    library_name: Any = vacuum.library_name
    if not hfss.materials.exists_material(library_name):
        raise RuntimeError(f"AEDT vacuum material is unavailable: {library_name!r}")
    region.material_name = library_name
    observed_material = native_object_property(region, "Material").strip('"')
    if (
        region.name != "Region"
        or observed_material.casefold() != library_name.casefold()
    ):
        raise RuntimeError("vacuum region readback failed")
    result = {
        "material_id": vacuum.material_id,
        "requested_library_name": library_name,
        "observed_material_name": observed_material,
        "padding_um": list(spec.region_padding_um),
    }
    if envelope is not None:
        result["requested_padding_um"] = list(spec.region_padding_um)
        result["padding_um"] = list(effective_padding)
        result["source_effective_envelope"] = envelope
        result["native_bounding_box_um"] = list(q3d_region_bounds(region))
    if isinstance(spec, Q3dSpec):
        result["native_region_object_id"] = int(region.id)
        if envelope is None:
            result["native_bounding_box_um"] = list(q3d_region_bounds(region))
        result["requested_padding_um"] = list(spec.region_padding_um)
    return result


def q3d_region_bounds(region: Any) -> tuple[float, float, float, float, float, float]:
    """Read finite ordered native Region bounds in model units."""
    try:
        values = tuple(float(value) for value in region.bounding_box)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Q3D Region native bounds are unavailable") from exc
    if (
        len(values) != 6
        or not all(math.isfinite(value) for value in values)
        or any(values[index] >= values[index + 3] for index in range(3))
    ):
        raise RuntimeError("Q3D Region native bounds are invalid")
    return values


def saved_setup_properties(app: Any, setup_name: str) -> dict[str, Any]:
    """Read the one exact saved setup after matching its declared name."""
    analysis = app.design_properties.get("AnalysisSetup")
    setups = analysis.get("SolveSetups") if isinstance(analysis, dict) else None
    setup_names = (
        [name for name, value in setups.items() if isinstance(value, dict)]
        if isinstance(setups, dict)
        else []
    )
    if setup_names != [setup_name]:
        raise RuntimeError("saved AEDT project does not contain one exact setup")
    setup = cast(dict[str, Any], setups)[setup_name]
    if not isinstance(setup, dict):
        raise TypeError("saved AEDT setup properties are invalid")
    return setup


def pyaedt_version() -> str:
    """Require the pinned PyAEDT distribution before native execution."""
    try:
        installed = version("pyaedt")
    except PackageNotFoundError as exc:
        raise RuntimeError("PyAEDT is not installed") from exc
    if installed != LOCKED_PYAEDT:
        raise RuntimeError(
            f"PyAEDT version mismatch: expected {LOCKED_PYAEDT}, got {installed}"
        )
    return installed


def native_object_property(obj: Any, property_name: str) -> str:
    # Direct AEDT property query avoids the mutable PyAEDT object-property cache.
    """Read one native string property directly, bypassing the SDK cache."""
    value = obj._oeditor.GetPropertyValue(  # noqa: SLF001 -- direct native property read bypasses the mutable SDK object cache.
        "Geometry3DAttributeTab", obj.name, property_name
    )
    if not isinstance(value, str) or not value:
        raise RuntimeError(
            f"AEDT native object property {property_name!r} is unavailable for {obj.name!r}"
        )
    return value


def native_object_boolean_property(obj: Any, property_name: str) -> bool:
    # Keep boolean readback direct too; AEDT transports it as either bool or text.
    """Read one native boolean property through its transport representation."""
    value = obj._oeditor.GetPropertyValue(  # noqa: SLF001 -- direct native property read bypasses the mutable SDK object cache.
        "Geometry3DAttributeTab", obj.name, property_name
    )
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.casefold() in {"true", "false"}:
        return value.casefold() == "true"
    raise RuntimeError(
        f"AEDT native boolean property {property_name!r} is unavailable for {obj.name!r}"
    )


def native_boundary_names(hfss: Any) -> list[str]:
    """Read native boundary names after confirming the collection is available."""
    if "GetBoundaries" not in hfss.oboundary.__dir__():
        raise RuntimeError("AEDT native boundary collection is unavailable")
    return list(hfss.oboundary.GetBoundaries())
