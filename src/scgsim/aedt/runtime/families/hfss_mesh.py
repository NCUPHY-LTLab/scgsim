"""Own native mesh assignment and saved mesh properties."""

from __future__ import annotations

from typing import Any

from scgsim.aedt.runtime.families.hfss_port_properties import native_value_matches
from scgsim.aedt.runtime.families.hfss_requests import HfssGdsSpec
from scgsim.aedt.specs.common import (
    SURFACE_APPROXIMATION_LEVEL,
)


def assign_mesh(hfss: Any, spec: HfssGdsSpec) -> dict[str, Any]:
    """Assign the requested mesh controls and verify their native saved properties."""
    grounds = [
        item.object_name for item in spec.object_bindings if item.role == "ground"
    ]
    signals = [
        item.object_name for item in spec.object_bindings if item.role == "signal"
    ]
    result: dict[str, Any] = {"surface": {}, "length": {}}
    for role, objects, name in (
        ("signal", signals, "SignalSurfaceApprox9"),
        ("ground", grounds, "GroundSurfaceApprox9"),
    ):
        operation = hfss.mesh.assign_surface_mesh(
            objects, SURFACE_APPROXIMATION_LEVEL, name
        )
        if operation is None or name not in _native_mesh_operation_names(hfss):
            raise RuntimeError(f"{role} surface mesh readback failed")
        native = _native_mesh_readback(hfss, name, objects)
        _require_native_mesh_properties(
            native,
            {
                "Name": name,
                "Type": "Surface Approximation Based",
                "Region": "On Selection",
                "Curved Mesh Approximation Type": "Use Slider",
                "Curved Surface Mesh Resolution": SURFACE_APPROXIMATION_LEVEL,
            },
            f"{role} surface mesh",
        )
        result["surface"][role] = {
            "requested": {
                "name": name,
                "objects": objects,
                "level": SURFACE_APPROXIMATION_LEVEL,
            },
            "native": native,
        }
    if spec.length_mesh is not None:
        for role, objects, name in (
            ("signal", list(spec.length_mesh.signal_objects), "UniformCpwSignalLength"),
            ("ground", list(spec.length_mesh.ground_objects), "UniformCpwGroundLength"),
        ):
            operation = hfss.mesh.assign_length_mesh(
                objects,
                inside_selection=False,
                maximum_length=f"{spec.length_mesh.maximum_length_um:g}um",
                maximum_elements=1_000_000,
                name=name,
            )
            requested = {
                "Objects": objects,
                "MaxLength": f"{spec.length_mesh.maximum_length_um:g}um",
                "NumMaxElem": "1000000",
                "RestrictElem": True,
                "RestrictLength": True,
                "RefineInside": False,
                "Enabled": True,
            }
            if operation is None or name not in _native_mesh_operation_names(hfss):
                raise RuntimeError(f"uniform CPW {role} length mesh readback failed")
            native = _native_mesh_readback(hfss, name, objects)
            _require_native_mesh_properties(
                native,
                {
                    "Name": name,
                    "Type": "Length Based",
                    "Region": "On Selection",
                    "Enabled": True,
                    "Restrict Length": True,
                    "Max Length": f"{spec.length_mesh.maximum_length_um:g}um",
                    "Restrict Max Elems": True,
                    "Max Elems": 1_000_000,
                },
                f"uniform CPW {role} length mesh",
            )
            result["length"][role] = {
                "requested": {"name": name, "properties": requested},
                "native": native,
            }
    return result


def _native_mesh_operation_names(hfss: Any) -> list[str]:
    names = hfss.get_oo_name(hfss.odesign, "Mesh")
    if not isinstance(names, list):
        raise TypeError("AEDT native mesh operation collection is unavailable")
    return names


def _native_mesh_readback(
    hfss: Any, expected_name: str, expected_objects: list[str]
) -> dict[str, Any]:
    """Read AEDT OOP mesh properties and MeshSetup assignments directly."""
    operation_names = _native_mesh_operation_names(hfss)
    if expected_name not in operation_names:
        raise RuntimeError(f"mesh operation is missing natively: {expected_name}")
    property_names = hfss.get_oo_properties(hfss.odesign, f"Mesh/{expected_name}")
    if not property_names:
        raise RuntimeError(
            f"mesh operation native properties are unavailable: {expected_name}"
        )
    properties = {
        property_name: hfss.get_oo_property_value(
            hfss.odesign, f"Mesh/{expected_name}", property_name
        )
        for property_name in property_names
    }
    assigned_ids = [
        int(object_id)
        for object_id in hfss.mesh.omeshmodule.GetMeshOpAssignment(expected_name)
    ]
    objects = [hfss.oeditor.GetObjectNameByID(object_id) for object_id in assigned_ids]
    if objects != expected_objects:
        raise RuntimeError(
            f"mesh operation native assignment mismatch: {expected_name}"
        )
    return {
        "operation_names": operation_names,
        "properties": properties,
        "object_ids": assigned_ids,
        "objects": objects,
    }


def _require_native_mesh_properties(
    native: dict[str, Any], expected: dict[str, Any], context: str
) -> None:
    """Compare required settings while retaining AEDT's raw native properties."""
    observed = native["properties"]
    mismatches = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items()
        if key not in observed or not native_value_matches(observed[key], value)
    }
    if mismatches:
        raise RuntimeError(f"{context} native property mismatch: {mismatches!r}")
