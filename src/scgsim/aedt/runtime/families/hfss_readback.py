"""Native HFSS readback records bound into a prepared solve.

Owns reading mesh operations, terminals, modal ports and terminal references
back from the live design and comparing them with what the request asked for.
It does not own assigning those objects or exporting results."""

from __future__ import annotations

import math
import re
from typing import Any

from scgsim.aedt.specs.common import (
    ModalPort,
    TerminalPort,
)
from scgsim.aedt.specs.hfss import HfssDrivenSpec, HfssEigenmodeSpec
from scgsim.aedt.specs.parse import HfssSpec


def native_mesh_operation_names(hfss: Any) -> list[str]:
    """Return the names of native HFSS mesh operations, raising if unavailable."""
    names = hfss.get_oo_name(hfss.odesign, "Mesh")
    if not isinstance(names, list):
        raise TypeError("AEDT native mesh operation collection is unavailable")
    return names


def native_mesh_readback(
    hfss: Any, expected_name: str, expected_objects: list[str]
) -> dict[str, Any]:
    """Read AEDT OOP mesh properties and MeshSetup assignments directly."""
    operation_names = native_mesh_operation_names(hfss)
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


def require_native_mesh_properties(
    native: dict[str, Any], expected: dict[str, Any], context: str
) -> None:
    """Compare required settings while retaining AEDT's raw native properties."""
    observed = native["properties"]
    mismatches = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items()
        if key not in observed or not _native_value_matches(observed[key], value)
    }
    if mismatches:
        raise RuntimeError(f"{context} native property mismatch: {mismatches!r}")


def _native_value_matches(observed: Any, expected: Any) -> bool:
    """Accept AEDT native scalar spelling without converting its recorded value."""
    if isinstance(expected, bool):
        return observed is expected or observed == str(expected).lower()
    if isinstance(expected, int):
        return observed == expected or observed == str(expected)
    return observed == expected


def native_terminal_readback(
    hfss: Any, boundary_name: str, terminal_name: str, port: TerminalPort
) -> dict[str, Any]:
    """Read port and terminal state from AEDT's Excitations object tree."""
    excitation_names = hfss.get_oo_name(hfss.odesign, "Excitations")
    if boundary_name not in excitation_names:
        raise RuntimeError(f"AEDT terminal boundary is missing: {boundary_name!r}")
    terminal_names = hfss.get_oo_name(hfss.odesign, f"Excitations\\{boundary_name}")
    if terminal_names != [terminal_name]:
        raise RuntimeError(
            f"AEDT terminal child mismatch for {boundary_name!r}: {terminal_names!r}"
        )
    boundary_properties = native_oo_properties(hfss, f"Excitations\\{boundary_name}")
    terminal_properties = native_oo_properties(
        hfss, f"Excitations\\{boundary_name}\\{terminal_name}"
    )
    require_native_properties(
        boundary_properties,
        {
            "Name": boundary_name,
            "Type": "Wave Port",
            "Wave Port Type": "Terminal",
            "Num Terminals": 1,
            "Deembed Dist": f"{port.deembed_um:g}um",
            "Renorm All Terminals": False,
        },
        f"terminal boundary {boundary_name!r}",
    )
    require_native_properties(
        terminal_properties,
        {"Name": terminal_name, "Port Name": boundary_name, "Type": "Terminal"},
        f"terminal child {terminal_name!r}",
    )
    if boundary_properties.get("Deembed") not in (True, False, "true", "false"):
        raise RuntimeError("native Wave deembed flag is unavailable")
    return {
        "excitation_names": excitation_names,
        "terminal_names": terminal_names,
        "boundary_properties": boundary_properties,
        "terminal_properties": terminal_properties,
        "inactive_renormalizing_impedance": terminal_properties.get(
            "Terminal Renormalizing Impedance"
        ),
    }


def native_modal_oo_readback(hfss: Any, boundary_name: str) -> dict[str, Any]:
    """Read the direct AEDT object-tree state of one modal wave port."""
    excitation_names = hfss.get_oo_name(hfss.odesign, "Excitations")
    if boundary_name not in excitation_names:
        raise RuntimeError(f"AEDT modal boundary is missing: {boundary_name!r}")
    properties = native_oo_properties(hfss, f"Excitations\\{boundary_name}")
    require_native_properties(
        properties,
        {
            "Name": boundary_name,
            "Type": "Wave Port",
            "Num Modes": 1,
            "Deembed": False,
            "Renorm All Modes": False,
        },
        f"modal boundary {boundary_name!r}",
    )
    return {
        "excitation_names": excitation_names,
        "boundary_properties": properties,
    }


def bind_port_evidence(
    hfss: Any, spec: HfssSpec, ports: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Attach native excitation evidence to port records by solution mode."""
    if isinstance(spec, HfssEigenmodeSpec):
        if ports:
            raise RuntimeError("HFSS Eigenmode must not bind port evidence")
        return ports
    if spec.mode == "terminal":
        return _bind_terminal_reference_evidence(hfss, spec, ports)
    return _bind_modal_evidence(hfss, spec, ports)


def _bind_modal_evidence(
    hfss: Any, spec: HfssDrivenSpec, ports: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Bind modal integration lines from AEDT's saved native design state."""
    try:
        boundaries = hfss.design_properties["BoundarySetup"]["Boundaries"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("AEDT native modal boundaries are unavailable") from exc
    if not isinstance(boundaries, dict):
        raise TypeError("AEDT native modal boundary data is invalid")
    for record, port, effective_port in zip(
        ports, spec.ports, spec.effective_ports, strict=True
    ):
        if not isinstance(port, ModalPort):
            raise TypeError("modal evidence requires ModalPort entries")
        if not isinstance(effective_port, ModalPort):
            raise TypeError("modal effective evidence requires ModalPort entries")
        boundary = boundaries.get(port.name)
        try:
            mode = boundary["Modes"]["Mode1"]
            positions = mode["IntLine"]["GeometryPosition"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"AEDT native modal integration line is unavailable: {port.name!r}"
            ) from exc
        observed = [
            [_native_position_um(item[f"{axis}Position"]) for axis in "XYZ"]
            for item in positions
        ]
        expected = [list(point) for point in effective_port.integration_line_um]
        if (
            boundary.get("BoundType") != "Wave Port"
            or boundary.get("WavePortType") != "Modal"
            or boundary.get("NumModes") != 1
            or boundary.get("Faces") != [record["face_id"]]
            or boundary.get("DoDeembed") is not False
            or mode.get("ModeNum") != 1
            or mode.get("UseIntLine") is not True
            or mode.get("CharImp") != "Zpi"
            or len(observed) != 2
            or any(
                not math.isclose(actual, wanted, abs_tol=1e-9)
                for actual_point, expected_point in zip(observed, expected, strict=True)
                for actual, wanted in zip(actual_point, expected_point, strict=True)
            )
        ):
            raise RuntimeError(
                f"AEDT native modal port readback mismatch: {port.name!r}"
            )
        native = record.get("native")
        if not isinstance(native, dict):
            raise TypeError("modal native OO evidence is unavailable")
        native["saved_boundary"] = {
            "bound_type": boundary["BoundType"],
            "wave_port_type": boundary["WavePortType"],
            "faces": boundary["Faces"],
            "num_modes": boundary["NumModes"],
            "deembed": boundary["DoDeembed"],
            "mode_number": mode["ModeNum"],
            "use_integration_line": mode["UseIntLine"],
            "integration_line_um": observed,
            "characteristic_impedance": mode["CharImp"],
        }
    return ports


def _native_position_um(value: Any) -> float:
    """Read one saved AEDT modal coordinate as a finite micrometer value."""
    if isinstance(value, bool):
        raise ValueError("AEDT modal coordinate must be a finite length")
    if isinstance(value, str):
        match = re.fullmatch(
            r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z]+)",
            value.strip(),
        )
        if match:
            from ansys.aedt.core.generic.constants import AEDT_UNITS, unit_converter

            units = match.group(2)
            if units not in AEDT_UNITS["Length"]:
                raise ValueError("AEDT modal coordinate units must be a length")
            value = unit_converter(
                float(match.group(1)),
                unit_system="Length",
                input_units=units,
                output_units="um",
            )
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("AEDT modal coordinate must be a finite length") from exc
    if not math.isfinite(result):
        raise ValueError("AEDT modal coordinate must be a finite length")
    return result


def _bind_terminal_reference_evidence(
    hfss: Any, spec: HfssDrivenSpec, ports: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Bind global AEDT terminal reference IDs after saving the owned project."""
    if len(ports) != len(spec.ports):
        raise RuntimeError("terminal port evidence count is invalid")
    reference_ids = native_terminal_reference_ids(hfss)
    reference_objects = [
        hfss.oeditor.GetObjectNameByID(object_id) for object_id in reference_ids
    ]
    expected = list(spec.ports[0].reference_objects)
    if reference_objects != expected:
        raise RuntimeError("AEDT native terminal reference conductor mismatch")
    for record, port in zip(ports, spec.ports, strict=True):
        if (
            not isinstance(port, TerminalPort)
            or list(port.reference_objects) != expected
        ):
            raise RuntimeError("terminal reference contract is not globally consistent")
        native = record.get("native")
        if not isinstance(native, dict):
            raise TypeError("terminal native evidence is unavailable")
        native["reference_conductor_ids"] = reference_ids
        native["reference_conductors"] = reference_objects
    return ports


def native_terminal_reference_ids(hfss: Any) -> list[int]:
    """Read terminal reference conductors from HFSS native design properties."""
    try:
        values = hfss.design_properties["BoundarySetup"]["ProductSpecificData"][
            "TerminalReferenceConductors"
        ]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            "AEDT native terminal reference conductors are unavailable"
        ) from exc
    if not isinstance(values, list) or not values:
        raise RuntimeError("AEDT native terminal reference conductors are invalid")
    try:
        return [int(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise RuntimeError("AEDT native terminal reference IDs are invalid") from exc


def native_oo_properties(hfss: Any, path: str) -> dict[str, Any]:
    """Return properties directly from an AEDT object-oriented child object."""
    child = hfss.get_oo_object(hfss.odesign, path)
    if not child:
        raise RuntimeError(f"AEDT native child object is unavailable: {path!r}")
    names = child.GetPropNames()
    if not names:
        raise RuntimeError(f"AEDT native child has no properties: {path!r}")
    return {name: child.GetPropValue(name) for name in names}


def require_native_properties(
    observed: dict[str, Any], expected: dict[str, Any], context: str
) -> None:
    """Require semantic settings from their direct AEDT native representation."""
    mismatches = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items()
        if key not in observed or not _native_value_matches(observed[key], value)
    }
    if mismatches:
        raise RuntimeError(f"{context} native property mismatch: {mismatches!r}")


def face_for_side(
    faces: list[tuple[int, tuple[float, float, float]]], side: str
) -> int:
    """Return the id of the single face lying on the extreme of the given side."""
    axis, sign = {
        "-X": (0, -1),
        "+X": (0, 1),
        "-Y": (1, -1),
        "+Y": (1, 1),
        "-Z": (2, -1),
        "+Z": (2, 1),
    }[side]
    edge = (
        max(center[axis] for _, center in faces)
        if sign > 0
        else min(center[axis] for _, center in faces)
    )
    matches = [face_id for face_id, center in faces if abs(center[axis] - edge) <= 1e-9]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one Region face on {side}, got {matches!r}"
        )
    return matches[0]
