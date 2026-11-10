"""Strict offline validation for terminal-port native readback and physical assignments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from scgsim.aedt.results.provenance import (
    SOURCE_SCHEMA_V19,
    SOURCE_SCHEMA_V20,
    SOURCE_SCHEMA_V21,
    SOURCE_SCHEMA_V22,
    SOURCE_SCHEMA_V23,
    SOURCE_SCHEMA_V24,
    SOURCE_SCHEMA_V25,
)
from scgsim.aedt.specs.common import LumpedTerminalPort, TerminalPort
from scgsim.aedt.specs.hfss import (
    HfssDrivenGeometrySpec,
    HfssDrivenSpec,
)

CURRENT_TERMINAL_NATIVE_SCHEMAS = {
    SOURCE_SCHEMA_V19,
    SOURCE_SCHEMA_V20,
    SOURCE_SCHEMA_V21,
    SOURCE_SCHEMA_V22,
    SOURCE_SCHEMA_V23,
    SOURCE_SCHEMA_V24,
    SOURCE_SCHEMA_V25,
}


@dataclass(frozen=True)
class BodyTerminalContext:
    """Source and native inventories shared by body-first terminal checks."""

    receipt: dict[str, Any]
    spec: HfssDrivenGeometrySpec
    geometry: dict[str, Any]
    conductors: list[dict[str, Any]]
    objects: list[dict[str, Any]]
    global_references: set[str]


def validate_body_terminal_readback(
    receipt: dict[str, Any], spec: HfssDrivenGeometrySpec
) -> None:
    """Verify body-first terminal ports against their source and native readback."""
    geometry = receipt.get("geometry", {})
    if geometry.get("source_sha256") != spec.geometry.source_sha256:
        raise RuntimeError("body-first Terminal geometry source differs")
    conductors = spec.geometry.source["conductors"]
    objects = geometry.get("objects", [])
    global_references = {
        name
        for port in spec.ports
        for name in _body_terminal_names(
            conductors,
            objects,
            port.reference_entity_ids
            if isinstance(port, LumpedTerminalPort)
            else port.reference_objects,
        )
    }
    context = BodyTerminalContext(
        receipt, spec, geometry, conductors, objects, global_references
    )
    for record, port in zip(receipt["ports"], spec.ports, strict=True):
        _validate_body_terminal_port(context, record, port)


def _body_terminal_names(
    conductors: list[dict[str, Any]], objects: list[dict[str, Any]], entity_ids: Any
) -> list[str]:
    """Map declared source conductor IDs to unique native body names."""
    result = []
    for entity_id in entity_ids:
        ids = {
            item["semantic_id"]
            for item in conductors
            if entity_id in {item["semantic_id"], item["source_semantic_id"]}
        }
        matches = [
            item["object_name"]
            for item in objects
            if item["kind"] == "conductor" and item["semantic_id"] in ids
        ]
        if not matches:
            raise RuntimeError("body-first Terminal source Entity has no native bodies")
        result.extend(name for name in matches if name not in result)
    return result


def _validate_body_terminal_port(
    context: BodyTerminalContext, record: dict[str, Any], port: Any
) -> None:
    terminal_port = cast(TerminalPort, port)
    reference_ids = (
        port.reference_entity_ids
        if isinstance(port, LumpedTerminalPort)
        else terminal_port.reference_objects
    )
    expected = {
        "signal_entity_ids": list(port.signal_entity_ids),
        "reference_entity_ids": list(reference_ids),
        "signal_objects": _body_terminal_names(
            context.conductors, context.objects, port.signal_entity_ids
        ),
        "reference_objects": _body_terminal_names(
            context.conductors, context.objects, reference_ids
        ),
    }
    native = record.get("native", {})
    requested = (
        port.to_payload()
        if isinstance(port, LumpedTerminalPort)
        else {
            "index": port.index,
            "name": port.name,
            "side": port.side,
            "reference_objects": list(terminal_port.reference_objects),
            "signal_entity_ids": list(port.signal_entity_ids),
            "renormalize": False,
            "deembed_um": terminal_port.deembed_um,
        }
    )
    if record.get("requested") != requested:
        raise RuntimeError("body-first Terminal authored treatment differs")
    if (
        record.get("index") != port.index
        or record.get("boundary") != port.name
        or record.get("source_binding") != expected
        or not native.get("signal_objects")
        or not set(native["signal_objects"]).issubset(expected["signal_objects"])
        or set(native.get("reference_conductors", ())) != context.global_references
    ):
        raise RuntimeError("body-first Terminal native ownership differs")
    terminal = native.get("saved_terminal", {})
    boundary = native.get("saved_boundary", {})
    expected_type = (
        "Lumped Port" if isinstance(port, LumpedTerminalPort) else "Wave Port"
    )
    if (
        boundary.get("BoundType") != expected_type
        or native.get("terminal_properties", {}).get("Port Name") != port.name
    ):
        raise RuntimeError("body-first Terminal native family differs")
    if [int(value) for value in boundary.get("Faces", ())] != [record.get("face_id")]:
        raise RuntimeError("body-first Terminal native support face differs")
    _validate_body_terminal_assignment(context, record, port, native, terminal)
    _validate_body_terminal_properties(context, record, port, native, expected_type)


def _validate_body_terminal_assignment(
    context: BodyTerminalContext,
    record: dict[str, Any],
    port: Any,
    native: dict[str, Any],
    terminal: dict[str, Any],
) -> None:
    assignment_kind = native.get("terminal_assignment_kind")
    if assignment_kind == "Faces":
        _validate_body_terminal_faces(context, record, native, terminal)
    elif assignment_kind is None:
        _validate_body_terminal_edges(record, native, terminal)
    else:
        raise RuntimeError("body-first Terminal native assignment kind is invalid")


def _validate_body_terminal_faces(
    context: BodyTerminalContext,
    record: dict[str, Any],
    native: dict[str, Any],
    terminal: dict[str, Any],
) -> None:
    face_records = native.get("terminal_faces")
    raw_faces = terminal.get("Faces")
    if (
        "terminal_edges" in native
        or not isinstance(face_records, list)
        or not face_records
        or not isinstance(raw_faces, list)
        or not raw_faces
        or terminal.get("Edges") not in (None, [], ())
    ):
        raise RuntimeError("body-first Terminal native face assignment is invalid")
    try:
        raw_face_ids = [int(value) for value in raw_faces]
    except (TypeError, ValueError) as exc:
        raise RuntimeError("body-first Terminal saved face IDs are invalid") from exc
    conductor_objects = [
        item
        for item in context.objects
        if isinstance(item, dict) and item.get("kind") == "conductor"
    ]
    face_ids = []
    face_owners = []
    for face_record in face_records:
        if (
            not isinstance(face_record, dict)
            or set(face_record) != {"face_id", "object_name", "native_object_id"}
            or not isinstance(face_record.get("face_id"), int)
            or isinstance(face_record.get("face_id"), bool)
            or not isinstance(face_record.get("object_name"), str)
            or not face_record["object_name"]
            or not isinstance(face_record.get("native_object_id"), int)
            or isinstance(face_record.get("native_object_id"), bool)
        ):
            raise RuntimeError(
                "body-first Terminal native face owner record is invalid"
            )
        face_id = face_record["face_id"]
        owners = [
            item
            for item in conductor_objects
            if isinstance(item.get("native_face_ids"), list)
            and face_id in item["native_face_ids"]
        ]
        if len(owners) != 1:
            raise RuntimeError("body-first Terminal native face ownership is ambiguous")
        owner = owners[0]
        if (
            owner.get("object_name") != face_record["object_name"]
            or owner.get("native_object_id") != face_record["native_object_id"]
        ):
            raise RuntimeError(
                "body-first Terminal native face owner differs from geometry"
            )
        face_ids.append(face_id)
        face_owners.append(face_record["object_name"])
    if face_ids != raw_face_ids:
        raise RuntimeError("body-first Terminal native face assignment differs")
    if set(face_owners) != set(native["signal_objects"]):
        raise RuntimeError("body-first Terminal contact ownership differs")


def _validate_body_terminal_edges(
    record: dict[str, Any], native: dict[str, Any], terminal: dict[str, Any]
) -> None:
    if terminal.get("Faces") not in (None, [], ()):
        raise RuntimeError("body-first Terminal child assignment kind is ambiguous")
    edges = native.get("terminal_edges", [])
    if not edges or [item["edge_id"] for item in edges] != [
        int(value) for value in terminal.get("Edges", ())
    ]:
        raise RuntimeError("body-first Terminal native edge assignment differs")
    if set(item["object_name"] for item in edges) != set(native["signal_objects"]):
        raise RuntimeError("body-first Terminal contact ownership differs")


def _validate_body_terminal_properties(
    context: BodyTerminalContext,
    record: dict[str, Any],
    port: Any,
    native: dict[str, Any],
    expected_type: str,
) -> None:
    properties = native.get("boundary_properties", {})
    terminal_properties = native.get("terminal_properties", {})
    renorm = port.renormalize if isinstance(port, LumpedTerminalPort) else False
    current_wave = (
        not isinstance(port, LumpedTerminalPort)
        and context.receipt.get("runtime_source", {}).get("schema_version")
        in CURRENT_TERMINAL_NATIVE_SCHEMAS
    )
    if (
        properties.get("Type") != expected_type
        or properties.get("Renorm All Terminals") != renorm
        or (
            properties.get("Deembed") not in (True, False, "true", "false")
            if current_wave
            else properties.get("Deembed") != (not isinstance(port, LumpedTerminalPort))
        )
        or (
            isinstance(port, LumpedTerminalPort)
            and terminal_properties.get("Terminal Renormalizing Impedance")
            != f"{port.impedance_ohm:g}ohm"
        )
        or (
            current_wave
            and native.get("inactive_renormalizing_impedance")
            != terminal_properties.get("Terminal Renormalizing Impedance")
        )
        or (
            not isinstance(port, LumpedTerminalPort)
            and not current_wave
            and terminal_properties.get("Terminal Renormalizing Impedance") != "50ohm"
        )
        or (current_wave and properties.get("Deembed Dist") != f"{port.deembed_um:g}um")
    ):
        raise RuntimeError("body-first Terminal native electrical treatment differs")


def validate_terminal_native_evidence(
    ports: Any, spec: HfssDrivenSpec, *, current: bool = False
) -> None:
    """Verify terminal excitation evidence against declared source ports."""
    if not isinstance(ports, list) or len(ports) != 2:
        raise RuntimeError("completed receipt has invalid terminal ports")
    for record, declared_port in zip(ports, spec.ports, strict=True):
        port = cast(TerminalPort, declared_port)
        if not isinstance(record, dict) or "unresolved_native" in record:
            raise RuntimeError(
                "completed receipt has unresolved terminal native evidence"
            )
        terminal_name = record.get("terminal_excitation")
        if (
            record.get("index") != port.index
            or record.get("boundary") != port.name
            or not isinstance(terminal_name, str)
            or not terminal_name
            or not isinstance(record.get("face_id"), int)
            or not isinstance(record.get("face_center_um"), list)
            or len(record["face_center_um"]) != 3
        ):
            raise RuntimeError("completed receipt terminal identity is invalid")
        requested = record.get("requested")
        if requested != {
            "reference_objects": list(port.reference_objects),
            "renormalize": False,
            "deembed_um": port.deembed_um,
        }:
            raise RuntimeError("completed receipt terminal request does not match spec")
        native = record.get("native")
        if not isinstance(native, dict):
            raise TypeError("completed receipt has no native terminal evidence")
        if (
            native.get("excitation_names")
            != [item.name for item in spec.ports[: port.index]]
            or native.get("terminal_names") != [terminal_name]
            or native.get("reference_conductors") != list(port.reference_objects)
            or not isinstance(native.get("reference_conductor_ids"), list)
            or not native["reference_conductor_ids"]
            or not all(
                isinstance(value, int) for value in native["reference_conductor_ids"]
            )
        ):
            raise RuntimeError("completed receipt terminal native identity mismatch")
        boundary = native.get("boundary_properties")
        if not isinstance(boundary, dict):
            raise TypeError(
                "completed receipt has invalid terminal boundary properties"
            )
        terminal = native.get("terminal_properties")
        if not isinstance(terminal, dict):
            raise TypeError(
                "completed receipt has invalid terminal excitation properties"
            )
        expected_boundary = {
            "Deembed": boundary.get("Deembed") if current else True,
            "Deembed Dist": f"{port.deembed_um:g}um",
            "Name": port.name,
            "Num Terminals": "1",
            "Renorm All Terminals": False,
            "Type": "Wave Port",
            "Wave Port Type": "Terminal",
        }
        expected_terminal = {
            "Name": terminal_name,
            "Port Name": port.name,
            "Terminal Renormalizing Impedance": (
                terminal.get("Terminal Renormalizing Impedance") if current else "50ohm"
            ),
            "Type": "Terminal",
        }
        if (
            boundary != expected_boundary
            or terminal != expected_terminal
            or (
                current
                and (
                    boundary.get("Deembed") not in (True, False, "true", "false")
                    or native.get("inactive_renormalizing_impedance")
                    != terminal.get("Terminal Renormalizing Impedance")
                )
            )
        ):
            raise RuntimeError("completed receipt terminal native property mismatch")
