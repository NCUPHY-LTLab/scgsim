"""Check lumped loads and body-first Terminal ports in a completed receipt.

Body-first means the geometry was built from bodies (conductors and lumped supports)
rather than imported from GDS. Owns: lumped RLC (resistor, inductor, capacitor)
readback and the body-first Driven Terminal port readback. It does not own plain
GDS-imported Terminal ports (``hfss_readback``).
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from scgsim.aedt._data import detached
from scgsim.aedt._rlc import rlc_native_topology, rlc_scalar_si
from scgsim.aedt.results.readback_common import (
    is_sha256_hex,
    uses_current_terminal_rules,
)
from scgsim.aedt.specs.common import LumpedTerminalPort
from scgsim.aedt.specs.hfss_prepared import HfssDrivenGeometrySpec

# ---------------------------------------------------------------------------
# Lumped RLC loads
# ---------------------------------------------------------------------------


def validate_lumped_readback(receipt: dict[str, Any], prepared: Any) -> None:
    """Bind the receipt's lumped supports and RLC loads to the prepared source."""
    supports = prepared.source.get("lumped_supports", ())
    if not supports and not prepared.lumped_rlcs:
        return
    geometry = receipt.get("geometry")
    if (
        not isinstance(geometry, dict)
        or geometry.get("source_sha256") != prepared.source_sha256
    ):
        raise RuntimeError("native support source identity differs")
    by_id = _check_native_supports(geometry, supports)
    records = geometry.get("lumped_rlcs", [])
    if len(records) != len(prepared.lumped_rlcs):
        raise RuntimeError("generic RLC inventory differs")
    for treatment, record in zip(prepared.lumped_rlcs, records, strict=True):
        support = by_id[treatment.support_id]
        expected = {
            "topology": treatment.topology,
            "native_topology": rlc_native_topology(treatment.topology),
            "contact_points_um": support["effective"]["contact_points_um"],
            "resistance_ohm": treatment.resistance_ohm,
            "inductance_h": treatment.inductance_h,
            "capacitance_f": treatment.capacitance_f,
        }
        _check_rlc_identity(record, treatment, support, expected)
        _check_rlc_saved_line(record, support, expected)
        _check_rlc_native_values(record, expected)


def _check_native_supports(
    geometry: dict[str, Any], supports: Any
) -> dict[str, dict[str, Any]]:
    """Each requested support must be a single native sheet face; index by support ID."""
    native_supports = geometry.get("lumped_supports")
    if not isinstance(native_supports, list) or len(native_supports) != len(supports):
        raise RuntimeError("native support inventory differs")
    by_id = {}
    for requested, native in zip(supports, native_supports, strict=True):
        requested = detached(requested)
        if (
            any(
                native.get(key) != requested[key]
                for key in ("support_id", "source", "effective")
            )
            or native.get("native_object_type") != "Sheet"
            or not isinstance(native.get("native_object_id"), int)
            or len(native.get("native_face_ids", ())) != 1
        ):
            raise RuntimeError("native neutral support attribution differs")
        by_id[native["support_id"]] = native
    return by_id


def _check_rlc_identity(
    record: dict[str, Any],
    treatment: Any,
    support: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    if (
        record.get("support_id") != treatment.support_id
        or record.get("source") != treatment.to_payload()
        or record.get("requested") != expected
        or record.get("object_name") != support["object_name"]
        or record.get("native_object_id") != support["native_object_id"]
        or record.get("native_face_ids") != support["native_face_ids"]
    ):
        raise RuntimeError("generic RLC source/native identity differs")
    assignment = record.get("assignment", {})
    if assignment.get("covered_objects") != [support["object_name"]] or assignment.get(
        "covered_faces"
    ) != sorted(support["native_face_ids"]):
        raise RuntimeError("generic RLC assignment differs")


def _check_rlc_saved_line(
    record: dict[str, Any], support: dict[str, Any], expected: dict[str, Any]
) -> None:
    """The saved boundary's directed current line must run between the contact points."""
    saved = record.get("saved_boundary", {})
    positions = saved.get("CurrentLine", {}).get("GeometryPosition", ())
    if (
        saved.get("BoundType") != "Lumped RLC"
        or record.get("saved_model_units") != "um"
        or len(positions) != 2
        or not is_sha256_hex(record.get("saved_project_sha256", ""))
        or record.get("saved_contact_points_um") != expected["contact_points_um"]
    ):
        raise RuntimeError("saved generic RLC directed line evidence differs")
    for point, position in zip(
        record["saved_contact_points_um"], positions, strict=True
    ):
        if (
            position.get("IsAttachedToEntity") is False
            and position.get("PositionType") == "AbsolutePosition"
        ):
            if [float(position[axis + "Position"]) for axis in "XYZ"] != point:
                raise RuntimeError("saved generic RLC absolute position differs")
        elif (
            position.get("IsAttachedToEntity") is True
            and position.get("PositionType") == "EdgeCenter"
        ):
            _check_attached_contact_edge(record, support, position, point)
        else:
            raise RuntimeError("saved generic RLC line position kind unsupported")


def _check_attached_contact_edge(
    record: dict[str, Any],
    support: dict[str, Any],
    position: dict[str, Any],
    point: Any,
) -> None:
    """A line end attached to an edge centre needs exactly one matching edge record."""
    contacts = [
        item
        for item in record.get("attached_contact_edges", ())
        if item.get("edge_id") == position.get("EntityID")
    ]
    if (
        len(contacts) != 1
        or contacts[0].get("midpoint_um") != point
        or contacts[0].get("object_name") != support["object_name"]
        or contacts[0].get("native_object_id") != support["native_object_id"]
    ):
        raise RuntimeError("saved generic RLC attached edge identity unavailable")


def _check_rlc_native_values(record: dict[str, Any], expected: dict[str, Any]) -> None:
    """Native enable flags and SI values must equal the requested component values."""
    properties = record.get("native_properties", {})
    if properties.get("RLC Type") != expected["native_topology"]:
        raise RuntimeError("generic RLC native topology differs")
    for key, enable, prop in (
        ("resistance_ohm", "Use Resist", "Resistance"),
        ("inductance_h", "Use Induct", "Inductance"),
        ("capacitance_f", "Use Cap", "Capacitance"),
    ):
        raw = properties.get(enable)
        if raw in (True, "true", "True", 1):
            enabled = True
        elif raw in (False, "false", "False", 0):
            enabled = False
        else:
            raise RuntimeError("generic RLC enable readback unavailable")
        value = expected[key]
        if enabled != (value is not None):
            raise RuntimeError("generic RLC enable differs")
        if enabled:
            observed = rlc_scalar_si(properties.get(prop), key)
            if observed != Decimal(str(value)) or observed != Decimal(
                record["normalized_values_si"][key]
            ):
                raise RuntimeError("generic RLC native component value differs")


# ---------------------------------------------------------------------------
# Body-first Driven Terminal ports
# ---------------------------------------------------------------------------


def validate_body_terminal_readback(
    receipt: dict[str, Any], spec: HfssDrivenGeometrySpec
) -> None:
    """Bind each Terminal port to the source Entities and native bodies it names."""
    geometry = receipt.get("geometry", {})
    if geometry.get("source_sha256") != spec.geometry.source_sha256:
        raise RuntimeError("body-first Terminal geometry source differs")
    validate_lumped_readback(receipt, spec.geometry)
    conductors = spec.geometry.source["conductors"]
    objects = geometry.get("objects", [])

    # Every port must see the same set of reference conductors.
    global_references = {
        name
        for port in spec.ports
        for name in _native_body_names(_reference_ids(port), conductors, objects)
    }
    for record, port in zip(receipt["ports"], spec.ports, strict=True):
        # Built first: it raises if a source Entity has no native body.
        expected = _expected_source_binding(port, conductors, objects)
        _check_port_request(record, port, geometry)
        native = record.get("native", {})
        _check_port_ownership(record, native, port, expected, global_references)
        _check_port_native_family(record, native, port)
        _check_port_contact_assignment(native, objects)
        _check_port_electrical_treatment(receipt, native, port)


def _reference_ids(port: Any) -> Any:
    """Lumped ports reference source Entities; wave ports reference object names."""
    if isinstance(port, LumpedTerminalPort):
        return port.reference_entity_ids
    return port.reference_objects


def _native_body_names(entity_ids: Any, conductors: Any, objects: Any) -> list[str]:
    """Native conductor body names for source Entities, in first-seen order.

    An Entity id may be a conductor's own ``semantic_id`` or the ``source_semantic_id``
    it was derived from, and one Entity can own several native bodies.
    """
    result: list[str] = []
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


def _check_port_request(
    record: dict[str, Any], port: Any, geometry: dict[str, Any]
) -> None:
    """The authored port treatment (and any lumped support) must be what was applied."""
    requested = (
        port.to_payload()
        if isinstance(port, LumpedTerminalPort)
        else {
            "index": port.index,
            "name": port.name,
            "side": port.side,
            "reference_objects": list(port.reference_objects),
            "signal_entity_ids": list(port.signal_entity_ids),
            "renormalize": False,
            "deembed_um": port.deembed_um,
        }
    )
    if record.get("requested") != requested:
        raise RuntimeError("body-first Terminal authored treatment differs")
    if isinstance(port, LumpedTerminalPort):
        support = next(
            item
            for item in geometry["lumped_supports"]
            if item["support_id"] == port.support_id
        )
        if (
            record.get("support") != support
            or record.get("object_name") != support["object_name"]
        ):
            raise RuntimeError("body-first Lumped support identity differs")


def _expected_source_binding(
    port: Any, conductors: Any, objects: Any
) -> dict[str, Any]:
    reference_ids = _reference_ids(port)
    return {
        "signal_entity_ids": list(port.signal_entity_ids),
        "reference_entity_ids": list(reference_ids),
        "signal_objects": _native_body_names(
            port.signal_entity_ids, conductors, objects
        ),
        "reference_objects": _native_body_names(reference_ids, conductors, objects),
    }


def _check_port_ownership(
    record: dict[str, Any],
    native: dict[str, Any],
    port: Any,
    expected: dict[str, Any],
    global_references: set[str],
) -> None:
    if (
        record.get("index") != port.index
        or record.get("boundary") != port.name
        or record.get("source_binding") != expected
        or not native.get("signal_objects")
        or not set(native["signal_objects"]).issubset(expected["signal_objects"])
        or set(native.get("reference_conductors", ())) != global_references
    ):
        raise RuntimeError("body-first Terminal native ownership differs")


def _check_port_native_family(
    record: dict[str, Any], native: dict[str, Any], port: Any
) -> None:
    boundary = native.get("saved_boundary", {})
    if (
        boundary.get("BoundType") != _native_port_type(port)
        or native.get("terminal_properties", {}).get("Port Name") != port.name
    ):
        raise RuntimeError("body-first Terminal native family differs")
    if [int(value) for value in boundary.get("Faces", ())] != [record.get("face_id")]:
        raise RuntimeError("body-first Terminal native support face differs")


def _native_port_type(port: Any) -> str:
    return "Lumped Port" if isinstance(port, LumpedTerminalPort) else "Wave Port"


def _check_port_contact_assignment(native: dict[str, Any], objects: Any) -> None:
    """The terminal is assigned to faces or to edges, and must own the signal bodies."""
    terminal = native.get("saved_terminal", {})
    assignment_kind = native.get("terminal_assignment_kind")
    if assignment_kind == "Faces":
        _check_face_assignment(native, terminal, objects)
    elif assignment_kind is None:
        _check_edge_assignment(native, terminal)
    else:
        raise RuntimeError("body-first Terminal native assignment kind is invalid")


def _check_face_assignment(
    native: dict[str, Any], terminal: dict[str, Any], objects: Any
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
        for item in objects
        if isinstance(item, dict) and item.get("kind") == "conductor"
    ]
    face_ids = []
    face_owners = []
    for face_record in face_records:
        _check_face_owner_record(face_record)
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


def _check_face_owner_record(face_record: Any) -> None:
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
        raise RuntimeError("body-first Terminal native face owner record is invalid")


def _check_edge_assignment(native: dict[str, Any], terminal: dict[str, Any]) -> None:
    if terminal.get("Faces") not in (None, [], ()):
        raise RuntimeError("body-first Terminal child assignment kind is ambiguous")
    edges = native.get("terminal_edges", [])
    if not edges or [item["edge_id"] for item in edges] != [
        int(value) for value in terminal.get("Edges", ())
    ]:
        raise RuntimeError("body-first Terminal native edge assignment differs")
    if {item["object_name"] for item in edges} != set(native["signal_objects"]):
        raise RuntimeError("body-first Terminal contact ownership differs")


def _check_port_electrical_treatment(
    receipt: dict[str, Any], native: dict[str, Any], port: Any
) -> None:
    """Renormalization, de-embedding and impedance as AEDT reported them.

    Under the current (V19) rules a wave port records its observed de-embed flag
    and distance; earlier runs must show the fixed values the runtime then forced.
    """
    properties = native.get("boundary_properties", {})
    terminal_properties = native.get("terminal_properties", {})
    lumped = isinstance(port, LumpedTerminalPort)
    renorm = port.renormalize if lumped else False
    current_wave = not lumped and uses_current_terminal_rules(receipt)
    if (
        properties.get("Type") != _native_port_type(port)
        or properties.get("Renorm All Terminals") != renorm
        or (
            properties.get("Deembed") not in (True, False, "true", "false")
            if current_wave
            else properties.get("Deembed") != (not lumped)
        )
        or (
            lumped
            and terminal_properties.get("Terminal Renormalizing Impedance")
            != f"{port.impedance_ohm:g}ohm"
        )
        or (
            current_wave
            and native.get("inactive_renormalizing_impedance")
            != terminal_properties.get("Terminal Renormalizing Impedance")
        )
        or (
            not lumped
            and not current_wave
            and terminal_properties.get("Terminal Renormalizing Impedance") != "50ohm"
        )
        or (current_wave and properties.get("Deembed Dist") != f"{port.deembed_um:g}um")
    ):
        raise RuntimeError("body-first Terminal native electrical treatment differs")
