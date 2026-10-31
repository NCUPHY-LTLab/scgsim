"""Body-first HFSS terminal-port assignment and its native evidence.

Owns assigning lumped terminal ports onto native conductor objects and binding
the live boundary, terminal and reference readback to each port record. It does
not own the body geometry or the explicit-port (non body-first) path."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from scgsim.aedt.runtime.families.hfss_readback import (
    _face_for_side,
    _native_oo_properties,
    _native_terminal_reference_ids,
    _require_native_properties,
)
from scgsim.aedt.runtime.native.common import (
    detached_data,
)
from scgsim.aedt.specs.common import (
    LumpedTerminalPort,
)
from scgsim.aedt.specs.hfss_prepared import (
    HfssDrivenGeometrySpec,
)


def _assign_body_terminal_ports(
    hfss: Any, spec: HfssDrivenGeometrySpec, geometry: dict[str, Any]
) -> list[dict[str, Any]]:
    """Use authored Entities and supports; AutoIdentify does not author ownership."""
    from scgsim.aedt.runtime.native.hfss_geometry import native_conductor_objects
    from scgsim.aedt.runtime.native.hfss_lumped import assign_terminal_lumped

    supports = {
        item["support_id"]: item for item in geometry.get("lumped_supports", ())
    }
    records = []
    for port in spec.ports:
        record: dict[str, Any] = {}
        try:
            signal_ids = port.signal_entity_ids
            reference_ids = (
                port.reference_entity_ids
                if isinstance(port, LumpedTerminalPort)
                else port.reference_objects
            )
            signals = native_conductor_objects(spec.geometry, geometry, signal_ids)
            references = native_conductor_objects(
                spec.geometry, geometry, reference_ids
            )
            if isinstance(port, LumpedTerminalPort):
                support = supports[port.support_id]
                record = assign_terminal_lumped(
                    hfss,
                    hfss.modeler[support["object_name"]],
                    name=port.name,
                    reference_objects=references,
                    impedance_ohm=port.impedance_ohm,
                    renormalize=port.renormalize,
                    deembed_um=port.deembed_um,
                )
                record["support"] = detached_data(support)
            else:
                region = hfss.modeler["Region"]
                faces = [
                    (int(face.id), tuple(float(value) for value in face.center))
                    for face in region.faces
                ]
                face_id = _face_for_side(faces, port.side)
                before = set(hfss.oboundary.GetExcitationsOfType("Terminal"))
                boundary = hfss.wave_port(
                    face_id,
                    reference=references,
                    name=port.name,
                    renormalize=False,
                    deembed=f"{port.deembed_um:g}um",
                    terminals_rename=False,
                )
                terminals = sorted(
                    set(hfss.oboundary.GetExcitationsOfType("Terminal")) - before
                )
                if boundary is False or boundary is None or len(terminals) != 1:
                    raise RuntimeError(
                        f"body-first Wave Terminal assignment failed: {port.name!r}"
                    )
                record = {
                    "boundary": boundary.name,
                    "terminal_excitation": terminals[0],
                    "face_id": face_id,
                    "native_terminal_names": terminals,
                    "face_center_um": list(dict(faces)[face_id]),
                    "assignment_api": "wave_port",
                }
            records.append(
                {
                    "index": port.index,
                    "kind": "lumped"
                    if isinstance(port, LumpedTerminalPort)
                    else "wave",
                    "requested": port.to_payload()
                    if isinstance(port, LumpedTerminalPort)
                    else {
                        "index": port.index,
                        "name": port.name,
                        "side": port.side,
                        "reference_objects": list(port.reference_objects),
                        "signal_entity_ids": list(signal_ids),
                        "renormalize": False,
                        "deembed_um": port.deembed_um,
                    },
                    "source_binding": {
                        "signal_entity_ids": list(signal_ids),
                        "reference_entity_ids": list(reference_ids),
                        "signal_objects": signals,
                        "reference_objects": references,
                    },
                    **record,
                }
            )
        except Exception as exc:
            exc.add_note(
                "body-first Terminal assignment: "
                + json.dumps(
                    {
                        "port_index": port.index,
                        "port_name": port.name,
                        "record": record,
                    },
                    default=str,
                    sort_keys=True,
                )
            )
            raise
    return records


# Length: over 150 lines. One loop over ports reads the saved boundary, terminal
# and reference records and keeps the raw native record before any validation, so
# a malformed record stays in the error; the validation order is the error order.
def _bind_body_terminal_evidence(
    hfss: Any,
    spec: HfssDrivenGeometrySpec,
    geometry: dict[str, Any],
    ports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Read saved Terminal assignments and their actual conductor owners."""
    boundaries = hfss.design_properties["BoundarySetup"]["Boundaries"]
    reference_ids = _native_terminal_reference_ids(hfss)
    reference_names = [hfss.oeditor.GetObjectNameByID(value) for value in reference_ids]
    for record, port in zip(ports, spec.ports, strict=True):
        try:
            terminal_name = record["terminal_excitation"]
            native_terminal = boundaries[terminal_name]
            native_boundary = boundaries[record["boundary"]]
            native = {
                "saved_boundary": detached_data(native_boundary),
                "saved_terminal": detached_data(native_terminal),
            }
            # Retain the raw child record before reading its assignment list so
            # any malformed or unsupported native shape remains in the error.
            record["native"] = native

            def assignment_ids(key: str) -> list[int] | None:
                if key not in native_terminal:  # noqa: B023 - called within the iteration that defines it
                    return None
                values = native_terminal[key]  # noqa: B023 - called within the iteration that defines it
                if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                    raise RuntimeError(
                        f"saved Terminal {key} assignment is invalid: {port.name!r}"  # noqa: B023 - called within the iteration that defines it
                    )
                if not values:
                    return None
                return [int(value) for value in values]

            edges = assignment_ids("Edges")
            faces = assignment_ids("Faces")
            if (edges is None) == (faces is None):
                raise RuntimeError(
                    f"saved Terminal child assignment must have exactly one "
                    f"nonempty Edges or Faces list: {port.name!r}"
                )
            owners = []
            if edges is not None:
                edge_records = []
                native["terminal_edges"] = edge_records
                for edge_id in edges:
                    matches = [
                        item["object_name"]
                        for item in geometry["objects"]
                        if item["kind"] == "conductor"
                        and edge_id
                        in {
                            int(edge.id)
                            for edge in hfss.modeler[item["object_name"]].edges
                        }
                    ]
                    if len(matches) != 1:
                        raise RuntimeError(
                            f"native Terminal edge owner is ambiguous: {edge_id}"
                        )
                    edge_records.append(
                        {
                            "edge_id": edge_id,
                            "object_name": matches[0],
                            "native_object_id": int(hfss.modeler[matches[0]].id),
                        }
                    )
                    if matches[0] not in owners:
                        owners.append(matches[0])
            else:
                if faces is None:
                    raise RuntimeError(
                        f"saved Terminal child assignment is unavailable: {port.name!r}"
                    )
                face_records = []
                native["terminal_assignment_kind"] = "Faces"
                native["terminal_faces"] = face_records
                for face_id in faces:
                    matches = [
                        item["object_name"]
                        for item in geometry["objects"]
                        if item["kind"] == "conductor"
                        and face_id
                        in {
                            int(face.id)
                            for face in hfss.modeler[item["object_name"]].faces
                        }
                    ]
                    if len(matches) != 1:
                        raise RuntimeError(
                            f"native Terminal face owner is ambiguous: {face_id}"
                        )
                    face_records.append(
                        {
                            "face_id": face_id,
                            "object_name": matches[0],
                            "native_object_id": int(hfss.modeler[matches[0]].id),
                        }
                    )
                    if matches[0] not in owners:
                        owners.append(matches[0])
            native["signal_objects"] = owners
            native["reference_conductor_ids"] = reference_ids
            native["reference_conductors"] = reference_names
            native["boundary_properties"] = _native_oo_properties(
                hfss, f"Excitations\\{record['boundary']}"
            )
            native["terminal_properties"] = _native_oo_properties(
                hfss, f"Excitations\\{record['boundary']}\\{terminal_name}"
            )
            if not owners or not set(owners).issubset(
                record["source_binding"]["signal_objects"]
            ):
                raise RuntimeError(
                    f"native Terminal signal ownership differs: {port.name!r}"
                )
            if not set(record["source_binding"]["reference_objects"]).issubset(
                reference_names
            ):
                raise RuntimeError(f"native Terminal references differ: {port.name!r}")
            expected_type = (
                "Lumped Port" if isinstance(port, LumpedTerminalPort) else "Wave Port"
            )
            if (
                native_boundary["BoundType"] != expected_type
                or native["terminal_properties"].get("Port Name") != port.name
            ):
                raise RuntimeError(
                    f"saved Terminal family/parent differs: {port.name!r}"
                )
            if [int(value) for value in native_boundary["Faces"]] != [
                record["face_id"]
            ]:
                raise RuntimeError(
                    f"saved Terminal support assignment differs: {port.name!r}"
                )
            renorm = port.renormalize if isinstance(port, LumpedTerminalPort) else False
            _require_native_properties(
                native["boundary_properties"],
                {
                    "Type": expected_type,
                    "Renorm All Terminals": renorm,
                    **(
                        {"Deembed": False}
                        if isinstance(port, LumpedTerminalPort)
                        else {}
                    ),
                },
                port.name,
            )
            if isinstance(port, LumpedTerminalPort):
                _require_native_properties(
                    native["terminal_properties"],
                    {"Terminal Renormalizing Impedance": f"{port.impedance_ohm:g}ohm"},
                    terminal_name,
                )
            else:
                # With renormalization disabled this is native metadata, not a
                # source-authored load or solved characteristic impedance.
                if native["boundary_properties"].get("Deembed") not in (
                    True,
                    False,
                    "true",
                    "false",
                ):
                    raise RuntimeError("native Wave deembed flag is unavailable")
                native["inactive_renormalizing_impedance"] = native[
                    "terminal_properties"
                ].get("Terminal Renormalizing Impedance")
                _require_native_properties(
                    native["boundary_properties"],
                    {"Deembed Dist": f"{port.deembed_um:g}um"},
                    port.name,
                )
        except Exception as exc:
            exc.add_note(
                "saved body-first Terminal readback: "
                + json.dumps(
                    {
                        "port_index": port.index,
                        "port_name": port.name,
                        "record": record,
                    },
                    default=str,
                    sort_keys=True,
                )
            )
            raise
    expected_references = {
        name
        for record in ports
        for name in record["source_binding"]["reference_objects"]
    }
    if set(reference_names) != expected_references:
        raise RuntimeError(
            "body-first global native Terminal reference inventory differs"
        )
    return ports
