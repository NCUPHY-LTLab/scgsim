"""Body-first HFSS terminal-port assignment and its native evidence.

Owns assigning lumped terminal ports onto native conductor objects and binding
the live boundary, terminal and reference readback to each port record. It does
not own the body geometry or the explicit-port (non body-first) path."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from scgsim.aedt.runtime.families.hfss_readback import (
    face_for_side,
    native_oo_properties,
    native_terminal_reference_ids,
    require_native_properties,
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


def assign_body_terminal_ports(
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
                    hfss, hfss.modeler[support["object_name"]], port, references
                )
                record["support"] = detached_data(support)
            else:
                region = hfss.modeler["Region"]
                faces = [
                    (int(face.id), tuple(float(value) for value in face.center))
                    for face in region.faces
                ]
                face_id = face_for_side(faces, port.side)
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


@dataclass(frozen=True)
class _SavedTerminalSetup:
    """Saved boundary records and the global terminal reference conductors."""

    boundaries: dict[str, Any]
    reference_ids: list[int]
    reference_names: list[str]


def bind_body_terminal_evidence(
    hfss: Any,
    spec: HfssDrivenGeometrySpec,
    geometry: dict[str, Any],
    ports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Read saved Terminal assignments and their actual conductor owners."""
    saved = _read_saved_terminal_setup(hfss)
    for record, port in zip(ports, spec.ports, strict=True):
        try:
            _bind_port_terminal(hfss, geometry, record, port, saved)
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
    if set(saved.reference_names) != expected_references:
        raise RuntimeError(
            "body-first global native Terminal reference inventory differs"
        )
    return ports


def _read_saved_terminal_setup(hfss: Any) -> _SavedTerminalSetup:
    """Read the saved boundary records and the conductors AEDT uses as references."""
    boundaries = hfss.design_properties["BoundarySetup"]["Boundaries"]
    reference_ids = native_terminal_reference_ids(hfss)
    reference_names = [hfss.oeditor.GetObjectNameByID(value) for value in reference_ids]
    return _SavedTerminalSetup(boundaries, reference_ids, reference_names)


def _bind_port_terminal(
    hfss: Any,
    geometry: dict[str, Any],
    record: dict[str, Any],
    port: Any,
    saved: _SavedTerminalSetup,
) -> None:
    """Attach the saved boundary and terminal readback to ``record`` and verify it."""
    terminal_name = record["terminal_excitation"]
    native_terminal = saved.boundaries[terminal_name]
    native_boundary = saved.boundaries[record["boundary"]]
    native = {
        "saved_boundary": detached_data(native_boundary),
        "saved_terminal": detached_data(native_terminal),
    }
    # Retain the raw child record before reading its assignment list so
    # any malformed or unsupported native shape remains in the error.
    record["native"] = native

    owners = _read_terminal_owners(hfss, geometry, native, native_terminal, port.name)
    native["signal_objects"] = owners
    native["reference_conductor_ids"] = saved.reference_ids
    native["reference_conductors"] = saved.reference_names
    native["boundary_properties"] = native_oo_properties(
        hfss, f"Excitations\\{record['boundary']}"
    )
    native["terminal_properties"] = native_oo_properties(
        hfss, f"Excitations\\{record['boundary']}\\{terminal_name}"
    )
    if not owners or not set(owners).issubset(
        record["source_binding"]["signal_objects"]
    ):
        raise RuntimeError(f"native Terminal signal ownership differs: {port.name!r}")
    if not set(record["source_binding"]["reference_objects"]).issubset(
        saved.reference_names
    ):
        raise RuntimeError(f"native Terminal references differ: {port.name!r}")
    _verify_terminal_family(native, native_boundary, record, port, terminal_name)


def _terminal_assignment_ids(
    native_terminal: dict[str, Any], key: str, port_name: str
) -> list[int] | None:
    """Return the ``Edges`` or ``Faces`` ID list of a saved terminal, or None if empty/absent."""
    if key not in native_terminal:
        return None
    values = native_terminal[key]
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise RuntimeError(f"saved Terminal {key} assignment is invalid: {port_name!r}")
    if not values:
        return None
    return [int(value) for value in values]


def _read_terminal_owners(
    hfss: Any,
    geometry: dict[str, Any],
    native: dict[str, Any],
    native_terminal: dict[str, Any],
    port_name: str,
) -> list[str]:
    """Find the conductor object that owns each edge or face the terminal is assigned to.

    A saved terminal carries exactly one nonempty ``Edges`` or ``Faces`` list. The
    per-ID records are stored on ``native``; the owner names are returned in order of
    first appearance.
    """
    edges = _terminal_assignment_ids(native_terminal, "Edges", port_name)
    faces = _terminal_assignment_ids(native_terminal, "Faces", port_name)
    if (edges is None) == (faces is None):
        raise RuntimeError(
            f"saved Terminal child assignment must have exactly one "
            f"nonempty Edges or Faces list: {port_name!r}"
        )
    owners: list[str] = []
    if edges is not None:
        kind, ids = "edge", edges
        owner_records = native["terminal_edges"] = []
    else:
        kind, ids = "face", faces
        native["terminal_assignment_kind"] = "Faces"
        owner_records = native["terminal_faces"] = []
    for native_id in ids:
        owner = _conductor_owning(hfss, geometry, kind, native_id)
        owner_records.append(
            {
                f"{kind}_id": native_id,
                "object_name": owner,
                "native_object_id": int(hfss.modeler[owner].id),
            }
        )
        if owner not in owners:
            owners.append(owner)
    return owners


def _conductor_owning(
    hfss: Any, geometry: dict[str, Any], kind: str, native_id: int
) -> str:
    """Return the one conductor object that has the native edge or face ``native_id``."""
    matches = [
        item["object_name"]
        for item in geometry["objects"]
        if item["kind"] == "conductor"
        and native_id
        in {
            int(part.id)
            for part in getattr(hfss.modeler[item["object_name"]], f"{kind}s")
        }
    ]
    if len(matches) != 1:
        raise RuntimeError(f"native Terminal {kind} owner is ambiguous: {native_id}")
    return matches[0]


def _verify_terminal_family(
    native: dict[str, Any],
    native_boundary: dict[str, Any],
    record: dict[str, Any],
    port: Any,
    terminal_name: str,
) -> None:
    """Check the saved boundary type, parent and renormalization settings match the port."""
    expected_type = (
        "Lumped Port" if isinstance(port, LumpedTerminalPort) else "Wave Port"
    )
    if (
        native_boundary["BoundType"] != expected_type
        or native["terminal_properties"].get("Port Name") != port.name
    ):
        raise RuntimeError(f"saved Terminal family/parent differs: {port.name!r}")
    if [int(value) for value in native_boundary["Faces"]] != [record["face_id"]]:
        raise RuntimeError(f"saved Terminal support assignment differs: {port.name!r}")
    renorm = port.renormalize if isinstance(port, LumpedTerminalPort) else False
    require_native_properties(
        native["boundary_properties"],
        {
            "Type": expected_type,
            "Renorm All Terminals": renorm,
            **({"Deembed": False} if isinstance(port, LumpedTerminalPort) else {}),
        },
        port.name,
    )
    if isinstance(port, LumpedTerminalPort):
        require_native_properties(
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
        native["inactive_renormalizing_impedance"] = native["terminal_properties"].get(
            "Terminal Renormalizing Impedance"
        )
        require_native_properties(
            native["boundary_properties"],
            {"Deembed Dist": f"{port.deembed_um:g}um"},
            port.name,
        )
