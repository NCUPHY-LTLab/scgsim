"""Own HFSS port construction and saved source-to-terminal evidence."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

from scgsim.aedt.runtime.families.hfss_port_properties import (
    native_modal_oo_readback,
    native_oo_properties,
    native_position_um,
    native_terminal_readback,
    native_terminal_reference_ids,
    require_native_properties,
    text_value,
)
from scgsim.aedt.runtime.families.hfss_requests import HfssGdsSpec
from scgsim.aedt.runtime.native.common import (
    detached_data,
)
from scgsim.aedt.runtime.native.hfss_geometry import native_conductor_objects
from scgsim.aedt.runtime.native.hfss_lumped import assign_terminal_lumped
from scgsim.aedt.specs.common import (
    LumpedTerminalPort,
    ModalPort,
    TerminalPort,
)
from scgsim.aedt.specs.hfss import (
    HfssDrivenGeometrySpec,
    HfssDrivenSpec,
    HfssEigenmodeSpec,
)


def assign_body_terminal_ports(
    hfss: Any, spec: HfssDrivenGeometrySpec, geometry: dict[str, Any]
) -> list[dict[str, Any]]:
    """Use authored Entities and supports; AutoIdentify does not author ownership."""

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
                    port,
                    reference_objects=references,
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


def bind_body_terminal_evidence(
    hfss: Any,
    spec: HfssDrivenGeometrySpec,
    geometry: dict[str, Any],
    ports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Read saved Terminal assignments and their actual conductor owners."""
    boundaries = hfss.design_properties["BoundarySetup"]["Boundaries"]
    reference_ids = native_terminal_reference_ids(hfss)
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

            edges = _terminal_assignment_ids(native_terminal, "Edges", port.name)
            faces = _terminal_assignment_ids(native_terminal, "Faces", port.name)
            if (edges is None) == (faces is None):
                raise RuntimeError(
                    f"saved Terminal child assignment must have exactly one "
                    f"nonempty Edges or Faces list: {port.name!r}"
                )
            owners = _read_body_terminal_owners(
                hfss, geometry, native, edges, faces, port.name
            )
            native["signal_objects"] = owners
            native["reference_conductor_ids"] = reference_ids
            native["reference_conductors"] = reference_names
            native["boundary_properties"] = native_oo_properties(
                hfss, f"Excitations\\{record['boundary']}"
            )
            native["terminal_properties"] = native_oo_properties(
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
            require_native_properties(
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
                native["inactive_renormalizing_impedance"] = native[
                    "terminal_properties"
                ].get("Terminal Renormalizing Impedance")
                require_native_properties(
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


def assign_ports(hfss: Any, spec: HfssGdsSpec) -> list[dict[str, Any]]:
    """Assign GDS-family modal or terminal ports and retain their native identities."""
    if isinstance(spec, HfssEigenmodeSpec):
        if hfss.get_oo_name(hfss.odesign, "Excitations"):
            raise RuntimeError("HFSS Eigenmode design must not contain excitations")
        return []
    region = hfss.modeler.get_object_from_name("Region")
    if region is None:
        raise RuntimeError("Region is missing")
    faces = [
        (int(face.id), tuple(float(value) for value in face.center))
        for face in region.faces
    ]
    centers = {face_id: center for face_id, center in faces}
    records: list[dict[str, Any]] = []
    for port, effective_port in zip(spec.ports, spec.effective_ports, strict=True):
        face_id = _face_for_side(faces, port.side)
        if isinstance(port, TerminalPort):
            before = set(hfss.oboundary.GetExcitationsOfType("Terminal"))
            boundary = hfss.wave_port(
                face_id,
                reference=list(port.reference_objects),
                name=port.name,
                renormalize=False,
                deembed=f"{port.deembed_um:g}um",
                terminals_rename=False,
            )
            after = set(hfss.oboundary.GetExcitationsOfType("Terminal"))
            names = sorted(after - before)
            if boundary is None or len(names) != 1:
                raise RuntimeError(
                    f"terminal excitation readback failed for {port.name!r}: {names!r}"
                )
            boundary_name = text_value(
                getattr(boundary, "name", port.name), "boundary.name"
            )
            native = native_terminal_readback(hfss, boundary_name, names[0], port)
            records.append(
                {
                    "index": port.index,
                    "boundary": boundary_name,
                    "terminal_excitation": names[0],
                    "face_id": face_id,
                    "face_center_um": list(centers[face_id]),
                    "native_terminal_names": names,
                    "requested": {
                        "reference_objects": list(port.reference_objects),
                        "renormalize": False,
                        "deembed_um": port.deembed_um,
                    },
                    "native": native,
                }
            )
        elif isinstance(port, ModalPort):
            if not isinstance(effective_port, ModalPort):
                raise TypeError("modal effective port must preserve ModalPort type")
            before = set(hfss.get_oo_name(hfss.odesign, "Excitations"))
            boundary = hfss.wave_port(
                face_id,
                integration_line=[
                    list(point) for point in effective_port.integration_line_um
                ],
                modes=1,
                impedance=50,
                name=port.name,
                renormalize=False,
                deembed=0,
                characteristic_impedance="Zpi",
            )
            after = set(hfss.get_oo_name(hfss.odesign, "Excitations"))
            names = sorted(after - before)
            if boundary is None or names != [port.name]:
                raise RuntimeError(
                    f"modal excitation readback failed for {port.name!r}: {names!r}"
                )
            native = native_modal_oo_readback(hfss, port.name)
            records.append(
                {
                    "index": port.index,
                    "boundary": port.name,
                    "modal_excitation": port.name,
                    "face_id": face_id,
                    "face_center_um": list(centers[face_id]),
                    "requested": {
                        "integration_line_um": [
                            list(point) for point in port.integration_line_um
                        ],
                        "modes": 1,
                        "renormalize": False,
                        "deembed_um": 0.0,
                        "characteristic_impedance": "Zpi",
                    },
                    "effective": {
                        "integration_line_um": [
                            list(point) for point in effective_port.integration_line_um
                        ]
                    },
                    "native": native,
                }
            )
        else:  # pragma: no cover - HfssDrivenSpec already closes this boundary.
            raise TypeError("unsupported HFSS driven port type")
    return records


def _face_for_side(faces: list[tuple[int, tuple[float, ...]]], side: str) -> int:
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


def bind_port_evidence(
    hfss: Any, spec: HfssGdsSpec, ports: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Bind saved modal or terminal evidence for the selected GDS-family request."""
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
        boundary = cast(dict[str, Any], boundaries.get(port.name))
        try:
            mode = boundary["Modes"]["Mode1"]
            positions = mode["IntLine"]["GeometryPosition"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"AEDT native modal integration line is unavailable: {port.name!r}"
            ) from exc
        observed = [
            [native_position_um(item[f"{axis}Position"]) for axis in "XYZ"]
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
    expected = list(cast(TerminalPort, spec.ports[0]).reference_objects)
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


def _terminal_assignment_ids(
    native_terminal: Mapping[str, Any], key: str, port_name: str
) -> list[int] | None:
    if key not in native_terminal:
        return None
    values = native_terminal[key]
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise RuntimeError(f"saved Terminal {key} assignment is invalid: {port_name!r}")
    if not values:
        return None
    return [int(value) for value in values]


def _read_body_terminal_owners(  # noqa: PLR0913 -- native owner queries retain explicit assignment inputs and the failing port identity.
    hfss: Any,
    geometry: dict[str, Any],
    native: dict[str, Any],
    edges: list[int] | None,
    faces: list[int] | None,
    port_name: str,
) -> list[str]:
    """Read the exact source conductor owner for every assigned edge or face."""
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
                in {int(edge.id) for edge in hfss.modeler[item["object_name"]].edges}
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
                f"saved Terminal child assignment is unavailable: {port_name!r}"
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
                in {int(face.id) for face in hfss.modeler[item["object_name"]].faces}
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
    return owners
