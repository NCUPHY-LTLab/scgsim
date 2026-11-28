"""Strict offline check of a Driven Terminal receipt's lumped ports against its request and SGB model."""

from __future__ import annotations

from typing import Any

from scgsim.aedt.preparation.sgb_model import port_group
from scgsim.aedt.results.readback_common import validate_imported_sgb_model
from scgsim.aedt.specs.hfss import HfssDrivenGeometrySpec


def validate_body_terminal_readback(
    receipt: dict[str, Any], spec: HfssDrivenGeometrySpec
) -> None:
    """The receipt's geometry is the request's model imported, and its ports are the request's, as assigned.

    Each port is a Terminal ``Lumped Port`` named as requested, on its port sheet's imported faces, with one
    Terminal on the edges the import recorded; the saved design read both back the same way.
    """
    validate_imported_sgb_model(receipt, spec.model)
    assigned = receipt["geometry"].get("lumped_ports")
    ports = receipt.get("ports")
    if not isinstance(ports, list) or not isinstance(assigned, list):
        raise RuntimeError("the Driven receipt has no lumped-port records")
    if len(ports) != len(spec.ports) or len(assigned) != len(spec.ports):
        raise RuntimeError("the Driven receipt's ports are not the request's two ports")
    for record, imported, port in zip(ports, assigned, spec.ports, strict=True):
        native = record.get("native", {})
        saved = native.get("saved_boundary", {})
        saved_terminal = native.get("saved_terminal", {})
        faces = sorted(item["face_id"] for item in imported.get("faces", ()))
        if (
            {key: value for key, value in record.items() if key != "native"} != imported
            or record.get("index") != port.index
            or record.get("boundary") != port.name
            or record.get("requested") != port.to_payload()
            or record.get("port_group") != port_group(port.port_sheet_id)
            or not isinstance(record.get("terminal_excitation"), str)
            or saved.get("BoundType") != "Lumped Port"
            or saved.get("LumpedPortType") != "Terminal"
            or sorted(int(value) for value in saved.get("Faces", ())) != faces
            or saved_terminal.get("BoundType") != "Terminal"
            or sorted(int(value) for value in saved_terminal.get("Edges", ()))
            != imported.get("terminal_edges")
        ):
            raise RuntimeError(
                f"the Driven receipt's port {port.name!r} is not the requested lumped port as assigned"
            )
