"""Read the saved Driven lumped Terminal ports back from the project the import assigned them in.

:func:`~scgsim.aedt.runtime.families.sgb_ports.assign_lumped_port` records each port as it is assigned.
After the project is saved, :func:`read_saved_lumped_ports` checks every record against the saved design:
the port is a Terminal ``Lumped Port`` on exactly the faces it was given, and its one Terminal is the saved
child of that port on exactly the edges it was given.  The reference conductors AEDT records, when it records
any, must be the conductor solids of the ports' reference hosts (each sheet's host that is not its signal
host); a Route A host is a Perfect E sheet, not a solid, so there it records none.  The saved records, the
port's and Terminal's native properties and the reference conductors AEDT chose are kept in the receipt.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from scgsim.aedt.runtime.native.common import detached_data
from scgsim.contracts.sgb_sidecar import Sidecar


def read_saved_lumped_ports(
    hfss: Any, ports: list[dict[str, Any]], sidecar: Sidecar
) -> list[dict[str, Any]]:
    """Each assigned lumped-port record with its saved port and Terminal; a difference raises."""
    setup = hfss.design_properties["BoundarySetup"]
    boundaries = setup["Boundaries"]
    references = [
        hfss.oeditor.GetObjectNameByID(int(value))
        for value in setup.get("ProductSpecificData", {}).get(
            "TerminalReferenceConductors", ()
        )
    ]
    declared = declared_reference_objects(sidecar, ports)
    # AEDT 2024.2 recorded none in the development runs of both routes, so an empty record is kept as read.
    if references and sorted(references) != declared:
        raise RuntimeError(
            f"the saved design's Terminal reference conductors are {sorted(references)}; the ports' "
            f"reference hosts are the conductors {declared}"
        )
    out = []
    for record in ports:
        name, terminal = record["boundary"], record["terminal_excitation"]
        saved, saved_terminal = boundaries.get(name), boundaries.get(terminal)
        if not isinstance(saved, Mapping) or not isinstance(saved_terminal, Mapping):
            raise RuntimeError(
                f"the saved design has no lumped port {name!r} with Terminal {terminal!r}"
            )
        faces = sorted(item["face_id"] for item in record["faces"])
        if (
            saved.get("BoundType") != "Lumped Port"
            or saved.get("LumpedPortType") != "Terminal"
            or sorted(int(value) for value in saved.get("Faces", ())) != faces
            or list(saved.get("TerminalIDList") or ()) != [saved_terminal.get("ID")]
            or saved_terminal.get("BoundType") != "Terminal"
            or saved_terminal.get("ParentBndID") != saved.get("ID")
            or sorted(int(value) for value in saved_terminal.get("Edges", ()))
            != record["terminal_edges"]
        ):
            raise RuntimeError(
                f"the saved lumped port {name!r} is not a Terminal Lumped Port on faces {faces} with "
                f"Terminal {terminal!r} on edges {record['terminal_edges']}: "
                f"{dict(saved)!r}, {dict(saved_terminal)!r}"
            )
        out.append(
            {
                **record,
                "native": {
                    "saved_boundary": detached_data(dict(saved)),
                    "saved_terminal": detached_data(dict(saved_terminal)),
                    "boundary_properties": _oo_properties(hfss, f"Excitations\\{name}"),
                    "terminal_properties": _oo_properties(
                        hfss, f"Excitations\\{name}\\{terminal}"
                    ),
                    "reference_conductors": references,
                },
            }
        )
    return out


def declared_reference_objects(
    sidecar: Sidecar, ports: list[dict[str, Any]]
) -> list[str]:
    """The AEDT objects of the ports' reference hosts: each conductor solid of a sheet host that is not its signal host."""
    hosts = {
        host
        for record in ports
        for host in record["host_entity_ids"]
        if host != record["signal_host"]
    }
    return sorted(
        str(volume["aedt_name"])
        for volume in sidecar["volume"].values()
        if volume["kind"] == "conductor"
        and volume["emitted"]
        and volume["entity"] in hosts
    )


def _oo_properties(hfss: Any, path: str) -> dict[str, Any]:
    """The properties of one AEDT object-oriented child, as AEDT spells them."""
    child = hfss.get_oo_object(hfss.odesign, path)
    if not child:
        raise RuntimeError(f"AEDT native child object is unavailable: {path!r}")
    names = child.GetPropNames()
    if not names:
        raise RuntimeError(f"AEDT native child has no properties: {path!r}")
    return detached_data({name: child.GetPropValue(name) for name in names})
