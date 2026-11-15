"""Read the saved Driven lumped Terminal ports back from the project the import assigned them in.

:func:`~scgsim.aedt.runtime.families.sgb_ports.assign_lumped_port` records each port as it is assigned.
After the project is saved, :func:`read_saved_lumped_ports` checks every record against the saved design:
the port is a Terminal ``Lumped Port`` on exactly the faces it was given, and its one Terminal is the saved
child of that port on exactly the edges it was given.  The saved records, the port's and Terminal's native
properties and the reference conductors AEDT chose are kept in the receipt.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from scgsim.aedt.runtime.native.common import detached_data


def read_saved_lumped_ports(
    hfss: Any, ports: list[dict[str, Any]]
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


def _oo_properties(hfss: Any, path: str) -> dict[str, Any]:
    """The properties of one AEDT object-oriented child, as AEDT spells them."""
    child = hfss.get_oo_object(hfss.odesign, path)
    if not child:
        raise RuntimeError(f"AEDT native child object is unavailable: {path!r}")
    names = child.GetPropNames()
    if not names:
        raise RuntimeError(f"AEDT native child has no properties: {path!r}")
    return detached_data({name: child.GetPropValue(name) for name in names})
