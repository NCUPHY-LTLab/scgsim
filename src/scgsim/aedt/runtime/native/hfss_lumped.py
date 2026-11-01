"""Native assignments on source-bound support Sheets.

The caller owns source/effective geometry and Entity-to-object bindings. This
module never constructs a closest-edge electrode or chooses a reference Net.
Generic RLC boundaries remain separate from Josephson-junction EPR records.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from scgsim.aedt.data import detached as detached_data
from scgsim.aedt.preparation.planar_source import native_name
from scgsim.aedt.rlc import rlc_native_topology, rlc_scalar_si
from scgsim.aedt.runtime.native.common import resolve_native_assignment
from scgsim.aedt.runtime.native.saved_geometry import read_saved_hfss_design
from scgsim.aedt.specs.common import LumpedRlc, LumpedTerminalPort
from scgsim.aedt.specs.geometry import PreparedPlanarGeometry


def assign_terminal_lumped(
    app: Any, sheet: Any, port: LumpedTerminalPort, *, reference_objects: list[str]
) -> dict[str, Any]:
    """Use the pinned Terminal AutoIdentify route on one already-bound Sheet."""
    if app.solution_type != "Terminal":
        raise RuntimeError("source-bound Lumped ports require DrivenTerminal")
    if port.deembed_um != 0:
        raise ValueError("native Lumped Terminal supports zero deembedding only")
    if len(sheet.faces) != 1:
        raise RuntimeError(f"Lumped support is not one native face: {sheet.name!r}")
    before = set(app.oboundary.GetExcitationsOfType("Terminal"))
    boundary = app.lumped_port(
        sheet.name,
        reference=reference_objects,
        create_port_sheet=False,
        auto_identify=True,
        impedance=port.impedance_ohm,
        name=port.name,
        renormalize=port.renormalize,
        deembed=bool(port.deembed_um),
        terminals_rename=False,
    )
    if boundary is False or boundary is None:
        raise RuntimeError(f"native Lumped Terminal assignment failed: {port.name!r}")
    terminals = sorted(set(app.oboundary.GetExcitationsOfType("Terminal")) - before)
    if len(terminals) != 1:
        raise RuntimeError(f"native Lumped Terminal inventory differs: {terminals!r}")
    return {
        "boundary": boundary.name,
        "terminal_excitation": terminals[0],
        "native_terminal_names": terminals,
        "object_name": sheet.name,
        "native_object_id": int(sheet.id),
        "face_id": int(sheet.faces[0].id),
        "assignment_api": "lumped_port(auto_identify=True, create_port_sheet=False)",
    }


def assign_rlc(
    app: Any,
    sheet: Any,
    treatment: LumpedRlc,
    contact_points_um: list[list[float]],
    *,
    name: str,
) -> dict[str, Any]:
    """Assign exactly the authored enabled R/L/C branches on a bound support."""
    native_topology = rlc_native_topology(treatment.topology)
    # The convenience writer rejects all-falsy values before native assignment.
    # Author the same native boundary once, with None alone disabling a branch.
    props = {
        "Objects": [sheet.name],
        "CurrentLine": {
            "Start": [f"{value}um" for value in contact_points_um[0]],
            "End": [f"{value}um" for value in contact_points_um[1]],
        },
        "RLC Type": native_topology,
    }
    for enabled, value_key, value, unit in (
        ("UseResist", "Resistance", treatment.resistance_ohm, "ohm"),
        ("UseInduct", "Inductance", treatment.inductance_h, "H"),
        ("UseCap", "Capacitance", treatment.capacitance_f, "F"),
    ):
        props[enabled] = value is not None
        if value is not None:
            props[value_key] = f"{value}{unit}"
    boundary = app._create_boundary(name, props, "Lumped RLC")  # noqa: SLF001 -- pinned native writer preserves authored zero-valued RLC branches.
    if boundary is False or boundary is None:
        raise RuntimeError(f"native generic RLC assignment failed: {name!r}")
    return rlc_record(sheet, treatment, contact_points_um, name=boundary.name)


def rlc_record(
    sheet: Any, treatment: LumpedRlc, contact_points_um: list[list[float]], *, name: str
) -> dict[str, Any]:
    """The same authored request ledger is used on construction and saved rebind."""
    return {
        "boundary": name,
        "object_name": sheet.name,
        "native_object_id": int(sheet.id),
        "native_face_ids": [int(face.id) for face in sheet.faces],
        "requested": {
            "topology": treatment.topology,
            "native_topology": rlc_native_topology(treatment.topology),
            "contact_points_um": contact_points_um,
            "resistance_ohm": treatment.resistance_ohm,
            "inductance_h": treatment.inductance_h,
            "capacitance_f": treatment.capacitance_f,
        },
    }


def read_rlc_parameters(app: Any, record: dict[str, Any]) -> dict[str, Any]:
    """Retain live native enable/value evidence; disabled branches stay disabled."""
    from ansys.aedt.core.modules.boundary.common import BoundaryObject  # noqa: PLC0415 -- optional PyAEDT loads only at the authorized native execution boundary.

    live = BoundaryObject(app, record["boundary"], auto_update=False)._child_object  # noqa: SLF001 -- pinned boundary property object supplies uncached native electrical readback.
    if live is None:
        raise RuntimeError("native generic RLC property object is unavailable")
    names = list(live.GetPropNames())
    values = {name: live.GetPropValue(name) for name in names}
    record["native_properties"] = values

    raw_ids = [
        int(value) for value in app.oboundary.GetBoundaryAssignment(record["boundary"])
    ]
    typed, covered_faces, covered_objects = resolve_native_assignment(app, raw_ids)
    record["assignment"] = {
        "raw_ids": raw_ids,
        "typed": typed,
        "covered_faces": sorted(covered_faces),
        "covered_objects": sorted(covered_objects),
    }
    if covered_objects != {record["object_name"]} or covered_faces != set(
        record["native_face_ids"]
    ):
        raise RuntimeError("native generic RLC assignment differs from its support")
    requested = record["requested"]
    record["normalized_values_si"] = {}
    if values.get("RLC Type") != requested["native_topology"]:
        raise RuntimeError("native generic RLC topology differs")
    for key, enable, prop in (
        ("resistance_ohm", "Use Resist", "Resistance"),
        ("inductance_h", "Use Induct", "Inductance"),
        ("capacitance_f", "Use Cap", "Capacitance"),
    ):
        raw_enabled = values.get(enable)
        if raw_enabled in (True, "true", "True", 1):
            enabled = True
        elif raw_enabled in (False, "false", "False", 0):
            enabled = False
        else:
            raise RuntimeError(f"native generic RLC enable unavailable: {enable}")
        if enabled != (requested[key] is not None):
            raise RuntimeError(f"native generic RLC enable differs: {enable}")
        if enabled:
            observed = rlc_scalar_si(values.get(prop), key)
            record["normalized_values_si"][key] = str(observed)
            if observed != Decimal(str(requested[key])):
                raise RuntimeError(f"native generic RLC value differs: {prop}")
    return record


def assign_generic_rlcs(app: Any, spec: Any, geometry: dict[str, Any]) -> None:
    """Electrical loads use neutral source supports, independently of JJ EPR."""

    supports = {
        item["support_id"]: item for item in geometry.get("lumped_supports", ())
    }
    if spec.lumped_rlcs:
        geometry["lumped_rlcs"] = []
    for treatment in spec.lumped_rlcs:
        support = supports[treatment.support_id]
        try:
            record = assign_rlc(
                app,
                app.modeler[support["object_name"]],
                treatment,
                name=native_name("rlc", treatment.support_id),
                contact_points_um=support["effective"]["contact_points_um"],
            )
        except Exception as exc:
            exc.add_note(
                "generic RLC assignment: "
                + json.dumps(
                    {"source": treatment.to_payload(), "support": support},
                    default=str,
                    sort_keys=True,
                )
            )
            raise
        geometry["lumped_rlcs"].append(
            {
                "support_id": treatment.support_id,
                "source": treatment.to_payload(),
                **record,
            }
        )


def bind_saved_generic_rlcs(
    app: Any, prepared: PreparedPlanarGeometry, geometry: dict[str, Any]
) -> None:
    """Bind authored RLC requests to saved support objects and verify their values."""
    supports = {
        item["support_id"]: item for item in geometry.get("lumped_supports", ())
    }
    if prepared.lumped_rlcs:
        geometry["lumped_rlcs"] = []
    for treatment in prepared.lumped_rlcs:
        support = supports[treatment.support_id]
        record = rlc_record(
            app.modeler[support["object_name"]],
            treatment,
            name=native_name("rlc", treatment.support_id),
            contact_points_um=support["effective"]["contact_points_um"],
        )
        geometry["lumped_rlcs"].append(
            {
                "support_id": treatment.support_id,
                "source": treatment.to_payload(),
                **record,
            }
        )
    read_generic_rlcs(app, geometry)


def read_generic_rlcs(app: Any, geometry: dict[str, Any]) -> None:
    """Verify live values and saved directed contact lines for generic RLC loads."""
    records = geometry.get("lumped_rlcs", ())
    if not records:
        return

    digest, design = read_saved_hfss_design(
        Path(app.project_file), app.design_name, purpose="generic RLC"
    )
    boundaries = design["BoundarySetup"]["Boundaries"]
    for record in records:
        try:
            read_rlc_parameters(app, record)
            boundary = detached_data(boundaries[record["boundary"]])
            record["saved_boundary"] = boundary
            record["saved_project_sha256"] = digest
            if boundary["BoundType"] != "Lumped RLC":
                raise RuntimeError("saved generic RLC boundary type differs")
            record["saved_model_units"] = design["ModelSetup"]["GeometryCore"]["Units"]
            if record["saved_model_units"] != "um":
                raise RuntimeError("saved generic RLC model units differ")
            record["attached_contact_edges"] = []
            positions = boundary["CurrentLine"]["GeometryPosition"]
            observed = []
            for position in positions:
                if (
                    position.get("IsAttachedToEntity") is False
                    and position.get("PositionType") == "AbsolutePosition"
                ):
                    point = [float(position[axis + "Position"]) for axis in "XYZ"]
                elif (
                    position.get("IsAttachedToEntity") is True
                    and position.get("PositionType") == "EdgeCenter"
                ):
                    edge_id = position["EntityID"]
                    edges = [
                        edge
                        for edge in app.modeler[record["object_name"]].edges
                        if int(edge.id) == edge_id
                    ]
                    if len(edges) != 1:
                        raise RuntimeError(
                            "saved generic RLC contact edge does not belong to its support"
                        )
                    point = list(edges[0].midpoint)
                    record["attached_contact_edges"].append(
                        {
                            "edge_id": edge_id,
                            "object_name": record["object_name"],
                            "native_object_id": record["native_object_id"],
                            "midpoint_um": point,
                        }
                    )
                else:
                    raise RuntimeError(
                        "saved generic RLC line position kind is unsupported"
                    )
                observed.append(point)
            record["saved_contact_points_um"] = observed
            if observed != record["requested"]["contact_points_um"]:
                raise RuntimeError("saved generic RLC directed contact line differs")
        except Exception as exc:
            exc.add_note(
                "generic RLC readback: "
                + json.dumps(record, default=str, sort_keys=True)
            )
            raise
