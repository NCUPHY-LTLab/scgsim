"""Native assignments on source-bound support Sheets.

The caller owns source/effective geometry and Entity-to-object bindings. This
module never constructs a closest-edge electrode or chooses a reference Net.
Generic RLC boundaries remain separate from Josephson-junction EPR records.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from scgsim.aedt._rlc import rlc_native_topology, rlc_scalar_si
from pathlib import Path
import json
from scgsim.aedt._data import detached as detached_data
from scgsim.aedt.specs.geometry import PreparedPlanarGeometry


def assign_terminal_lumped(
    app: Any,
    sheet: Any,
    *,
    name: str,
    reference_objects: list[str],
    impedance_ohm: float,
    renormalize: bool,
    deembed_um: float,
) -> dict[str, Any]:
    """Use the pinned Terminal AutoIdentify route on one already-bound Sheet."""
    if app.solution_type != "Terminal":
        raise RuntimeError("source-bound Lumped ports require DrivenTerminal")
    if deembed_um != 0:
        raise ValueError("native Lumped Terminal supports zero deembedding only")
    if len(sheet.faces) != 1:
        raise RuntimeError(f"Lumped support is not one native face: {sheet.name!r}")
    before = set(app.oboundary.GetExcitationsOfType("Terminal"))
    boundary = app.lumped_port(
        sheet.name,
        reference=reference_objects,
        create_port_sheet=False,
        auto_identify=True,
        impedance=impedance_ohm,
        name=name,
        renormalize=renormalize,
        deembed=bool(deembed_um),
        terminals_rename=False,
    )
    if boundary is False or boundary is None:
        raise RuntimeError(f"native Lumped Terminal assignment failed: {name!r}")
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
    *,
    name: str,
    contact_points_um: list[list[float]],
    topology: str,
    resistance_ohm: float | None,
    inductance_h: float | None,
    capacitance_f: float | None,
) -> dict[str, Any]:
    """Assign exactly the authored enabled R/L/C branches on a bound support."""
    native_topology = rlc_native_topology(topology)
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
        ("UseResist", "Resistance", resistance_ohm, "ohm"),
        ("UseInduct", "Inductance", inductance_h, "H"),
        ("UseCap", "Capacitance", capacitance_f, "F"),
    ):
        props[enabled] = value is not None
        if value is not None:
            props[value_key] = f"{value}{unit}"
    boundary = app._create_boundary(name, props, "Lumped RLC")
    if boundary is False or boundary is None:
        raise RuntimeError(f"native generic RLC assignment failed: {name!r}")
    return rlc_record(
        sheet,
        name=boundary.name,
        contact_points_um=contact_points_um,
        topology=topology,
        resistance_ohm=resistance_ohm,
        inductance_h=inductance_h,
        capacitance_f=capacitance_f,
    )


def rlc_record(
    sheet: Any,
    *,
    name: str,
    contact_points_um: list[list[float]],
    topology: str,
    resistance_ohm: float | None,
    inductance_h: float | None,
    capacitance_f: float | None,
) -> dict[str, Any]:
    """The same authored request ledger is used on construction and saved rebind."""
    return {
        "boundary": name,
        "object_name": sheet.name,
        "native_object_id": int(sheet.id),
        "native_face_ids": [int(face.id) for face in sheet.faces],
        "requested": {
            "topology": topology,
            "native_topology": rlc_native_topology(topology),
            "contact_points_um": contact_points_um,
            "resistance_ohm": resistance_ohm,
            "inductance_h": inductance_h,
            "capacitance_f": capacitance_f,
        },
    }


def read_rlc_parameters(app: Any, record: dict[str, Any]) -> dict[str, Any]:
    """Retain live native enable/value evidence; disabled branches stay disabled."""
    from ansys.aedt.core.modules.boundary.common import BoundaryObject

    live = BoundaryObject(app, record["boundary"], auto_update=False)._child_object
    if live is None:
        raise RuntimeError("native generic RLC property object is unavailable")
    names = list(live.GetPropNames())
    values = {name: live.GetPropValue(name) for name in names}
    record["native_properties"] = values
    from scgsim.aedt.runtime.native.common import _resolve_native_assignment

    raw_ids = [
        int(value) for value in app.oboundary.GetBoundaryAssignment(record["boundary"])
    ]
    typed, covered_faces, covered_objects = _resolve_native_assignment(app, raw_ids)
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


def _assign_generic_rlcs(app: Any, spec: Any, geometry: dict[str, Any]) -> None:
    """Electrical loads use neutral source supports, independently of JJ EPR."""
    from scgsim.aedt.preparation.planar import _native_name

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
                name=_native_name("rlc", treatment.support_id),
                contact_points_um=support["effective"]["contact_points_um"],
                topology=treatment.topology,
                resistance_ohm=treatment.resistance_ohm,
                inductance_h=treatment.inductance_h,
                capacitance_f=treatment.capacitance_f,
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


def _bind_saved_generic_rlcs(
    app: Any, prepared: PreparedPlanarGeometry, geometry: dict[str, Any]
) -> None:
    from scgsim.aedt.preparation.planar import _native_name

    supports = {
        item["support_id"]: item for item in geometry.get("lumped_supports", ())
    }
    if prepared.lumped_rlcs:
        geometry["lumped_rlcs"] = []
    for treatment in prepared.lumped_rlcs:
        support = supports[treatment.support_id]
        record = rlc_record(
            app.modeler[support["object_name"]],
            name=_native_name("rlc", treatment.support_id),
            contact_points_um=support["effective"]["contact_points_um"],
            topology=treatment.topology,
            resistance_ohm=treatment.resistance_ohm,
            inductance_h=treatment.inductance_h,
            capacitance_f=treatment.capacitance_f,
        )
        geometry["lumped_rlcs"].append(
            {
                "support_id": treatment.support_id,
                "source": treatment.to_payload(),
                **record,
            }
        )
    _read_generic_rlcs(app, geometry)


def _read_generic_rlcs(app: Any, geometry: dict[str, Any]) -> None:

    records = geometry.get("lumped_rlcs", ())
    if not records:
        return
    from scgsim.aedt.runtime.native.hfss_geometry import _read_saved_hfss_design

    digest, design = _read_saved_hfss_design(
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
