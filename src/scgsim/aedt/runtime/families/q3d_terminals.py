"""Own Q3D net and terminal assignment with saved readback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from scgsim.aedt.specs.q3d import Q3dSpec
from scgsim.aedt.terminal_chain import q3d_terminal_segments


def _assign_sheet_terminal(
    app: Any, spec: Q3dSpec, terminal: _TerminalAssignment
) -> dict[str, Any]:
    """Bind authored exterior segments to actual Sheet edges and native excitation."""
    object_name, side = terminal.object_name, terminal.side
    name, net_name = terminal.name, terminal.net_name
    body = next(body for body in spec.bodies if body.body_id == object_name)
    effective = next(
        record for record in spec.effective_bodies if record["body_id"] == object_name
    )
    segments = q3d_terminal_segments(body, side)
    ring = body.exterior_um
    if ring[0] == ring[-1]:
        ring = ring[:-1]
    obj = app.modeler.get_object_from_name(object_name)
    edge_ids = []
    source_bindings = []
    for index in segments:
        endpoints = {
            (*ring[index], effective["effective_z_min_um"]),
            (*ring[(index + 1) % len(ring)], effective["effective_z_min_um"]),
        }
        matches = [
            edge
            for edge in obj.edges
            if {
                tuple(float(value) for value in vertex.position)
                for vertex in edge.vertices
            }
            == endpoints
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Q3D Sheet terminal source segment {index} has no unique native edge"
            )
        edge_id = int(matches[0].id)
        edge_ids.append(edge_id)
        source_bindings.append(
            {"source_segment_index": index, "native_edge_id": edge_id}
        )
    assignment = [
        f"NAME:{name}",
        "Edges:=",
        edge_ids,
        "Net:=",
        net_name,
        "TerminalType:=",
        "ConstantVoltage",
    ]
    method = (
        app.oboundary.AssignSource
        if name == f"{net_name}Source"
        else app.oboundary.AssignSink
    )
    stage = "assign_source_or_sink"
    native_assignment = None
    try:
        method(assignment)
        stage = "native_assignment_readback"
        native_assignment = app.oboundary.GetExcitationAssignment(name)
        assigned = [int(value) for value in native_assignment]
        if assigned != edge_ids:
            raise RuntimeError(
                f"Q3D Sheet terminal native assignment differs: {name!r}"
            )
    except Exception as exc:  # noqa: BLE001 -- preserve the native primary error.
        exc.add_note(
            f"Q3D Sheet terminal stage={stage}; name={name!r}; net={net_name!r}; "
            f"object={object_name!r}; side={side!r}; source_edges={source_bindings!r}; "
            f"selected_native_edge_ids={edge_ids!r}; native_assignment={native_assignment!r}"
        )
        raise
    return {
        "name": name,
        "object_name": object_name,
        "side": side,
        "native_edge_ids": assigned,
        "source_edges": source_bindings,
        "terminal_type": "ConstantVoltage",
    }


def verify_saved_q3d_sheet_terminals(app: Any, nets: list[dict[str, Any]]) -> None:
    """Read saved Sheet terminal properties only after the preparation Save succeeds."""
    for net in nets:
        for kind in ("source", "sink"):
            terminal = net.get(kind)
            if terminal is None or "native_edge_ids" not in terminal:
                continue
            try:
                saved = app.design_properties["BoundarySetup"]["Boundaries"][
                    terminal["name"]
                ]
                if (
                    saved.get("Edges") != terminal["native_edge_ids"]
                    or saved.get("Net") != net["name"]
                    or saved.get("TerminalType") != terminal["terminal_type"]
                ):
                    raise RuntimeError(
                        f"Q3D saved Sheet terminal assignment differs: {terminal['name']!r}"
                    )
            except Exception as exc:  # noqa: BLE001 -- preserve failed saved readback.
                exc.add_note(
                    f"Q3D Sheet terminal stage=after_preparation_save_readback; "
                    f"net={net['name']!r}; terminal={terminal!r}"
                )
                raise


def assign_q3d_nets(app: Any, spec: Q3dSpec) -> list[dict[str, Any]]:
    """Assign declared nets and their source/sink terminals in source order."""
    directions = {"-X": 0, "-Y": 1, "-Z": 2, "+X": 3, "+Y": 4, "+Z": 5}
    records: list[dict[str, Any]] = []
    for net in spec.nets:
        boundary = app.assign_net(list(net.object_names), net.name, net.net_type)
        if boundary is None or net.name not in app.net_names:
            raise RuntimeError(f"Q3D net assignment failed for {net.name!r}")
        expected_ids = [
            int(app.modeler.get_object_from_name(name).id) for name in net.object_names
        ]
        observed_ids = [
            int(value) for value in app.oboundary.GetExcitationAssignment(net.name)
        ]
        if observed_ids != expected_ids:
            raise RuntimeError(f"Q3D native net assignment mismatch for {net.name!r}")
        record: dict[str, Any] = {
            "name": net.name,
            "net_type": net.net_type,
            "object_names": list(net.object_names),
            "native_object_ids": observed_ids,
        }
        if net.net_type == "Signal" and net.source_object is not None:
            source_name = f"{net.name}Source"
            sink_name = f"{net.name}Sink"
            source_direction = directions[cast(str, net.source_side)]
            sink_direction = directions[cast(str, net.sink_side)]
            effective = {item["body_id"]: item for item in spec.effective_bodies}
            for kind, object_name, side, direction, terminal_name in (
                (
                    "source",
                    net.source_object,
                    net.source_side,
                    source_direction,
                    source_name,
                ),
                ("sink", net.sink_object, net.sink_side, sink_direction, sink_name),
            ):
                if effective[object_name]["representation"] == "sheet":
                    record[kind] = _assign_sheet_terminal(
                        app,
                        spec,
                        _TerminalAssignment(
                            cast(str, object_name),
                            cast(str, side),
                            terminal_name,
                            net.name,
                        ),
                    )
                    continue
                assign = app.source if kind == "source" else app.sink
                boundary = assign(
                    object_name,
                    direction=direction,
                    name=terminal_name,
                    net_name=net.name,
                )
                if boundary is None:
                    raise RuntimeError(f"Q3D {kind} assignment failed for {net.name!r}")
                faces = [
                    int(value)
                    for value in app.oboundary.GetExcitationAssignment(terminal_name)
                ]
                expected = int(app.modeler._get_faceid_on_axis(object_name, direction))  # noqa: SLF001 -- pinned axis lookup independently verifies the assigned native terminal face.
                if faces != [expected]:
                    raise RuntimeError(
                        f"Q3D native {kind} assignment mismatch for {net.name!r}"
                    )
                record[kind] = {
                    "name": terminal_name,
                    "object_name": object_name,
                    "side": side,
                    "native_face_ids": faces,
                }
        records.append(record)
    if app.net_names != [net.name for net in spec.nets]:
        raise RuntimeError("Q3D native net order does not match the structured spec")
    return records


@dataclass(frozen=True)
class _TerminalAssignment:
    """The declared object side and native terminal/net identity."""

    object_name: str
    side: str
    name: str
    net_name: str
