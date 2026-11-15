"""Assign the Q3D Nets of an imported SGB model, with their sources and sinks, and read them back.

Each Net of the request is assigned over its conductor objects with its Q3D type.  A source or sink sits on
the authored global side of its object, read from the imported object itself (issue #67), never from source
polygons:

* on a solid, the planar faces whose every vertex lies at the object's extreme along the side;
* on a sheet (a Route A conductor), the straight edges whose two vertices lie there, which must form one
  unbroken segment, an in-plane side only.

The faces of a solid's side must be one connected patch.  A side that selects nothing, or pieces that do not
join, is refused rather than narrowed to one of them.  The comparison uses the import's readback tolerance.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from scgsim.aedt.specs.q3d import Q3dNetSpec, Q3dSpec

Point = tuple[float, float, float]
TERMINAL_TYPE = "ConstantVoltage"


class Q3dTerminalError(RuntimeError):
    """A Net or a terminal did not take, or its side selects no single patch of the object."""


def assign_q3d_nets(
    app: Any, spec: Q3dSpec, tolerance_um: float
) -> list[dict[str, Any]]:
    """Assign the request's Nets in their order, each with its source and sink, and read each back."""
    records: list[dict[str, Any]] = []
    for net in spec.nets:
        boundary = app.assign_net(list(net.object_names), net.name, net.net_type)
        if not boundary or net.name not in app.net_names:
            raise Q3dTerminalError(f"Q3D net assignment failed for {net.name!r}")
        expected = sorted(int(app.modeler[name].id) for name in net.object_names)
        observed = sorted(
            int(value) for value in app.oboundary.GetExcitationAssignment(net.name)
        )
        if observed != expected:
            raise Q3dTerminalError(
                f"Q3D native net assignment differs for {net.name!r}"
            )
        record: dict[str, Any] = {
            "name": net.name,
            "net_type": net.net_type,
            "object_names": list(net.object_names),
            "native_object_ids": observed,
        }
        if net.has_terminals:
            record["source"] = _assign_terminal(app, net, "source", tolerance_um)
            record["sink"] = _assign_terminal(app, net, "sink", tolerance_um)
        records.append(record)
    if list(app.net_names) != [net.name for net in spec.nets]:
        raise Q3dTerminalError("Q3D native net order does not match the request")
    return records


def _assign_terminal(
    app: Any, net: Q3dNetSpec, kind: str, tolerance_um: float
) -> dict[str, Any]:
    object_name = net.source_object if kind == "source" else net.sink_object
    side = net.source_side if kind == "source" else net.sink_side
    assert object_name is not None and side is not None  # noqa: S101 -- has_terminals guarantees both
    name = f"{net.name}{kind.title()}"
    obj = app.modeler[object_name]
    sheet = str(obj.object_type) == "Sheet"
    if sheet:
        key = "Edges"
        ids = sheet_side_edges(_edges(obj), side, tolerance_um)
    else:
        key = "Faces"
        ids = solid_side_faces(_faces(obj), side, tolerance_um)
    method = (
        app.oboundary.AssignSource if kind == "source" else app.oboundary.AssignSink
    )
    method(
        [
            f"NAME:{name}",
            f"{key}:=",
            ids,
            "Net:=",
            net.name,
            "TerminalType:=",
            TERMINAL_TYPE,
        ]
    )
    assigned = sorted(
        int(value) for value in app.oboundary.GetExcitationAssignment(name)
    )
    if assigned != sorted(ids):
        raise Q3dTerminalError(
            f"Q3D {kind} {name!r} is on {assigned}, the side {side} of {object_name!r} is {sorted(ids)}"
        )
    return {
        "name": name,
        "object_name": object_name,
        "side": side,
        "native_face_ids" if key == "Faces" else "native_edge_ids": sorted(ids),
        "terminal_type": TERMINAL_TYPE,
    }


def verify_saved_q3d_terminals(app: Any, nets: list[dict[str, Any]]) -> None:
    """Read every terminal back from the saved design: its faces or edges, Net and terminal type."""
    boundaries = app.design_properties["BoundarySetup"]["Boundaries"]
    for net in nets:
        for kind in ("source", "sink"):
            terminal = net.get(kind)
            if terminal is None:
                continue
            key = "Faces" if "native_face_ids" in terminal else "Edges"
            saved = boundaries[terminal["name"]]
            ids = terminal["native_face_ids" if key == "Faces" else "native_edge_ids"]
            if (
                sorted(int(value) for value in saved.get(key, ())) != ids
                or saved.get("Net") != net["name"]
                or saved.get("TerminalType") != terminal["terminal_type"]
            ):
                raise Q3dTerminalError(
                    f"the saved Q3D {kind} {terminal['name']!r} differs from its assignment"
                )


# ---------------------------------------------------------------- side selection (no AEDT)


def side_axis(side: str) -> tuple[int, float]:
    """The axis index and sign (+1 or -1) of a global side such as ``+X``."""
    return "XYZ".index(side[1]), 1.0 if side[0] == "+" else -1.0


def solid_side_faces(
    faces: Sequence[tuple[int, bool, Sequence[Point], Sequence[int]]],
    side: str,
    tolerance_um: float,
) -> list[int]:
    """The ids of the planar faces whose every vertex lies at the extreme along ``side``.

    Each face is ``(id, planar, vertices, edge ids)``.  The selected faces must join, edge to edge, into one
    patch; none, or two patches, is refused.
    """
    axis, sign = side_axis(side)
    extreme = max(sign * p[axis] for _, _, vertices, _ in faces for p in vertices)
    selected = [
        (face_id, set(edges))
        for face_id, planar, vertices, edges in faces
        if planar
        and vertices
        and all(abs(sign * p[axis] - extreme) <= tolerance_um for p in vertices)
    ]
    if not selected:
        raise Q3dTerminalError(f"no planar face of the object lies on its side {side}")
    patch = {selected[0][0]}
    reached = set(selected[0][1])
    grew = True
    while grew:
        grew = False
        for face_id, edges in selected:
            if face_id not in patch and edges & reached:
                patch.add(face_id)
                reached |= edges
                grew = True
    if len(patch) != len(selected):
        raise Q3dTerminalError(
            f"the faces on side {side} are {len(selected)} faces that do not join into one patch"
        )
    return sorted(face_id for face_id, _ in selected)


def sheet_side_edges(
    edges: Sequence[tuple[int, float, Sequence[Point]]],
    side: str,
    tolerance_um: float,
) -> list[int]:
    """The ids of the straight edges whose two vertices lie at the sheet's extreme along ``side``.

    Each edge is ``(id, length, vertices)``.  The selected edges must lie end to end as one segment (their
    lengths add up to the distance between its farthest ends); a sheet's side is in its plane.
    """
    axis, sign = side_axis(side)
    if axis == 2:
        raise Q3dTerminalError(
            f"a sheet terminal side must be in its plane, not {side}"
        )
    extreme = max(sign * p[axis] for _, _, vertices in edges for p in vertices)
    selected = [
        (edge_id, length, vertices)
        for edge_id, length, vertices in edges
        if len(vertices) == 2
        and abs(length - math.dist(*vertices)) <= tolerance_um
        and all(abs(sign * p[axis] - extreme) <= tolerance_um for p in vertices)
    ]
    if not selected:
        raise Q3dTerminalError(f"no straight edge of the sheet lies on its side {side}")
    points = [p for _, _, vertices in selected for p in vertices]
    span = max(math.dist(p, q) for p in points for q in points)
    covered = sum(length for _, length, _ in selected)
    if abs(covered - span) > len(selected) * tolerance_um:
        raise Q3dTerminalError(
            f"the sheet's edges on side {side} do not form one unbroken segment"
        )
    return sorted(edge_id for edge_id, _, _ in selected)


def _faces(obj: Any) -> list[tuple[int, bool, list[Point], list[int]]]:
    return [
        (
            int(face.id),
            bool(face.is_planar),
            [_point(vertex.position) for vertex in face.vertices],
            [int(edge.id) for edge in face.edges],
        )
        for face in obj.faces
    ]


def _edges(obj: Any) -> list[tuple[int, float, list[Point]]]:
    return [
        (
            int(edge.id),
            float(edge.length),
            [_point(vertex.position) for vertex in edge.vertices],
        )
        for edge in obj.edges
    ]


def _point(values: Sequence[Any]) -> Point:
    x, y, z = (float(value) for value in values)
    return (x, y, z)
