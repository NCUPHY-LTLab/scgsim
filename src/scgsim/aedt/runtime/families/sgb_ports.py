"""HFSS assignments on the faces of an imported port sheet, read from those faces' own edges.

A port sheet's faces are where the import found them (``face_copies``).  :func:`assign_port_rlc` puts a
junction's or a generic ``LumpedRlc``'s Lumped RLC on them, with the current line across them, read from the
faces' own edges along the sidecar port's ``direction`` (:func:`current_line_um`), never from the sidecar's
coordinates: a face two solids share imports twice, about 4e-5 um apart on arcs (E4).

:func:`assign_lumped_port` puts a Driven lumped Terminal port on the port sheet, with its Terminal on the
edges where the sheet meets the signal conductor (:func:`port_end_edges`).  A Terminal port needs the
conductors to touch it, and AEDT 2024.2 imports a substrate face about 2e-8 um away from the conductor
solids beside it, too far for AEDT to find them (development check, 2026-11-15).  So a port face on an
imported solid is drawn again as a sheet object from the guard's exact vertices, as a free face is from its
guarded loops, and checked against the guard's area.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from scgsim.aedt.runtime.families.sgb_boundaries import PortRlc, PortTerminal
from scgsim.aedt.runtime.families.sgb_faces import SgbImportError, area_tolerance_um2
from scgsim.aedt.runtime.native.common import native_boundary_type
from scgsim.geometry.sgb.output.sidecar.aedt import sanitise_aedt_name

Point = tuple[float, float, float]


def assign_port_rlc(
    app: Any,
    rlc: PortRlc,
    face_copies: Mapping[str, Mapping[str, int]],
    tolerance_um: float,
) -> dict[str, Any]:
    """One Lumped RLC over every face of its port sheet, its current line read from those faces' own edges.

    SGB partitions a port sheet where another face meets it (a pad wall, a band), so the RLC takes all of
    its faces and the current line crosses their union along the port's ``direction``.  A face two solids
    share imports twice, about 4e-5 um apart on arcs (E4), so the line ends come from the imported faces
    themselves, never from the sidecar's coordinates.
    """
    faces, (start, end) = port_faces_and_line(
        app, rlc.port, rlc.direction, face_copies, tolerance_um
    )
    props = {
        "Faces": [face_id for _, face_id in faces],
        "CurrentLine": {
            "Start": [f"{value}um" for value in start],
            "End": [f"{value}um" for value in end],
        },
        **rlc.branches,
    }
    # PyAEDT's Lumped RLC helpers take sheet names only and refuse all-falsy values; the native writer takes
    # the face ids (E4) and keeps an authored zero branch.
    boundary = app._create_boundary(rlc.name, props, "Lumped RLC")  # noqa: SLF001 -- the pinned native writer is the only face-id route.
    if not boundary or native_boundary_type(app, rlc.name) != "Lumped RLC":
        raise SgbImportError(f"AEDT did not assign the Lumped RLC {rlc.name!r}")
    return {
        **rlc.evidence,
        "boundary": rlc.name,
        "port_group": rlc.port["group"],
        "faces": [
            {"object_name": object_name, "face_id": face_id}
            for object_name, face_id in faces
        ],
        "host_entity_ids": list(rlc.port["host_entity_ids"]),
        "current_line_um": [list(start), list(end)],
        "branches": dict(rlc.branches),
    }


def port_faces_and_line(
    app: Any,
    port: Mapping[str, Any],
    direction: Point,
    face_copies: Mapping[str, Mapping[str, int]],
    tolerance_um: float,
) -> tuple[list[tuple[str, int]], tuple[Point, Point]]:
    """The imported faces of a sidecar port, as (object, face id), and the line across them along ``direction``."""
    faces, edges = _port_faces(app, port, face_copies)
    return faces, current_line_um(
        [points for _, points in edges], direction, tolerance_um
    )


def _port_faces(
    app: Any, port: Mapping[str, Any], face_copies: Mapping[str, Mapping[str, int]]
) -> tuple[list[tuple[str, int]], list[tuple[int, tuple[Point, ...]]]]:
    """The imported faces of a sidecar port, as (object, face id), and their edges, as (edge id, vertices)."""
    faces: list[tuple[str, int]] = []
    edges: list[tuple[int, tuple[Point, ...]]] = []
    for planned in port["faces"]:
        ((object_name, face_id),) = face_copies[planned].items()
        face = next(f for f in app.modeler[object_name].faces if int(f.id) == face_id)
        faces.append((object_name, face_id))
        edges += [
            (int(edge.id), tuple(_point(vertex.position) for vertex in edge.vertices))
            for edge in face.edges
        ]
    return faces, edges


def assign_lumped_port(
    app: Any,
    terminal: PortTerminal,
    port_faces: Mapping[str, Any],
    tolerance_um: float,
) -> dict[str, Any]:
    """One Driven lumped Terminal port over every face of its port sheet, and its one Terminal.

    The port takes all the faces of its sheet, as a Lumped RLC does.  Its Terminal is the edges of those
    faces at the signal conductor's end of the sheet along the sidecar ``direction``, found as the RLC's
    current-line ends are (:func:`port_end_edges`).  The sidecar lists the sheet's hosts in the order that
    direction meets them, start host then end host, so the signal host's place in that list is its end.  The
    reference is the conductor at the other end; AEDT records it in ``TerminalReferenceConductors``.
    ``port_faces`` maps each planned face of the sheet to its imported place: the name of the sheet object a
    free face was rebuilt as, or the guard record of a face on an imported solid, which is drawn again
    (:func:`port_sheet`).

    AEDT 2024.2 makes a lumped port with an integration line a Modal lumped port with no Terminal, even in a
    Driven Terminal design (development check, 2026-11-15), so the Terminal is assigned on the edges.
    """
    port = terminal.port
    hosts = list(terminal.sheet["host_entity_ids"])
    ends = [host in port.signal_entity_ids for host in hosts]
    if len(hosts) != 2 or sum(ends) != 1:
        raise SgbImportError(
            f"port {port.name!r}: its sheet joins {hosts}, not one signal Entity of "
            f"{list(port.signal_entity_ids)} and one other conductor"
        )
    faces, edges = port_sheet(app, terminal, port_faces, tolerance_um)
    signal_edges = port_end_edges(
        edges, terminal.direction, tolerance_um, upper=ends[1]
    )
    face_ids = [face_id for _, face_id in faces]
    impedance = f"{port.impedance_ohm}ohm"
    name = port.name
    terminal_name = f"{name}_T1"
    before = set(app.oboundary.GetExcitationsOfType("Terminal"))
    # The native writers, as AEDT 2024.2 records a Terminal lumped port and its edge Terminal.
    app.oboundary.AssignLumpedPort(
        [
            f"NAME:{name}",
            "Faces:=",
            face_ids,
            "LumpedPortType:=",
            "Terminal",
            "DoDeembed:=",
            False,
            "RenormalizeAllTerminals:=",
            port.renormalize,
            "TerminalIDList:=",
            [],
            "ShowReporterFilter:=",
            False,
            "ReporterFilter:=",
            [True],
            "Impedance:=",
            impedance,
        ]
    )
    app.oboundary.AssignTerminal(
        [
            f"NAME:{terminal_name}",
            "Edges:=",
            signal_edges,
            "ParentBndID:=",
            name,
            "TerminalResistance:=",
            impedance,
        ]
    )
    terminals = sorted(set(app.oboundary.GetExcitationsOfType("Terminal")) - before)
    if name not in set(app.oboundary.GetExcitationsOfType("Lumped Port")):
        raise SgbImportError(f"AEDT did not assign the lumped port {name!r}")
    if terminals != [terminal_name]:
        raise SgbImportError(
            f"the lumped port {name!r} created the Terminals {terminals}, not one {terminal_name!r}"
        )
    return {
        "index": port.index,
        "kind": "lumped",
        "requested": port.to_payload(),
        "boundary": name,
        "terminal_excitation": terminal_name,
        "port_group": terminal.sheet["group"],
        "faces": [
            {"object_name": object_name, "face_id": face_id}
            for object_name, face_id in faces
        ],
        "host_entity_ids": hosts,
        "signal_host": hosts[ends.index(True)],
        "terminal_edges": signal_edges,
    }


def port_sheet(
    app: Any,
    terminal: PortTerminal,
    port_faces: Mapping[str, Any],
    tolerance_um: float,
) -> tuple[list[tuple[str, int]], list[tuple[int, tuple[Point, ...]]]]:
    """The faces a lumped port takes, as (object, face id), and their edges, as (edge id, vertices).

    The free faces of a port sheet are already sheet objects at the guard's exact loops.  The faces on an
    imported solid are drawn again into one sheet object, ``port_sheet_<id>``, each from its guarded vertex
    loop; a face whose loop is not a flat polygon of its guarded area (an arc, a hole) is refused.  A sheet
    holds its faces of one kind only.
    """
    planned = list(terminal.sheet["faces"])
    free = [face for face in planned if isinstance(port_faces[face], str)]
    if free and len(free) != len(planned):
        raise SgbImportError(
            f"port sheet {terminal.sheet['group']!r} has free faces and faces on solids"
        )
    if free:
        names = [port_faces[face] for face in planned]
    else:
        name = sanitise_aedt_name(f"port_sheet_{terminal.port.port_sheet_id}")
        names = [
            _draw_port_sheet(app, name, [port_faces[f] for f in planned], tolerance_um)
        ]
    faces = [(name, int(face.id)) for name in names for face in app.modeler[name].faces]
    edges = [
        (int(edge.id), tuple(_point(vertex.position) for vertex in edge.vertices))
        for name in names
        for face in app.modeler[name].faces
        for edge in face.edges
    ]
    return faces, edges


def _draw_port_sheet(
    app: Any, name: str, records: Sequence[Mapping[str, Any]], tolerance_um: float
) -> str:
    """One sheet object ``name`` drawn from the guarded vertex loops of ``records``; its area must be theirs."""
    parts = []
    total_perimeter = 0.0
    for k, record in enumerate(records):
        points = [_point(p) for p in record["vertices_um"]]
        perimeter = sum(
            math.dist(a, b)
            for a, b in zip(points, points[1:] + points[:1], strict=True)
        )
        total_perimeter += perimeter
        limit = area_tolerance_um2(record["area_um2"], perimeter, tolerance_um)
        flat = max(p[2] for p in points) - min(p[2] for p in points) <= tolerance_um
        if (
            record["curved"]
            or not flat
            or abs(_shoelace(points) - record["area_um2"]) > limit
        ):
            raise SgbImportError(
                f"port face {record['face']} is not a flat polygon of its vertices; it cannot be drawn again"
            )
        sheet = app.modeler.create_polyline(
            [list(p) for p in [*points, points[0]]],
            cover_surface=True,
            close_surface=True,
            name=f"{name}__{k}",
        )
        if not sheet:
            raise SgbImportError(f"AEDT did not draw port face {record['face']}")
        parts.append(sheet.name)
    if len(parts) > 1 and not app.modeler.unite(parts):
        raise SgbImportError(f"AEDT did not unite the faces of {name}")
    sheet = app.modeler[parts[0]]
    sheet.name = name
    expected = sum(record["area_um2"] for record in records)
    area = sum(float(face.area) for face in sheet.faces)
    if sheet.name != name or abs(area - expected) > area_tolerance_um2(
        expected, total_perimeter, tolerance_um
    ):
        raise SgbImportError(
            f"the port sheet {sheet.name!r} has {area} um2, the guard planned {expected} um2"
        )
    return name


def _shoelace(points: Sequence[Point]) -> float:
    """The area of a flat horizontal polygon."""
    twice = sum(
        a[0] * b[1] - b[0] * a[1]
        for a, b in zip(points, [*points[1:], points[0]], strict=True)
    )
    return abs(twice) / 2


def port_end_edges(
    edges: Sequence[tuple[int, Sequence[Point]]],
    direction: Point,
    tolerance_um: float,
    *,
    upper: bool,
) -> list[int]:
    """The ids of the straight edges at one end of a port face's extent along ``direction``, sorted.

    ``upper`` picks the end ``direction`` points to.  The edges there must form one unbroken segment across
    the face (SGB may split it where another face meets the port), as at each end of the current line.
    """
    vertices = [point for _, points in edges for point in points]
    along = [_dot(point, direction) for point in vertices]
    extreme = max(along) if upper else min(along)
    _port_end_midpoint(
        [points for _, points in edges], direction, extreme, tolerance_um
    )
    return sorted(
        edge_id
        for edge_id, points in edges
        if len(points) == 2
        and all(abs(_dot(p, direction) - extreme) <= tolerance_um for p in points)
    )


def current_line_um(
    edges: Sequence[Sequence[Point]], direction: Point, tolerance_um: float
) -> tuple[Point, Point]:
    """Start and end of the current line across a port face along ``direction``.

    At each end of the face's extent along ``direction`` its straight edges there must form one unbroken
    segment across it (SGB may split it where another face meets the port); the line joins the midpoints of
    the two segments, from the lower end to the upper.  The sidecar's ``direction`` points from the first host
    conductor to the second, so the line does too.
    """
    vertices = [point for edge in edges for point in edge]
    low = min(_dot(point, direction) for point in vertices)
    high = max(_dot(point, direction) for point in vertices)
    start, end = (
        _port_end_midpoint(edges, direction, extreme, tolerance_um)
        for extreme in (low, high)
    )
    return start, end


def _port_end_midpoint(
    edges: Sequence[Sequence[Point]],
    direction: Point,
    extreme: float,
    tolerance_um: float,
) -> Point:
    """The midpoint of the one unbroken segment the face's edges at ``extreme`` along ``direction`` form."""
    across = [
        edge
        for edge in edges
        if len(edge) == 2
        and all(abs(_dot(p, direction) - extreme) <= tolerance_um for p in edge)
    ]
    points = [p for edge in across for p in edge]
    if not across:
        raise SgbImportError(
            f"the port face has no edge across its direction at {extreme} um"
        )
    a, b = max(
        ((p, q) for p in points for q in points), key=lambda pair: math.dist(*pair)
    )
    covered = sum(math.dist(*edge) for edge in across)
    if abs(covered - math.dist(a, b)) > len(across) * tolerance_um:
        raise SgbImportError(
            f"the port face's edges at {extreme} um along its direction do not form one segment"
        )
    return _point([(a[k] + b[k]) / 2 for k in range(3)])


def _point(values: Sequence[Any]) -> Point:
    x, y, z = (float(value) for value in values)
    return (x, y, z)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))
