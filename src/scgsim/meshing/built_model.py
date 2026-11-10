"""The built model as the sidecar records it, and the sizes of its fine edges and faces.

Owns reading the geometry and lineage that the meshing-stage check needs from an SGB sidecar, by native tag
(tag = 1 + position in ``binding.ids_by_tag``):

* ``expected``: where every vertex sits, the two vertices of every edge, the edges of every face and the face
  areas.  These are the shapes the XAO holds, so measuring them measures the model the mesher receives;
* ``mesh_features``: the radius of every circular-arc edge (``arc``) and the margins of every band family
  (``band_boundary``);
* ``faces``: each face's kind (``horizontal`` or ``wall``) and physical group;
* ``lineage``: the carriers of every horizontal edge and what each carrier comes from (a source polygon edge
  or a member of a cut family), the Entities a wall column stands under, and the sources of every face.

It measures the four sizes the meshing-stage classes are stated in (issue 125): the column gap of a wall
face, the width ``2 A / P`` of a horizontal face, the corner (wedge) angle of a horizontal face, and the
length of a horizontal edge that bounds no wall.  Corner angles are taken between the chords of the boundary
edges, as SGB's sliver report takes them.  Nothing here decides what is fine: that is the limits' job.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .errors import SidecarError

Point3 = tuple[float, float, float]
Bounds = tuple[Point3, Point3]

_EDGE_DIM = 1
_FACE_DIM = 2
_HORIZONTAL = "horizontal"
_FULL_TURN_DEG = 360.0


@dataclass(frozen=True)
class Carrier:
    """What one carrier of a horizontal edge comes from, as ``lineage.carriers`` records it.

    ``kind`` is ``polygon_edge`` (a source polygon edge of ``entity``), ``family_member`` (a curve that
    SGB generated for the cut family ``family``, named ``member``) or ``unknown``.
    """

    id: str
    kind: str
    entity: str | None = None
    family: str | None = None
    member: str | None = None


@dataclass(frozen=True)
class BuiltModel:
    """The geometry and lineage of one SGB build, by native tag."""

    grid_um: float
    vertices: tuple[Point3, ...]
    edge_vertices: tuple[tuple[int, int], ...]
    face_edges: tuple[tuple[int, ...], ...]
    face_areas: tuple[float, ...]
    edge_ids: tuple[str, ...]
    face_ids: tuple[str, ...]
    arc_radius: Mapping[
        int, float
    ]  # native edge tag -> radius of its circle, micrometres
    face_kind: Mapping[str, str]  # face id -> "horizontal" or "wall"
    face_group: Mapping[str, str | None]
    face_sources: Mapping[str, tuple[str, ...]]
    edge_carriers: Mapping[str, tuple[str, ...]]  # horizontal edge id -> carrier ids
    vertical_edges: Mapping[str, tuple[str, ...]]  # wall column id -> Entities
    carriers: Mapping[str, Carrier]
    band_margins: Mapping[
        str, tuple[float, ...]
    ]  # band family id -> margins, micrometres

    def point(self, vertex: int) -> Point3:
        """Where native vertex ``vertex`` sits."""
        return self.vertices[vertex - 1]

    def ends(self, edge: int) -> tuple[Point3, Point3]:
        """The two end points of native edge ``edge``."""
        a, b = self.edge_vertices[edge - 1]
        return self.point(a), self.point(b)

    def edge_id(self, edge: int) -> str:
        """The planned id of native edge ``edge``."""
        return self.edge_ids[edge - 1]

    def face_id(self, face: int) -> str:
        """The planned id of native face ``face``."""
        return self.face_ids[face - 1]

    def is_vertical(self, edge: int) -> bool:
        """Whether ``edge`` is a wall column: its two ends differ only in z."""
        return self.edge_id(edge) in self.vertical_edges

    def edge_length(self, edge: int) -> float:
        """Length of ``edge``: its chord, or the arc over that chord on the edge's circle."""
        a, b = self.ends(edge)
        chord = math.dist(a, b)
        radius = self.arc_radius.get(edge)
        if radius is None:
            return chord
        # SGB splits every circle into arcs of at most a quarter turn, so the minor arc is the edge.
        return 2 * radius * math.asin(min(1.0, chord / (2 * radius)))

    def is_horizontal_face(self, face: int) -> bool:
        """Whether native face ``face`` lies in one z plane."""
        return self.face_kind.get(self.face_id(face)) == _HORIZONTAL

    def bounds(self, edges: tuple[int, ...]) -> Bounds:
        """The axis-aligned box of the end points of ``edges``."""
        points = [p for edge in edges for p in self.ends(edge)]
        return (
            (
                min(p[0] for p in points),
                min(p[1] for p in points),
                min(p[2] for p in points),
            ),
            (
                max(p[0] for p in points),
                max(p[1] for p in points),
                max(p[2] for p in points),
            ),
        )


def read_built_model(sidecar: Mapping[str, Any]) -> BuiltModel:
    """The built model recorded in a decoded ``scgsim.sgb.sidecar.v2`` sidecar."""
    expected = _mapping(sidecar.get("expected"), "expected")
    ids_by_tag = _mapping(
        _mapping(sidecar.get("binding"), "binding").get("ids_by_tag"),
        "binding.ids_by_tag",
    )
    lineage = _mapping(sidecar.get("lineage"), "lineage")
    faces = _mapping(sidecar.get("faces"), "faces")
    features = _list(sidecar.get("mesh_features"), "mesh_features")
    return BuiltModel(
        grid_um=float(sidecar["grid_um"]),
        vertices=tuple(
            (float(x), float(y), float(z))
            for x, y, z in _list(expected.get("vertices_um"), "expected.vertices_um")
        ),
        edge_vertices=tuple(
            (int(a), int(b))
            for a, b in _list(expected.get("edge_vertices"), "expected.edge_vertices")
        ),
        face_edges=tuple(
            tuple(int(e) for e in edges)
            for edges in _list(expected.get("face_edges"), "expected.face_edges")
        ),
        face_areas=tuple(
            float(a)
            for a in _list(expected.get("face_area_um2"), "expected.face_area_um2")
        ),
        edge_ids=tuple(_list(ids_by_tag.get(str(_EDGE_DIM)), "edge ids")),
        face_ids=tuple(_list(ids_by_tag.get(str(_FACE_DIM)), "face ids")),
        arc_radius={
            int(tag): float(entry["facts"]["radius_um"])
            for entry in features
            if entry.get("kind") == "arc"
            for tag in entry["native_tags"]
        },
        face_kind={fid: str(record.get("kind")) for fid, record in faces.items()},
        face_group={fid: record.get("group") for fid, record in faces.items()},
        face_sources={
            fid: tuple(sources)
            for fid, sources in _mapping(lineage.get("faces"), "lineage.faces").items()
        },
        edge_carriers={
            eid: tuple(carriers)
            for eid, carriers in _mapping(lineage.get("edges"), "lineage.edges").items()
        },
        vertical_edges={
            eid: tuple(entities)
            for eid, entities in _mapping(
                lineage.get("vertical_edges"), "lineage.vertical_edges"
            ).items()
        },
        carriers={
            cid: Carrier(
                cid,
                str(record.get("kind")),
                record.get("entity"),
                record.get("family"),
                record.get("member"),
            )
            for cid, record in _mapping(
                lineage.get("carriers"), "lineage.carriers"
            ).items()
        },
        band_margins={
            str(entry["source"]): tuple(float(m) for m in entry["facts"]["margins_um"])
            for entry in features
            if entry.get("kind") == "band_boundary"
        },
    )


@dataclass(frozen=True)
class Measured:
    """One measured edge, face or face corner, before any limit is applied.

    ``measure`` names the size: ``wall column gap``, ``face width 2A/P`` or ``edge length`` (micrometres), or
    ``wedge angle`` (degrees).  ``edges`` are the native edges whose lineage says where it comes from: a
    wall's horizontal edges, a face's boundary, the two edges of a corner, or the edge itself.
    """

    measure: str
    dimension: int
    tag: int
    value: float
    location: Point3
    bounds: Bounds
    edges: tuple[int, ...]


def wall_gaps(model: BuiltModel) -> list[Measured]:
    """The column gap of every wall face: the shortest distance in plan between two of its wall columns.

    This is SGB's wall measure without the height: a wall stands on a horizontal edge, so its column gap
    is the length of the outline edge it stands on.
    """
    measured = []
    for face, edges in enumerate(model.face_edges, start=1):
        if model.is_horizontal_face(face):
            continue
        columns = sorted(
            {model.ends(edge)[0][:2] for edge in edges if model.is_vertical(edge)}
        )
        if len(columns) < 2:  # noqa: PLR2004 -- a gap needs two columns
            continue
        gap = min(
            math.dist(p, q) for i, p in enumerate(columns) for q in columns[i + 1 :]
        )
        bounds = model.bounds(edges)
        measured.append(
            Measured(
                "wall column gap",
                _FACE_DIM,
                face,
                gap,
                _centre(bounds),
                bounds,
                tuple(e for e in edges if not model.is_vertical(e)),
            )
        )
    return measured


def face_widths(model: BuiltModel) -> list[Measured]:
    """The width ``2 A / P`` of every horizontal face, with the area SGB planned and the edge lengths."""
    measured = []
    for face, edges in enumerate(model.face_edges, start=1):
        if not model.is_horizontal_face(face):
            continue
        perimeter = sum(model.edge_length(edge) for edge in edges)
        area = model.face_areas[face - 1]
        if perimeter <= 0:
            continue
        bounds = model.bounds(edges)
        measured.append(
            Measured(
                "face width 2A/P",
                _FACE_DIM,
                face,
                2 * area / perimeter,
                _centre(bounds),
                bounds,
                edges,
            )
        )
    return measured


def closest_far_edges(
    model: BuiltModel, edges: tuple[int, ...]
) -> tuple[int, int] | None:
    """The two edges of a face that share no vertex and whose chords come closest: where the face is narrowest.

    ``2 A / P`` says a face is narrow, not where; this pair says between which two curves it is.  Two
    boundary edges of one face do not cross, so their distance is that of an end point to the other chord.
    """
    best: tuple[float, int, int] | None = None
    for i, first in enumerate(edges):
        a = model.edge_vertices[first - 1]
        for second in edges[i + 1 :]:
            b = model.edge_vertices[second - 1]
            if set(a) & set(b):
                continue
            pa, pb = model.ends(first), model.ends(second)
            distance = min(
                _point_chord_distance(pa[0], pb),
                _point_chord_distance(pa[1], pb),
                _point_chord_distance(pb[0], pa),
                _point_chord_distance(pb[1], pa),
            )
            if best is None or distance < best[0]:
                best = (distance, first, second)
    return None if best is None else (best[1], best[2])


def wedge_angles(model: BuiltModel) -> list[Measured]:
    """The smallest corner angle of every horizontal face, measured inside the face between edge chords."""
    measured = []
    for face, edges in enumerate(model.face_edges, start=1):
        if not model.is_horizontal_face(face):
            continue
        corner = _sharpest_corner(model, edges)
        if corner is None:
            continue
        angle, vertex, pair = corner
        point = model.point(vertex)
        measured.append(
            Measured("wedge angle", _FACE_DIM, face, angle, point, (point, point), pair)
        )
    return measured


def free_edges(model: BuiltModel) -> list[Measured]:
    """The length of every horizontal edge that bounds no wall face (a band arc, a seam piece, a sheet edge)."""
    on_walls = {
        edge
        for face, edges in enumerate(model.face_edges, start=1)
        if not model.is_horizontal_face(face)
        for edge in edges
    }
    measured = []
    for edge in range(1, len(model.edge_vertices) + 1):
        if edge in on_walls or model.is_vertical(edge):
            continue
        bounds = model.bounds((edge,))
        measured.append(
            Measured(
                "edge length",
                _EDGE_DIM,
                edge,
                model.edge_length(edge),
                _centre(bounds),
                bounds,
                (edge,),
            )
        )
    return measured


def _sharpest_corner(
    model: BuiltModel, edges: tuple[int, ...]
) -> tuple[float, int, tuple[int, int]] | None:
    """The smallest interior corner angle of a horizontal face, its vertex and its two edges.

    The face's edges are walked into loops; the loop of largest area is the outer one, turned
    counter-clockwise, and the others are holes, turned clockwise, so the face lies to the left of every loop.
    A loop of fewer than three vertices (two arcs) has no chord corner and is skipped.
    """
    best: tuple[float, int, tuple[int, int]] | None = None
    loops = _loops(model, edges)
    if not loops:
        return None
    areas = [_signed_area(model, loop) for loop in loops]
    outer = max(range(len(loops)), key=lambda i: abs(areas[i]))
    for index, loop in enumerate(loops):
        if len(loop) < 3:  # noqa: PLR2004 -- a polygon needs three vertices
            continue
        counter_clockwise = index == outer
        if (areas[index] > 0) != counter_clockwise:
            loop = _reversed(loop)  # noqa: PLW2901 -- the same loop, walked the other way
        for k, (vertex, incoming) in enumerate(loop):
            following_vertex, outgoing = loop[(k + 1) % len(loop)]
            previous_vertex = loop[k - 1][0]
            angle = _interior_angle(
                model.point(previous_vertex),
                model.point(vertex),
                model.point(following_vertex),
            )
            if best is None or angle < best[0]:
                best = (angle, vertex, (incoming, outgoing))
    return best


def _loops(model: BuiltModel, edges: tuple[int, ...]) -> list[list[tuple[int, int]]]:
    """The closed loops of a face as (vertex, edge arriving at it) pairs; empty when a vertex is not on two edges."""
    around: dict[int, list[int]] = {}
    for edge in edges:
        for vertex in model.edge_vertices[edge - 1]:
            around.setdefault(vertex, []).append(edge)
    if any(len(incident) != 2 for incident in around.values()):  # noqa: PLR2004 -- a pinch has no single corner
        return []
    unused = set(edges)
    loops = []
    while unused:
        first = min(unused)
        _, vertex = model.edge_vertices[first - 1]
        loop, edge = [], first
        while True:
            unused.discard(edge)
            loop.append((vertex, edge))
            following = next((e for e in around[vertex] if e != edge), None)
            if following is None or following == first:
                break  # back at the first edge (an edge closed on itself has no other)
            edge = following
            a, b = model.edge_vertices[edge - 1]
            vertex = b if a == vertex else a
        loops.append(loop)
    return loops


def _reversed(loop: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """The loop walked the other way, still as (vertex, edge arriving at it) pairs."""
    vertices = [vertex for vertex, _ in loop]
    arriving = [edge for _, edge in loop]
    count = len(loop)
    # Walking backwards, the edge that arrives at vertex k is the one that used to leave it.
    return [(vertices[k], arriving[(k + 1) % count]) for k in range(count - 1, -1, -1)]


def _signed_area(model: BuiltModel, loop: list[tuple[int, int]]) -> float:
    points = [model.point(vertex) for vertex, _ in loop]
    return 0.5 * sum(
        p[0] * q[1] - q[0] * p[1]
        for p, q in zip(points, points[1:] + points[:1], strict=True)
    )


def _interior_angle(previous: Point3, vertex: Point3, following: Point3) -> float:
    """The angle at ``vertex`` on the left of the walk previous -> vertex -> following, in degrees."""
    back = math.atan2(previous[1] - vertex[1], previous[0] - vertex[0])
    ahead = math.atan2(following[1] - vertex[1], following[0] - vertex[0])
    # Turning counter-clockwise from the edge ahead to the edge behind sweeps the region on the left.
    return math.degrees(back - ahead) % _FULL_TURN_DEG


def _point_chord_distance(point: Point3, chord: tuple[Point3, Point3]) -> float:
    """Distance in plan from ``point`` to the segment between the chord's ends."""
    (ax, ay, _), (bx, by, _) = chord
    dx, dy = bx - ax, by - ay
    span = dx * dx + dy * dy
    t = (
        0.0
        if span == 0
        else max(0.0, min(1.0, ((point[0] - ax) * dx + (point[1] - ay) * dy) / span))
    )
    return math.hypot(point[0] - (ax + t * dx), point[1] - (ay + t * dy))


def _centre(bounds: Bounds) -> Point3:
    low, high = bounds
    return (
        (low[0] + high[0]) / 2,
        (low[1] + high[1]) / 2,
        (low[2] + high[2]) / 2,
    )


def _mapping(value: Any, what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SidecarError(f"sidecar {what} must be an object.")
    return value


def _list(value: Any, what: str) -> list[Any]:
    if not isinstance(value, list):
        raise SidecarError(f"sidecar {what} must be a list.")
    return value
