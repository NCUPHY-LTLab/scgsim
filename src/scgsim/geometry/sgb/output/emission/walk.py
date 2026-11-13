"""Numbering of every shape in OpenCASCADE's ``TopExp::MapShapes`` order (design brief decision D8).

Owns the depth-first walk over the emitted volumes -- compound, solid, shell, face, wire, edge,
vertices -- that assigns every sub-shape its index.  The order is exactly OCCT's pre-order map, so
the native tag Gmsh gives a shape after loading the file is ``index + 1`` and the planned id of each
native entity is known by construction, never by matching geometry.  A shared face, edge or vertex is
numbered once, where it is first reached, and referenced by index everywhere else.

The walk also decides, per face, its surface (``surfaces``), the wires of its loops, and the free
shapes of the compound: the embedded faces (a sheet inside a volume).

Does not own the surfaces' geometry (``surfaces``), the arrays (``plan``) or the text (``records``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from scgsim.geometry.sgb.errors import SgbError
from scgsim.geometry.sgb.output.emission.surfaces import SurfaceBuilder
from scgsim.geometry.sgb.topology.assembly.records import Loop, TopoFace, Topology
from scgsim.geometry.sgb.topology.routes.groups import RoutedModel

EdgeUse = tuple[
    str, int
]  # (edge id, +1 / -1 relative to the edge's v0 -> v1 direction)


@dataclass
class Numbering:
    """What the walk found: ids in tag order and the wires, shells and references that connect them."""

    vertex_ids: list[str] = field(default_factory=list)
    edge_ids: list[str] = field(default_factory=list)
    face_ids: list[str] = field(default_factory=list)
    solid_ids: list[str] = field(default_factory=list)
    edge_ends: list[tuple[int, int]] = field(
        default_factory=list
    )  # (start, end) vertex indices
    wires: list[list[int]] = field(default_factory=list)  # signed 1-based edge numbers
    face_surface: list[int] = field(default_factory=list)
    face_flipped: list[bool] = field(default_factory=list)
    face_first_wire: list[int] = field(default_factory=list)
    face_wire_count: list[int] = field(default_factory=list)
    pcurves: dict[int, int] = field(
        default_factory=dict
    )  # (edge << 32 | surface) -> pcurve number
    shells: list[list[int]] = field(default_factory=list)  # signed 1-based face numbers
    solid_first_shell: list[int] = field(default_factory=list)
    solid_shell_count: list[int] = field(default_factory=list)
    free_faces: list[int] = field(default_factory=list)
    solid_ref: list[int] = field(
        default_factory=list
    )  # MapShapes index of each shape (for the XAO)
    face_ref: list[int] = field(default_factory=list)
    edge_ref: list[int] = field(default_factory=list)
    vertex_ref: list[int] = field(default_factory=list)


def walk_emitted_volumes(
    topology: Topology, routed: RoutedModel, surfaces: SurfaceBuilder
) -> Numbering:
    """Number the shapes of every emitted volume, then the free faces of the compound."""
    emitted = [
        (vid, volume)
        for vid, volume in sorted(topology.volumes.items())
        if vid in routed.emitted_volumes
    ]
    if not emitted:
        raise SgbError(
            f"Route {routed.policy.name} emits no volume of this model, so there is no solid to write; "
            "give the input a solution region (vacuum or dielectric), or choose Route C to keep the conductors"
        )
    walk = _Walk(topology, surfaces)
    for vid, volume in emitted:
        walk.solid(vid, [list(shell.faces) for shell in volume.shells])
    for _, volume in emitted:
        for face_id in volume.embedded_faces:
            walk.free_face(face_id)
    return walk.found


class _Walk:
    """The depth-first walk; ``ref`` counts shapes in pre-order starting with the compound."""

    def __init__(self, topology: Topology, surfaces: SurfaceBuilder) -> None:
        self.topology = topology
        self.surfaces = surfaces
        self.found = Numbering()
        self.ref = 1  # the compound
        self._face_no: dict[str, int] = {}
        self._edge_no: dict[str, int] = {}
        self._vertex_no: dict[str, int] = {}

    def solid(self, solid_id: str, shells: list[list[EdgeUse]]) -> None:
        """Number a solid and, below it, its shells and everything they reach."""
        found = self.found
        self.ref += 1
        found.solid_ref.append(self.ref)
        found.solid_ids.append(solid_id)
        found.solid_first_shell.append(len(found.shells))
        found.solid_shell_count.append(len(shells))
        for faces in shells:
            self.ref += 1
            found.shells.append(
                [self._oriented(sign, self._face(fid)) for fid, sign in faces]
            )

    def free_face(self, face_id: str) -> None:
        """An embedded face that bounds no shell: a child of the compound next to the solids."""
        known = face_id in self._face_no
        number = self._face(face_id)
        if not known:
            self.found.free_faces.append(number)

    def _oriented(self, sign: int, face_number: int) -> int:
        """Signed 1-based face reference; a flipped face is used with the opposite orientation."""
        flipped = self.found.face_flipped[face_number]
        return -sign * (face_number + 1) if flipped else sign * (face_number + 1)

    def _face(self, face_id: str) -> int:
        number = self._face_no.get(face_id)
        if number is not None:
            return number
        found = self.found
        face = self.topology.faces[face_id]
        number = self._face_no[face_id] = len(found.face_ids)
        found.face_ids.append(face_id)
        self.ref += 1
        found.face_ref.append(self.ref)
        surface, flipped = self.surfaces.surface_of(face)
        found.face_surface.append(surface)
        found.face_flipped.append(flipped)
        found.face_first_wire.append(len(found.wires))
        loops = self._wires_of(face)
        found.face_wire_count.append(len(loops))
        for loop in loops:
            self.ref += 1  # the wire
            if flipped:  # visit edges in the order they are stored in the written (reversed) wire
                loop = [(eid, -sign) for eid, sign in reversed(loop)]
            signed = [sign * (self._edge(eid) + 1) for eid, sign in loop]
            if len({abs(s) for s in signed}) != len(signed):
                raise SgbError(
                    f"internal: face {face_id}: an edge occurs twice in one wire (seam/bridge edges are not written)"
                )
            for s in signed:
                found.pcurves.setdefault(
                    ((abs(s) - 1) << 32) | surface, len(found.pcurves)
                )
            found.wires.append(signed)
        return number

    def _wires_of(self, face: TopoFace) -> list[list[EdgeUse]]:
        """The wires of a face: its outer loop, then every hole loop split into simple loops.

        Two holes that touch at a vertex are one boundary cycle that visits the vertex twice (a
        figure eight).  OCCT treats such a wire as unorientable (``BRepCheck_UnorientableShape``), so
        each lobe is written as its own wire; the lobes share the vertex.
        """
        out = [list(face.loops[0])]
        for loop in face.loops[1:]:
            out.extend(self._simple_lobes(loop))
        return out

    def _simple_lobes(self, loop: Loop) -> list[list[EdgeUse]]:
        edges = self.topology.edges

        def start(item: EdgeUse) -> str:
            return edges[item[0]].v0 if item[1] > 0 else edges[item[0]].v1

        lobes: list[list[EdgeUse]] = []
        path: list[EdgeUse] = []
        at: dict[
            str, int
        ] = {}  # vertex -> index in ``path`` of the item that starts there
        for item in loop:
            v = start(item)
            if v in at:
                i = at[v]
                lobes.append(path[i:])
                for dropped in path[i:]:
                    at.pop(start(dropped), None)
                del path[i:]
            at[v] = len(path)
            path.append(item)
        lobes.append(path)
        return [lobe for lobe in lobes if lobe]

    def _edge(self, edge_id: str) -> int:
        number = self._edge_no.get(edge_id)
        if number is not None:
            return number
        found = self.found
        edge = self.topology.edges[edge_id]
        number = self._edge_no[edge_id] = len(found.edge_ids)
        found.edge_ids.append(edge_id)
        self.ref += 1
        found.edge_ref.append(self.ref)
        # children are written end vertex first, then start vertex (as Gmsh does); pre-order follows that order
        end, start = self._vertex(edge.v1), self._vertex(edge.v0)
        found.edge_ends.append((start, end))
        return number

    def _vertex(self, vertex_id: str) -> int:
        number = self._vertex_no.get(vertex_id)
        if number is None:
            found = self.found
            number = self._vertex_no[vertex_id] = len(found.vertex_ids)
            found.vertex_ids.append(vertex_id)
            self.ref += 1
            found.vertex_ref.append(self.ref)
        return number
