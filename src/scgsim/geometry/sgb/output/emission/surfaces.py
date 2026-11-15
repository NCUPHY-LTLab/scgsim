"""The surfaces a BREP face lies on: one plane per level and per wall, one cylinder per curved wall.

Owns turning a planned :class:`~scgsim.geometry.sgb.topology.assembly.records.TopoFace` into an OpenCASCADE
surface record -- origin, normal (the axis of a cylinder), x axis, y axis and, for a cylinder, its
radius -- and the check that every edge lies on the surfaces that will carry its pcurve.

* a horizontal face lies on the plane of its level (shared by every face of the level);
* a straight wall lies on the plane through its boundary (its own plane, Newell normal);
* a curved wall is, after assembly (design brief section 14, review R3), a patch of *one* cylinder:
  its bottom and top boundary are each one arc of the same circle.  The axis is +z.  A face that is
  neither is refused (``SgbError``), never written approximately.

Does not own numbering of shapes (``walk``) or the text of the records (``records``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from math import lcm

import numpy as np

from scgsim.geometry.exact.errors import SgbError
from scgsim.geometry.sgb.output.emission.numbers import PI, angle_of
from scgsim.geometry.sgb.topology.assembly.records import TopoEdge, TopoFace, Topology
from scgsim.refusal import internal

PLANE, CYLINDER = 0, 1  # surface kinds
CIRCLE_SPREAD_LIMIT_UM = (
    1e-8  # arcs of one cylinder wall must share one circle to this distance
)
FULL_TURN_MARGIN = (
    1e-6  # radians: a wall spanning more than 2 pi - this wraps the cylinder
)

Vec3 = tuple[float, float, float]
Frame = tuple[Vec3, Vec3, Vec3, Vec3]  # origin, normal (axis), x axis, y axis


@dataclass(frozen=True)
class SurfaceRecord:
    """A plane (``radius`` 0) or a cylinder about the +z axis through ``frame[0]``."""

    kind: int
    frame: Frame
    radius: float = 0.0


class SurfaceBuilder:
    """Creates the surface of each face in turn; faces of one level share a plane."""

    def __init__(self, topology: Topology) -> None:
        self.topology = topology
        self.grid_um = topology.grid_um
        self.zden = (
            lcm(*(z.denominator for z in topology.levels)) if topology.levels else 1
        )
        self.records: list[SurfaceRecord] = []
        self._level_plane: dict[int, int] = {}

    def point(self, vertex_id: str) -> Vec3:
        """Position of a vertex in micrometres (a lattice integer times the grid, exact scaling)."""
        v = self.topology.vertices[vertex_id]
        return (v.x * self.grid_um, v.y * self.grid_um, v.zs / self.zden)

    def surface_of(self, face: TopoFace) -> tuple[int, bool]:
        """(surface number, wire flipped) of a face; a flipped wire is written reversed (cylinders only)."""
        if face.curved:
            return self._cylinder_of(face)
        if not face.planar:
            raise SgbError(
                internal(
                    f"face {face.id}: non-planar, non-cylindrical face (needs a B-spline record; not implemented)"
                )
            )
        if face.kind == "horizontal":
            return self._level_plane_of(face), False
        topology = self.topology
        corners = [
            self.point(topology.edges[eid].v0 if sign > 0 else topology.edges[eid].v1)
            for eid, sign in face.loops[0]
        ]
        self.records.append(SurfaceRecord(PLANE, plane_through(corners, face.id)))
        return len(self.records) - 1, False

    def _level_plane_of(self, face: TopoFace) -> int:
        topology = self.topology
        if face.twice_xy_area <= 0:
            raise SgbError(
                internal(f"horizontal face {face.id} is not counter-clockwise about +z")
            )
        zs = topology.vertices[topology.edges[face.loops[0][0][0]].v0].zs
        plane = self._level_plane.get(zs)
        if plane is None:
            plane = self._level_plane[zs] = len(self.records)
            frame = (
                (0.0, 0.0, zs / self.zden),
                (0.0, 0.0, 1.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
            )
            self.records.append(SurfaceRecord(PLANE, frame))
        return plane

    def _cylinder_of(self, face: TopoFace) -> tuple[int, bool]:
        """The cylinder carrying a curved wall whose horizontal edges are arcs of one circle.

        The surface's natural normal points away from the axis.  The loop's normal (towards the
        positive side of the wall) is the opposite when the top arcs run counter-clockwise; then the
        wire is reversed and every shell uses the face with the opposite orientation (``flipped``).
        The x axis is the unit radial vector of the clockwise-most vertex, so the angle ``u`` of the
        parametrisation starts there and needs no trigonometry to define.
        """
        topology, g = self.topology, self.grid_um
        if len(face.loops) != 1:
            raise SgbError(internal(f"curved wall {face.id} has holes"))
        arcs = self._wall_arcs(face)
        circle = arcs[0][0].curve
        assert circle is not None  # _wall_arcs keeps arc edges only
        cx, cy, radius = circle.centre[0] * g, circle.centre[1] * g, circle.radius * g
        _check_one_circle(face, arcs, (cx, cy, radius), g)
        ends = [self.point(vid) for edge, _ in arcs for vid in (edge.v0, edge.v1)]
        radial = np.array([(x - cx, y - cy) for x, y, _ in ends])
        reference = radial[0]
        cross = reference[0] * radial[:, 1] - reference[1] * radial[:, 0]
        dot = reference[0] * radial[:, 0] + reference[1] * radial[:, 1]
        offsets = angle_of(cross, dot)
        if offsets.max() - offsets.min() > 2 * PI - FULL_TURN_MARGIN:
            raise SgbError(internal(f"curved wall {face.id} wraps the whole cylinder"))
        start = radial[int(np.argmin(offsets))]
        length = math.sqrt(start[0] * start[0] + start[1] * start[1])
        x_axis = (start[0] / length, start[1] / length, 0.0)
        y_axis = (-x_axis[1], x_axis[0], 0.0)
        top = max(topology.vertices[edge.v0].zs for edge, _ in arcs)
        senses = {
            edge.curve.ccw == (sign > 0)
            for edge, sign in arcs
            if edge.curve is not None and topology.vertices[edge.v0].zs == top
        }
        if len(senses) != 1:
            raise SgbError(
                internal(f"curved wall {face.id}: top arcs run in different senses")
            )
        self.records.append(
            SurfaceRecord(
                CYLINDER, ((cx, cy, 0.0), (0.0, 0.0, 1.0), x_axis, y_axis), radius
            )
        )
        return len(self.records) - 1, senses.pop()

    def _wall_arcs(self, face: TopoFace) -> list[tuple[TopoEdge, int]]:
        arcs = []
        for eid, sign in face.loops[0]:
            edge = self.topology.edges[eid]
            if edge.kind == "vertical":
                continue
            if edge.curve is None:
                raise SgbError(
                    internal(
                        f"curved wall {face.id} has a straight horizontal edge (not a cylinder patch)"
                    )
                )
            arcs.append((edge, sign))
        return arcs


def _check_one_circle(
    face: TopoFace,
    arcs: list[tuple[TopoEdge, int]],
    circle: tuple[float, float, float],
    g: float,
) -> None:
    cx, cy, radius = circle
    spread = max(
        max(
            abs(e.curve.centre[0] * g - cx),
            abs(e.curve.centre[1] * g - cy),
            abs(e.curve.radius * g - radius),
        )
        for e, _ in arcs
        if e.curve is not None
    )
    if spread > CIRCLE_SPREAD_LIMIT_UM:
        raise SgbError(
            internal(
                f"curved wall {face.id}: its {len(arcs)} horizontal arcs lie on different circles (centres differ by "
                f"{spread * 1000:.3g} nm); a single cylinder cannot carry the wall, and assembly should have split "
                "the arc at the foreign vertex"
            )
        )


def plane_through(points: list[Vec3], face_id: str) -> Frame:
    """Plane of a planar loop (Newell normal about the loop, right-hand rule); Y = N x X, so (X, Y, N) is right-handed."""
    p0 = points[0]
    rel = [(x - p0[0], y - p0[1], z - p0[2]) for x, y, z in points]
    nx = ny = nz = 0.0
    for (ax, ay, az), (bx, by, bz) in zip(rel, rel[1:] + rel[:1], strict=True):
        nx += (ay - by) * (az + bz)
        ny += (az - bz) * (ax + bx)
        nz += (ax - bx) * (ay + by)
    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if norm == 0.0:
        raise SgbError(internal(f"face {face_id}: degenerate loop"))
    nx, ny, nz = nx / norm, ny / norm, nz / norm
    if (
        abs(nz) < 1e-9
    ):  # vertical plane: y axis is +z, x axis = y x N, so that x x y = N
        scale = math.sqrt(nx * nx + ny * ny)
        nx, ny = nx / scale, ny / scale
        return (p0, (nx, ny, 0.0), (-ny, nx, 0.0), (0.0, 0.0, 1.0))
    ax, ay, az = (
        1.0 - nx * nx,
        -nx * ny,
        -nx * nz,
    )  # tilted plane: x axis is +x projected into the plane
    scale = math.sqrt(ax * ax + ay * ay + az * az)
    x_axis = (ax / scale, ay / scale, az / scale)
    y_axis = (
        ny * x_axis[2] - nz * x_axis[1],
        nz * x_axis[0] - nx * x_axis[2],
        nx * x_axis[1] - ny * x_axis[0],
    )
    return (p0, (nx, ny, nz), x_axis, y_axis)


def check_edges_lie_on_surfaces(  # noqa: PLR0913 - the edge and surface tables travel together
    verts: np.ndarray,
    edge_v: np.ndarray,
    kind: np.ndarray,
    centre: np.ndarray,
    surf_kind: np.ndarray,
    surf_geom: np.ndarray,
    surf_radius: np.ndarray,
    pcurve_edge: np.ndarray,
    pcurve_surface: np.ndarray,
    tolerance: float,
    arc_kind: int,
) -> None:
    """Every edge must lie on every surface that carries a pcurve of it, within a tenth of the tolerance."""
    if not len(pcurve_edge):
        return
    limit = tolerance / 10
    origin, normal = surf_geom[pcurve_surface, 0], surf_geom[pcurve_surface, 1]
    x_axis, y_axis = surf_geom[pcurve_surface, 2], surf_geom[pcurve_surface, 3]
    is_plane = surf_kind[pcurve_surface] == PLANE
    worst = 0.0
    for end in (0, 1):
        rel = verts[edge_v[pcurve_edge, end]] - origin
        plane_distance = np.abs(np.einsum("ij,ij->i", rel, normal))
        axis_distance = np.hypot(
            np.einsum("ij,ij->i", rel, x_axis), np.einsum("ij,ij->i", rel, y_axis)
        )
        off = np.where(
            is_plane,
            plane_distance,
            np.abs(axis_distance - surf_radius[pcurve_surface]),
        )
        worst = max(worst, float(off.max()))
    arcs = kind[pcurve_edge] == arc_kind
    if arcs.any():  # a circle lies on a plane when its centre does (axis +z), on a cylinder when its centre is on the axis
        rel = centre[pcurve_edge] - origin
        plane_distance = np.abs(np.einsum("ij,ij->i", rel, normal))
        axis_distance = np.hypot(
            np.einsum("ij,ij->i", rel, x_axis), np.einsum("ij,ij->i", rel, y_axis)
        )
        worst = max(
            worst, float(np.where(is_plane, plane_distance, axis_distance)[arcs].max())
        )
    if worst > limit:
        raise SgbError(
            internal(f"an edge leaves its surface by {worst:.3g} um (limit {limit:g})")
        )
