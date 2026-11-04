"""Direct OpenCASCADE BREP / XAO serialisation of an already-planned topology (experiment E3, brief D8).

Instead of replaying the plan through the Gmsh/OCC API (``emit``: ``addPoint`` ... ``addVolume``,
``synchronize``, ``gmsh.write``), this module *prints* the plan as an ASCII ``CASCADE Topology V1``
BREP: one COMPOUND of SOLIDs (the emitted volumes), where every vertex, edge and face is a TShape
written **once** and referenced by index from every shell/wire that uses it. That sharing is what
makes the model conformal; no Boolean, no healing, no CAD kernel is involved in writing.

Units: model units (um). All shape tolerances are ``TOLERANCE`` = 1e-7 um (OCCT's own
``Precision::Confusion``), which is 1e4 times smaller than the model grid (1 nm).

Pipeline (two stages, both owned here; nothing else in the package is touched):

1. ``plan_brep`` (serial, small): walks volumes -> shells -> faces -> wires -> edges -> vertices in
   exactly the order of OCCT's ``TopExp::MapShapes`` pre-order and numbers every sub-shape in
   that order. That numbering *is* the Gmsh tag order (tag = index + 1, verified by tests) and the
   XAO ``reference`` of each sub-shape, so the planned-id -> native-tag map is known without the
   mesher (never by geometry matching). It also builds the surfaces and the (edge, surface) pcurve table,
   and checks every edge lies on its surfaces within a tenth of the tolerance.
2. ``render_*`` (chunked, parallelisable): turns index ranges of every section into text. Chunks
   depend only on the read-only plan, so ``workers > 1`` forks a pool that inherits the plan and
   returns encoded text in file order; the result is byte-identical for every ``workers``.

Geometry written: straight edges (Line) on planes (one Plane per horizontal level, one per wall), circular
arc edges (Circle, normal +z or -z by travel sense) on horizontal planes or on the Cylinder (axis +z) of a curved
wall. A new kind plugs in through the dispatch tables ``CURVE_RECORDS``, ``SURFACE_RECORDS`` and ``PCURVE_RECORDS``
(a B-spline wall would add a surface record and a pcurve record per edge kind); numbering, sharing and section
layout do not change. Faces that are neither planar nor a one-circle cylinder raise ``SgbError``.

Does not own topology decisions (``assembly``), planning identity, or meshing.
"""

from __future__ import annotations

import gc
import json
import math
import multiprocessing
import time
from dataclasses import dataclass, field
from math import lcm
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np

from sgbproto.assembly import Topology
from sgbproto.errors import SgbError

TOLERANCE = 1e-7  # um; every vertex, edge and face carries this tolerance
CHUNK_ITEMS = 40_000  # items of one section per text-generation task

# Shape flag words copied from files written by Gmsh 4.15.2 (free, modified, checked, orientable, closed, ...).
FLAGS = {"Ve": "0101101", "Ed": "0101000", "Wi": "0101100", "Fa": "0101000", "Sh": "0101100", "So": "0100000",
         "Co": "1100000"}
HORIZONTAL = "horizontal"
LINE, ARC = 0, 1  # edge kinds
PLANE, CYLINDER = 0, 1  # surface kinds


@dataclass
class BrepPlan:
    """Numbered, flat description of the shapes to print (indices are 0-based in tag order)."""

    tolerance: float
    vertex_ids: list[str]
    edge_ids: list[str]
    face_ids: list[str]
    solid_ids: list[str]
    verts: np.ndarray  # (nV, 3) um
    edge_v: np.ndarray  # (nE, 2) vertex indices, start/end
    edge_kind: np.ndarray  # (nE,) LINE | ARC
    edge_dir: np.ndarray  # (nE, 3) unit direction (line) / unit centre -> start vector (arc)
    edge_span: np.ndarray  # (nE,) parameter range end: length (line) / angle (arc); the range starts at 0
    arc_centre: np.ndarray  # (nE, 3) (zeros for lines)
    arc_radius: np.ndarray  # (nE,)
    arc_sense: np.ndarray  # (nE,) +1 / -1: the circle's normal is +z / -z (zero for lines)
    surf_kind: np.ndarray  # (nS,) PLANE | CYLINDER
    surf_geom: np.ndarray  # (nS, 4, 3): origin, normal (axis), x axis, y axis
    surf_radius: np.ndarray  # (nS,) cylinders only
    pc_edge: np.ndarray  # (nPC,) edge of each (edge, surface) pcurve
    pc_surf: np.ndarray  # (nPC,)
    edge_pcs: list[list[int]]  # per edge: the pcurves it carries
    wires: list[list[int]]  # signed 1-based edge numbers
    face_surf: list[int]
    face_flipped: list[bool]  # wire reversed because the loop's normal is opposite to the surface's (cylinders)
    face_wire0: list[int]
    face_nwires: list[int]
    shells: list[list[int]]  # signed 1-based face numbers (+: outward normal of the solid)
    solid_shell0: list[int]
    solid_nshells: list[int]
    free_faces: list[int]  # embedded faces: children of the compound next to the solids
    free_edges: list[int]  # refinement-only curves
    refs: dict[str, list[int]]  # MapShapes pre-order index of each solid/face/edge/vertex (for XAO)
    groups: list[tuple[str, str, int, list[int]]]  # (name, "face"|"solid", physical tag, element indices)
    timings: dict[str, float] = field(default_factory=dict)
    write_pcurves: bool = True

    @property
    def counts(self) -> dict[str, int]:
        return {"vertices": len(self.vertex_ids), "edges": len(self.edge_ids), "faces": len(self.face_ids),
                "solids": len(self.solid_ids), "wires": len(self.wires), "shells": len(self.shells),
                "surfaces": len(self.surf_kind), "cylinders": int((self.surf_kind == CYLINDER).sum()),
                "arcs": int((self.edge_kind == ARC).sum()),
                "pcurves": len(self.pc_edge) if self.write_pcurves else 0,
                "free_faces": len(self.free_faces), "free_edges": len(self.free_edges)}

    def tags(self) -> dict[str, tuple[int, int]]:
        """Planned id -> (dim, Gmsh tag) after loading the file (tag = index + 1), same shape as ``EmitResult.tags``."""
        out: dict[str, tuple[int, int]] = {}
        for dim, ids in ((0, self.vertex_ids), (1, self.edge_ids), (2, self.face_ids), (3, self.solid_ids)):
            out.update({pid: (dim, i + 1) for i, pid in enumerate(ids)})
        return out

    # reference of the i-th shape of a kind inside the TShapes section (counted from the end of the section)
    def _bases(self) -> dict[str, int]:
        c = self.counts
        total = 1 + c["vertices"] + c["edges"] + c["wires"] + c["faces"] + c["shells"] + c["solids"]
        base_e = total - c["vertices"]
        base_w = base_e - c["edges"]
        base_f = base_w - c["wires"]
        base_s = base_f - c["faces"]
        base_so = base_s - c["shells"]
        return {"total": total, "V": total, "E": base_e, "W": base_w, "F": base_f, "S": base_s, "SO": base_so}


@dataclass
class BrepResult:
    path: Path
    plan: BrepPlan
    bytes_written: int
    timings: dict[str, float]
    occt: object | None = None  # strict mode: the ``occt_check.OcctReport`` of the written file

    @property
    def tags(self) -> dict[str, tuple[int, int]]:
        return self.plan.tags()


# --------------------------------------------------------------------------- stage 1: plan


def plan_brep(topo: Topology, tolerance: float = TOLERANCE, pcurves: bool = True) -> BrepPlan:
    """Number every shape of the emitted volumes in ``TopExp::MapShapes`` order and collect surfaces/pcurves.

    The cyclic collector is paused (as in ``pipeline.build``): the walk allocates millions of acyclic objects next to
    a heap of the same size, and re-scanning it cost a third of the planning time at 20000 features.
    """
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        t0 = time.perf_counter()
        w = _Walk(topo)
        w.visit_all()
        t1 = time.perf_counter()
        plan = w.finish(tolerance)
        plan.write_pcurves = pcurves
        plan.timings = {"walk": t1 - t0, "arrays_and_checks": time.perf_counter() - t1}
        return plan
    finally:
        if was_enabled:
            gc.enable()


class _Walk:
    """Depth-first numbering identical to OCCT's pre-order map: compound, solid, shell, face, wire, edge, vertices."""

    def __init__(self, topo: Topology) -> None:
        self.topo = topo
        self.g = topo.grid_g
        self.zden = lcm(*(z.denominator for z in topo.levels)) if topo.levels else 1
        self.face_no: dict[str, int] = {}
        self.edge_no: dict[str, int] = {}
        self.vert_no: dict[str, int] = {}
        self.vertex_ids: list[str] = []
        self.edge_ids: list[str] = []
        self.face_ids: list[str] = []
        self.solid_ids: list[str] = []
        self.edge_v: list[tuple[int, int]] = []
        self.edge_arc: dict[int, object] = {}  # edge number -> RealisedArc
        self.wires: list[list[int]] = []
        self.face_surf: list[int] = []
        self.face_flipped: list[bool] = []
        self.face_wire0: list[int] = []
        self.face_nwires: list[int] = []
        self.surfaces: list[tuple] = []  # (kind, (origin, normal, x, y), radius)
        self.level_plane: dict[int, int] = {}
        self.pcs: dict[int, int] = {}  # (edge << 32 | surface) -> pcurve number
        self.shells: list[list[int]] = []
        self.solid_shell0: list[int] = []
        self.solid_nshells: list[int] = []
        self.free_faces: list[int] = []
        self.free_edges: list[int] = []
        self.ref = 1  # the compound
        self.solid_ref: list[int] = []
        self.face_ref: list[int] = []
        self.edge_ref: list[int] = []
        self.vert_ref: list[int] = []

    def visit_all(self) -> None:
        topo = self.topo
        emitted = [(vid, v) for vid, v in sorted(topo.volumes.items()) if v.emitted]
        if not emitted:
            raise SgbError("no emitted volume to serialise")
        for vid, vol in emitted:
            self.ref += 1
            self.solid_ref.append(self.ref)
            self.solid_ids.append(vid)
            self.solid_shell0.append(len(self.shells))
            self.solid_nshells.append(len(vol.shells))
            for shell in vol.shells:
                self.ref += 1
                self.shells.append([self._oriented(sign, self._face(fid)) for fid, sign in shell.faces])
        for _, vol in emitted:
            for fid in vol.embedded_faces:
                known = fid in self.face_no
                number = self._face(fid)
                if not known:
                    self.free_faces.append(number)
        for fid in list(self.face_ids):
            for eid in topo.faces[fid].refinement_edges:
                if eid not in self.edge_no:
                    self.free_edges.append(self._edge(eid))

    def _oriented(self, sign: int, face_number: int) -> int:
        """Signed 1-based face reference; a flipped face is used with the opposite orientation."""
        return -sign * (face_number + 1) if self.face_flipped[face_number] else sign * (face_number + 1)

    def _face(self, fid: str) -> int:
        number = self.face_no.get(fid)
        if number is not None:
            return number
        face = self.topo.faces[fid]
        number = self.face_no[fid] = len(self.face_ids)
        self.face_ids.append(fid)
        self.ref += 1
        self.face_ref.append(self.ref)
        surface, flipped = self._surface_of(face)
        self.face_surf.append(surface)
        self.face_flipped.append(flipped)
        self.face_wire0.append(len(self.wires))
        loops = self._wires_of(face)
        self.face_nwires.append(len(loops))
        pcs = self.pcs
        for loop in loops:
            self.ref += 1  # the wire
            if flipped:  # visit edges in the order they are stored in the written (reversed) wire
                loop = [(eid, -sign) for eid, sign in reversed(loop)]
            signed = [sign * (self._edge(eid) + 1) for eid, sign in loop]
            if len({abs(s) for s in signed}) != len(signed):
                raise SgbError(f"face {fid}: an edge occurs twice in one wire (seam/bridge edges are not written)")
            for s in signed:
                key = ((abs(s) - 1) << 32) | surface
                if key not in pcs:
                    pcs[key] = len(pcs)
            self.wires.append(signed)
        return number

    def _wires_of(self, face) -> list[list[tuple[str, int]]]:
        """The wires of a face: its outer loop, then every hole loop split into simple loops.

        Two holes that touch at a vertex are one boundary cycle that visits the vertex twice (a figure eight). OCCT
        treats such a wire as unorientable (``BRepCheck_UnorientableShape``), so each lobe is written as its own wire;
        the lobes share the vertex.
        """
        out = [list(face.loops[0])]
        for loop in face.loops[1:]:
            out.extend(self._simple_lobes(loop))
        return out

    def _simple_lobes(self, loop) -> list[list[tuple[str, int]]]:
        edges = self.topo.edges
        start = lambda item: edges[item[0]].v0 if item[1] > 0 else edges[item[0]].v1
        lobes: list[list[tuple[str, int]]] = []
        path: list[tuple[str, int]] = []
        at: dict[str, int] = {}  # vertex -> index in ``path`` of the item that starts there
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

    def _edge(self, eid: str) -> int:
        number = self.edge_no.get(eid)
        if number is not None:
            return number
        e = self.topo.edges[eid]
        number = self.edge_no[eid] = len(self.edge_ids)
        self.edge_ids.append(eid)
        self.ref += 1
        self.edge_ref.append(self.ref)
        # children are written end vertex first, then start vertex (as Gmsh does); pre-order follows that order
        end, start = self._vertex(e.v1), self._vertex(e.v0)
        self.edge_v.append((start, end))
        if getattr(e, "curve", None) is not None:
            self.edge_arc[number] = e.curve
        return number

    def _vertex(self, vid: str) -> int:
        number = self.vert_no.get(vid)
        if number is None:
            number = self.vert_no[vid] = len(self.vertex_ids)
            self.vertex_ids.append(vid)
            self.ref += 1
            self.vert_ref.append(self.ref)
        return number

    # -- surfaces -----------------------------------------------------------------------

    def _point(self, vid: str) -> tuple[float, float, float]:
        v = self.topo.vertices[vid]
        return (v.x * self.g, v.y * self.g, v.zs / self.zden)

    def _surface_of(self, face) -> tuple[int, bool]:
        """(surface number, wire flipped): a plane per horizontal level and per wall, a cylinder per curved wall."""
        if getattr(face, "curved", False):
            return self._cylinder_of(face)
        if not face.planar:
            raise SgbError(f"face {face.id}: non-planar, non-cylindrical face (needs a B-spline record; not implemented)")
        topo = self.topo
        if face.kind == HORIZONTAL:
            if face.twice_xy_area <= 0:
                raise SgbError(f"horizontal face {face.id} is not counter-clockwise about +z")
            zs = topo.vertices[topo.edges[face.loops[0][0][0]].v0].zs
            plane = self.level_plane.get(zs)
            if plane is None:
                plane = self.level_plane[zs] = len(self.surfaces)
                self.surfaces.append((PLANE, ((0.0, 0.0, zs / self.zden), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), 0.0))
            return plane, False
        pts = [self._point(topo.edges[eid].v0 if sign > 0 else topo.edges[eid].v1) for eid, sign in face.loops[0]]
        self.surfaces.append((PLANE, _plane_through(pts, face.id), 0.0))
        return len(self.surfaces) - 1, False

    def _cylinder_of(self, face) -> tuple[int, bool]:
        """The cylinder (axis +z) carrying a curved wall whose horizontal edges are arcs of one circle.

        The surface's natural normal points away from the axis. The loop's normal (towards the positive side of
        the wall) is the opposite when the top arcs run counter-clockwise; then the wire is reversed and every
        shell uses the face with the opposite orientation (``flipped``).
        """
        topo, g = self.topo, self.g
        if len(face.loops) != 1:
            raise SgbError(f"curved wall {face.id} has holes")
        arcs = []
        for eid, sign in face.loops[0]:
            edge = topo.edges[eid]
            if edge.kind == "vertical":
                continue
            if getattr(edge, "curve", None) is None:
                raise SgbError(f"curved wall {face.id} has a straight horizontal edge (not a cylinder patch)")
            arcs.append((edge, sign))
        first = arcs[0][0].curve
        cx, cy, radius = first.centre[0] * g, first.centre[1] * g, first.radius * g
        spread = max(max(abs(e.curve.centre[0] * g - cx), abs(e.curve.centre[1] * g - cy), abs(e.curve.radius * g - radius))
                     for e, _ in arcs)
        if spread > 1e-8:
            raise SgbError(
                f"curved wall {face.id}: its {len(arcs)} horizontal arcs lie on different circles (centres differ by "
                f"{spread * 1000:.3g} nm), because a vertex of another prism or curve splits the arc on one of its levels; "
                "a single cylinder cannot carry the wall and a B-spline record is not implemented (the gmsh-API emitter "
                "can still write this model)")
        angles = []
        for edge, _ in arcs:
            for vid in (edge.v0, edge.v1):
                x, y, _ = self._point(vid)
                angles.append(math.atan2(y - cy, x - cx))
        reference = angles[0]
        offsets = [(a - reference + math.pi) % (2 * math.pi) - math.pi for a in angles]
        if max(offsets) - min(offsets) > 2 * math.pi - 1e-6:
            raise SgbError(f"curved wall {face.id} wraps the whole cylinder")
        a_min = reference + min(offsets)  # u = angle from the radial direction at the clockwise-most vertex
        top = max(topo.vertices[edge.v0].zs for edge, _ in arcs)
        directions = {edge.curve.ccw == (sign > 0) for edge, sign in arcs if topo.vertices[edge.v0].zs == top}
        if len(directions) != 1:
            raise SgbError(f"curved wall {face.id}: top arcs run in different senses")
        flipped = directions.pop()  # top arcs counter-clockwise: the loop's normal points to the axis
        x_axis = (math.cos(a_min), math.sin(a_min), 0.0)
        y_axis = (-math.sin(a_min), math.cos(a_min), 0.0)
        self.surfaces.append((CYLINDER, ((cx, cy, 0.0), (0.0, 0.0, 1.0), x_axis, y_axis), radius))
        return len(self.surfaces) - 1, flipped

    # -- arrays -------------------------------------------------------------------------

    def finish(self, tolerance: float) -> BrepPlan:
        topo = self.topo
        verts = np.array([self._point(v) for v in self.vertex_ids], dtype=float).reshape(-1, 3)
        edge_v = np.array(self.edge_v, dtype=np.int64).reshape(-1, 2)
        n_edges = len(self.edge_ids)
        delta = verts[edge_v[:, 1]] - verts[edge_v[:, 0]]
        span = np.linalg.norm(delta, axis=1)
        if span.size and span.min() <= 10 * tolerance:
            raise SgbError(f"edge shorter than 10 x tolerance ({span.min():.3g} um)")
        direction = delta / span[:, None]
        kind = np.zeros(n_edges, dtype=np.int8)
        centre, radius, sense = np.zeros((n_edges, 3)), np.zeros(n_edges), np.zeros(n_edges)
        if self.edge_arc:
            self._arcs_to_arrays(tolerance, verts, edge_v, kind, direction, span, centre, radius, sense)
        surf_kind = np.array([s[0] for s in self.surfaces], dtype=np.int8)
        surf_geom = np.array([s[1] for s in self.surfaces], dtype=float).reshape(-1, 4, 3)
        surf_radius = np.array([s[2] for s in self.surfaces], dtype=float)
        pc_keys = np.fromiter(self.pcs, dtype=np.int64, count=len(self.pcs))
        pc_edge, pc_surf = pc_keys >> 32, pc_keys & 0xFFFFFFFF
        _check_edges_lie_on_surfaces(verts, edge_v, kind, direction, centre, surf_kind, surf_geom, surf_radius,
                                     pc_edge, pc_surf, tolerance)
        edge_pcs: list[list[int]] = [[] for _ in self.edge_ids]
        for pc, e in enumerate(pc_edge.tolist()):
            edge_pcs[e].append(pc)
        refs = {"solid": self.solid_ref, "face": self.face_ref, "edge": self.edge_ref, "vertex": self.vert_ref}
        return BrepPlan(
            tolerance, self.vertex_ids, self.edge_ids, self.face_ids, self.solid_ids, verts, edge_v, kind, direction,
            span, centre, radius, sense, surf_kind, surf_geom, surf_radius, pc_edge, pc_surf, edge_pcs, self.wires,
            self.face_surf, self.face_flipped, self.face_wire0, self.face_nwires, self.shells, self.solid_shell0,
            self.solid_nshells, self.free_faces, self.free_edges, refs, self._groups(topo),
        )

    def _arcs_to_arrays(self, tolerance, verts, edge_v, kind, direction, span, centre, radius, sense) -> None:
        """Circle records: normal +z (counter-clockwise travel) or -z, x axis towards the start point."""
        g = self.g
        for number, arc in self.edge_arc.items():
            start, end = verts[edge_v[number, 0]], verts[edge_v[number, 1]]
            c = np.array([arc.centre[0] * g, arc.centre[1] * g, start[2]])
            r = arc.radius * g
            rel0, rel1 = start - c, end - c
            if max(abs(np.hypot(rel0[0], rel0[1]) - r), abs(np.hypot(rel1[0], rel1[1]) - r)) > tolerance / 10 or abs(end[2] - start[2]) > tolerance / 10:
                raise SgbError(f"arc edge {self.edge_ids[number]}: end points are not on its circle within {tolerance / 10:g} um")
            s = 1.0 if arc.ccw else -1.0
            x_dir = rel0 / np.hypot(rel0[0], rel0[1])
            angle = math.atan2(s * (rel0[0] * rel1[1] - rel0[1] * rel1[0]), rel0[0] * rel1[0] + rel0[1] * rel1[1])
            if angle <= 0.0:
                angle += 2 * math.pi
            kind[number], direction[number], span[number] = ARC, x_dir, angle  # the circle is parametrised by its angle (rad)
            centre[number], radius[number], sense[number] = c, r, s

    def _groups(self, topo: Topology) -> list[tuple[str, str, int, list[int]]]:
        """Physical groups as element index lists, tagged like ``emit`` does (per dimension, from 1, skipping empty)."""
        next_tag = {2: 1, 3: 1}
        out = []
        for name, group in topo.groups.items():
            dim = group["dim"]
            lookup = self.face_no if dim == 2 else {vid: i for i, vid in enumerate(self.solid_ids)}
            elements = [lookup[i] for i in group["ids"] if i in lookup]
            if not elements:
                continue
            out.append((name, "face" if dim == 2 else "solid", next_tag[dim], elements))
            next_tag[dim] += 1
        return out


def _plane_through(points: list[tuple[float, float, float]], face_id: str):
    """Plane of a planar loop (Newell normal about the loop, right-hand rule); Y = N x X, so (X, Y, N) is right-handed."""
    p0 = points[0]
    rel = [(x - p0[0], y - p0[1], z - p0[2]) for x, y, z in points]
    nx = ny = nz = 0.0
    for (ax, ay, az), (bx, by, bz) in zip(rel, rel[1:] + rel[:1]):
        nx += (ay - by) * (az + bz)
        ny += (az - bz) * (ax + bx)
        nz += (ax - bx) * (ay + by)
    norm = (nx * nx + ny * ny + nz * nz) ** 0.5
    if norm == 0.0:
        raise SgbError(f"face {face_id}: degenerate loop")
    nx, ny, nz = nx / norm, ny / norm, nz / norm
    if abs(nz) < 1e-9:  # vertical plane: y axis is +z, x axis = y x N, so that x x y = N
        scale = (nx * nx + ny * ny) ** 0.5
        nx, ny = nx / scale, ny / scale
        return (p0, (nx, ny, 0.0), (-ny, nx, 0.0), (0.0, 0.0, 1.0))
    ax, ay, az = 1.0 - nx * nx, -nx * ny, -nx * nz  # tilted plane: x axis is +x projected into the plane
    scale = (ax * ax + ay * ay + az * az) ** 0.5
    x_axis = (ax / scale, ay / scale, az / scale)
    y_axis = (ny * x_axis[2] - nz * x_axis[1], nz * x_axis[0] - nx * x_axis[2], nx * x_axis[1] - ny * x_axis[0])
    return (p0, (nx, ny, nz), x_axis, y_axis)


def _check_edges_lie_on_surfaces(verts, edge_v, kind, direction, centre, surf_kind, surf_geom, surf_radius,
                                 pc_edge, pc_surf, tolerance) -> None:
    """Every edge must lie on every surface that carries a pcurve of it, within a tenth of the tolerance."""
    if not len(pc_edge):
        return
    limit = tolerance / 10
    origin, normal = surf_geom[pc_surf, 0], surf_geom[pc_surf, 1]
    ends = [verts[edge_v[pc_edge, 0]] - origin, verts[edge_v[pc_edge, 1]] - origin]
    is_plane = surf_kind[pc_surf] == PLANE
    worst = 0.0
    for rel in ends:
        plane_dist = np.abs(np.einsum("ij,ij->i", rel, normal))
        axis_dist = np.hypot(np.einsum("ij,ij->i", rel, surf_geom[pc_surf, 2]), np.einsum("ij,ij->i", rel, surf_geom[pc_surf, 3]))
        worst = max(worst, np.where(is_plane, plane_dist, np.abs(axis_dist - surf_radius[pc_surf])).max())
    arcs = kind[pc_edge] == ARC
    if arcs.any():  # a circle lies on a plane when its centre does (axis +z), on a cylinder when its centre is on the axis
        rel = centre[pc_edge] - origin
        plane_dist = np.abs(np.einsum("ij,ij->i", rel, normal))
        axis_dist = np.hypot(np.einsum("ij,ij->i", rel, surf_geom[pc_surf, 2]), np.einsum("ij,ij->i", rel, surf_geom[pc_surf, 3]))
        worst = max(worst, np.where(is_plane, plane_dist, axis_dist)[arcs].max())
    if worst > limit:
        raise SgbError(f"an edge leaves its surface by {worst:.3g} um (limit {limit:g})")


# --------------------------------------------------------------------------- stage 2: text sections


_PLAN: BrepPlan | None = None  # inherited by forked workers; set before a pool is created


def _rows(template: str, array: np.ndarray) -> list[str]:
    """One formatted string per row of ``array`` (``-0`` is normalised to ``0``)."""
    return list(map(template.__mod__, map(tuple, (array + 0.0).tolist())))


def _tol(plan: BrepPlan) -> str:
    return "%.9g" % plan.tolerance


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", a, b)


# -- pcurves: one record per (edge kind, surface kind) combination ---------------------


def _pcurve_line_on_plane(plan: BrepPlan, e: np.ndarray, p: np.ndarray) -> list[str]:
    d0 = plan.verts[plan.edge_v[e, 0]] - plan.surf_geom[p, 0]
    direction, x_axis, y_axis = plan.edge_dir[e], plan.surf_geom[p, 2], plan.surf_geom[p, 3]
    rows = np.stack([_dot(d0, x_axis), _dot(d0, y_axis), _dot(direction, x_axis), _dot(direction, y_axis)], axis=1)
    return _rows("1 %.12g %.12g %.15g %.15g\n", rows)


def _pcurve_arc_on_plane(plan: BrepPlan, e: np.ndarray, p: np.ndarray) -> list[str]:
    """Circle2d: centre, x axis, y axis (indirect when the circle runs clockwise), radius."""
    c0 = plan.arc_centre[e] - plan.surf_geom[p, 0]
    x3 = plan.edge_dir[e]
    y3 = np.stack([-plan.arc_sense[e] * x3[:, 1], plan.arc_sense[e] * x3[:, 0], np.zeros(len(e))], axis=1)
    x_axis, y_axis = plan.surf_geom[p, 2], plan.surf_geom[p, 3]
    rows = np.stack([_dot(c0, x_axis), _dot(c0, y_axis), _dot(x3, x_axis), _dot(x3, y_axis),
                     _dot(y3, x_axis), _dot(y3, y_axis), plan.arc_radius[e]], axis=1)
    return _rows("2 %.15g %.15g %.15g %.15g %.15g %.15g %.15g\n", rows)


def _angle_on_cylinder(plan: BrepPlan, e: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d0 = plan.verts[plan.edge_v[e, 0]] - plan.surf_geom[p, 0]
    u = np.arctan2(_dot(d0, plan.surf_geom[p, 3]), _dot(d0, plan.surf_geom[p, 2]))
    return np.where(u < -1e-12, u + 2 * np.pi, u), _dot(d0, plan.surf_geom[p, 1])


def _pcurve_line_on_cylinder(plan: BrepPlan, e: np.ndarray, p: np.ndarray) -> list[str]:
    """A vertical edge: a straight line along v (the height)."""
    u, v = _angle_on_cylinder(plan, e, p)
    rows = np.stack([u, v, np.zeros(len(e)), _dot(plan.edge_dir[e], plan.surf_geom[p, 1])], axis=1)
    return _rows("1 %.15g %.15g %.15g %.15g\n", rows)


def _pcurve_arc_on_cylinder(plan: BrepPlan, e: np.ndarray, p: np.ndarray) -> list[str]:
    """An arc about the axis: a straight line along u (the angle), which increases for counter-clockwise travel."""
    u, v = _angle_on_cylinder(plan, e, p)
    rows = np.stack([u, v, plan.arc_sense[e], np.zeros(len(e))], axis=1)
    return _rows("1 %.15g %.15g %.15g %.15g\n", rows)


PCURVE_RECORDS = {  # (edge kind, surface kind) -> record writer; the extension point for new curve/surface pairs
    (LINE, PLANE): _pcurve_line_on_plane, (ARC, PLANE): _pcurve_arc_on_plane,
    (LINE, CYLINDER): _pcurve_line_on_cylinder, (ARC, CYLINDER): _pcurve_arc_on_cylinder,
}


def _interleave(codes: np.ndarray, writers: dict, plan: BrepPlan, *index_arrays: np.ndarray) -> str:
    """Records of mixed kinds in item order: ``codes[i]`` selects the writer of item ``i``."""
    out = np.empty(len(codes), dtype=object)
    for code in np.unique(codes).tolist():
        picked = np.nonzero(codes == code)[0]
        out[picked] = writers[code](plan, *(a[picked] for a in index_arrays))
    return "".join(out.tolist())


def _text_curve2ds(plan: BrepPlan, lo: int, hi: int) -> str:
    e, p = plan.pc_edge[lo:hi], plan.pc_surf[lo:hi]
    codes = plan.edge_kind[e].astype(int) * 2 + plan.surf_kind[p]
    writers = {ek * 2 + sk: fn for (ek, sk), fn in PCURVE_RECORDS.items()}
    return _interleave(codes, writers, plan, e, p)


# -- 3D curves and surfaces --------------------------------------------------------------


def _curve_line(plan: BrepPlan, e: np.ndarray) -> list[str]:
    rows = np.concatenate([plan.verts[plan.edge_v[e, 0]], plan.edge_dir[e]], axis=1)
    return _rows("1 %.12g %.12g %.12g %.15g %.15g %.15g\n", rows)


def _curve_circle(plan: BrepPlan, e: np.ndarray) -> list[str]:
    x3 = plan.edge_dir[e]
    s = plan.arc_sense[e]
    n3 = np.stack([np.zeros(len(e)), np.zeros(len(e)), s], axis=1)
    y3 = np.stack([-s * x3[:, 1], s * x3[:, 0], np.zeros(len(e))], axis=1)  # N x X
    rows = np.concatenate([plan.arc_centre[e], n3, x3, y3, plan.arc_radius[e][:, None]], axis=1)
    return _rows("2 %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g\n", rows)


CURVE_RECORDS = {LINE: _curve_line, ARC: _curve_circle}  # extension point: a new curve kind adds an entry


def _text_curves(plan: BrepPlan, lo: int, hi: int) -> str:
    e = np.arange(lo, hi)
    return _interleave(plan.edge_kind[lo:hi], CURVE_RECORDS, plan, e)


def _surface_plane(plan: BrepPlan, s: np.ndarray) -> list[str]:
    return _rows("1 " + "%.12g %.12g %.12g " + "%.15g " * 8 + "%.15g\n", plan.surf_geom[s].reshape(-1, 12))


def _surface_cylinder(plan: BrepPlan, s: np.ndarray) -> list[str]:
    rows = np.concatenate([plan.surf_geom[s].reshape(-1, 12), plan.surf_radius[s][:, None]], axis=1)
    return _rows("2 " + "%.15g " * 12 + "%.15g\n", rows)


SURFACE_RECORDS = {PLANE: _surface_plane, CYLINDER: _surface_cylinder}


def _text_surfaces(plan: BrepPlan, lo: int, hi: int) -> str:
    return _interleave(plan.surf_kind[lo:hi], SURFACE_RECORDS, plan, np.arange(lo, hi))


# -- shapes ------------------------------------------------------------------------------


def _text_vertices(plan: BrepPlan, lo: int, hi: int) -> str:
    return "".join(_rows("Ve\n" + _tol(plan) + "\n%.12g %.12g %.12g\n0 0\n\n" + FLAGS["Ve"] + "\n*\n", plan.verts[lo:hi]))


def _text_edges(plan: BrepPlan, lo: int, hi: int) -> str:
    b = plan._bases()
    ref_v, tol = b["V"], _tol(plan)
    spans = ["%.15g" % x for x in plan.edge_span[lo:hi].tolist()]
    starts, ends = plan.edge_v[lo:hi, 0].tolist(), plan.edge_v[lo:hi, 1].tolist()
    pc_surf, with_pcs = plan.pc_surf, plan.write_pcurves
    head, tail = "Ed\n " + tol + " 1 1 0\n1  %d 0 0 %s\n", "0\n\n" + FLAGS["Ed"] + "\n-%d 0 +%d 0 *\n"
    out = []
    for k, i in enumerate(range(lo, hi)):
        reps = "".join("2  %d %d 0 0 %s\n" % (pc + 1, pc_surf[pc] + 1, spans[k]) for pc in plan.edge_pcs[i]) if with_pcs else ""
        out.append(head % (i + 1, spans[k]) + reps + tail % (ref_v - ends[k], ref_v - starts[k]))
    return "".join(out)


def _wire_children(base: int, items: list[int]) -> str:
    return "".join("%s%d 0 " % ("+" if s > 0 else "-", base - abs(s) + 1) for s in items)


def _text_wires(plan: BrepPlan, lo: int, hi: int) -> str:
    base_e = plan._bases()["E"]
    head = "Wi\n\n" + FLAGS["Wi"] + "\n"
    return "".join(head + _wire_children(base_e, plan.wires[i]) + "*\n" for i in range(lo, hi))


def _text_faces(plan: BrepPlan, lo: int, hi: int) -> str:
    base_w = plan._bases()["W"]
    tol, flags = _tol(plan), FLAGS["Fa"]
    out = []
    for i in range(lo, hi):
        w0 = plan.face_wire0[i]
        wires = "".join("+%d 0 " % (base_w - w) for w in range(w0, w0 + plan.face_nwires[i]))
        out.append("Fa\n0  %s %d 0\n\n%s\n%s*\n" % (tol, plan.face_surf[i] + 1, flags, wires))
    return "".join(out)


def _text_shells(plan: BrepPlan, lo: int, hi: int) -> str:
    base_f = plan._bases()["F"]
    head = "Sh\n\n" + FLAGS["Sh"] + "\n"
    return "".join(head + _wire_children(base_f, plan.shells[i]) + "*\n" for i in range(lo, hi))


def _text_solids(plan: BrepPlan, lo: int, hi: int) -> str:
    base_s = plan._bases()["S"]
    head = "So\n\n" + FLAGS["So"] + "\n"
    out = []
    for i in range(lo, hi):
        s0 = plan.solid_shell0[i]
        out.append(head + "".join("+%d 0 " % (base_s - s) for s in range(s0, s0 + plan.solid_nshells[i])) + "*\n")
    return "".join(out)


def _text_xao_vertices(plan: BrepPlan, lo: int, hi: int) -> str:
    return "".join('        <vertex index="%d" name="" reference="%d"/>\n' % (i, plan.refs["vertex"][i]) for i in range(lo, hi))


def _text_xao_edges(plan: BrepPlan, lo: int, hi: int) -> str:
    return "".join('        <edge index="%d" name="" reference="%d"/>\n' % (i, plan.refs["edge"][i]) for i in range(lo, hi))


def _text_xao_faces(plan: BrepPlan, lo: int, hi: int) -> str:
    return "".join('        <face index="%d" name="" reference="%d"/>\n' % (i, plan.refs["face"][i]) for i in range(lo, hi))


SECTION_TEXT = {
    "curve2ds": _text_curve2ds, "curves": _text_curves, "surfaces": _text_surfaces, "vertices": _text_vertices,
    "edges": _text_edges, "wires": _text_wires, "faces": _text_faces, "shells": _text_shells,
    "solids": _text_solids, "xao_vertices": _text_xao_vertices, "xao_edges": _text_xao_edges,
    "xao_faces": _text_xao_faces,
}


def _compound_text(plan: BrepPlan) -> str:
    b = plan._bases()
    children = ["+%d 0 " % (b["SO"] - i) for i in range(len(plan.solid_ids))]
    children += ["+%d 0 " % (b["F"] - i) for i in plan.free_faces]
    children += ["+%d 0 " % (b["E"] - i) for i in plan.free_edges]
    return "Co\n\n" + FLAGS["Co"] + "\n" + "".join(children) + "*\n\n+1 0 \n"


def _pieces(plan: BrepPlan, prefix: str):
    """The BREP as an ordered list of literal strings and (section, lo, hi) chunk tasks."""
    c = plan.counts
    total = plan._bases()["total"]
    yield prefix + "CASCADE Topology V1, (c) Matra-Datavision\nLocations 0\nCurve2ds %d\n" % c["pcurves"]
    yield from _tasks("curve2ds", c["pcurves"])
    yield "Curves %d\n" % c["edges"]
    yield from _tasks("curves", c["edges"])
    yield "Polygon3D 0\nPolygonOnTriangulations 0\nSurfaces %d\n" % c["surfaces"]
    yield from _tasks("surfaces", c["surfaces"])
    yield "Triangulations 0\n\nTShapes %d\n" % total
    for section in ("vertices", "edges", "wires", "faces", "shells", "solids"):
        yield from _tasks(section, c[section])
    yield _compound_text(plan)


def _tasks(section: str, count: int, chunk: int | None = None):
    chunk = chunk or CHUNK_ITEMS
    for lo in range(0, count, chunk):
        yield (section, lo, min(count, lo + chunk))


def _run_task(task) -> bytes:
    section, lo, hi = task
    return SECTION_TEXT[section](_PLAN, lo, hi).encode("ascii")


def _write_pieces(out, plan: BrepPlan, pieces: list, workers: int) -> int:
    """Write literals and chunk tasks in order; tasks run in a forked pool when ``workers > 1``."""
    global _PLAN
    _PLAN = plan
    tasks = [p for p in pieces if isinstance(p, tuple)]
    written = 0
    if workers <= 1 or len(tasks) < 2:
        results = map(_run_task, tasks)
        pool = None
    else:
        pool = multiprocessing.get_context("fork").Pool(workers)
        results = pool.imap(_run_task, tasks, chunksize=1)
    try:
        for piece in pieces:
            data = next(results) if isinstance(piece, tuple) else piece.encode("ascii")
            out.write(data)
            written += len(data)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        _PLAN = None
    return written


# --------------------------------------------------------------------------- public writers


def write_brep(topo: Topology, path: str | Path, *, workers: int = 1, tolerance: float = TOLERANCE,
               pcurves: bool = True, plan: BrepPlan | None = None, sidecar: bool | str | Path = True,
               lineage: bool = True, strict: bool | dict = False) -> BrepResult:
    """Write ``path`` as an ASCII BREP: a COMPOUND of one SOLID per emitted volume, shared sub-shapes written once.

    ``sidecar`` (default on) also writes ``<stem>.sidecar.json`` next to it (or at the given path), see
    ``build_sidecar``; ``lineage=False`` leaves the lineage pointers out of it. ``strict=True`` reads the written
    file back with OCP and runs the official OCCT checks (``occt_check``); any finding raises. It needs OCP (the dev
    extra) and says so before writing anything when OCP is missing. A dict passes options to
    ``occt_check.check_file`` (e.g. ``strict={"self_interference": False}`` for large models).
    """
    _require_strict(strict)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    plan = plan or plan_brep(topo, tolerance, pcurves)
    timings["plan"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    with path.open("wb") as out:
        n = _write_pieces(out, plan, list(_pieces(plan, "DBRep_DrawableShape\n\n")), workers)
    timings["render"] = time.perf_counter() - t0
    _finish(topo, plan, path, sidecar, lineage, timings)
    return BrepResult(path, plan, n, timings, _strict_check(strict, topo, plan, path))


def write_xao(topo: Topology, path: str | Path, *, workers: int = 1, name: str | None = None,
              tolerance: float = TOLERANCE, pcurves: bool = True, plan: BrepPlan | None = None,
              sidecar: bool | str | Path = True, lineage: bool = True, strict: bool | dict = False) -> BrepResult:
    """Write ``path`` as XAO 1.0: the BREP in a CDATA block, sub-shape references, and the physical groups.

    The ``reference`` of each sub-shape is its ``TopExp::MapShapes`` index (Gmsh resolves groups through it).
    Groups are the planned primary groups: ``face`` or ``solid`` elements by 0-based index in tag order.
    ``strict=True`` runs the official OCCT checks on the BREP inside the written XAO (see ``write_brep``).
    """
    _require_strict(strict)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = name or path.stem
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    plan = plan or plan_brep(topo, tolerance, pcurves)
    timings["plan"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    c = plan.counts
    pieces: list = ['<?xml version="1.0" encoding="UTF-8"?>\n<XAO version="1.0" author="sgbproto">\n'
                    '  <geometry name=%s>\n    <shape format="BREP"><![CDATA[\n' % quoteattr(name)]
    pieces += list(_pieces(plan, ""))
    pieces.append("]]></shape>\n    <topology>\n")
    for kind, section, count in (("vertices", "xao_vertices", c["vertices"]), ("edges", "xao_edges", c["edges"]),
                                 ("faces", "xao_faces", c["faces"])):
        pieces.append('      <%s count="%d">\n' % (kind, count))
        pieces += list(_tasks(section, count))
        pieces.append("      </%s>\n" % kind)
    solids = "".join('        <solid index="%d" name="" reference="%d"/>\n' % (i, r) for i, r in enumerate(plan.refs["solid"]))
    pieces.append('      <solids count="%d">\n%s      </solids>\n    </topology>\n  </geometry>\n' % (c["solids"], solids))
    pieces.append(_groups_xml(plan))
    pieces.append("</XAO>\n")
    with path.open("wb") as out:
        n = _write_pieces(out, plan, pieces, workers)
    timings["render"] = time.perf_counter() - t0
    _finish(topo, plan, path, sidecar, lineage, timings)
    return BrepResult(path, plan, n, timings, _strict_check(strict, topo, plan, path))


def _require_strict(strict: bool | dict) -> None:
    if strict:
        from sgbproto.occt_check import require_ocp

        require_ocp()


def _strict_check(strict: bool | dict, topo: Topology, plan: BrepPlan, path: Path):
    if not strict:
        return None
    from sgbproto.occt_check import check_file

    options = strict if isinstance(strict, dict) else {}
    # self-interference is the one expensive Boolean pre-check (about 25-40 s at 1000 features): strict mode runs the
    # cheap checks by default; ask for it with strict={"self_interference": True} (the fixture catalogue always does)
    return check_file(path, topo, plan, **{"tolerance": plan.tolerance, "self_interference": False, **options, "strict": True})


def _finish(topo: Topology, plan: BrepPlan, path: Path, sidecar: bool | str | Path, lineage: bool,
            timings: dict[str, float]) -> None:
    t0 = time.perf_counter()
    if sidecar:
        target = path.with_name(path.stem + ".sidecar.json") if sidecar is True else Path(sidecar)
        target.write_text(json.dumps(build_sidecar(topo, plan, lineage), indent=1, sort_keys=True) + "\n")
        timings["sidecar"] = time.perf_counter() - t0
    timings["total"] = timings["plan"] + timings["render"] + timings.get("sidecar", 0.0)


# --------------------------------------------------------------------------- production sidecar


def build_sidecar(topo: Topology, plan: BrepPlan, lineage: bool = True) -> dict:
    """What the mesher needs besides the geometry file, with planned ids only (no coordinates, no native handles).

    * ``ids_by_tag[dim]``: planned ids in native-tag order, so ``tag = 1 + position`` (dim 0 vertices, 1 edges,
      2 faces, 3 volumes); the numbering is the ``TopExp::MapShapes`` pre-order of the written compound.
    * ``physical_groups``: the primary groups (they partition each dimension) with their Gmsh physical tag; the
      same names and tags the XAO carries. ``unions``: lists of primary group names, never overlapping groups.
    * ``embedded_faces`` / ``embedded``: faces inside a volume ((face, host volume) pairs; neither XAO nor BREP can
      carry mesh embedding, the mesher re-applies them). ``refinement_edges``: embedded size curves with their host
      face; ``written_as_free_edge`` says the file holds the curve as a free edge of the compound.
    * ``lineage``: edge id -> source carrier ids, face id -> source entity ids (pointers, no geometry).
    Everything is derived from the plan in a fixed order, so it is independent of the number of workers.
    """
    by_tag = {"0": plan.vertex_ids, "1": plan.edge_ids, "2": plan.face_ids, "3": plan.solid_ids}
    tag_of = {"2": {fid: i + 1 for i, fid in enumerate(plan.face_ids)}, "3": {vid: i + 1 for i, vid in enumerate(plan.solid_ids)}}
    groups = {}
    for name, dimension, physical_tag, elements in plan.groups:
        dim = 3 if dimension == "solid" else 2
        groups[name] = {"dim": dim, "physical_tag": physical_tag, "ids": list(topo.groups[name]["ids"]),
                        "native_tags": [i + 1 for i in elements]}
    free_edges = {plan.edge_ids[i] for i in plan.free_edges}
    refinement = [{"edge": eid, "host_face": fid, "written_as_free_edge": eid in free_edges}
                  for fid in plan.face_ids for eid in topo.faces[fid].refinement_edges]
    embedded = {vid: list(topo.volumes[vid].embedded_faces) for vid in plan.solid_ids if topo.volumes[vid].embedded_faces}
    out = {
        "format": "sgbproto-sidecar-2",
        "writer": "sgbproto.brep_writer",
        "route": topo.route,
        "grid_um": topo.grid_g,
        "tolerance_um": plan.tolerance,
        "binding": "planned id -> native tag: tag = 1 + position in ids_by_tag[dim] (TopExp::MapShapes pre-order); "
                   "never by geometry",
        "ids_by_tag": by_tag,
        "counts": plan.counts,
        "physical_groups": groups,
        "unions": topo.unions,
        "surface_epr_rows": topo.surface_epr_rows,
        "union_note": "unions are lists of primary physical-group names (e.g. all PEC faces of a Net); "
                      "they are never overlapping Gmsh groups",
        "embedded": embedded,
        "embedded_faces": [{"face": fid, "host_volume": vid} for vid, fids in embedded.items() for fid in fids],
        "refinement_edges": refinement,
        "embedded_note": "XAO/BREP cannot carry mesh embedding: the mesher must re-apply (face, host volume) and "
                         "(edge, host face) pairs",
        "curved_wall_faces": [fid for fid in plan.face_ids if getattr(topo.faces[fid], "curved", False)],
        "face_surface_kind": {"plane": int((plan.surf_kind == PLANE).sum()), "cylinder": int((plan.surf_kind == CYLINDER).sum())},
    }
    if lineage:
        out["lineage"] = {"edges": {eid: list(topo.lineage.get(eid, ())) for eid in plan.edge_ids},
                          "faces": {fid: list(topo.faces[fid].sources) for fid in plan.face_ids},
                          "membership": {fid: dict(topo.faces[fid].membership) for fid in plan.face_ids},
                          "note": "source carrier ids (edges), source entity ids (faces); membership: face -> {entity: "
                                  "fraction of the face area (walls: of the bottom boundary length) covered by that "
                                  "entity; non-exclusive, may sum to more than 1}"}
    return out


def read_sidecar_tags(path: str | Path) -> dict[str, tuple[int, int]]:
    """planned id -> (dim, tag) from a sidecar written by this module."""
    data = json.loads(Path(path).read_text())
    return {pid: (int(dim), i + 1) for dim, ids in data["ids_by_tag"].items() for i, pid in enumerate(ids)}


def _groups_xml(plan: BrepPlan) -> str:
    lines = ['  <groups count="%d">\n' % len(plan.groups)]
    for gname, dimension, tag, elements in plan.groups:
        lines.append('    <group name=%s dimension="%s" tag="%d" count="%d">\n' % (quoteattr(gname), dimension, tag, len(elements)))
        lines.append("".join('      <element index="%d"/>\n' % i for i in elements))
        lines.append("    </group>\n")
    lines.append('  </groups>\n  <fields count="0"/>\n')
    return "".join(lines)


# --------------------------------------------------------------------------- verification of a loaded file


def planned_counts(topo: Topology) -> dict[int, int]:
    """Entities per dimension that ``emit`` would create (independent of the writer's numbering)."""
    faces: set[str] = set()
    for v in topo.volumes.values():
        if v.emitted:
            for shell in v.shells:
                faces.update(fid for fid, _ in shell.faces)
            faces.update(v.embedded_faces)
    edges = {eid for fid in faces for lp in topo.faces[fid].loops for eid, _ in lp}
    edges |= {eid for fid in faces for eid in topo.faces[fid].refinement_edges}
    vertices = {x for eid in edges for x in (topo.edges[eid].v0, topo.edges[eid].v1)}
    return {0: len(vertices), 1: len(edges), 2: len(faces), 3: sum(v.emitted for v in topo.volumes.values())}


def verify_loaded(topo: Topology, plan: BrepPlan, mesh_size: float | None = None, mesh_size_min: float = 0.0,
                  groups: bool = True, supports: dict | None = None, refine_families: tuple[str, ...] = (),
                  refine_size: float = 1.0, refine_distance: tuple[float, float] = (2.0, 20.0)) -> dict:
    """Check the model currently loaded in Gmsh against the plan (the file must have been merged/opened).

    (a) entity counts per dimension equal the planned counts; (b) the tag predicted for every planned id
    sits at the planned place (vertex coordinates) and every face / volume is bounded by exactly the planned
    edges / faces (``getBoundary``); (c) physical groups equal the planned groups; (e) volumes (mass, i.e.
    orientation) are positive and equal the planned signed volumes (arc bulge terms included); (d) with ``mesh_size``, a coarse 3D mesh is conformal
    (the prototype's own ``check_mesh_conformity`` after re-applying the planned embeddings); (f) every face's
    area equals ``planned_face_area_um2`` and so does every group's total; with ``supports`` (inset bands) the
    prototype's ``band_mesh_report`` shows that no triangle or tet straddles a band boundary.
    """
    import gmsh

    tags = plan.tags()
    report: dict = {"problems": []}
    problems = report["problems"]
    clock, stamp = {}, [time.perf_counter()]

    def lap(name: str) -> None:
        now = time.perf_counter()
        clock[name] = now - stamp[0]
        stamp[0] = now

    report["timings_s"] = clock
    native = {d: len(gmsh.model.getEntities(d)) for d in range(4)}
    planned = planned_counts(topo)
    report["counts"] = {"native": native, "planned": planned}
    if native != planned:
        problems.append(f"entity counts {native} != planned {planned}")
    # vertex placement proves the numbering; every other check uses it
    off = 0.0
    for i, vid in enumerate(plan.vertex_ids):
        x, y, z, *_ = gmsh.model.getBoundingBox(0, i + 1)
        off = max(off, abs(x - plan.verts[i, 0]), abs(y - plan.verts[i, 1]), abs(z - plan.verts[i, 2]))
    report["max_vertex_offset_um"] = off
    lap("counts_and_vertex_placement")
    if off > 1e-9:
        problems.append(f"vertex tags are not where the plan put them (max offset {off:.3g} um)")
    bad_edges = 0
    for i, eid in enumerate(plan.edge_ids):
        e = topo.edges[eid]
        got = {t for _, t in gmsh.model.getBoundary([(1, i + 1)], combined=False, oriented=False)}
        if got != {tags[e.v0][1], tags[e.v1][1]}:
            bad_edges += 1
    lap("edge_boundaries")
    bad_faces = 0
    for i, fid in enumerate(plan.face_ids):
        want = {tags[eid][1] for lp in topo.faces[fid].loops for eid, _ in lp}
        got = {t for _, t in gmsh.model.getBoundary([(2, i + 1)], combined=False, oriented=False)}
        if got != want:
            bad_faces += 1
    lap("face_boundaries")
    bad_solids = 0
    for i, vid in enumerate(plan.solid_ids):
        want = {tags[fid][1] for sh in topo.volumes[vid].shells for fid, _ in sh.faces}
        got = {t for _, t in gmsh.model.getBoundary([(3, i + 1)], combined=False, oriented=False)}
        if got != want:
            bad_solids += 1
    lap("solid_boundaries")
    report["boundaries_mismatching"] = {"edges": bad_edges, "faces": bad_faces, "solids": bad_solids}
    if bad_edges or bad_faces or bad_solids:
        problems.append(f"boundary mismatch: {report['boundaries_mismatching']}")
    # (c) physical groups
    # ``getEntitiesForPhysicalGroup`` scans every entity of the dimension per group (groups x entities: quadratic);
    # asking each entity for its groups touches every entity once
    members: dict[tuple[int, int], list[int]] = {}
    for d in (2, 3):
        for _, entity in gmsh.model.getEntities(d):
            for group_tag in gmsh.model.getPhysicalGroupsForEntity(d, entity):
                members.setdefault((d, int(group_tag)), []).append(int(entity))
    got_groups = {gmsh.model.getPhysicalName(d, t): (d, t, sorted(members.get((d, t), []))) for d, t in gmsh.model.getPhysicalGroups()}
    expect = {}
    for name, dim, tag, elements in plan.groups:
        expect[name] = (3 if dim == "solid" else 2, tag, sorted(i + 1 for i in elements))
    report["groups"] = {"planned": len(expect), "loaded": len(got_groups)}
    if groups and expect != got_groups:
        problems.append("physical groups differ from the planned groups")
    lap("physical_groups")
    # (e) volumes: mass is orientation sensitive, so a wrongly oriented shell shows up as a negative volume
    g = topo.grid_g
    worst, masses = 0.0, []
    for i, vid in enumerate(plan.solid_ids):
        planned_volume = float(sum(sh.signed_volume for sh in topo.volumes[vid].shells)) * g * g
        mass = gmsh.model.occ.getMass(3, i + 1)
        masses.append(mass)
        worst = max(worst, abs(mass - planned_volume) / max(abs(planned_volume), 1e-30))
    lap("volumes")
    report["volumes_um3"] = masses
    report["max_relative_volume_error_vs_plan"] = worst
    if min(masses) <= 0:
        problems.append(f"a volume has non-positive mass {min(masses):.6g} (shell orientation)")
    if worst > 1e-9:
        problems.append(f"volume differs from plan (relative {worst:.3g})")
    # (f) areas: per face and per physical group, against the planned (arc-realised) areas
    from sgbproto.emit import planned_face_area_um2

    face_err, group_area = 0.0, {}
    for i, fid in enumerate(plan.face_ids):
        face = topo.faces[fid]
        planned_area, mass = planned_face_area_um2(topo, face), gmsh.model.occ.getMass(2, i + 1)
        face_err = max(face_err, abs(mass - planned_area) / max(planned_area, 1e-30))
        if face.group is not None:
            row = group_area.setdefault(face.group, [0.0, 0.0])
            row[0] += planned_area
            row[1] += mass
    lap("face_areas")
    report["max_relative_face_area_error"] = face_err
    report["group_areas_um2"] = {k: {"planned": v[0], "loaded": v[1]} for k, v in sorted(group_area.items())}
    if face_err > 1e-9:
        problems.append(f"a face area differs from plan (relative {face_err:.3g})")
    if mesh_size is not None:
        report["mesh"] = _mesh_loaded(topo, tags, mesh_size, mesh_size_min, supports, refine_families, refine_size,
                                      refine_distance)
        if not report["mesh"]["conformal"]:
            problems.append(f"mesh not conformal: {report['mesh']['problems'][:3]}")
        for name, row in report["mesh"].get("bands", {}).items():
            if not (row["triangles"] > 0 and row["all_shared_by_tets"] and row["centroids_in_band"]):
                problems.append(f"band {name}: a triangle straddles a band boundary or is not shared by the tets")
    report["ok"] = not problems
    return report


def _mesh_loaded(topo: Topology, tags: dict[str, tuple[int, int]], size: float, size_min: float,
                 supports: dict | None = None, refine_families: tuple[str, ...] = (), refine_size: float = 1.0,
                 refine_distance: tuple[float, float] = (2.0, 20.0)) -> dict:
    import gmsh
    from types import SimpleNamespace

    from sgbproto.emit import _refine_near_curves, band_mesh_report, check_mesh_conformity

    for vid, vol in sorted(topo.volumes.items()):  # the mesher re-applies the planned embeddings (XAO/BREP cannot carry them)
        if vol.emitted:
            for fid in vol.embedded_faces:
                gmsh.model.mesh.embed(2, [tags[fid][1]], 3, tags[vid][1])
    for fid in {f for v in topo.volumes.values() if v.emitted for f in _faces_of(topo, v)}:
        refinement = topo.faces[fid].refinement_edges
        if refinement:
            gmsh.model.mesh.embed(1, [tags[e][1] for e in refinement], 2, tags[fid][1])
    if refine_families:  # the same size field the gmsh-API path uses around the band curves
        _refine_near_curves(topo, SimpleNamespace(tags=tags), refine_families, size, refine_size, refine_distance)
    gmsh.option.setNumber("Mesh.MeshSizeMax", size)
    gmsh.option.setNumber("Mesh.MeshSizeMin", size_min)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    t0 = time.perf_counter()
    gmsh.model.mesh.generate(3)
    seconds = time.perf_counter() - t0
    out = check_mesh_conformity(topo, tags)
    out["mesh_seconds"] = seconds
    if supports:
        out["bands"] = band_mesh_report(topo, tags, supports)
    return out


def _faces_of(topo: Topology, vol) -> set[str]:
    return {fid for sh in vol.shells for fid, _ in sh.faces} | set(vol.embedded_faces)
