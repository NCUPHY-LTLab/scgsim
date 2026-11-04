"""Bottom-up emission through the Gmsh/OCC API (brief section 12, stage S10).

Owns replaying an already-decided :class:`Topology` into a CAD model with *no*
Boolean operation: points (once per vertex) -> lines (once per atomic edge) ->
curve loops -> plane surfaces (walls that are not planar use surface filling)
-> surface loops -> volumes. ``Geometry.OCCAutoFix`` is off: a fix would be a
planning defect. Faces that lie inside a volume (a sheet with the same volume
on both sides) are embedded after synchronisation. Physical groups come from
the semantic names assigned during assembly.

Outputs a ``model.xao`` and a JSON sidecar that binds planned ids to native
tags *by construction order* -- never by geometry matching -- plus a mesh
conformity check used by the examples and tests.

Does not own topology decisions; it reads them.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import gmsh
import numpy as np

from sgbproto.errors import SgbError
from sgbproto.assembly import TopoFace, Topology


ATTEMPTS = 12  # face orders tried per solid, see _Emitter._add_volume


@dataclass
class EmitResult:
    xao_path: Path
    sidecar_path: Path
    tags: dict[str, tuple[int, int]]  # planned id -> (dim, native tag)
    order: list[dict] = field(default_factory=list)
    mesh: dict | None = None
    timings: dict = field(default_factory=dict)
    wall_deviation: list[dict] = field(default_factory=list)  # curved walls: realised surface vs cylinder (grid units)
    masses: dict = field(default_factory=dict)  # measure_masses: planned vs OCC area/volume per face and volume


def emit(
    topo: Topology,
    out_dir: Path,
    name: str = "model",
    mesh_size: float | None = None,
    mesh_size_min: float = 0.0,
    gmsh_options: dict[str, float] | None = None,
    write_xao: bool = True,
    measure_masses: bool = False,
    orientation_attempts: int = ATTEMPTS,
    supports: dict | None = None,
    refine_families: tuple[str, ...] = (),
    refine_size: float = 1.0,
    refine_distance: tuple[float, float] = (2.0, 20.0),
) -> EmitResult:
    """Emit XAO + sidecar; optionally mesh and verify conformity when ``mesh_size`` is given."""
    out_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    gmsh.initialize(["-noenv"])
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Geometry.OCCAutoFix", 0)
        for key, value in (gmsh_options or {}).items():
            gmsh.option.setNumber(key, value)
        gmsh.model.add(name)
        t0 = time.perf_counter()
        emitter = _Emitter(topo)
        emitter.attempts = orientation_attempts
        emitter.build()
        timings["occ_build"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        gmsh.model.occ.synchronize()
        timings["synchronize"] = time.perf_counter() - t0
        wall_deviation = emitter.measure_curved_walls()
        masses = emitter.measure_masses() if measure_masses else {}
        t0 = time.perf_counter()
        emitter.embed_faces()
        timings["embed"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        emitter.add_physical_groups()
        timings["physical_groups"] = time.perf_counter() - t0
        _check_one_planned_id_per_native(emitter)
        xao = out_dir / f"{name}.xao"
        t0 = time.perf_counter()
        if write_xao:
            gmsh.write(str(xao))
        timings["write_xao"] = time.perf_counter() - t0
        sidecar = out_dir / f"{name}.sidecar.json"
        sidecar.write_text(json.dumps(emitter.sidecar(), indent=1, sort_keys=True))
        mesh_report = None
        if mesh_size is not None:
            t0 = time.perf_counter()
            mesh_report = _mesh_and_check(
                topo, emitter, mesh_size, mesh_size_min, supports or {}, refine_families, refine_size, refine_distance
            )
            timings["mesh_and_check"] = time.perf_counter() - t0
        return EmitResult(xao, sidecar, dict(emitter.tags), emitter.order, mesh_report, timings, wall_deviation, masses)
    finally:
        gmsh.finalize()


class _Emitter:
    def __init__(self, topo: Topology) -> None:
        self.topo = topo
        self.tags: dict[str, tuple[int, int]] = {}
        self.order: list[dict] = []
        self.group_tags: dict[str, dict] = {}
        self.nonplanar_faces: list[str] = []
        self.emitted_faces: dict[str, TopoFace] = {}
        self.aux_centres: dict[str, list[float]] = {}
        self.attempts = ATTEMPTS
        self.inside_out: list[str] = []  # solids whose sewn shell has a negative sense (see _add_volume)

    def _record(self, dim: int, planned: str, tag: int) -> int:
        self.tags[planned] = (dim, tag)
        self.order.append({"order": len(self.order), "dim": dim, "id": planned, "tag": tag})
        return tag

    def build(self) -> None:
        occ = gmsh.model.occ
        topo = self.topo
        emitted_vols = {vid: v for vid, v in sorted(topo.volumes.items()) if v.emitted}
        # Faces needed: the boundary faces of emitted volumes plus their embedded faces.
        needed: set[str] = set()
        for v in emitted_vols.values():
            for shell in v.shells:
                needed.update(fid for fid, _ in shell.faces)
            needed.update(v.embedded_faces)
        self.emitted_faces = {fid: topo.faces[fid] for fid in sorted(needed)}
        used_edges = {eid for f in self.emitted_faces.values() for lp in f.loops for eid, _ in lp}
        used_edges |= {eid for f in self.emitted_faces.values() for eid in f.refinement_edges}
        used_vertices = {v for eid in used_edges for v in (topo.edges[eid].v0, topo.edges[eid].v1)}
        centres = {topo.edges[eid].centre_vertex for eid in used_edges if topo.edges[eid].centre_vertex}
        g = topo.grid_g
        # Explicit tags everywhere (construction order, from 1): Gmsh then never searches for a free tag.
        for n, vid in enumerate(sorted(used_vertices), start=1):
            v = topo.vertices[vid]
            self._record(0, vid, occ.addPoint(v.x * g, v.y * g, float(v.z), 0.0, n))
        # Arc centres are construction aids: a point is created, the arc is built about it, the point is removed
        # again (a free point would be meshed as a stray node). They are listed in the sidecar, not as entities.
        centre_tag = {}
        for cid in sorted(centres):
            v = topo.vertices[cid]
            centre_tag[cid] = occ.addPoint(v.x * g, v.y * g, float(v.z))
        for n, eid in enumerate(sorted(used_edges), start=1):
            e = topo.edges[eid]
            if e.curve is None:
                self._record(1, eid, occ.addLine(self.tags[e.v0][1], self.tags[e.v1][1], n))
            else:  # minor arc v0 -> v1 about the planned centre point
                self._record(1, eid, occ.addCircleArc(self.tags[e.v0][1], centre_tag[e.centre_vertex], self.tags[e.v1][1], n))
        if centre_tag:
            occ.remove([(0, t) for t in centre_tag.values()])
            self.aux_centres = {cid: [topo.vertices[cid].x, topo.vertices[cid].y] for cid in sorted(centre_tag)}
        for n, (fid, f) in enumerate(self.emitted_faces.items(), start=1):
            # OCC wants hole wires oriented opposite to the outer wire as it sees them; our stored holes are
            # clockwise like the outer boundary's complement, which OCC reads as same-sense (area = outer + holes).
            wires = [occ.addCurveLoop([sign * self.tags[eid][1] for eid, sign in (lp if i == 0 else _reversed_loop(lp))])
                     for i, lp in enumerate(f.loops)]
            if f.planar:
                tag = occ.addPlaneSurface(wires, n)
            else:
                if len(wires) != 1:
                    raise SgbError(f"non-planar face {fid} with holes cannot be emitted")
                self.nonplanar_faces.append(fid)
                if len(f.loops[0]) == 4:
                    # Four shared edges (two arcs, two verticals): a Coons/B-spline patch is the exact cylinder.
                    tag = occ.addBSplineFilling(wires[0], n)
                else:  # longer boundary: constrained filling is only approximate (see measure_curved_walls)
                    tag = occ.addSurfaceFilling(wires[0], n, [], 3, 50, 2, False, 1e-5, 1e-4, 1e-2, 1e-1, 8, 20)
            self._record(2, fid, tag)
        for n, (vid, v) in enumerate(emitted_vols.items(), start=1):
            self._record(3, vid, self._add_volume(vid, v, n))

    def _add_volume(self, vid: str, v, n: int) -> int:
        """A solid with positive mass where Gmsh's sewing allows it.

        ``addSurfaceLoop(sewing=True)`` makes a shell mutually consistent, but its *absolute* sense is not
        controllable from the API (measured: not the first face, not the lowest tag, not the face list order in a
        predictable way), so a solid can come out inside out (negative mass). Up to ``ATTEMPTS`` deterministic
        face orders are tried (outward face first, reversed, rotations); if none gives a positive mass the solid is
        kept as built and listed in the sidecar (``inside_out_solids``). Meshing is unaffected; a consumer that
        needs a positive solid uses the direct BREP writer (D8), which sets every orientation explicitly.

        ``orientation_attempts=0`` skips sewing and the check (sewing a shell of 10^5 faces needs tens of GB): the
        scaling runs use it, and the shells' senses are then unchecked.
        """
        occ = gmsh.model.occ
        if self.attempts <= 0:  # large models: no sewing (it is quadratic in memory and time), shell senses unchecked
            loops = [occ.addSurfaceLoop([self.tags[fid][1] for fid, _ in shell.faces]) for shell in v.shells]
            return occ.addVolume(loops, n)
        for attempt, orders in enumerate(self._shell_orders(v, self.attempts)):
            loops = [occ.addSurfaceLoop([self.tags[fid][1] for fid in order], sewing=True) for order in orders]
            tag = occ.addVolume(loops, n)
            if occ.getMass(3, tag) > 0:
                return tag
            if attempt + 1 < self.attempts:
                occ.remove([(3, tag)])
        self.inside_out.append(vid)
        return tag

    def _shell_orders(self, v, limit: int):
        """Deterministic candidate face orders per shell: (sorted with an outward horizontal face first), its
        reverse, then rotations of the sorted list."""
        per_shell = []
        for shell in v.shells:
            ids = sorted(fid for fid, _ in shell.faces)
            wanted = 1 if shell.role == "outer" else -1
            anchor = next((fid for fid, sign in sorted(shell.faces) if sign == wanted
                           and self.topo.faces[fid].kind == "horizontal" and self.topo.faces[fid].planar), None)
            first = ([anchor] + [f for f in ids if f != anchor]) if anchor else ids
            per_shell.append([first, first[::-1]] + [ids[k:] + ids[:k] for k in range(1, len(ids))])
        for k in range(min(limit, max(len(c) for c in per_shell))):
            yield [candidates[min(k, len(candidates) - 1)] for candidates in per_shell]

    def measure_masses(self) -> dict:
        """OCC ``getMass`` of every emitted face (area, um^2) and volume (um^3) against the planned values."""
        g2 = self.topo.grid_g**2
        faces = {fid: {"occ": gmsh.model.occ.getMass(2, self.tags[fid][1]), "planned": planned_face_area_um2(self.topo, f),
                       "group": f.group} for fid, f in self.emitted_faces.items()}
        volumes = {}
        for vid, v in self.topo.volumes.items():
            if v.emitted:
                volumes[vid] = {"occ": gmsh.model.occ.getMass(3, self.tags[vid][1]), "group": f"VOL::{v.domain.name}",
                                "planned": float(sum(s.signed_volume for s in v.shells)) * g2}
        return {"faces": faces, "volumes": volumes}

    def measure_curved_walls(self) -> list[dict]:
        """Distance of each emitted curved wall surface from the exact cylinder and from the realised one."""
        g = self.topo.grid_g
        rows = []
        for fid, f in self.emitted_faces.items():
            if not f.curved:
                continue
            cylinders = {}
            for eid, _ in f.loops[0]:
                curve = self.topo.edges[eid].curve
                if curve is not None:
                    cylinders[curve.carrier] = curve
            tag = self.tags[fid][1]
            lo, hi = gmsh.model.getParametrizationBounds(2, tag)
            exact = realised = 0.0
            for i in range(11):
                for j in range(11):
                    u = [lo[0] + (hi[0] - lo[0]) * i / 10, lo[1] + (hi[1] - lo[1]) * j / 10]
                    x, y, _ = gmsh.model.getValue(2, tag, u)
                    px, py = x / g, y / g
                    exact = max(exact, min(abs(math.hypot(px - c.root_centre[0], py - c.root_centre[1]) - c.root_radius)
                                           for c in cylinders.values()))
                    realised = max(realised, min(abs(math.hypot(px - c.centre[0], py - c.centre[1]) - c.radius)
                                                 for c in cylinders.values()))
            rows.append({"face": fid, "group": f.group, "z_range": [str(f.z_range[0]), str(f.z_range[1])],
                         "vs_exact_cylinder_grid": exact, "vs_realised_cylinder_grid": realised})
        return rows

    def embed_faces(self) -> None:
        for fid, f in self.emitted_faces.items():
            if f.refinement_edges:
                lines = [self.tags[eid][1] for eid in f.refinement_edges]
                gmsh.model.mesh.embed(1, lines, 2, self.tags[fid][1])
        for vid, v in sorted(self.topo.volumes.items()):
            if v.emitted:
                for fid in v.embedded_faces:
                    gmsh.model.mesh.embed(2, [self.tags[fid][1]], 3, self.tags[vid][1])

    def add_physical_groups(self) -> None:
        # Explicit physical tags: letting Gmsh pick the next free tag scans all groups (quadratic in their number).
        next_tag = {2: 1, 3: 1}
        for name, group in self.topo.groups.items():
            tags = [self.tags[i][1] for i in group["ids"] if i in self.tags]
            if not tags:
                continue
            dim = group["dim"]
            ptag = gmsh.model.addPhysicalGroup(dim, tags, tag=next_tag[dim], name=name)
            next_tag[dim] += 1
            self.group_tags[name] = {"dim": group["dim"], "physical_tag": ptag, "ids": group["ids"], "native_tags": tags}

    def sidecar(self) -> dict:
        return {
            "format": "sgbproto-sidecar-1",
            "route": self.topo.route,
            "grid_um": self.topo.grid_g,
            "binding": "planned id -> native tag by construction order (never by geometry)",
            "construction_order": self.order,
            "physical_groups": self.group_tags,
            "unions": self.topo.unions,
            "surface_epr_rows": self.topo.surface_epr_rows,
            "union_note": "unions are lists of primary physical-group names (e.g. all PEC faces of a Net); "
                          "they are never overlapping Gmsh groups",
            "nonplanar_wall_faces": self.nonplanar_faces,
            "inside_out_solids": self.inside_out,
            "arc_centre_points": self.aux_centres,
            "arc_centre_note": "construction aids (planned id -> centre in grid units); removed after the arcs were built",
            "curved_wall_faces": [fid for fid, f in self.emitted_faces.items() if f.curved],
            "embedded": {
                vid: v.embedded_faces for vid, v in sorted(self.topo.volumes.items()) if v.emitted and v.embedded_faces
            },
            "embedded_faces": [{"face": fid, "host_volume": vid} for vid, v in sorted(self.topo.volumes.items())
                               if v.emitted for fid in v.embedded_faces],
            "refinement_edges": [{"edge": eid, "host_face": fid} for fid, f in self.emitted_faces.items()
                                 for eid in f.refinement_edges],
            "embedded_note": "XAO cannot carry mesh embedding: the mesher must re-apply (face, host volume) and "
                             "(edge, host face) pairs",
        }


def _reversed_loop(loop) -> list[tuple[str, int]]:
    """The same boundary traversed the other way."""
    return [(eid, -sign) for eid, sign in reversed(loop)]


def _check_one_planned_id_per_native(em: _Emitter) -> None:
    """Every native entity must correspond to exactly one planned id (and vice versa)."""
    for dim in range(4):
        native = sorted(t for d, t in gmsh.model.getEntities(dim))
        planned = sorted(tag for d, tag in em.tags.values() if d == dim)
        if native != planned:
            raise SgbError(f"dim {dim}: native entities {len(native)} != planned ids {len(planned)}")


# --------------------------------------------------------------------------- mesh conformity


def _mesh_and_check(
    topo: Topology, em: _Emitter, size: float, size_min: float, supports: dict, refine_families: tuple[str, ...],
    refine_size: float, refine_distance: tuple[float, float],
) -> dict:
    if refine_families:
        _refine_near_curves(topo, em, refine_families, size, refine_size, refine_distance)
    gmsh.option.setNumber("Mesh.MeshSizeMax", size)
    gmsh.option.setNumber("Mesh.MeshSizeMin", size_min)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    t0 = time.perf_counter()
    gmsh.model.mesh.generate(3)
    mesh_time = time.perf_counter() - t0
    report = check_mesh_conformity(topo, em.tags)
    report["mesh_seconds"] = mesh_time
    if supports:
        report["bands"] = band_mesh_report(topo, em.tags, supports)
    return report


def _refine_near_curves(topo, em, families, size, refine_size, distance) -> None:
    """Background size field: small elements near the curves of the given families (e.g. inset band edges)."""
    curves = [em.tags[eid][1] for eid, e in topo.edges.items()
              if eid in em.tags and any(src.startswith(f + "/") for src in e.sources for f in families)]
    field = gmsh.model.mesh.field
    dist = field.add("Distance")
    field.setNumbers(dist, "CurvesList", curves)
    field.setNumber(dist, "Sampling", 50)
    thr = field.add("Threshold")
    field.setNumber(thr, "InField", dist)
    field.setNumber(thr, "SizeMin", refine_size)
    field.setNumber(thr, "SizeMax", size)
    field.setNumber(thr, "DistMin", distance[0])
    field.setNumber(thr, "DistMax", distance[1])
    field.setAsBackgroundMesh(thr)
    for option in ("Mesh.MeshSizeFromPoints", "Mesh.MeshSizeFromCurvature", "Mesh.MeshSizeExtendFromBoundary"):
        gmsh.option.setNumber(option, 0)


def band_mesh_report(topo: Topology, tags: dict[str, tuple[int, int]], supports: dict) -> dict:
    """Per band group: triangles, tets on each side, and whether every triangle lies inside its band.

    A band group is conformal when each of its triangles is a face of exactly one tet of every adjacent emitted
    volume (so no element straddles a band boundary) and its centroid lies in the band by the exact-carrier
    distance test.
    """
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    xyz = {int(t): (coords[3 * i], coords[3 * i + 1]) for i, t in enumerate(node_tags)}
    tet_faces = _tet_face_counters(topo, tags)
    g = topo.grid_g
    report: dict[str, dict] = {}
    for name, group in topo.groups.items():
        if group["dim"] != 2 or "::band=" not in name:
            continue
        k = int(name.rsplit("=", 1)[1])
        row = {"faces": len(group["ids"]), "triangles": 0, "tets_per_side": {}, "all_shared_by_tets": True,
               "centroids_in_band": True, "triangles_per_face_min": None}
        for fid in group["ids"]:
            f = topo.faces[fid]
            _, _, nodes = gmsh.model.mesh.getElements(2, tags[fid][1])
            tri = np.array(nodes[0]).reshape(-1, 3)
            row["triangles"] += len(tri)
            row["triangles_per_face_min"] = min(row["triangles_per_face_min"] or len(tri), len(tri))
            support = supports[f.bands[0][0]]
            for t in tri:
                key = tuple(sorted(int(n) for n in t))
                for vid in (f.neg_volume, f.pos_volume):
                    if vid is not None and topo.volumes[vid].emitted:
                        hit = tet_faces[vid].get(key, 0) == 1
                        row["tets_per_side"][vid] = row["tets_per_side"].get(vid, 0) + int(hit)
                        row["all_shared_by_tets"] &= hit
                cx = sum(xyz[int(n)][0] for n in t) / 3 / g
                cy = sum(xyz[int(n)][1] for n in t) / 3 / g
                row["centroids_in_band"] &= support.band_of((cx, cy)) == k
        report[name] = row
    return report


def _tet_face_counters(topo: Topology, tags: dict[str, tuple[int, int]]) -> dict[str, Counter]:
    """For every emitted volume: how many tets of that volume own each (sorted) triangle of node tags."""
    out: dict[str, Counter] = {}
    for vid, v in topo.volumes.items():
        if not v.emitted:
            continue
        _, _, nodes = gmsh.model.mesh.getElements(3, tags[vid][1])
        tets = np.array(nodes[0]).reshape(-1, 4) if nodes else np.zeros((0, 4), dtype=np.int64)
        cnt: Counter = Counter()
        for combo in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
            tri = np.sort(tets[:, combo], axis=1)
            cnt.update(map(tuple, tri.tolist()))
        out[vid] = cnt
    return out


def duplicate_node_count() -> tuple[int, int]:
    """(node count, number of nodes coinciding with another node to 1e-7 um) of the current mesh."""
    _, coords, _ = gmsh.model.mesh.getNodes()
    xyz = np.round(np.array(coords).reshape(-1, 3), 7)
    return int(len(xyz)), int(len(xyz) - len(np.unique(xyz, axis=0)))


def _count_tets(topo: Topology, tags: dict[str, tuple[int, int]]) -> int:
    total = 0
    for vid, v in topo.volumes.items():
        if v.emitted:
            _, ids, _ = gmsh.model.mesh.getElements(3, tags[vid][1])
            total += sum(len(i) for i in ids)
    return total


def check_mesh_conformity(topo: Topology, tags: dict[str, tuple[int, int]]) -> dict:
    """Verify the current Gmsh mesh is conformal across every planned face.

    * no two nodes coincide (after rounding to 1e-7 um);
    * for each volume, the faces seen exactly once among its tets equal the
      union of the triangles of its planned surfaces;
    * each planned face's triangles belong to the tets of every volume that
      the plan says is adjacent (twice for an embedded face).
    """
    n_nodes, duplicate_nodes = duplicate_node_count()

    tet_faces = _tet_face_counters(topo, tags)

    problems: list[str] = []
    surface_tris: dict[str, set] = {}
    for fid, (dim, tag) in tags.items():
        if dim != 2:
            continue
        _, _, nodes = gmsh.model.mesh.getElements(2, tag)
        tri = np.sort(np.array(nodes[0]).reshape(-1, 3), axis=1) if nodes else np.zeros((0, 3), dtype=np.int64)
        surface_tris[fid] = set(map(tuple, tri.tolist()))
        if len(tri) == 0:
            problems.append(f"face {fid} has no triangles")
    shared_faces = 0
    for fid, tris in surface_tris.items():
        f = topo.faces[fid]
        sides = [v for v in (f.neg_volume, f.pos_volume) if v is not None and topo.volumes[v].emitted]
        if f.neg_volume is not None and f.neg_volume == f.pos_volume:
            for t in tris:
                if tet_faces[f.neg_volume].get(t, 0) != 2:
                    problems.append(f"embedded face {fid}: triangle not on both sides")
                    break
            continue
        if len(sides) == 2:
            shared_faces += 1
        for vid in sides:
            counts = tet_faces[vid]
            if any(counts.get(t, 0) != 1 for t in tris):
                problems.append(f"face {fid}: triangles missing from tets of volume {vid}")
    for vid, cnt in tet_faces.items():
        boundary = {t for t, c in cnt.items() if c == 1}
        planned = set()
        for fid, tris in surface_tris.items():
            f = topo.faces[fid]
            if vid in (f.neg_volume, f.pos_volume) and f.neg_volume != f.pos_volume:
                planned |= tris
        if boundary != planned:
            problems.append(f"volume {vid}: tet boundary has {len(boundary - planned)} unplanned and {len(planned - boundary)} missing triangles")
        if any(c > 2 for c in cnt.values()):
            problems.append(f"volume {vid}: a face is shared by more than two tets")
    return {
        "nodes": n_nodes,
        "duplicate_nodes": duplicate_nodes,
        "tets": _count_tets(topo, tags),
        "shared_interior_faces": shared_faces,
        "problems": problems,
        "conformal": duplicate_nodes == 0 and not problems,
    }


def surface_triangles(topo: Topology, mesh_size: float, mesh_size_min: float = 0.0) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Surface mesh of every emitted face: ``face id -> (points in um, triangles as point indices)``. For figures."""
    gmsh.initialize(["-noenv"])
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Geometry.OCCAutoFix", 0)
        gmsh.model.add("surfaces")
        emitter = _Emitter(topo)
        emitter.build()
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size_min)
        gmsh.model.mesh.generate(2)
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        index = {int(t): i for i, t in enumerate(node_tags)}
        xyz = np.array(coords).reshape(-1, 3)
        out = {}
        for fid, (dim, tag) in emitter.tags.items():
            if dim != 2:
                continue
            _, _, nodes = gmsh.model.mesh.getElements(2, tag)
            tri = np.array([index[int(n)] for n in nodes[0]]).reshape(-1, 3)
            out[fid] = (xyz, tri)
        return out
    finally:
        gmsh.finalize()


def planned_face_area_um2(topo: Topology, face: TopoFace) -> float:
    """Area (um^2) of a planned face with its arcs realised (what OCC should report for the emitted surface)."""
    g = topo.grid_g
    if face.kind == "horizontal":
        twice = float(face.twice_xy_area)
        for loop in face.loops:
            for eid, sign in loop:
                curve = topo.edges[eid].curve
                if curve is not None:
                    twice += curve.signed_segment_twice_area * sign  # the loop keeps the face on its left
        return twice / 2 * g * g
    # A wall is vertical: height times the length of its bottom boundary (arcs by arc length).
    z_lo, z_hi = face.z_range
    length = 0.0
    for eid, _ in face.loops[0]:
        e = topo.edges[eid]
        if e.kind != "horizontal" or topo.vertices[e.v0].z != z_lo:
            continue
        a, b = topo.vertices[e.v0], topo.vertices[e.v1]
        chord = math.hypot(b.x - a.x, b.y - a.y)
        if e.curve is None:
            length += chord
        else:
            length += e.curve.radius * 2 * math.asin(min(1.0, chord / (2 * e.curve.radius)))
    return length * g * float(z_hi - z_lo)
