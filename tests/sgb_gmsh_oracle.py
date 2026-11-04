"""Test oracles that use the Gmsh API: an independent bottom-up emitter and mesh conformity checks.

The production writer serialises the planned topology directly (``emission``).  These helpers rebuild the
same topology the slow way -- ``addPoint`` ... ``addVolume`` through Gmsh's OCC kernel -- so a test can
compare the two, and check that a coarse mesh of a loaded model is conformal across every planned face.
They never ship: Gmsh's API cannot express the sense of a solid, which is one reason the writer exists.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import gmsh
import numpy as np

from scgsim.geometry.sgb.model import ConformalModel


@contextmanager
def gmsh_session(path: Path | None = None) -> Iterator[None]:
    """A fresh Gmsh session, optionally with ``path`` (BREP or XAO) opened and synchronised; autofix off."""
    gmsh.initialize(["-noenv"])
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Geometry.OCCAutoFix", 0)
        if path is not None:
            (gmsh.open if str(path).endswith(".xao") else gmsh.merge)(str(path))
            gmsh.model.occ.synchronize()
        yield
    finally:
        gmsh.finalize()


@dataclass
class OracleResult:
    """What the API emitter made: planned id -> (dim, tag), the groups it added and the faces it could not make exact."""

    tags: dict[str, tuple[int, int]] = field(default_factory=dict)
    groups: dict[str, tuple[int, list[int]]] = field(
        default_factory=dict
    )  # name -> (dim, native tags)
    filled_walls: list[str] = field(
        default_factory=list
    )  # curved walls built by surface filling (approximate)


def emit_with_gmsh_api(
    model: ConformalModel, xao_path: Path | None = None
) -> OracleResult:
    """Replay the model through the Gmsh/OCC API in the current session (optionally writing an XAO)."""
    result = OracleResult()
    _build_entities(model, result)
    gmsh.model.occ.synchronize()
    for name, group in model.routed.groups.items():
        tags = [result.tags[i][1] for i in group.ids if i in result.tags]
        if tags:
            gmsh.model.addPhysicalGroup(group.dim, tags, name=name)
            result.groups[name] = (group.dim, sorted(tags))
    if xao_path is not None:
        gmsh.write(str(xao_path))
    return result


def _build_entities(model: ConformalModel, result: OracleResult) -> None:
    occ = gmsh.model.occ
    topo = model.topology
    volumes = {
        vid: v
        for vid, v in sorted(topo.volumes.items())
        if vid in model.routed.emitted_volumes
    }
    needed: set[str] = set()
    for v in volumes.values():
        for shell in v.shells:
            needed.update(fid for fid, _ in shell.faces)
        needed.update(v.embedded_faces)
    faces = {fid: topo.faces[fid] for fid in sorted(needed)}
    edges = {eid for f in faces.values() for lp in f.loops for eid, _ in lp}
    edges |= {eid for f in faces.values() for eid in f.refinement_edges}
    vertices = {v for eid in edges for v in (topo.edges[eid].v0, topo.edges[eid].v1)}
    centres = {
        topo.edges[eid].centre_vertex for eid in edges if topo.edges[eid].centre_vertex
    }
    g = topo.grid_um
    for vid in sorted(vertices):
        v = topo.vertices[vid]
        result.tags[vid] = (
            0,
            occ.addPoint(v.x * g, v.y * g, float(v.z), 0.0, len(result.tags) + 1),
        )
    centre_tag = {
        cid: occ.addPoint(
            topo.vertices[cid].x * g,
            topo.vertices[cid].y * g,
            float(topo.vertices[cid].z),
        )
        for cid in sorted(centres)
    }
    for n, eid in enumerate(sorted(edges), start=1):
        e = topo.edges[eid]
        a, b = result.tags[e.v0][1], result.tags[e.v1][1]
        tag = (
            occ.addLine(a, b, n)
            if e.curve is None
            else occ.addCircleArc(a, centre_tag[e.centre_vertex], b, n)
        )
        result.tags[eid] = (1, tag)
    if centre_tag:
        occ.remove(
            [(0, t) for t in centre_tag.values()]
        )  # construction aids; a free point would be meshed
    for n, (fid, f) in enumerate(faces.items(), start=1):
        wires = [
            occ.addCurveLoop(
                [
                    sign * result.tags[eid][1]
                    for eid, sign in (lp if i == 0 else _reversed(lp))
                ]
            )
            for i, lp in enumerate(f.loops)
        ]
        if f.planar and not f.curved:
            tag = occ.addPlaneSurface(wires, n)
        elif len(f.loops[0]) == 4:
            tag = occ.addBSplineFilling(
                wires[0], n
            )  # four shared edges: a Coons patch is the exact cylinder
        else:
            result.filled_walls.append(fid)
            tag = occ.addSurfaceFilling(
                wires[0], n, [], 3, 50, 2, False, 1e-5, 1e-4, 1e-2, 1e-1, 8, 20
            )
        result.tags[fid] = (2, tag)
    for n, (vid, v) in enumerate(volumes.items(), start=1):
        loops = [
            occ.addSurfaceLoop([result.tags[fid][1] for fid, _ in shell.faces])
            for shell in v.shells
        ]
        result.tags[vid] = (3, occ.addVolume(loops, n))


def _reversed(loop: tuple[tuple[str, int], ...]) -> list[tuple[str, int]]:
    return [(eid, -sign) for eid, sign in reversed(loop)]


def embed_planned(model: ConformalModel, tags: dict[str, tuple[int, int]]) -> None:
    """Re-apply the planned embeddings from the topology (what the mesher does from the sidecar)."""
    topo = model.topology
    for vid, v in sorted(topo.volumes.items()):
        if vid in model.routed.emitted_volumes:
            for fid in v.embedded_faces:
                gmsh.model.mesh.embed(2, [tags[fid][1]], 3, tags[vid][1])
    for fid, f in topo.faces.items():
        if f.refinement_edges and fid in tags:
            gmsh.model.mesh.embed(
                1, [tags[e][1] for e in f.refinement_edges], 2, tags[fid][1]
            )


# --------------------------------------------------------------------------- mesh conformity


def mesh_coarsely(size: float, size_min: float = 0.0) -> float:
    """Generate a 3D mesh with the given maximum size in the current session; returns the seconds it took."""
    import time  # noqa: PLC0415 -- used here only

    gmsh.option.setNumber("Mesh.MeshSizeMax", size)
    gmsh.option.setNumber("Mesh.MeshSizeMin", size_min)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    started = time.perf_counter()
    gmsh.model.mesh.generate(3)
    return time.perf_counter() - started


def refine_near(  # noqa: PLR0913 -- a test helper
    model: ConformalModel,
    tags: dict[str, tuple[int, int]],
    families: tuple[str, ...],
    size: float,
    refine_size: float,
    distance: tuple[float, float],
) -> None:  # noqa: PLR0913 -- a test helper
    """Background size field: small elements near the curves of the given families (e.g. inset band edges)."""
    topo = model.topology
    curves = [
        tags[eid][1]
        for eid, e in topo.edges.items()
        if eid in tags
        and any(src.startswith(f + "/") for src in e.sources for f in families)
    ]
    field_api = gmsh.model.mesh.field
    dist = field_api.add("Distance")
    field_api.setNumbers(dist, "CurvesList", curves)
    field_api.setNumber(dist, "Sampling", 50)
    thr = field_api.add("Threshold")
    field_api.setNumber(thr, "InField", dist)
    field_api.setNumber(thr, "SizeMin", refine_size)
    field_api.setNumber(thr, "SizeMax", size)
    field_api.setNumber(thr, "DistMin", distance[0])
    field_api.setNumber(thr, "DistMax", distance[1])
    field_api.setAsBackgroundMesh(thr)
    for option in (
        "Mesh.MeshSizeFromPoints",
        "Mesh.MeshSizeFromCurvature",
        "Mesh.MeshSizeExtendFromBoundary",
    ):
        gmsh.option.setNumber(option, 0)


def size_from_mesh_hints(  # noqa: PLR0913 -- the hint kinds and the three sizing numbers of the recipe
    sidecar: dict,
    *,
    coarse: float,
    near: float,
    growth: float,
    kinds: tuple[str, ...] = ("band_boundary",),
) -> None:
    """Background size field driven by the sidecar's mesh hints (the recipe ``sidecar.hints`` documents).

    The element size is ``near`` on the hinted edges and grows by ``growth`` per micrometre of distance
    (so it is ``coarse`` from ``(coarse - near) / growth`` away), in the current session.
    """
    field_api = gmsh.model.mesh.field
    curves = sorted(
        {
            t
            for h in sidecar["mesh_hints"]
            if h["kind"] in kinds
            for t in h["native_tags"]
        }
    )
    dist = field_api.add("Distance")
    field_api.setNumbers(dist, "CurvesList", curves)
    field_api.setNumber(dist, "Sampling", 50)
    ramp = field_api.add("Threshold")
    field_api.setNumber(ramp, "InField", dist)
    field_api.setNumber(ramp, "SizeMin", near)
    field_api.setNumber(ramp, "SizeMax", coarse)
    field_api.setNumber(ramp, "DistMin", 0.0)
    field_api.setNumber(ramp, "DistMax", (coarse - near) / growth)
    field_api.setAsBackgroundMesh(ramp)
    for option in (
        "Mesh.MeshSizeFromPoints",
        "Mesh.MeshSizeFromCurvature",
        "Mesh.MeshSizeExtendFromBoundary",
    ):
        gmsh.option.setNumber(option, 0)


def _tet_face_counters(
    model: ConformalModel, tags: dict[str, tuple[int, int]]
) -> dict[str, Counter]:
    out: dict[str, Counter] = {}
    for vid in sorted(model.routed.emitted_volumes):
        _, _, nodes = gmsh.model.mesh.getElements(3, tags[vid][1])
        tets = (
            np.array(nodes[0]).reshape(-1, 4)
            if nodes
            else np.zeros((0, 4), dtype=np.int64)
        )
        counter: Counter = Counter()
        for combo in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
            counter.update(map(tuple, np.sort(tets[:, combo], axis=1).tolist()))
        out[vid] = counter
    return out


def check_mesh_conformity(  # noqa: PLR0912 -- one check per way a mesh can be non-conformal
    model: ConformalModel, tags: dict[str, tuple[int, int]]
) -> dict:
    """Verify the current Gmsh mesh is conformal across every planned face.

    No two nodes coincide; for each volume the faces seen exactly once among its tets equal the union of the
    triangles of its planned surfaces; each planned face's triangles belong to the tets of every adjacent
    emitted volume (twice for an embedded face).
    """
    topo = model.topology
    emitted = model.routed.emitted_volumes
    _, coords, _ = gmsh.model.mesh.getNodes()
    xyz = np.round(np.array(coords).reshape(-1, 3), 7)
    duplicate_nodes = int(len(xyz) - len(np.unique(xyz, axis=0)))
    tet_faces = _tet_face_counters(model, tags)
    problems: list[str] = []
    surface_tris: dict[str, set] = {}
    for fid, (dim, tag) in tags.items():
        if dim != 2:
            continue
        _, _, nodes = gmsh.model.mesh.getElements(2, tag)
        tri = (
            np.sort(np.array(nodes[0]).reshape(-1, 3), axis=1)
            if nodes
            else np.zeros((0, 3), dtype=np.int64)
        )
        surface_tris[fid] = set(map(tuple, tri.tolist()))
        if len(tri) == 0:
            problems.append(f"face {fid} has no triangles")
    for fid, tris in surface_tris.items():
        f = topo.faces[fid]
        sides = [
            v for v in (f.neg_volume, f.pos_volume) if v is not None and v in emitted
        ]
        if f.neg_volume is not None and f.neg_volume == f.pos_volume:
            if any(tet_faces[f.neg_volume].get(t, 0) != 2 for t in tris):
                problems.append(f"embedded face {fid}: triangle not on both sides")
            continue
        for vid in sides:
            if any(tet_faces[vid].get(t, 0) != 1 for t in tris):
                problems.append(
                    f"face {fid}: triangles missing from tets of volume {vid}"
                )
    for vid, counter in tet_faces.items():
        boundary = {t for t, c in counter.items() if c == 1}
        planned: set = set()
        for fid, tris in surface_tris.items():
            f = topo.faces[fid]
            if vid in (f.neg_volume, f.pos_volume) and f.neg_volume != f.pos_volume:
                planned |= tris
        if boundary != planned:
            problems.append(
                f"volume {vid}: tet boundary has {len(boundary - planned)} unplanned and {len(planned - boundary)} missing triangles"
            )
        if any(c > 2 for c in counter.values()):
            problems.append(f"volume {vid}: a face is shared by more than two tets")
    tets = sum(
        len(i)
        for vid in emitted
        for i in gmsh.model.mesh.getElements(3, tags[vid][1])[1]
    )
    return {"nodes": len(xyz), "duplicate_nodes": duplicate_nodes, "tets": tets, "problems": problems,
            "conformal": duplicate_nodes == 0 and not problems}  # fmt: skip


def band_mesh_report(
    model: ConformalModel, tags: dict[str, tuple[int, int]], supports: dict
) -> dict:
    """Per band group: triangles, tets on each side, and whether every triangle lies inside its band.

    A band group is conformal when each of its triangles is a face of exactly one tet of every adjacent emitted
    volume (so no element straddles a band boundary) and its centroid lies in the band by the exact-carrier
    distance test.
    """
    topo = model.topology
    g = topo.grid_um
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    xyz = {int(t): (coords[3 * i], coords[3 * i + 1]) for i, t in enumerate(node_tags)}
    tet_faces = _tet_face_counters(model, tags)
    report: dict[str, dict] = {}
    for name, group in model.routed.groups.items():
        if group.dim != 2 or "::band=" not in name:
            continue
        k = int(name.rsplit("=", 1)[1])
        row = {"faces": len(group.ids), "triangles": 0, "tets_per_side": {}, "all_shared_by_tets": True,
               "centroids_in_band": True}  # fmt: skip
        for fid in group.ids:
            f = topo.faces[fid]
            _, _, nodes = gmsh.model.mesh.getElements(2, tags[fid][1])
            tri = np.array(nodes[0]).reshape(-1, 3)
            row["triangles"] += len(tri)
            support = supports[f.bands[0][0]]
            for t in tri:
                key = tuple(sorted(int(n) for n in t))
                for vid in (f.neg_volume, f.pos_volume):
                    if vid is not None and vid in model.routed.emitted_volumes:
                        hit = tet_faces[vid].get(key, 0) == 1
                        row["tets_per_side"][vid] = row["tets_per_side"].get(
                            vid, 0
                        ) + int(hit)
                        row["all_shared_by_tets"] &= hit
                cx = sum(xyz[int(n)][0] for n in t) / 3 / g
                cy = sum(xyz[int(n)][1] for n in t) / 3 / g
                row["centroids_in_band"] &= support.band_of((cx, cy, 1)) == k
        report[name] = row
    return report


def mesh_quality() -> dict[str, float]:
    """Quality of the tetrahedra of the current mesh: min and 1st-percentile ``minSICN`` and ``gamma``.

    ``minSICN`` is Gmsh's signed inverse condition number (1 is a regular tetrahedron, <= 0 an inverted or flat
    one); ``gamma`` is the inscribed-to-circumscribed radius ratio scaled to 1 for a regular tetrahedron.
    """
    types, ids, _ = gmsh.model.mesh.getElements(3)
    out: dict[str, float] = {}
    tets = [
        int(i) for t, chunk in zip(types, ids, strict=True) if t == 4 for i in chunk
    ]
    if not tets:
        return {"tets": 0}
    for quality in ("minSICN", "gamma"):
        values = np.array(gmsh.model.mesh.getElementQualities(tets, quality))
        out[f"{quality}_min"] = float(values.min())
        out[f"{quality}_p1"] = float(np.percentile(values, 1))
    out["tets"] = len(tets)
    return out
