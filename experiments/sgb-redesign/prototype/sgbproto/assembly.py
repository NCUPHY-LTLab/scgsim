"""Topology assembly from arrangements (brief section 2, stage S6; S7/S8 grouping).

Owns turning the dissolved slab and level arrangements into one shared,
conformal boundary representation -- *without any CAD kernel*:

* **vertices / edges**: every vertex and edge of a level arrangement, plus
  vertical edges on each wall column;
* **horizontal faces**: cells of ``A_i`` whose level label needs a face
  (``below != above`` or a sheet is present);
* **walls**: one vertical face per atomic slab edge and side-label pair, with
  *coalescing*: stacked walls of consecutive slabs merge into one face unless a
  level cuts the wall -- an ``A_i`` edge or vertex touching the wall's chain
  (e.g. a port sheet edge on a metal sidewall) or a sheet face along it.
  Bottom boundary = chain of ``A`` edges covering the slab edge at the lowest
  level of the run, top boundary = the chain at the highest;
* **vertical edges**: each wall column (x, y) is cut only at the levels where
  a wall run starts or ends or a horizontal face has a vertex, so neighbouring
  faces always share whole edges;
* **volumes**: connected components of slab cells joined across level cells
  whose below and above domains are equal and that carry no sheet;
* **shells**: faces around each volume grouped into connected closed surfaces,
  classified outer/inner by exact signed volume (integer arithmetic);
* **groups**: each face has exactly one primary physical group (``routes``).

A face with the same volume on both sides is a sheet inside a volume
(``embedded``); it is listed on the volume but is not part of its shell.

Parallel stages (see ``parallel``): wall runs (per wall key, owned by the tile
of its first endpoint) and shells (per volume). Volume connectivity is the
global reduction.

Does not own arrangements (``layers``) or emission (``emit``).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from math import lcm

from sgbproto import identity
from sgbproto.arcs import RealisedArc
from sgbproto.errors import SgbError
from sgbproto.layers import Layer
from sgbproto.levels import LevelStructure
from sgbproto.linking import TAG
from sgbproto.parallel import Runtime
from sgbproto.records import Domain, LevelLabel, SheetOwner
from sgbproto.routes import RoutePolicy, pec_net, surface_epr_rows, surface_group

Loop = tuple[tuple[str, int], ...]  # ((edge id, +1/-1 relative to the edge's canonical direction), ...)


@dataclass
class TopoVertex:
    id: str
    x: float  # grid units; an int for arrangement vertices, a float for the centre point of an arc edge
    y: float
    z: Fraction
    zs: int  # z times the common level denominator (exact integer for volume arithmetic)


@dataclass
class TopoEdge:
    id: str
    kind: str  # "horizontal" | "vertical" | "refinement"
    v0: str
    v1: str
    plane: str
    sources: tuple[str, ...]
    curve: RealisedArc | None = None  # arc edges: the realised circle arc v0 -> v1
    centre_vertex: str | None = None  # arc edges: the planned point carrying the arc's centre


@dataclass
class TopoFace:
    id: str
    kind: str  # "horizontal" | "wall"
    plane: str
    loops: tuple[Loop, ...]  # first loop is the outer boundary
    neg_domain: Domain | None  # domain on the side opposite the face normal
    pos_domain: Domain | None
    owner: SheetOwner | None
    neg_volume: str | None
    pos_volume: str | None
    group: str | None  # the single primary physical group
    planar: bool
    z_range: tuple[Fraction, Fraction]
    twice_xy_area: int = 0
    sources: tuple[str, ...] = ()
    membership: dict[str, float] = field(default_factory=dict)  # entity -> covered fraction of the face (non-exclusive)
    bands: tuple = ()  # inset band labels ((family id, band index), ...) of a horizontal face
    curved: bool = False  # a wall whose bottom/top boundary contains arcs (a cylinder patch, emitted as a filling)
    bulge_volume: float = 0.0  # walls: sum of (arc segment area x height), + when the arcs bulge into the pos side
    refinement_edges: list[str] = field(default_factory=list)  # embedded mesh-size curves (refinement_only)


@dataclass
class Shell:
    faces: tuple[tuple[str, int], ...]  # (face id, +1 if its normal points out of the volume)
    signed_volume: Fraction  # grid^2 * um
    role: str  # "outer" | "inner"


@dataclass
class TopoVolume:
    id: str
    domain: Domain
    cells: tuple[str, ...]
    emitted: bool
    shells: list[Shell] = field(default_factory=list)
    embedded_faces: list[str] = field(default_factory=list)


@dataclass
class Topology:
    route: str
    grid_g: float
    levels: tuple[Fraction, ...]
    vertices: dict[str, TopoVertex]
    edges: dict[str, TopoEdge]
    faces: dict[str, TopoFace]
    volumes: dict[str, TopoVolume]
    groups: dict[str, dict]  # primary groups: name -> {"dim": 2|3, "ids": [...]}; they partition each dimension
    unions: dict[str, list[str]]  # e.g. "PEC::Ground" -> list of primary group names
    surface_epr_rows: dict[str, str]  # Surface-EPR row ("MA::pad::band=2") -> the primary group that carries it
    lineage: dict[str, list[str]]  # edge id -> source carrier ids

    def counts(self) -> dict[str, int]:
        return {
            "vertices": len(self.vertices),
            "edges": len(self.edges),
            "faces": len(self.faces),
            "volumes": len(self.volumes),
            "emitted_volumes": sum(v.emitted for v in self.volumes.values()),
            "groups": len(self.groups),
        }


def assemble(
    levels: LevelStructure,
    slabs: list[Layer],
    level_layers: list[Layer],
    policy: RoutePolicy,
    grid_g: float,
    rt: Runtime | None = None,
) -> Topology:
    """Build the full shared topology from dissolved arrangements."""
    return _Builder(levels, slabs, level_layers, policy, grid_g, rt or Runtime()).run()


class _Builder:
    def __init__(self, levels, slabs, level_layers, policy, grid_g, rt) -> None:
        self.levels, self.slabs, self.A, self.policy, self.g, self.rt = levels, slabs, level_layers, policy, grid_g, rt
        self.zden = lcm(*(z.denominator for z in levels.levels))
        self.vertices: dict[str, TopoVertex] = {}
        self.edges: dict[str, TopoEdge] = {}
        self.faces: dict[str, TopoFace] = {}
        self.lineage: dict[str, list[str]] = {}
        self.volume_of: dict[tuple[int, int], str] = {}
        self.vid_at: list[dict] = []  # per level: grid point -> vertex id
        self.level_index = {z: i for i, z in enumerate(levels.levels)}
        self.cuts: dict[tuple[int, int], set[int]] = defaultdict(set)  # wall column -> level indices it is cut at
        self._domains: dict[str, Domain] = {}

    def run(self) -> Topology:
        rt = self.rt
        with rt.serial("assembly_registry"):
            self._register_level_geometry()
        with rt.serial("volume_connectivity"):
            volumes = self._volumes()
        with rt.serial("assembly_faces"):
            self._horizontal_faces()
        runs = self._wall_runs()
        with rt.serial("assembly_faces"):
            self._wall_faces(runs)
            self._refinement_curves()
            self._prune_unused()
        self._shells(volumes)
        with rt.serial("assembly_groups"):
            groups, unions = self._groups(volumes)
        return Topology(self.policy.name, self.g, self.levels.levels, self.vertices, self.edges, self.faces,
                        volumes, groups, unions, surface_epr_rows(groups), self.lineage)

    # -- vertices and edges ------------------------------------------------------------

    def _register_level_geometry(self) -> None:
        for layer in self.A:
            z = layer.z
            zs = int(z * self.zden)
            ids = {}
            for p, v in zip(layer.arr.vertices, layer.vertex_ids):
                ids[p] = v
                self.vertices[v] = TopoVertex(v, p[0], p[1], z, zs)
            self.vid_at.append(ids)
            for e, eid, src, curve in zip(layer.arr.edges, layer.edge_ids, layer.edge_sources, layer.edge_curve):
                centre = None
                if curve is not None:
                    centre = identity.digest("vc", [eid])
                    self.vertices[centre] = TopoVertex(centre, curve.centre[0], curve.centre[1], z, zs)
                self.edges[eid] = TopoEdge(eid, "horizontal", ids[e.a], ids[e.b], layer.plane, src, curve, centre)
                self.lineage[eid] = list(src)

    # -- volumes -----------------------------------------------------------------------

    def _volumes(self) -> dict[str, TopoVolume]:
        parent: dict[tuple[int, int], tuple[int, int]] = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for k, slab in enumerate(self.slabs):
            for fi in range(1, len(slab.arr.faces)):
                if slab.labels[fi] is not None:
                    find((k, fi))
        for k, layer in enumerate(self.A):
            if k == 0 or k == len(self.A) - 1:
                continue
            for fa in range(1, len(layer.arr.faces)):
                lab: LevelLabel = layer.labels[fa]
                if lab.below is not None and lab.below == lab.above and lab.sheet is None:
                    ra, rb = find((k - 1, layer.below_cell[fa])), find((k, layer.above_cell[fa]))
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)
        groups: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        for node in list(parent):
            groups[find(node)].append(node)
        volumes: dict[str, TopoVolume] = {}
        for members in groups.values():
            members.sort()
            domain = self.slabs[members[0][0]].labels[members[0][1]]
            cells = tuple(sorted(self.slabs[k].face_ids[fi] for k, fi in members))
            vid = identity.volume_id(domain, list(cells))
            volumes[vid] = TopoVolume(vid, domain, cells, self.policy.is_emitted(domain))
            self._domains[vid] = domain
            for node in members:
                self.volume_of[node] = vid
        return volumes

    def _side_volume(self, slab_index: int, cell: int, domain) -> str | None:
        if domain is None or slab_index < 0 or slab_index >= len(self.slabs):
            return None
        return self.volume_of[(slab_index, cell)]

    # -- horizontal faces --------------------------------------------------------------

    def _horizontal_faces(self) -> None:
        for k, layer in enumerate(self.A):
            arr = layer.arr
            for fa in range(1, len(arr.faces)):
                lab: LevelLabel = layer.labels[fa]
                if not lab.needs_face():
                    continue
                loops = tuple(self._cycle_loop(layer, c) for c in arr.face_cycles(fa))
                neg = self._side_volume(k - 1, layer.below_cell[fa], lab.below)
                pos = self._side_volume(k, layer.above_cell[fa], lab.above)
                group = surface_group(lab.below, lab.above, lab.sheet, self.policy, lab.bands)
                boundary = [eid for lp in loops for eid, _ in lp]
                fid = layer.face_ids[fa]
                self.faces[fid] = TopoFace(
                    fid, "horizontal", layer.plane, loops, lab.below, lab.above, lab.sheet, neg, pos, group,
                    True, (layer.z, layer.z), arr.face_twice_area(fa), self._sources(boundary), bands=lab.bands,
                    membership=layer.membership[fa],
                )
                for lp in loops:
                    for eid, _ in lp:
                        e = self.edges[eid]
                        for vid in (e.v0, e.v1):
                            v = self.vertices[vid]
                            self.cuts[(v.x, v.y)].add(k)

    def _cycle_loop(self, layer: Layer, cycle_index: int) -> Loop:
        return tuple((layer.edge_ids[h >> 1], -1 if h & 1 else 1) for h in layer.arr.cycles[cycle_index].half_edges)

    # -- walls (parallel by wall key) --------------------------------------------------

    def _wall_runs(self) -> list[dict]:
        rt = self.rt
        with rt.serial("wall_prep"):
            members = [_tag_members(layer) for layer in self.A]
            keys: dict[tuple, list[tuple[int, int]]] = defaultdict(list)
            for k, slab in enumerate(self.slabs):
                for si, e in enumerate(slab.arr.edges):
                    left, right = slab.arr.he_face[2 * si], slab.arr.he_face[2 * si + 1]
                    if slab.labels[left] != slab.labels[right]:
                        keys[(e.a, e.b)].append((k, si))
            key_list = sorted(keys)
            owners = [rt.owner(key[0]) for key in key_list]
        shared = (self.slabs, self.A, members, [keys[k] for k in key_list], key_list, self.zden,
                  [int(z * self.zden) for z in self.levels.levels], self.levels.levels)
        out = rt.run("wall_runs", _wall_run_job, shared, owners)
        return [run for runs in out for run in runs]

    def _wall_faces(self, runs: list[dict]) -> None:
        for run in runs:
            for p in (run["u"], run["v"]):
                self.cuts[p].update((run["k0"], run["k1"] + 1))
        for run in runs:
            k0, k1 = run["k0"], run["k1"] + 1
            u, v = run["u"], run["v"]
            bottom_layer, top_layer = self.A[k0], self.A[k1]
            loop = (
                *self._column(u, k0, k1, up=True),
                *self._chain(top_layer, run["top"], reverse=False),
                *self._column(v, k0, k1, up=False),
                *self._chain(bottom_layer, run["bottom"], reverse=True),
            )
            for eid, _ in loop:
                e = self.edges[eid]
                if e.kind == "vertical":  # a wall column inherits the lineage of the walls that meet there
                    e.sources = tuple(sorted(set(e.sources) | set(run["sources"])))
                    self.lineage[eid] = list(e.sources)
            dom_l, dom_r = run["dom_l"], run["dom_r"]
            left, right = run["left"], run["right"]
            neg = self._side_volume(k0, right, dom_r)
            pos = self._side_volume(k0, left, dom_l)
            if (neg, pos) != (self._side_volume(run["k1"], run["right_top"], dom_r),
                              self._side_volume(run["k1"], run["left_top"], dom_l)):
                raise SgbError(f"wall {run['id']} joins cells of different volumes")
            z_range = (self.levels.levels[k0], self.levels.levels[k1])
            self.faces[run["id"]] = TopoFace(
                run["id"], "wall", f"wall[{z_range[0]},{z_range[1]}]", (loop,), dom_r, dom_l, None, neg, pos,
                surface_group(dom_r, dom_l, None, self.policy), run["planar"] and not run["curved"], z_range, 0,
                run["sources"], curved=run["curved"], bulge_volume=run["bulge_volume"], membership=run["membership"],
            )

    def _chain(self, layer: Layer, half_edges: list[int], reverse: bool) -> list[tuple[str, int]]:
        seq = [(layer.edge_ids[h >> 1], -1 if h & 1 else 1) for h in half_edges]
        if reverse:
            seq = [(e, -s) for e, s in reversed(seq)]
        return seq

    def _column(self, p, k0: int, k1: int, up: bool) -> list[tuple[str, int]]:
        """Vertical edge segments of column ``p`` between levels k0 and k1 (split at the column's cuts)."""
        stops = sorted(j for j in self.cuts[p] if k0 <= j <= k1)
        segments = []
        for lo, hi in zip(stops, stops[1:]):
            z_lo, z_hi = self.levels.levels[lo], self.levels.levels[hi]
            eid = identity.vertical_edge_id(p, z_lo, z_hi)
            if eid not in self.edges:
                self.edges[eid] = TopoEdge(eid, "vertical", self.vid_at[lo][p], self.vid_at[hi][p],
                                           f"column{p}[{z_lo},{z_hi}]", ())
            segments.append((eid, 1 if up else -1))
        return segments if up else list(reversed(segments))

    def _sources(self, edge_ids: list[str]) -> tuple[str, ...]:
        entities: set[str] = set()
        for eid in edge_ids:
            for s in self.lineage.get(eid, ()):
                entities.add(s.split("/")[0])
        return tuple(sorted(entities))

    def _refinement_curves(self) -> None:
        """Attach ``refinement_only`` curves to the face that contains them (kept as embedded edges)."""
        for k, layer in enumerate(self.A):
            for edge, face_index in layer.dissolved.refinement:
                face = self.faces.get(layer.face_ids[face_index])
                carriers = ", ".join(sorted(edge.carriers))
                where = f"({edge.a[0] * self.g:.4f}, {edge.a[1] * self.g:.4f}) um on {layer.plane}"
                if face is None:
                    raise SgbError(f"refinement curve {carriers} at {where} lies in a cell that has no face "
                                   "(inside a volume): it cannot be embedded and is not dropped silently")
                if not any(v is not None and self.policy.is_emitted(self._domain_of(v)) for v in (face.neg_volume, face.pos_volume)):
                    raise SgbError(f"refinement curve {carriers} at {where} lies in a face of no emitted volume")
                sources = tuple(sorted(c for c in edge.carriers))
                eid = identity.edge_id(layer.plane, edge.a, edge.b, sources)
                ids = []
                for p in (edge.a, edge.b):
                    v = identity.vertex_id(layer.plane, p)
                    self.vertices.setdefault(v, TopoVertex(v, p[0], p[1], layer.z, int(layer.z * self.zden)))
                    ids.append(v)
                self.edges[eid] = TopoEdge(eid, "refinement", ids[0], ids[1], layer.plane, sources)
                self.lineage[eid] = list(sources)
                face.refinement_edges.append(eid)

    def _domain_of(self, volume_id: str):
        return self._domains[volume_id]

    def _prune_unused(self) -> None:
        used_edges = {eid for f in self.faces.values() for lp in f.loops for eid, _ in lp}
        used_edges |= {eid for f in self.faces.values() for eid in f.refinement_edges}
        self.edges = {k: v for k, v in self.edges.items() if k in used_edges}
        used_vertices = {v for e in self.edges.values() for v in (e.v0, e.v1)}
        used_vertices |= {e.centre_vertex for e in self.edges.values() if e.centre_vertex}
        self.vertices = {k: v for k, v in self.vertices.items() if k in used_vertices}
        self.lineage = {k: v for k, v in self.lineage.items() if k in used_edges}

    # -- shells (parallel by volume) -----------------------------------------------------

    def _shells(self, volumes: dict[str, TopoVolume]) -> None:
        rt = self.rt
        with rt.serial("shell_prep"):
            incident: dict[str, list[tuple[str, int]]] = defaultdict(list)
            for fid, f in sorted(self.faces.items()):
                if f.neg_volume is not None and f.neg_volume == f.pos_volume:
                    volumes[f.neg_volume].embedded_faces.append(fid)
                    continue
                if f.neg_volume is not None:
                    incident[f.neg_volume].append((fid, 1))
                if f.pos_volume is not None:
                    incident[f.pos_volume].append((fid, -1))
            vids = sorted(incident)
            owners = [self._volume_owner(incident[v]) for v in vids]
        shared = (self.faces, self.edges, self.vertices, self.zden, [incident[v] for v in vids])
        out = rt.run("shells", _shell_job, shared, owners)
        for vid, shells in zip(vids, out):
            volumes[vid].shells = shells

    def _volume_owner(self, entries: list[tuple[str, int]]) -> int:
        face = self.faces[entries[0][0]]
        v = self.vertices[self.edges[face.loops[0][0][0]].v0]
        return self.rt.owner((v.x, v.y))

    # -- groups --------------------------------------------------------------------------

    def _groups(self, volumes: dict[str, TopoVolume]) -> tuple[dict[str, dict], dict[str, list[str]]]:
        groups: dict[str, dict] = {}
        pec: dict[str, set[str]] = defaultdict(set)
        for fid, f in sorted(self.faces.items()):
            if not any(v is not None and volumes[v].emitted for v in (f.neg_volume, f.pos_volume)):
                continue
            if f.group is None:
                continue
            groups.setdefault(f.group, {"dim": 2, "ids": []})["ids"].append(fid)
            net = pec_net(f.neg_domain, f.pos_domain, f.owner)
            if net is not None:
                pec[f"PEC::{net}"].add(f.group)
        for vid, v in sorted(volumes.items()):
            if v.emitted:
                groups.setdefault(f"VOL::{v.domain.name}", {"dim": 3, "ids": []})["ids"].append(vid)
        unions = {name: sorted(members) for name, members in sorted(pec.items())}
        return dict(sorted(groups.items())), unions


# --------------------------------------------------------------------------- wall jobs


def _tag_members(layer: Layer) -> dict:
    """Level edges carrying each slab-edge tag, and the number of edges at each vertex."""
    tags: dict[str, list[int]] = defaultdict(list)
    degree = [0] * len(layer.arr.vertices)
    for ei, e in enumerate(layer.arr.edges):
        a, b = layer.arr.edge_ends[ei]
        degree[a] += 1
        degree[b] += 1
        for c in e.carriers:
            if c.startswith(TAG):
                tags[c].append(ei)
    return {"tags": tags, "degree": degree}


def _trace(layer: Layer, tag_edges: list[int], start, end) -> list[int]:
    """Half-edges of the level path that carries a slab-edge tag, from ``start`` to ``end``."""
    arr = layer.arr
    at: dict = defaultdict(list)
    for ei in tag_edges:
        e = arr.edges[ei]
        at[e.a].append(ei)
        at[e.b].append(ei)
    cur, path, used = start, [], set()
    while cur != end:
        options = [ei for ei in at[cur] if ei not in used]
        if len(options) != 1:
            raise SgbError(f"slab edge chain is ambiguous or broken at {cur}")
        ei = options[0]
        used.add(ei)
        e = arr.edges[ei]
        path.append(2 * ei if e.a == cur else 2 * ei + 1)
        cur = e.b if e.a == cur else e.a
    if len(used) != len(tag_edges):
        raise SgbError("slab edge chain does not use all tagged edges")
    return path


def _level_cuts_wall(layer: Layer, degree: list[int], below: list[int], above: list[int]) -> bool:
    """True when the level ``layer`` must cut the wall whose chains just below/above it are given."""
    if {h >> 1 for h in below} != {h >> 1 for h in above}:
        return True
    arr = layer.arr
    last = len(below) - 1
    for i, h in enumerate(below):
        head = arr.edge_ends[h >> 1][0 if h & 1 else 1]
        if i < last and degree[head] != 2:
            return True  # something else meets the wall line at this vertex
        if any(lab is not None and lab.needs_face() for lab in (layer.labels[arr.he_face[h]], layer.labels[arr.he_face[h ^ 1]])):
            return True  # a sheet face lies along the wall line
    return False


def _wall_run_job(shared, items):
    slabs, A, members, entries_by_key, key_list, zden, zs, levels = shared
    out = []
    for i in items:
        (u, v), entries = key_list[i], sorted(entries_by_key[i])
        runs, current = [], None
        for k, si in entries:
            slab = slabs[k]
            tag = TAG + slab.edge_ids[si]
            bottom = _trace(A[k], members[k]["tags"][tag], u, v)
            top = _trace(A[k + 1], members[k + 1]["tags"][tag], u, v)
            left, right = slab.arr.he_face[2 * si], slab.arr.he_face[2 * si + 1]
            dom = (slab.labels[left], slab.labels[right])
            if current is not None and current["k1"] == k - 1 and current["dom"] == dom and not _level_cuts_wall(
                A[k], members[k]["degree"], current["top"], bottom
            ):
                current.update(k1=k, top=top, left_top=left, right_top=right, last_edge=si)
                continue
            if current is not None:
                runs.append(current)
            current = {"k0": k, "k1": k, "dom": dom, "bottom": bottom, "top": top, "left": left, "right": right,
                       "left_top": left, "right_top": right, "last_edge": si}
        if current is not None:
            runs.append(current)
        out.append([_finish_run(r, u, v, slabs, A, zs, levels) for r in runs])
    return out


def _finish_run(r: dict, u, v, slabs, A, zs, levels) -> dict:
    bottom_layer, top_layer = A[r["k0"]], A[r["k1"] + 1]
    z_lo, z_hi = levels[r["k0"]], levels[r["k1"] + 1]
    bottom_ids = [bottom_layer.edge_ids[h >> 1] for h in r["bottom"]]
    top_ids = [top_layer.edge_ids[h >> 1] for h in r["top"]]
    dom_l, dom_r = r["dom"]
    fid = identity.face_id("wall", f"wall[{z_lo},{z_hi}]", [dom_r, dom_l, list(u), list(v)], bottom_ids + top_ids)
    height = float(z_hi - z_lo)
    bulge, curved = 0.0, False
    for layer, half_edges in ((bottom_layer, r["bottom"]),):  # top and bottom carry the same arcs
        for h in half_edges:
            curve = layer.edge_curve[h >> 1]
            if curve is not None:
                curved = True
                toward_left = curve.bulge_left if h % 2 == 0 else not curve.bulge_left
                bulge += (1 if toward_left else -1) * curve.segment_area * height
    pts = [(*u, zs[r["k0"]]), (*u, zs[r["k1"] + 1])]
    pts += [(*top_layer.arr.origin(h), zs[r["k1"] + 1]) for h in r["top"]] + [(*v, zs[r["k1"] + 1]), (*v, zs[r["k0"]])]
    pts += [(*bottom_layer.arr.origin(h), zs[r["k0"]]) for h in r["bottom"][:0:-1]]
    sources: set[str] = set()
    for layer, ids in ((bottom_layer, r["bottom"]), (top_layer, r["top"])):
        for h in ids:
            for s in layer.edge_sources[h >> 1]:
                sources.add(s.split("/")[0])
    return {"id": fid, "membership": _wall_membership(bottom_layer, r["bottom"]), "u": u, "v": v, "k0": r["k0"], "k1": r["k1"], "dom_l": dom_l, "dom_r": dom_r,
            "bottom": r["bottom"], "top": r["top"], "left": r["left"], "right": r["right"],
            "left_top": r["left_top"], "right_top": r["right_top"], "planar": _is_planar(pts),
            "curved": curved, "bulge_volume": bulge,
            "sources": tuple(sorted(sources))}


def _wall_membership(layer: Layer, half_edges: list[int]) -> dict[str, float]:
    """A wall has no footprint: the share of an entity is the fraction of the wall's bottom boundary length carried
    by edges that come from that entity (non-exclusive when edges have several sources)."""
    lengths: dict[str, float] = {}
    total = 0.0
    for h in half_edges:
        e = layer.arr.edges[h >> 1]
        curve = layer.edge_curve[h >> 1]
        chord = math.hypot(e.b[0] - e.a[0], e.b[1] - e.a[1])
        length = chord if curve is None else curve.radius * 2 * math.asin(min(1.0, chord / (2 * curve.radius)))
        total += length
        for entity in {s.split("/")[0] for s in layer.edge_sources[h >> 1]}:
            lengths[entity] = lengths.get(entity, 0.0) + length
    return {e: round(v / total, 9) for e, v in sorted(lengths.items())} if total else {}


def _is_planar(pts) -> bool:
    if len(pts) <= 3:
        return True
    p0 = pts[0]
    n = None
    for i in range(1, len(pts) - 1):
        a = _sub(pts[i], p0)
        for j in range(i + 1, len(pts)):
            c = _crossv(a, _sub(pts[j], p0))
            if any(c):
                n = c
                break
        if n:
            break
    return n is None or all(_dot(n, _sub(p, p0)) == 0 for p in pts)


# --------------------------------------------------------------------------- shell job


def _shell_job(shared, items):
    faces, edges, vertices, zden, entries_per_volume = shared
    return [_volume_shells(faces, edges, vertices, zden, entries_per_volume[i]) for i in items]


def _volume_shells(faces, edges, vertices, zden, entries) -> list[Shell]:
    by_edge: dict[str, list[int]] = defaultdict(list)
    for i, (fid, _) in enumerate(entries):
        for lp in faces[fid].loops:
            for eid, _ in lp:
                by_edge[eid].append(i)
    parent = list(range(len(entries)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for members in by_edge.values():
        for m in members[1:]:
            ra, rb = find(members[0]), find(m)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    comps: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for i, ent in enumerate(entries):
        comps[find(i)].append(ent)
    shells = []
    for ents in comps.values():
        six_v = sum(_six_volume(faces, edges, vertices, fid, s) for fid, s in ents)
        # Arc walls: the chord polyhedron is exact; the cylinder patches add/remove segment-area x height.
        arc_term = sum(s * faces[fid].bulge_volume for fid, s in ents if faces[fid].curved)
        volume = Fraction(six_v, 6 * zden) + (Fraction(arc_term) if arc_term else 0)
        shells.append(Shell(tuple(sorted(ents)), volume, "outer" if six_v > 0 else "inner"))
    shells.sort(key=lambda s: (s.role != "outer", s.faces))
    return shells


def _six_volume(faces, edges, vertices, fid: str, sign: int) -> int:
    """Six times the divergence-theorem volume contribution of an outward-oriented face (z scaled to ints)."""
    f = faces[fid]
    if f.kind == "horizontal":
        z = vertices[edges[f.loops[0][0][0]].v0].zs
        return sign * z * f.twice_xy_area
    pts = []
    for eid, esign in f.loops[0]:
        e = edges[eid]
        w = vertices[e.v0 if esign > 0 else e.v1]
        pts.append((w.x, w.y, w.zs))
    if sign < 0:
        pts.reverse()
    return sum(_det(pts[0], pts[i], pts[i + 1]) for i in range(1, len(pts) - 1))


def shell_checks(faces, edges, ents) -> tuple[list[str], int]:
    """Every edge used twice with opposite orientation; Euler characteristic 2 - 2*genus (or even for voids)."""
    uses: dict[str, list[int]] = defaultdict(list)
    vertices: set[str] = set()
    chi = 0
    for fid, sign in ents:
        f = faces[fid]
        for loop in f.loops:
            for eid, esign in loop:
                uses[eid].append(esign * sign)
                vertices.update((edges[eid].v0, edges[eid].v1))
        chi += 1 - (len(f.loops) - 1)
    issues = []
    for eid, signs in uses.items():
        if len(signs) != 2:
            issues.append(f"edge {eid} used {len(signs)} times")
        elif signs[0] + signs[1] != 0:
            issues.append(f"edge {eid} not used with opposite orientation")
    chi += len(vertices) - len(uses)
    return issues, chi


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _crossv(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _det(a, b, c) -> int:
    return _dot(a, _crossv(b, c))
