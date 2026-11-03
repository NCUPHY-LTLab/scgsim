"""Topology assembly from the labelled arrangements (design brief section 2, stage S6).

Owns turning the dissolved slab and level arrangements into one shared, conformal
boundary representation -- *without any CAD kernel*:

* **vertices and edges**: every vertex and edge of a level arrangement, plus the
  vertical edges of the wall columns, plus the pieces a curved wall splits its arcs into;
* **horizontal faces**: cells of ``A_i`` whose level label needs a face (the domains below
  and above differ, or a sheet is present);
* **walls**: one vertical face per wall run (``walls``), one cylinder per piece on an arc
  (``curved_walls``);
* **vertical edges**: a wall column ``(x, y)`` is cut only at the levels where a wall run
  starts or ends or a horizontal face has a vertex, so neighbouring faces share whole edges;
* **volumes**: connected components of slab cells joined across level cells whose
  below and above domains are equal and that carry no sheet;
* **shells**: the faces around each volume, grouped into closed surfaces (``shells``).

A face with the same volume on both sides is a sheet inside a volume (*embedded*); it is
listed on the volume but is not part of its shell.

The result is route independent.  Which volumes are emitted and which primary group a face
joins is decided afterwards (``routes``).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from math import lcm

from scgsim.geometry.sgb import identity
from scgsim.geometry.sgb.arrangement.arcs import RealisedArc, realise
from scgsim.geometry.sgb.assembly.curved_walls import (
    EdgeAt,
    is_curved,
    plan_curved_splits,
)
from scgsim.geometry.sgb.assembly.lineage import CarrierEntities
from scgsim.geometry.sgb.assembly.records import (
    Loop,
    TopoEdge,
    TopoFace,
    Topology,
    TopoVertex,
    TopoVolume,
)
from scgsim.geometry.sgb.assembly.shells import is_planar, shell_job
from scgsim.geometry.sgb.assembly.walls import WallRun, find_wall_runs
from scgsim.geometry.sgb.derivation.labelled import LabelledArrangements
from scgsim.geometry.sgb.derivation.layer import Layer
from scgsim.geometry.sgb.errors import SgbError
from scgsim.geometry.sgb.parallel import Runtime
from scgsim.geometry.sgb.plane import Point
from scgsim.geometry.sgb.records.carriers import ArcCarrier
from scgsim.geometry.sgb.records.labels import Domain, LevelLabel

Entry = tuple[str, int]  # (edge id, +1 / -1 relative to the edge's v0 -> v1 direction)


@dataclass(frozen=True)
class Link:
    """One edge of a chain in travel order."""

    edge_id: str
    sign: int
    start: Point
    end: Point


def assemble_topology(labelled: LabelledArrangements, rt: Runtime) -> Topology:
    """Build the full shared topology from the labelled arrangements."""
    return _Assembly(labelled, rt).run()


class _Assembly:
    """The state that travels through the assembly steps."""

    def __init__(self, labelled: LabelledArrangements, rt: Runtime) -> None:
        self.labelled = labelled
        self.rt = rt
        self.levels = labelled.levels
        self.slabs = labelled.slabs
        self.A = labelled.level_layers
        self.grid = labelled.catalogue.grid
        self.entities = CarrierEntities(labelled.catalogue)
        self.zden = lcm(*(z.denominator for z in self.levels.levels))
        self.vertices: dict[str, TopoVertex] = {}
        self.edges: dict[str, TopoEdge] = {}
        self.faces: dict[str, TopoFace] = {}
        self.lineage: dict[str, list[str]] = {}
        self.volume_of: dict[tuple[int, int], str] = {}
        self.domain_of: dict[str, Domain] = {}
        self.vid_at: list[
            dict[Point, str]
        ] = []  # per level: lattice point -> vertex id
        self.cuts: dict[Point, set[int]] = defaultdict(
            set
        )  # wall column -> level indices it is cut at
        self.piece_ids: dict[
            EdgeAt, list[str]
        ] = {}  # split level edge -> its piece edge ids, from a to b
        self.piece_points: dict[EdgeAt, list[Point]] = {}

    def run(self) -> Topology:
        """Run the assembly steps in order."""
        rt = self.rt
        with rt.serial("assembly_registry"):
            self._register_level_geometry()
        with rt.serial("volume_connectivity"):
            volumes = self._volumes()
        runs = find_wall_runs(self.slabs, self.A, rt)
        with rt.serial("assembly_faces"):
            self._register_splits(plan_curved_splits(self.A, runs))
            self._horizontal_faces()
            self._wall_faces(runs)
            self._refinement_curves()
            self._prune_unused()
        self._shells(volumes)
        return Topology(
            self.grid.size_um,
            self.levels.levels,
            self.vertices,
            self.edges,
            self.faces,
            volumes,
            self.lineage,
        )

    # -- vertices and edges ------------------------------------------------------------

    def _register_level_geometry(self) -> None:
        for layer in self.A:
            zs = int(layer.level_z * self.zden)
            ids: dict[Point, str] = {}
            for p, vertex_id in zip(layer.arr.vertices, layer.vertex_ids, strict=True):
                ids[p] = vertex_id
                self.vertices[vertex_id] = TopoVertex(
                    vertex_id, p[0], p[1], layer.level_z, zs
                )
            self.vid_at.append(ids)
            for e, eid, sources, curve in zip(
                layer.arr.edges,
                layer.edge_ids,
                layer.edge_sources,
                layer.edge_curve,
                strict=True,
            ):
                self._add_horizontal_edge(
                    layer, eid, ids[e.a], ids[e.b], sources, curve
                )

    def _add_horizontal_edge(  # noqa: PLR0913 - an edge's identity, ends, sources and curve
        self,
        layer: Layer,
        eid: str,
        v0: str,
        v1: str,
        sources: tuple[str, ...],
        curve: RealisedArc | None,
    ) -> None:
        centre = None
        if curve is not None:
            centre = identity.digest("vc", [eid])
            zs = int(layer.level_z * self.zden)
            self.vertices[centre] = TopoVertex(
                centre, curve.centre[0], curve.centre[1], layer.level_z, zs
            )
        self.edges[eid] = TopoEdge(
            eid, "horizontal", v0, v1, layer.plane, sources, curve, centre
        )
        self.lineage[eid] = list(sources)

    def _vertex_at(self, level: int, p: Point) -> str:
        """The vertex id of a lattice point on a level plane, created when the arrangement has none there."""
        ids = self.vid_at[level]
        if p not in ids:
            layer = self.A[level]
            vertex_id = identity.vertex_id(layer.plane, p)
            self.vertices[vertex_id] = TopoVertex(
                vertex_id, p[0], p[1], layer.level_z, int(layer.level_z * self.zden)
            )
            ids[p] = vertex_id
        return ids[p]

    def _register_splits(self, splits: dict[EdgeAt, tuple[Point, ...]]) -> None:
        """Replace each planned arc edge by its pieces: new vertices, edges and realised arcs."""
        arcs = self.labelled.arcs
        for (level, ei), interior in sorted(splits.items()):
            layer = self.A[level]
            edge = layer.arr.edges[ei]
            carriers = sorted(edge.carriers)
            arc: ArcCarrier = arcs[carriers[0]]
            root = arcs.get(arc.root_id, arc)
            points = [edge.a, *interior, edge.b]
            sources = layer.edge_sources[ei]
            ids = []
            for p, q in zip(points, points[1:], strict=False):
                eid = identity.edge_id(layer.plane, p, q, sources)
                self._add_horizontal_edge(
                    layer,
                    eid,
                    self._vertex_at(level, p),
                    self._vertex_at(level, q),
                    sources,
                    realise(arc, root, p, q),
                )
                ids.append(eid)
            self.piece_ids[(level, ei)] = ids
            self.piece_points[(level, ei)] = points

    # -- volumes -----------------------------------------------------------------------

    def _volumes(self) -> dict[str, TopoVolume]:
        parent: dict[tuple[int, int], tuple[int, int]] = {}

        def find(x: tuple[int, int]) -> tuple[int, int]:
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
                label: LevelLabel = layer.labels[fa]
                if (
                    label.below is not None
                    and label.below == label.above
                    and label.sheet is None
                ):
                    below = find((k - 1, layer.below_cell[fa]))
                    above = find((k, layer.above_cell[fa]))
                    if below != above:
                        parent[max(below, above)] = min(below, above)
        groups: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        for node in list(parent):
            groups[find(node)].append(node)
        volumes: dict[str, TopoVolume] = {}
        for members in groups.values():
            members.sort()
            domain = self.slabs[members[0][0]].labels[members[0][1]]
            cells = tuple(sorted(self.slabs[k].face_ids[fi] for k, fi in members))
            vid = identity.volume_id(domain, list(cells))
            volumes[vid] = TopoVolume(vid, domain, cells)
            self.domain_of[vid] = domain
            for node in members:
                self.volume_of[node] = vid
        return volumes

    def _side_volume(
        self, slab_index: int, cell: int, domain: Domain | None
    ) -> str | None:
        if domain is None or slab_index < 0 or slab_index >= len(self.slabs):
            return None
        return self.volume_of[(slab_index, cell)]

    # -- chains and loops --------------------------------------------------------------

    def _links(self, level: int, half_edges: tuple[int, ...] | list[int]) -> list[Link]:
        """The edges of a chain of half-edges in travel order, split edges expanded into their pieces."""
        layer = self.A[level]
        links: list[Link] = []
        for h in half_edges:
            ei = h >> 1
            forward = h % 2 == 0
            if (level, ei) in self.piece_ids:
                ids = self.piece_ids[(level, ei)]
                points = self.piece_points[(level, ei)]
                spans = list(zip(ids, points, points[1:], strict=False))
                if forward:
                    links.extend(Link(eid, 1, p, q) for eid, p, q in spans)
                else:
                    links.extend(Link(eid, -1, q, p) for eid, p, q in reversed(spans))
            else:
                links.append(
                    Link(
                        layer.edge_ids[ei],
                        1 if forward else -1,
                        layer.arr.origin(h),
                        layer.arr.target(h),
                    )
                )
        return links

    def _cycle_loop(self, level: int, cycle_index: int) -> Loop:
        layer = self.A[level]
        links = self._links(level, layer.arr.cycles[cycle_index].half_edges)
        return tuple((link.edge_id, link.sign) for link in links)

    # -- horizontal faces --------------------------------------------------------------

    def _horizontal_faces(self) -> None:
        for k, layer in enumerate(self.A):
            arr = layer.arr
            for fa in range(1, len(arr.faces)):
                label: LevelLabel = layer.labels[fa]
                if not label.needs_face():
                    continue
                loops = tuple(self._cycle_loop(k, c) for c in arr.face_cycles(fa))
                neg = self._side_volume(k - 1, layer.below_cell[fa], label.below)
                pos = self._side_volume(k, layer.above_cell[fa], label.above)
                boundary = [eid for loop in loops for eid, _ in loop]
                area = arr.face_twice_area(fa)
                if self._uses_split_edge(k, fa):
                    area = self._loops_twice_area(loops)
                face_id = layer.face_ids[fa]
                self.faces[face_id] = TopoFace(
                    face_id,
                    "horizontal",
                    layer.plane,
                    loops,
                    label.below,
                    label.above,
                    label.sheet,
                    neg,
                    pos,
                    True,
                    (layer.level_z, layer.level_z),
                    area,
                    self._sources(boundary),
                    bands=label.bands,
                    membership=layer.membership[fa],
                )
                for loop in loops:
                    for eid, _ in loop:
                        edge = self.edges[eid]
                        for vid in (edge.v0, edge.v1):
                            vertex = self.vertices[vid]
                            self.cuts[(int(vertex.x), int(vertex.y))].add(k)

    def _uses_split_edge(self, level: int, face: int) -> bool:
        arr = self.A[level].arr
        return any(
            (level, h >> 1) in self.piece_ids
            for c in arr.face_cycles(face)
            for h in arr.cycles[c].half_edges
        )

    def _loops_twice_area(self, loops: tuple[Loop, ...]) -> int:
        """Twice the signed area enclosed by loops of (possibly split) edges, from their vertices."""
        total = 0
        for loop in loops:
            points = []
            for eid, sign in loop:
                edge = self.edges[eid]
                vertex = self.vertices[edge.v0 if sign > 0 else edge.v1]
                points.append((int(vertex.x), int(vertex.y)))
            total += sum(
                x0 * y1 - x1 * y0
                for (x0, y0), (x1, y1) in zip(
                    points, [*points[1:], points[0]], strict=True
                )
            )
        return total

    def _sources(self, edge_ids: list[str]) -> tuple[str, ...]:
        carriers = {s for eid in edge_ids for s in self.lineage.get(eid, ())}
        return self.entities.entities_of(list(carriers))

    # -- walls -------------------------------------------------------------------------

    def _wall_faces(self, runs: list[WallRun]) -> None:
        pieces = [piece for run in runs for piece in self._wall_pieces(run)]
        for piece in pieces:
            for p in (piece.u, piece.v):
                self.cuts[p].update((piece.run.k0, piece.run.k1 + 1))
        for piece in pieces:
            self._add_wall(piece)

    def _wall_pieces(self, run: WallRun) -> list[_WallPiece]:
        """One piece for a straight run, one cylinder piece per arc edge for a curved run."""
        bottom = self._links(run.k0, run.bottom)
        top = self._links(run.k1 + 1, run.top)
        if not is_curved(run, self.A):
            return [_WallPiece(run, run.u, run.v, bottom, top)]
        if len(bottom) != len(top) or any(
            b.start != t.start or b.end != t.end
            for b, t in zip(bottom, top, strict=True)
        ):
            raise SgbError(
                f"curved wall from {run.u} to {run.v} has different vertices at its bottom and top "
                "after splitting: the cylinder cannot be formed"
            )
        return [
            _WallPiece(run, b.start, b.end, [b], [t])
            for b, t in zip(bottom, top, strict=True)
        ]

    def _add_wall(self, piece: _WallPiece) -> None:
        run = piece.run
        k0, k1 = run.k0, run.k1 + 1
        loop: Loop = (
            *self._column(piece.u, k0, k1, up=True),
            *((link.edge_id, link.sign) for link in piece.top),
            *self._column(piece.v, k0, k1, up=False),
            *((link.edge_id, -link.sign) for link in reversed(piece.bottom)),
        )
        sources = self._sources([link.edge_id for link in (*piece.bottom, *piece.top)])
        for eid, _ in loop:
            edge = self.edges[eid]
            if (
                edge.kind == "vertical"
            ):  # a wall column inherits the lineage of the walls that meet there
                edge.sources = tuple(sorted(set(edge.sources) | set(sources)))
                self.lineage[eid] = list(edge.sources)
        neg = self._side_volume(k0, run.right, run.dom_r)
        pos = self._side_volume(k0, run.left, run.dom_l)
        if (neg, pos) != (
            self._side_volume(run.k1, run.right_top, run.dom_r),
            self._side_volume(run.k1, run.left_top, run.dom_l),
        ):
            raise SgbError(
                f"wall from {run.u} to {run.v} joins cells of different volumes"
            )
        z_lo, z_hi = self.levels.levels[k0], self.levels.levels[k1]
        plane = f"wall[{z_lo},{z_hi}]"
        bottom_ids = [link.edge_id for link in piece.bottom]
        top_ids = [link.edge_id for link in piece.top]
        face_id = identity.face_id(
            "wall",
            plane,
            [run.dom_r, run.dom_l, list(piece.u), list(piece.v)],
            bottom_ids + top_ids,
        )
        curved = is_curved(run, self.A)
        self.faces[face_id] = TopoFace(
            face_id,
            "wall",
            plane,
            (loop,),
            run.dom_r,
            run.dom_l,
            None,
            neg,
            pos,
            curved or self._planar(piece),
            (z_lo, z_hi),
            0,
            sources,
            cylinder=(bottom_ids[0], top_ids[0]) if curved else None,
            bulge_volume=self._bulge(piece, float(z_hi - z_lo)) if curved else 0.0,
            membership=self._wall_membership(piece),
        )

    def _column(self, p: Point, k0: int, k1: int, *, up: bool) -> list[Entry]:
        """Vertical edge segments of column ``p`` between levels k0 and k1 (split at the column's cuts)."""
        stops = sorted(j for j in self.cuts[p] if k0 <= j <= k1)
        segments: list[Entry] = []
        levels = self.levels.levels
        for lo, hi in zip(stops, stops[1:], strict=False):
            eid = identity.vertical_edge_id(p, levels[lo], levels[hi])
            if eid not in self.edges:
                self.edges[eid] = TopoEdge(
                    eid,
                    "vertical",
                    self._vertex_at(lo, p),
                    self._vertex_at(hi, p),
                    f"column{p}[{levels[lo]},{levels[hi]}]",
                    (),
                )
            segments.append((eid, 1 if up else -1))
        return segments if up else list(reversed(segments))

    def _planar(self, piece: _WallPiece) -> bool:
        run = piece.run
        zs = [int(z * self.zden) for z in self.levels.levels]
        top_z, bottom_z = zs[run.k1 + 1], zs[run.k0]
        points = [(*piece.u, bottom_z), (*piece.u, top_z)]
        points += [(*link.end, top_z) for link in piece.top]
        points += [(*link.start, bottom_z) for link in piece.bottom[1:]]
        return is_planar(points)

    def _bulge(self, piece: _WallPiece, height: float) -> float:
        """Sum of (arc segment area x height), positive when the arcs bulge into the positive side."""
        total = 0.0
        for link in piece.bottom:
            curve = self.edges[link.edge_id].curve
            if curve is not None:
                toward_left = (
                    curve.bulge_left if link.sign > 0 else not curve.bulge_left
                )
                total += (1 if toward_left else -1) * curve.segment_area * height
        return total

    def _wall_membership(self, piece: _WallPiece) -> dict[str, float]:
        """A wall has no footprint: an entity's share is the fraction of the bottom boundary length it contributes."""
        lengths: dict[str, float] = {}
        total = 0.0
        for link in piece.bottom:
            edge = self.edges[link.edge_id]
            chord = math.hypot(link.end[0] - link.start[0], link.end[1] - link.start[1])
            curve = edge.curve
            length = (
                chord
                if curve is None
                else curve.radius * 2 * math.asin(min(1.0, chord / (2 * curve.radius)))
            )
            total += length
            for entity in self.entities.entities_of(list(edge.sources)):
                lengths[entity] = lengths.get(entity, 0.0) + length
        return (
            {e: round(v / total, 9) for e, v in sorted(lengths.items())}
            if total
            else {}
        )

    # -- refinement curves -------------------------------------------------------------

    def _refinement_curves(self) -> None:
        """Attach ``refinement_only`` curves to the face that contains them (kept as embedded edges)."""
        for layer in self.A:
            for edge, face_index in layer.dissolved.refinement:
                face = self.faces.get(layer.face_ids[face_index])
                carriers = ", ".join(sorted(edge.carriers))
                where = f"({self.grid.to_um(edge.a[0]):.4f}, {self.grid.to_um(edge.a[1]):.4f}) um on {layer.plane}"
                if face is None:
                    raise SgbError(
                        f"refinement curve {carriers} at {where} lies in a cell that has no face (inside a "
                        "volume): it cannot be embedded and is not dropped silently"
                    )
                sources = tuple(sorted(edge.carriers))
                eid = identity.edge_id(layer.plane, edge.a, edge.b, sources)
                ids = []
                for p in (edge.a, edge.b):
                    vid = identity.vertex_id(layer.plane, p)
                    self.vertices.setdefault(
                        vid,
                        TopoVertex(
                            vid,
                            p[0],
                            p[1],
                            layer.level_z,
                            int(layer.level_z * self.zden),
                        ),
                    )
                    ids.append(vid)
                self.edges[eid] = TopoEdge(
                    eid, "refinement", ids[0], ids[1], layer.plane, sources
                )
                self.lineage[eid] = list(sources)
                face.refinement_edges.append(eid)

    def _prune_unused(self) -> None:
        used = {eid for f in self.faces.values() for loop in f.loops for eid, _ in loop}
        used |= {eid for f in self.faces.values() for eid in f.refinement_edges}
        self.edges = {k: v for k, v in self.edges.items() if k in used}
        used_vertices = {v for e in self.edges.values() for v in (e.v0, e.v1)}
        used_vertices |= {
            e.centre_vertex for e in self.edges.values() if e.centre_vertex
        }
        self.vertices = {k: v for k, v in self.vertices.items() if k in used_vertices}
        self.lineage = {k: v for k, v in self.lineage.items() if k in used}

    # -- shells ------------------------------------------------------------------------

    def _shells(self, volumes: dict[str, TopoVolume]) -> None:
        rt = self.rt
        with rt.serial("shell_prep"):
            incident: dict[str, list[Entry]] = defaultdict(list)
            for fid, face in sorted(self.faces.items()):
                if face.neg_volume is not None and face.neg_volume == face.pos_volume:
                    volumes[face.neg_volume].embedded_faces.append(fid)
                    continue
                if face.neg_volume is not None:
                    incident[face.neg_volume].append((fid, 1))
                if face.pos_volume is not None:
                    incident[face.pos_volume].append((fid, -1))
            vids = sorted(incident)
            owners = [self._volume_owner(incident[v]) for v in vids]
        shared = (
            self.faces,
            self.edges,
            self.vertices,
            self.zden,
            [incident[v] for v in vids],
        )
        out = rt.run("shells", shell_job, shared, owners)
        for vid, shells in zip(vids, out, strict=True):
            volumes[vid].shells = shells

    def _volume_owner(self, entries: list[Entry]) -> int:
        face = self.faces[entries[0][0]]
        vertex = self.vertices[self.edges[face.loops[0][0][0]].v0]
        return self.rt.owner((int(vertex.x), int(vertex.y)))


@dataclass
class _WallPiece:
    """A wall run, or one cylinder piece of a curved run, with its bottom and top chains in travel order."""

    run: WallRun
    u: Point
    v: Point
    bottom: list[Link]
    top: list[Link]
