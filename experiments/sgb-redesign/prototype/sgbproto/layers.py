"""Slab and level arrangements (brief section 2, stages S4 and S5), stage by stage.

Owns building, for one model, every slab arrangement ``S_i`` (curves of the
prisms active in the slab) and every level arrangement ``A_i`` (curves of the
dissolved ``S_{i-1}``, ``S_i``, the sheets on the level and the curve families
targeting it) -- each as: snap-rounded edges, half-edge faces, exact labels,
dissolution, and content-addressed ids.

The work is organised *stage by stage over all layers* so each parallel stage
(labelling, atomic id hashing) is one fork/join over per-tile item lists, see
``parallel``. Slab arrangements do not depend on each other, and a level only
needs the two slabs next to it, so all slabs are built first, then all levels.

A level arrangement receives each slab edge as a carrier tagged ``~<edge id>``;
that tag is how assembly later finds the chain of level edges that covers a
slab edge (the wall bottom and top). Source lineage of every edge is the
expansion of those tags back to polygon-edge carriers.

Does not own topology assembly (``assembly``) or the tile runtime (``parallel``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from sgbproto import identity
from sgbproto.arcs import RealisedArc, derived_arc, realise
from sgbproto.arrangement import Arrangement, build_arrangement, interior_point
from sgbproto.dissolve import Dissolved, FamilyPolicies, dissolve
from sgbproto.labels import PrismSet, resolve_domain
from sgbproto.levels import LevelStructure
from sgbproto.linking import TAG, SlabEdges, link_faces
from sgbproto.parallel import Runtime
from sgbproto.records import MEMBER_ONLY, ArcCarrier, LevelLabel, ModelSpec, base_carrier_id
from sgbproto.snap_rounding import Segment, snap_round


@dataclass
class Layer:
    """One dissolved, labelled and identified planar arrangement."""

    kind: str  # "slab" | "level"
    index: int
    plane: str  # content key, e.g. "slab[0,100]" or "level[0]"
    z: Fraction | tuple[Fraction, Fraction]
    dissolved: Dissolved
    cells: PrismSet | None  # prisms for slab layers
    vertex_ids: list[str] = field(default_factory=list)
    edge_ids: list[str] = field(default_factory=list)
    edge_sources: list[tuple[str, ...]] = field(default_factory=list)
    face_ids: list[str] = field(default_factory=list)
    membership: list[dict[str, float]] = field(default_factory=list)  # faces/cells: entity -> covered area fraction
    removed_edges: list = field(default_factory=list)  # slab layers: raw edges that dissolved (cut cells at levels)
    edge_curve: list[RealisedArc | None] = field(default_factory=list)  # realised arc of each arc edge, else None
    below_cell: list[int] = field(default_factory=list)  # level layers: slab cell below each face
    above_cell: list[int] = field(default_factory=list)  # level layers: slab cell above each face
    raw: dict = field(default_factory=dict)  # pre-dissolution measurements for validation
    stats: dict = field(default_factory=dict)

    @property
    def arr(self) -> Arrangement:
        return self.dissolved.arrangement

    @property
    def labels(self) -> list:
        return self.dissolved.labels


def build_layers(
    spec: ModelSpec, levels: LevelStructure, rt: Runtime
) -> tuple[list[Layer], list[Layer], dict[str, ArcCarrier]]:
    """All slab layers, then all level layers, plus the arc registry (source arcs and realised slab-edge arcs)."""
    arcs = spec.arc_registry()
    slabs = _build_slabs(spec, levels, rt, arcs)
    return slabs, _build_levels(spec, levels, slabs, rt, arcs), arcs


# --------------------------------------------------------------------------- slabs


def _build_slabs(spec: ModelSpec, levels: LevelStructure, rt: Runtime, arcs: dict[str, ArcCarrier]) -> list[Layer]:
    engine = rt.engine()
    raws, stats, sets = [], [], []
    for k in range(levels.slab_count):
        active = levels.slab_active[k]
        segments = [Segment(c.p, c.q, frozenset([c.id])) for i in active for c in spec.prisms[i].carriers()]
        with rt.noding("noding_driver"):
            snapped = snap_round(segments, engine)
        with rt.serial("dcel_faces"):
            raws.append(build_arrangement(snapped.edges))
        stats.append({"segments": len(segments), "hot_pixels": snapped.hot_pixels, "snap_rounds": snapped.rounds})
        sets.append(PrismSet(spec.prisms, active))

    flat = [(k, fi) for k, arr in enumerate(raws) for fi in range(1, len(arr.faces))]
    owners = [rt.owner(_anchor(raws[k], fi)) for k, fi in flat]
    out = rt.run("labelling", _label_slab_job, (raws, sets, flat, spec.prisms), owners)
    labels = [[None] for _ in raws]
    hits: list[list[list[int]]] = [[[]] for _ in raws]
    for (k, _), (domain, inside) in zip(flat, out):
        labels[k].append(domain)
        hits[k].append(inside)

    layers = []
    with rt.serial("dissolution"):
        for k, raw_arr in enumerate(raws):
            z = (levels.levels[k], levels.levels[k + 1])
            dis = dissolve(raw_arr, labels[k])
            layer = Layer("slab", k, identity.plane_key_slab(*z), z, dis, sets[k], raw=_raw_measurements(raw_arr, hits[k]))
            layer.stats = {**stats[k], "raw_faces": len(raw_arr.faces) - 1, **dis.stats}
            layer.removed_edges = dis.removed
            entities = [()] + [tuple(sorted({spec.prisms[j].entity_id for j in hits[k][fi]})) for fi in range(1, len(raw_arr.faces))]
            layer.membership = _membership(raw_arr, dis, entities[1:])
            if rt.keep_raw:
                layer.raw.update(arrangement=raw_arr, labels=labels[k])
            layers.append(layer)
    _assign_ids(layers, [{} for _ in layers], rt)
    with rt.serial("arc_realisation"):
        for layer in layers:
            _realise_edges(layer, arcs, register=True)
    return layers


def _label_slab_job(shared, items):
    raws, sets, flat, prisms = shared
    out = []
    for i in items:
        k, fi = flat[i]
        p = interior_point(raws[k], fi)
        inside = sets[k].hit(p)
        out.append((resolve_domain([prisms[j] for j in inside], p), inside))
    return out


# --------------------------------------------------------------------------- levels


def _build_levels(
    spec: ModelSpec, levels: LevelStructure, slabs: list[Layer], rt: Runtime, arcs: dict[str, ArcCarrier]
) -> list[Layer]:
    engine = rt.engine()
    n_levels = len(levels.levels)
    raws, stats, links, sheet_sets, policies, supports = [], [], [], [], [], []
    for k in range(n_levels):
        segments: list[Segment] = []
        for slab_index in (k - 1, k):
            if 0 <= slab_index < len(slabs):
                segments.extend(_slab_edge_segments(slabs[slab_index], arcs))
                segments.extend(_member_segments(slabs[slab_index]))
        sheet_idx = levels.level_sheets[k]
        for i in sheet_idx:
            segments.extend(Segment(c.p, c.q, frozenset([c.id])) for c in spec.prisms[i].carriers())
        required: set[str] = set()
        refinement: set[str] = set()
        for fi in levels.level_families[k]:
            fam = spec.families[fi]
            members = [(c.id, c.p, c.q) for c in fam.carriers]
            for arc in fam.arcs:
                pts = arc.witness_points()
                members.extend(zip(arc.piece_ids(), pts, pts[1:]))
            for cid, p, q in members:
                segments.append(Segment(p, q, frozenset([cid])))
                if fam.edge_policy == "required":
                    required.add(base_carrier_id(cid))
                elif fam.edge_policy == "refinement_only":
                    refinement.add(base_carrier_id(cid))
        supports.append([(fam.family_id, fam.support) for fam in (spec.families[i] for i in levels.level_families[k])
                         if fam.support is not None])
        with rt.noding("noding_driver"):
            snapped = snap_round(segments, engine)
        with rt.serial("dcel_faces"):
            raw_arr = build_arrangement(snapped.edges)
        with rt.serial("linking"):
            below = _slab_link(raw_arr, slabs[k - 1]) if k > 0 else None
            above = _slab_link(raw_arr, slabs[k]) if k < len(slabs) else None
        raws.append(raw_arr)
        links.append((below, above))
        sheet_sets.append(PrismSet(spec.prisms, sheet_idx) if sheet_idx else None)
        policies.append(FamilyPolicies(frozenset(required), frozenset(refinement)))
        stats.append({"segments": len(segments), "hot_pixels": snapped.hot_pixels, "snap_rounds": snapped.rounds})

    flat = [(k, fi) for k, arr in enumerate(raws) for fi in range(1, len(arr.faces))]
    owners = [rt.owner(_anchor(raws[k], fi)) for k, fi in flat]
    cover_sets = [
        [s for s in (slabs[k - 1].cells if k > 0 else None, slabs[k].cells if k < len(slabs) else None, sheet_sets[k])
         if s is not None]
        for k in range(n_levels)
    ]
    out = rt.run("labelling", _level_extras_job, (raws, sheet_sets, supports, cover_sets, flat, spec.prisms), owners)
    extras: dict[tuple[int, int], tuple] = dict(zip(flat, out))

    layers: list[Layer] = []
    with rt.serial("dissolution"):
        for k, raw_arr in enumerate(raws):
            below_link, above_link = links[k]
            labels = [LevelLabel(None, None, None, ())]
            for fi in range(1, len(raw_arr.faces)):
                labels.append(
                    LevelLabel(
                        slabs[k - 1].labels[below_link[fi]] if below_link else None,
                        slabs[k].labels[above_link[fi]] if above_link else None,
                        extras[(k, fi)][0],
                        extras[(k, fi)][1],
                    )
                )
            dis = dissolve(raw_arr, labels, policies[k])
            z = levels.levels[k]
            layer = Layer("level", k, identity.plane_key_level(z), z, dis, None, raw=_raw_measurements(raw_arr, None))
            layer.stats = {**stats[k], "raw_faces": len(raw_arr.faces) - 1, **dis.stats}
            _record_cells(layer, dis, below_link, above_link)
            layer.membership = _membership(raw_arr, dis, [extras[(k, fi)][2] for fi in range(1, len(raw_arr.faces))])
            if rt.keep_raw:
                layer.raw.update(arrangement=raw_arr, labels=labels)
            layers.append(layer)

    tag_sources = []
    for k in range(n_levels):
        ts: dict[str, tuple[str, ...]] = {}
        for slab_index in (k - 1, k):
            if 0 <= slab_index < len(slabs):
                s = slabs[slab_index]
                ts.update({TAG + eid: src for eid, src in zip(s.edge_ids, s.edge_sources)})
        tag_sources.append(ts)
    _assign_ids(layers, tag_sources, rt)
    with rt.serial("arc_realisation"):
        for layer in layers:
            _realise_edges(layer, arcs, register=False)
    return layers


def _level_extras_job(shared, items):
    """Per face: sheet owner (footprint test), inset band labels (distance to the support region) and the source
    entities whose footprint covers it (prisms of the slabs below/above and the sheets on the level)."""
    raws, sheet_sets, supports, cover_sets, flat, prisms = shared
    out = []
    for i in items:
        k, fi = flat[i]
        p = interior_point(raws[k], fi)
        owner = sheet_sets[k].owner_at(p) if sheet_sets[k] is not None else None
        point = (p[0] / p[2], p[1] / p[2])
        bands = ()
        for fid, support in supports[k]:
            band = support.band_of(point)
            if band is not None:
                bands += ((fid, band),)
        entities = tuple(sorted({prisms[j].entity_id for s in cover_sets[k] for j in s.hit(p)}))
        out.append((owner, bands, entities))
    return out


def _membership(raw_arr: Arrangement, dis: Dissolved, raw_entities: list[tuple[str, ...]]) -> list[dict[str, float]]:
    """Area-weighted membership of each dissolved face: for every entity, the fraction of the face covered by its
    footprint. Shares are non-exclusive (overlaps count for each entity) and may sum to more than 1. Computed from
    the raw cells (each lies wholly inside or outside every footprint) with integer chord areas."""
    n = len(dis.arrangement.faces)
    total = [0] * n
    covered: list[dict[str, int]] = [dict() for _ in range(n)]
    for fi in range(1, len(raw_arr.faces)):
        new = dis.face_map[fi]
        area = raw_arr.face_twice_area(fi)
        total[new] += area
        for entity in raw_entities[fi - 1]:
            covered[new][entity] = covered[new].get(entity, 0) + area
    return [{e: round(c / total[f], 12) for e, c in sorted(covered[f].items())} if total[f] else {} for f in range(n)]


def _slab_link(raw_arr: Arrangement, slab: Layer) -> list[int]:
    return link_faces(raw_arr, SlabEdges(slab.arr, slab.edge_ids))[0]


def _record_cells(layer: Layer, dis: Dissolved, below_link: list[int] | None, above_link: list[int] | None) -> None:
    """Slab cell below/above each dissolved face (merged faces lie in one slab cell on each side)."""
    representative: dict[int, int] = {}
    for old, new in enumerate(dis.face_map):
        representative.setdefault(new, old)
    for new in range(len(layer.arr.faces)):
        old = representative.get(new, 0)
        layer.below_cell.append(below_link[old] if below_link else 0)
        layer.above_cell.append(above_link[old] if above_link else 0)


def _slab_edge_segments(slab: Layer, arcs: dict[str, ArcCarrier]) -> list[Segment]:
    """Slab edges as level-arrangement input: a chord for a line edge, the arc's witness chords for an arc edge."""
    out: list[Segment] = []
    for e, eid in zip(slab.arr.edges, slab.edge_ids):
        tag = TAG + eid
        if tag in arcs:
            pts = arcs[tag].witness_points()
            out.extend(Segment(p, q, frozenset([piece])) for piece, p, q in zip(arcs[tag].piece_ids(), pts, pts[1:]))
        else:
            out.append(Segment(e.a, e.b, frozenset([tag])))
    return out


def _member_segments(slab: Layer) -> list[Segment]:
    """Raw slab edges that dissolved, as membership-only carriers: they cut the level cells along every prism
    boundary, so each cell lies wholly inside or outside every footprint and the coverage is exact; they are never
    required and dissolve again at the level (their cells have equal labels)."""
    return [Segment(e.a, e.b, frozenset([f"{MEMBER_ONLY}{slab.plane}#{i}"])) for i, e in enumerate(slab.removed_edges)]


def _realise_edges(layer: Layer, arcs: dict[str, ArcCarrier], register: bool) -> None:
    """Arc edges (all carriers are arcs of one source arc) get their realised arc; slab arcs become carriers."""
    layer.edge_curve = []
    for e, eid in zip(layer.arr.edges, layer.edge_ids):
        carriers = sorted(e.carriers)
        if not carriers or not all(c in arcs for c in carriers) or len({arcs[c].root_id for c in carriers}) != 1:
            layer.edge_curve.append(None)
            continue
        arc = arcs[carriers[0]]
        root = arcs.get(arc.root_id, arc)
        realised = realise(arc, root, e.a, e.b)
        layer.edge_curve.append(realised)
        if register:
            tag = TAG + eid
            arcs[tag] = derived_arc(tag, root, realised, e.a, e.b, arc.lineage)


def _anchor(arr: Arrangement, face_index: int):
    """The point that decides which tile owns a face: the origin of its outer cycle's first half-edge."""
    return arr.origin(arr.cycles[arr.faces[face_index].outer].half_edges[0])


def _raw_measurements(arr: Arrangement, hits: list | None) -> dict:
    out = {"total_twice_area": arr.total_bounded_twice_area(), "top_twice_area": arr.top_level_twice_area()}
    if hits is not None:
        per_prism: dict[int, int] = {}
        for fi in range(1, len(arr.faces)):
            for k in hits[fi]:
                per_prism[k] = per_prism.get(k, 0) + arr.face_twice_area(fi)
        out["prism_twice_area"] = per_prism
    return out


# --------------------------------------------------------------------------- ids (parallel)


def _assign_ids(layers: list[Layer], tag_sources: list[dict], rt: Runtime) -> None:
    """Vertex and edge ids, then face ids: atomic content hashing, owned per tile."""
    arrs = [layer.arr for layer in layers]
    planes = [layer.plane for layer in layers]
    tasks, owners = [], []
    for li, arr in enumerate(arrs):
        for vi, p in enumerate(arr.vertices):
            tasks.append((0, li, vi))
            owners.append(rt.owner(p))
        for ei, e in enumerate(arr.edges):
            tasks.append((1, li, ei))
            owners.append(rt.owner(e.a))
    out = rt.run("atomic_ids", _vertex_edge_id_job, (arrs, planes, tag_sources, tasks), owners)
    for layer in layers:
        layer.vertex_ids = [""] * len(layer.arr.vertices)
        layer.edge_ids = [""] * len(layer.arr.edges)
        layer.edge_sources = [()] * len(layer.arr.edges)
    for (kind, li, idx), value in zip(tasks, out):
        if kind == 0:
            layers[li].vertex_ids[idx] = value
        else:
            layers[li].edge_sources[idx], layers[li].edge_ids[idx] = value

    face_tasks = [(li, fi) for li, arr in enumerate(arrs) for fi in range(1, len(arr.faces))]
    face_owners = [rt.owner(_anchor(arrs[li], fi)) for li, fi in face_tasks]
    shared = (arrs, [layer.edge_ids for layer in layers], [layer.labels for layer in layers], planes,
              [layer.kind for layer in layers], face_tasks)
    face_out = rt.run("atomic_ids", _face_id_job, shared, face_owners)
    for layer in layers:
        layer.face_ids = [""] * len(layer.arr.faces)
    for (li, fi), fid in zip(face_tasks, face_out):
        layers[li].face_ids[fi] = fid


def _vertex_edge_id_job(shared, items):
    arrs, planes, tag_sources, tasks = shared
    out = []
    for i in items:
        kind, li, idx = tasks[i]
        arr, plane = arrs[li], planes[li]
        if kind == 0:
            out.append(identity.vertex_id(plane, arr.vertices[idx]))
            continue
        e = arr.edges[idx]
        sources: set[str] = set()
        for c in e.carriers:
            sources.update(tag_sources[li][c] if c.startswith(TAG) else (c,))
        src = tuple(sorted(sources))
        out.append((src, identity.edge_id(plane, e.a, e.b, src)))
    return out


def _face_id_job(shared, items):
    arrs, edge_ids, labels, planes, kinds, tasks = shared
    out = []
    for i in items:
        li, fi = tasks[i]
        arr = arrs[li]
        boundary = [edge_ids[li][h >> 1] for c in arr.face_cycles(fi) for h in arr.cycles[c].half_edges]
        if kinds[li] == "slab":
            out.append(identity.cell_id(planes[li], labels[li][fi], boundary))
        else:
            out.append(identity.face_id("horizontal", planes[li], labels[li][fi], boundary))
    return out
