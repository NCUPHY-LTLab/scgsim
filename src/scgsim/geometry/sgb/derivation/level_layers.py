"""Level arrangements ``A_i`` (design brief section 2, stages S4 and S5).

Owns building, for every level, the arrangement of the curves of the dissolved
slab arrangements below and above it, the sheets lying on the level and the
cut families targeting it: snap-rounded edges, half-edge faces, labels linked
from the slab cells, dissolution, membership, ids and realised arcs.

A level arrangement receives each slab edge as a carrier tagged ``~<edge id>``;
that tag is how assembly later finds the chain of level edges that covers a slab
edge (the wall bottom and top).  The raw slab edges that dissolved enter as
membership-only carriers: they cut the level cells along every prism boundary, so
each cell lies wholly inside or outside every footprint and coverage is exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scgsim.geometry.sgb import identity
from scgsim.geometry.sgb.arrangement.dcel import (
    Arrangement,
    build_arrangement,
    interior_point,
)
from scgsim.geometry.sgb.derivation.arc_edges import realise_arc_edges
from scgsim.geometry.sgb.derivation.ids import assign_ids
from scgsim.geometry.sgb.derivation.layer import Layer, RawMeasurements
from scgsim.geometry.sgb.labelling.bands import face_band
from scgsim.geometry.sgb.labelling.dissolve import Dissolved, FamilyPolicies, dissolve
from scgsim.geometry.sgb.labelling.linking import TAG, SlabEdges, link_faces
from scgsim.geometry.sgb.labelling.membership import membership_shares
from scgsim.geometry.sgb.labelling.prism_set import PrismSet
from scgsim.geometry.sgb.noding.snap_rounding import Segment, snap_round
from scgsim.geometry.sgb.parallel import Runtime
from scgsim.geometry.sgb.prisms.levels import LevelStructure
from scgsim.geometry.sgb.records.carriers import (
    MEMBER_ONLY,
    ArcCarrier,
    base_carrier_id,
)
from scgsim.geometry.sgb.records.labels import LevelLabel, SheetOwner
from scgsim.geometry.sgb.records.prisms import PrismCatalogue
from scgsim.geometry.sgb.records.support import SupportRegion


@dataclass
class _RawLevel:
    """One level before labelling: its raw arrangement and what links it to the slabs."""

    arrangement: Arrangement
    below_link: list[int] | None
    above_link: list[int] | None
    sheets: PrismSet | None
    policies: FamilyPolicies
    supports: list[tuple[str, SupportRegion]]  # inset families targeting this level
    stats: dict[str, int]


def build_level_layers(
    catalogue: PrismCatalogue,
    levels: LevelStructure,
    slabs: list[Layer],
    rt: Runtime,
    arcs: dict[str, ArcCarrier],
) -> list[Layer]:
    """All level layers, given the finished slab layers (slab arcs are already registered in ``arcs``)."""
    engine = rt.engine()
    raw_levels = []
    for level in range(len(levels.levels)):
        segments, policies = _level_segments(catalogue, levels, slabs, level, arcs)
        with rt.noding("noding_driver"):
            snapped = snap_round(segments, engine)
        with rt.serial("dcel_faces"):
            arrangement = build_arrangement(snapped.edges)
        with rt.serial("linking"):
            below = _link_to_slab(arrangement, slabs[level - 1]) if level > 0 else None
            above = (
                _link_to_slab(arrangement, slabs[level]) if level < len(slabs) else None
            )
        sheet_indices = levels.level_sheets[level]
        sheets = (
            PrismSet(catalogue.prisms, sheet_indices, catalogue.grid)
            if sheet_indices
            else None
        )
        stats = {
            "segments": len(segments),
            "hot_pixels": snapped.hot_pixels,
            "snap_rounds": snapped.rounds,
        }
        supports = [
            (catalogue.families[i].family_id, support)
            for i in levels.level_families[level]
            if (support := catalogue.families[i].support) is not None
        ]
        raw_levels.append(
            _RawLevel(arrangement, below, above, sheets, policies, supports, stats)
        )

    extras = _label_level_cells(catalogue, slabs, raw_levels, rt)
    layers = _dissolve_levels(levels, slabs, raw_levels, extras, rt)
    tag_sources = [_tag_sources(slabs, level) for level in range(len(levels.levels))]
    assign_ids(layers, tag_sources, rt)
    with rt.serial("arc_realisation"):
        for layer in layers:
            realise_arc_edges(layer, arcs, register=False)
    return layers


def _level_segments(
    catalogue: PrismCatalogue,
    levels: LevelStructure,
    slabs: list[Layer],
    level: int,
    arcs: dict[str, ArcCarrier],
) -> tuple[list[Segment], FamilyPolicies]:
    """Noding input of one level, and the edge policies of the cut families that target it."""
    segments: list[Segment] = []
    for slab_index in (level - 1, level):
        if 0 <= slab_index < len(slabs):
            segments.extend(_slab_edge_segments(slabs[slab_index], arcs))
            segments.extend(_member_segments(slabs[slab_index]))
    for i in levels.level_sheets[level]:
        segments.extend(
            Segment(c.p, c.q, frozenset([c.id])) for c in catalogue.prisms[i].carriers()
        )
    required: set[str] = set()
    refinement: set[str] = set()
    for family_index in levels.level_families[level]:
        family = catalogue.families[family_index]
        members = [(c.id, c.p, c.q) for c in family.carriers]
        for arc in family.arcs:
            points = arc.witness_points()
            members.extend(zip(arc.piece_ids(), points, points[1:], strict=False))
        for carrier_id, p, q in members:
            segments.append(Segment(p, q, frozenset([carrier_id])))
            policy = required if family.edge_policy == "required" else refinement
            policy.add(base_carrier_id(carrier_id))
    return segments, FamilyPolicies(frozenset(required), frozenset(refinement))


def _slab_edge_segments(slab: Layer, arcs: dict[str, ArcCarrier]) -> list[Segment]:
    """Slab edges as level-arrangement input: a chord for a line edge, the arc's witness chords for an arc edge."""
    segments: list[Segment] = []
    for edge, edge_id in zip(slab.arr.edges, slab.edge_ids, strict=False):
        tag = TAG + edge_id
        if tag in arcs:
            points = arcs[tag].witness_points()
            segments.extend(
                Segment(p, q, frozenset([piece]))
                for piece, p, q in zip(
                    arcs[tag].piece_ids(), points, points[1:], strict=False
                )
            )
        else:
            segments.append(Segment(edge.a, edge.b, frozenset([tag])))
    return segments


def _member_segments(slab: Layer) -> list[Segment]:
    """Raw slab edges that dissolved, as membership-only carriers (never required; they dissolve again)."""
    return [
        Segment(e.a, e.b, frozenset([f"{MEMBER_ONLY}{slab.plane}#{i}"]))
        for i, e in enumerate(slab.removed_edges)
    ]


def _link_to_slab(level_arr: Arrangement, slab: Layer) -> list[int]:
    return link_faces(level_arr, SlabEdges(slab.arr, slab.edge_ids))[0]


def _label_level_cells(
    catalogue: PrismCatalogue,
    slabs: list[Layer],
    raw_levels: list[_RawLevel],
    rt: Runtime,
) -> dict[
    tuple[int, int],
    tuple[SheetOwner | None, tuple[tuple[str, int], ...], tuple[str, ...]],
]:
    """Per raw face: sheet owner (footprint test) and the entities whose footprint covers it (a parallel stage)."""
    flat = [
        (k, fi)
        for k, raw in enumerate(raw_levels)
        for fi in range(1, len(raw.arrangement.faces))
    ]
    owners = [rt.owner(raw_levels[k].arrangement.face_anchor(fi)) for k, fi in flat]
    cover_sets = []
    for k, raw in enumerate(raw_levels):
        below = slabs[k - 1].cells if k > 0 else None
        above = slabs[k].cells if k < len(slabs) else None
        cover_sets.append([s for s in (below, above, raw.sheets) if s is not None])
    shared = (
        [raw.arrangement for raw in raw_levels],
        [raw.sheets for raw in raw_levels],
        [raw.supports for raw in raw_levels],
        cover_sets,
        flat,
        catalogue.prisms,
    )
    out = rt.run("labelling", _level_extras_job, shared, owners)
    return dict(zip(flat, out, strict=False))


def _level_extras_job(shared: Any, items: list[int]) -> list[Any]:
    """Per face: its sheet owner, its bands and the entities whose footprint covers it.

    Owner and coverage are footprint tests at one interior point: every footprint boundary is an
    arrangement edge, so no face straddles one.  A band is a distance test, whose boundary can be
    a line inside a face, so a band comes from the face's area (``face_band``), not from the point.
    """
    arrangements, sheet_sets, supports, cover_sets, flat, prisms = shared
    out = []
    for i in items:
        k, fi = flat[i]
        point = interior_point(arrangements[k], fi)
        sheets = sheet_sets[k]
        owner = sheets.owner_at(point) if sheets is not None else None
        bands = tuple(
            (family_id, band)
            for family_id, support in supports[k]
            if (band := face_band(arrangements[k], fi, support)) is not None
        )
        entities = tuple(
            sorted({prisms[j].entity_id for s in cover_sets[k] for j in s.hit(point)})
        )
        out.append((owner, bands, entities))
    return out


def _dissolve_levels(
    levels: LevelStructure,
    slabs: list[Layer],
    raw_levels: list[_RawLevel],
    extras: dict[
        tuple[int, int],
        tuple[SheetOwner | None, tuple[tuple[str, int], ...], tuple[str, ...]],
    ],
    rt: Runtime,
) -> list[Layer]:
    layers = []
    with rt.serial("dissolution"):
        for k, raw in enumerate(raw_levels):
            arrangement = raw.arrangement
            labels = [LevelLabel(None, None, None)]
            for fi in range(1, len(arrangement.faces)):
                labels.append(
                    LevelLabel(
                        slabs[k - 1].labels[raw.below_link[fi]]
                        if raw.below_link
                        else None,
                        slabs[k].labels[raw.above_link[fi]] if raw.above_link else None,
                        extras[(k, fi)][0],
                        extras[(k, fi)][1],
                    )
                )
            dissolved = dissolve(arrangement, labels, raw.policies)
            z = levels.levels[k]
            layer = Layer(
                "level",
                k,
                identity.plane_key_level(z),
                z,
                dissolved,
                None,
                RawMeasurements(
                    arrangement.total_bounded_twice_area(),
                    arrangement.top_level_twice_area(),
                ),
            )
            layer.stats = {
                **raw.stats,
                "raw_faces": len(arrangement.faces) - 1,
                **dissolved.stats,
            }
            _record_slab_cells(layer, dissolved, raw.below_link, raw.above_link)
            entities = [extras[(k, fi)][2] for fi in range(1, len(arrangement.faces))]
            layer.membership = membership_shares(arrangement, dissolved, entities)
            if rt.keep_raw:
                layer.raw_arrangement, layer.raw_labels = arrangement, labels
            layers.append(layer)
    return layers


def _record_slab_cells(
    layer: Layer,
    dissolved: Dissolved,
    below_link: list[int] | None,
    above_link: list[int] | None,
) -> None:
    """Slab cell below and above each dissolved face (a merged face lies in one slab cell on each side)."""
    representative: dict[int, int] = {}
    for old, new in enumerate(dissolved.face_map):
        representative.setdefault(new, old)
    for new in range(len(layer.arr.faces)):
        old = representative.get(new, 0)
        layer.below_cell.append(below_link[old] if below_link else 0)
        layer.above_cell.append(above_link[old] if above_link else 0)


def _tag_sources(slabs: list[Layer], level: int) -> dict[str, tuple[str, ...]]:
    """Tagged slab-edge id -> its polygon-edge sources, for the slabs adjacent to a level."""
    tag_sources: dict[str, tuple[str, ...]] = {}
    for slab_index in (level - 1, level):
        if 0 <= slab_index < len(slabs):
            slab = slabs[slab_index]
            tag_sources.update(
                {
                    TAG + eid: src
                    for eid, src in zip(slab.edge_ids, slab.edge_sources, strict=False)
                }
            )
    return tag_sources
