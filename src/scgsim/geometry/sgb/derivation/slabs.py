"""Slab arrangements ``S_i`` (design brief section 2, stages S4 and S5).

Owns building, for every slab, the arrangement of the boundary curves of the
prisms active in it: snap-rounded edges, half-edge faces, exact domain labels,
dissolution, area-weighted membership, content-addressed ids and realised arcs.

The work is organised stage by stage over all slabs, so each parallel stage
(labelling, atomic ids) is one fork/join over per-tile item lists.  Slab
arrangements do not depend on each other.
"""

from __future__ import annotations

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
from scgsim.geometry.sgb.labelling.dissolve import dissolve
from scgsim.geometry.sgb.labelling.membership import membership_shares
from scgsim.geometry.sgb.labelling.prism_set import PrismSet, resolve_domain
from scgsim.geometry.sgb.noding.snap_rounding import Segment, snap_round
from scgsim.geometry.sgb.parallel import Runtime
from scgsim.geometry.sgb.prisms.levels import LevelStructure
from scgsim.geometry.sgb.records.carriers import ArcCarrier
from scgsim.geometry.sgb.records.labels import Domain
from scgsim.geometry.sgb.records.prisms import PrismCatalogue


def build_slab_layers(
    catalogue: PrismCatalogue,
    levels: LevelStructure,
    rt: Runtime,
    arcs: dict[str, ArcCarrier],
) -> list[Layer]:
    """All slab layers; slab arcs are registered in ``arcs`` as derived carriers."""
    prisms = catalogue.prisms
    engine = rt.engine()
    raw_arrangements: list[Arrangement] = []
    noding_stats: list[dict[str, int]] = []
    prism_sets: list[PrismSet] = []
    for slab in range(levels.slab_count):
        active = levels.slab_active[slab]
        segments = [
            Segment(c.p, c.q, frozenset([c.id]))
            for i in active
            for c in prisms[i].carriers()
        ]
        with rt.noding("noding_driver"):
            snapped = snap_round(segments, engine)
        with rt.serial("dcel_faces"):
            raw_arrangements.append(build_arrangement(snapped.edges))
        noding_stats.append(
            {
                "segments": len(segments),
                "hot_pixels": snapped.hot_pixels,
                "snap_rounds": snapped.rounds,
            }
        )
        prism_sets.append(PrismSet(prisms, active, catalogue.grid))

    labels, hits = _label_slab_cells(raw_arrangements, prism_sets, prisms, rt)
    layers = _dissolve_slabs(
        catalogue, levels, rt, raw_arrangements, labels, hits, noding_stats, prism_sets
    )
    assign_ids(layers, [{} for _ in layers], rt)
    with rt.serial("arc_realisation"):
        for layer in layers:
            realise_arc_edges(layer, arcs, register=True)
    return layers


def _label_slab_cells(
    raws: list[Arrangement], sets: list[PrismSet], prisms: Any, rt: Runtime
) -> tuple[list[list[Domain | None]], list[list[list[int]]]]:
    """Domain label and containing prisms of every raw face (a parallel stage, owned per face)."""
    flat = [(k, fi) for k, arr in enumerate(raws) for fi in range(1, len(arr.faces))]
    owners = [rt.owner(raws[k].face_anchor(fi)) for k, fi in flat]
    out = rt.run("labelling", _label_slab_job, (raws, sets, flat, prisms), owners)
    labels: list[list[Domain | None]] = [[None] for _ in raws]
    hits: list[list[list[int]]] = [[[]] for _ in raws]
    for (k, _), (domain, inside) in zip(flat, out, strict=False):
        labels[k].append(domain)
        hits[k].append(inside)
    return labels, hits


def _label_slab_job(shared: Any, items: list[int]) -> list[Any]:
    raws, sets, flat, prisms = shared
    out = []
    for i in items:
        k, fi = flat[i]
        point = interior_point(raws[k], fi)
        inside = sets[k].hit(point)
        domain = resolve_domain([prisms[j] for j in inside], point, sets[k].grid)
        out.append((domain, inside))
    return out


def _dissolve_slabs(  # noqa: PLR0913 - the per-slab tables travel together
    catalogue: PrismCatalogue,
    levels: LevelStructure,
    rt: Runtime,
    raws: list[Arrangement],
    labels: list[list[Domain | None]],
    hits: list[list[list[int]]],
    noding_stats: list[dict[str, int]],
    prism_sets: list[PrismSet],
) -> list[Layer]:
    prisms = catalogue.prisms
    layers = []
    with rt.serial("dissolution"):
        for k, raw in enumerate(raws):
            z = (levels.levels[k], levels.levels[k + 1])
            dissolved = dissolve(raw, labels[k])
            layer = Layer(
                "slab",
                k,
                identity.plane_key_slab(*z),
                z,
                dissolved,
                prism_sets[k],
                _slab_measurements(raw, hits[k]),
            )
            layer.stats = {
                **noding_stats[k],
                "raw_faces": len(raw.faces) - 1,
                **dissolved.stats,
            }
            layer.removed_edges = dissolved.removed
            entities = [
                tuple(sorted({prisms[j].entity_id for j in hits[k][fi]}))
                for fi in range(1, len(raw.faces))
            ]
            layer.membership = membership_shares(raw, dissolved, entities)
            if rt.keep_raw:
                layer.raw_arrangement, layer.raw_labels = raw, labels[k]
            layers.append(layer)
    return layers


def _slab_measurements(raw: Arrangement, hits: list[list[int]]) -> RawMeasurements:
    """Raw area totals and the area covered by each prism (for the area-conservation check)."""
    per_prism: dict[int, int] = {}
    for fi in range(1, len(raw.faces)):
        for prism in hits[fi]:
            per_prism[prism] = per_prism.get(prism, 0) + raw.face_twice_area(fi)
    return RawMeasurements(
        raw.total_bounded_twice_area(), raw.top_level_twice_area(), per_prism
    )
