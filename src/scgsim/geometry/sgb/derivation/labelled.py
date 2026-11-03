"""Stages S2 to S5 as one call: the labelled arrangements of a model.

This is the entry point sub-stage 1a delivers.  Given an effective
``GeometryBuildInput`` and its ``RepresentationContext`` (or an already built prism
catalogue) it returns every slab arrangement and every level arrangement, snap
rounded, labelled, dissolved and content-addressed, after the different-Net contact
and non-manifold checks.  Assembly of faces, walls, shells and volumes, route
policies, cut families and emission build on this result in later sub-stages.

``tiles`` (``T x T``) and ``workers`` configure the per-tile runtime: every parallel
stage is a fork/join over items owned by a tile, and the result is identical for
every ``tiles`` and ``workers`` value.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field

from scgsim.geometry.models.input import GeometryBuildInput
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.sgb import identity
from scgsim.geometry.sgb.derivation.contacts import check_contacts, check_manifold
from scgsim.geometry.sgb.derivation.layer import Layer
from scgsim.geometry.sgb.derivation.level_layers import build_level_layers
from scgsim.geometry.sgb.derivation.slabs import build_slab_layers
from scgsim.geometry.sgb.parallel import Runtime
from scgsim.geometry.sgb.prisms.catalogue import build_prism_catalogue
from scgsim.geometry.sgb.prisms.levels import LevelStructure, build_levels
from scgsim.geometry.sgb.records.carriers import ArcCarrier
from scgsim.geometry.sgb.records.prisms import PrismCatalogue


@dataclass
class LabelledArrangements:
    """Every slab and level arrangement of a model, with the records they came from."""

    catalogue: PrismCatalogue
    levels: LevelStructure
    slabs: list[Layer]
    level_layers: list[Layer]
    arcs: dict[
        str, ArcCarrier
    ]  # every arc carrier by id: source arcs and realised slab-edge arcs
    timings: dict[str, float] = field(default_factory=dict)  # stage -> seconds
    stage_kind: dict[str, str] = field(
        default_factory=dict
    )  # stage -> "serial" | "parallel"
    amdahl: dict[str, float] = field(
        default_factory=dict
    )  # serial/parallel wall, worker work, Amdahl limit
    counts: dict[str, int] = field(default_factory=dict)

    def identity_sets(self) -> dict[str, frozenset[str]]:
        """Every entity id by kind: equal for two builds iff their identity is equal."""
        sets: dict[str, frozenset[str]] = {}
        for layer in (*self.slabs, *self.level_layers):
            prefix = "S" if layer.kind == "slab" else "A"
            sets[f"{prefix}:{layer.plane}:cells"] = frozenset(layer.face_ids[1:])
            sets[f"{prefix}:{layer.plane}:edges"] = frozenset(layer.edge_ids)
            sets[f"{prefix}:{layer.plane}:vertices"] = frozenset(layer.vertex_ids)
        return sets

    def identity_digest(self) -> str:
        """One hash over every entity id."""
        sets = self.identity_sets()
        return identity.digest("build", {k: sorted(v) for k, v in sorted(sets.items())})


def derive_labelled_arrangements(
    build_input: GeometryBuildInput,
    representation: RepresentationContext,
    *,
    tiles: int = 1,
    workers: int = 1,
) -> LabelledArrangements:
    """The labelled arrangements of an effective input (stages S2 to S5)."""
    catalogue = build_prism_catalogue(build_input, representation)
    return derive_from_catalogue(catalogue, tiles=tiles, workers=workers)


def derive_from_catalogue(
    catalogue: PrismCatalogue,
    *,
    tiles: int = 1,
    workers: int = 1,
    keep_raw: bool = False,
) -> LabelledArrangements:
    """The labelled arrangements of a prism catalogue (stages S4 and S5).

    ``keep_raw`` keeps each layer's pre-dissolution arrangement (figures and teaching examples).
    """
    # The pipeline allocates millions of small acyclic objects; the cyclic collector would only
    # re-scan them (about 30 % of the run time), so it is paused for the build.
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        with Runtime(tiles, workers, _model_bbox(catalogue)) as rt:
            rt.keep_raw = keep_raw
            return _derive(catalogue, rt)
    finally:
        if was_enabled:
            gc.enable()


def _derive(catalogue: PrismCatalogue, rt: Runtime) -> LabelledArrangements:
    with rt.serial("levels"):
        levels = build_levels(catalogue.prisms, catalogue.families)
    arcs = catalogue.arc_registry()
    slabs = build_slab_layers(catalogue, levels, rt, arcs)
    level_layers = build_level_layers(catalogue, levels, slabs, rt, arcs)
    with rt.serial("contact_check"):
        check_contacts(catalogue, (*slabs, *level_layers))
    with rt.serial("manifold_check"):
        check_manifold(catalogue, slabs)
    timings = {**rt.watch.serial, **rt.watch.parallel}
    kinds = {
        **{name: "serial" for name in rt.watch.serial},
        **{name: "parallel" for name in rt.watch.parallel},
    }
    return LabelledArrangements(
        catalogue,
        levels,
        slabs,
        level_layers,
        arcs,
        dict(timings),
        kinds,
        rt.watch.totals(),
        _counts(catalogue, slabs, level_layers),
    )


def _counts(
    catalogue: PrismCatalogue, slabs: list[Layer], level_layers: list[Layer]
) -> dict[str, int]:
    return {
        "prisms": len(catalogue.prisms),
        "slabs": len(slabs),
        "levels": len(level_layers),
        "slab_edges": sum(len(s.arr.edges) for s in slabs),
        "slab_cells": sum(len(s.arr.faces) - 1 for s in slabs),
        "level_edges": sum(len(a.arr.edges) for a in level_layers),
        "level_cells": sum(len(a.arr.faces) - 1 for a in level_layers),
        "hot_pixels": sum(s.stats["hot_pixels"] for s in (*slabs, *level_layers)),
    }


def _model_bbox(catalogue: PrismCatalogue) -> tuple[int, int, int, int]:
    points = [
        p
        for prism in catalogue.prisms
        for ring in prism.footprint.rings()
        for p in ring
    ]
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)
