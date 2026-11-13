"""Stages S2 to S5 as one call: the labelled arrangements of a model.

This is the entry point sub-stage 1a delivers.  Given an effective
``GeometryBuildInput`` and its ``RepresentationContext`` (or an already built prism
catalogue) it returns every slab arrangement and every level arrangement, snap
rounded, labelled, dissolved and content-addressed, after the different-Net contact
and non-manifold checks.  Assembly of faces, walls, shells and volumes, route
policies, cut families and emission build on this result in later sub-stages.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field, replace

from scgsim.geometry.models.input import GeometryBuildInput
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.sgb.arrangement.derivation.contacts import (
    check_contacts,
    check_manifold,
)
from scgsim.geometry.sgb.arrangement.derivation.layer import Layer
from scgsim.geometry.sgb.arrangement.derivation.level_layers import build_level_layers
from scgsim.geometry.sgb.arrangement.derivation.slabs import build_slab_layers
from scgsim.geometry.sgb.exact import identity
from scgsim.geometry.sgb.exact.carriers import ArcCarrier
from scgsim.geometry.sgb.exact.prisms import PrismCatalogue
from scgsim.geometry.sgb.exact.stopwatch import Stopwatch
from scgsim.geometry.sgb.prisms.catalogue import build_prism_catalogue
from scgsim.geometry.sgb.prisms.levels import LevelStructure, build_levels


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
    counts: dict[str, int] = field(default_factory=dict)
    structural_checks: list[str] = field(
        default_factory=list
    )  # checks that ran while labelling and passed

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
    build_input: GeometryBuildInput, representation: RepresentationContext
) -> LabelledArrangements:
    """The labelled arrangements of an effective input (stages S2 to S5)."""
    catalogue = build_prism_catalogue(build_input, representation)
    return derive_from_catalogue(catalogue)


def derive_from_catalogue(
    catalogue: PrismCatalogue, *, keep_raw: bool = False
) -> LabelledArrangements:
    """The labelled arrangements of a prism catalogue (stages S4 and S5).

    ``keep_raw`` keeps each layer's pre-dissolution arrangement (figures and teaching examples).
    """
    # The pipeline allocates millions of small acyclic objects; the cyclic collector would only
    # re-scan them (about 30 % of the run time), so it is paused for the build.
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        return derive_arrangements(catalogue, keep_raw=keep_raw, watch=Stopwatch())
    finally:
        if was_enabled:
            gc.enable()


def derive_physical_arrangements(catalogue: PrismCatalogue) -> LabelledArrangements:
    """The physical pre-pass: the labelled arrangements of the catalogue's prisms without ports or cut families.

    ``bands`` finds every inset support face on it, so the faces and their gap edges are the physical ones:
    ports, contact caps and other bands never move a band.
    """
    physical = replace(
        catalogue,
        prisms=tuple(p for p in catalogue.prisms if p.material_kind != "port"),
        families=(),
    )
    return derive_from_catalogue(physical)


def derive_arrangements(
    catalogue: PrismCatalogue, *, keep_raw: bool, watch: Stopwatch
) -> LabelledArrangements:
    """The labelled arrangements of a catalogue, for a caller that already paused the cyclic collector.

    ``watch`` receives the time of every stage; the caller's later stages book on the same watch.
    """
    with watch.stage("levels"):
        levels = build_levels(catalogue.prisms, catalogue.families)
    arcs = catalogue.arc_registry()
    slabs = build_slab_layers(catalogue, levels, arcs, keep_raw=keep_raw, watch=watch)
    level_layers = build_level_layers(
        catalogue, levels, slabs, arcs, keep_raw=keep_raw, watch=watch
    )
    with watch.stage("contact_check"):
        check_contacts(catalogue, (*slabs, *level_layers))
    with watch.stage("manifold_check"):
        check_manifold(catalogue, slabs)
    return LabelledArrangements(
        catalogue,
        levels,
        slabs,
        level_layers,
        arcs,
        dict(watch.seconds),
        _counts(catalogue, slabs, level_layers),
        ["no_different_net_contact", "no_non_manifold_pinch"],
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
