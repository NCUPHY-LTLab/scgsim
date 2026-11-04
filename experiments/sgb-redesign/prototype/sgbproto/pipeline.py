"""End-to-end derivation S2..S9 with per-stage timings, counts and the Amdahl split.

Owns the orchestration only: levels -> slab arrangements -> level arrangements
-> assembly -> validation. Emission (S10) stays separate because it is
sequential and needs Gmsh.

``tiles`` (``T x T``) and ``workers`` configure the :class:`~sgbproto.parallel.Runtime`:
every parallel stage is a per-tile fork/join over items owned by tile; results
are identical for every ``tiles`` and ``workers`` value (property-tested).
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field

from sgbproto.assembly import Topology, assemble
from sgbproto.contacts import check_contacts, check_manifold
from sgbproto.layers import Layer, build_layers
from sgbproto.levels import LevelStructure, build_levels
from sgbproto.parallel import Runtime
from sgbproto.records import ModelSpec
from sgbproto.routes import ROUTES, RoutePolicy, check_route
from sgbproto.validate import ValidationReport, validate


@dataclass
class BuildResult:
    spec: ModelSpec
    policy: RoutePolicy
    levels: LevelStructure
    slabs: list[Layer]
    level_layers: list[Layer]
    topology: Topology
    report: ValidationReport | None
    timings: dict[str, float] = field(default_factory=dict)  # stage -> seconds (flat)
    stage_kind: dict[str, str] = field(default_factory=dict)  # stage -> "serial" | "parallel"
    amdahl: dict[str, float] = field(default_factory=dict)  # serial/parallel wall, worker work, Amdahl limit
    stage_work: dict[str, float] = field(default_factory=dict)  # parallel stage -> summed tile-job compute (s)
    counts: dict[str, int] = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "model": self.spec.name,
            "route": self.policy.name,
            "grid_um": self.spec.grid.g,
            "levels": [str(z) for z in self.levels.levels],
            "counts": self.counts,
            "timings_s": {k: round(v, 4) for k, v in self.timings.items()},
            "stage_kind": self.stage_kind,
            "amdahl": {k: round(v, 4) for k, v in self.amdahl.items()},
            "parallel_stage_worker_compute_s": {k: round(v, 4) for k, v in self.stage_work.items()},
            "validation": self.report.to_dict() if self.report else None,
        }


def model_bbox(spec: ModelSpec) -> tuple[int, int, int, int]:
    pts = [p for prism in spec.prisms for ring in prism.footprint.rings() for p in ring]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def build(
    spec: ModelSpec,
    route: str | None = None,
    tiles: int = 1,
    workers: int = 1,
    run_validation: bool = True,
    check_planarity: bool = True,
    keep_raw: bool = False,
) -> BuildResult:
    """Derive the full conformal topology for ``spec`` (fail closed on any invariant).

    ``run_validation`` and ``check_planarity`` are test-only switches: validation is not optional in a production
    build (brief section 14).
    """
    # The pipeline allocates millions of small acyclic objects; the cyclic collector would only re-scan them
    # (about 30 % of the run time and growing with the live heap), so it is paused for the build.
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        with Runtime(tiles, workers, model_bbox(spec)) as rt:
            rt.keep_raw = keep_raw
            return _build(spec, route, rt, run_validation, check_planarity)
    finally:
        if was_enabled:
            gc.enable()


def _build(spec: ModelSpec, route: str | None, rt: Runtime, run_validation: bool, check_planarity: bool) -> BuildResult:
    policy = ROUTES[route or spec.meta["route"]]
    check_route(policy, spec.prisms)
    with rt.serial("S2_levels"):
        levels = build_levels(spec.prisms, spec.families)
    slabs, level_layers, _ = build_layers(spec, levels, rt)
    with rt.serial("contact_check"):
        check_contacts(spec, slabs + level_layers)
    with rt.serial("manifold_check"):
        check_manifold(spec, slabs)
    topology = assemble(levels, slabs, level_layers, policy, spec.grid.g, rt)
    report = None
    if run_validation:
        report = validate(spec, levels, slabs, level_layers, topology, check_planarity, rt)
        report.raise_if_failed()
    counts = {
        "prisms": len(spec.prisms),
        "slabs": len(slabs),
        "levels": len(level_layers),
        "slab_edges": sum(len(s.arr.edges) for s in slabs),
        "slab_cells": sum(len(s.arr.faces) - 1 for s in slabs),
        "level_edges": sum(len(a.arr.edges) for a in level_layers),
        "level_cells": sum(len(a.arr.faces) - 1 for a in level_layers),
        "hot_pixels": sum(s.stats["hot_pixels"] for s in slabs + level_layers),
        **topology.counts(),
    }
    timings = {**rt.watch.serial, **rt.watch.parallel}
    kinds = {**{k: "serial" for k in rt.watch.serial}, **{k: "parallel" for k in rt.watch.parallel}}
    return BuildResult(spec, policy, levels, slabs, level_layers, topology, report, dict(timings), kinds,
                       rt.watch.totals(), dict(rt.watch.work), counts)


def id_sets(result: BuildResult) -> dict[str, frozenset[str]]:
    """The identity of a build: every entity id by kind (used by determinism and tiling tests)."""
    topo = result.topology
    return {
        "vertices": frozenset(topo.vertices),
        "edges": frozenset(topo.edges),
        "faces": frozenset(topo.faces),
        "volumes": frozenset(topo.volumes),
        **{f"S:{s.plane}:cells": frozenset(s.face_ids[1:]) for s in result.slabs},
        **{f"S:{s.plane}:edges": frozenset(s.edge_ids) for s in result.slabs},
        **{f"A:{a.plane}:cells": frozenset(a.face_ids[1:]) for a in result.level_layers},
        **{f"A:{a.plane}:edges": frozenset(a.edge_ids) for a in result.level_layers},
        **{f"G:{name}": frozenset(g["ids"]) for name, g in topo.groups.items()},
    }


def ids_digest(result: BuildResult) -> str:
    """One hash over every entity id: equal iff two builds have identical identity."""
    from sgbproto.identity import digest

    sets = id_sets(result)
    return digest("build", {k: sorted(v) for k, v in sorted(sets.items())})
