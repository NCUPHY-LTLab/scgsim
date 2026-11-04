"""Structural invariants and failure behaviour (brief section 11, stage S9).

Owns independent checks over the assembled model. Each check yields a named
result; any failure raises :class:`ValidationError` (fail closed, no fallback
geometry). Checks implemented for lines-only phase 1:

* every vertex is a grid point and no edge is shorter than ``g`` (also vertical);
* every arrangement is planar: no crossing edges, no vertex inside an edge;
* area conservation: raw arrangement area = dissolved area = area enclosed by
  the outermost boundaries; per-prism footprint area conserved across cells
  (within the snap-rounding bound);
* every shell is closed and orientable (each edge used twice, opposite sense);
* Euler characteristic of each shell is 2 - 2*genus;
* each face is used by at most two volumes; an interior face by exactly two sides;
* every physical group is non-empty and every emitted face has a group;
* realised arc edges cross no other edge (sampled) and stay near their exact carrier circle.

* inset bands partition their support region and ``area(R_m)`` does not grow with ``m`` (brief section 14).
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from fractions import Fraction

from sgbproto.arcs import face_exact_area, sampled_crossings
from sgbproto.assembly import Topology, shell_checks
from sgbproto.errors import ValidationError
from sgbproto.layers import Layer
from sgbproto.levels import LevelStructure
from sgbproto.parallel import Runtime
from sgbproto.records import Grid, ModelSpec


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""
    warning: bool = False


@dataclass
class ValidationReport:
    checks: list[Check] = field(default_factory=list)

    timings: dict[str, float] = field(default_factory=dict)  # check function -> seconds (wall)

    @contextmanager
    def timed(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = self.timings.get(name, 0.0) + time.perf_counter() - t0

    def add(self, name: str, passed: bool, detail: str = "", warning: bool = False) -> None:
        self.checks.append(Check(name, passed, detail, warning))

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed and not c.warning]

    def raise_if_failed(self) -> None:
        if self.failures:
            lines = "; ".join(f"{c.name}: {c.detail}" for c in self.failures)
            raise ValidationError(f"{len(self.failures)} invariant(s) failed: {lines}")

    def to_dict(self) -> dict:
        return {
            "passed": not self.failures,
            "timings_s": {k: round(v, 5) for k, v in self.timings.items()},
            "checks": [
                {"name": c.name, "passed": c.passed, "warning": c.warning, "detail": c.detail} for c in self.checks
            ],
        }


def validate(
    spec: ModelSpec,
    levels: LevelStructure,
    slabs: list[Layer],
    level_layers: list[Layer],
    topo: Topology,
    check_planarity: bool = True,
    rt: Runtime | None = None,
) -> ValidationReport:
    report = ValidationReport()
    rt = rt or Runtime()
    with rt.serial("validation_serial"), report.timed("grid_and_lengths"):
        _check_grid_and_lengths(report, slabs + level_layers, topo, spec.grid)
    if check_planarity:
        with report.timed("planarity"):
            _check_planar(report, slabs + level_layers, rt)
    with rt.serial("validation_serial"):
        with report.timed("arcs"):
            _check_arcs(report, slabs + level_layers)
        with report.timed("bands"):
            _check_bands(report, spec, levels, level_layers)
        with report.timed("areas"):
            _check_areas(report, slabs, level_layers)
        with report.timed("prism_areas"):
            _check_prism_areas(report, spec, levels, slabs)
    with report.timed("shells"):
        _check_shells(report, topo, rt)
    with rt.serial("validation_serial"):
        with report.timed("face_sides"):
            _check_face_sides(report, topo)
        with report.timed("groups"):
            _check_groups(report, topo)
        with report.timed("edges_in_faces"):
            _check_edges_in_faces(report, topo)
    return report


def _check_grid_and_lengths(report: ValidationReport, layers: list[Layer], topo: Topology, grid: Grid) -> None:
    shortest = None
    off_grid = 0
    for layer in layers:
        for e in layer.arr.edges:
            if not all(isinstance(c, int) for c in (*e.a, *e.b)):
                off_grid += 1
            length_sq = (e.a[0] - e.b[0]) ** 2 + (e.a[1] - e.b[1]) ** 2
            shortest = length_sq if shortest is None else min(shortest, length_sq)
    report.add("vertices_on_grid", off_grid == 0, f"{off_grid} off-grid edges")
    report.add("no_edge_shorter_than_g", shortest is None or shortest >= 1, f"shortest horizontal edge = {shortest} grid^2")
    z_gaps = [b - a for a, b in zip(topo.levels, topo.levels[1:])]
    shortest_z = min(z_gaps) if z_gaps else Fraction(1)
    report.add("no_vertical_edge_shorter_than_g", float(shortest_z) >= grid.g, f"shortest slab = {float(shortest_z)} um")


def _check_planar(report: ValidationReport, layers: list[Layer], rt: Runtime) -> None:
    """No crossing edge and no vertex inside an edge, using the (tile-parallel) noding kernels."""
    engine = rt.engine()
    bad = []
    for layer in layers:
        keys = [(e.a, e.b) for e in layer.arr.edges]
        if len(keys) < 2:
            continue
        flags = [True] * len(keys)
        crossings = engine.crossing_pixels(keys, flags)
        vertices = {p for k in keys for p in k}
        hits = engine.hits(keys, flags, vertices, set())
        pass_through = sum(1 for (a, b), found in zip(keys, hits) for c in found if c not in (a, b))
        if crossings or pass_through:
            bad.append(f"{layer.plane}: {len(crossings)} crossings, {pass_through} pass-throughs")
    report.add("arrangements_planar", not bad, "; ".join(bad) or "no crossing edge, no vertex inside an edge")


def _check_arcs(report: ValidationReport, layers: list[Layer]) -> None:
    """Realised arc edges cross no other edge; their deviation from the exact carrier stays near the snap bound."""
    crossing, worst, count = [], 0.0, 0
    for layer in layers:
        if not any(c is not None for c in layer.edge_curve):
            continue
        count += sum(c is not None for c in layer.edge_curve)
        worst = max([worst] + [c.deviation for c in layer.edge_curve if c is not None])
        crossing.extend(f"{layer.plane}: {layer.edge_ids[a]} x {layer.edge_ids[b]}"
                        for a, b in sampled_crossings(layer.arr, layer.edge_curve))
    report.add("arc_edges_do_not_cross", not crossing, "; ".join(crossing[:3]) or f"{count} arc edges sampled")
    report.add("arc_deviation_within_snap_bound", worst <= 1.0,
               f"max deviation of a realised arc from its exact carrier = {worst:.4f} grid units", warning=True)


def _check_bands(report: ValidationReport, spec: ModelSpec, levels: LevelStructure, level_layers: list[Layer]) -> None:
    """Inset bands partition their support region and ``area(R_m)`` (union of bands >= k) does not grow with m."""
    problems, checked = [], 0
    for fam in spec.families:
        support = fam.support
        if support is None:
            continue
        layer = level_layers[levels.levels.index(fam.target_levels[0])]
        areas = [0.0] * support.band_count
        for fi in range(1, len(layer.arr.faces)):
            for family_id, band in layer.labels[fi].bands:
                if family_id == fam.family_id:
                    areas[band] += face_exact_area(layer.arr, layer.edge_curve, fi)
        checked += 1
        cumulative = [sum(areas[k:]) for k in range(len(areas))]  # area(R_{m_k}); R_0 = the region itself
        tolerance = (support.perimeter() * support.band_count)  # each boundary moves by < 1 grid unit
        if abs(cumulative[0] - support.area()) > tolerance:
            problems.append(f"{fam.family_id}: bands cover {cumulative[0]:.1f} but the region has {support.area():.1f} grid^2")
        if any(b > a + 1e-6 for a, b in zip(cumulative, cumulative[1:])) or any(a < -1e-6 for a in areas):
            problems.append(f"{fam.family_id}: band areas {[round(a) for a in areas]} are not non-increasing")
    report.add("band_area_monotone", not problems, "; ".join(problems) or f"{checked} inset families: bands partition "
               "the support region and area(R_m) is non-increasing in m")


def _check_areas(report: ValidationReport, slabs: list[Layer], level_layers: list[Layer]) -> None:
    bad = []
    for layer in slabs + level_layers:
        raw = layer.raw
        dis_total = layer.arr.total_bounded_twice_area()
        dis_top = layer.arr.top_level_twice_area()
        if not (raw["total_twice_area"] == raw["top_twice_area"] == dis_total == dis_top):
            bad.append(f"{layer.plane}: raw {raw['total_twice_area']}/{raw['top_twice_area']} dissolved {dis_total}/{dis_top}")
    report.add("area_conservation_per_arrangement", not bad, "; ".join(bad) or "raw = dissolved = enclosed area for every S_i and A_i")


def _check_prism_areas(report: ValidationReport, spec: ModelSpec, levels: LevelStructure, slabs: list[Layer]) -> None:
    worst = 0.0
    bad = []
    for k, slab in enumerate(slabs):
        measured = slab.raw.get("prism_twice_area", {})
        for pi in levels.slab_active[k]:
            prism = spec.prisms[pi]
            want = prism.footprint.twice_area()
            got = measured.get(pi, 0)
            perimeter = sum(
                abs(r[(i + 1) % len(r)][0] - r[i][0]) + abs(r[(i + 1) % len(r)][1] - r[i][1])
                for r in prism.footprint.rings()
                for i in range(len(r))
            )
            # Snapping moves a boundary by at most sqrt(2)/2 grid units: twice-area error <= 2 * perimeter * 0.71
            # (iterated rounding can add a little: allow a factor of 2).
            tolerance = 4 * perimeter
            worst = max(worst, abs(got - want) / max(tolerance, 1))
            if abs(got - want) > tolerance:
                bad.append(f"{prism.entity_id}@{slab.plane}: {got} vs {want}")
    report.add("prism_area_conserved_over_cells", not bad,
               "; ".join(bad) or f"worst deviation = {worst:.3f} of the snap bound")


def _check_shells(report: ValidationReport, topo: Topology, rt: Runtime) -> None:
    """Recompute closure, orientation and Euler characteristic of every shell (parallel per volume)."""
    shells = [(vid, si, shell) for vid, vol in sorted(topo.volumes.items()) for si, shell in enumerate(vol.shells)]
    owners = [i % (rt.tiles * rt.tiles) for i in range(len(shells))]
    results = rt.run("validation_shells", _shell_check_job, (topo.faces, topo.edges, [s[2] for s in shells]), owners)
    problems, euler_bad = [], []
    for (vid, si, shell), (issues, chi) in zip(shells, results):
        problems.extend(f"{vid}/{si}: {msg}" for msg in issues)
        if chi % 2 != 0 or (shell.role == "outer" and chi > 2):
            euler_bad.append(f"{vid}/{si}: chi={chi}")
    report.add("shells_closed_and_oriented", not problems,
               "; ".join(problems[:5]) or f"{len(shells)} shells: every edge used twice with opposite orientation")
    report.add("shell_euler_characteristic", not euler_bad, "; ".join(euler_bad[:5]) or "chi = 2 - 2*genus for every shell")
    no_outer = [v for v, vol in topo.volumes.items() if vol.shells and vol.shells[0].role != "outer"]
    report.add("every_volume_has_outer_shell", not no_outer, f"{len(no_outer)} volumes without positive-volume shell")


def _shell_check_job(shared, items):
    faces, edges, shells = shared
    return [shell_checks(faces, edges, list(shells[i].faces)) for i in items]


def _check_face_sides(report: ValidationReport, topo: Topology) -> None:
    uses = Counter()
    for f in topo.faces.values():
        vols = {v for v in (f.neg_volume, f.pos_volume) if v is not None}
        uses[len(vols)] += 1
    over = sum(1 for f in topo.faces.values() if len({f.neg_volume, f.pos_volume} - {None}) > 2)
    interior_bad = [
        f.id for f in topo.faces.values()
        if f.neg_domain is not None and f.pos_domain is not None and (f.neg_volume is None or f.pos_volume is None)
    ]
    report.add("each_face_used_by_at_most_two_volumes", over == 0, f"face-volume count histogram {dict(uses)}")
    report.add("interior_faces_have_two_sides", not interior_bad, f"{len(interior_bad)} interior faces missing a side")


def _check_groups(report: ValidationReport, topo: Topology) -> None:
    empty = [n for n, g in topo.groups.items() if not g["ids"]]
    report.add("physical_groups_non_empty", not empty and bool(topo.groups), f"{len(topo.groups)} groups; empty: {empty}")
    seen: Counter = Counter()
    for g in topo.groups.values():
        seen.update((g["dim"], i) for i in g["ids"])
    overlapping = [k for k, n in seen.items() if n > 1]
    report.add("groups_partition_each_dimension", not overlapping, f"{len(overlapping)} entities in more than one group")
    volumes = topo.volumes
    emitted_faces = [
        f for f in topo.faces.values()
        if any(v is not None and volumes[v].emitted for v in (f.neg_volume, f.pos_volume))
    ]
    missing = [f.id for f in emitted_faces if (2, f.id) not in seen]
    report.add("every_emitted_face_in_one_group", not missing, f"{len(missing)} emitted faces without group")
    missing_v = [v.id for v in volumes.values() if v.emitted and (3, v.id) not in seen]
    report.add("every_emitted_volume_in_one_group", not missing_v, f"{len(missing_v)} emitted volumes without group")
    unknown = [m for members in topo.unions.values() for m in members if m not in topo.groups]
    report.add("unions_reference_primary_groups", not unknown, f"unknown members: {unknown[:3]}")
    bad_walls = [f.id for f in topo.faces.values() if f.kind == "wall" and not f.planar and not f.curved]
    report.add("walls_planar", not bad_walls, f"{len(bad_walls)} non-planar walls (emitted as filled surfaces)", warning=True)


def _check_edges_in_faces(report: ValidationReport, topo: Topology) -> None:
    around: dict[str, int] = Counter()
    for f in topo.faces.values():
        for lp in f.loops:
            for eid, _ in lp:
                around[eid] += 1
    embedded = {fid for v in topo.volumes.values() for fid in v.embedded_faces}
    free = []
    for fid in embedded:
        for lp in topo.faces[fid].loops:
            free.extend(eid for eid, _ in lp if around[eid] == 1)
    bad = [eid for eid, n in around.items() if n < 2 and eid not in set(free)]
    report.add("every_edge_shared_by_two_faces", not bad, f"{len(bad)} dangling edges; {len(set(free))} free edges of embedded faces")
