"""Invariants of the slab and level arrangements (design brief section 11, stage S9).

* every vertex is a lattice point and no horizontal edge is shorter than ``g``;
* slabs thinner than ``g`` are *reported*, never a failure: levels are the exact declared Z
  values and no Z threshold exists (decision D2);
* every arrangement is planar: no crossing edges, no vertex inside an edge;
* area conservation: raw arrangement area = dissolved area = area enclosed by the outermost
  boundaries, and every prism's footprint area is conserved across its cells within the
  snap-rounding bound;
* realised arc edges cross no other edge (sampled) and stay near their exact carrier circle;
* inset-band areas (``area(R_m)`` non-increasing in ``m``) -- the hook sub-stage 1c fills.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from scgsim.geometry.sgb.arrangement.arcs import face_exact_area, sampled_crossings
from scgsim.geometry.sgb.arrangement.derivation.layer import Layer
from scgsim.geometry.sgb.arrangement.noding.snap_rounding import crossing_pixels_kernel, hits_kernel
from scgsim.geometry.sgb.prisms.levels import LevelStructure
from scgsim.geometry.sgb.exact.grid import Grid
from scgsim.geometry.sgb.exact.prisms import PrismCatalogue
from scgsim.geometry.sgb.exact.stopwatch import Stopwatch
from scgsim.geometry.sgb.checks.validation.report import ValidationReport

SNAP_BOUND_GRID_UNITS = 1.0  # iterated snap rounding on layout inputs stays below this
PRISM_AREA_TOLERANCE_FACTOR = (
    4  # snapping moves a boundary by < sqrt(2)/2 g; iteration may add a little
)


def check_grid_and_lengths(
    report: ValidationReport,
    layers: Sequence[Layer],
    levels: LevelStructure,
    grid: Grid,
) -> None:
    """Lattice vertices, edge lengths, and the (report-only) slabs thinner than the grid."""
    shortest = None
    off_grid = 0
    for layer in layers:
        for e in layer.arr.edges:
            if not all(isinstance(c, int) for c in (*e.a, *e.b)):
                off_grid += 1
            length_sq = (e.a[0] - e.b[0]) ** 2 + (e.a[1] - e.b[1]) ** 2
            shortest = length_sq if shortest is None else min(shortest, length_sq)
    report.add("vertices_on_grid", off_grid == 0, f"{off_grid} off-grid edges")
    report.add(
        "no_edge_shorter_than_g",
        shortest is None or shortest >= 1,
        f"shortest horizontal edge = {shortest} grid^2",
    )
    thin = [
        f"[{lo}, {hi}] um ({float(hi - lo) / grid.size_um:.3g} g)"
        for lo, hi in zip(levels.levels, levels.levels[1:], strict=False)
        if float(hi - lo) < grid.size_um
    ]
    report.add(
        "slabs_thinner_than_g_are_kept",
        not thin,
        f"{len(thin)} slab(s) thinner than the grid, kept as declared: {'; '.join(thin[:5])}"
        if thin
        else "every slab is at least one grid unit thick",
        warning=True,
    )


def check_planar(
    report: ValidationReport, layers: Sequence[Layer], watch: Stopwatch
) -> None:
    """No crossing edge and no vertex inside an edge, using the noding kernels (booked on ``watch``)."""
    bad = []
    for layer in layers:
        keys = [(e.a, e.b) for e in layer.arr.edges]
        if len(keys) < 2:
            continue
        flags = [True] * len(keys)
        with watch.stage("noding_kernels"):
            crossings = crossing_pixels_kernel(keys, flags)
        vertices = {p for key in keys for p in key}
        with watch.stage("noding_kernels"):
            hits = hits_kernel(keys, flags, vertices, set())
        pass_through = sum(
            1
            for (a, b), found in zip(keys, hits, strict=True)
            for c in found
            if c not in (a, b)
        )
        if crossings or pass_through:
            bad.append(
                f"{layer.plane}: {len(crossings)} crossings, {pass_through} pass-throughs"
            )
    report.add(
        "arrangements_planar",
        not bad,
        "; ".join(bad) or "no crossing edge, no vertex inside an edge",
    )


def check_arcs(report: ValidationReport, layers: Sequence[Layer]) -> None:
    """Realised arc edges cross no other edge; their deviation from the exact carrier stays near the snap bound."""
    crossing, worst, count = [], 0.0, 0
    for layer in layers:
        curves = [c for c in layer.edge_curve if c is not None]
        if not curves:
            continue
        count += len(curves)
        worst = max([worst, *(c.deviation for c in curves)])
        crossing.extend(
            f"{layer.plane}: {layer.edge_ids[a]} x {layer.edge_ids[b]}"
            for a, b in sampled_crossings(layer.arr, layer.edge_curve)
        )
    report.add(
        "arc_edges_do_not_cross",
        not crossing,
        "; ".join(crossing[:3]) or f"{count} arc edges sampled",
    )
    report.add(
        "arc_deviation_within_snap_bound",
        worst <= SNAP_BOUND_GRID_UNITS,
        f"max deviation of a realised arc from its exact carrier = {worst:.4f} grid units",
        warning=True,
    )


def check_areas(report: ValidationReport, layers: Sequence[Layer]) -> None:
    """Raw, dissolved and enclosed area agree for every arrangement."""
    bad = []
    for layer in layers:
        raw = layer.raw
        dissolved_total = layer.arr.total_bounded_twice_area()
        dissolved_top = layer.arr.top_level_twice_area()
        if not (
            raw.total_twice_area
            == raw.top_twice_area
            == dissolved_total
            == dissolved_top
        ):
            bad.append(
                f"{layer.plane}: raw {raw.total_twice_area}/{raw.top_twice_area} "
                f"dissolved {dissolved_total}/{dissolved_top}"
            )
    report.add(
        "area_conservation_per_arrangement",
        not bad,
        "; ".join(bad) or "raw = dissolved = enclosed area for every S_i and A_i",
    )


def check_prism_areas(
    report: ValidationReport,
    catalogue: PrismCatalogue,
    levels: LevelStructure,
    slabs: Sequence[Layer],
) -> None:
    """Each prism's footprint area is conserved over the cells it covers, within the snap bound."""
    worst = 0.0
    bad = []
    for k, slab in enumerate(slabs):
        measured = slab.raw.prism_twice_area or {}
        for pi in levels.slab_active[k]:
            prism = catalogue.prisms[pi]
            want = prism.footprint.twice_area()
            got = measured.get(pi, 0)
            perimeter = sum(
                abs(r[(i + 1) % len(r)][0] - r[i][0])
                + abs(r[(i + 1) % len(r)][1] - r[i][1])
                for r in prism.footprint.rings()
                for i in range(len(r))
            )
            tolerance = PRISM_AREA_TOLERANCE_FACTOR * perimeter
            worst = max(worst, abs(got - want) / max(tolerance, 1))
            if abs(got - want) > tolerance:
                bad.append(f"{prism.entity_id}@{slab.plane}: {got} vs {want}")
    report.add(
        "prism_area_conserved_over_cells",
        not bad,
        "; ".join(bad) or f"worst deviation = {worst:.3f} of the snap bound",
    )


def check_band_areas(
    report: ValidationReport,
    level_layers: Sequence[Layer],
    supports: Mapping[str, tuple[float, float]] | None = None,
) -> None:
    """Inset bands partition their support region and ``area(R_m)`` does not grow with ``m``.

    ``area(R_m)`` is the union of the bands from ``m`` on (selections are cumulative, never added as
    disjoint contributions).  ``supports`` maps a family id to ``(support area, tolerance)`` in grid^2:
    the bands must cover the support within the tolerance (every boundary moves by less than one grid unit).
    Without it only monotonicity and non-negative band areas are checked.
    """
    areas: dict[str, dict[int, float]] = {}
    for layer in level_layers:
        for fi in range(1, len(layer.arr.faces)):
            for family_id, band in layer.labels[fi].bands:
                by_band = areas.setdefault(family_id, {})
                by_band[band] = by_band.get(band, 0.0) + face_exact_area(
                    layer.arr, layer.edge_curve, fi
                )
    problems = []
    family_ids: set[str] = {*areas, *(supports or {})}
    for family_id in sorted(family_ids):
        by_band = areas.get(family_id, {})
        ordered = [by_band.get(k, 0.0) for k in range(max(by_band, default=-1) + 1)]
        cumulative = [sum(ordered[k:]) for k in range(len(ordered))]
        if any(a < -1e-6 for a in ordered) or any(
            b > a + 1e-6 for a, b in zip(cumulative, cumulative[1:], strict=False)
        ):
            problems.append(
                f"{family_id}: band areas {[round(a) for a in ordered]} are not non-increasing"
            )
        if supports and family_id in supports:
            expected, tolerance = supports[family_id]
            covered = cumulative[0] if cumulative else 0.0
            if abs(covered - expected) > tolerance:
                problems.append(
                    f"{family_id}: bands cover {covered:.1f} but the region has {expected:.1f} grid^2"
                )
    report.add(
        "band_area_monotone",
        not problems,
        "; ".join(problems)
        or f"{len(areas)} inset families: bands partition their support and area(R_m) is non-increasing in m",
    )
