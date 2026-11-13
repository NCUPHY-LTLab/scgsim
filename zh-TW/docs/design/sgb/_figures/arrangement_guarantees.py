"""Figures of arrangement.qmd: measured guarantees, complexity and the arc witness.

Computed here (the page quotes these numbers):

* guarantees of iterated snap rounding (``scgsim.geometry.sgb.arrangement.noding``) on random segment sets and on layout-like inputs: no crossings, no vertex on an
  edge interior, minimum edge length, minimum vertex-edge distance, maximum deviation from the carriers, and the same
  measurements after round 1 only (classic snap rounding);
* cost: snap-rounding plus DCEL time against the number of segments, and hot pixels against N from the production
  scaling run (``figures/why-scaling-measured.json``, written by ``why_scaling.py measure``);
* the polyline witness of a circular arc: chord count against radius, and a worked example with a foreign segment.

Writes ``figures/arrangement-guarantees.html/.svg``, ``arrangement-complexity.html/.svg``,
``arrangement-arc-witness.svg`` and ``figures/arrangement-guarantees-numbers.json``.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import Counter
from fractions import Fraction

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from intro_common import (
    COLOURS,
    FIG,
    build_example,
    save_plotly,
    save_svg,
    write_numbers,
)
from plotly.subplots import make_subplots
from sgb_emit_cases import pad_requests
from sgb_scenes import airbridge, random_layout

from scgsim.geometry.sgb.arrangement.arcs import realise, sample_arc
from scgsim.geometry.sgb.arrangement.dcel import build_arrangement
from scgsim.geometry.sgb.arrangement.noding.snap_rounding import (
    Segment,
    assemble_chain,
    crossing_pixels_kernel,
    hits_kernel,
    snap_round,
)
from scgsim.geometry.sgb.arrangement.noding.snap_rounding import (
    canonical_key as canonical,
)
from scgsim.geometry.sgb.arrangement.noding.spatial_hash import HotGrid
from scgsim.geometry.sgb.checks.slivers.report import sliver_report
from scgsim.geometry.sgb.exact.carriers import ArcCarrier, EdgeCarrier, Lineage
from scgsim.geometry.sgb.model import build_model_from_catalogue

SQRT_HALF = math.sqrt(2) / 2


# --------------------------------------------------------------------------- snap rounding with a round limit


def snap_state(segments, max_rounds=None):
    """Edges (a, b) -> carrier names after ``max_rounds`` rounds (``None``: until nothing changes), and the round count."""
    edges: dict = {}
    for seg in segments:
        if seg.a != seg.b:
            edges.setdefault(canonical(seg.a, seg.b), set()).update(seg.carriers)
    hot = {p for k in edges for p in k}
    changed, delta, rounds = set(edges), set(hot), 0
    while (changed or delta) and (max_rounds is None or rounds < max_rounds):
        rounds += 1
        keys = sorted(edges)
        flags = [k in changed for k in keys]
        new_hot = crossing_pixels_kernel(keys, flags) - hot
        hot |= new_hot
        delta |= new_hot
        hits = hits_kernel(keys, flags, hot, delta)
        new_edges, verified = {}, set()
        for key, h in zip(keys, hits, strict=True):
            chain = assemble_chain(key[0], key[1], h)
            if len(chain) == 2:
                verified.add(key)
            for u, v in zip(chain, chain[1:], strict=False):
                new_edges.setdefault(canonical(u, v), set()).update(edges[key])
        changed = set(new_edges) - verified
        delta = set()
        edges = new_edges
    return edges, rounds


# --------------------------------------------------------------------------- measurements


def _seg_dist(pt, p, q) -> float:
    dx, dy = q[0] - p[0], q[1] - p[1]
    t = max(
        0.0, min(1.0, ((pt[0] - p[0]) * dx + (pt[1] - p[1]) * dy) / (dx * dx + dy * dy))
    )
    return math.hypot(pt[0] - (p[0] + t * dx), pt[1] - (p[1] + t * dy))


def _proper_cross(a, b, c, d) -> bool:
    def side(p, q, r):
        v = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        return (v > 0) - (v < 0)

    return side(a, b, c) * side(a, b, d) < 0 and side(c, d, a) * side(c, d, b) < 0


def _measure(edges: dict, carriers: dict) -> dict:
    keys = sorted(edges)
    verts = sorted({p for k in keys for p in k})
    crossings = 0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if _proper_cross(*keys[i], *keys[j]):
                crossings += 1
    min_vertex_edge = math.inf
    on_interior = 0
    for v in verts:
        for a, b in keys:
            if v == a or v == b:
                continue
            d = _seg_dist(v, a, b)
            min_vertex_edge = min(min_vertex_edge, d)
            if d < 1e-9:
                on_interior += 1
    deviation = 0.0
    for (a, b), names in edges.items():
        best = min(
            max(_seg_dist(a, *carriers[n]), _seg_dist(b, *carriers[n])) for n in names
        )
        deviation = max(deviation, best)
    return {
        "crossings": crossings,
        "vertex_on_edge_interior": on_interior,
        "min_edge_length": min(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in keys),
        "min_vertex_edge_distance": min_vertex_edge,
        "max_deviation": deviation,
    }


def _random_segments(rng, n, box):
    out = []
    while len(out) < n:
        a = (rng.randint(0, box), rng.randint(0, box))
        b = (rng.randint(0, box), rng.randint(0, box))
        if a != b:
            out.append((a, b))
    return out


def _trials(count=250, n=24, box=48, seed=3) -> dict:
    rng = random.Random(seed)
    rows = {"round_1": [], "iterated": []}
    rounds = []
    for _ in range(count):
        raw = _random_segments(rng, n, box)
        carriers = {f"c{i}": s for i, s in enumerate(raw)}
        segs = [Segment(a, b, frozenset([f"c{i}"])) for i, (a, b) in enumerate(raw)]
        e1, _ = snap_state(segs, 1)
        ef, r = snap_state(segs)
        rounds.append(r)
        rows["round_1"].append(_measure(e1, carriers))
        rows["iterated"].append(_measure(ef, carriers))
    out = {
        "trials": count,
        "segments_per_trial": n,
        "box_g": box,
        "rounds_histogram": dict(sorted(Counter(rounds).items())),
    }
    for name, items in rows.items():
        out[name] = {
            "trials_with_crossings": sum(1 for m in items if m["crossings"]),
            "trials_with_vertex_on_edge": sum(
                1 for m in items if m["vertex_on_edge_interior"]
            ),
            "min_edge_length_g": round(min(m["min_edge_length"] for m in items), 4),
            "min_vertex_edge_distance_g": round(
                min(m["min_vertex_edge_distance"] for m in items), 4
            ),
            "max_deviation_g": round(max(m["max_deviation"] for m in items), 4),
            "deviations": [m["max_deviation"] for m in items],
        }
    return out


def dense_drift(count=1500, n=40, box=30, seed=11) -> dict:
    """Largest deviation after iteration on denser random sets (more crossings per segment); deviation only."""
    rng = random.Random(seed)
    worst, rounds = 0.0, Counter()
    for _ in range(count):
        raw = _random_segments(rng, n, box)
        carriers = {f"c{i}": sg for i, sg in enumerate(raw)}
        edges, r = snap_state(
            [Segment(a, b, frozenset([f"c{i}"])) for i, (a, b) in enumerate(raw)]
        )
        rounds[r] += 1
        for (a, b), names in edges.items():
            worst = max(
                worst,
                min(
                    max(_seg_dist(a, *carriers[nm]), _seg_dist(b, *carriers[nm]))
                    for nm in names
                ),
            )
    return {
        "trials": count,
        "segments_per_trial": n,
        "box_g": box,
        "max_deviation_g": round(worst, 4),
        "rounds_histogram": dict(sorted(rounds.items())),
    }


def example_models() -> dict:
    """The running examples of the page (port on the support plane) and the airbridge, as production models."""
    return {
        "pad_in_ground_A": lambda: build_example("A"),
        "pad_in_ground_B": lambda: build_example("B"),
        "pad_in_ground_A_bands": lambda: build_example("A", insets=pad_requests("A")),
        "pad_in_ground_B_bands": lambda: build_example("B", insets=pad_requests("B")),
        "airbridge_B": lambda: build_model_from_catalogue(airbridge(), "B"),
    }


def _layout_like() -> dict:
    model = build_model_from_catalogue(random_layout(200, seed=1, route="B"), "B")
    drift = sliver_report(model.labelled, model.topology, model.routed)["snap_drift"]
    layers = model.labelled.slabs + model.labelled.level_layers
    rounds = Counter(layer.stats["snap_rounds"] for layer in layers)
    cases = {}
    for case, build in example_models().items():
        built = build()
        report = sliver_report(built.labelled, built.topology, built.routed)
        cases[case] = {
            "max_snap_drift_g": report["snap_drift"]["max_grid_units"],
            "edges_measured": report["snap_drift"]["edges_measured"],
            "arc_edges": report["arc_edges"]["count"],
            "max_arc_deviation_g": report["arc_edges"]["max_deviation_grid"],
            "mean_arc_deviation_g": report["arc_edges"]["mean_deviation_grid"],
            "snap_rounds": sorted(
                layer.stats["snap_rounds"]
                for layer in built.labelled.slabs + built.labelled.level_layers
            ),
        }
    return {
        "random_layout_n200": {
            "max_snap_drift_g": drift["max_grid_units"],
            "edges_measured": drift["edges_measured"],
            "edges_over_bound": drift["edges_over_bound"],
            "snap_rounds_histogram": dict(sorted(rounds.items())),
        },
        "examples": cases,
    }


def level_stats() -> dict:
    """What the page quotes about single level arrangements of the running example (segments, hot pixels, rounds, edges)."""
    out = {}
    for name, build in (
        ("pad_in_ground_B", lambda: build_example("B")),
        ("pad_in_ground_A_bands", lambda: build_example("A", insets=pad_requests("A"))),
        (
            "pad_in_ground_B_midplane_port_stage_scene",
            lambda: build_example("B", port_plane="mid"),
        ),
    ):
        model = build()
        out[name] = {
            f"level_z={float(layer.level_z):g}": {
                **{
                    k: layer.stats[k]
                    for k in (
                        "segments",
                        "hot_pixels",
                        "snap_rounds",
                        "edges_before",
                        "edges_after",
                    )
                },
                "cells": len(layer.arr.faces) - 1,
            }
            for layer in model.labelled.level_layers
        }
    return out


def pixel_corners() -> dict:
    """Which segments make the pixel of the grid point (0, 0) hot: the pixel is the half-open square [-1/2, 1/2)^2."""

    grid = HotGrid([(0, 0)])
    cases = {
        "passes through the centre (0, 0)": ((-1, -1), (1, 1)),
        "touches only the lower-left corner (-1/2, -1/2)": ((-1, 0), (0, -1)),
        "touches only the upper-right corner (1/2, 1/2)": ((1, 0), (0, 1)),
        "touches only the lower-right corner (1/2, -1/2)": ((1, 0), (0, -1)),
        "touches only the upper-left corner (-1/2, 1/2)": ((-1, 0), (0, 1)),
        "cuts across the pixel above its centre (at x = 0 it is at y = 1/2)": (
            (-2, 0),
            (2, 1),
        ),
    }
    return {name: bool(grid.touching(a, b)) for name, (a, b) in cases.items()}


# --------------------------------------------------------------------------- cost


def timed_snap(n: int, seed=7) -> dict:
    """snap_round and DCEL time for n short segments at constant density (about one crossing per segment)."""
    rng = random.Random(seed)
    box = int(math.sqrt(n) * 40)
    segs = []
    for i in range(n):
        x, y = rng.randint(0, box), rng.randint(0, box)
        ang = rng.uniform(0, math.pi)
        length = rng.uniform(15, 60)
        segs.append(
            Segment(
                (x, y),
                (x + round(length * math.cos(ang)), y + round(length * math.sin(ang))),
                frozenset([f"s{i}"]),
            )
        )
    segs = [s for s in segs if s.a != s.b]
    t_snap = t_dcel = math.inf
    for _ in range(3):  # best of three
        t0 = time.perf_counter()
        result = snap_round(segs)
        t_snap = min(t_snap, time.perf_counter() - t0)
        t0 = time.perf_counter()
        arr = build_arrangement(result.edges)
        t_dcel = min(t_dcel, time.perf_counter() - t0)
    return {
        "segments": len(segs),
        "snap_s": round(t_snap, 3),
        "dcel_s": round(t_dcel, 3),
        "hot_pixels": result.hot_pixels,
        "edges": len(result.edges),
        "rounds": result.rounds,
        "faces": len(arr.faces) - 1,
    }


def scaling_rows():
    """Arrangement sizes and stage times of the production runs of ``why_scaling.py measure``."""
    rows = json.loads((FIG / "why-scaling-measured.json").read_text())["rows"]
    return [
        {
            "N": r["n_features"],
            "hot_pixels": r["counts"]["hot_pixels"],
            "level_edges": r["counts"]["level_edges"],
            "slab_edges": r["counts"]["slab_edges"],
            "noding_kernels_s": r["stage_s"]["noding_kernels"],
            "noding_driver_s": r["stage_s"]["noding_driver"],
            "dcel_faces_s": r["stage_s"]["dcel_faces"],
            "build_s": r["build_wall_s"],
        }
        for r in rows
    ]


def _slopes(xs, ys):
    return [
        round(math.log(y1 / y0) / math.log(x1 / x0), 2)
        for (x0, y0), (x1, y1) in zip(
            zip(xs, ys, strict=True), zip(xs[1:], ys[1:], strict=True), strict=False
        )
    ]


# --------------------------------------------------------------------------- arc witness


def _quarter(radius: int) -> ArcCarrier:
    return ArcCarrier(
        "arc",
        (Fraction(0), Fraction(0)),
        Fraction(radius),
        True,
        (Fraction(radius), Fraction(0)),
        (Fraction(0), Fraction(radius)),
        Lineage("polygon_edge", "demo"),
    )


def _chord_counts(radii) -> list[int]:
    return [len(_quarter(r).witness_points()) - 1 for r in radii]


FOREIGN = EdgeCarrier("line")  # the foreign segment of the arc example


def _arc_example():
    arc = _quarter(24)
    pts = arc.witness_points()
    pieces = arc.pieces()
    segs = [
        Segment(a, b, frozenset([piece]))
        for piece, a, b in zip(pieces, pts, pts[1:], strict=False)
    ]
    segs.append(
        Segment((14, -2), (14, 26), frozenset([FOREIGN]))
    )  # a foreign vertical segment crossing the arc
    result = snap_round(segs)
    edges = {(e.a, e.b): set(e.carriers) for e in result.edges}
    # merge runs of witness pieces with no foreign vertex between them (what dissolution does)
    degree = Counter(p for k in edges for p in k)
    arc_edges = [(k, v) for k, v in edges.items() if all(c.curve == arc.id for c in v)]
    nxt: dict = {}
    for (a, b), _ in arc_edges:
        nxt.setdefault(a, []).append(b)
        nxt.setdefault(b, []).append(a)
    ends = [p for p, nb in nxt.items() if len(nb) == 1 or degree[p] != 2]
    runs, seen = [], set()
    for start in sorted(ends):
        if start in seen:
            continue
        run, cur, prev = [start], start, None
        while True:
            options = [q for q in nxt[cur] if q != prev]
            if not options or (len(run) > 1 and degree[cur] != 2):
                break
            prev, cur = cur, options[0]
            run.append(cur)
            if degree[cur] != 2:
                break
        seen.update(run[:1] + run[-1:])
        runs.append(run)
    return arc, pts, edges, runs, result


def _arc_figure(radii, counts) -> dict:
    arc, pts, edges, runs, result = _arc_example()
    fig, axes = plt.subplots(
        1, 3, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.1]}
    )
    th = [i * math.pi / 2 / 200 for i in range(201)]
    exact = [(24 * math.cos(t), 24 * math.sin(t)) for t in th]
    for ax in axes[:2]:
        ax.plot(
            *zip(*exact, strict=True),
            color="#999",
            lw=1.2,
            label="exact carrier (circle, r = 24 g)",
        )
        ax.set_aspect("equal")
        ax.set_xlim(-1, 27)
        ax.set_ylim(-4, 27)
        ax.grid(True, color="#e5e5e5", lw=0.5)
        ax.set_xticks(range(0, 27, 4))
        ax.set_yticks(range(0, 27, 4))
        ax.tick_params(labelsize=7)
    a = axes[0]
    a.plot(
        *zip(*pts, strict=True),
        color=COLOURS["pad"],
        lw=1.6,
        marker="o",
        ms=3,
        label=f"witness: {len(pts) - 1} chords, sagitta ≤ g/4",
    )
    a.plot([14, 14], [-2, 26], color=COLOURS["bad"], lw=1.4, label="foreign segment")
    a.legend(fontsize=7, frameon=False, loc="upper right")
    a.set_title(
        "1  Witness chords and a foreign segment (input)", fontsize=9, loc="left"
    )
    b = axes[1]
    for (u, v), names in sorted(edges.items()):
        colour = COLOURS["bad"] if FOREIGN in names else COLOURS["pad"]
        b.plot([u[0], v[0]], [u[1], v[1]], color=colour, lw=1.8, marker="o", ms=3)
    foreign = sorted({p for (u, v), n in edges.items() if FOREIGN in n for p in (u, v)})
    b.set_title(
        f"2  After snap rounding: {len(edges)} edges, {result.hot_pixels} hot pixels, {result.rounds} rounds",
        fontsize=9,
        loc="left",
    )
    c = axes[2]
    c.plot(*zip(*exact, strict=True), color="#999", lw=1.0, ls=":")
    devs = []
    for run in runs:
        if len(run) < 2:
            continue
        a_pt, b_pt = run[0], run[-1]
        real = realise(arc, arc, a_pt, b_pt)
        samples = sample_arc(real, a_pt, b_pt, 40)
        c.plot(*zip(*samples, strict=True), color=COLOURS["pad"], lw=2.4)
        c.plot(
            [a_pt[0], b_pt[0]],
            [a_pt[1], b_pt[1]],
            marker="o",
            color=COLOURS["pad"],
            ls="none",
            ms=4,
        )
        devs.append(real.deviation)
    c.plot([14, 14], [-2, 26], color=COLOURS["bad"], lw=1.4)
    for p in foreign:
        c.plot([p[0]], [p[1]], marker="s", color=COLOURS["bad"], ms=4)
    c.set_aspect("equal")
    c.set_xlim(-1, 27)
    c.set_ylim(-4, 27)
    c.grid(True, color="#e5e5e5", lw=0.5)
    c.tick_params(labelsize=7)
    c.set_title(
        f"3  Merged: {len([r for r in runs if len(r) > 1])} arc edges, deviation ≤ {max(devs):.2f} g",
        fontsize=9,
        loc="left",
    )
    fig.tight_layout()
    save_svg(fig, "arrangement-arc-witness")
    return {
        "radius_g": 24,
        "chords": len(pts) - 1,
        "edges_after_snap": len(edges),
        "hot_pixels": result.hot_pixels,
        "rounds": result.rounds,
        "arc_edges_after_merge": len([r for r in runs if len(r) > 1]),
        "arc_edge_deviations_g": [round(d, 4) for d in devs],
        "chord_count_per_quarter_arc": dict(zip(radii, counts, strict=True)),
    }


# --------------------------------------------------------------------------- figures


def _guarantees_figure(t: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bins = [i * 0.1 for i in range(0, 21)]
    ax.hist(
        t["round_1"]["deviations"],
        bins=bins,
        color="#bbbbbb",
        label="after round 1 (classic snap rounding)",
        alpha=0.9,
    )
    ax.hist(
        t["iterated"]["deviations"],
        bins=bins,
        color=COLOURS["pad"],
        label="after iteration (final)",
        alpha=0.75,
    )
    ax.axvline(SQRT_HALF, color=COLOURS["bad"], ls="--", lw=1.2)
    ax.text(
        SQRT_HALF + 0.03,
        ax.get_ylim()[1] * 0.9,
        "√2/2 g",
        color=COLOURS["bad"],
        fontsize=8,
    )
    ax.set_xlabel(
        "largest deviation of an edge end point from its carrier, per trial (grid units g)"
    )
    ax.set_ylabel("trials")
    ax.set_title(
        f"{t['trials']} random sets of {t['segments_per_trial']} segments (computed)"
    )
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save_svg(fig, "arrangement-guarantees")
    fig2 = go.Figure()
    fig2.add_trace(
        go.Histogram(
            x=t["round_1"]["deviations"],
            xbins=dict(start=0, end=2, size=0.1),
            name="after round 1",
            marker_color="#bbbbbb",
        )
    )
    fig2.add_trace(
        go.Histogram(
            x=t["iterated"]["deviations"],
            xbins=dict(start=0, end=2, size=0.1),
            name="after iteration",
            marker_color=COLOURS["pad"],
            opacity=0.75,
        )
    )
    fig2.add_vline(
        x=SQRT_HALF,
        line_dash="dash",
        line_color=COLOURS["bad"],
        annotation_text="√2/2 g",
    )
    fig2.update_layout(
        barmode="overlay",
        height=420,
        xaxis_title="largest deviation per trial (g)",
        yaxis_title="trials",
        title=f"Deviation of snapped edges from their carriers, {t['trials']} random trials",
        legend=dict(orientation="h", y=-0.25),
    )
    save_plotly(fig2, "arrangement-guarantees")


def _complexity_figure(sizes, rows, radii, counts) -> None:
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(14, 4.2))
    ns = [r["N"] for r in rows]
    a.loglog(
        ns,
        [r["hot_pixels"] for r in rows],
        "o-",
        color=COLOURS["pad"],
        label="hot pixels",
    )
    a.loglog(
        ns,
        [r["level_edges"] + r["slab_edges"] for r in rows],
        "s-",
        color=COLOURS["accent"],
        label="noded edges (all arrangements)",
    )
    a.set_xticks(ns)
    a.set_xticklabels([str(n) for n in ns])
    a.minorticks_off()
    a.set_xlabel("features N")
    a.set_ylabel("count")
    for n, r in zip(ns, rows, strict=True):
        a.annotate(
            f"{r['hot_pixels']:,}",
            (n, r["hot_pixels"]),
            textcoords="offset points",
            xytext=(-4, 7),
            fontsize=7,
            ha="right",
        )
    a.set_title("Production pipeline: size of the arrangements")
    a.legend(fontsize=8, frameon=False)
    a.grid(True, color="#ddd", lw=0.5)
    b.loglog(
        [s["segments"] for s in sizes],
        [s["snap_s"] + s["dcel_s"] for s in sizes],
        "o-",
        color=COLOURS["pad"],
        label="snap rounding + DCEL",
    )
    b.loglog(
        [s["segments"] for s in sizes],
        [
            s["segments"]
            / sizes[0]["segments"]
            * (sizes[0]["snap_s"] + sizes[0]["dcel_s"])
            for s in sizes
        ],
        ":",
        color="#999",
        label="slope 1",
    )
    b.set_xticks([s["segments"] for s in sizes])
    b.set_xticklabels([f"{s['segments'] // 1000}k" for s in sizes])
    b.minorticks_off()
    b.set_xlabel("segments n")
    b.set_ylabel("seconds")
    b.set_title("One arrangement, constant density (synthetic)")
    b.legend(fontsize=8, frameon=False)
    b.grid(True, color="#ddd", lw=0.5)
    c.loglog(radii, counts, "o-", color=COLOURS["pad"], label="chords per quarter arc")
    c.loglog(
        radii,
        [counts[0] * math.sqrt(r / radii[0]) for r in radii],
        ":",
        color="#999",
        label="slope 1/2",
    )
    c.set_xlabel("radius (grid units g)")
    c.set_ylabel("chords")
    c.set_title("Arc witness size, sagitta ≤ g/4")
    c.legend(fontsize=8, frameon=False)
    c.grid(True, color="#ddd", lw=0.5)
    fig.tight_layout()
    save_svg(fig, "arrangement-complexity")
    fi = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Pipeline: arrangement size",
            "Synthetic: time against segments",
            "Arc witness size",
        ),
    )
    fi.add_trace(
        go.Scatter(
            x=ns,
            y=[r["hot_pixels"] for r in rows],
            mode="lines+markers",
            name="hot pixels",
        ),
        row=1,
        col=1,
    )
    fi.add_trace(
        go.Scatter(
            x=ns,
            y=[r["level_edges"] + r["slab_edges"] for r in rows],
            mode="lines+markers",
            name="noded edges",
        ),
        row=1,
        col=1,
    )
    fi.add_trace(
        go.Scatter(
            x=[s["segments"] for s in sizes],
            y=[s["snap_s"] + s["dcel_s"] for s in sizes],
            mode="lines+markers",
            name="snap + DCEL (s)",
        ),
        row=1,
        col=2,
    )
    fi.add_trace(
        go.Scatter(
            x=radii, y=counts, mode="lines+markers", name="chords per quarter arc"
        ),
        row=1,
        col=3,
    )
    for col in (1, 2, 3):
        fi.update_xaxes(type="log", row=1, col=col)
        fi.update_yaxes(type="log", row=1, col=col)
    fi.update_layout(
        height=440,
        legend=dict(orientation="h", y=-0.2),
        title="Cost of building an arrangement (computed)",
    )
    save_plotly(fi, "arrangement-complexity")


def main() -> None:
    """Write the arrangement-guarantee and complexity figures, the arc witness and the numbers the page quotes."""
    t = _trials()
    _guarantees_figure(t)
    layout = _layout_like()
    sizes = [timed_snap(n) for n in (2000, 4000, 8000, 16000, 32000, 64000)]
    rows = scaling_rows()
    radii = [3, 6, 12, 24, 100, 400, 1600, 5000, 8000, 20000, 100000]
    counts = _chord_counts(radii)
    arc = _arc_figure(radii, counts)
    _complexity_figure(sizes, rows, radii, counts)
    numbers = {
        "random_trials": {
            k: (
                {kk: vv for kk, vv in v.items() if kk != "deviations"}
                if isinstance(v, dict)
                else v
            )
            for k, v in t.items()
        },
        "dense_random": dense_drift(),
        "layout_like": layout,
        "level_stats": level_stats(),
        "timing": {
            "rows": sizes,
            "slope_snap_plus_dcel": _slopes(
                [s["segments"] for s in sizes],
                [s["snap_s"] + s["dcel_s"] for s in sizes],
            ),
        },
        "pipeline_rows": rows,
        "pipeline_slopes": {
            "hot_pixels": _slopes(
                [r["N"] for r in rows], [r["hot_pixels"] for r in rows]
            ),
            "level_edges": _slopes(
                [r["N"] for r in rows], [r["level_edges"] for r in rows]
            ),
        },
        "pixel_corners": pixel_corners(),
        "arc_witness": arc,
        "chord_slope": _slopes(radii[3:], counts[3:]),
    }
    write_numbers("arrangement-guarantees", numbers)
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
