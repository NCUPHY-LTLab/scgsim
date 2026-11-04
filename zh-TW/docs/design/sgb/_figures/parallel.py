"""Figures and tables for ``docs/design/sgb/parallelism.qmd``.

Run, with the Python of the checkout that contains the production SGB core (``scgsim.geometry.sgb``)::

    SGB_CORE=<sgb-core> <sgb-core>/.artifacts/env/bin/python docs/design/sgb/_figures/parallel.py

The tile ownership of the running example (the z = 0 level of the Route A pad-in-ground with bands) is computed with the production
runtime: the pre-dissolution arrangement of the level (``keep_raw``), the model's tile cut lines (``noding.tiling.tile_cut_lines``) and
the owner of every hot pixel (``tile_index``).  The snap-round counts are read from the arrangements of the seven running examples.

The scaling data are not measured here.  They are read from ``figures/why-scaling-measured.json`` (``why_scaling.py measure``): 27 cases
of random Route B layouts of seed 2 (N = 1000, 5000, 20000; 1 x 1, 4 x 4 and 8 x 8 tiles; 1, 8, 16 and 32 workers), each in its own
interpreter so that the peak memory is that case's own, built with ``build_model_from_catalogue(..., tiles=T, workers=W)``.  The one-tile,
one-worker case of each N also wrote the files with ``emit_model``, opened the XAO in Gmsh and ran ``verify_loaded``; its step times
are the emission and load steps quoted here.  Outputs go to ``docs/design/sgb/figures/`` (prefix ``parallel-``):

* ``parallel-tiles``      between-pixel tile cuts on the z = 0 level of the running example (computed) and a
                          conceptual pixel-level zoom; html + svg
* ``parallel-speedup``    measured speed-up against workers with the Amdahl curve, and the time of every stage of the
                          one-process run at N = 20000 followed by the emission and load steps; html + svg
* ``parallel-tables.md``  every table and range of the data in English, for checking
* ``parallel-table-*.md`` the tables the page includes: ``excerpt``, ``full``, ``stages``, ``workers``, ``endtoend``, ``memory``, ``tiles``
* ``parallel-numbers.json``  the numbers of the text that are not in a table
"""

from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import inset_kit as kit  # noqa: E402  (also puts the production core's tests on sys.path)
import matplotlib.pyplot as plt  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from inset_kit import Label, Line, Panel, Poly, Pts  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402
from scgsim.geometry.sgb.derivation.labelled import model_bbox  # noqa: E402
from scgsim.geometry.sgb.model import build_model_from_catalogue  # noqa: E402
from scgsim.geometry.sgb.noding.tiling import tile_cut_lines, tile_index  # noqa: E402
from sgb_emit_cases import CASES  # noqa: E402

G = kit.G_UM
TILE_COLOURS = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#CC79A7",
    "#0072B2",
    "#D55E00",
    "#999999",
    "#8c6d31",
    "#7570b3",
]
SHARED = (
    kit.FIGURES / "why-scaling-measured.json"
)  # production scaling, written by why_scaling.py; read here, not re-measured


# --------------------------------------------------------------------------- tiles


def tile_statistics(model, tiles: int) -> dict:
    """Ownership of the hot pixels of the z = 0 level and the edges that cross a cut without a vertex on it."""
    layer = model.labelled.level_layers[kit.LEVEL_Z0]
    raw = layer.raw_arrangement
    xmin, ymin, xmax, ymax = model_bbox(model.labelled.catalogue)
    x_cuts, y_cuts = (
        tile_cut_lines(xmin, xmax, tiles),
        tile_cut_lines(ymin, ymax, tiles),
    )
    pixels = sorted({p for e in raw.edges for p in (e.a, e.b)})
    owners = {p: tile_index(x_cuts, y_cuts, tiles, p) for p in pixels}
    crossing_edges = [e for e in raw.edges if owners[e.a] != owners[e.b]]
    crossing = [(e.a, e.b) for e in crossing_edges]
    chords = sum(
        any("#" in carrier for carrier in e.carriers) for e in crossing_edges
    )  # a carrier id with "#k" is a chord of an arc's witness
    lengths = sorted(
        math.hypot(e.b[0] - e.a[0], e.b[1] - e.a[1]) * G for e in crossing_edges
    )
    per_tile = [sum(1 for t in owners.values() if t == k) for k in range(tiles * tiles)]
    return {
        "tiles": f"{tiles}x{tiles}",
        "hot_pixels": len(pixels),
        "pixels_per_tile": per_tile,
        "edges": len(raw.edges),
        "edges_between_tiles": len(crossing),
        "arc_chords_between_tiles": chords,
        "median_length_um_between_tiles": lengths[len(lengths) // 2],
        "x_cuts_um": [(c - 0.5) * G for c in x_cuts[1:-1]],
        "y_cuts_um": [(c - 0.5) * G for c in y_cuts[1:-1]],
        "_pixels": pixels,
        "_owners": owners,
        "_crossing": crossing,
    }


def conceptual_zoom() -> Panel:
    """Pixel-level picture of one tile cut (conceptual, hand-placed example)."""
    cut = 4.5
    own = {"L": "#56B4E9", "R": "#E69F00"}
    hot = {(0, 1): "L", (1, 4): "L", (4, 2): "L", (9, 4): "R", (8, 0): "R"}
    items: list = []
    for x in range(-1, 11):
        items.append(Line([x - 0.5, x - 0.5], [-1.5, 5.5], "#d0d0d0", 0.6))
    for y in range(-1, 7):
        items.append(Line([-1.5, 10.5], [y - 0.5, y - 0.5], "#d0d0d0", 0.6))
    for (x, y), tile in hot.items():
        items.append(
            Poly(
                [
                    [
                        (x - 0.5, y - 0.5),
                        (x + 0.5, y - 0.5),
                        (x + 0.5, y + 0.5),
                        (x - 0.5, y + 0.5),
                    ]
                ],
                own[tile],
                "rgba(0,0,0,0.6)",
                0.8,
                0.85,
                hover=f"hot pixel ({x}, {y}) owned by tile {tile}",
            )
        )
    items.append(
        Line(
            [0, 9],
            [1, 4],
            "#555",
            1.0,
            "dash",
            name="source carrier A (whole, unclipped)",
        )
    )
    items.append(Line([1, 8], [4, 0], "#555", 1.0, "dash", name="source carrier B"))
    items.append(
        Line([0, 4, 9], [1, 2, 4], kit.INK, 2.4, name="carrier A after snap rounding")
    )
    items.append(
        Line([1, 4, 8], [4, 2, 0], kit.INK, 2.4, name="carrier B after snap rounding")
    )
    items.append(
        Line(
            [cut, cut],
            [-1.5, 5.5],
            kit.PORT,
            2.0,
            "dot",
            name="tile cut between pixel columns 4 and 5 (x = c - 1/2)",
        )
    )
    items.append(Label(1.8, 5.1, "tile L owns pixel columns <= 4", "#0b5d8c", 10))
    items.append(Label(7.6, 5.1, "tile R owns pixel columns >= 5", "#a15d00", 10))
    items.append(
        Label(
            6.6,
            -1.0,
            "edge (4, 2) - (9, 4) crosses the cut with no vertex on it",
            kit.INK,
            9,
        )
    )
    return Panel(
        "conceptual: one cut at a pixel boundary, two carriers",
        (-1.5, 10.5),
        (-1.5, 5.5),
        xlabel="pixel column",
        ylabel="pixel row",
        items=items,
    )


def figure_tiles(model) -> dict:
    stats = {t: tile_statistics(model, t) for t in (2, 3, 4)}
    chosen = stats[2]
    items: list = []
    # pixel markers grouped by owning tile
    for tile in range(4):
        pts = [p for p in chosen["_pixels"] if chosen["_owners"][p] == tile]
        if pts:
            items.append(
                Pts(
                    [p[0] * G for p in pts],
                    [p[1] * G for p in pts],
                    TILE_COLOURS[tile],
                    3.5,
                    name=f"hot pixels owned by tile {tile}",
                    hover=[f"tile {tile}: ({p[0]}, {p[1]}) grid units" for p in pts],
                )
            )
    for a, b in chosen["_crossing"]:
        items.append(
            Line(
                [a[0] * G, b[0] * G],
                [a[1] * G, b[1] * G],
                kit.PORT,
                1.4,
                name="edge whose ends belong to different tiles",
            )
        )
    for x in chosen["x_cuts_um"]:
        items.append(
            Line(
                [x, x],
                [-155, 155],
                "#555",
                1.0,
                "dot",
                name="tile cut (between pixel columns / rows)",
            )
        )
    for y in chosen["y_cuts_um"]:
        items.append(
            Line(
                [-155, 155],
                [y, y],
                "#555",
                1.0,
                "dot",
                name="tile cut (between pixel columns / rows)",
            )
        )
    left = Panel(
        "z = 0 level of the running example (Route A, bands), 2 x 2 tiles",
        (-60, 60),
        (-40, 40),
        xlabel="x (um)",
        ylabel="y (um)",
        items=items,
    )
    kit.write_both(
        [left, conceptual_zoom()],
        "parallel-tiles",
        ncols=2,
        svg_size=(13, 6.3),
        html_size=(1300, 640),
        shared_legend=True,
        title="Between-pixel tile cuts: every hot pixel has one owning tile and no seam vertex exists",
    )
    return {
        t: {k: v for k, v in s.items() if not k.startswith("_")}
        for t, s in stats.items()
    }


def snap_round_histogram() -> dict:
    """How many snap-rounding rounds the arrangements of the running examples need (``layer.stats['snap_rounds']``)."""
    counts: collections.Counter = collections.Counter()
    builders = {
        "pad_a": lambda: kit.pad_model("A", None),
        "pad_b": lambda: kit.pad_model("B", None),
        "pad_c": lambda: build_model_from_catalogue(kit.pad_catalogue("B"), "C"),
        "pad_a_bands": lambda: kit.pad_model("A", kit.MARGINS),
        "pad_b_bands": lambda: kit.pad_model("B", kit.MARGINS),
        "pad_c_bands": lambda: build_model_from_catalogue(
            kit.pad_catalogue("B"), "C", insets=kit.pad_requests("B")
        ),
    }
    models = {name: build() for name, build in builders.items()}
    models["bridge_b"] = CASES["bridge_b"]()
    arrangements = 0
    for model in models.values():
        for layer in (*model.labelled.slabs, *model.labelled.level_layers):
            counts[layer.stats["snap_rounds"]] += 1
            arrangements += 1
    return {
        "examples": len(models),
        "arrangements": arrangements,
        "by_rounds": dict(sorted(counts.items())),
    }


# --------------------------------------------------------------------------- scaling (read from the shared measurement)


def scaling_rows() -> list[dict]:
    return json.loads(SHARED.read_text())["rows"]


def rows_by(scaling: list[dict], n: int, tiles: int, workers: int) -> dict:
    return next(
        r
        for r in scaling
        if r["n_features"] == n
        and r["tiles_per_side"] == tiles
        and r["workers"] == workers
    )


def emission_seconds(base: dict) -> dict[str, float]:
    """The sequential steps after the derivation in the one-process run: the CAD files, the reports, the sidecar, the Gmsh load and the load-side check."""
    e = base["emission"]
    t = e["direct_w1"]
    return {
        "plan + BREP + XAO files": t["plan"] + t["write_brep"] + t["write_xao"],
        "sliver and validation reports": t["reports"],
        "sidecar": t["sidecar"],
        "Gmsh load of the XAO": e["gmsh_load_s"],
        "verify_loaded": e["verify_loaded_s"],
    }


def speedups(scaling: list[dict]) -> list[dict]:
    base = {
        r["n_features"]: r
        for r in scaling
        if r["tiles_per_side"] == 1 and r["workers"] == 1
    }
    out = []
    for r in scaling:
        a = r["amdahl"]
        out.append(
            {
                "n": r["n_features"],
                "tiles": r["tiles_per_side"],
                "workers": r["workers"],
                "build_s": r["build_wall_s"],
                "speedup": base[r["n_features"]]["build_wall_s"] / r["build_wall_s"],
                "serial_s": a["serial_stages_s"],
                "parallel_wall_s": a["parallel_stages_wall_s"],
                "worker_compute_s": a["parallel_stages_work_s"],
                "fraction": a["amdahl_parallel_fraction"],
                "limit": a["amdahl_speedup_limit"],
                "rss_mb": r["peak_rss_mb"],
                "digest": r["identity_digest"],
                "valid": r["validation_passed"],
                "worker_rss_mb": r["peak_worker_rss_mb"],
            }
        )
    return out


def figure_speedup(scaling: list[dict], rows: list[dict]) -> dict:
    base_row = rows_by(scaling, 20000, 1, 1)
    fraction = base_row["amdahl"]["amdahl_parallel_fraction"]
    workers = [1, 8, 16, 32]
    series = [(n, t) for n in (1000, 5000, 20000) for t in (4, 8)]
    colours = {1000: "#56B4E9", 5000: "#E69F00", 20000: "#0072B2"}
    amdahl = lambda p: 1 / ((1 - fraction) + fraction / p)  # noqa: E731
    # right panel: stage times of the serial N = 20000 run plus the sequential emission steps
    stage_s = base_row["stage_s"]
    kind = base_row["stage_kind"]
    names = sorted(stage_s, key=lambda k: stage_s[k])
    emission = emission_seconds(base_row)
    labels = [*names, *emission]
    values = [*[stage_s[k] for k in names], *emission.values()]
    bar_colours = [
        ("#0072B2" if kind[k] == "parallel" else "#D55E00") for k in names
    ] + ["#777777"] * len(emission)
    top = max(values) * 1.18

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.45, 0.55],
        subplot_titles=(
            "measured speed-up of the derivation",
            "where the time goes, N = 20000, one process",
        ),
    )
    for n, t in series:
        pts = [(1, 1.0)] + [
            (
                w,
                next(
                    r["speedup"]
                    for r in rows
                    if r["n"] == n and r["tiles"] == t and r["workers"] == w
                ),
            )
            for w in workers[1:]
        ]
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in pts],
                y=[p[1] for p in pts],
                mode="lines+markers",
                name=f"N = {n}, {t}x{t} tiles",
                line=dict(color=colours[n], dash="solid" if t == 4 else "dash"),
                marker=dict(size=6),
            ),
            row=1,
            col=1,
        )
    xs = list(range(1, 33))
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=[amdahl(p) for p in xs],
            mode="lines",
            name=f"Amdahl model, serial share {1 - fraction:.2f} (measured at N = 20000)",
            line=dict(color="#555", dash="dot"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            y=labels,
            x=values,
            orientation="h",
            marker_color=bar_colours,
            showlegend=False,
            text=[f"{v:.1f} s" for v in values],
            textposition="outside",
        ),
        row=1,
        col=2,
    )
    all_speedups = [r["speedup"] for r in rows]
    y_range = [
        min(0.7, min(all_speedups) - 0.05),
        max(2.4, max(all_speedups) + 0.1, amdahl(32) + 0.1),
    ]
    fig.update_xaxes(
        title="workers",
        row=1,
        col=1,
        type="log",
        tickvals=[1, 2, 4, 8, 16, 32],
        ticktext=["1", "2", "4", "8", "16", "32"],
    )
    fig.update_yaxes(
        title="speed-up over one process (build wall time)", row=1, col=1, range=y_range
    )
    fig.update_xaxes(
        title="seconds (blue: parallel stage, orange: serial stage, grey: files, reports, sidecar, Gmsh load and check after the derivation)",
        row=1,
        col=2,
        range=[0, top],
    )
    fig.update_layout(
        width=1300,
        height=640,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color=kit.INK, size=12),
        legend=dict(orientation="h", y=-0.18),
        title="Tile-parallel derivation of the production core (figures/why-scaling-measured.json)",
    )
    kit.ensure_plotly_js()
    fig.write_html(
        kit.FIGURES / "parallel-speedup.html",
        include_plotlyjs="directory",
        full_html=True,
    )

    f, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [0.45, 0.55]}
    )
    f.patch.set_facecolor("white")
    for n, t in series:
        pts = [(1, 1.0)] + [
            (
                w,
                next(
                    r["speedup"]
                    for r in rows
                    if r["n"] == n and r["tiles"] == t and r["workers"] == w
                ),
            )
            for w in workers[1:]
        ]
        ax1.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            "o-" if t == 4 else "o--",
            color=colours[n],
            label=f"N = {n}, {t}x{t} tiles",
        )
    ax1.plot(
        xs,
        [amdahl(p) for p in xs],
        ":",
        color="#555",
        label=f"Amdahl model, serial share {1 - fraction:.2f} (measured at N = 20000)",
    )
    ax1.set_xscale("log", base=2)
    ax1.set_xticks([1, 2, 4, 8, 16, 32])
    ax1.set_xticklabels(["1", "2", "4", "8", "16", "32"])
    ax1.set_ylim(*y_range)
    ax1.set_xlabel("workers")
    ax1.set_ylabel("speed-up over one process (build wall time)")
    ax1.set_title("measured speed-up of the derivation", fontsize=10)
    ax1.legend(fontsize=7, loc="upper left")
    ax1.grid(alpha=0.3)
    ax2.barh(labels, values, color=bar_colours)
    for i, v in enumerate(values):
        ax2.text(v + top * 0.01, i, f"{v:.1f} s", va="center", fontsize=7)
    ax2.set_xlim(0, top)
    ax2.tick_params(axis="y", labelsize=7)
    ax2.set_xlabel(
        "seconds (blue: parallel stage, orange: serial stage, grey: files, reports, sidecar, Gmsh load and check after the derivation)"
    )
    ax2.set_title("where the time goes, N = 20000, one process", fontsize=10)
    f.tight_layout()
    kit.save_svg(f, "parallel-speedup")
    return {
        "fraction_n20000_serial": fraction,
        "amdahl_curve_at_8": amdahl(8),
        "amdahl_limit": 1 / (1 - fraction),
    }


# --------------------------------------------------------------------------- tables


def tables(scaling: list[dict], rows: list[dict], tile_stats: dict, extra: dict) -> str:
    out = [
        "## Measured speed-up (figures/why-scaling-measured.json; speed-up = build time of the 1x1, 1-worker run / build time)",
        "",
        "| N | T x T | workers | build (s) | speed-up | serial stages (s) | parallel stages, wall (s) | worker compute (s) | Amdahl parallel fraction | speed-up limit | peak RSS (MB) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        out.append(
            f"| {r['n']} | {r['tiles']}x{r['tiles']} | {r['workers']} | {r['build_s']:.1f} | {r['speedup']:.2f} | {r['serial_s']:.1f} | {r['parallel_wall_s']:.1f} | "
            f"{r['worker_compute_s']:.1f} | {r['fraction']:.2f} | {r['limit']:.2f} | {r['rss_mb']:.0f} |"
        )
    par = [r for r in rows if r["workers"] > 1]
    big = [r for r in par if r["n"] >= 5000]
    base = [r for r in rows if r["workers"] == 1 and r["tiles"] == 1]
    one_worker_tiled = [r for r in rows if r["workers"] == 1 and r["tiles"] > 1]
    out += [
        "",
        "## Ranges quoted in the text",
        "",
        f"- largest speed-up of any run: {max(r['speedup'] for r in rows):.2f}x",
        f"- smallest speed-up of any run: {min(r['speedup'] for r in rows):.2f}x",
        f"- speed-up of runs with workers > 1: {min(r['speedup'] for r in par):.2f}x to {max(r['speedup'] for r in par):.2f}x",
        f"- speed-up of runs with workers > 1 at N = 20000: {min(r['speedup'] for r in par if r['n'] == 20000):.2f}x to {max(r['speedup'] for r in par if r['n'] == 20000):.2f}x",
        f"- speed-up of tiled runs with one worker: {min(r['speedup'] for r in one_worker_tiled):.2f}x to {max(r['speedup'] for r in one_worker_tiled):.2f}x",
        f"- Amdahl parallel fraction, one process (1x1, 1 worker): {min(r['fraction'] for r in base):.2f} to {max(r['fraction'] for r in base):.2f}",
        f"- Amdahl parallel fraction, N >= 5000 with workers: {min(r['fraction'] for r in big):.2f} to {max(r['fraction'] for r in big):.2f}",
        f"- speed-up limit, N >= 5000 with workers: {min(r['limit'] for r in big):.2f}x to {max(r['limit'] for r in big):.2f}x",
        f"- speed-up limit, one process: {min(r['limit'] for r in base):.2f}x to {max(r['limit'] for r in base):.2f}x",
        f"- same identity digest for every (T, workers) of one N: {all(len({r['digest'] for r in rows if r['n'] == n}) == 1 for n in {r['n'] for r in rows})}",
        f"- every case passed validation: {all(r['valid'] for r in rows)}",
        f"- Amdahl curve with serial share from N = 20000, one process: fraction {extra['fraction_n20000_serial']:.3f}, S(8) = {extra['amdahl_curve_at_8']:.2f}x, limit {extra['amdahl_limit']:.2f}x",
    ]
    base20 = rows_by(scaling, 20000, 1, 1)
    out += [
        "",
        "## Stage times of the serial run (N = 20000, 1x1 tile, 1 worker) and the steps after it",
        "",
        "| stage | kind | seconds |",
        "|---|---|---|",
    ]
    for k in sorted(base20["stage_s"], key=lambda k: -base20["stage_s"][k]):
        out.append(
            f"| `{k}` | {base20['stage_kind'][k]} | {base20['stage_s'][k]:.2f} |"
        )
    steps = emission_seconds(base20)
    for name, seconds in steps.items():
        out.append(f"| {name} | after the derivation | {seconds:.2f} |")
    serial_total = sum(
        v for k, v in base20["stage_s"].items() if base20["stage_kind"][k] == "serial"
    )
    parallel_total = sum(
        v for k, v in base20["stage_s"].items() if base20["stage_kind"][k] == "parallel"
    )
    derive = base20["build_wall_s"]
    files = steps["plan + BREP + XAO files"]
    limit = base20["amdahl"]["amdahl_speedup_limit"]
    best = max(r["speedup"] for r in rows if r["n"] == 20000)
    best_row = next(r for r in rows if r["n"] == 20000 and r["speedup"] == best)
    after = sum(steps.values())
    out += [
        "",
        f"serial stages total {serial_total:.1f} s, parallel stages total {parallel_total:.1f} s, sum of stages {serial_total + parallel_total:.1f} s, build wall {derive:.1f} s",
        "",
        "## End to end at N = 20000 (one process: derivation, the files, the reports, the sidecar, Gmsh load, verify_loaded)",
        "",
        f"- derivation {derive:.1f} s, steps after it {after:.1f} s, total {derive + after:.1f} s; the derivation is {derive / (derive + after):.0%} of it",
        f"- derivation at its Amdahl limit ({limit:.2f}x, {derive / limit:.1f} s): total {derive / limit + after:.1f} s, {(derive + after) / (derive / limit + after):.2f}x end to end",
        f"- derivation at the best measured speed-up ({best:.2f}x, {best_row['tiles']}x{best_row['tiles']} tiles, {best_row['workers']} workers): total {derive / best + after:.1f} s, "
        f"{(derive + after) / (derive / best + after):.2f}x end to end",
        f"- only the derivation and the CAD files: {derive + files:.1f} s; with the best speed-up {derive / best + files:.1f} s, {(derive + files) / (derive / best + files):.2f}x",
        "",
        "## Stage by stage at N = 20000: one process against 4x4 tiles with 16 workers",
        "",
        "| stage | kind | 1x1, 1 worker (s) | 4x4, 16 workers (s) | ratio |",
        "|---|---|---|---|---|",
    ]
    wide20 = rows_by(scaling, 20000, 4, 16)
    for k in sorted(base20["stage_s"], key=lambda k: -base20["stage_s"][k]):
        a1, a2 = base20["stage_s"][k], wide20["stage_s"][k]
        out.append(
            f"| `{k}` | {base20['stage_kind'][k]} | {a1:.2f} | {a2:.2f} | {a1 / a2:.2f}x |"
        )
    ws = (8, 16, 32)
    out += [
        "",
        "## Worker compute against workers (N = 20000, 4x4 tiles)",
        "",
        "| workers | worker compute (s) | serial stages (s) | parallel stages wall (s) | build (s) |",
        "|---|---|---|---|---|",
    ]
    for w in (1, *ws):
        r = rows_by(scaling, 20000, 4, w)
        a = r["amdahl"]
        out.append(
            f"| {w} | {a['parallel_stages_work_s']:.1f} | {a['serial_stages_s']:.1f} | {a['parallel_stages_wall_s']:.1f} | {r['build_wall_s']:.1f} |"
        )
    out += [
        "",
        "## Peak memory of the parent process (MB) by tiling and workers",
        "",
        "| N | 1x1, 1 worker | 4x4, 1 worker | 4x4, 8 | 4x4, 16 | 4x4, 32 | 8x8, 1 worker | 8x8, 8 | 8x8, 16 | 8x8, 32 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for n in sorted({r["n_features"] for r in scaling}):
        cells = [
            rows_by(scaling, n, t, w)["peak_rss_mb"]
            for t, w in (
                (1, 1),
                (4, 1),
                (4, 8),
                (4, 16),
                (4, 32),
                (8, 1),
                (8, 8),
                (8, 16),
                (8, 32),
            )
        ]
        out.append(f"| {n} | " + " | ".join(f"{c:.0f}" for c in cells) + " |")
    out += [
        "",
        "## Tile statistics of the running example (z = 0 level, Route A with bands; computed here with the production runtime)",
        "",
        "| tiles | hot pixels | pixels per tile | noded edges | edges whose ends lie in different tiles | of them arc witness chords | median length (um) | x cuts (um) | y cuts (um) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for t, s in tile_stats.items():
        out.append(
            f"| {s['tiles']} | {s['hot_pixels']} | {s['pixels_per_tile']} | {s['edges']} | {s['edges_between_tiles']} | {s['arc_chords_between_tiles']} | {s['median_length_um_between_tiles']:.1f} | "
            f"{', '.join(f'{x:.4f}' for x in s['x_cuts_um'])} | {', '.join(f'{y:.4f}' for y in s['y_cuts_um'])} |"
        )
    return "\n".join(out) + "\n"


def scaling_summary(scaling: list[dict], rows: list[dict]) -> dict:
    """The numbers of the page's text: speed-ups, the serial share, the stages that limit it, the memory."""
    out: dict = {}
    for n in sorted({r["n"] for r in rows}):
        mine = [r for r in rows if r["n"] == n]
        with_workers = [r for r in mine if r["workers"] > 1]
        best = max(with_workers, key=lambda r: r["speedup"])
        base = rows_by(scaling, n, 1, 1)
        serial = sorted(
            (
                (k, v)
                for k, v in base["stage_s"].items()
                if base["stage_kind"][k] == "serial"
            ),
            key=lambda kv: -kv[1],
        )
        parallel = sorted(
            (
                (k, v)
                for k, v in base["stage_s"].items()
                if base["stage_kind"][k] == "parallel"
            ),
            key=lambda kv: -kv[1],
        )
        total = sum(base["stage_s"].values())
        out[n] = {
            "best": {k: best[k] for k in ("tiles", "workers", "speedup", "build_s")},
            "range_with_workers": [
                min(r["speedup"] for r in with_workers),
                max(r["speedup"] for r in with_workers),
            ],
            "range_one_worker_tiled": [
                min(r["speedup"] for r in mine if r["workers"] == 1 and r["tiles"] > 1),
                max(r["speedup"] for r in mine if r["workers"] == 1 and r["tiles"] > 1),
            ],
            "serial_share_of_stages": sum(v for _, v in serial) / total,
            "serial_stages_s": sum(v for _, v in serial),
            "parallel_stages_s": sum(v for _, v in parallel),
            "sum_of_stages_s": total,
            "build_s": base["build_wall_s"],
            "amdahl_fraction": base["amdahl"]["amdahl_parallel_fraction"],
            "amdahl_limit": base["amdahl"]["amdahl_speedup_limit"],
            "largest_serial": serial[:4],
            "largest_parallel": parallel[:4],
            "peak_rss_mb": {
                "1x1": base["peak_rss_mb"],
                "tiled_min": min(r["rss_mb"] for r in mine if r["tiles"] > 1),
                "tiled_max": max(r["rss_mb"] for r in mine if r["tiles"] > 1),
                "tiled_one_worker_max": max(
                    r["rss_mb"] for r in mine if r["tiles"] > 1 and r["workers"] == 1
                ),
            },
            "digests": len({r["digest"] for r in mine}),
            "all_valid": all(r["valid"] for r in mine),
        }
    return out


def _md(header: list[str], rows: list[list[str]], align: str) -> str:
    rule = ["---:" if a == "r" else ":---" for a in align]
    return (
        "\n".join(
            [
                "| " + " | ".join(header) + " |",
                "|" + "|".join(rule) + "|",
                *["| " + " | ".join(row) + " |" for row in rows],
            ]
        )
        + "\n"
    )


def page_tables(scaling: list[dict], rows: list[dict], tile_stats: dict) -> None:
    """The tables the page includes (Traditional Chinese headers), each in its own file so that no number on the page is typed by hand."""
    full, excerpt = [], []
    for r in rows:
        cells = [
            f"{r['n']:,}",
            f"{r['tiles']}x{r['tiles']}",
            str(r["workers"]),
            f"{r['build_s']:.1f}",
            f"{r['speedup']:.2f}",
            f"{r['serial_s']:.1f}",
            f"{r['parallel_wall_s']:.1f}",
            f"{r['worker_compute_s']:.1f}",
            f"{r['fraction']:.2f}",
            f"{r['limit']:.2f}",
            f"{r['rss_mb']:,.0f}",
        ]
        full.append(cells)
        if (r["tiles"], r["workers"]) in ((1, 1), (4, 1), (4, 16), (8, 8)):
            excerpt.append(cells)
    header = [
        "N",
        "Tiles",
        "Workers",
        "Build (s)",
        "加速",
        "Serial stage (s)",
        "Parallel stage，wall (s)",
        "Worker compute (s)",
        "Parallel fraction",
        "Limit",
        "Parent peak RSS (MB)",
    ]
    (kit.FIGURES / "parallel-table-full.md").write_text(
        _md(header, full, "lllrrrrrrrr")
    )
    (kit.FIGURES / "parallel-table-excerpt.md").write_text(
        _md(header, excerpt, "lllrrrrrrrr")
    )

    base, wide = rows_by(scaling, 20000, 1, 1), rows_by(scaling, 20000, 4, 16)
    stage_rows = []
    for k in sorted(base["stage_s"], key=lambda k: -base["stage_s"][k]):
        a, b = base["stage_s"][k], wide["stage_s"][k]
        stage_rows.append(
            [
                f"`{k}`",
                "平行" if base["stage_kind"][k] == "parallel" else "序列",
                f"{a:.2f}",
                f"{b:.2f}",
                f"{a / b:.2f}x",
            ]
        )
    serial_a = sum(
        v for k, v in base["stage_s"].items() if base["stage_kind"][k] == "serial"
    )
    serial_b = sum(
        v for k, v in wide["stage_s"].items() if wide["stage_kind"][k] == "serial"
    )
    parallel_a = sum(
        v for k, v in base["stage_s"].items() if base["stage_kind"][k] == "parallel"
    )
    parallel_b = sum(
        v for k, v in wide["stage_s"].items() if wide["stage_kind"][k] == "parallel"
    )
    stage_rows.append(
        [
            "**序列 stage 合計**",
            "",
            f"**{serial_a:.1f}**",
            f"**{serial_b:.1f}**",
            f"**{serial_a / serial_b:.2f}x**",
        ]
    )
    stage_rows.append(
        [
            "**平行 stage 合計**",
            "",
            f"**{parallel_a:.1f}**",
            f"**{parallel_b:.1f}**",
            f"**{parallel_a / parallel_b:.2f}x**",
        ]
    )
    stage_rows.append(
        [
            "**build 的 wall time**",
            "",
            f"**{base['build_wall_s']:.1f}**",
            f"**{wide['build_wall_s']:.1f}**",
            f"**{base['build_wall_s'] / wide['build_wall_s']:.2f}x**",
        ]
    )
    (kit.FIGURES / "parallel-table-stages.md").write_text(
        _md(
            [
                "Stage",
                "種類",
                "1×1 tile、1 worker (s)",
                "4×4 tile、16 worker (s)",
                "倍數",
            ],
            stage_rows,
            "llrrr",
        )
    )

    work_rows = []
    for w in (1, 8, 16, 32):
        r = rows_by(scaling, 20000, 4, w)
        a = r["amdahl"]
        work_rows.append(
            [
                str(w),
                f"{a['parallel_stages_work_s']:.1f}",
                f"{a['parallel_stages_wall_s']:.1f}",
                f"{a['serial_stages_s']:.1f}",
                f"{r['build_wall_s']:.1f}",
                f"{rows_by(scaling, 20000, 1, 1)['build_wall_s'] / r['build_wall_s']:.2f}",
            ]
        )
    (kit.FIGURES / "parallel-table-workers.md").write_text(
        _md(
            [
                "Workers",
                "Worker compute 合計 (s)",
                "Parallel stage，wall (s)",
                "Serial stage (s)",
                "Build (s)",
                "加速",
            ],
            work_rows,
            "rrrrrr",
        )
    )

    steps = emission_seconds(base)
    after = sum(steps.values())
    e2e = [
        ["derivation（`build_model_from_catalogue`）", f"{base['build_wall_s']:.1f}"]
    ]
    names = {
        "plan + BREP + XAO files": "寫出 BREP 與 XAO（`plan`、`write_brep`、`write_xao`，1 worker）",
        "sliver and validation reports": "sliver report 與 validation report",
        "sidecar": "sidecar",
        "Gmsh load of the XAO": "Gmsh 載入 XAO",
        "verify_loaded": "`verify_loaded`",
    }
    e2e += [[names[k], f"{v:.1f}"] for k, v in steps.items()]
    e2e.append(["**合計**", f"**{base['build_wall_s'] + after:.1f}**"])
    (kit.FIGURES / "parallel-table-endtoend.md").write_text(
        _md(["步驟（N = 20,000，一個 process）", "秒"], e2e, "lr")
    )

    memory = []
    for n in sorted({r["n_features"] for r in scaling}):
        cells = [
            rows_by(scaling, n, t, w)["peak_rss_mb"]
            for t, w in ((1, 1), (4, 1), (4, 16), (8, 1), (8, 16))
        ]
        memory.append([f"{n:,}", *[f"{c:,.0f}" for c in cells]])
    (kit.FIGURES / "parallel-table-memory.md").write_text(
        _md(
            [
                "N",
                "1×1、1 worker",
                "4×4、1 worker",
                "4×4、16 worker",
                "8×8、1 worker",
                "8×8、16 worker",
            ],
            memory,
            "lrrrrr",
        )
    )

    tile_rows = [
        [
            s["tiles"].replace("x", " × "),
            ", ".join(f"{c:,}" for c in s["pixels_per_tile"]),
            f"{s['edges_between_tiles']}（其中 {s['arc_chords_between_tiles']} 條是 arc 的 witness chord）",
        ]
        for s in tile_stats.values()
    ]
    (kit.FIGURES / "parallel-table-tiles.md").write_text(
        _md(
            [
                "Tiles",
                "每個 tile 的 hot pixel 數（tile index = 欄 × T + 列）",
                "兩端分屬不同 tile 的 edge",
            ],
            tile_rows,
            "lll",
        )
    )


def main() -> None:
    kit.ensure_plotly_js()
    model = kit.pad_model("A", kit.MARGINS, keep_raw=True)
    tile_stats = figure_tiles(model)
    scaling = scaling_rows()
    rows = speedups(scaling)
    extra = figure_speedup(scaling, rows)
    (kit.FIGURES / "parallel-tables.md").write_text(
        tables(scaling, rows, tile_stats, extra)
    )
    page_tables(scaling, rows, tile_stats)
    numbers = {
        "snap_rounds": snap_round_histogram(),
        "tiles": tile_stats,
        "scaling": scaling_summary(scaling, rows),
    }
    (kit.FIGURES / "parallel-numbers.json").write_text(json.dumps(numbers, indent=1))
    print(json.dumps(numbers["snap_rounds"]))
    print("wrote parallel-*")


if __name__ == "__main__":
    main()
