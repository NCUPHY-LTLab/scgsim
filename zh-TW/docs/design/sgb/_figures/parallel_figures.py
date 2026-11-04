"""Figures of the parallelism page: tile ownership of the production runtime, its pixel zoom and the speed-up curves."""

from __future__ import annotations

import collections
import math
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import inset_kit as kit  # noqa: E402  (also puts the production core's tests on sys.path)
import matplotlib.pyplot as plt  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from inset_kit import Label, Line, Panel, Poly, Pts  # noqa: E402
from parallel_tables import G, emission_seconds, rows_by
from plotly.subplots import make_subplots  # noqa: E402
from sgb_emit_cases import CASES  # noqa: E402

from scgsim.geometry.sgb.derivation.labelled import model_bbox  # noqa: E402
from scgsim.geometry.sgb.model import build_model_from_catalogue  # noqa: E402
from scgsim.geometry.sgb.noding.tiling import tile_cut_lines, tile_index  # noqa: E402

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
    """Tile ownership for several tile counts and a conceptual pixel zoom (``parallel-tiles``); returns the tile statistics."""
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


@dataclass(frozen=True)
class _SpeedupPanel:
    """The left panel: measured speed-up curves, the Amdahl curve and the y range that shows both."""

    curves: list[
        tuple[str, str, bool, list[int], list[float]]
    ]  # label, colour, dashed, workers, speed-ups
    amdahl_label: str
    amdahl_workers: list[int]
    amdahl_speedups: list[float]
    y_range: list[float]


@dataclass(frozen=True)
class _StageBars:
    """The right panel: seconds of every stage of the serial N = 20000 run, then of the steps after the derivation."""

    labels: list[str]
    values: list[float]
    colours: list[str]
    top: float


def figure_speedup(scaling: list[dict], rows: list[dict]) -> dict:
    """Speed-up against workers, the Amdahl curve and the stage times of the measured cases (``parallel-speedup``); returns the numbers it draws."""
    base_row = rows_by(scaling, 20000, 1, 1)
    fraction = base_row["amdahl"]["amdahl_parallel_fraction"]
    panel = _speedup_panel(rows, fraction)
    bars = _stage_bars(base_row)
    _speedup_html(panel, bars)
    _speedup_svg(panel, bars)
    return {
        "fraction_n20000_serial": fraction,
        "amdahl_curve_at_8": _amdahl(8, fraction),
        "amdahl_limit": 1 / (1 - fraction),
    }


def _amdahl(workers: float, fraction: float) -> float:
    """Amdahl's speed-up for a parallel fraction on that many workers."""
    return 1 / ((1 - fraction) + fraction / workers)


def _speedup_panel(rows: list[dict], fraction: float) -> _SpeedupPanel:
    workers = [1, 8, 16, 32]
    colours = {1000: "#56B4E9", 5000: "#E69F00", 20000: "#0072B2"}
    curves = []
    for n, t in [(n, t) for n in (1000, 5000, 20000) for t in (4, 8)]:
        speedups = [1.0] + [
            next(
                r["speedup"]
                for r in rows
                if r["n"] == n and r["tiles"] == t and r["workers"] == w
            )
            for w in workers[1:]
        ]
        curves.append(
            (f"N = {n}, {t}x{t} tiles", colours[n], t != 4, workers, speedups)
        )
    xs = list(range(1, 33))
    all_speedups = [r["speedup"] for r in rows]
    return _SpeedupPanel(
        curves=curves,
        amdahl_label=f"Amdahl model, serial share {1 - fraction:.2f} (measured at N = 20000)",
        amdahl_workers=xs,
        amdahl_speedups=[_amdahl(p, fraction) for p in xs],
        y_range=[
            min(0.7, min(all_speedups) - 0.05),
            max(2.4, max(all_speedups) + 0.1, _amdahl(32, fraction) + 0.1),
        ],
    )


def _stage_bars(base_row: dict) -> _StageBars:
    # right panel: stage times of the serial N = 20000 run plus the sequential emission steps
    stage_s = base_row["stage_s"]
    kind = base_row["stage_kind"]
    names = sorted(stage_s, key=lambda k: stage_s[k])
    emission = emission_seconds(base_row)
    values = [*[stage_s[k] for k in names], *emission.values()]
    return _StageBars(
        labels=[*names, *emission],
        values=values,
        colours=[("#0072B2" if kind[k] == "parallel" else "#D55E00") for k in names]
        + ["#777777"] * len(emission),
        top=max(values) * 1.18,
    )


def _speedup_html(panel: _SpeedupPanel, bars: _StageBars) -> None:
    """The interactive speed-up figure."""
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.45, 0.55],
        subplot_titles=(
            "measured speed-up of the derivation",
            "where the time goes, N = 20000, one process",
        ),
    )
    for label, colour, dashed, workers, speedups in panel.curves:
        fig.add_trace(
            go.Scatter(
                x=workers,
                y=speedups,
                mode="lines+markers",
                name=label,
                line=dict(color=colour, dash="dash" if dashed else "solid"),
                marker=dict(size=6),
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=panel.amdahl_workers,
            y=panel.amdahl_speedups,
            mode="lines",
            name=panel.amdahl_label,
            line=dict(color="#555", dash="dot"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            y=bars.labels,
            x=bars.values,
            orientation="h",
            marker_color=bars.colours,
            showlegend=False,
            text=[f"{v:.1f} s" for v in bars.values],
            textposition="outside",
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(
        title="workers",
        row=1,
        col=1,
        type="log",
        tickvals=[1, 2, 4, 8, 16, 32],
        ticktext=["1", "2", "4", "8", "16", "32"],
    )
    fig.update_yaxes(
        title="speed-up over one process (build wall time)",
        row=1,
        col=1,
        range=panel.y_range,
    )
    fig.update_xaxes(
        title="seconds (blue: parallel stage, orange: serial stage, grey: files, reports, sidecar, Gmsh load and check after the derivation)",
        row=1,
        col=2,
        range=[0, bars.top],
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


def _speedup_svg(panel: _SpeedupPanel, bars: _StageBars) -> None:
    """The static copy of the speed-up figure."""
    f, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [0.45, 0.55]}
    )
    f.patch.set_facecolor("white")
    for label, colour, dashed, workers, speedups in panel.curves:
        ax1.plot(
            workers, speedups, "o--" if dashed else "o-", color=colour, label=label
        )
    ax1.plot(
        panel.amdahl_workers,
        panel.amdahl_speedups,
        ":",
        color="#555",
        label=panel.amdahl_label,
    )
    ax1.set_xscale("log", base=2)
    ax1.set_xticks([1, 2, 4, 8, 16, 32])
    ax1.set_xticklabels(["1", "2", "4", "8", "16", "32"])
    ax1.set_ylim(*panel.y_range)
    ax1.set_xlabel("workers")
    ax1.set_ylabel("speed-up over one process (build wall time)")
    ax1.set_title("measured speed-up of the derivation", fontsize=10)
    ax1.legend(fontsize=7, loc="upper left")
    ax1.grid(alpha=0.3)
    ax2.barh(bars.labels, bars.values, color=bars.colours)
    for i, v in enumerate(bars.values):
        ax2.text(v + bars.top * 0.01, i, f"{v:.1f} s", va="center", fontsize=7)
    ax2.set_xlim(0, bars.top)
    ax2.tick_params(axis="y", labelsize=7)
    ax2.set_xlabel(
        "seconds (blue: parallel stage, orange: serial stage, grey: files, reports, sidecar, Gmsh load and check after the derivation)"
    )
    ax2.set_title("where the time goes, N = 20000, one process", fontsize=10)
    f.tight_layout()
    kit.save_svg(f, "parallel-speedup")
