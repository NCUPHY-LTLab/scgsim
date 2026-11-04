"""Figures and tables for ``docs/design/sgb/parallelism.qmd``.

Run (from anywhere)::

    .venv/bin/python zh-TW/docs/design/sgb/_figures/parallel.py

Reads the measured scaling data of the prototype (``outputs/scaling.json``, produced by its ``examples/scaling.py``;
this script does not re-measure anything) and recomputes the tile ownership of the running example from the
prototype's own runtime. Outputs go to ``docs/design/sgb/figures/`` (prefix ``parallel-``):

* ``parallel-tiles``      between-pixel tile cuts on the z = 0 level of the running example (computed) and a
                          conceptual pixel-level zoom; html + svg
* ``parallel-speedup``    measured speed-up against workers with the Amdahl curve, and the stage time split of the
                          serial run at N = 20000; html + svg
* ``parallel-tables.md``  markdown tables quoted by the page (speed-up, Amdahl split, tile statistics)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import inset_kit as kit  # noqa: E402  (also puts the prototype on sys.path)
from inset_kit import Label, Line, Panel, Poly, Pts  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from sgbproto.parallel import Runtime  # noqa: E402
from sgbproto.pipeline import build, model_bbox  # noqa: E402
from sgbproto.scenes import pad_in_ground  # noqa: E402

G = kit.G_UM
TILE_COLOURS = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#0072B2", "#D55E00", "#999999", "#8c6d31", "#7570b3"]
SCALING = json.loads((kit.PROTOTYPE_OUTPUTS / "scaling.json").read_text())


# --------------------------------------------------------------------------- tiles


def tile_statistics(spec, result, tiles: int) -> dict:
    """Ownership of the hot pixels of the z = 0 level and the edges that cross a cut without a vertex on it."""
    layer = result.level_layers[1]
    raw = layer.raw["arrangement"]
    rt = Runtime(tiles, 1, model_bbox(spec))
    pixels = sorted({p for e in raw.edges for p in (e.a, e.b)})
    owners = {p: rt.owner(p) for p in pixels}
    crossing = [(e.a, e.b) for e in raw.edges if owners[e.a] != owners[e.b]]
    per_tile = [sum(1 for t in owners.values() if t == k) for k in range(tiles * tiles)]
    return {"tiles": f"{tiles}x{tiles}", "hot_pixels": len(pixels), "pixels_per_tile": per_tile, "edges": len(raw.edges),
            "edges_between_tiles": len(crossing), "x_cuts_um": [(c - 0.5) * G for c in rt._xc[1:-1]], "y_cuts_um": [(c - 0.5) * G for c in rt._yc[1:-1]],
            "_pixels": pixels, "_owners": owners, "_crossing": crossing}


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
        items.append(Poly([[(x - 0.5, y - 0.5), (x + 0.5, y - 0.5), (x + 0.5, y + 0.5), (x - 0.5, y + 0.5)]], own[tile], "rgba(0,0,0,0.6)", 0.8, 0.85,
                          hover=f"hot pixel ({x}, {y}) owned by tile {tile}"))
    items.append(Line([0, 9], [1, 4], "#555", 1.0, "dash", name="source carrier A (whole, unclipped)"))
    items.append(Line([1, 8], [4, 0], "#555", 1.0, "dash", name="source carrier B"))
    items.append(Line([0, 4, 9], [1, 2, 4], kit.INK, 2.4, name="carrier A after snap rounding"))
    items.append(Line([1, 4, 8], [4, 2, 0], kit.INK, 2.4, name="carrier B after snap rounding"))
    items.append(Line([cut, cut], [-1.5, 5.5], kit.PORT, 2.0, "dot", name="tile cut between pixel columns 4 and 5 (x = c - 1/2)"))
    items.append(Label(1.8, 5.1, "tile L owns pixel columns <= 4", "#0b5d8c", 10))
    items.append(Label(7.6, 5.1, "tile R owns pixel columns >= 5", "#a15d00", 10))
    items.append(Label(6.6, -1.0, "edge (4, 2) - (9, 4) crosses the cut with no vertex on it", kit.INK, 9))
    return Panel("conceptual: one cut at a pixel boundary, two carriers", (-1.5, 10.5), (-1.5, 5.5), xlabel="pixel column", ylabel="pixel row", items=items)


def figure_tiles(spec, result) -> dict:
    stats = {t: tile_statistics(spec, result, t) for t in (2, 3, 4)}
    chosen = stats[2]
    items: list = []
    # pixel markers grouped by owning tile
    for tile in range(4):
        pts = [p for p in chosen["_pixels"] if chosen["_owners"][p] == tile]
        if pts:
            items.append(Pts([p[0] * G for p in pts], [p[1] * G for p in pts], TILE_COLOURS[tile], 3.5, name=f"hot pixels owned by tile {tile}",
                             hover=[f"tile {tile}: ({p[0]}, {p[1]}) grid units" for p in pts]))
    for a, b in chosen["_crossing"]:
        items.append(Line([a[0] * G, b[0] * G], [a[1] * G, b[1] * G], kit.PORT, 1.4, name="edge whose ends belong to different tiles"))
    for x in chosen["x_cuts_um"]:
        items.append(Line([x, x], [-155, 155], "#555", 1.0, "dot", name="tile cut (between pixel columns / rows)"))
    for y in chosen["y_cuts_um"]:
        items.append(Line([-155, 155], [y, y], "#555", 1.0, "dot", name="tile cut (between pixel columns / rows)"))
    left = Panel("z = 0 level of the running example (Route A, bands), 2 x 2 tiles", (-60, 60), (-40, 40), xlabel="x (um)", ylabel="y (um)", items=items)
    kit.write_both([left, conceptual_zoom()], "parallel-tiles", ncols=2, svg_size=(13, 6.3), html_size=(1300, 640), shared_legend=True,
                   title="Between-pixel tile cuts: every hot pixel has one owning tile and no seam vertex exists")
    return {t: {k: v for k, v in s.items() if not k.startswith("_")} for t, s in stats.items()}


# --------------------------------------------------------------------------- speed-up


def rows_by(n: int, tiles: int, workers: int):
    return next(r for r in SCALING["rows"] if r["n_features"] == n and r["tiles_per_side"] == tiles and r["workers"] == workers)


def speedups() -> list[dict]:
    base = {r["n_features"]: r for r in SCALING["rows"] if r["tiles_per_side"] == 1 and r["workers"] == 1}
    out = []
    for r in SCALING["rows"]:
        a = r["amdahl"]
        out.append({"n": r["n_features"], "tiles": r["tiles_per_side"], "workers": r["workers"], "build_s": r["build_wall_s"],
                    "speedup": base[r["n_features"]]["build_wall_s"] / r["build_wall_s"], "serial_s": a["serial_stages_s"],
                    "parallel_wall_s": a["parallel_stages_wall_s"], "worker_compute_s": a["parallel_stages_work_s"],
                    "fraction": a["amdahl_parallel_fraction"], "limit": a["amdahl_speedup_limit"], "rss_mb": r["peak_rss_mb"],
                    "digest": r["ids_digest"], "valid": r["validation_passed"]})
    return out


def figure_speedup(rows: list[dict]) -> dict:
    base_row = rows_by(20000, 1, 1)
    fraction = base_row["amdahl"]["amdahl_parallel_fraction"]
    workers = [1, 8, 16, 32]
    series = [(n, t) for n in (1000, 5000, 20000) for t in (4, 8)]
    colours = {1000: "#56B4E9", 5000: "#E69F00", 20000: "#0072B2"}
    amdahl = lambda p: 1 / ((1 - fraction) + fraction / p)  # noqa: E731
    # right panel: stage times of the serial N = 20000 run plus emission
    stage_s = base_row["stage_s"]
    kind = base_row["stage_kind"]
    names = sorted(stage_s, key=lambda k: stage_s[k])
    labels = [*names, "S10 emission (sequential)"]
    values = [*[stage_s[k] for k in names], base_row["emit_s"]]
    bar_colours = [("#0072B2" if kind[k] == "parallel" else "#D55E00") for k in names] + ["#777777"]

    fig = make_subplots(rows=1, cols=2, column_widths=[0.45, 0.55], subplot_titles=("measured speed-up of the derivation", "where the time goes, N = 20000, one process"))
    for n, t in series:
        pts = [(1, 1.0)] + [(w, next(r["speedup"] for r in rows if r["n"] == n and r["tiles"] == t and r["workers"] == w)) for w in workers[1:]]
        fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="lines+markers", name=f"N = {n}, {t}x{t} tiles",
                                 line=dict(color=colours[n], dash="solid" if t == 4 else "dash"), marker=dict(size=6)), row=1, col=1)
    xs = list(range(1, 33))
    fig.add_trace(go.Scatter(x=xs, y=[amdahl(p) for p in xs], mode="lines", name=f"Amdahl model, serial share {1 - fraction:.2f} (measured at N = 20000)", line=dict(color="#555", dash="dot")), row=1, col=1)
    fig.add_trace(go.Bar(y=labels, x=values, orientation="h", marker_color=bar_colours, showlegend=False, text=[f"{v:.1f} s" for v in values], textposition="outside"), row=1, col=2)
    fig.update_xaxes(title="workers", row=1, col=1, type="log", tickvals=[1, 2, 4, 8, 16, 32], ticktext=["1", "2", "4", "8", "16", "32"])
    fig.update_yaxes(title="speed-up over one process (build wall time)", row=1, col=1, range=[0.7, 2.4])
    fig.update_xaxes(title="seconds (blue: parallel stage, orange: serial stage)", row=1, col=2, range=[0, 86])
    fig.update_layout(width=1300, height=640, paper_bgcolor="white", plot_bgcolor="white", font=dict(color=kit.INK, size=12), legend=dict(orientation="h", y=-0.18),
                      title="Tile-parallel derivation in the Python prototype (outputs/scaling.json)")
    kit.ensure_plotly_js()
    fig.write_html(kit.FIGURES / "parallel-speedup.html", include_plotlyjs="directory", full_html=True)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [0.45, 0.55]})
    f.patch.set_facecolor("white")
    for n, t in series:
        pts = [(1, 1.0)] + [(w, next(r["speedup"] for r in rows if r["n"] == n and r["tiles"] == t and r["workers"] == w)) for w in workers[1:]]
        ax1.plot([p[0] for p in pts], [p[1] for p in pts], "o-" if t == 4 else "o--", color=colours[n], label=f"N = {n}, {t}x{t} tiles")
    ax1.plot(xs, [amdahl(p) for p in xs], ":", color="#555", label=f"Amdahl model, serial share {1 - fraction:.2f} (measured at N = 20000)")
    ax1.set_xscale("log", base=2); ax1.set_xticks([1, 2, 4, 8, 16, 32]); ax1.set_xticklabels(["1", "2", "4", "8", "16", "32"])
    ax1.set_ylim(0.7, 2.4); ax1.set_xlabel("workers"); ax1.set_ylabel("speed-up over one process (build wall time)")
    ax1.set_title("measured speed-up of the derivation", fontsize=10); ax1.legend(fontsize=7, loc="upper left"); ax1.grid(alpha=0.3)
    ax2.barh(labels, values, color=bar_colours)
    for i, v in enumerate(values):
        ax2.text(v + 0.8, i, f"{v:.1f} s", va="center", fontsize=7)
    ax2.set_xlim(0, 86); ax2.tick_params(axis="y", labelsize=7)
    ax2.set_xlabel("seconds (blue: parallel stage, orange: serial stage, grey: emission)")
    ax2.set_title("where the time goes, N = 20000, one process", fontsize=10)
    f.tight_layout()
    kit.save_svg(f, "parallel-speedup")
    return {"fraction_n20000_serial": fraction, "amdahl_curve_at_8": amdahl(8), "amdahl_limit": 1 / (1 - fraction)}


# --------------------------------------------------------------------------- tables


def tables(rows: list[dict], tile_stats: dict, extra: dict) -> str:
    out = ["## Measured speed-up (outputs/scaling.json; speed-up = build time of the 1x1, 1-worker run / build time)", "",
           "| N | T x T | workers | build (s) | speed-up | serial stages (s) | parallel stages, wall (s) | worker compute (s) | Amdahl parallel fraction | speed-up limit | peak RSS (MB) |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['n']} | {r['tiles']}x{r['tiles']} | {r['workers']} | {r['build_s']:.1f} | {r['speedup']:.2f} | {r['serial_s']:.1f} | {r['parallel_wall_s']:.1f} | "
                   f"{r['worker_compute_s']:.1f} | {r['fraction']:.2f} | {r['limit']:.2f} | {r['rss_mb']:.0f} |")
    par = [r for r in rows if r["workers"] > 1]
    big = [r for r in par if r["n"] >= 5000]
    base = [r for r in rows if r["workers"] == 1 and r["tiles"] == 1]
    out += ["", "## Ranges quoted in the text", "",
            f"- largest speed-up of any run: {max(r['speedup'] for r in rows):.2f}x",
            f"- speed-up of runs with workers > 1: {min(r['speedup'] for r in par):.2f}x to {max(r['speedup'] for r in par):.2f}x",
            f"- Amdahl parallel fraction, one process (1x1, 1 worker): {min(r['fraction'] for r in base):.2f} to {max(r['fraction'] for r in base):.2f}",
            f"- Amdahl parallel fraction, N >= 5000 with workers: {min(r['fraction'] for r in big):.2f} to {max(r['fraction'] for r in big):.2f}",
            f"- speed-up limit, N >= 5000 with workers: {min(r['limit'] for r in big):.2f}x to {max(r['limit'] for r in big):.2f}x",
            f"- speed-up limit, one process: {min(r['limit'] for r in base):.2f}x to {max(r['limit'] for r in base):.2f}x",
            f"- same identity digest for every (T, workers) of one N: {all(len({r['digest'] for r in rows if r['n'] == n}) == 1 for n in {r['n'] for r in rows})}",
            f"- every case passed validation: {all(r['valid'] for r in rows)}",
            f"- Amdahl curve with serial share from N = 20000, one process: fraction {extra['fraction_n20000_serial']:.3f}, S(8) = {extra['amdahl_curve_at_8']:.2f}x, limit {extra['amdahl_limit']:.2f}x"]
    # serial stage names and their time at N = 20000
    base20 = rows_by(20000, 1, 1)
    out += ["", "## Stage times of the serial run (N = 20000, 1x1 tile, 1 worker)", "", "| stage | kind | seconds |", "|---|---|---|"]
    for k in sorted(base20["stage_s"], key=lambda k: -base20["stage_s"][k]):
        out.append(f"| `{k}` | {base20['stage_kind'][k]} | {base20['stage_s'][k]:.2f} |")
    out.append(f"| S10 emission (no meshing) | sequential | {base20['emit_s']:.2f} |")
    serial_total = sum(v for k, v in base20["stage_s"].items() if base20["stage_kind"][k] == "serial")
    parallel_total = sum(v for k, v in base20["stage_s"].items() if base20["stage_kind"][k] == "parallel")
    derive, emit = base20["build_wall_s"], base20["emit_s"]
    limit = base20["amdahl"]["amdahl_speedup_limit"]
    best = max(r["speedup"] for r in rows if r["n"] == 20000)
    out += ["", f"serial stages total {serial_total:.1f} s, parallel stages total {parallel_total:.1f} s, build wall {base20['build_wall_s']:.1f} s",
            "", "## End to end at N = 20000 (derivation + emission, one process)", "",
            f"- derivation {derive:.1f} s + emission {emit:.1f} s = {derive + emit:.1f} s; emission is {emit / (derive + emit):.0%} of it",
            f"- derivation at its speed-up limit ({limit:.2f}x, {derive / limit:.1f} s): total {derive / limit + emit:.1f} s, {(derive + emit) / (derive / limit + emit):.2f}x end to end",
            f"- derivation at the best measured speed-up ({best:.2f}x): total {derive / best + emit:.1f} s, {(derive + emit) / (derive / best + emit):.2f}x end to end",
            "", "## Worker compute against workers (N = 20000, 4x4 tiles)", "", "| workers | worker compute (s) | serial stages (s) | parallel stages wall (s) |", "|---|---|---|---|"]
    for w in (1, 8, 16, 32):
        a = rows_by(20000, 4, w)["amdahl"]
        out.append(f"| {w} | {a['parallel_stages_work_s']:.1f} | {a['serial_stages_s']:.1f} | {a['parallel_stages_wall_s']:.1f} |")
    out += ["", "## Tile statistics of the running example (z = 0 level, Route A with bands; computed here from the prototype runtime)", "",
            "| tiles | hot pixels | pixels per tile | noded edges | edges whose ends lie in different tiles | x cuts (um) | y cuts (um) |", "|---|---|---|---|---|---|---|"]
    for t, s in tile_stats.items():
        out.append(f"| {s['tiles']} | {s['hot_pixels']} | {s['pixels_per_tile']} | {s['edges']} | {s['edges_between_tiles']} | "
                   f"{', '.join(f'{x:.4f}' for x in s['x_cuts_um'])} | {', '.join(f'{y:.4f}' for y in s['y_cuts_um'])} |")
    return "\n".join(out) + "\n"


def main() -> None:
    kit.ensure_plotly_js()
    spec = pad_in_ground("A", bands_um=(0.1, 0.5, 2.0))
    result = build(spec, keep_raw=True)
    tile_stats = figure_tiles(spec, result)
    rows = speedups()
    extra = figure_speedup(rows)
    (kit.FIGURES / "parallel-tables.md").write_text(tables(rows, tile_stats, extra))
    print("wrote parallel-*")


if __name__ == "__main__":
    main()
