"""The tables and numbers the parallelism page quotes, read from the measured scaling file."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import inset_kit as kit  # noqa: E402  (also puts the production core's tests on sys.path)

G = kit.G_UM
SHARED = (
    kit.FIGURES / "why-scaling-measured.json"
)  # production scaling, written by why_scaling.py; read here, not re-measured


# --------------------------------------------------------------------------- scaling (read from the shared measurement)


def scaling_rows() -> list[dict]:
    """The measured cases of the shared ``why-scaling-measured.json``."""
    return json.loads(SHARED.read_text())["rows"]


def rows_by(scaling: list[dict], n: int, tiles: int, workers: int) -> dict:
    """The measured case for one N, tile count and worker count."""
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
    """Every measured case with its speed-up over the one-tile, one-worker run of the same N."""
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


# --------------------------------------------------------------------------- tables


def tables(scaling: list[dict], rows: list[dict], tile_stats: dict, extra: dict) -> str:
    """The Markdown tables of ``parallel-tables.md``: measured speed-up, tile ownership and the Amdahl limit."""
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
    for s in tile_stats.values():
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
    _speedup_tables(rows)
    _stage_table(scaling)
    _worker_table(scaling)
    _end_to_end_table(scaling)
    _memory_table(scaling)
    _tile_table(tile_stats)


def _speedup_tables(rows):
    """The speed-up table of every measured case, in full and as an excerpt."""
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


def _stage_table(scaling):
    """The stage times at N = 20,000 of the serial run against the 4 x 4 tile, 16-worker run."""
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


def _worker_table(scaling):
    """The effect of the worker count at N = 20,000 on 4 x 4 tiles."""
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


def _end_to_end_table(scaling):
    """The steps from topology to a loadable model at N = 20,000, one process."""
    base = rows_by(scaling, 20000, 1, 1)
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


def _memory_table(scaling):
    """The parent's peak memory against N for each tiling and worker count."""
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


def _tile_table(tile_stats):
    """The edges whose two ends lie in different tiles, for each tile count."""
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
