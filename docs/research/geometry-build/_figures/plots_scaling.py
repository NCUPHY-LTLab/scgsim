"""Scaling, API-comparison, tail and parallelism figures and tables of the geometry-build study."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plots_common import (
    COLOUR,
    SCHEDULES,
    SIZES,
    comma,
    log_axes,
    save_svg,
    seconds_text,
    slopes,
    table,
    tail_steps,
    write_table,
)

# --------------------------------------------------------------------------- scaling page


def scaling_figure(
    t: dict, ones: dict[int, dict], by_schedule: dict, folder: Path
) -> None:
    """Time and memory of the derivation against N, with a linear reference."""
    derive = [ones[n]["build_wall_s"] for n in SIZES]
    memory = [ones[n]["peak_rss_mb"] for n in SIZES]
    tiled = [by_schedule[(n, 4, 1)]["peak_rss_mb"] for n in SIZES]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    left.plot(SIZES, derive, marker="o", color=COLOUR["derive"], label=t["derive"])
    left.plot(
        SIZES,
        [derive[0] * n / SIZES[0] for n in SIZES],
        ls=":",
        color=COLOUR["reference"],
        label=t["slope_one"],
    )
    right.plot(
        SIZES, memory, marker="o", color=COLOUR["memory"], label=t["memory_one_tile"]
    )
    right.plot(
        SIZES,
        tiled,
        marker="s",
        ls="--",
        color=COLOUR["memory"],
        label=t["memory_tiles"],
    )
    right.plot(
        SIZES,
        [memory[0] * n / SIZES[0] for n in SIZES],
        ls=":",
        color=COLOUR["reference"],
        label=t["slope_one"],
    )
    for ax, ylabel in ((left, t["seconds"]), (right, t["megabytes"])):
        log_axes(ax)
        ax.set_xlabel(t["features"])
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    save_svg(fig, folder, "scaling")


def scaling_tables(language: str, ones: dict[int, dict], folder: Path) -> None:
    """The table of build time, memory and counts against N."""
    zh = language == "zh-TW"
    header = [
        "N",
        "prism 數" if zh else "Prisms",
        "face 數" if zh else "Faces",
        "推導與驗證（s）" if zh else "Derivation and validation (s)",
        "每個 face（µs）" if zh else "Per face (µs)",
        "Peak RSS（MB）" if zh else "Peak RSS (MB)",
    ]
    rows = []
    for n in SIZES:
        r = ones[n]
        rows.append(
            [
                comma(n),
                comma(r["counts"]["prisms"]),
                comma(r["counts"]["faces"]),
                seconds_text(r["build_wall_s"]),
                comma(r["build_wall_s"] / r["counts"]["faces"] * 1e6),
                comma(r["peak_rss_mb"]),
            ]
        )
    write_table(folder, "scaling-table", table(header, rows, "rrrrrr"), language)
    quantities = [
        (
            "推導與驗證時間" if zh else "Derivation and validation time",
            [ones[n]["build_wall_s"] for n in SIZES],
        ),
        ("face 數" if zh else "Faces", [ones[n]["counts"]["faces"] for n in SIZES]),
        (
            "Peak RSS，單一 tile" if zh else "Peak RSS, one tile",
            [ones[n]["peak_rss_mb"] for n in SIZES],
        ),
    ]
    header = [
        "量" if zh else "Quantity",
        "N = 1,000 到 5,000" if zh else "N = 1,000 to 5,000",
        "N = 5,000 到 20,000" if zh else "N = 5,000 to 20,000",
    ]
    rows = [
        [name] + [f"{s:.2f}" for s in slopes(list(SIZES), values)]
        for name, values in quantities
    ]
    write_table(folder, "scaling-slope-table", table(header, rows, "lrr"), language)


# --------------------------------------------------------------------------- emission and load page


def api_comparison_figure(t: dict, timing: dict, folder: Path) -> None:
    """The Gmsh-API oracle against the direct writer: time against N."""
    heads = timing["head"]
    ns = [h["n"] for h in heads]
    api = [h["api"]["total_s"] for h in heads]
    direct = [
        h["direct"]["plan_s"]
        + h["direct"]["render_brep_s"]
        + h["direct"]["render_xao_s"]
        for h in heads
    ]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    left.plot(ns, api, marker="o", color=COLOUR["api"], label=t["api"])
    left.plot(ns, direct, marker="o", color=COLOUR["write"], label=t["direct"])
    left.plot(
        ns,
        [api[0] * n / ns[0] for n in ns],
        ls=":",
        color=COLOUR["reference"],
        label=t["slope_one"].replace("1,000", "500"),
    )
    log_axes(left, ns)
    left.set_xlabel(t["features"])
    left.set_ylabel(t["seconds"])
    left.legend(fontsize=8, frameon=False, loc="upper left")
    stages = [
        ("occ_build_s", "occ_build", "#e6a08a"),
        ("synchronize_s", "synchronize", "#d97b5b"),
        ("physical_groups_s", "groups", COLOUR["api"]),
        ("write_xao_s", "write_xao", "#7f2a20"),
    ]
    bottom = [0.0] * len(heads)
    positions = range(len(heads))
    for key, label, colour in stages:
        values = [h["api"][key] for h in heads]
        right.bar(
            positions,
            values,
            bottom=bottom,
            color=colour,
            label=t[label],
            edgecolor="white",
        )
        bottom = [b + v for b, v in zip(bottom, values, strict=True)]
    for x, h in zip(positions, heads, strict=True):
        share = h["api"]["physical_groups_s"] / h["api"]["total_s"]
        right.text(
            x,
            h["api"]["total_s"] * 1.02,
            f"{h['api']['total_s']:.1f} s\n({share:.0%})",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    right.set_xticks(list(positions))
    right.set_xticklabels([f"{n:,}" for n in ns])
    right.set_xlabel(t["features"])
    right.set_ylabel(t["seconds"])
    right.set_ylim(0, max(api) * 1.2)
    right.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    save_svg(fig, folder, "api-vs-direct")


def tail_figure(t: dict, ones: dict[int, dict], folder: Path) -> None:
    """The stacked steps from topology to a loadable model at the largest N."""
    row = ones[20000]
    tail = tail_steps(row)
    derive = row["build_wall_s"]
    labels = [
        ("t_derive", COLOUR["derive"]),
        ("t_write", COLOUR["write"]),
        ("t_reports", COLOUR["reports"]),
        ("t_sidecar", COLOUR["sidecar"]),
        ("t_load", COLOUR["load"]),
        ("t_verify", COLOUR["verify"]),
    ]
    stacks = {
        t["tail_one"]: [
            derive,
            tail["write_w1"],
            tail["reports"],
            tail["sidecar"],
            tail["load"],
            tail["verify"],
        ],
        t["tail_sixteen"]: [
            derive,
            tail["write_w16"],
            tail["reports"],
            tail["sidecar"],
            tail["load"],
            tail["verify"],
        ],
    }
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(11.5, 3.9), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    offsets = [0.0, 0.0]
    for i, (key, colour) in enumerate(labels):
        values = [stack[i] for stack in stacks.values()]
        left.barh(
            list(stacks),
            values,
            left=offsets,
            color=colour,
            label=t[key],
            edgecolor="white",
        )
        for y, (start, value) in enumerate(zip(offsets, values, strict=True)):
            if value >= 10:
                left.text(
                    start + value / 2,
                    y,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                    fontweight="bold",
                )
        offsets = [a + b for a, b in zip(offsets, values, strict=True)]
    for y, total in enumerate(offsets):
        left.text(total + 2, y, f"{total:.0f} s", va="center", fontsize=8)
    left.invert_yaxis()
    left.set_xlim(0, max(offsets) * 1.12)
    left.set_xlabel(t["seconds_at_20000"])
    left.legend(
        fontsize=7.5,
        frameon=False,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.2),
    )
    steps = [
        ("t_write", "write_w1", COLOUR["write"]),
        ("t_reports", "reports", COLOUR["reports"]),
        ("t_sidecar", "sidecar", COLOUR["sidecar"]),
        ("t_load", "load", COLOUR["load"]),
        ("t_verify", "verify", COLOUR["verify"]),
    ]
    for label, key, colour in steps:
        right.plot(
            SIZES,
            [tail_steps(ones[n])[key] for n in SIZES],
            marker="o",
            color=colour,
            label=t[label],
        )
    right.plot(
        SIZES,
        [ones[1000]["build_wall_s"] * n / 1000 for n in SIZES],
        ls=":",
        color=COLOUR["reference"],
        label=t["slope_one"] + " (" + t["derive"] + ")",
    )
    log_axes(right)
    right.set_xlabel(t["features"])
    right.set_ylabel(t["seconds"])
    right.legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.tight_layout()
    save_svg(fig, folder, "tail")


# --------------------------------------------------------------------------- parallelism page


def speedups(by_schedule: dict) -> dict[int, dict[tuple[int, int], float]]:
    """The derivation speed-up of every tiled schedule over the serial one, by N."""
    result = {}
    for n in SIZES:
        base = by_schedule[(n, 1, 1)]["build_wall_s"]
        result[n] = {
            (t, w): base / by_schedule[(n, t, w)]["build_wall_s"] for t, w in SCHEDULES
        }
    return result


def parallel_figure(t: dict, by_schedule: dict, folder: Path) -> None:
    """Speed-up against workers with the Amdahl bound, and the serial and parallel stage seconds."""
    gain = speedups(by_schedule)
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(11.5, 4.1), gridspec_kw={"width_ratios": [1, 1.1]}
    )
    colours = {1000: "#6baed6", 5000: "#2171b5", 20000: "#08306b"}
    workers = [1, 8, 16, 32]
    for n in SIZES:
        for tiles, style in ((4, "-"), (8, "--")):
            left.plot(
                range(4),
                [gain[n][(tiles, w)] for w in workers],
                marker="o",
                ls=style,
                color=colours[n],
                label=f"N = {n:,}, {t['tiles_4'] if tiles == 4 else t['tiles_8']}",
            )
    left.axhline(
        by_schedule[(20000, 1, 1)]["amdahl"]["amdahl_speedup_limit"],
        color=COLOUR["serial"],
        ls=":",
        label=t["amdahl"],
    )
    left.axhline(1.0, color=COLOUR["reference"], lw=0.8)
    left.set_xticks(range(4))
    left.set_xticklabels([str(w) for w in workers])
    left.set_xlabel(t["workers"])
    left.set_ylabel(t["speedup"])
    left.set_ylim(0.8, 2.2)
    left.legend(
        fontsize=7.5,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=2,
    )
    one = by_schedule[(20000, 1, 1)]
    stages = sorted(one["stage_s"].items(), key=lambda kv: kv[1], reverse=True)[:12]
    names = [name for name, _ in stages]
    right.barh(
        names,
        [seconds for _, seconds in stages],
        color=[
            COLOUR["serial"]
            if one["stage_kind"][name] == "serial"
            else COLOUR["parallel"]
            for name in names
        ],
    )
    right.invert_yaxis()
    right.tick_params(axis="y", labelsize=8)
    right.set_xlabel(t["stage_seconds"])
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOUR["serial"]),
        plt.Rectangle((0, 0), 1, 1, color=COLOUR["parallel"]),
    ]
    right.legend(
        handles,
        [t["serial_stage"], t["parallel_stage"]],
        fontsize=8,
        frameon=False,
        loc="lower right",
    )
    fig.tight_layout()
    save_svg(fig, folder, "parallel")


def parallel_tables(
    language: str, by_schedule: dict, ones: dict[int, dict], folder: Path
) -> None:
    """The tables of speed-ups and stage times of the tiled schedules."""
    zh = language == "zh-TW"
    gain = speedups(by_schedule)
    header = ["tile × worker" if zh else "Tiles × workers"] + [
        f"N = {n:,}" for n in SIZES
    ]
    rows = []
    for tiles, workers in SCHEDULES:
        label = (
            f"{tiles} × {tiles} tile，{workers} worker"
            if zh
            else f"{tiles} × {tiles} tiles, {workers} worker"
            + ("s" if workers > 1 else "")
        )
        if (tiles, workers) == (1, 1):
            label = "1 個 tile（基準）" if zh else "one tile (baseline)"
        rows.append(
            [label]
            + [
                f"{by_schedule[(n, tiles, workers)]['build_wall_s']:.1f} s（{gain[n][(tiles, workers)]:.2f}×）"
                if zh
                else f"{by_schedule[(n, tiles, workers)]['build_wall_s']:.1f} s ({gain[n][(tiles, workers)]:.2f}×)"
                for n in SIZES
            ]
        )
    write_table(folder, "parallel-table", table(header, rows, "lrrr"), language)

    header = [
        "N",
        "序列 stage（s）" if zh else "Serial stages (s)",
        "平行 stage 的 wall time（s）" if zh else "Parallel stages, wall time (s)",
        "序列比例" if zh else "Serial share",
        "Amdahl 上限" if zh else "Amdahl limit",
    ]
    rows = []
    for n in SIZES:
        a = ones[n]["amdahl"]
        total = a["serial_stages_s"] + a["parallel_stages_wall_s"]
        rows.append(
            [
                comma(n),
                f"{a['serial_stages_s']:.1f}",
                f"{a['parallel_stages_wall_s']:.1f}",
                f"{a['serial_stages_s'] / total:.0%}",
                f"{a['amdahl_speedup_limit']:.2f}×",
            ]
        )
    write_table(folder, "amdahl-table", table(header, rows, "rrrrr"), language)

    header = ["tile × worker" if zh else "Tiles × workers"] + [
        f"N = {n:,}" for n in SIZES
    ]
    rows = []
    for tiles, workers in SCHEDULES:
        label = (
            f"{tiles} × {tiles} tile，{workers} worker"
            if zh
            else f"{tiles} × {tiles} tiles, {workers} worker"
            + ("s" if workers > 1 else "")
        )
        if (tiles, workers) == (1, 1):
            label = "1 個 tile" if zh else "one tile"
        rows.append(
            [label]
            + [comma(by_schedule[(n, tiles, workers)]["peak_rss_mb"]) for n in SIZES]
        )
    write_table(folder, "memory-table", table(header, rows, "lrrr"), language)

    one, many = ones[20000]["stage_s"], by_schedule[(20000, 4, 16)]["stage_s"]
    kinds = ones[20000]["stage_kind"]
    header = [
        "stage",
        "種類" if zh else "Kind",
        "單一行程（s）" if zh else "One process (s)",
        "4 × 4 tile，16 個 worker（s）" if zh else "4 × 4 tiles, 16 workers (s)",
        "倍數" if zh else "Speed-up",
    ]
    rows = []
    for kind in ("serial", "parallel"):
        top = sorted(
            (name for name in one if kinds[name] == kind),
            key=lambda name: one[name],
            reverse=True,
        )[:4]
        for name in top:
            rows.append(
                [
                    f"`{name}`",
                    ("序列" if zh else "serial")
                    if kind == "serial"
                    else ("平行" if zh else "parallel"),
                    f"{one[name]:.1f}",
                    f"{many[name]:.1f}",
                    f"{one[name] / many[name]:.2f}×",
                ]
            )
    write_table(folder, "stage-table", table(header, rows, "llrrr"), language)
