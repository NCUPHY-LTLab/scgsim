"""Scaling, API-comparison and tail figures and tables of the geometry-build study."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plots_common import (
    COLOUR,
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


def scaling_figure(t: dict, ones: dict[int, dict], folder: Path) -> None:
    """Time and memory of the derivation against N, with a linear reference."""
    derive = [ones[n]["build_wall_s"] for n in SIZES]
    memory = [ones[n]["peak_rss_mb"] for n in SIZES]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    left.plot(SIZES, derive, marker="o", color=COLOUR["derive"], label=t["derive"])
    left.plot(
        SIZES,
        [derive[0] * n / SIZES[0] for n in SIZES],
        ls=":",
        color=COLOUR["reference"],
        label=t["slope_one"],
    )
    right.plot(SIZES, memory, marker="o", color=COLOUR["memory"], label=t["memory"])
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
            "Peak RSS",
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
            tail["write"],
            tail["reports"],
            tail["sidecar"],
            tail["load"],
            tail["verify"],
        ],
    }
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(11.5, 3.9), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    offsets = [0.0]
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
        ("t_write", "write", COLOUR["write"]),
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
