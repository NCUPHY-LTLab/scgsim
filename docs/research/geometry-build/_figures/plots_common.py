"""Shared helpers of the geometry-build figure scripts: the data and output folders, the palette, the measured-row readers, the SVG writer and the Markdown table writer."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "zh-TW/docs/design/sgb/figures"
OUTPUT = {
    "en": ROOT / "docs/research/geometry-build/figures",
    "zh-TW": ROOT / "zh-TW/docs/research/geometry-build/figures",
}

CHINESE_FONTS = (
    "'Noto Sans CJK TC', 'PingFang TC', 'Microsoft JhengHei', 'Noto Sans CJK JP'"
)
SIZES = (1000, 5000, 20000)
SCHEDULES = [(1, 1)] + [
    (tiles, workers) for tiles in (4, 8) for workers in (1, 8, 16, 32)
]  # (tiles per side, workers)
COLOUR = {
    "derive": "#2a6f9e",
    "api": "#c0392b",
    "write": "#e08a1e",
    "load": "#7f7f7f",
    "memory": "#2a8f5a",
    "reports": "#8c6bb1",
    "sidecar": "#2a8f5a",
    "verify": "#b8a76a",
    "reference": "#999999",
    "serial": "#d55e00",
    "parallel": "#0072b2",
}

# --------------------------------------------------------------------------- reading the data


def read(name: str) -> dict:
    """A JSON file of the design pages' measurements, parsed."""
    return json.loads((DATA / name).read_text())


def one_process_rows(measured: dict) -> dict[int, dict]:
    """The measured rows of the one-tile, one-worker schedule, by N."""
    return {
        r["n_features"]: r
        for r in measured["rows"]
        if (r["tiles_per_side"], r["workers"]) == (1, 1)
    }


def schedule_rows(measured: dict) -> dict[tuple[int, int, int], dict]:
    """Every measured row, by (N, tiles per side, workers)."""
    return {
        (r["n_features"], r["tiles_per_side"], r["workers"]): r
        for r in measured["rows"]
    }


def slope(x1: float, y1: float, x2: float, y2: float) -> float:
    """Local log-log slope between two points."""
    return math.log(y2 / y1) / math.log(x2 / x1)


def slopes(xs: list[float], ys: list[float]) -> list[float]:
    """The local log-log slopes between consecutive points of a series."""
    return [slope(xs[i], ys[i], xs[i + 1], ys[i + 1]) for i in range(len(xs) - 1)]


def structural_seconds(run: dict) -> dict[str, float]:
    """Time of each structural check of one validator run, including the two checks that run in the earlier stages."""
    checks = dict(run["validator_s"])
    checks["contact_check"] = run["stage_s"]["contact_check"]
    checks["manifold_check"] = run["stage_s"]["manifold_check"]
    return checks


def validator_checks(timing: dict) -> dict[str, list[float]]:
    """Seconds of every check at each size, structural checks and checks on the loaded model together."""
    runs = {r["n"]: r for r in timing["validator"]}
    table: dict[str, list[float]] = {}
    for n in SIZES:
        run = runs[n]
        for name, seconds in {
            **structural_seconds(run),
            **run["verify_phases_s"],
        }.items():
            table.setdefault(name, []).append(seconds)
    return table


def tail_steps(row: dict) -> dict[str, float]:
    """The steps after the derivation, for one size, one process (reports, sidecar and load are the same in both stacks)."""
    e = row["emission"]
    return {
        "write_w1": sum(e["direct_w1"][k] for k in ("plan", "write_brep", "write_xao")),
        "write_w16": sum(
            e["direct_w16"][k] for k in ("plan", "write_brep", "write_xao")
        ),
        "reports": e["direct_w1"]["reports"],
        "sidecar": e["direct_w1"]["sidecar"],
        "load": e["gmsh_load_s"],
        "verify": e["verify_loaded_s"],
    }


def end_to_end(row: dict, derive_seconds: float) -> float:
    """The time from topology to a loadable model: derivation plus the direct writer, reports, sidecar and Gmsh load."""
    tail = tail_steps(row)
    return (
        derive_seconds
        + tail["write_w1"]
        + tail["reports"]
        + tail["sidecar"]
        + tail["load"]
        + tail["verify"]
    )


# --------------------------------------------------------------------------- output helpers


def save_svg(fig, folder: Path, name: str) -> None:
    """Write a Matplotlib figure as an SVG into a figures folder (text kept as text, no date)."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.svg"
    fig.savefig(
        path,
        format="svg",
        bbox_inches="tight",
        facecolor="white",
        metadata={"Date": None},
    )
    plt.close(fig)
    # the text stays text, so the reader's browser picks the Traditional Chinese glyph forms
    path.write_text(path.read_text().replace("'Noto Sans CJK JP'", CHINESE_FONTS))


def table(header: list[str], rows: list[list[str]], align: str) -> str:
    """A Markdown table with the given header, rows and per-column alignment (``l`` or ``r``)."""
    rule = "|" + "|".join({"l": ":---", "r": "---:"}[a] for a in align) + "|"
    lines = ["| " + " | ".join(header) + " |", rule] + [
        "| " + " | ".join(row) + " |" for row in rows
    ]
    return "\n".join(lines) + "\n"


def write_table(folder: Path, name: str, content: str, language: str) -> None:
    """Write a Markdown table into a figures folder; English files keep ASCII parentheses."""
    folder.mkdir(parents=True, exist_ok=True)
    if (
        language == "en"
    ):  # headers are written with Chinese parentheses once, English keeps ASCII ones
        content = content.replace("（", " (").replace("）", ")")
    (folder / f"{name}.md").write_text(content)


def comma(value: float, digits: int = 0) -> str:
    """A number with thousands separators."""
    return f"{value:,.{digits}f}"


def seconds_text(value: float) -> str:
    """Seconds with two decimals below ten and one above."""
    return f"{value:.2f}" if value < 10 else f"{value:.1f}"


def log_axes(ax, sizes=SIZES) -> None:
    """Log-log axes with the measured sizes as ticks."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{n:,}" for n in sizes])
    ax.minorticks_off()
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.grid(True, which="major", color="#ddd", lw=0.6)
