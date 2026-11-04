"""Figures, tables and quoted numbers of the "Geometry build performance" study.

Reads the measurements stored with the SGB design pages and writes, for both languages, the SVG figures and the Markdown
tables the study pages use.  Nothing is measured here and the site render never runs this file::

    python3 docs/research/geometry-build/_figures/plots.py

Only ``matplotlib`` is needed.  The editable data is the set of JSON files below (written by the design figure scripts
``why_scaling.py`` and ``emission.py``; the study's reproduction page lists the commands).  Outputs go to
``docs/research/geometry-build/figures/`` (English) and ``zh-TW/docs/research/geometry-build/figures/`` (Traditional Chinese),
with identical file names; ``numbers.json`` in each folder holds the values the prose quotes.
"""

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

CHINESE_FONTS = "'Noto Sans CJK TC', 'PingFang TC', 'Microsoft JhengHei', 'Noto Sans CJK JP'"
SIZES = (1000, 5000, 20000)
SCHEDULES = [(1, 1)] + [(tiles, workers) for tiles in (4, 8) for workers in (1, 8, 16, 32)]  # (tiles per side, workers)
COLOUR = {"derive": "#2a6f9e", "api": "#c0392b", "write": "#e08a1e", "load": "#7f7f7f", "memory": "#2a8f5a",
          "reports": "#8c6bb1", "sidecar": "#2a8f5a", "verify": "#b8a76a", "reference": "#999999",
          "serial": "#d55e00", "parallel": "#0072b2"}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "figure.facecolor": "white",
    "axes.facecolor": "white", "svg.fonttype": "none",
    "font.family": ["DejaVu Sans", "Noto Sans CJK JP"],  # the second only supplies the Chinese glyphs for text layout
})

TEXT = {
    "en": {
        "features": "features N", "seconds": "seconds", "megabytes": "megabytes",
        "derive": "derivation and validation", "slope_one": "slope 1 from N = 1,000",
        "memory_one_tile": "one tile", "memory_tiles": "4 × 4 tiles, one worker",
        "api": "Gmsh API", "direct": "direct writer", "occ_build": "OCC build", "synchronize": "synchronize",
        "groups": "addPhysicalGroup", "write_xao": "write XAO",
        "tail_one": "one process", "tail_sixteen": "16 workers",
        "t_derive": "derivation and validation", "t_write": "write BREP and XAO", "t_reports": "reports",
        "t_sidecar": "sidecar", "t_load": "Gmsh load", "t_verify": "verify_loaded",
        "seconds_at_20000": "seconds at N = 20,000",
        "check_seconds": "seconds at N = 20,000", "structural": "structural check", "loaded": "check on the loaded model",
        "structural_total": "structural checks, total", "loaded_total": "verify_loaded, total",
        "faces": "faces F",
        "workers": "workers", "speedup": "speed-up of the derivation", "tiles_4": "4 × 4 tiles", "tiles_8": "8 × 8 tiles",
        "amdahl": "limit at N = 20,000", "stage_seconds": "seconds, N = 20,000, one process",
        "serial_stage": "serial stage", "parallel_stage": "parallel stage",
    },
    "zh-TW": {
        "features": "特徵數 N", "seconds": "秒", "megabytes": "MB",
        "derive": "推導與驗證", "slope_one": "以 N = 1,000 為起點的斜率 1",
        "memory_one_tile": "單一 tile", "memory_tiles": "4 × 4 tile，單一 worker",
        "api": "Gmsh API", "direct": "直接寫出", "occ_build": "OCC 建構", "synchronize": "synchronize",
        "groups": "addPhysicalGroup", "write_xao": "寫出 XAO",
        "tail_one": "單一行程", "tail_sixteen": "16 個 worker",
        "t_derive": "推導與驗證", "t_write": "寫出 BREP 與 XAO", "t_reports": "報告",
        "t_sidecar": "sidecar", "t_load": "Gmsh 載入", "t_verify": "verify_loaded",
        "seconds_at_20000": "N = 20,000 時的秒數",
        "check_seconds": "N = 20,000 時的秒數", "structural": "結構檢查", "loaded": "載入後模型的檢查",
        "structural_total": "結構檢查合計", "loaded_total": "verify_loaded 合計",
        "faces": "face 數 F",
        "workers": "worker 數", "speedup": "推導的加速", "tiles_4": "4 × 4 tile", "tiles_8": "8 × 8 tile",
        "amdahl": "N = 20,000 的上限", "stage_seconds": "秒，N = 20,000，單一行程",
        "serial_stage": "序列 stage", "parallel_stage": "平行 stage",
    },
}

CHECKS = {  # name: (kind, English, Chinese); kind is "structural" or "loaded"
    "planarity": ("structural", "each arrangement edge, with the noding kernel", "各 arrangement 的 edge，用 noding kernel"),
    "face_areas": ("loaded", "one Gmsh mass call per face", "每個 face 一次 Gmsh mass 呼叫"),
    "shells": ("structural", "the faces of each shell", "每個 shell 的 face"),
    "counts_and_vertex_placement": ("loaded", "one Gmsh call per vertex", "每個 vertex 一次 Gmsh 呼叫"),
    "boundaries": ("loaded", "one getBoundary call per edge, face and solid", "每個 edge、face、solid 一次 getBoundary"),
    "volumes": ("loaded", "one mass integral per solid", "每個 solid 一次 mass 積分"),
    "physical_groups": ("loaded", "one group query per entity", "每個 entity 查詢一次 group"),
    "groups": ("structural", "group membership of every face and volume", "每個 face 與 volume 的 group 歸屬"),
    "edges_in_faces": ("structural", "the edges and vertical edges of every face", "每個 face 的 edge 與 vertical edge"),
    "contact_check": ("structural", "contact between different Nets", "不同 Net 的接觸"),
    "manifold_check": ("structural", "non-manifold pinches", "non-manifold pinch"),
    "atomic_ids": ("structural", "uniqueness of vertex, edge and face ids", "vertex、edge、face id 的唯一性"),
    "grid_and_lengths": ("structural", "edge lengths and vertex grid positions", "edge 長度與 vertex 的 grid 位置"),
    "face_sides": ("structural", "the two sides of every face", "每個 face 的兩側"),
    "areas": ("structural", "area conservation per arrangement", "各 arrangement 的面積守恆"),
    "prism_areas": ("structural", "prism areas conserved on the cells", "prism 面積在 cell 上守恆"),
    "cylinders": ("structural", "each curved wall is one cylinder", "每面 curved wall 是單一 cylinder"),
    "bands": ("structural", "band areas are monotone", "band 面積單調"),
    "arcs": ("structural", "arc edges do not intersect and stay within the deviation", "arc edge 互不相交且在容許偏差內"),
}

S9_CHECKS = ("grid_and_lengths", "atomic_ids", "planarity", "arcs", "bands", "areas", "prism_areas", "shells", "face_sides", "cylinders", "groups", "edges_in_faces")

# --------------------------------------------------------------------------- reading the data


def read(name: str) -> dict:
    return json.loads((DATA / name).read_text())


def one_process_rows(measured: dict) -> dict[int, dict]:
    return {r["n_features"]: r for r in measured["rows"] if (r["tiles_per_side"], r["workers"]) == (1, 1)}


def schedule_rows(measured: dict) -> dict[tuple[int, int, int], dict]:
    return {(r["n_features"], r["tiles_per_side"], r["workers"]): r for r in measured["rows"]}


def slope(x1: float, y1: float, x2: float, y2: float) -> float:
    """Local log-log slope between two points."""
    return math.log(y2 / y1) / math.log(x2 / x1)


def slopes(xs: list[float], ys: list[float]) -> list[float]:
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
        for name, seconds in {**structural_seconds(run), **run["verify_phases_s"]}.items():
            table.setdefault(name, []).append(seconds)
    return table


def tail_steps(row: dict) -> dict[str, float]:
    """The steps after the derivation, for one size, one process (reports, sidecar and load are the same in both stacks)."""
    e = row["emission"]
    return {"write_w1": sum(e["direct_w1"][k] for k in ("plan", "write_brep", "write_xao")),
            "write_w16": sum(e["direct_w16"][k] for k in ("plan", "write_brep", "write_xao")),
            "reports": e["direct_w1"]["reports"], "sidecar": e["direct_w1"]["sidecar"],
            "load": e["gmsh_load_s"], "verify": e["verify_loaded_s"]}


def end_to_end(row: dict, derive_seconds: float) -> float:
    tail = tail_steps(row)
    return derive_seconds + tail["write_w1"] + tail["reports"] + tail["sidecar"] + tail["load"] + tail["verify"]


# --------------------------------------------------------------------------- output helpers


def save_svg(fig, folder: Path, name: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.svg"
    fig.savefig(path, format="svg", bbox_inches="tight", facecolor="white", metadata={"Date": None})
    plt.close(fig)
    # the text stays text, so the reader's browser picks the Traditional Chinese glyph forms
    path.write_text(path.read_text().replace("'Noto Sans CJK JP'", CHINESE_FONTS))


def table(header: list[str], rows: list[list[str]], align: str) -> str:
    rule = "|" + "|".join({"l": ":---", "r": "---:"}[a] for a in align) + "|"
    lines = ["| " + " | ".join(header) + " |", rule] + ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def write_table(folder: Path, name: str, content: str, language: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    if language == "en":  # headers are written with Chinese parentheses once, English keeps ASCII ones
        content = content.replace("（", " (").replace("）", ")")
    (folder / f"{name}.md").write_text(content)


def comma(value: float, digits: int = 0) -> str:
    return f"{value:,.{digits}f}"


def seconds_text(value: float) -> str:
    return f"{value:.2f}" if value < 10 else f"{value:.1f}"


def log_axes(ax, sizes=SIZES) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{n:,}" for n in sizes])
    ax.minorticks_off()
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.grid(True, which="major", color="#ddd", lw=0.6)


# --------------------------------------------------------------------------- scaling page


def scaling_figure(t: dict, ones: dict[int, dict], by_schedule: dict, folder: Path) -> None:
    derive = [ones[n]["build_wall_s"] for n in SIZES]
    memory = [ones[n]["peak_rss_mb"] for n in SIZES]
    tiled = [by_schedule[(n, 4, 1)]["peak_rss_mb"] for n in SIZES]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    left.plot(SIZES, derive, marker="o", color=COLOUR["derive"], label=t["derive"])
    left.plot(SIZES, [derive[0] * n / SIZES[0] for n in SIZES], ls=":", color=COLOUR["reference"], label=t["slope_one"])
    right.plot(SIZES, memory, marker="o", color=COLOUR["memory"], label=t["memory_one_tile"])
    right.plot(SIZES, tiled, marker="s", ls="--", color=COLOUR["memory"], label=t["memory_tiles"])
    right.plot(SIZES, [memory[0] * n / SIZES[0] for n in SIZES], ls=":", color=COLOUR["reference"], label=t["slope_one"])
    for ax, ylabel in ((left, t["seconds"]), (right, t["megabytes"])):
        log_axes(ax)
        ax.set_xlabel(t["features"])
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    save_svg(fig, folder, "scaling")


def scaling_tables(language: str, ones: dict[int, dict], folder: Path) -> None:
    zh = language == "zh-TW"
    header = ["N", "prism 數" if zh else "Prisms", "face 數" if zh else "Faces", "推導與驗證（s）" if zh else "Derivation and validation (s)",
              "每個 face（µs）" if zh else "Per face (µs)", "Peak RSS（MB）" if zh else "Peak RSS (MB)"]
    rows = []
    for n in SIZES:
        r = ones[n]
        rows.append([comma(n), comma(r["counts"]["prisms"]), comma(r["counts"]["faces"]), seconds_text(r["build_wall_s"]),
                     comma(r["build_wall_s"] / r["counts"]["faces"] * 1e6), comma(r["peak_rss_mb"])])
    write_table(folder, "scaling-table", table(header, rows, "rrrrrr"), language)
    quantities = [("推導與驗證時間" if zh else "Derivation and validation time", [ones[n]["build_wall_s"] for n in SIZES]),
                  ("face 數" if zh else "Faces", [ones[n]["counts"]["faces"] for n in SIZES]),
                  ("Peak RSS，單一 tile" if zh else "Peak RSS, one tile", [ones[n]["peak_rss_mb"] for n in SIZES])]
    header = ["量" if zh else "Quantity", "N = 1,000 到 5,000" if zh else "N = 1,000 to 5,000", "N = 5,000 到 20,000" if zh else "N = 5,000 to 20,000"]
    rows = [[name] + [f"{s:.2f}" for s in slopes(list(SIZES), values)] for name, values in quantities]
    write_table(folder, "scaling-slope-table", table(header, rows, "lrr"), language)


# --------------------------------------------------------------------------- emission and load page


def api_comparison_figure(t: dict, timing: dict, folder: Path) -> None:
    heads = timing["head"]
    ns = [h["n"] for h in heads]
    api = [h["api"]["total_s"] for h in heads]
    direct = [h["direct"]["plan_s"] + h["direct"]["render_brep_s"] + h["direct"]["render_xao_s"] for h in heads]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    left.plot(ns, api, marker="o", color=COLOUR["api"], label=t["api"])
    left.plot(ns, direct, marker="o", color=COLOUR["write"], label=t["direct"])
    left.plot(ns, [api[0] * n / ns[0] for n in ns], ls=":", color=COLOUR["reference"], label=t["slope_one"].replace("1,000", "500"))
    log_axes(left, ns)
    left.set_xlabel(t["features"])
    left.set_ylabel(t["seconds"])
    left.legend(fontsize=8, frameon=False, loc="upper left")
    stages = [("occ_build_s", "occ_build", "#e6a08a"), ("synchronize_s", "synchronize", "#d97b5b"),
              ("physical_groups_s", "groups", COLOUR["api"]), ("write_xao_s", "write_xao", "#7f2a20")]
    bottom = [0.0] * len(heads)
    positions = range(len(heads))
    for key, label, colour in stages:
        values = [h["api"][key] for h in heads]
        right.bar(positions, values, bottom=bottom, color=colour, label=t[label], edgecolor="white")
        bottom = [b + v for b, v in zip(bottom, values)]
    for x, h in zip(positions, heads):
        share = h["api"]["physical_groups_s"] / h["api"]["total_s"]
        right.text(x, h["api"]["total_s"] * 1.02, f"{h['api']['total_s']:.1f} s\n({share:.0%})", ha="center", va="bottom", fontsize=8)
    right.set_xticks(list(positions))
    right.set_xticklabels([f"{n:,}" for n in ns])
    right.set_xlabel(t["features"])
    right.set_ylabel(t["seconds"])
    right.set_ylim(0, max(api) * 1.2)
    right.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    save_svg(fig, folder, "api-vs-direct")


def tail_figure(t: dict, ones: dict[int, dict], folder: Path) -> None:
    row = ones[20000]
    tail = tail_steps(row)
    derive = row["build_wall_s"]
    labels = [("t_derive", COLOUR["derive"]), ("t_write", COLOUR["write"]), ("t_reports", COLOUR["reports"]),
              ("t_sidecar", COLOUR["sidecar"]), ("t_load", COLOUR["load"]), ("t_verify", COLOUR["verify"])]
    stacks = {t["tail_one"]: [derive, tail["write_w1"], tail["reports"], tail["sidecar"], tail["load"], tail["verify"]],
              t["tail_sixteen"]: [derive, tail["write_w16"], tail["reports"], tail["sidecar"], tail["load"], tail["verify"]]}
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 3.9), gridspec_kw={"width_ratios": [1.15, 1]})
    offsets = [0.0, 0.0]
    for i, (key, colour) in enumerate(labels):
        values = [stack[i] for stack in stacks.values()]
        left.barh(list(stacks), values, left=offsets, color=colour, label=t[key], edgecolor="white")
        for y, (start, value) in enumerate(zip(offsets, values)):
            if value >= 10:
                left.text(start + value / 2, y, f"{value:.1f}", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
        offsets = [a + b for a, b in zip(offsets, values)]
    for y, total in enumerate(offsets):
        left.text(total + 2, y, f"{total:.0f} s", va="center", fontsize=8)
    left.invert_yaxis()
    left.set_xlim(0, max(offsets) * 1.12)
    left.set_xlabel(t["seconds_at_20000"])
    left.legend(fontsize=7.5, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.2))
    steps = [("t_write", "write_w1", COLOUR["write"]), ("t_reports", "reports", COLOUR["reports"]), ("t_sidecar", "sidecar", COLOUR["sidecar"]),
             ("t_load", "load", COLOUR["load"]), ("t_verify", "verify", COLOUR["verify"])]
    for label, key, colour in steps:
        right.plot(SIZES, [tail_steps(ones[n])[key] for n in SIZES], marker="o", color=colour, label=t[label])
    right.plot(SIZES, [ones[1000]["build_wall_s"] * n / 1000 for n in SIZES], ls=":", color=COLOUR["reference"], label=t["slope_one"] + " (" + t["derive"] + ")")
    log_axes(right)
    right.set_xlabel(t["features"])
    right.set_ylabel(t["seconds"])
    right.legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.tight_layout()
    save_svg(fig, folder, "tail")


def emission_tables(language: str, ones: dict[int, dict], timing: dict, oracle: dict, folder: Path) -> None:
    zh = language == "zh-TW"
    header = ["N", "寫出 BREP 與 XAO，單一行程（s）" if zh else "Write BREP and XAO, one process (s)",
              "寫出，16 個 worker（s）" if zh else "Write, 16 workers (s)", "報告（s）" if zh else "Reports (s)", "Sidecar（s）",
              "Gmsh 載入 XAO（s）" if zh else "Gmsh load of the XAO (s)", "verify_loaded（s）"]
    rows = []
    for n in SIZES:
        tail = tail_steps(ones[n])
        rows.append([comma(n)] + [seconds_text(tail[k]) for k in ("write_w1", "write_w16", "reports", "sidecar", "load", "verify")])
    write_table(folder, "tail-table", table(header, rows, "rrrrrrr"), language)

    runs = {r["n"]: r for r in timing["validator"]}
    header = ["N", "寫出的 face 數" if zh else "Faces written", "BREP（MB）", "XAO（MB）", "Sidecar（MB）",
              "整條路徑的 peak RSS（MB）" if zh else "Peak RSS of the whole path (MB)"]
    rows = []
    for n in SIZES:
        e = ones[n]["emission"]
        mb = {k: v / 1e6 for k, v in e["file_bytes"].items()}
        rows.append([comma(n), comma(e["written_counts"]["faces"]), comma(mb["brep"], 1), comma(mb["xao"], 1), comma(mb["sidecar"], 1),
                     comma(runs[n]["peak_rss_mb"])])
    write_table(folder, "files-table", table(header, rows, "rrrrrr"), language)

    header = ["N", "寫出的 face 數" if zh else "Faces", "physical group 數" if zh else "Physical groups", "OCC build（s）", "synchronize（s）",
              "addPhysicalGroup（s）", "寫出 XAO（s）" if zh else "Write XAO (s)", "Gmsh API 合計（s）" if zh else "Gmsh API total (s)",
              "直接寫出（s）" if zh else "Direct writer (s)", "倍數" if zh else "Ratio", "group 佔比" if zh else "Group share"]
    rows = []
    for h in timing["head"]:
        api, direct = h["api"], h["direct"]
        direct_total = direct["plan_s"] + direct["render_brep_s"] + direct["render_xao_s"]
        rows.append([comma(h["n"]), comma(direct["plan_counts"]["faces"]), comma(h["groups"]), f"{api['occ_build_s']:.2f}", f"{api['synchronize_s']:.2f}",
                     seconds_text(api["physical_groups_s"]), f"{api['write_xao_s']:.2f}", seconds_text(api["total_s"]), f"{direct_total:.2f}",
                     f"{api['total_s'] / direct_total:.0f}×", f"{api['physical_groups_s'] / api['total_s']:.0%}"])
    write_table(folder, "api-table", table(header, rows, "rrrrrrrrrrr"), language)

    header = ["N", "Gmsh API：substrate（µm³）", "Gmsh API：vacuum（µm³）", "直接寫出：兩個 solid（µm³）" if zh else "Direct writer: the two solids (µm³)"]
    if not zh:
        header[1:3] = ["Gmsh API: substrate (µm³)", "Gmsh API: vacuum (µm³)"]
    rows = []
    for r in oracle["rows"]:
        mass = r["solid_mass_um3"]
        direct_masses = ", ".join(f"{m:.3e}" for m in mass["direct_writer"])
        rows.append([comma(r["n_features"]), f"{mass['gmsh_api_oracle']['substrate']:.3e}", f"{mass['gmsh_api_oracle']['vacuum']:.3e}", direct_masses])
    write_table(folder, "mass-table", table(header, rows, "rrrr"), language)

    header = ["N", "API 寫出的 XAO（s）" if zh else "XAO written by the Gmsh API (s)", "直接寫出的 XAO（s）" if zh else "XAO from the direct writer (s)",
              "直接寫出的 BREP（s）" if zh else "BREP from the direct writer (s)"]
    rows = [[comma(h["n"]), f"{h['load']['api_xao_s']:.2f}", f"{h['load']['direct_xao_s']:.2f}", f"{h['load']['direct_brep_s']:.2f}"] for h in timing["head"]]
    write_table(folder, "load-table", table(header, rows, "rrrr"), language)


# --------------------------------------------------------------------------- validation page


def validation_figure(t: dict, timing: dict, folder: Path) -> None:
    checks = validator_checks(timing)
    ordered = sorted(checks, key=lambda name: checks[name][-1], reverse=True)[:12]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})
    colours = [COLOUR["derive"] if CHECKS[name][0] == "structural" else COLOUR["verify"] for name in ordered]
    left.barh(ordered, [checks[name][-1] for name in ordered], color=colours)
    left.invert_yaxis()
    left.set_xlabel(t["check_seconds"])
    left.tick_params(axis="y", labelsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOUR["derive"]), plt.Rectangle((0, 0), 1, 1, color=COLOUR["verify"])]
    left.legend(handles, [t["structural"], t["loaded"]], fontsize=8, frameon=False, loc="lower right")
    structural = [sum(v[i] for k, v in checks.items() if CHECKS[k][0] == "structural") for i in range(3)]
    loaded = [sum(v[i] for k, v in checks.items() if CHECKS[k][0] == "loaded") for i in range(3)]
    right.plot(SIZES, structural, marker="o", color=COLOUR["derive"], label=t["structural_total"])
    right.plot(SIZES, loaded, marker="o", color=COLOUR["verify"], label=t["loaded_total"])
    right.plot(SIZES, [structural[0] * n / 1000 for n in SIZES], ls=":", color=COLOUR["reference"], label=t["slope_one"])
    log_axes(right)
    right.set_xlabel(t["features"])
    right.set_ylabel(t["seconds"])
    right.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    save_svg(fig, folder, "validation")


def validation_tables(language: str, timing: dict, folder: Path) -> None:
    zh = language == "zh-TW"
    checks = validator_checks(timing)
    faces = [r["counts"]["faces"] for r in sorted(timing["validator"], key=lambda r: r["n"])]
    header = ["檢查" if zh else "Check", "種類" if zh else "Kind", "檢查什麼" if zh else "What it checks",
              "N = 1,000（s）", "N = 5,000（s）", "N = 20,000（s）", "斜率" if zh else "Slope"]
    rows = []
    for name in sorted(checks, key=lambda k: checks[k][-1], reverse=True):
        kind, english, chinese = CHECKS[name]
        kind_text = {"structural": "結構" if zh else "structural", "loaded": "載入後" if zh else "loaded model"}[kind]
        values = checks[name]
        overall = slope(faces[0], values[0], faces[-1], values[-1])
        rows.append([f"`{name}`", kind_text, chinese if zh else english] + [f"{v:.3f}" for v in values] + [f"{overall:.2f}"])
    write_table(folder, "validator-table", table(header, rows, "lllrrrr"), language)

    header = ["N", "face 數" if zh else "Faces", "結構檢查合計（s）" if zh else "Structural checks, total (s)",
              "載入後檢查合計（s）" if zh else "Checks on the loaded model, total (s)", "結構檢查佔該次 build 的比例" if zh else "Structural share of that run's build"]
    rows = []
    for run in sorted(timing["validator"], key=lambda r: r["n"]):
        rows.append([comma(run["n"]), comma(run["counts"]["faces"]), f"{sum(structural_seconds(run).values()):.2f}",
                     f"{run['verify_s']:.2f}", f"{sum(structural_seconds(run).values()) / run['build_s']:.0%}"])
    write_table(folder, "validator-total-table", table(header, rows, "rrrrr"), language)


def occt_table(language: str, example: dict, folder: Path) -> None:
    zh = language == "zh-TW"
    catalogue, occt = example["catalogue"], example["occt"]
    header = ["模型組" if zh else "Model set", "報告數" if zh else "Reports", "BRepCheck_Analyzer（ms）", "BOPAlgo_ArgumentAnalyzer（ms）",
              "OCCT 檢查合計（ms）" if zh else "All OCCT checks (ms)"]
    ms = catalogue["occt_ms"]
    rows = [["63 個組合中通過的 54 個（23 個 fixture）" if zh else "54 passing combinations of 63 (23 fixtures)", str(catalogue["passed"]),
             f"{ms['brep_check_ms'][0]:.1f}–{ms['brep_check_ms'][1]:.1f}", f"{ms['bop_ms'][0]:.1f}–{ms['bop_ms'][1]:.1f}",
             f"{ms['occt_total_ms'][0]:.1f}–{ms['occt_total_ms'][1]:.1f}"],
            ["7 個範例模型，BREP 與 XAO" if zh else "Seven example models, BREP and XAO", str(occt["reports"]),
             f"{occt['brep_check_ms'][0]:.1f}–{occt['brep_check_ms'][1]:.1f}", f"{occt['bop_ms'][0]:.1f}–{occt['bop_ms'][1]:.1f}",
             f"{occt['total_ms'][0]:.1f}–{occt['total_ms'][1]:.1f}"]]
    write_table(folder, "occt-table", table(header, rows, "lrrrr"), language)


# --------------------------------------------------------------------------- parallelism page


def speedups(by_schedule: dict) -> dict[int, dict[tuple[int, int], float]]:
    result = {}
    for n in SIZES:
        base = by_schedule[(n, 1, 1)]["build_wall_s"]
        result[n] = {(t, w): base / by_schedule[(n, t, w)]["build_wall_s"] for t, w in SCHEDULES}
    return result


def parallel_figure(t: dict, by_schedule: dict, folder: Path) -> None:
    gain = speedups(by_schedule)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.1), gridspec_kw={"width_ratios": [1, 1.1]})
    colours = {1000: "#6baed6", 5000: "#2171b5", 20000: "#08306b"}
    workers = [1, 8, 16, 32]
    for n in SIZES:
        for tiles, style in ((4, "-"), (8, "--")):
            left.plot(range(4), [gain[n][(tiles, w)] for w in workers], marker="o", ls=style, color=colours[n],
                      label=f"N = {n:,}, {t['tiles_4'] if tiles == 4 else t['tiles_8']}")
    left.axhline(by_schedule[(20000, 1, 1)]["amdahl"]["amdahl_speedup_limit"], color=COLOUR["serial"], ls=":", label=t["amdahl"])
    left.axhline(1.0, color=COLOUR["reference"], lw=0.8)
    left.set_xticks(range(4))
    left.set_xticklabels([str(w) for w in workers])
    left.set_xlabel(t["workers"])
    left.set_ylabel(t["speedup"])
    left.set_ylim(0.8, 2.2)
    left.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    one = by_schedule[(20000, 1, 1)]
    stages = sorted(one["stage_s"].items(), key=lambda kv: kv[1], reverse=True)[:12]
    names = [name for name, _ in stages]
    right.barh(names, [seconds for _, seconds in stages],
               color=[COLOUR["serial"] if one["stage_kind"][name] == "serial" else COLOUR["parallel"] for name in names])
    right.invert_yaxis()
    right.tick_params(axis="y", labelsize=8)
    right.set_xlabel(t["stage_seconds"])
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOUR["serial"]), plt.Rectangle((0, 0), 1, 1, color=COLOUR["parallel"])]
    right.legend(handles, [t["serial_stage"], t["parallel_stage"]], fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    save_svg(fig, folder, "parallel")


def parallel_tables(language: str, by_schedule: dict, ones: dict[int, dict], folder: Path) -> None:
    zh = language == "zh-TW"
    gain = speedups(by_schedule)
    header = ["tile × worker" if zh else "Tiles × workers"] + [f"N = {n:,}" for n in SIZES]
    rows = []
    for tiles, workers in SCHEDULES:
        label = f"{tiles} × {tiles} tile，{workers} worker" if zh else f"{tiles} × {tiles} tiles, {workers} worker" + ("s" if workers > 1 else "")
        if (tiles, workers) == (1, 1):
            label = "1 個 tile（基準）" if zh else "one tile (baseline)"
        rows.append([label] + [f"{by_schedule[(n, tiles, workers)]['build_wall_s']:.1f} s（{gain[n][(tiles, workers)]:.2f}×）" if zh
                               else f"{by_schedule[(n, tiles, workers)]['build_wall_s']:.1f} s ({gain[n][(tiles, workers)]:.2f}×)" for n in SIZES])
    write_table(folder, "parallel-table", table(header, rows, "lrrr"), language)

    header = ["N", "序列 stage（s）" if zh else "Serial stages (s)", "平行 stage 的 wall time（s）" if zh else "Parallel stages, wall time (s)",
              "序列比例" if zh else "Serial share", "Amdahl 上限" if zh else "Amdahl limit"]
    rows = []
    for n in SIZES:
        a = ones[n]["amdahl"]
        total = a["serial_stages_s"] + a["parallel_stages_wall_s"]
        rows.append([comma(n), f"{a['serial_stages_s']:.1f}", f"{a['parallel_stages_wall_s']:.1f}", f"{a['serial_stages_s'] / total:.0%}",
                     f"{a['amdahl_speedup_limit']:.2f}×"])
    write_table(folder, "amdahl-table", table(header, rows, "rrrrr"), language)

    header = ["tile × worker" if zh else "Tiles × workers"] + [f"N = {n:,}" for n in SIZES]
    rows = []
    for tiles, workers in SCHEDULES:
        label = f"{tiles} × {tiles} tile，{workers} worker" if zh else f"{tiles} × {tiles} tiles, {workers} worker" + ("s" if workers > 1 else "")
        if (tiles, workers) == (1, 1):
            label = "1 個 tile" if zh else "one tile"
        rows.append([label] + [comma(by_schedule[(n, tiles, workers)]["peak_rss_mb"]) for n in SIZES])
    write_table(folder, "memory-table", table(header, rows, "lrrr"), language)

    one, many = ones[20000]["stage_s"], by_schedule[(20000, 4, 16)]["stage_s"]
    kinds = ones[20000]["stage_kind"]
    header = ["stage", "種類" if zh else "Kind", "單一行程（s）" if zh else "One process (s)", "4 × 4 tile，16 個 worker（s）" if zh else "4 × 4 tiles, 16 workers (s)",
              "倍數" if zh else "Speed-up"]
    rows = []
    for kind in ("serial", "parallel"):
        top = sorted((name for name in one if kinds[name] == kind), key=lambda name: one[name], reverse=True)[:4]
        for name in top:
            rows.append([f"`{name}`", ("序列" if zh else "serial") if kind == "serial" else ("平行" if zh else "parallel"),
                         f"{one[name]:.1f}", f"{many[name]:.1f}", f"{one[name] / many[name]:.2f}×"])
    write_table(folder, "stage-table", table(header, rows, "llrrr"), language)


# --------------------------------------------------------------------------- quoted numbers


def quoted_numbers(by_schedule: dict, ones: dict[int, dict], timing: dict, oracle: dict, example: dict, validator: dict) -> dict:
    """The values the prose of the pages states, so a reviewer can compare the sentences with the data."""
    gain = speedups(by_schedule)
    row = ones[20000]
    tail = tail_steps(row)
    base = end_to_end(row, row["build_wall_s"])
    best = by_schedule[(20000, 8, 8)]["build_wall_s"]
    limit = row["amdahl"]["amdahl_speedup_limit"]
    checks = validator_checks(timing)
    faces = [r["counts"]["faces"] for r in sorted(timing["validator"], key=lambda r: r["n"])]
    structural = [sum(v[i] for k, v in checks.items() if CHECKS[k][0] == "structural") for i in range(3)]
    check_slopes = {k: round(slope(faces[0], v[0], faces[-1], v[-1]), 2) for k, v in checks.items()}
    workers_rss = {n: [by_schedule[(n, t, w)]["peak_worker_rss_mb"] for t, w in SCHEDULES if w > 1] for n in SIZES}
    return {
        "derive_s": {n: ones[n]["build_wall_s"] for n in SIZES},
        "faces": {n: ones[n]["counts"]["faces"] for n in SIZES},
        "derive_slopes_against_n": [round(s, 2) for s in slopes(list(SIZES), [ones[n]["build_wall_s"] for n in SIZES])],
        "rss_slopes_against_n": [round(s, 2) for s in slopes(list(SIZES), [ones[n]["peak_rss_mb"] for n in SIZES])],
        "faces_slopes_against_n": [round(s, 2) for s in slopes(list(SIZES), [ones[n]["counts"]["faces"] for n in SIZES])],
        "us_per_face": {n: round(ones[n]["build_wall_s"] / ones[n]["counts"]["faces"] * 1e6) for n in SIZES},
        "tail_s_at_20000": {k: round(v, 1) for k, v in tail.items()},
        "tail_total_one_process": round(tail["write_w1"] + tail["reports"] + tail["sidecar"] + tail["load"] + tail["verify"], 1),
        "tail_slopes_against_n": {k: [round(s, 2) for s in slopes(list(SIZES), [tail_steps(ones[n])[k] for n in SIZES])]
                                  for k in ("write_w1", "reports", "sidecar", "load", "verify")},
        "end_to_end_one_process_s": round(base, 1),
        "end_to_end_with_best_derivation_s": round(end_to_end(row, best), 1),
        "end_to_end_gain_best": round(base / end_to_end(row, best), 3),
        "end_to_end_gain_at_amdahl_limit": round(base / end_to_end(row, row["build_wall_s"] / limit), 3),
        "sidecar_mb": {n: round(ones[n]["emission"]["file_bytes"]["sidecar"] / 1e6, 1) for n in SIZES},
        "api_to_direct_ratio": {h["n"]: round(h["api"]["total_s"] / (h["direct"]["plan_s"] + h["direct"]["render_brep_s"] + h["direct"]["render_xao_s"]), 1)
                                for h in timing["head"]},
        "api_group_share": {h["n"]: round(h["api"]["physical_groups_s"] / h["api"]["total_s"], 3) for h in timing["head"]},
        "api_slopes_against_n": [round(s, 2) for s in slopes([h["n"] for h in timing["head"]], [h["api"]["total_s"] for h in timing["head"]])],
        "oracle_runs_from_scaling_pass_s": {r["n_features"]: r["oracle_emit_s"] for r in oracle["rows"]},
        "structural_total_s": [round(v, 2) for v in structural],
        "validator_cross_check_s9_total": {r["n_features"]: [r["validation_total_s"], round(sum(checks[k][i] for k in S9_CHECKS), 3)]
                                          for i, r in enumerate(validator["rows"])},
        "structural_total_slopes_against_faces": [round(s, 2) for s in slopes(faces, structural)],
        "check_slopes_against_faces": check_slopes,
        "check_slope_range_over_0.1s_at_largest": [min(s for k, s in check_slopes.items() if checks[k][-1] > 0.1), max(s for k, s in check_slopes.items() if checks[k][-1] > 0.1)],
        "speedup_best": {n: round(max(v for (t, w), v in gain[n].items() if w > 1), 2) for n in SIZES},
        "speedup_one_worker_tiled": {n: [round(min(gain[n][(t, 1)] for t in (4, 8)), 2), round(max(gain[n][(t, 1)] for t in (4, 8)), 2)] for n in SIZES},
        "serial_share_of_stages": {n: round(ones[n]["amdahl"]["serial_stages_s"] / (ones[n]["amdahl"]["serial_stages_s"] + ones[n]["amdahl"]["parallel_stages_wall_s"]), 3) for n in SIZES},
        "amdahl_limit": {n: ones[n]["amdahl"]["amdahl_speedup_limit"] for n in SIZES},
        "worker_peak_rss_range_mb": {n: [min(v), max(v)] for n, v in workers_rss.items()},
        "parent_peak_rss_20000_mb": {f"{t}x{t}/{w}": by_schedule[(20000, t, w)]["peak_rss_mb"] for t, w in SCHEDULES},
        "distinct_identity_digests": {n: len({by_schedule[(n, t, w)]["identity_digest"] for t, w in SCHEDULES}) for n in SIZES},
        "all_validation_passed": all(r["validation_passed"] for r in by_schedule.values()),
        "occt_all_ok_on_examples": example["occt"]["all_ok"],
    }


# --------------------------------------------------------------------------- main

def main() -> None:
    measured = read("why-scaling-measured.json")
    timing = read("emission-timing-data.json")
    oracle = read("why-scaling-oracle.json")
    example = read("emission-example-numbers.json")
    ones = one_process_rows(measured)
    by_schedule = schedule_rows(measured)
    numbers = quoted_numbers(by_schedule, ones, timing, oracle, example, read("why-validator-scaling.json"))
    for language, folder in OUTPUT.items():
        t = TEXT[language]
        scaling_figure(t, ones, by_schedule, folder)
        scaling_tables(language, ones, folder)
        api_comparison_figure(t, timing, folder)
        tail_figure(t, ones, folder)
        emission_tables(language, ones, timing, oracle, folder)
        validation_figure(t, timing, folder)
        validation_tables(language, timing, folder)
        occt_table(language, example, folder)
        parallel_figure(t, by_schedule, folder)
        parallel_tables(language, by_schedule, ones, folder)
        (folder / "numbers.json").write_text(json.dumps(numbers, indent=1, sort_keys=True) + "\n")
        print(f"wrote {folder.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
