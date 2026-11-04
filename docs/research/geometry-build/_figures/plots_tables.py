"""Emission, validation and OCCT figures and tables of the geometry-build study."""

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
    slope,
    structural_seconds,
    table,
    tail_steps,
    validator_checks,
    write_table,
)
from plots_text import CHECKS


def emission_tables(
    language: str, ones: dict[int, dict], timing: dict, oracle: dict, folder: Path
) -> None:
    """The emission tables: write and load times, file sizes and the oracle comparison."""
    zh = language == "zh-TW"
    header = [
        "N",
        "寫出 BREP 與 XAO，單一行程（s）"
        if zh
        else "Write BREP and XAO, one process (s)",
        "寫出，16 個 worker（s）" if zh else "Write, 16 workers (s)",
        "報告（s）" if zh else "Reports (s)",
        "Sidecar（s）",
        "Gmsh 載入 XAO（s）" if zh else "Gmsh load of the XAO (s)",
        "verify_loaded（s）",
    ]
    rows = []
    for n in SIZES:
        tail = tail_steps(ones[n])
        rows.append(
            [comma(n)]
            + [
                seconds_text(tail[k])
                for k in (
                    "write_w1",
                    "write_w16",
                    "reports",
                    "sidecar",
                    "load",
                    "verify",
                )
            ]
        )
    write_table(folder, "tail-table", table(header, rows, "rrrrrrr"), language)

    runs = {r["n"]: r for r in timing["validator"]}
    header = [
        "N",
        "寫出的 face 數" if zh else "Faces written",
        "BREP（MB）",
        "XAO（MB）",
        "Sidecar（MB）",
        "整條路徑的 peak RSS（MB）" if zh else "Peak RSS of the whole path (MB)",
    ]
    rows = []
    for n in SIZES:
        e = ones[n]["emission"]
        mb = {k: v / 1e6 for k, v in e["file_bytes"].items()}
        rows.append(
            [
                comma(n),
                comma(e["written_counts"]["faces"]),
                comma(mb["brep"], 1),
                comma(mb["xao"], 1),
                comma(mb["sidecar"], 1),
                comma(runs[n]["peak_rss_mb"]),
            ]
        )
    write_table(folder, "files-table", table(header, rows, "rrrrrr"), language)

    header = [
        "N",
        "寫出的 face 數" if zh else "Faces",
        "physical group 數" if zh else "Physical groups",
        "OCC build（s）",
        "synchronize（s）",
        "addPhysicalGroup（s）",
        "寫出 XAO（s）" if zh else "Write XAO (s)",
        "Gmsh API 合計（s）" if zh else "Gmsh API total (s)",
        "直接寫出（s）" if zh else "Direct writer (s)",
        "倍數" if zh else "Ratio",
        "group 佔比" if zh else "Group share",
    ]
    rows = []
    for h in timing["head"]:
        api, direct = h["api"], h["direct"]
        direct_total = (
            direct["plan_s"] + direct["render_brep_s"] + direct["render_xao_s"]
        )
        rows.append(
            [
                comma(h["n"]),
                comma(direct["plan_counts"]["faces"]),
                comma(h["groups"]),
                f"{api['occ_build_s']:.2f}",
                f"{api['synchronize_s']:.2f}",
                seconds_text(api["physical_groups_s"]),
                f"{api['write_xao_s']:.2f}",
                seconds_text(api["total_s"]),
                f"{direct_total:.2f}",
                f"{api['total_s'] / direct_total:.0f}×",
                f"{api['physical_groups_s'] / api['total_s']:.0%}",
            ]
        )
    write_table(folder, "api-table", table(header, rows, "rrrrrrrrrrr"), language)

    header = [
        "N",
        "Gmsh API：substrate（µm³）",
        "Gmsh API：vacuum（µm³）",
        "直接寫出：兩個 solid（µm³）" if zh else "Direct writer: the two solids (µm³)",
    ]
    if not zh:
        header[1:3] = ["Gmsh API: substrate (µm³)", "Gmsh API: vacuum (µm³)"]
    rows = []
    for r in oracle["rows"]:
        mass = r["solid_mass_um3"]
        direct_masses = ", ".join(f"{m:.3e}" for m in mass["direct_writer"])
        rows.append(
            [
                comma(r["n_features"]),
                f"{mass['gmsh_api_oracle']['substrate']:.3e}",
                f"{mass['gmsh_api_oracle']['vacuum']:.3e}",
                direct_masses,
            ]
        )
    write_table(folder, "mass-table", table(header, rows, "rrrr"), language)

    header = [
        "N",
        "API 寫出的 XAO（s）" if zh else "XAO written by the Gmsh API (s)",
        "直接寫出的 XAO（s）" if zh else "XAO from the direct writer (s)",
        "直接寫出的 BREP（s）" if zh else "BREP from the direct writer (s)",
    ]
    rows = [
        [
            comma(h["n"]),
            f"{h['load']['api_xao_s']:.2f}",
            f"{h['load']['direct_xao_s']:.2f}",
            f"{h['load']['direct_brep_s']:.2f}",
        ]
        for h in timing["head"]
    ]
    write_table(folder, "load-table", table(header, rows, "rrrr"), language)


# --------------------------------------------------------------------------- validation page


def validation_figure(t: dict, timing: dict, folder: Path) -> None:
    """The slowest validation checks and the structural against load-side totals, by N."""
    checks = validator_checks(timing)
    ordered = sorted(checks, key=lambda name: checks[name][-1], reverse=True)[:12]
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.2, 1]}
    )
    colours = [
        COLOUR["derive"] if CHECKS[name][0] == "structural" else COLOUR["verify"]
        for name in ordered
    ]
    left.barh(ordered, [checks[name][-1] for name in ordered], color=colours)
    left.invert_yaxis()
    left.set_xlabel(t["check_seconds"])
    left.tick_params(axis="y", labelsize=8)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOUR["derive"]),
        plt.Rectangle((0, 0), 1, 1, color=COLOUR["verify"]),
    ]
    left.legend(
        handles,
        [t["structural"], t["loaded"]],
        fontsize=8,
        frameon=False,
        loc="lower right",
    )
    structural = [
        sum(v[i] for k, v in checks.items() if CHECKS[k][0] == "structural")
        for i in range(3)
    ]
    loaded = [
        sum(v[i] for k, v in checks.items() if CHECKS[k][0] == "loaded")
        for i in range(3)
    ]
    right.plot(
        SIZES,
        structural,
        marker="o",
        color=COLOUR["derive"],
        label=t["structural_total"],
    )
    right.plot(
        SIZES, loaded, marker="o", color=COLOUR["verify"], label=t["loaded_total"]
    )
    right.plot(
        SIZES,
        [structural[0] * n / 1000 for n in SIZES],
        ls=":",
        color=COLOUR["reference"],
        label=t["slope_one"],
    )
    log_axes(right)
    right.set_xlabel(t["features"])
    right.set_ylabel(t["seconds"])
    right.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    save_svg(fig, folder, "validation")


def validation_tables(language: str, timing: dict, folder: Path) -> None:
    """The tables of the check times and their growth against the number of faces."""
    zh = language == "zh-TW"
    checks = validator_checks(timing)
    faces = [
        r["counts"]["faces"] for r in sorted(timing["validator"], key=lambda r: r["n"])
    ]
    header = [
        "檢查" if zh else "Check",
        "種類" if zh else "Kind",
        "檢查什麼" if zh else "What it checks",
        "N = 1,000（s）",
        "N = 5,000（s）",
        "N = 20,000（s）",
        "斜率" if zh else "Slope",
    ]
    rows = []
    for name in sorted(checks, key=lambda k: checks[k][-1], reverse=True):
        kind, english, chinese = CHECKS[name]
        kind_text = {
            "structural": "結構" if zh else "structural",
            "loaded": "載入後" if zh else "loaded model",
        }[kind]
        values = checks[name]
        overall = slope(faces[0], values[0], faces[-1], values[-1])
        rows.append(
            [f"`{name}`", kind_text, chinese if zh else english]
            + [f"{v:.3f}" for v in values]
            + [f"{overall:.2f}"]
        )
    write_table(folder, "validator-table", table(header, rows, "lllrrrr"), language)

    header = [
        "N",
        "face 數" if zh else "Faces",
        "結構檢查合計（s）" if zh else "Structural checks, total (s)",
        "載入後檢查合計（s）" if zh else "Checks on the loaded model, total (s)",
        "結構檢查佔該次 build 的比例" if zh else "Structural share of that run's build",
    ]
    rows = []
    for run in sorted(timing["validator"], key=lambda r: r["n"]):
        rows.append(
            [
                comma(run["n"]),
                comma(run["counts"]["faces"]),
                f"{sum(structural_seconds(run).values()):.2f}",
                f"{run['verify_s']:.2f}",
                f"{sum(structural_seconds(run).values()) / run['build_s']:.0%}",
            ]
        )
    write_table(folder, "validator-total-table", table(header, rows, "rrrrr"), language)


def occt_table(language: str, example: dict, folder: Path) -> None:
    """The table of the OCCT checks of the example models and the extreme-case catalogue."""
    zh = language == "zh-TW"
    catalogue, occt = example["catalogue"], example["occt"]
    header = [
        "模型組" if zh else "Model set",
        "報告數" if zh else "Reports",
        "BRepCheck_Analyzer（ms）",
        "BOPAlgo_ArgumentAnalyzer（ms）",
        "OCCT 檢查合計（ms）" if zh else "All OCCT checks (ms)",
    ]
    ms = catalogue["occt_ms"]
    rows = [
        [
            "63 個組合中通過的 54 個（23 個 fixture）"
            if zh
            else "54 passing combinations of 63 (23 fixtures)",
            str(catalogue["passed"]),
            f"{ms['brep_check_ms'][0]:.1f}–{ms['brep_check_ms'][1]:.1f}",
            f"{ms['bop_ms'][0]:.1f}–{ms['bop_ms'][1]:.1f}",
            f"{ms['occt_total_ms'][0]:.1f}–{ms['occt_total_ms'][1]:.1f}",
        ],
        [
            "7 個範例模型，BREP 與 XAO" if zh else "Seven example models, BREP and XAO",
            str(occt["reports"]),
            f"{occt['brep_check_ms'][0]:.1f}–{occt['brep_check_ms'][1]:.1f}",
            f"{occt['bop_ms'][0]:.1f}–{occt['bop_ms'][1]:.1f}",
            f"{occt['total_ms'][0]:.1f}–{occt['total_ms'][1]:.1f}",
        ],
    ]
    write_table(folder, "occt-table", table(header, rows, "lrrrr"), language)
