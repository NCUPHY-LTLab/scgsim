"""Plotting half of ``why_scaling.py``: the scaling and tail figures and ``why-scaling-numbers.json``, drawn from the measured files."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from intro_common import save_plotly, save_svg, write_numbers
from plotly.subplots import make_subplots
from why_scaling_measure import exponent

# --------------------------------------------------------------------------- figures (from the measured file)


def load_measured(path: Path) -> list[dict]:
    """The rows of the measured file; stop if any case failed or timed out."""
    rows = json.loads(path.read_text())["rows"]
    bad = [r for r in rows if "error" in r or "timeout_s" in r]
    if bad:
        raise SystemExit(f"measured cases failed: {bad}")
    return rows


def _series(rows: list[dict], oracle: dict) -> dict:
    ns = sorted({r["n_features"] for r in rows})
    base = {r["n_features"]: r for r in rows}
    direct = {n: base[n]["emission"]["direct"] for n in ns}
    return {
        "n": ns,
        "faces": [base[n]["counts"]["faces"] for n in ns],
        "derive_s": [base[n]["build_wall_s"] for n in ns],
        "brep_write_s": [
            round(
                direct[n]["plan"] + direct[n]["write_brep"] + direct[n]["write_xao"], 3
            )
            for n in ns
        ],
        "reports_s": [direct[n]["reports"] for n in ns],
        "sidecar_s": [direct[n]["sidecar"] for n in ns],
        "gmsh_load_s": [base[n]["emission"]["gmsh_load_s"] for n in ns],
        "verify_loaded_s": [base[n]["emission"]["verify_loaded_s"] for n in ns],
        "oracle": {
            "n": sorted(n for n in oracle if "oracle_emit_s" in oracle[n]),
            "emit_s": [
                oracle[n]["oracle_emit_s"]
                for n in sorted(oracle)
                if "oracle_emit_s" in oracle[n]
            ],
            "load_s": [
                oracle[n]["oracle_load_s"]
                for n in sorted(oracle)
                if "oracle_emit_s" in oracle[n]
            ],
            "timed_out_at_n": [n for n in sorted(oracle) if "timeout_s" in oracle[n]],
        },
        "rss_mb": [base[n]["peak_rss_mb"] for n in ns],
        "file_mb": [
            {k: round(v / 1e6, 1) for k, v in base[n]["emission"]["file_bytes"].items()}
            for n in ns
        ],
        "written_counts": [base[n]["emission"]["written_counts"] for n in ns],
    }


COLOUR = {
    "derive": "#2a6f9e",
    "api": "#c0392b",
    "write": "#e08a1e",
    "load": "#7f7f7f",
    "rss": "#2a8f5a",
    "ref": "#999999",
}
LINES = (
    ("derive_s", "derivation and validation S2-S9", "derive", "solid"),
    ("brep_write_s", "direct BREP/XAO writer (D8)", "write", "solid"),
    ("gmsh_load_s", "Gmsh loading the XAO (outside SGB)", "load", "dash"),
)


def measured_points(ns, values):
    """The (N, value) pairs that were measured (the oracle has no value at the largest N)."""
    pairs = [(n, v) for n, v in zip(ns, values, strict=False) if v is not None]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _scaling_figure(s: dict) -> None:
    _scaling_html(s)
    _scaling_svg(s)


def _scaling_html(s: dict) -> None:
    """The interactive time and memory figure."""
    ns = s["n"]
    ref = [s["derive_s"][0] * n / ns[0] for n in ns]
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Wall time against model size",
            "Peak memory against model size",
        ),
        horizontal_spacing=0.12,
    )
    for key, name, colour, dash in LINES:
        xs, ys = measured_points(ns, s[key])
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                name=name,
                line=dict(color=COLOUR[colour], dash=dash),
                hovertemplate="N = %{x}<br>%{y:.2f} s<extra>" + name + "</extra>",
            ),
            row=1,
            col=1,
        )
    o = s["oracle"]
    fig.add_trace(
        go.Scatter(
            x=o["n"],
            y=o["emit_s"],
            mode="lines+markers",
            name="Gmsh-API emitter (test oracle; run up to N = 2000 only)",
            line=dict(color=COLOUR["api"]),
            hovertemplate="N = %{x}<br>%{y:.2f} s<extra>Gmsh-API oracle</extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ns,
            y=ref,
            mode="lines",
            name="slope 1 from the derivation at N = 1000",
            line=dict(color=COLOUR["ref"], dash="dot"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ns,
            y=s["rss_mb"],
            mode="lines+markers",
            name="peak RSS after the build",
            line=dict(color=COLOUR["rss"]),
            hovertemplate="N = %{x}<br>%{y:.0f} MB<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=ns,
            y=[s["rss_mb"][0] * n / ns[0] for n in ns],
            mode="lines",
            name="slope 1 from N = 1000",
            line=dict(color=COLOUR["ref"], dash="dot"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    for col in (1, 2):
        fig.update_xaxes(
            type="log", title="features N (log)", tickvals=ns, row=1, col=col
        )
        fig.update_yaxes(type="log", row=1, col=col)
    fig.update_yaxes(title="seconds (log)", row=1, col=1)
    fig.update_yaxes(title="megabytes (log)", row=1, col=2)
    fig.update_layout(
        height=520,
        legend=dict(orientation="h", y=-0.28),
        title="Production pipeline scgsim.geometry.sgb, Route B random layouts (measured by why_scaling.py)",
    )
    save_plotly(fig, "why-scaling")


def _scaling_svg(s: dict) -> None:
    """The static copy of the time and memory figure."""
    ns = s["n"]
    ref = [s["derive_s"][0] * n / ns[0] for n in ns]
    f, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for key, name, colour, dash in LINES:
        xs, ys = measured_points(ns, s[key])
        a1.loglog(
            xs,
            ys,
            marker="o",
            color=COLOUR[colour],
            ls="-" if dash == "solid" else "--",
            label=name,
        )
    a1.loglog(
        s["oracle"]["n"],
        s["oracle"]["emit_s"],
        marker="o",
        color=COLOUR["api"],
        label="Gmsh-API emitter (test oracle; run up to N = 2000 only)",
    )
    a1.loglog(
        ns,
        ref,
        color=COLOUR["ref"],
        ls=":",
        label="slope 1 from the derivation at N = 1000",
    )
    a2.loglog(ns, s["rss_mb"], marker="o", color=COLOUR["rss"], label="peak RSS")
    a2.loglog(
        ns,
        [s["rss_mb"][0] * n / ns[0] for n in ns],
        color=COLOUR["ref"],
        ls=":",
        label="slope 1 from N = 1000",
    )
    for ax, ylabel, title in (
        (a1, "seconds", "Wall time against model size"),
        (a2, "megabytes", "Peak memory after the build (parent process)"),
    ):
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        ax.minorticks_off()
        ax.set_xlabel("features N")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="major", color="#ddd", lw=0.6)
    a1.legend(fontsize=7, frameon=False, loc="upper left")
    a2.legend(fontsize=7, frameon=False, loc="upper left")
    f.tight_layout()
    save_svg(f, "why-scaling")


def _tail_figure(rows: list[dict]) -> dict:
    base = next(r for r in rows if r["n_features"] == 20000)
    e = base["emission"]
    t = e["direct"]
    stacks = {
        "direct BREP/XAO": (
            base["build_wall_s"],
            t["plan"] + t["write_brep"] + t["write_xao"],
            t["reports"],
            t["sidecar"],
            e["gmsh_load_s"],
            e["verify_loaded_s"],
        ),
    }
    labels = [
        "derivation and validation S2-S9",
        "write the CAD file (plan, BREP, XAO)",
        "sliver and validation reports",
        "sidecar with lineage",
        "Gmsh loads the XAO",
        "load-side check verify_loaded",
    ]
    colours = [
        COLOUR["derive"],
        COLOUR["write"],
        "#8c6bb1",
        "#2a8f5a",
        COLOUR["load"],
        "#b8a76a",
    ]
    fig = go.Figure()
    for i, label in enumerate(labels):
        fig.add_trace(
            go.Bar(
                y=list(stacks),
                x=[v[i] for v in stacks.values()],
                orientation="h",
                name=label,
                marker_color=colours[i],
                text=[f"{v[i]:.1f} s" if v[i] >= 0.5 else "" for v in stacks.values()],
                textposition="inside",
            )
        )
    fig.update_layout(
        barmode="stack",
        height=360,
        xaxis_title="seconds, N = 20000 (measured by why_scaling.py)",
        legend=dict(orientation="h", y=-0.4),
        title="From the validated model to a model Gmsh can mesh, N = 20000",
    )
    fig.update_yaxes(autorange="reversed")
    save_plotly(fig, "why-tail")
    f, ax = plt.subplots(figsize=(9.5, 3.6))
    left = [0.0] * len(stacks)
    for i, label in enumerate(labels):
        vals = [v[i] for v in stacks.values()]
        ax.barh(
            list(stacks),
            vals,
            left=left,
            color=colours[i],
            label=label,
            edgecolor="white",
        )
        for y, (start, v) in enumerate(zip(left, vals, strict=False)):
            if v >= 4.0:
                ax.text(
                    start + v / 2,
                    y,
                    f"{v:.1f} s",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                    fontweight="bold",
                )
        left = [a + b for a, b in zip(left, vals, strict=False)]
    for y, total in enumerate(left):
        ax.text(total + 1, y, f"{total:.0f} s", va="center", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("seconds at N = 20000")
    ax.set_xlim(0, max(left) * 1.12)
    ax.legend(
        fontsize=7,
        frameon=False,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.25),
    )
    ax.set_title("From the validated model to a model Gmsh can mesh")
    f.tight_layout()
    save_svg(f, "why-tail")
    return {
        name: dict(
            zip(
                (
                    "derive_s",
                    "write_s",
                    "reports_s",
                    "sidecar_s",
                    "gmsh_load_s",
                    "verify_loaded_s",
                ),
                (round(x, 1) for x in v),
                strict=False,
            ),
            total_s=round(sum(v), 1),
        )
        for name, v in stacks.items()
    }


def _local_slopes(ns, values) -> list[float]:
    xs, ys = measured_points(ns, values)
    return exponent(xs, ys)


def plot(rows: list[dict], oracle: dict, machine: dict) -> None:
    """Draw the scaling and tail figures and write ``why-scaling-numbers.json``."""
    s = _series(rows, oracle)
    _scaling_figure(s)
    tails = _tail_figure(rows)
    numbers = {
        "series": s,
        "local_log_slope": {
            k: _local_slopes(s["n"], s[k])
            for k in (
                "derive_s",
                "brep_write_s",
                "gmsh_load_s",
                "verify_loaded_s",
                "reports_s",
                "sidecar_s",
                "rss_mb",
                "faces",
            )
        },
        "stack_n20000": tails,
        "oracle_local_log_slope": exponent(s["oracle"]["n"], s["oracle"]["emit_s"])
        if len(s["oracle"]["n"]) > 1
        else [],
        "oracle_ratio_to_direct_write_at_1000": round(
            s["oracle"]["emit_s"][s["oracle"]["n"].index(1000)]
            / _series_value(s, "brep_write_s", 1000),
            1,
        ),
        "machine": machine,
        "source": "figures/why-scaling-measured.json (python why_scaling.py measure), figures/why-scaling-oracle.json (python why_scaling.py oracle)",
    }
    write_numbers("why-scaling", numbers)
    print(json.dumps(numbers, indent=1))


def _series_value(s: dict, key: str, n: int) -> float:
    return s[key][s["n"].index(n)]
