"""Figures of why-plan-first.qmd: measured scaling of the planned pipeline.

Data: ``experiments/sgb-redesign/prototype/outputs/scaling.json`` (derivation, Gmsh-API emission, peak RSS,
identity digests; Route B random layouts, N = 1000, 5000, 20000) and ``outputs/e3_brep/timing.json`` (direct
BREP/XAO write and Gmsh load times for the same layouts). Nothing here measures a global 3-D fragment.

Writes ``figures/why-scaling.html`` and ``.svg`` (log-log time and memory against N), ``figures/why-tail.html``
and ``.svg`` (end-to-end stacks at N = 20000) and ``figures/why-scaling-numbers.json``.
"""

from __future__ import annotations

import json
import math

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from _intro_common import OUTPUTS, save_plotly, save_svg, write_numbers

COLOUR = {"derive": "#2a6f9e", "api": "#c0392b", "write": "#e08a1e", "load": "#7f7f7f", "rss": "#2a8f5a", "ref": "#999999"}


def load():
    scaling = json.loads((OUTPUTS / "scaling.json").read_text())["rows"]
    timing = {r["n_features"]: r for r in json.loads((OUTPUTS / "e3_brep" / "timing.json").read_text())}
    return scaling, timing


def series(scaling, timing) -> dict:
    ns = sorted({r["n_features"] for r in scaling})
    base = {r["n_features"]: r for r in scaling if r["tiles_per_side"] == 1 and r["workers"] == 1}
    tiled = {r["n_features"]: r for r in scaling if r["tiles_per_side"] == 4 and r["workers"] == 1}
    return {
        "n": ns,
        "faces": [base[n]["counts"]["faces"] for n in ns],
        "derive_s": [base[n]["build_wall_s"] for n in ns],
        "api_emit_s": [base[n]["emit_s"] for n in ns],
        "brep_write_s": [timing[n]["direct"]["brep_w1"]["total"] for n in ns],
        "brep_write_w16_s": [timing[n]["direct"]["brep_w16"]["total"] for n in ns],
        "gmsh_load_s": [timing[n]["load"]["direct_brep"]["load_s"] for n in ns],
        "rss_mb": [base[n]["peak_rss_mb"] for n in ns],
        "rss_tiled_mb": [tiled[n]["peak_rss_mb"] for n in ns],
        "native_entities": [base[n]["native_entities"] for n in ns],
    }


def speedups(scaling) -> dict:
    """Best tiled/parallel derivation speed-up over the serial one-tile run, per N (all schedules in scaling.json)."""
    out = {}
    for n in sorted({r["n_features"] for r in scaling}):
        rows = [r for r in scaling if r["n_features"] == n]
        base = next(r for r in rows if r["tiles_per_side"] == 1)["build_wall_s"]
        best = max(rows, key=lambda r: base / r["build_wall_s"])
        out[str(n)] = {"best_speedup": round(base / best["build_wall_s"], 2), "tiles_per_side": best["tiles_per_side"], "workers": best["workers"]}
    return out


def sidecar() -> dict:
    """Time to build and serialise the sidecar JSON with lineage (outputs/e3_brep/sidecar_timing.json)."""
    raw = json.loads((OUTPUTS / "e3_brep" / "sidecar_timing.json").read_text())
    return {n: {"with_lineage_s": round(v["lineage"]["build_s"] + v["lineage"]["json_s"], 2), "with_lineage_MB": v["lineage"]["MB"],
                "without_lineage_s": round(v["no_lineage"]["build_s"] + v["no_lineage"]["json_s"], 2)} for n, v in raw.items()}


def exponent(ns, ys) -> list[float]:
    """Local log-log slope between consecutive N."""
    return [round(math.log(y1 / y0) / math.log(n1 / n0), 2) for (n0, y0), (n1, y1) in zip(zip(ns, ys), zip(ns[1:], ys[1:]))]


def identity(scaling) -> dict:
    out = {}
    for n in sorted({r["n_features"] for r in scaling}):
        rows = [r for r in scaling if r["n_features"] == n]
        out[str(n)] = {
            "configurations": len(rows),
            "tiles_per_side": sorted({r["tiles_per_side"] for r in rows}),
            "workers": sorted({r["workers"] for r in rows}),
            "distinct_ids_digests": len({r["ids_digest"] for r in rows}),
            "ids_digest": rows[0]["ids_digest"],
            "all_validation_passed": all(r["validation_passed"] for r in rows),
        }
    return out


LINES = (
    ("derive_s", "derivation S2-S9 (serial, T = 1)", "derive", "solid"),
    ("api_emit_s", "Gmsh-API emitter (test oracle; no sewing)", "api", "solid"),
    ("brep_write_s", "direct BREP/XAO writer (D8, one process)", "write", "solid"),
    ("gmsh_load_s", "Gmsh loading the BREP (outside SGB)", "load", "dash"),
)


def scaling_figure(s: dict) -> None:
    ns = s["n"]
    ref = [s["derive_s"][0] * n / ns[0] for n in ns]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Wall time against model size", "Peak memory against model size"),
                        horizontal_spacing=0.12)
    for key, name, colour, dash in LINES:
        fig.add_trace(go.Scatter(x=ns, y=s[key], mode="lines+markers", name=name, line=dict(color=COLOUR[colour], dash=dash),
                                 hovertemplate="N = %{x}<br>%{y:.2f} s<extra>" + name + "</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ns, y=ref, mode="lines", name="slope 1 from the derivation at N = 1000", line=dict(color=COLOUR["ref"], dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ns, y=s["rss_mb"], mode="lines+markers", name="peak RSS, one tile (T = 1)", line=dict(color=COLOUR["rss"]),
                             hovertemplate="N = %{x}<br>%{y:.0f} MB<extra>T = 1</extra>"), row=1, col=2)
    fig.add_trace(go.Scatter(x=ns, y=s["rss_tiled_mb"], mode="lines+markers", name="peak RSS, 4 × 4 tiles, one worker",
                             line=dict(color=COLOUR["rss"], dash="dash"), hovertemplate="N = %{x}<br>%{y:.0f} MB<extra>4 x 4 tiles</extra>"), row=1, col=2)
    fig.add_trace(go.Scatter(x=ns, y=[s["rss_mb"][0] * n / ns[0] for n in ns], mode="lines", name="slope 1 from T = 1 at N = 1000",
                             line=dict(color=COLOUR["ref"], dash="dot"), showlegend=False), row=1, col=2)
    for col in (1, 2):
        fig.update_xaxes(type="log", title="features N (log)", tickvals=ns, row=1, col=col)
        fig.update_yaxes(type="log", row=1, col=col)
    fig.update_yaxes(title="seconds (log)", row=1, col=1)
    fig.update_yaxes(title="megabytes (log)", row=1, col=2)
    fig.update_layout(height=520, legend=dict(orientation="h", y=-0.28), title="Planned pipeline, Route B random layouts (computed by examples/scaling.py)")
    save_plotly(fig, "why-scaling")

    f, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for key, name, colour, dash in LINES:
        a1.loglog(ns, s[key], marker="o", color=COLOUR[colour], ls="-" if dash == "solid" else "--", label=name)
    a1.loglog(ns, ref, color=COLOUR["ref"], ls=":", label="slope 1 from the derivation at N = 1000")
    a2.loglog(ns, s["rss_mb"], marker="o", color=COLOUR["rss"], label="one tile (T = 1)")
    a2.loglog(ns, s["rss_tiled_mb"], marker="s", color=COLOUR["rss"], ls="--", label="4 × 4 tiles, one worker")
    a2.loglog(ns, [s["rss_mb"][0] * n / ns[0] for n in ns], color=COLOUR["ref"], ls=":", label="slope 1 from T = 1 at N = 1000")
    for ax, ylabel, title in ((a1, "seconds", "Wall time against model size"), (a2, "megabytes", "Peak memory (parent process)")):
        ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns]); ax.minorticks_off()
        ax.set_xlabel("features N"); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(True, which="major", color="#ddd", lw=0.6)
    a1.legend(fontsize=7, frameon=False, loc="upper left")
    a2.legend(fontsize=7, frameon=False, loc="upper left")
    f.tight_layout()
    save_svg(f, "why-scaling")


def tail_figure(timing) -> dict:
    t = timing[20000]
    side = sidecar()["20000"]
    side20 = side["with_lineage_s"]
    stacks = {
        "Gmsh-API oracle": (t["build_s"], t["gmsh_api"]["total_s"], t["load"]["gmsh_api_xao"]["load_s"], 0.0),
        "direct BREP/XAO, one process": (t["build_s"], t["direct"]["xao_w1"]["total"], t["load"]["direct_xao"]["load_s"], side20),
        "direct BREP/XAO, 16 workers": (t["build_s"], t["direct"]["xao_w16"]["total"], t["load"]["direct_xao"]["load_s"], side20),
    }
    labels = ["derivation S2-S9", "write the CAD file", "Gmsh loads it", "sidecar with lineage (not measured for the oracle)"]
    colours = [COLOUR["derive"], COLOUR["write"], COLOUR["load"], "#2a8f5a"]
    fig = go.Figure()
    for i, label in enumerate(labels):
        fig.add_trace(go.Bar(y=list(stacks), x=[v[i] for v in stacks.values()], orientation="h", name=label, marker_color=colours[i],
                             text=[f"{v[i]:.1f} s" for v in stacks.values()], textposition="inside"))
    fig.update_layout(barmode="stack", height=360, xaxis_title="seconds, N = 20000 (computed by examples/scaling.py and e3 timing)",
                      legend=dict(orientation="h", y=-0.3), title="From planned topology to a model Gmsh can mesh, N = 20000")
    fig.update_yaxes(autorange="reversed")
    save_plotly(fig, "why-tail")
    f, ax = plt.subplots(figsize=(9, 3.6))
    left = [0.0] * len(stacks)
    for i, label in enumerate(labels):
        vals = [v[i] for v in stacks.values()]
        ax.barh(list(stacks), vals, left=left, color=colours[i], label=label, edgecolor="white")
        for y, (l, v) in enumerate(zip(left, vals)):
            if v >= 3.0:
                ax.text(l + v / 2, y, f"{v:.1f} s", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
        left = [a + b for a, b in zip(left, vals)]
    for y, total in enumerate(left):
        ax.text(total + 1, y, f"{total:.0f} s", va="center", fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel("seconds at N = 20000"); ax.set_xlim(0, max(left) * 1.12)
    ax.legend(fontsize=8, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.28))
    ax.set_title("From planned topology to a model Gmsh can mesh")
    f.tight_layout()
    save_svg(f, "why-tail")
    return {k: {"derive_s": round(v[0], 1), "write_s": round(v[1], 1), "load_s": round(v[2], 1), "sidecar_s": round(v[3], 1), "total_s": round(sum(v), 1)} for k, v in stacks.items()}


def main() -> None:
    scaling, timing = load()
    s = series(scaling, timing)
    scaling_figure(s)
    tails = tail_figure(timing)
    numbers = {
        "series": s,
        "local_log_slope": {k: exponent(s["n"], s[k]) for k in ("derive_s", "api_emit_s", "brep_write_s", "gmsh_load_s", "rss_mb", "rss_tiled_mb", "faces")},
        "identity": identity(scaling),
        "speedups": speedups(scaling),
        "sidecar": sidecar(),
        "stack_n20000": tails,
        "source": ["outputs/scaling.json", "outputs/e3_brep/timing.json"],
    }
    write_numbers("why-scaling", numbers)
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
