"""Figures of why-plan-first.qmd: measured scaling of the production pipeline.

Measures ``scgsim.geometry.sgb`` itself.  Each case runs in its own interpreter (so peak memory is that case's
own) on the Route B random layouts of ``tests/sgb_scenes.random_layout`` (seed 2, pitch 260 um): N = 1000, 5000 and
20000 features, nine schedules (one tile, and 4 x 4 and 8 x 8 tiles with 1, 8, 16 and 32 workers).  The one-tile
case also writes the CAD files with the direct writer (1 and 16 workers), opens the XAO in Gmsh, runs the load-side
check ``verify_loaded``.  A second mode replays the same models through the Gmsh API oracle
(``tests/sgb_gmsh_oracle``, the test-side bottom-up emitter) for the comparison of decision D8; its time grows
much faster than linearly (N = 5000 did not finish in 10 minutes in a trial run), so it is run only up to N = 2000.  Nothing here measures a global 3-D fragment.

Four modes::

    python why_scaling.py measure   # runs the cases (about 15 minutes) and writes figures/why-scaling-measured.json
    python why_scaling.py oracle    # the Gmsh-API oracle at N = 500, 1000, 2000 -> figures/why-scaling-oracle.json
    python why_scaling.py validators  # time of each structural validation check at N = 1000, 5000, 20000 -> figures/why-validator-scaling.json
    python why_scaling.py           # draws the figures and writes figures/why-scaling-numbers.json from those files

Writes ``figures/why-scaling.html`` and ``.svg`` (log-log time and memory against N), ``figures/why-tail.html`` and
``.svg`` (end-to-end stacks at N = 20000), ``figures/why-scaling-numbers.json`` and the raw
``figures/why-scaling-measured.json``.  ``arrangement_guarantees.py`` reads the raw file for the arrangement sizes.
"""

from __future__ import annotations

import json
import math
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# the test-side scene builders of the sgb-core checkout (a measuring child imports nothing heavier than it needs,
# because it reports its own peak memory)
sys.path.insert(
    0, str(Path(os.environ.get("SGB_CORE", str(HERE.parents[4]))) / "tests")
)
sys.path.insert(0, str(HERE))

NS = (1000, 5000, 20000)
# Seed 1 is not used: at N = 20000 it draws two shapes of one Net that touch at a single vertex, which production
# rejects (``NonManifoldError``, fail closed); seeds 2 to 9 all build at N = 20000.
SEED = 2
SCHEDULES = [(1, 1)] + [
    (t, w) for t in (4, 8) for w in (1, 8, 16, 32)
]  # (tiles per side, workers)
CAP_S = 900.0


# --------------------------------------------------------------------------- measuring (one case per interpreter)


def run_child(n: int, tiles: int, workers: int) -> dict:
    """Build one layout under one schedule; for the one-tile schedule also emit, load, check and replay it."""
    from scgsim.geometry.sgb.model import build_model_from_catalogue
    from sgb_scenes import random_layout

    started = time.perf_counter()
    catalogue = random_layout(n, seed=SEED, route="B")
    generate = time.perf_counter() - started
    started = time.perf_counter()
    model = build_model_from_catalogue(catalogue, "B", tiles=tiles, workers=workers)
    build_wall = time.perf_counter() - started
    row = {
        "n_features": n,
        "tiles_per_side": tiles,
        "workers": workers,
        "generate_s": round(generate, 3),
        "build_wall_s": round(build_wall, 3),
        "stage_s": {k: round(v, 3) for k, v in model.timings.items()},
        "stage_kind": model.stage_kind,
        "amdahl": {k: round(v, 3) for k, v in model.amdahl.items()},
        "counts": model.counts,
        "identity_digest": model.identity_digest(),
        "validation_passed": True,
        "peak_rss_mb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
        ),
    }
    if (tiles, workers) == (1, 1):
        row["emission"] = emit_and_load(model)
    row["peak_worker_rss_mb"] = round(
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, 1
    )
    return row


def emit_and_load(model) -> dict:
    """Direct writer (1 and 16 workers), Gmsh load and the load-side check of the written files."""
    from scgsim.geometry.sgb.build import emit_model
    from scgsim.geometry.sgb.emission.plan import plan_brep
    from scgsim.geometry.sgb.emission.writer import write_brep_file, write_xao_file
    from scgsim.geometry.sgb.sidecar.files import read_sidecar
    from scgsim.geometry.sgb.verification.loaded import verify_loaded
    from sgb_gmsh_oracle import gmsh_session

    out: dict = {}
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        timings: dict[str, float] = {}
        artifacts = emit_model(model, folder, workers=1, timings=timings)
        out["direct_w1"] = {k: round(v, 3) for k, v in timings.items()}
        # the same files again with 16 writer processes; only the three steps the workers touch are repeated
        started = time.perf_counter()
        plan = plan_brep(model.topology, model.routed)
        wide = {"plan": time.perf_counter() - started}
        wide["write_brep"] = write_brep_file(plan, folder / "wide.brep", workers=16)
        wide["write_xao"] = write_xao_file(
            plan,
            folder / "wide.xao",
            name="wide",
            workers=16,
            brep=folder / "wide.brep",
        )
        out["direct_w16"] = {k: round(v, 3) for k, v in wide.items()}
        out["file_bytes"] = {
            key: path.stat().st_size
            for key, path in (
                ("xao", artifacts.xao),
                ("brep", artifacts.brep),
                ("sidecar", artifacts.sidecar),
                ("slivers", artifacts.sliver_report),
                ("validation", artifacts.validation_report),
            )
        }
        out["written_counts"] = {
            k: v
            for k, v in artifacts.counts.items()
            if k
            in ("vertices", "edges", "faces", "solids", "surfaces", "cylinders", "arcs")
        }
        sidecar = read_sidecar(artifacts.sidecar)
        started = time.perf_counter()
        with gmsh_session(artifacts.xao):
            out["gmsh_load_s"] = round(time.perf_counter() - started, 3)
            started = time.perf_counter()
            report = verify_loaded(sidecar)
            out["verify_loaded_s"] = round(time.perf_counter() - started, 3)
            out["verify_loaded_ok"] = bool(report.ok)
    return out


ORACLE_NS = (500, 1000, 2000)
ORACLE_CAP_S = 300.0


def run_validator_child(n: int) -> dict:
    """Build one layout and report the wall time of every structural validation check (``ValidationReport.timings``)."""
    from scgsim.geometry.sgb.model import build_model_from_catalogue
    from sgb_scenes import random_layout

    model = build_model_from_catalogue(random_layout(n, seed=SEED, route="B"), "B")
    return {
        "n_features": n,
        "faces": model.counts["faces"],
        "checks": len(model.report.checks),
        "check_timings_s": {k: round(v, 4) for k, v in model.report.timings.items()},
        "validation_total_s": round(sum(model.report.timings.values()), 3),
        "validation_stage_s": round(
            model.timings["validation_serial"] + model.timings["validation_shells"], 3
        ),
    }


def measure_validators(path: Path) -> None:
    rows = []
    for n in NS:
        proc = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--validator-child",
                str(n),
            ],
            capture_output=True,
            text=True,
        )
        rows.append(json.loads(proc.stdout.strip().splitlines()[-1]))
        print(n, rows[-1]["validation_total_s"], flush=True)
    names = sorted(rows[0]["check_timings_s"])
    slopes = {
        name: exponent(
            [r["faces"] for r in rows],
            [max(r["check_timings_s"][name], 1e-4) for r in rows],
        )
        for name in names
    }
    path.write_text(
        json.dumps(
            {
                "rows": rows,
                "local_slope_against_faces": slopes,
                "validation_total_slope_against_faces": exponent(
                    [r["faces"] for r in rows], [r["validation_total_s"] for r in rows]
                ),
            },
            indent=1,
        )
    )


def run_oracle_child(n: int) -> dict:
    """Replay one validated Route B layout through the Gmsh API (and write its XAO), then load that XAO.

    Also records the mass (signed volume, um^3) of every solid of the replay and of the direct writer's XAO: a solid
    whose faces are listed in the wrong sense has a negative mass.
    """
    import gmsh
    from scgsim.geometry.sgb.build import emit_model
    from scgsim.geometry.sgb.model import build_model_from_catalogue
    from sgb_gmsh_oracle import emit_with_gmsh_api, gmsh_session
    from sgb_scenes import random_layout

    model = build_model_from_catalogue(random_layout(n, seed=SEED, route="B"), "B")
    with tempfile.TemporaryDirectory() as tmp:
        xao = Path(tmp) / "oracle.xao"
        started = time.perf_counter()
        with gmsh_session():
            result = emit_with_gmsh_api(model, xao)
            emit = time.perf_counter() - started
            oracle_mass = {
                model.topology.volumes[v].domain.name: round(
                    gmsh.model.occ.getMass(3, tag), 1
                )
                for v, (dim, tag) in result.tags.items()
                if dim == 3
            }
        started = time.perf_counter()
        with gmsh_session(xao):
            load = time.perf_counter() - started
        artifacts = emit_model(model, Path(tmp) / "direct")
        with gmsh_session(artifacts.xao):
            direct_mass = sorted(
                round(gmsh.model.occ.getMass(3, tag), 1)
                for dim, tag in gmsh.model.getEntities(3)
            )
    return {
        "n_features": n,
        "oracle_emit_s": round(emit, 3),
        "oracle_load_s": round(load, 3),
        "solid_mass_um3": {
            "gmsh_api_oracle": oracle_mass,
            "direct_writer": direct_mass,
        },
    }


def measure_oracle(path: Path) -> None:
    rows = []
    for n in ORACLE_NS:
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--oracle-child",
                    str(n),
                ],
                capture_output=True,
                text=True,
                timeout=ORACLE_CAP_S,
            )
            rows.append(json.loads(proc.stdout.strip().splitlines()[-1]))
        except subprocess.TimeoutExpired:
            rows.append({"n_features": n, "timeout_s": ORACLE_CAP_S})
        print(rows[-1], flush=True)
        path.write_text(json.dumps({"rows": rows}, indent=1))


def measure(path: Path) -> None:
    rows = []
    started = time.perf_counter()
    for n in NS:
        for tiles, workers in SCHEDULES:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--child",
                json.dumps([n, tiles, workers]),
            ]
            t0 = time.perf_counter()
            try:
                proc = subprocess.run(
                    command, capture_output=True, text=True, timeout=CAP_S
                )
                row = (
                    json.loads(proc.stdout.strip().splitlines()[-1])
                    if proc.returncode == 0
                    else {
                        "n_features": n,
                        "tiles_per_side": tiles,
                        "workers": workers,
                        "error": proc.stderr.strip().splitlines()[-1]
                        if proc.stderr
                        else "failed",
                    }
                )
            except subprocess.TimeoutExpired:
                row = {
                    "n_features": n,
                    "tiles_per_side": tiles,
                    "workers": workers,
                    "timeout_s": CAP_S,
                }
            row["case_wall_s"] = round(time.perf_counter() - t0, 2)
            rows.append(row)
            print(
                f"N={n:>5} T={tiles} w={workers:>2} build={row.get('build_wall_s')} case={row['case_wall_s']} s",
                flush=True,
            )
            path.write_text(json.dumps({"rows": rows}, indent=1))
    machine = {
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "total_wall_s": round(time.perf_counter() - started, 1),
    }
    path.write_text(
        json.dumps(
            {"machine": machine, "route": "B", "seed": SEED, "rows": rows}, indent=1
        )
    )
    print("wrote", path, machine)


# --------------------------------------------------------------------------- figures (from the measured file)


def load_measured(path: Path) -> list[dict]:
    rows = json.loads(path.read_text())["rows"]
    bad = [r for r in rows if "error" in r or "timeout_s" in r]
    if bad:
        raise SystemExit(f"measured cases failed: {bad}")
    return rows


def series(rows: list[dict], oracle: dict) -> dict:
    ns = sorted({r["n_features"] for r in rows})
    base = {
        r["n_features"]: r
        for r in rows
        if (r["tiles_per_side"], r["workers"]) == (1, 1)
    }
    tiled = {
        r["n_features"]: r
        for r in rows
        if (r["tiles_per_side"], r["workers"]) == (4, 1)
    }
    direct = {n: base[n]["emission"]["direct_w1"] for n in ns}
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
        "brep_write_w16_s": [
            round(
                sum(
                    base[n]["emission"]["direct_w16"][k]
                    for k in ("plan", "write_brep", "write_xao")
                ),
                3,
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
        "rss_tiled_mb": [tiled[n]["peak_rss_mb"] for n in ns],
        "file_mb": [
            {k: round(v / 1e6, 1) for k, v in base[n]["emission"]["file_bytes"].items()}
            for n in ns
        ],
        "written_counts": [base[n]["emission"]["written_counts"] for n in ns],
    }


def speedups(rows: list[dict]) -> dict:
    """Best tiled/parallel derivation speed-up over the serial one-tile run, per N (all schedules measured)."""
    out = {}
    for n in sorted({r["n_features"] for r in rows}):
        subset = [r for r in rows if r["n_features"] == n]
        base = next(r for r in subset if r["tiles_per_side"] == 1)["build_wall_s"]
        best = max(subset, key=lambda r: base / r["build_wall_s"])
        out[str(n)] = {
            "best_speedup": round(base / best["build_wall_s"], 2),
            "tiles_per_side": best["tiles_per_side"],
            "workers": best["workers"],
            "all": {
                f"T{r['tiles_per_side']}w{r['workers']}": round(
                    base / r["build_wall_s"], 2
                )
                for r in subset
            },
        }
    return out


def exponent(ns, ys) -> list[float]:
    """Local log-log slope between consecutive N."""
    return [
        round(math.log(y1 / y0) / math.log(n1 / n0), 2)
        for (n0, y0), (n1, y1) in zip(zip(ns, ys), zip(ns[1:], ys[1:]))
    ]


def identity(rows: list[dict]) -> dict:
    out = {}
    for n in sorted({r["n_features"] for r in rows}):
        subset = [r for r in rows if r["n_features"] == n]
        out[str(n)] = {
            "configurations": len(subset),
            "tiles_per_side": sorted({r["tiles_per_side"] for r in subset}),
            "workers": sorted({r["workers"] for r in subset}),
            "distinct_identity_digests": len({r["identity_digest"] for r in subset}),
            "identity_digest": subset[0]["identity_digest"],
            "all_validation_passed": all(r["validation_passed"] for r in subset),
        }
    return out


COLOUR = {
    "derive": "#2a6f9e",
    "api": "#c0392b",
    "write": "#e08a1e",
    "load": "#7f7f7f",
    "rss": "#2a8f5a",
    "ref": "#999999",
}
LINES = (
    ("derive_s", "derivation and validation S2-S9 (serial, T = 1)", "derive", "solid"),
    ("brep_write_s", "direct BREP/XAO writer (D8, one process)", "write", "solid"),
    ("gmsh_load_s", "Gmsh loading the XAO (outside SGB)", "load", "dash"),
)


def measured_points(ns, values):
    """The (N, value) pairs that were measured (the oracle has no value at the largest N)."""
    pairs = [(n, v) for n, v in zip(ns, values) if v is not None]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def scaling_figure(s: dict) -> None:
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from _intro_common import save_plotly, save_svg
    from plotly.subplots import make_subplots

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
            name="peak RSS after the build, one tile (T = 1)",
            line=dict(color=COLOUR["rss"]),
            hovertemplate="N = %{x}<br>%{y:.0f} MB<extra>T = 1</extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=ns,
            y=s["rss_tiled_mb"],
            mode="lines+markers",
            name="peak RSS after the build, 4 × 4 tiles, one worker",
            line=dict(color=COLOUR["rss"], dash="dash"),
            hovertemplate="N = %{x}<br>%{y:.0f} MB<extra>4 x 4 tiles</extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=ns,
            y=[s["rss_mb"][0] * n / ns[0] for n in ns],
            mode="lines",
            name="slope 1 from T = 1 at N = 1000",
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
    a2.loglog(
        ns, s["rss_mb"], marker="o", color=COLOUR["rss"], label="one tile (T = 1)"
    )
    a2.loglog(
        ns,
        s["rss_tiled_mb"],
        marker="s",
        color=COLOUR["rss"],
        ls="--",
        label="4 × 4 tiles, one worker",
    )
    a2.loglog(
        ns,
        [s["rss_mb"][0] * n / ns[0] for n in ns],
        color=COLOUR["ref"],
        ls=":",
        label="slope 1 from T = 1 at N = 1000",
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


def tail_figure(rows: list[dict]) -> dict:
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from _intro_common import save_plotly, save_svg

    base = next(
        r
        for r in rows
        if r["n_features"] == 20000 and (r["tiles_per_side"], r["workers"]) == (1, 1)
    )
    e = base["emission"]
    derive = base["build_wall_s"]

    def direct(workers: int) -> tuple[float, ...]:
        t = e[f"direct_w{workers}"]
        same = e[
            "direct_w1"
        ]  # reports and sidecar are written by the one-process run in both stacks
        return (
            derive,
            t["plan"] + t["write_brep"] + t["write_xao"],
            same["reports"],
            same["sidecar"],
            e["gmsh_load_s"],
            e["verify_loaded_s"],
        )

    stacks = {
        "direct BREP/XAO, one process": direct(1),
        "direct BREP/XAO, 16 workers": direct(16),
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
        for y, (l, v) in enumerate(zip(left, vals)):
            if v >= 4.0:
                ax.text(
                    l + v / 2,
                    y,
                    f"{v:.1f} s",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                    fontweight="bold",
                )
        left = [a + b for a, b in zip(left, vals)]
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
            ),
            total_s=round(sum(v), 1),
        )
        for name, v in stacks.items()
    }


def local_slopes(ns, values) -> list[float]:
    xs, ys = measured_points(ns, values)
    return exponent(xs, ys)


def plot(rows: list[dict], oracle: dict, machine: dict) -> None:
    from _intro_common import write_numbers

    s = series(rows, oracle)
    scaling_figure(s)
    tails = tail_figure(rows)
    numbers = {
        "series": s,
        "local_log_slope": {
            k: local_slopes(s["n"], s[k])
            for k in (
                "derive_s",
                "brep_write_s",
                "gmsh_load_s",
                "verify_loaded_s",
                "reports_s",
                "sidecar_s",
                "rss_mb",
                "rss_tiled_mb",
                "faces",
            )
        },
        "identity": identity(rows),
        "speedups": speedups(rows),
        "stack_n20000": tails,
        "oracle_local_log_slope": exponent(s["oracle"]["n"], s["oracle"]["emit_s"])
        if len(s["oracle"]["n"]) > 1
        else [],
        "oracle_ratio_to_direct_write_at_1000": round(
            s["oracle"]["emit_s"][s["oracle"]["n"].index(1000)]
            / series_value(s, "brep_write_s", 1000),
            1,
        ),
        "schedules": [f"T{t}w{w}" for t, w in SCHEDULES],
        "machine": machine,
        "source": "figures/why-scaling-measured.json (python why_scaling.py measure), figures/why-scaling-oracle.json (python why_scaling.py oracle)",
    }
    write_numbers("why-scaling", numbers)
    print(json.dumps(numbers, indent=1))


def series_value(s: dict, key: str, n: int) -> float:
    return s[key][s["n"].index(n)]


def main() -> None:
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        print(json.dumps(run_child(*json.loads(sys.argv[2]))))
        return
    if len(sys.argv) > 2 and sys.argv[1] == "--validator-child":
        print(json.dumps(run_validator_child(int(sys.argv[2]))))
        return
    if len(sys.argv) > 2 and sys.argv[1] == "--oracle-child":
        print(json.dumps(run_oracle_child(int(sys.argv[2]))))
        return
    figures = HERE.parent / "figures"
    measured, oracle_file = (
        figures / "why-scaling-measured.json",
        figures / "why-scaling-oracle.json",
    )
    if len(sys.argv) > 1 and sys.argv[1] == "measure":
        measure(measured)
    elif len(sys.argv) > 1 and sys.argv[1] == "validators":
        measure_validators(figures / "why-validator-scaling.json")
    elif len(sys.argv) > 1 and sys.argv[1] == "oracle":
        measure_oracle(oracle_file)
    else:
        oracle = {
            r["n_features"]: r for r in json.loads(oracle_file.read_text())["rows"]
        }  # N -> row
        plot(
            load_measured(measured),
            oracle,
            json.loads(measured.read_text()).get("machine", {}),
        )


if __name__ == "__main__":
    main()
