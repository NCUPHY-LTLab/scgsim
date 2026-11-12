"""Measuring half of ``why_scaling.py``: one case per interpreter, so each case reports its own peak memory.

Run by ``why_scaling.py``; the children are started as ``python why_scaling_measure.py --child '[n, tiles, workers]'``,
``--validator-child N`` and ``--oracle-child N``, and print one JSON row."""

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

from sgb_scenes import random_layout  # noqa: E402

from scgsim.geometry.sgb.model import build_model_from_catalogue  # noqa: E402

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
    # loaded here, after the build's peak memory was recorded, so the emission modules and Gmsh stay out of that number
    from sgb_gmsh_oracle import gmsh_session  # noqa: PLC0415 -- loaded after the build's peak memory is recorded

    from scgsim.contracts.sgb_sidecar import read_sidecar  # noqa: PLC0415 -- loaded after the build's peak memory is recorded
    from scgsim.geometry.sgb.build import emit_model  # noqa: PLC0415 -- loaded after the build's peak memory is recorded
    from scgsim.geometry.sgb.emission.plan import plan_brep  # noqa: PLC0415 -- loaded after the build's peak memory is recorded
    from scgsim.geometry.sgb.emission.writer import (  # noqa: PLC0415 -- loaded after the build's peak memory is recorded
        write_brep_file,
        write_xao_file,
    )
    from scgsim.geometry.sgb.verification.loaded import verify_loaded  # noqa: PLC0415 -- loaded after the build's peak memory is recorded

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
    """Time every structural validation check at each N, one interpreter each, and write the validator file."""
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
    import gmsh  # noqa: PLC0415 -- only the oracle case needs Gmsh; the build cases report peak memory
    from sgb_gmsh_oracle import emit_with_gmsh_api, gmsh_session  # noqa: PLC0415 -- only the oracle case needs Gmsh; the build cases report peak memory

    from scgsim.geometry.sgb.build import emit_model  # noqa: PLC0415 -- only the oracle case emits; the build cases report peak memory and load no emission code

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
    """Replay the models through the Gmsh-API oracle, one interpreter each with a time cap, and write the oracle file."""
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
    """Run every case of the scaling study, one interpreter each, and write the raw measured file."""
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


def exponent(ns, ys) -> list[float]:
    """Local log-log slope between consecutive N."""
    return [
        round(math.log(y1 / y0) / math.log(n1 / n0), 2)
        for (n0, y0), (n1, y1) in zip(
            zip(ns, ys, strict=False), zip(ns[1:], ys[1:], strict=False), strict=False
        )
    ]


def main() -> None:
    """Run one measuring child: print the JSON row of one case, one validation run or one oracle replay."""
    mode, argument = sys.argv[1], sys.argv[2]
    if mode == "--child":
        print(json.dumps(run_child(*json.loads(argument))))
    elif mode == "--validator-child":
        print(json.dumps(run_validator_child(int(argument))))
    elif mode == "--oracle-child":
        print(json.dumps(run_oracle_child(int(argument))))
    else:
        raise SystemExit(f"unknown child mode {mode!r}")


if __name__ == "__main__":
    main()
