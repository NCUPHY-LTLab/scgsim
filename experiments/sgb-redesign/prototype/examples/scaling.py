"""Scaling of the derivation stages on the randomized layout generator.

Each (N, tiles, workers) case runs in its own subprocess with a wall-clock cap
(default 600 s) so memory and failures are isolated. ``tiles = T`` means a
``T x T`` tile grid; ``workers`` is the number of worker processes of the
per-tile parallel stages (``workers`` without tiles has nothing to split).
Every stage is booked as serial or parallel, with the summed worker compute of
the parallel stages, which gives the Amdahl split.

Usage:  python examples/scaling.py [--ns 1000 5000 20000] [--cap 600] [--emit-max-n 20000]
Writes: outputs/scaling.json (and prints a table).
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONFIGS = [(1, 1)] + [(t, w) for t in (4, 8) for w in (1, 8, 16, 32)]  # (tiles per side T, workers)


def run_child(n: int, tiles: int, workers: int, emit_it: bool, dense: bool) -> dict:
    """Executed in the subprocess: build, optionally emit, report timings and memory."""
    from sgbproto.emit import emit
    from sgbproto.pipeline import build, ids_digest
    from sgbproto.scenes import random_layout

    t0 = time.perf_counter()
    kwargs = {"pitch": 120.0, "overlap_fraction": 0.6} if dense else {}
    spec = random_layout(n, seed=1, route="B", **kwargs)
    generate = time.perf_counter() - t0
    t0 = time.perf_counter()
    result = build(spec, tiles=tiles, workers=workers)
    wall_build = time.perf_counter() - t0
    row = {
        "n_features": n, "dense": dense, "tiles_per_side": tiles, "tile_count": tiles * tiles, "workers": workers,
        "generate_s": round(generate, 3), "build_wall_s": round(wall_build, 3),
        "stage_s": {k: round(v, 3) for k, v in result.timings.items()},
        "stage_kind": result.stage_kind,
        "stage_worker_compute_s": {k: round(v, 3) for k, v in result.stage_work.items()},
        "amdahl": {k: round(v, 3) for k, v in result.amdahl.items()},
        "counts": result.counts, "ids_digest": ids_digest(result),
        "validation_passed": not result.report.failures,
    }
    if emit_it:
        t0 = time.perf_counter()
        emitted = emit(result.topology, ROOT / "outputs" / "_scaling_tmp", f"n{n}", orientation_attempts=0)
        row["emit_s"] = round(time.perf_counter() - t0, 3)
        row["emit_stage_s"] = {k: round(v, 3) for k, v in emitted.timings.items()}
        row["native_entities"] = len(emitted.tags)
    row["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    row["peak_worker_rss_mb"] = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, 1)
    return row


def run_parent(args: argparse.Namespace) -> None:
    rows = []
    for dense in (False, True) if args.dense else (False,):
        for n in args.ns:
            for tiles, workers in CONFIGS:
                emit_it = n <= args.emit_max_n and (tiles, workers) == (1, 1)
                cmd = [sys.executable, __file__, "--child", json.dumps([n, tiles, workers, emit_it, dense])]
                t0 = time.perf_counter()
                try:
                    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.cap)
                    if proc.returncode != 0:
                        row = {"n_features": n, "dense": dense, "tiles_per_side": tiles, "workers": workers,
                               "error": proc.stderr.strip().splitlines()[-1] if proc.stderr else "failed"}
                    else:
                        row = json.loads(proc.stdout.strip().splitlines()[-1])
                except subprocess.TimeoutExpired:
                    row = {"n_features": n, "dense": dense, "tiles_per_side": tiles, "workers": workers,
                           "timeout_s": args.cap}
                row["case_wall_s"] = round(time.perf_counter() - t0, 2)
                rows.append(row)
                print(summarise(row), flush=True)
    out = ROOT / "outputs" / "scaling.json"
    out.write_text(json.dumps({"cap_s": args.cap, "route": "B", "rows": rows}, indent=1))
    print(f"wrote {out}")


def summarise(row: dict) -> str:
    head = f"N={row['n_features']:>5} dense={int(row['dense'])} T={row['tiles_per_side']} w={row['workers']:>2}"
    if "error" in row or "timeout_s" in row:
        return f"{head}  {row.get('error', 'TIMEOUT')}"
    emit = f" emit={row['emit_s']:.1f}s" if "emit_s" in row else ""
    a = row["amdahl"]
    return (f"{head}  build={row['build_wall_s']:.2f}s{emit} rss={row['peak_rss_mb']:.0f}MB "
            f"serial={a['serial_stages_s']:.1f}s parallel_wall={a['parallel_stages_wall_s']:.1f}s "
            f"worker_compute={a['parallel_stages_work_s']:.1f}s faces={row['counts']['faces']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ns", type=int, nargs="+", default=[1000, 5000, 20000])
    parser.add_argument("--cap", type=float, default=600.0)
    parser.add_argument("--emit-max-n", type=int, default=20000)
    parser.add_argument("--dense", action="store_true", help="also run the dense/overlapping variant")
    parser.add_argument("--child", type=str, default=None)
    ns = parser.parse_args()
    if ns.child:
        print(json.dumps(run_child(*json.loads(ns.child))))
    else:
        run_parent(ns)
