"""Scaling of SGB's own runtime validators at N = 1000 / 5000 / 20000 (no OCCT at these sizes, phase 4).

For each N (a fresh subprocess, so memory is isolated) on the random layout (Route B, seed 1):

* the build with every structural validator timed separately (``ValidationReport.timings``), the different-Net
  contact check and the same-Net non-manifold check (separate build stages);
* the direct writer's own plan checks (edges on their surfaces, arc end points on their circles, short edges);
* the load checks (``verify_loaded`` without a mesh): counts, vertex placement, edge / face / solid boundaries,
  physical groups, volumes (orientation), face and group areas against the plan.

The parent derives, per check, the time per face and the log-log slope against the number of faces (1.0 = linear);
a check whose slope exceeds 1.15 is flagged with its complexity.

Usage:  python examples/validator_scaling.py [--ns 1000 5000 20000] [--cap 1800]
Output: outputs/validator_scaling.json and outputs/validator_scaling.md
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs"

# what each check does and its expected complexity (E = edges, F = faces, V = vertices, N = prisms)
COMPLEXITY = {
    "grid_and_lengths": "O(E)", "planarity": "O(E) with a hot-pixel grid", "arcs": "O(arc edges) with sampled crossings",
    "bands": "O(F of the banded levels)", "areas": "O(E)", "prism_areas": "O(E)", "shells": "O(F) per shell (parallel per tile)",
    "face_sides": "O(F)", "groups": "O(F)", "edges_in_faces": "O(E + F)",
    "contact_check": "O(V + F)", "manifold_check": "O(E)",
    "writer_plan_checks": "O(E + pcurves)", "counts_and_vertex_placement": "O(V) gmsh calls",
    "edge_boundaries": "O(E) gmsh calls", "face_boundaries": "O(F) gmsh calls", "solid_boundaries": "O(solids x faces)",
    "physical_groups": "O(entities) (was O(groups x entities): quadratic, fixed)", "volumes": "O(solids) mass integrals", "face_areas": "O(F) gmsh mass calls",
}


def child(n: int) -> dict:
    import gmsh

    from sgbproto.brep_writer import verify_loaded, write_xao
    from sgbproto.pipeline import build
    from sgbproto.scenes import random_layout

    spec = random_layout(n, seed=1, route="B")
    t0 = time.perf_counter()
    result = build(spec)
    build_wall = time.perf_counter() - t0
    topo = result.topology
    checks = {k: v for k, v in result.report.timings.items()}
    checks["contact_check"] = result.timings["contact_check"]
    checks["manifold_check"] = result.timings["manifold_check"]
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.perf_counter()
        written = write_xao(topo, Path(tmp) / "m.xao", name="model", workers=8, sidecar=False)
        write_wall = time.perf_counter() - t0
        checks["writer_plan_checks"] = written.plan.timings["arrays_and_checks"]
        gmsh.initialize(["-noenv"])
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("Geometry.OCCAutoFix", 0)
            t0 = time.perf_counter()
            gmsh.open(str(written.path))
            gmsh.model.occ.synchronize()
            load_wall = time.perf_counter() - t0
            t0 = time.perf_counter()
            report = verify_loaded(topo, written.plan, mesh_size=None, groups=True)
            verify_wall = time.perf_counter() - t0
        finally:
            gmsh.finalize()
    checks.update(report["timings_s"])
    return {
        "n_features": n, "counts": result.counts, "build_wall_s": round(build_wall, 3), "write_xao_s": round(write_wall, 3),
        "gmsh_load_s": round(load_wall, 3), "verify_loaded_s": round(verify_wall, 3), "verify_ok": report["ok"],
        "validation_passed": not result.report.failures,
        "checks_s": {k: round(v, 5) for k, v in checks.items()}, "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
    }


def slope(xs: list[float], ys: list[float]) -> float:
    """Least-squares slope of log y against log x."""
    pairs = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if y > 0]
    if len(pairs) < 2:
        return float("nan")
    mx, my = sum(p[0] for p in pairs) / len(pairs), sum(p[1] for p in pairs) / len(pairs)
    return sum((a - mx) * (b - my) for a, b in pairs) / sum((a - mx) ** 2 for a, _ in pairs)


def summarise(rows: list[dict]) -> list[dict]:
    faces = [r["counts"]["faces"] for r in rows]
    names = sorted({k for r in rows for k in r["checks_s"]}, key=lambda k: -rows[-1]["checks_s"].get(k, 0))
    table = []
    for name in names:
        times = [r["checks_s"].get(name, 0.0) for r in rows]
        s = slope(faces, times) if min(times) > 0 else float("nan")
        table.append({"check": name, "complexity": COMPLEXITY.get(name, ""), "seconds": times,
                      "us_per_face": [round(t / f * 1e6, 2) for t, f in zip(times, faces)],
                      "slope_vs_faces": round(s, 3) if s == s else None,
                      "flag": bool(s == s and s > 1.15 and times[-1] > 0.5)})
    return table


def markdown(rows: list[dict], table: list[dict]) -> str:
    head = "| check | complexity | " + " | ".join(f"N={r['n_features']} (F={r['counts']['faces']})" for r in rows) + " | us/face (first -> last) | slope vs F |"
    lines = [head, "|---|---|" + "---|" * len(rows) + "---|---|"]
    for t in table:
        cells = " | ".join(f"{s:.3f} s" for s in t["seconds"])
        slope_text = "n/a" if t["slope_vs_faces"] is None else f"{t['slope_vs_faces']:.2f}" + (" FLAG" if t["flag"] else "")
        lines.append(f"| {t['check']} | {t['complexity']} | {cells} | {t['us_per_face'][0]} -> {t['us_per_face'][-1]} | {slope_text} |")
    extra = ["", "| N | faces | edges | build wall | write XAO (8 workers) | Gmsh load | verify_loaded | peak RSS |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        extra.append(f"| {r['n_features']} | {r['counts']['faces']} | {r['counts']['edges']} | {r['build_wall_s']} s | {r['write_xao_s']} s | "
                     f"{r['gmsh_load_s']} s | {r['verify_loaded_s']} s | {r['peak_rss_mb']} MB |")
    return "\n".join(lines + extra) + "\n"


def parent(ns: list[int], cap: float) -> None:
    rows = []
    for n in ns:
        proc = subprocess.run([sys.executable, __file__, "--child", str(n)], capture_output=True, text=True, timeout=cap)
        if proc.returncode:
            raise SystemExit(f"N={n}: {proc.stderr.strip().splitlines()[-3:]}")
        rows.append(json.loads(proc.stdout.strip().splitlines()[-1]))
        print(f"N={n}: build {rows[-1]['build_wall_s']} s, verify_loaded {rows[-1]['verify_loaded_s']} s, ok={rows[-1]['verify_ok']}", flush=True)
    table = summarise(rows)
    (OUT / "validator_scaling.json").write_text(json.dumps({"rows": rows, "table": table}, indent=1))
    (OUT / "validator_scaling.md").write_text(markdown(rows, table))
    print(markdown(rows, table))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[1000, 5000, 20000])
    ap.add_argument("--cap", type=float, default=1800.0)
    ap.add_argument("--child", type=int)
    args = ap.parse_args()
    if args.child:
        print(json.dumps(child(args.child)))
    else:
        parent(args.ns, args.cap)
