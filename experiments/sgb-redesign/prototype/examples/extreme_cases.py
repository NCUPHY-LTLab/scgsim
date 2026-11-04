"""Run every minimal extreme-case fixture through every check (phase 4).

For each (fixture, route) of ``sgbproto.extreme_cases.FIXTURES``: build + structural validation, direct BREP / XAO
write, Gmsh load and load checks, a conformal 3D mesh, and the official OCCT checks (BRepCheck_Analyzer,
BOPAlgo_ArgumentAnalyzer with every mode including self-interference, ShapeAnalysis, plan cross-check). Each case runs
``--repeat`` times; the table shows the median time of every check (milliseconds) -- the point is that the official
checkers are cheap at this size. Refused fixtures show the early error instead.

Output: ``outputs/extreme/rows.json`` (every row, every time), ``outputs/extreme/catalogue.md`` (the table) and
``outputs/extreme/band_klayout.json`` (inset-band areas against the KLayout erosion oracle for the banded fixtures).

Usage:  python examples/extreme_cases.py [--repeat 3] [names ...]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs" / "extreme"

BANDED = ("neck_split_by_inset", "band_vanishes", "reflex_small_angle", "refinement_crossing_band_edge")


def median_row(rows: list[dict]) -> dict:
    """The first row with every time replaced by the median over the repeats."""
    out = dict(rows[0])
    out["times_s"] = {k: statistics.median(r["times_s"][k] for r in rows) for k in rows[0]["times_s"]}
    out["repeats"] = len(rows)
    return out


def cell(seconds: float | None, ok: bool = True, text: str = "ok") -> str:
    if seconds is None:
        return "-"
    return f"{text if ok else 'FAIL'} {seconds * 1000:.1f}"


def sum_of(times: dict, *prefixes: str) -> float:
    return sum(v for k, v in times.items() if any(k.startswith(p) for p in prefixes))


def table(rows: list[dict]) -> str:
    head = ("| fixture | route | risk | faces / edges | build + validation | write BREP + XAO | Gmsh load + checks | mesh | "
            "BRepCheck_Analyzer | BOPAlgo_ArgumentAnalyzer (all modes) | ShapeAnalysis + plan | total |")
    lines = [head, "|" + "---|" * 12]
    for r in rows:
        t = r["times_s"]
        counts = f"{r['plan_counts']['faces']} / {r['plan_counts']['edges']}" if "plan_counts" in r else "-"
        base = f"| {r['fixture']} | {r['route']} | {r['risk']} | {counts} |"
        if r.get("expected_error") and r.get("error"):
            lines.append(f"{base} refused: **{r['error']['type']}** at build ({t['build'] * 1000:.1f} ms) | - | - | - | - | - | - | "
                         f"{t['build'] * 1000:.1f} |")
            continue
        if "gmsh" not in r and not r.get("writer_refused"):
            lines.append(f"{base} UNEXPECTED: " + (r["error"]["type"] if r.get("error") else "no error raised") + " | - | - | - | - | - | - | - |")
            continue
        if r.get("writer_refused"):
            oracle = r.get("oracle") or {}
            lines.append(f"{base} ok {t['build'] * 1000:.1f} | writer refused: 5-edge wall (see notes) | "
                         f"gmsh-API oracle: conformal={oracle.get('mesh_conformal')} {t.get('oracle_emit_and_mesh', 0) * 1000:.1f} | - | - | - | - | "
                         f"{(t['build'] + t.get('oracle_emit_and_mesh', 0)) * 1000:.1f} |")
            continue
        g_ok = all(g["ok"] for g in r["gmsh"].values())
        o_ok = all(o["ok"] for o in r["occt"].values())
        gmsh_load = sum_of(t, "gmsh_load", "gmsh_verify")
        mesh = sum_of(t, "gmsh_mesh")
        brep_check = sum_of(t, "occt_brep_check")
        bop = sum_of(t, "occt_bop_argument_analyzer")
        other = sum_of(t, "occt_shape_analysis", "occt_plan_properties")
        total = sum(v for v in t.values())
        lines.append(f"{base} {cell(t['build'])} | {cell(t['write_brep'] + t['write_xao'])} | {cell(gmsh_load, g_ok)} | "
                     f"{cell(mesh, g_ok, 'conformal')} | {cell(brep_check, o_ok)} | {cell(bop, o_ok)} | {cell(other, o_ok)} | {total * 1000:.1f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    from sgbproto.extreme_cases import FIXTURES, band_klayout_rows, cases, run_fixture

    OUT.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for fixture, route in cases():
            if args.names and fixture.name not in args.names:
                continue
            runs = [run_fixture(fixture, route, Path(tmp) / f"{fixture.name}_{route}_{i}") for i in range(args.repeat)]
            row = median_row(runs)
            rows.append(row)
            outcome = "ok" if row["ok"] else "UNEXPECTED"
            if not row["ok"]:
                failures.append(f"{fixture.name} {route}")
            detail = row["error"]["type"] if row.get("error") else ("writer refused" if row.get("writer_refused") else
                     {k: v["ok"] for k, v in row.get("occt", {}).items()})
            print(f"{fixture.name:36s} {route} {outcome:10s} {sum(row['times_s'].values()) * 1000:7.1f} ms  {detail}", flush=True)
    (OUT / "rows.json").write_text(json.dumps(rows, indent=1, default=str))
    notes = [f"- **{f.name}**: {f.note}" for f in FIXTURES.values() if f.note]
    (OUT / "catalogue.md").write_text(table(rows) + "\nNotes\n\n" + "\n".join(notes) + "\n\nTimes are medians of "
                                      f"{args.repeat} runs in milliseconds; 'total' is the sum of every listed check.\n")
    klayout = {}
    for name in BANDED:
        if args.names and name not in args.names:
            continue
        for route in FIXTURES[name].routes:
            klayout[f"{name}_{route}"] = band_klayout_rows(FIXTURES[name], route)
    (OUT / "band_klayout.json").write_text(json.dumps(klayout, indent=1))
    print(table(rows))
    if failures:
        raise SystemExit("unexpected outcome: " + ", ".join(failures))


if __name__ == "__main__":
    main()
