"""Official OCCT checks (OCP) on the written models, negative controls and timings (phase 4).

Default run -> ``outputs/occt/``:
  ``reports/<case>.json``       BRepCheck / BOPAlgo_ArgumentAnalyzer / ShapeAnalysis / plan cross-check of the BREP
                                and of the BREP inside the XAO, for the example and band cases;
  ``negative_controls.json/.md`` which official check sees each deliberate defect;
The checks are not scale-tested: the OCCT checkers run on the minimal extreme-case fixtures
(``examples/extreme_cases.py``) and on the five example cases here, never above N = 1000 features.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

OUT = ROOT / "outputs" / "occt"


def cases():
    from sgbproto.scenes import airbridge, pad_in_ground

    return {
        "pad_in_ground_A": pad_in_ground("A"), "pad_in_ground_B": pad_in_ground("B"), "airbridge_B": airbridge(),
        "pad_in_ground_A_bands": pad_in_ground("A", bands_um=(0.1, 0.5, 2.0)),
        "pad_in_ground_B_bands": pad_in_ground("B", bands_um=(0.1, 0.5, 2.0)),
    }


def default_run() -> None:
    import tempfile

    import occt_mutations as mutations

    from sgbproto.brep_writer import write_brep, write_xao
    from sgbproto.occt_check import check_file, check_shape, read_brep
    from sgbproto.extreme_cases import FIXTURES
    from sgbproto.pipeline import build

    (OUT / "reports").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, spec in cases().items():
            topo = build(spec).topology
            brep, xao = write_brep(topo, tmp / f"{name}.brep", sidecar=False), write_xao(topo, tmp / f"{name}.xao", sidecar=False)
            reports = {"brep": check_file(brep.path, topo, brep.plan), "xao": check_file(xao.path, topo, xao.plan)}
            (OUT / "reports" / f"{name}.json").write_text(json.dumps({k: r.to_dict() for k, r in reports.items()}, indent=1))
            print(name, {k: ("clean" if r.ok else r.problems) for k, r in reports.items()})

        # the negative controls run on a minimal fixture (three nested hole loops, 39 faces), not on the 300 um example
        topo = build(FIXTURES["nested_holes"].build("B")).topology
        written = write_brep(topo, tmp / "neg.brep", sidecar=False)
        rows = {}
        for control, mutate in mutations.CONTROLS.items():
            t0 = time.perf_counter()
            report = check_shape(mutate(read_brep(written.path)), topo, written.plan, source=control)
            rows[control] = {
                "check_ms": round((time.perf_counter() - t0) * 1000, 1),
                "BRepCheck_Analyzer": not report.brepcheck["valid"],
                "BRepCheck_statuses": report.brepcheck["histogram"],
                "BOPAlgo_ArgumentAnalyzer": report.arguments["faulty"],
                "BOP_statuses": sorted({r["status"] for r in report.arguments["results"]}),
                "ShapeAnalysis_Shell": bool(report.shape_analysis["shells_with_free_edges"] or report.shape_analysis["shells_with_bad_edges"]),
                "plan_cross_check_area_volume": bool(report.properties["mismatches"]),
            }
        (OUT / "negative_controls.json").write_text(json.dumps(rows, indent=1))
        (OUT / "negative_controls.md").write_text(control_table(rows))
        print(control_table(rows))


def control_table(rows: dict) -> str:
    mark = lambda b: "caught" if b else "-"
    lines = ["| defect | BRepCheck_Analyzer | BOPAlgo_ArgumentAnalyzer | ShapeAnalysis_Shell | plan area/volume | all checks (ms) |", "|---|---|---|---|---|---|"]
    for name, r in rows.items():
        statuses = f" ({', '.join(r['BRepCheck_statuses'])})" if r["BRepCheck_statuses"] else " (invalid; no status on the shape itself)"
        bop = f" ({', '.join(r['BOP_statuses'])})" if r["BOP_statuses"] else ""
        lines.append(f"| {name} | {mark(r['BRepCheck_Analyzer'])}{statuses if r['BRepCheck_Analyzer'] else ''} | "
                     f"{mark(r['BOPAlgo_ArgumentAnalyzer'])}{bop} | {mark(r['ShapeAnalysis_Shell'])} | {mark(r['plan_cross_check_area_volume'])} | {r['check_ms']} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    default_run()
