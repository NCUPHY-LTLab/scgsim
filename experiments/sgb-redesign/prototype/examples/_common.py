"""Shared driver for the example scripts: build, emit, mesh-check and write outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sgbproto.emit import emit  # noqa: E402
from sgbproto.pipeline import build  # noqa: E402
from sgbproto.records import ModelSpec  # noqa: E402
from sgbproto.report import write_case  # noqa: E402

OUTPUTS = ROOT / "outputs"


def run_case(spec: ModelSpec, case: str, mesh_size: float, mesh_size_min: float = 0.0, **emit_kwargs) -> dict:
    """Run S2-S10 on ``spec`` and write every artefact under ``outputs/<case>/``."""
    out_dir = OUTPUTS / case
    result = build(spec)
    emitted = emit(result.topology, out_dir, "model", mesh_size=mesh_size, mesh_size_min=mesh_size_min, **emit_kwargs)
    write_case(result, out_dir, emitted)
    summary = {
        "case": case,
        "counts": result.counts,
        "validation_passed": not result.report.failures,
        "mesh_conformal": emitted.mesh["conformal"],
        "mesh": {k: emitted.mesh[k] for k in ("nodes", "tets", "duplicate_nodes", "shared_interior_faces")},
        "arc_edges": sum(c is not None for layer in result.level_layers for c in layer.edge_curve),
        "curved_wall_max_deviation_grid": max((w["vs_exact_cylinder_grid"] for w in emitted.wall_deviation), default=None),
        "timings_s": {k: round(v, 4) for k, v in result.timings.items()},
    }
    print(json.dumps(summary, indent=1))
    if not emitted.mesh["conformal"]:
        raise SystemExit(f"{case}: mesh is not conformal: {emitted.mesh['problems']}")
    return summary
