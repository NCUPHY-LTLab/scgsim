"""Machine-readable outputs for the design documentation (examples, tests, figures).

Owns turning a :class:`BuildResult` (and optionally an :class:`EmitResult`)
into JSON files: a stage summary (counts per stage, timings), the validation
report, a lineage/identity table and a plain dump of every arrangement (edges
with carriers, faces with labels and ring coordinates in grid units) from
which computed figures can be drawn without re-running the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

from sgbproto.arcs import face_exact_area
from sgbproto.emit import EmitResult
from sgbproto.layers import Layer
from sgbproto.records import Domain
from sgbproto.pipeline import BuildResult
from sgbproto.sliver import sliver_report


def stage_summary(result: BuildResult, emitted: EmitResult | None = None) -> dict:
    topo = result.topology
    slab_stats = [{"plane": s.plane, **{k: v for k, v in s.stats.items() if k != "seconds"}} for s in result.slabs]
    level_stats = [{"plane": a.plane, **{k: v for k, v in a.stats.items() if k != "seconds"}} for a in result.level_layers]
    summary = {
        "model": result.spec.name,
        "route": result.policy.name,
        "grid_um": result.spec.grid.g,
        "stages": {
            "S2_prism_catalogue": {
                "prisms": len(result.spec.prisms),
                "levels": [str(z) for z in result.levels.levels],
                "slabs": result.levels.slab_count,
                "active_prisms_per_slab": [len(a) for a in result.levels.slab_active],
                "sheets_per_level": [len(s) for s in result.levels.level_sheets],
            },
            "S4_slab_arrangements": slab_stats,
            "S4_level_arrangements": level_stats,
            "S5_labelling_and_dissolution": {
                "edges_before": sum(x["edges_before"] for x in slab_stats + level_stats),
                "edges_after": sum(x["edges_after"] for x in slab_stats + level_stats),
                "cells_before": sum(x["faces_before"] for x in slab_stats + level_stats),
                "cells_after": sum(x["faces_after"] for x in slab_stats + level_stats),
            },
            "S6_topology": {k: v for k, v in topo.counts().items() if k != "groups"},
            "S7_route": {"route": result.policy.name, "excluded_volumes": [
                v.domain.name for v in topo.volumes.values() if not v.emitted
            ]},
            "S8_groups": {name: {"dim": g["dim"], "members": len(g["ids"])} for name, g in topo.groups.items()},
            "S9_validation": {
                "passed": result.report.to_dict()["passed"] if result.report else None,
                "checks": len(result.report.checks) if result.report else 0,
            },
        },
        "timings_s": {k: round(v, 5) for k, v in result.timings.items()},
        "stage_kind": result.stage_kind,
        "amdahl": {k: round(v, 5) for k, v in result.amdahl.items()},
        "parallel_stage_worker_compute_s": {k: round(v, 5) for k, v in result.stage_work.items()},
    }
    if emitted is not None:
        by_dim = {d: sum(1 for dim, _ in emitted.tags.values() if dim == d) for d in range(4)}
        summary["stages"]["S10_emission"] = {
            "native_entities_by_dim": by_dim,
            "timings_s": {k: round(v, 5) for k, v in emitted.timings.items()},
            "mesh": emitted.mesh,
            "curved_walls": {
                "count": len(emitted.wall_deviation),
                "max_vs_exact_cylinder_grid": max((w["vs_exact_cylinder_grid"] for w in emitted.wall_deviation), default=0.0),
                "walls": emitted.wall_deviation,
            },
        }
    return summary


def arrangement_dump(layers: list[Layer]) -> list[dict]:
    out = []
    for layer in layers:
        arr = layer.arr
        edges = [
            {"id": eid, "a": list(e.a), "b": list(e.b), "sources": list(src),
             **({"arc": {"centre": list(c.centre), "radius": c.radius, "ccw": c.ccw, "deviation": c.deviation}} if c else {})}
            for e, eid, src, c in zip(arr.edges, layer.edge_ids, layer.edge_sources, layer.edge_curve)
        ]
        faces = []
        for fi in range(1, len(arr.faces)):
            faces.append({
                "id": layer.face_ids[fi],
                "label": _label_text(layer.labels[fi]),
                "twice_area": arr.face_twice_area(fi),
                "exact_area": face_exact_area(arr, layer.edge_curve, fi) * 2,
                "outer": [list(p) for p in arr.cycle_points(arr.faces[fi].outer)],
                "holes": [[list(p) for p in arr.cycle_points(c)] for c in arr.faces[fi].holes],
            })
        out.append({"kind": layer.kind, "plane": layer.plane, "edges": edges, "faces": faces})
    return out


def _label_text(label) -> str:
    if label is None:
        return "outside"
    if hasattr(label, "below"):
        parts = [getattr(label.below, "name", "-"), getattr(label.above, "name", "-")]
        text = f"below={parts[0]} above={parts[1]}"
        if label.sheet is not None:
            text += f" sheet={label.sheet.kind}:{label.sheet.name}"
        if label.bands:
            text += " bands=" + ",".join(f"{f}:{k}" for f, k in label.bands)
        return text
    if isinstance(label, Domain):
        return label.name
    return str(label)


def write_case(result: BuildResult, out_dir: Path, emitted: EmitResult | None = None) -> dict[str, Path]:
    """Write summary, validation, lineage and arrangement dumps; return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "stage_summary": out_dir / "stage_summary.json",
        "validation_report": out_dir / "validation_report.json",
        "identity": out_dir / "identity.json",
        "arrangements": out_dir / "arrangements.json",
        "sliver_report": out_dir / "sliver_report.json",
    }
    paths["stage_summary"].write_text(json.dumps(stage_summary(result, emitted), indent=1))
    paths["validation_report"].write_text(json.dumps(result.report.to_dict(), indent=1))
    topo = result.topology
    identity = {
        "groups": {n: g for n, g in topo.groups.items()},
        "unions": topo.unions,
        "faces": {
            fid: {"kind": f.kind, "group": f.group, "membership": f.membership, "z_range": [str(f.z_range[0]), str(f.z_range[1])], "neg_volume": f.neg_volume, "pos_volume": f.pos_volume,
                  "sources": list(f.sources)}
            for fid, f in sorted(topo.faces.items())
        },
        "volumes": {
            vid: {"domain": v.domain.name, "kind": v.domain.kind, "emitted": v.emitted, "cells": list(v.cells),
                  "shells": [{"role": s.role, "faces": len(s.faces)} for s in v.shells],
                  "embedded_faces": v.embedded_faces}
            for vid, v in sorted(topo.volumes.items())
        },
        "edge_lineage": {eid: src for eid, src in sorted(topo.lineage.items())},
    }
    paths["identity"].write_text(json.dumps(identity, indent=1))
    paths["arrangements"].write_text(json.dumps(arrangement_dump(result.slabs + result.level_layers)))
    paths["sliver_report"].write_text(json.dumps(sliver_report(result), indent=1))
    return paths
