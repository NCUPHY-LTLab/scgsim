"""Build, count and mesh one sample scene: the numbers of the spline-curve study (issue 109).

A scene is a ``sgb_supported_kit.Scene`` (polygon outlines) or an ``arcchain.ArcScene`` (outlines of lines and
circular arcs).  For each it records

* the build: seconds of ``build_model_from_catalogue`` (Route B, MA/MS/SA bands at 0.5 and 1.0 um by type) and of
  ``emit_model``, or the error that refused it;
* the boundary: pieces of the conductor outlines as SGB receives them (straight edges, arc pieces);
* the emitted topology: vertices, edges, faces, and the horizontal band edges shorter than 25 grid steps (the
  count of issue 138);
* the mesh: ``mesh_sgb`` with its defaults (``FeatureRecipe``, ``RepairPolicy``, ``Budgets``) and the provisional
  threshold 0.01 on minSICN: tetrahedra, minimum quality, rounds, stop reason and seconds.

Usage (repository root, locked environment)::

    uv run --locked --no-sync python experiments/spline-curves/measure.py OUT_DIR SCENE [SCENE ...]

``OUT_DIR/<scene>.json`` receives the numbers; meshes and model files stay under ``OUT_DIR`` (not committed).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
sys.path[:0] = [str(REPOSITORY / "tests"), str(HERE)]

import scenes  # noqa: E402
from sgb_supported_kit import Scene, band_areas, reference, type_bands  # noqa: E402

from scgsim.geometry.sgb.build import emit_model  # noqa: E402
from scgsim.geometry.sgb.model import build_model_from_catalogue  # noqa: E402
from scgsim.meshing import MeshRepairError, mesh_sgb  # noqa: E402

THRESHOLD = (
    0.01  # provisional minSICN bound of the meshing benchmark (decision C is open)
)
SHORT_EDGE_STEPS = 25  # issue 138's count: band edges under 25 grid steps


def measure(
    name: str, out_dir: Path, *, mesh: bool = True, bands: bool = True
) -> dict[str, Any]:
    """Every number of one scene; a refused build or a rejected mesh is a result.

    ``bands=False`` builds without Surface-EPR bands: the control that separates the outline from its band seams.
    """
    scene = scenes.SCENES[name]()
    result: dict[str, Any] = {
        "scene": name,
        "bands_requested": bands,
        "boundary": scenes.boundary_pieces(scene),
    }
    started = time.perf_counter()
    try:
        insets = type_bands() if bands else ()
        model = build_model_from_catalogue(
            scenes.catalogue_of(scene), "B", insets=insets
        )
    except Exception as error:  # noqa: BLE001 - a refusal is the measurement
        result["build"] = {
            "status": "refused",
            "error": f"{type(error).__name__}: {str(error)[:300]}",
            "seconds": round(time.perf_counter() - started, 2),
        }
        return result
    build_seconds = time.perf_counter() - started
    started = time.perf_counter()
    artifacts = emit_model(model, out_dir / name)
    result["build"] = {
        "status": "built",
        "build_seconds": round(build_seconds, 2),
        "emit_seconds": round(time.perf_counter() - started, 2),
    }
    result["topology"] = topology_counts(model)
    if bands:
        result["bands"] = band_check(scene, model)
    if mesh:
        result["mesh"] = mesh_numbers(
            artifacts.xao, artifacts.sidecar, out_dir / name / "mesh"
        )
    return result


def topology_counts(model: Any) -> dict[str, Any]:
    """Entity counts, and the horizontal band edges shorter than :data:`SHORT_EDGE_STEPS` with where they are."""
    topo = model.topology
    short: list[dict[str, Any]] = []
    band_edges = 0
    arc_edges = 0
    for edge in topo.edges.values():
        if edge.kind != "horizontal":
            continue
        if edge.curve is not None:
            arc_edges += 1
        if not any("@m" in source for source in edge.sources):
            continue
        band_edges += 1
        a, b = topo.vertices[edge.v0], topo.vertices[edge.v1]
        length = math.hypot(b.x - a.x, b.y - a.y)
        if length < SHORT_EDGE_STEPS:
            short.append(
                {
                    "steps": round(length, 2),
                    "at_um": [
                        round(a.x * topo.grid_um, 4),
                        round(a.y * topo.grid_um, 4),
                    ],
                    "z": str(a.z),
                    "sources": sorted(edge.sources)[:2],
                }
            )
    return {
        **topo.counts(),
        "horizontal_arc_edges": arc_edges,
        "band_edges": band_edges,
        "short_band_edges": len(short),
        "short_band_edge_examples": short[:12],
    }


def band_check(scene: Any, model: Any) -> dict[str, Any]:
    """Band areas of the build against the exact erosion of the same outline (``sgb_supported_kit.reference``).

    A polygon scene is its own reference.  An arc-chain scene is compared with the polygon of its arcs'
    witness points (chords within g/4 of each arc), so the difference may also hold up to g/4 times the
    outline length per band on top of the snap bound; ``worst_excess_um2`` is the largest difference beyond
    the snap bound, and ``witness_allowance_um2`` that extra allowance.
    """
    reference_scene = scene if isinstance(scene, Scene) else scenes.witness_scene(scene)
    expected = reference(reference_scene)
    got = band_areas(model)
    allowance = scenes.witness_allowance_um2(scene)
    worst, families = 0.0, 0
    for family, ref in expected.items():
        families += 1
        for area, exact, bound in zip(
            got.get(family, []), ref.areas, ref.bounds, strict=False
        ):
            worst = max(worst, abs(area - exact) - bound)
    return {
        "families": families,
        "families_built": len(got),
        "same_families": sorted(got) == sorted(expected),
        "worst_excess_um2": worst,
        "witness_allowance_um2": allowance,
    }


def mesh_numbers(xao: Path, sidecar: Path, out_dir: Path) -> dict[str, Any]:
    """``mesh_sgb`` with its defaults; a rejection is read from the report it carries."""
    try:
        report = mesh_sgb(xao, sidecar, out_dir, threshold=THRESHOLD).report
        status = "accepted"
    except MeshRepairError as error:
        report = error.report
        status = "rejected"
    mesh = report["mesh"] or {}
    return {
        "status": status,
        "stop": report["stop"],
        "rounds": report["rounds"],
        "tetrahedra": mesh.get("tetrahedra"),
        "min_sicn": mesh.get("min_quality"),
        "bad": mesh.get("bad_count"),
        "seconds": round(report["timing"]["total_seconds"], 2),
        "round_seconds": [round(s, 2) for s in report["timing"]["round_seconds"]],
    }


def main(argv: list[str] | None = None) -> int:
    """Measure the named scenes, one JSON file each."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("scenes", nargs="+", choices=sorted(scenes.SCENES))
    parser.add_argument("--no-mesh", action="store_true")
    parser.add_argument(
        "--no-bands",
        action="store_true",
        help="build without Surface-EPR bands (the control)",
    )
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.scenes:
        try:
            result = measure(
                name, args.out_dir, mesh=not args.no_mesh, bands=not args.no_bands
            )
        except Exception:  # noqa: BLE001 - record and go on with the next scene
            result = {"scene": name, "crash": traceback.format_exc()[-2000:]}
        (args.out_dir / f"{name}.json").write_text(
            json.dumps(result, indent=1) + "\n", encoding="utf-8"
        )
        print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
