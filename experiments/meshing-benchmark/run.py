"""Benchmark of ``scgsim.meshing``: three sizing recipes through one pipeline, on fixed SGB cases.

Usage (from the repository root, with the test and palace dependencies synced)::

    uv run python experiments/meshing-benchmark/run.py OUTPUT_DIR [--cases NAME ...]
        [--no-determinism] [--compare experiments/meshing-benchmark/baseline.json]

For each case it builds the SGB XAO and v2 sidecar with the test input builders (``tests/`` goes on
``sys.path``), then meshes it with ``mesh_sgb`` in fresh worker processes, once per method (see ``METHODS``):

* ``module``: ``FeatureRecipe`` with repair, 3D Delaunay and the provisional threshold 0.01;
* ``module_hxt``: the same with ``MesherOptions(algorithm_3d=HXT)``, the former default;
* ``module_outline``: the same with the optional conductor-outline rule on (5 um on the conductors'
  outline edges, growing to the coarse size over 300 um); the table calls it "module + conductor-outline";
* ``R8``: ``GlobalSizeRecipe`` (only the global maximum size, bounding-box diagonal / 20), HXT, one mesh;
* ``RP``: ``UniformBoundaryRecipe`` (5 um on every curve growing to 300 um, as Palace does today),
  2D algorithm 5, one mesh;
* ``repair_only``: ``module`` on a sidecar without film facts.

``R8`` and ``RP`` run with ``Budgets(max_rounds=0)``: no repair, but the threshold still applies, so a mesh
below it is rejected and its numbers come from the report the error carries.

It writes ``OUTPUT_DIR/results.json`` (numbers only: no paths) and prints the table of the README.  The
module is run twice per case, in separate directories, to check that mesh and report are identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from importlib.metadata import version
from pathlib import Path
from typing import Any

import gdstk

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPOSITORY / "tests")]

from sgb_inputs import EffectiveInput, gds_pad_in_ground, thin_film_pair  # noqa: E402
from test_sgb_band_labels import PORT_CASES, all_interface_insets  # noqa: E402

from scgsim.geometry.models.input import GdsBoundaryReconstructionSpec  # noqa: E402
from scgsim.geometry.sgb.build import build_sgb_geometry  # noqa: E402
from scgsim.meshing import (  # noqa: E402
    HXT,
    Budgets,
    FeatureRecipe,
    GlobalSizeRecipe,
    MesherOptions,
    MeshRepairError,
    SizingRecipe,
    UniformBoundaryRecipe,
    mesh_sgb,
)

# Decision C (quality measure and threshold) is the Human's; 0.01 is what the feasibility prototype used.
PROVISIONAL_THRESHOLD = 0.01
PAD_RADIUS_UM = 10.0
# Cases whose repair-only variant runs too (no sidecar facts: the loop alone must find the sidewalls).
REPAIR_ONLY_CASES = ("B_1mm",)
# The references are not repaired: one mesh, judged by the threshold.
NO_REPAIR = Budgets(max_rounds=0)


@dataclass(frozen=True)
class Method:
    """What one benchmark method passes to ``mesh_sgb``."""

    recipe: SizingRecipe
    mesher: MesherOptions = MesherOptions()
    budgets: Budgets = Budgets()


# The mesher settings are those the former benchmark code gave each reference: R8 meshed with HXT, and RP
# with 2D algorithm 5 and the default 3D Delaunay.
METHODS: dict[str, Method] = {
    "module": Method(FeatureRecipe()),
    "module_outline": Method(FeatureRecipe(conductor_outline=True)),
    "module_hxt": Method(FeatureRecipe(), MesherOptions(algorithm_3d=HXT)),
    "R8": Method(GlobalSizeRecipe(), MesherOptions(algorithm_3d=HXT), NO_REPAIR),
    "RP": Method(UniformBoundaryRecipe(), MesherOptions(algorithm_2d=5), NO_REPAIR),
    "repair_only": Method(FeatureRecipe()),
}
METHOD_LABELS = {
    "module_outline": "module + conductor-outline",
    "module_hxt": "module, HXT",
    "repair_only": "module, repair only",
}


def main(argv: list[str] | None = None) -> int:
    """Run the selected cases and write ``results.json``."""
    args = _parse_arguments(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "environment": _environment(),
        "threshold": PROVISIONAL_THRESHOLD,
    }
    results["threshold_status"] = "provisional (decision C is open)"
    results["cases"] = {}
    for name in args.cases:
        print(f"== {name}", flush=True)
        results["cases"][name] = _run_case(name, args)
        (args.output_dir / "results.json").write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8"
        )
    print(_table(results))
    if args.write_baseline is not None:
        args.write_baseline.write_text(
            json.dumps(compact(results), indent=2) + "\n", encoding="utf-8"
        )
    if args.compare is not None:
        return _compare(results, json.loads(args.compare.read_text(encoding="utf-8")))
    return 0


def _parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument(
        "--no-determinism", action="store_true", help="run the module once per case"
    )
    parser.add_argument(
        "--write-baseline",
        type=Path,
        help="write the compact numbers of this run to this file",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        help="baseline.json to compare the deterministic numbers with",
    )
    return parser.parse_args(argv)


def _environment() -> dict[str, str]:
    return {
        "gmsh": version("gmsh"),
        "python": sys.version.split()[0],
        "scgsim": version("scgsim"),
    }


# ----------------------------------------------------------------------------------------------- cases


def _build_a(directory: Path) -> tuple[EffectiveInput, str]:
    return thin_film_pair(), "A"


def _rectangular(
    die_um: float, route: str
) -> Callable[[Path], tuple[EffectiveInput, str]]:
    return lambda directory: (gds_pad_in_ground(directory, die_um=die_um), route)


def _circular(die_um: float) -> Callable[[Path], tuple[EffectiveInput, str]]:
    def build(directory: Path) -> tuple[EffectiveInput, str]:
        """The pad-in-ground input with the pad's GDS polyline reconstructed as one exact circle."""
        centre = (die_um / 2, die_um / 2)
        effective = gds_pad_in_ground(
            directory,
            die_um=die_um,
            pad_polygon=gdstk.ellipse(centre, PAD_RADIUS_UM, tolerance=0.01),
        )
        (polygon,) = [
            p for p in effective.build_input.polygons if p.object_name == "pad"
        ]
        count = len(polygon.exterior)
        reconstruction = GdsBoundaryReconstructionSpec(
            "pad_circle",
            "pad",
            centre,
            "circular_arc",
            closed=True,
            segment_indices=tuple(range(count)),
            anchor_indices=(0, count // 3, 2 * count // 3),
        )
        curved = replace(
            effective.build_input, boundary_reconstruction=(reconstruction,)
        )
        return EffectiveInput(curved, effective.representation), "B"

    return build


CASES: dict[str, Callable[[Path], tuple[EffectiveInput, str]]] = {
    "A": _build_a,
    "B": _rectangular(100.0, "B"),
    "C": _rectangular(100.0, "C"),
    "circB": _circular(100.0),
    "B_1mm": _rectangular(1000.0, "B"),
    "circB_2mm": _circular(2000.0),
    "circB_4mm": _circular(4000.0),
}


def _build_case(name: str, directory: Path) -> tuple[Path, Path]:
    """Build the SGB XAO and sidecar of a case with every surface banded (0.5 and 1 um) and a port."""
    effective, route = CASES[name](directory)
    insets = all_interface_insets(
        effective, route, PORT_CASES["B" if route != "A" else "A"][1]
    )
    artifacts = build_sgb_geometry(
        effective.build_input,
        effective.representation,
        route=route,
        insets=insets,
        output_dir=directory,
    )
    return artifacts.xao, artifacts.sidecar


# -------------------------------------------------------------------------------------------- one case


def _run_case(name: str, args: argparse.Namespace) -> dict[str, Any]:
    directory = args.output_dir / "cases" / name
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    xao, sidecar = _build_case(name, directory)
    outcome: dict[str, Any] = {"build_seconds": round(time.perf_counter() - started, 2)}
    meshes = args.output_dir / "meshes" / name
    for method in ("module", "module_hxt", "module_outline"):
        outcome[method] = _run_method(xao, sidecar, meshes / method, method)
    if not args.no_determinism:
        outcome["determinism"] = _determinism(
            outcome["module"],
            xao,
            sidecar,
            args.output_dir / "meshes" / name / "module_again",
        )
    for method in ("R8", "RP"):
        outcome[method] = _run_method(xao, sidecar, meshes / method, method)
    if name in REPAIR_ONLY_CASES:
        outcome["repair_only"] = _run_method(
            xao, _without_film_facts(sidecar), meshes / "repair_only", "repair_only"
        )
    return outcome


def _run_method(
    xao: Path, sidecar: Path | dict[str, Any], directory: Path, method: str
) -> dict[str, Any]:
    """``mesh_sgb`` with the settings of ``method``; a rejected mesh is a result, not an error.

    A rejection carries the report of the mesh that failed, so its tetrahedra, minimum quality and count
    below the threshold are read from there; the stop reason is the run's status.
    """
    settings = METHODS[method]
    try:
        report = mesh_sgb(
            xao,
            sidecar,
            directory,
            threshold=PROVISIONAL_THRESHOLD,
            recipe=settings.recipe,
            budgets=settings.budgets,
            mesher=settings.mesher,
        ).report
        mesh_file = directory / f"{xao.stem}.msh"
    except MeshRepairError as failure:
        report = failure.report
        mesh_file = directory / f"{xao.stem}.rejected.msh"
    mesh = report["mesh"] or {}
    return {
        "recipe": {
            "name": report["recipe"]["name"],
            "version": report["recipe"]["version"],
        },
        "algorithm_2d": report["mesher"]["algorithm_2d"],
        "algorithm_3d": mesh.get("algorithm_3d"),
        "tetrahedra": mesh.get("tetrahedra"),
        "min_quality": mesh.get("min_quality"),
        "bad_count": mesh.get("bad_count"),
        "rounds": report["rounds"],
        "stop": report["stop"],
        "status": "accepted" if report["stop"] == "clean" else "rejected",
        "seconds": round(report["timing"]["total_seconds"], 2),
        "fallback": mesh.get("fallback"),
        "local_sizes": mesh.get("local_sizes"),
        "smallest_local_size_um": mesh.get("smallest_local_size_um"),
        "mesh_sha256": _sha256(mesh_file) if mesh_file.is_file() else None,
        "report": {key: value for key, value in report.items() if key != "timing"},
    }


def _without_film_facts(sidecar: Path) -> dict[str, Any]:
    """The sidecar without its ``film_sidewall`` features: what the repair loop does alone."""
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    document["mesh_features"] = [
        entry for entry in document["mesh_features"] if entry["kind"] != "film_sidewall"
    ]
    return document


def _determinism(
    first: dict[str, Any], xao: Path, sidecar: Path, directory: Path
) -> dict[str, Any]:
    """Run the module again in another directory; mesh bytes and report (without timing) must match."""
    second = _run_method(xao, sidecar, directory, "module")
    return {
        "same_mesh": first["mesh_sha256"] == second["mesh_sha256"],
        "same_report": first["report"] == second["report"],
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------ reporting


def compact(results: dict[str, Any]) -> dict[str, Any]:
    """The numbers worth committing: counts, qualities, stops and times; no meshes, hashes or reports."""
    keep = (
        "recipe",
        "algorithm_2d",
        "algorithm_3d",
        "tetrahedra",
        "min_quality",
        "bad_count",
        "rounds",
        "stop",
        "status",
        "seconds",
        "fallback",
        "local_sizes",
        "smallest_local_size_um",
    )
    cases = {}
    for name, case in results["cases"].items():
        cases[name] = {
            method: {key: case[method].get(key) for key in keep if key in case[method]}
            for method in METHODS
            if method in case
        }
        if "determinism" in case:
            cases[name]["determinism"] = case["determinism"]
    return {**{k: v for k, v in results.items() if k != "cases"}, "cases": cases}


def _table(results: dict[str, Any]) -> str:
    """The markdown table of the README."""
    header = (
        "| case | method | recipe | 2D / 3D | tetrahedra | min SICN | below | rounds | status | seconds | fallback |\n"
        "|---|---|---|---|---:|---:|---:|---:|---|---:|---|\n"
    )
    rows = []
    for name, case in results["cases"].items():
        for method in METHODS:
            if method in case:
                rows.append(_row(name, method, case[method]))
    return header + "\n".join(rows)


def _row(name: str, method: str, entry: dict[str, Any]) -> str:
    quality = "-" if entry["min_quality"] is None else f"{entry['min_quality']:.4g}"
    tetrahedra = "-" if entry["tetrahedra"] is None else f"{entry['tetrahedra']:,}"
    fallback = "yes" if entry["fallback"] else "no"
    algorithm_2d = "default" if entry["algorithm_2d"] is None else entry["algorithm_2d"]
    status = (
        entry["status"]
        if entry["status"] == "accepted"
        else f"rejected ({entry['stop']})"
    )
    recipe = entry["recipe"]["version"].removeprefix("scgsim.meshing.")
    return (
        f"| {name} | {METHOD_LABELS.get(method, method)} | {recipe} | {algorithm_2d} / {entry['algorithm_3d']} "
        f"| {tetrahedra} | {quality} | {entry['bad_count']} | {entry['rounds']} "
        f"| {status} | {entry['seconds']:.1f} | {fallback} |"
    )


def _same(old: Any, new: Any) -> bool:
    """Equal values; floats to a relative 1e-9, since only the last digits may differ between machines."""
    if isinstance(old, float) and isinstance(new, float):
        return math.isclose(old, new, rel_tol=1e-9)
    return bool(old == new)


def _compare(results: dict[str, Any], baseline: dict[str, Any]) -> int:
    """Print every difference in the deterministic numbers against the baseline; 1 if there is any."""
    differences = []
    for name, case in results["cases"].items():
        for method in METHODS:
            old, new = baseline["cases"].get(name, {}).get(method), case.get(method)
            if old is None or new is None:
                continue
            for field in ("tetrahedra", "min_quality", "bad_count", "rounds", "stop"):
                if not _same(old[field], new[field]):
                    differences.append(
                        f"{name} {method} {field}: {old[field]} -> {new[field]}"
                    )
    print("\n".join(differences) if differences else "identical to the baseline")
    return 1 if differences else 0


if __name__ == "__main__":
    raise SystemExit(main())
