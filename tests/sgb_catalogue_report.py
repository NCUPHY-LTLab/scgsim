"""Measurements over the extreme-case fixtures and the running examples (decisions D2 and D7).

Not a test module: ``measure`` is used by ``test_sgb_extreme_cases`` and ``main`` prints the tables the Lead
reads.  Per (fixture, route) it records what decides the reopen triggers of the two Human decisions:

* **D2** (no sliver thresholds; reopen if narrow faces cause malformed mesh elements): the quality of a
  coarse tetrahedral mesh -- minimum and 1st percentile of Gmsh's ``minSICN`` (signed inverse condition
  number, 1 = regular, <= 0 = inverted or flat) and ``gamma`` (inscribed / circumscribed radius ratio) --
  next to the narrowest face and the shortest edge of the model;
* **D7** (record drift only; reopen if drift above 1 g appears): the largest snap drift and the largest
  arc deviation, in grid units;
* the OCCT checks (when OCP is installed) and the load-side check.

Run it as ``python tests/sgb_catalogue_report.py [--out DIR] [--timings]`` from the package root.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import gmsh
from sgb_emit_cases import CASES, pad_requests
from sgb_extreme_cases import build_model, cases
from sgb_gmsh_oracle import gmsh_session, mesh_quality
from sgb_scenes import pad_in_ground, random_layout

from scgsim.contracts.sgb_sidecar import read_sidecar
from scgsim.geometry.sgb.build import emit_model
from scgsim.geometry.sgb.checks.verification.loaded import (
    apply_sidecar_embeds,
    verify_loaded,
)
from scgsim.geometry.sgb.checks.verification.ocp import ocp_available
from scgsim.geometry.sgb.model import ConformalModel, build_model_from_catalogue
from scgsim.geometry.sgb.output.emission.plan import plan_brep

MESH_SIZE_UM = 4.0  # coarse: the fixtures are 40 x 40 x 20 um boxes
NOT_MEASURED = {"mesh": "not meshed"}


def measure(
    model: ConformalModel,
    workdir: Path,
    *,
    mesh_size: float | None = MESH_SIZE_UM,
    with_occt: bool = True,
) -> dict:
    """Emit a model and measure it: counts, drift, load check, OCCT checks and coarse-mesh quality."""
    started = time.perf_counter()
    artifacts = emit_model(model, workdir)
    sidecar = read_sidecar(artifacts.sidecar)
    summary = sidecar["summary"]
    row: dict = {
        "route": sidecar["model"]["route"],
        "faces": summary["counts"]["faces"],
        "solids": summary["counts"]["solids"],
        "max_snap_drift_grid": summary["slivers"].get("max_snap_drift_grid"),
        "max_arc_deviation_grid": summary["slivers"].get("max_arc_deviation_grid"),
        "narrowest_face_um": summary["slivers"].get("narrowest_face_um"),
        "shortest_edge_um": summary["slivers"].get("shortest_edge_um"),
    }
    with gmsh_session(artifacts.xao):
        load = verify_loaded(sidecar)
        row["load_check"] = "ok" if load.ok else "; ".join(load.problems[:2])
        if mesh_size is not None:
            row.update(_mesh_row(sidecar, mesh_size))
    if with_occt and ocp_available():
        from scgsim.geometry.sgb.checks.verification.occt import check_file  # noqa: PLC0415 -- optional OCP (the occt group)

        plan = plan_brep(model.topology, model.routed)
        report = check_file(artifacts.xao, model.topology, plan, self_interference=True)
        row["occt"] = "ok" if report.ok else "; ".join(report.problems[:2])
    row["seconds"] = round(time.perf_counter() - started, 3)
    return row


def _mesh_row(sidecar: dict, size: float) -> dict:
    apply_sidecar_embeds(sidecar)
    gmsh.option.setNumber("Mesh.MeshSizeMax", size)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    started = time.perf_counter()
    try:
        gmsh.model.mesh.generate(3)
    except Exception as error:  # noqa: BLE001 -- a meshing failure is a measurement, not a crash
        return {"mesh": f"failed: {str(error)[:80]}"}
    return {
        "mesh": "ok",
        "mesh_seconds": round(time.perf_counter() - started, 3),
        **mesh_quality(),
    }


def fixture_rows(
    workdir: Path, *, mesh_size: float | None = MESH_SIZE_UM
) -> list[dict]:
    """One row per (fixture, route) the catalogue builds; refused fixtures are listed with their error."""
    rows = []
    for fixture, route in cases():
        row = {"fixture": fixture.name, "route": route}
        expected = fixture.expect_error.get(route)
        if expected is not None:
            row["refused"] = expected.__name__
        else:
            folder = workdir / f"{fixture.name}_{route}"
            row.update(
                measure(build_model(fixture, route), folder, mesh_size=mesh_size)
            )
        rows.append(row)
    return rows


def example_rows(workdir: Path) -> list[dict]:
    """The running examples (pad in ground with bands, airbridge ...) under every route they support."""
    rows = []
    for name in (
        "pad_a",
        "pad_b",
        "pad_c",
        "bridge_b",
        "pad_a_bands",
        "pad_b_bands",
        "pad_c_bands",
    ):
        rows.append(
            {"example": name, **measure(CASES[name](), workdir / name, mesh_size=60.0)}
        )
    return rows


def timing_rows(workdir: Path, *, features: int = 1000) -> list[dict]:
    """End-to-end seconds at the running examples and at one ``features``-feature random layout, by stage."""
    rows = []
    layouts = [
        (
            "pad_in_ground_B",
            lambda: build_model_from_catalogue(pad_in_ground("B"), "B"),
        ),
        (
            "pad_in_ground_B_bands",
            lambda: build_model_from_catalogue(
                pad_in_ground("B"), "B", insets=pad_requests("B")
            ),
        ),
        (
            f"random{features}_B",
            lambda: build_model_from_catalogue(
                random_layout(features, seed=1, route="B"), "B"
            ),
        ),
    ]
    for name, build in layouts:
        started = time.perf_counter()
        model = build()
        derive = time.perf_counter() - started
        timings: dict[str, float] = {}
        artifacts = emit_model(model, workdir / name, timings=timings)
        size = sum(
            p.stat().st_size for p in (artifacts.xao, artifacts.brep, artifacts.sidecar)
        )
        rows.append(
            {"layout": name, "faces": model.counts["faces"], "derive_and_validate_s": round(derive, 2),
             **{f"{k}_s": round(v, 2) for k, v in timings.items()}, "bytes_xao_brep_sidecar": size}
        )  # fmt: skip
    return rows


def print_table(rows: list[dict], keys: list[str]) -> None:
    """A Markdown table of selected keys."""
    print("| " + " | ".join(keys) + " |")
    print("|" + "---|" * len(keys))
    for row in rows:
        print("| " + " | ".join(_cell(row.get(k)) for k in keys) + " |")


def _cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return "" if value is None else str(value)


def main(argv: list[str] | None = None) -> int:
    """Print the tables; ``--out`` also writes ``catalogue.json`` there."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--timings",
        action="store_true",
        help="also time the examples and one N=1000 layout",
    )
    parser.add_argument("--no-mesh", action="store_true")
    args = parser.parse_args(argv)
    workdir = Path(tempfile.mkdtemp(prefix="sgb-catalogue-"))
    result = {"fixtures": fixture_rows(workdir, mesh_size=None if args.no_mesh else MESH_SIZE_UM),
              "examples": example_rows(workdir)}  # fmt: skip
    if args.timings:
        result["timings"] = timing_rows(workdir)
    columns = ["fixture", "route", "faces", "max_snap_drift_grid", "max_arc_deviation_grid", "narrowest_face_um",
               "load_check", "occt", "mesh", "tets", "minSICN_min", "minSICN_p1", "gamma_min", "gamma_p1", "refused"]  # fmt: skip
    print_table(result["fixtures"], columns)
    print()
    print_table(result["examples"], ["example", *columns[2:-1], "seconds"])
    if args.timings:
        print()
        print_table(result["timings"], list(result["timings"][0]))
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "catalogue.json").write_text(
            json.dumps(result, indent=1, sort_keys=True)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
