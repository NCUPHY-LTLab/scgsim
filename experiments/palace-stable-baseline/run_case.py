# ruff: noqa: PLC0415 -- the PDK and SCGSim are imported where used, after the PDK is activated
"""Run one layout through plan, SGB build, simulation_drc and mesh(), and write its record as JSON.

    python run_case.py CASE OUT.json [--no-bands] [--notebook-vacuum]
    python run_case.py --scene SCENE OUT.json        # a supported test scene: SGB build and fine structure

Each stage records ``pass`` or the exact error class and message, and its wall time.  The SGB build stage
also records the fine structure of the built model (``fine_structure.summarize``).  The ``simulation_drc``
stage records the findings of its meshing stage: the warnings of a pass, or the refusals of the
``SimulationDrcError`` it raises.  ``mesh()`` runs the same meshing stage before the mesher starts, so the
mesh stage records either that refusal or the recipe, tetrahedra, minSICN, rounds, stop reason and stop
places of the quality report.  A stage runs only when the stage before it passed.  ``simulation_drc`` and
``mesh()`` are public; the SGB build stage calls the geometry build those two share (``_build_geometry``),
because no public call keeps the built files.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fine_structure  # noqa: E402

MARGINS_UM = [0.5, 1.0]
PORT_INDUCTANCE_H = 10e-9
VACUUM_PADDING_UM = (
    1.0  # what simulation_drc uses when the layout has no vacuum of its own
)
OUT: list[Path] = []


def surface_epr_specs(bands: bool) -> dict[str, dict[str, Any]]:
    """MA, MS and SA from the PDK's literature-central presets; with bands, margins (0.5, 1.0) um by type."""
    from orcapen_sc_pdk import (
        get_interface_preset_records,
        validate_interface_preset_records,
    )

    records = validate_interface_preset_records(get_interface_preset_records())
    specs = {}
    for kind in ("MA", "MS", "SA"):
        record = records[f"LiteratureCentral_{kind}"]
        specs[kind] = {
            key: record[key]
            for key in ("thickness", "permittivity", "loss_tangent", "source")
        }
        if bands:
            specs[kind]["inset_margins_um"] = list(MARGINS_UM)
    return specs


def run_layout(name: str, bands: bool, notebook_vacuum: bool = False) -> dict[str, Any]:
    """The four stages of one layout."""
    import gdsfactory as gf
    import orcapen_sc_pdk
    from cases import LAYOUTS
    from orcapen_sc_pdk import get_material_records

    orcapen_sc_pdk.activate()
    gf.clear_cache()
    record: dict[str, Any] = {
        "case": name,
        "bands": bands,
        "notebook_vacuum": notebook_vacuum,
        "stages": {},
    }
    _CHECKPOINT[:] = [OUT[0], record]
    stages = record["stages"]
    specs = surface_epr_specs(bands)

    layout = _stage(stages, "build_layout", LAYOUTS[name])
    if layout is None:
        return record
    record["simulation"] = layout.simulation
    snapshot = _stage(
        stages,
        "plan",
        lambda: prepare_plan(layout, get_material_records(), notebook_vacuum),
    )
    if snapshot is None:
        return record
    with TemporaryDirectory(prefix="baseline-sgb-") as folder:
        with fine_structure.capture() as captured:
            built = _stage(
                stages, "sgb_build", lambda: _sgb_build(layout, snapshot, specs, folder)
            )
        if built is not None:
            stages["sgb_build"]["sidecar_slivers"] = json.loads(
                built.artifacts.sidecar.read_text()
            )["slivers"]
        if captured:
            stages["sgb_build"]["fine_structure"] = fine_structure.summarize(
                captured[-1]
            )
    if built is None:
        return record
    drc_class = _simulation_class(layout)
    drc_report = _stage(
        stages,
        "simulation_drc",
        lambda: drc_class.simulation_drc(
            layout.component,
            layer_stack=layout.layer_stack,
            material_records=get_material_records(),
            instances=dict(layout.instances) or None,
            nets=layout.nets,
            surface_epr_specs=specs,
        ),
    )
    if drc_report is None:
        return record
    stages["simulation_drc"]["meshing_stage"] = _meshing_stage_findings(drc_report)
    with TemporaryDirectory(prefix="baseline-mesh-") as folder:
        sim = configured_simulation(layout, snapshot, specs)
        sim.set_output_dir(folder)
        _stage(stages, "mesh", sim.mesh)
        stages["mesh"].update(_mesh_outcome(stages["mesh"], Path(folder)))
        if stages["mesh"]["result"] != "pass":
            # A worker that fails after meshing (the Palace lowering) leaves its reason only in the run's logs.
            stages["mesh"]["worker_logs"] = {
                log.name: log.read_text(errors="replace")[-3000:]
                for log in sorted(Path(folder).glob("logs/*.log"))
            }
            _checkpoint()
    return record


def run_scene(name: str) -> dict[str, Any]:
    """SGB build of a supported test scene with MA, MS and SA bands at (0.5, 1.0) um by type."""
    sys.path.insert(0, str(HERE.parents[1] / "tests"))
    from cases import scene_builders
    from sgb_supported_kit import catalogue, type_bands

    from scgsim.geometry.sgb.model import build_model_from_catalogue

    record: dict[str, Any] = {"case": name, "bands": True, "scene": True, "stages": {}}
    _CHECKPOINT[:] = [OUT[0], record]
    model = _stage(
        record["stages"],
        "sgb_build",
        lambda: build_model_from_catalogue(
            catalogue(scene_builders()[name]()), "B", insets=type_bands()
        ),
    )
    if model is not None:
        captured = fine_structure.capture_model(model)
        record["stages"]["sgb_build"]["sidecar_slivers"] = captured["report"]["summary"]
        record["stages"]["sgb_build"]["fine_structure"] = fine_structure.summarize(
            captured
        )
    return record


def _sgb_build(layout, snapshot, specs, folder):
    # The geometry build mesh() and simulation_drc share; no public call keeps its files.
    return configured_simulation(layout, snapshot, specs)._build_geometry(folder)  # noqa: SLF001


def prepare_plan(layout, material_records, notebook_vacuum: bool = False):
    """The prepared plan as ``simulation_drc`` builds it (root ``chip``, no coupon padding), or with the notebook vacuum."""
    from scgsim.geometry import GeometryPlan

    plan = GeometryPlan(
        layout.component,
        root_id="chip",
        layer_stack=layout.layer_stack,
        material_records=material_records,
        coupon_padding_um=0.0,
    )
    for path, reference in layout.instances.items():
        plan.register_instance(path, reference)
    if layout.nets is not None:
        plan.set_nets(layout.nets)
    if notebook_vacuum:
        # The vacuum the PDK's Palace notebooks give the plan, instead of simulation_drc's 1 um shell.
        plan.set_vacuum_region(padding=layout.notebook_vacuum_padding_um)
    return plan.prepare()


def _simulation_class(layout):
    from scgsim.palace import EigenmodeSim, ElectrostaticSim

    return EigenmodeSim if layout.simulation == "eigenmode" else ElectrostaticSim


def configured_simulation(layout, snapshot, specs):
    """The simulation as simulation_drc sets it up: 1 um vacuum, every authored port, Route B Surface EPR."""
    sim = _simulation_class(layout)()
    sim.set_plan(snapshot)
    if "VACUUM_REGION" not in snapshot.stack["solution_regions"]:
        sim.set_vacuum_region(VACUUM_PADDING_UM)
    if layout.simulation == "eigenmode":
        for port in snapshot.stack["metadata"].get("port_sheet_source_layers", ()):
            sim.add_port(
                port["name"],
                layer=port["target_layer"],
                layout_sheet=True,
                inductance=PORT_INDUCTANCE_H,
            )
    sim.set_surface_epr(representation="B", specs=specs)
    return sim


_CHECKPOINT: list[
    Any
] = []  # (path, record): the record is rewritten around every stage


def _checkpoint() -> None:
    if _CHECKPOINT:
        path, record = _CHECKPOINT
        path.write_text(json.dumps(record, indent=1, default=_json_default))


def _stage(stages: dict[str, Any], name: str, action):
    stages[name] = {"result": "running"}
    _checkpoint()
    started = time.perf_counter()
    try:
        value = action()
        stages[name] = {"result": "pass"}
    except Exception as error:  # noqa: BLE001 -- the outcome is the record
        value = None
        stages[name] = {
            "result": "error",
            "error_class": type(error).__name__,
            "message": str(error),
            "trace_tail": traceback.format_exc()[-3000:],
        }
        report = getattr(error, "report", None)
        if isinstance(report, dict):
            # MeshRepairError: the quality report of the run that stopped.
            stages[name]["quality_report"] = report
        elif report is not None:
            # SimulationDrcError, from simulation_drc or from mesh() before the mesher starts.
            stages[name]["meshing_stage"] = _meshing_stage_findings(report)
    stages[name]["seconds"] = round(time.perf_counter() - started, 2)
    _checkpoint()
    return value


def _meshing_stage_findings(report) -> dict[str, Any]:
    """The meshing-stage findings of a ``SimulationDrcReport``: refusals and warnings, each with its place."""
    findings = report.meshing.findings
    return {
        "passed": report.meshing.passed,
        "refusing": sum(1 for finding in findings if finding.action == "refuse"),
        "warning": sum(1 for finding in findings if finding.action == "warn"),
        "findings": [
            {
                "rule": finding.rule,
                "measure": finding.measure,
                "action": finding.action,
                "owner": finding.owner,
                "value": finding.value,
                "value_steps": finding.value_steps,
                "location_um": finding.location_um,
                "entity": finding.entity,
                "sources": finding.sources[:4],
            }
            for finding in findings
        ],
    }


def _mesh_outcome(stage: dict[str, Any], folder: Path) -> dict[str, Any]:
    report = stage.pop("quality_report", None)
    if report is None:
        found = sorted(folder.rglob("*.quality.json"))
        report = json.loads(found[0].read_text()) if found else None
    if report is None:
        return {}
    mesh = report.get("mesh") or {}
    return {
        "recipe": report.get("recipe"),
        "tetrahedra": mesh.get("tetrahedra"),
        "min_sicn": mesh.get("min_quality"),
        "bad_count": mesh.get("bad_count"),
        "rounds": report.get("rounds"),
        "stop": report.get("stop"),
        "time_budget": report.get("time_budget"),
        "mesh_seconds": (report.get("timing") or {}).get("total_seconds"),
        "history": report.get("history"),
        "remaining_bad": report.get("remaining_bad"),
        "stop_places": report.get("stop_places"),
    }


def main() -> None:
    """Run one case from the command line."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("case")
    parser.add_argument("out", type=Path)
    parser.add_argument("--no-bands", action="store_true")
    parser.add_argument("--notebook-vacuum", action="store_true")
    parser.add_argument("--scene", action="store_true")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    OUT[:] = [args.out]
    started = time.perf_counter()
    record = (
        run_scene(args.case)
        if args.scene
        else run_layout(args.case, not args.no_bands, args.notebook_vacuum)
    )
    record["seconds"] = round(time.perf_counter() - started, 2)
    args.out.write_text(json.dumps(record, indent=1, default=_json_default))


def _json_default(value):
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, Path):
        return value.name
    return str(value)


if __name__ == "__main__":
    main()
