"""Check that every roster layout passes ``simulation_drc`` and meshes, and write one JSON record per layout.

    python scripts/check_roster.py RECORDS_DIR [--only CASE ...]

Run it with the interpreter of an environment that has SCGSim and the OrcaPen SC PDK revision under test, as
CI's ``pdk-tests`` job builds it.  Each layout of ``roster_cases.ROSTER`` is built with Surface-EPR bands
(Route B, MA/MS/SA at margins 0.5 and 1.0 um by type) and goes through two public calls:

1. ``simulation_drc``, with the settings it fixes itself, which must return its report;
2. ``mesh()`` in the vacuum the Human set for testing (padding 0 um in x and y, 1000 um below and above; issue
   182) with the default mesh controls, which returns only a mesh at the quality threshold (minSICN 0.01)
   within the default time budget, and otherwise raises ``MeshRepairError``.

A stage runs only when the stage before it passed.  ``RECORDS_DIR/<case>.json`` holds, for each stage, ``pass``
or the error class and message, and its wall time.  ``simulation_drc`` adds its report (the findings of its
meshing stage, each with its place); ``mesh()`` adds the quality report (minSICN, tetrahedra, time budget, stop
reason and stop places) and, when it failed, the tail of each worker log.  The record is rewritten when each
stage starts and ends, so a process that dies in a stage leaves a record naming that stage ``running``.  After
every layout has run, the exit status is 1 when any of them failed a stage.  ``RECORDS_DIR`` must be empty or
absent, so its records are those of one run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, NotRequired, TypedDict

from scgsim.meshing import MeshRepairError
from scgsim.palace import SimulationDrcError

# The roster lives beside this script; `python -I` does not put the script's folder on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from roster_cases import (  # noqa: E402
    ROSTER,
    Layout,
    configured_simulation,
    prepare_plan,
    simulation_class,
    surface_epr_specs,
)

STAGES = ("build_layout", "simulation_drc", "mesh")


class StageRecord(TypedDict):
    """One stage of a record: ``result`` is ``pass``, ``error`` or, while it runs, ``running``."""

    result: str
    seconds: NotRequired[float]
    error_class: NotRequired[str]
    message: NotRequired[str]
    trace_tail: NotRequired[str]
    # simulation_drc, and mesh() when its meshing stage refuses: the SimulationDrcReport of both stages.
    drc_report: NotRequired[dict[str, Any]]
    # mesh(): the quality report of scgsim.meshing, of the delivered mesh or of the stop.
    quality_report: NotRequired[dict[str, Any]]
    # mesh() when it failed: the tail of each worker log of the run, by file name.
    worker_logs: NotRequired[dict[str, str]]


class CaseRecord(TypedDict):
    """The record of one roster layout: its stages in the order they ran, and the whole wall time."""

    case: str
    simulation: str | None
    stages: dict[str, StageRecord]
    seconds: float | None


def main(argv: list[str] | None = None) -> int:
    """Run every roster layout, write its record, and fail when any layout failed a stage."""
    arguments = _parse_arguments(argv)
    arguments.records.mkdir(parents=True, exist_ok=True)
    material_records = _activate_pdk()
    records = []
    for name in arguments.only or ROSTER:
        record = check_case(name, arguments.records / f"{name}.json", material_records)
        print(summary_line(record), flush=True)
        records.append(record)
    failed = failed_cases(records)
    print(f"roster: {len(records) - len(failed)}/{len(records)} passed", flush=True)
    return 1 if failed else 0


def _parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("records", type=Path)
    parser.add_argument("--only", nargs="+", choices=list(ROSTER), metavar="CASE")
    arguments = parser.parse_args(argv)
    if arguments.records.exists() and any(arguments.records.iterdir()):
        parser.error(f"RECORDS_DIR must be empty or absent: {arguments.records}")
    return arguments


def _activate_pdk() -> dict[str, Any]:
    """Activate the PDK, and return its material records."""
    import orcapen_sc_pdk  # noqa: PLC0415 -- the PDK is not in SCGSim's own lock; the record handling loads without it

    orcapen_sc_pdk.activate()
    return orcapen_sc_pdk.get_material_records()


def check_case(name: str, path: Path, material_records: dict[str, Any]) -> CaseRecord:
    """Build one roster layout, run ``simulation_drc`` and then ``mesh()``, and write the record at ``path``."""
    record: CaseRecord = {
        "case": name,
        "simulation": None,
        "stages": {},
        "seconds": None,
    }
    started = time.perf_counter()
    _run_stages(record, path, material_records)
    record["seconds"] = round(time.perf_counter() - started, 2)
    write_record(path, record)
    return record


def _run_stages(record: CaseRecord, path: Path, material_records) -> None:
    """The stages in order, each only when the one before it passed."""
    specs = surface_epr_specs(bands=True)
    layout = _stage(record, path, "build_layout", lambda: _build_layout(record["case"]))
    if layout is None:
        return
    record["simulation"] = layout.simulation
    report = _stage(
        record,
        path,
        "simulation_drc",
        lambda: _simulation_drc(layout, material_records, specs),
    )
    if report is None:
        return
    record["stages"]["simulation_drc"]["drc_report"] = asdict(report)
    with TemporaryDirectory(prefix="roster-mesh-") as folder:
        run = Path(folder)
        quality = _stage(
            record, path, "mesh", lambda: _mesh(layout, material_records, specs, run)
        )
        if quality is None:
            # A worker that fails after meshing (the Palace lowering) leaves its reason only in the run's logs.
            record["stages"]["mesh"]["worker_logs"] = {
                log.name: log.read_text(errors="replace")[-3000:]
                for log in sorted(run.glob("logs/*.log"))
            }
        else:
            record["stages"]["mesh"]["quality_report"] = quality


def _build_layout(name: str) -> Layout:
    import gdsfactory as gf  # noqa: PLC0415 -- imported with the PDK it draws with, which the record handling does not need

    # Every case starts from the cells a fresh process would build, so no case sees another's cache.
    gf.clear_cache()
    return ROSTER[name]()


def _simulation_drc(layout: Layout, material_records, specs):
    return simulation_class(layout).simulation_drc(
        layout.component,
        layer_stack=layout.layer_stack,
        material_records=material_records,
        instances=dict(layout.instances) or None,
        nets=layout.nets,
        surface_epr_specs=specs,
    )


def _mesh(layout: Layout, material_records, specs, run: Path) -> dict[str, Any]:
    """``mesh()`` into ``run`` of the simulation ``configured_simulation`` sets up; the delivered mesh's quality report."""
    sim = configured_simulation(layout, prepare_plan(layout, material_records), specs)
    sim.set_output_dir(run)
    sim.mesh()
    # mesh_sgb writes `<xao stem>.quality.json` beside the mesh it delivers; the run folder holds one model.
    found = sorted(run.rglob("*.quality.json"))
    if len(found) != 1:
        raise RuntimeError(
            f"expected one quality report in the run folder, found {len(found)}"
        )
    return json.loads(found[0].read_text())


def _stage(record: CaseRecord, path: Path, name: str, action: Callable[[], Any]):
    """Run one stage, recording ``pass`` or its error and its wall time; the value, or None when it failed."""
    stage: StageRecord = {"result": "running"}
    record["stages"][name] = stage
    write_record(path, record)
    started = time.perf_counter()
    try:
        value = action()
        stage["result"] = "pass"
    except Exception as error:  # noqa: BLE001 -- the outcome is the record
        value = None
        stage.update(
            result="error",
            error_class=type(error).__name__,
            message=str(error),
            trace_tail=traceback.format_exc()[-3000:],
        )
        if isinstance(error, SimulationDrcError):
            stage["drc_report"] = asdict(error.report)
        elif isinstance(error, MeshRepairError):
            stage["quality_report"] = error.report
    stage["seconds"] = round(time.perf_counter() - started, 2)
    write_record(path, record)
    return value


def write_record(path: Path, record: CaseRecord) -> None:
    """Write a record whole or not at all: to a temporary file in the same folder, then ``os.replace``."""
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(record, indent=1, default=str))
    os.replace(partial, path)


def failed_cases(records: Sequence[CaseRecord]) -> list[str]:
    """The cases that did not pass every stage: a stage that failed, is still marked running, or never ran."""
    return [
        record["case"]
        for record in records
        if any(
            record["stages"].get(stage, {}).get("result") != "pass" for stage in STAGES
        )
    ]


def summary_line(record: CaseRecord) -> str:
    """One line for the run log: each stage's result and seconds, then the mesh's quality and time budget."""
    stages = ", ".join(
        f"{name} {stage['result']} {stage.get('seconds')} s"
        for name, stage in record["stages"].items()
    )
    line = f"{record['case']}: {stages}"
    failing = next(
        (stage for stage in record["stages"].values() if stage["result"] != "pass"),
        None,
    )
    if failing is not None:
        message = (failing.get("message") or "").partition("\n")[0]
        line += f"; {failing.get('error_class')}: {message}"
    report = record["stages"].get("mesh", {}).get("quality_report")
    if report:
        line += "; " + _mesh_summary(report)
    return line


def _mesh_summary(report: dict[str, Any]) -> str:
    budget, timing = report["time_budget"], report["timing"]
    # A stop before the first mesh (the time limit reached while reading the extent) has no mesh entry.
    mesh = report.get("mesh") or {}
    return (
        f"{mesh.get('tetrahedra')} tetrahedra, minSICN {mesh.get('min_quality')}, stop {report['stop']}, "
        f"{timing['total_seconds']:.1f} s of meshing (limit {budget['limit_seconds']:.0f} s, "
        f"target {budget['target_seconds']:.0f} s)"
    )


if __name__ == "__main__":
    raise SystemExit(main())
