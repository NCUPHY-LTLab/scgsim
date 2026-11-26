"""Run the launcher-to-launcher Driven case of issue 187 through the public ``DrivenSim`` API and record its checks.

The case is the roster's (``scripts/roster_cases.py``): two OrcaPen SC PDK launchers joined by a 900 um CPW, the
two port sheets declared as 50 ohm ports to ground, left first; MA, MS and SA Surface EPR from the PDK's
literature-central presets; Route B.  Stages, each a subcommand, run in order on one output folder:

* ``prepare``  ``simulation_drc``, ``mesh()``, ``write_config()`` and ``prepare_handoff()``.
* ``execute``  ``bash run_palace.sh``, the script the handoff wrote.
* ``readback`` ``resolve_palace_result(...).driven_result()``, the Touchstone file written and read back, and the
  checks of section 6 of issue 187's plan, with the run facts Palace reports.

It records numbers and judges none: every tolerance of section 6 is the Human's.  The Palace executable and the
folder holding the MPI launcher Palace was built with come from ``PALACE_EXECUTABLE`` and ``PALACE_MPI_BIN``;
nothing host-specific lives in this file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import orcapen_sc_pdk

from scgsim.palace import (
    DrivenSim,
    Linear,
    PalaceDrivenResult,
    Points,
    resolve_palace_result,
)

# The roster lives in scripts/, beside this folder's parent; `python -I` does not put it on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from roster_cases import (  # noqa: E402
    ROSTER,
    geometry_plan,
    surface_epr_specs,
)

CASE = "launcher_to_launcher"
START_HZ = 2e9
STOP_HZ = 10e9
# The frequencies at which the plan's fact 11 prints arg S21, in GHz.
PHASE_FREQUENCIES_GHZ = (2.0, 4.0, 6.0, 8.0, 10.0)
OUTPUT_PREFIX = "launcher_to_launcher"


def main() -> None:
    """Parse the stage and run it on one output folder."""
    args = _parse_arguments()
    output = Path(args.output_dir).resolve()
    if args.stage == "prepare":
        prepare(output, args)
    elif args.stage == "execute":
        execute(output)
    else:
        readback(output)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("prepare", "execute", "readback"))
    parser.add_argument("output_dir")
    parser.add_argument("--order", type=int, default=2, help="FEM order")
    parser.add_argument(
        "--points", type=int, default=81, help="frequencies from 2 to 10 GHz"
    )
    parser.add_argument(
        "--adaptive-tolerance",
        type=float,
        default=None,
        help="Palace's adaptive fast sweep to this relative error; omitted: a full solve at every frequency",
    )
    parser.add_argument(
        "--full-solve-hz",
        type=float,
        action="append",
        help="a frequency the adaptive sweep solves in full first (full_solve=True); may be repeated",
    )
    parser.add_argument(
        "--bands",
        action="store_true",
        help="cut the Surface-EPR films as bands at 0.5 and 1.0 um, as the roster does",
    )
    parser.add_argument("--route", choices=("A", "B"), default="B")
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--palace", default=os.environ.get("PALACE_EXECUTABLE"))
    parser.add_argument("--mpi-bin", default=os.environ.get("PALACE_MPI_BIN"))
    return parser.parse_args()


# --------------------------------------------------------------------------- prepare


def prepare(output: Path, args: argparse.Namespace) -> None:
    """Check, mesh and configure the case, and write the handoff into ``output/run``."""
    if not args.palace or not args.mpi_bin:
        raise SystemExit(
            "--palace and --mpi-bin (or PALACE_EXECUTABLE, PALACE_MPI_BIN) are required"
        )
    output.mkdir(parents=True, exist_ok=True)
    orcapen_sc_pdk.activate()
    layout = ROSTER[CASE]()
    plan = geometry_plan(
        layout,
        orcapen_sc_pdk.get_material_records(),
        surface_epr_specs(bands=args.bands),
    )
    record: dict[str, Any] = {"settings": _settings(args)}

    started = time.perf_counter()
    report = DrivenSim.simulation_drc(plan, route=args.route)
    record["simulation_drc"] = {
        "geometry_passed": report.geometry.passed,
        "meshing_passed": report.meshing.passed,
        "lumped_elements": list(report.lumped_elements),
        "seconds": round(time.perf_counter() - started, 2),
    }

    sim = DrivenSim()
    sim.set_plan(plan.prepare())
    sim.set_output_dir(output / "run")
    sim.set_route(args.route)
    frequencies: list[Linear | Points] = [Linear(START_HZ, STOP_HZ, points=args.points)]
    if args.full_solve_hz:
        frequencies.append(Points(args.full_solve_hz, full_solve=True))
    sim.set_driven(frequencies, adaptive_tolerance=args.adaptive_tolerance)
    sim.set_numerical(order=args.order)
    started = time.perf_counter()
    sim.mesh()
    summary = sim.mesh_summary()
    record["mesh"] = {
        "seconds": round(time.perf_counter() - started, 2),
        "tetrahedra": summary["meshing"]["statistics"]["tetrahedral_topology"],
        "dof_estimate": summary["dof_estimate"],
    }
    config = json.loads(sim.write_config().read_text(encoding="utf-8"))
    record["config"] = {
        "lumped_ports": config["Boundaries"]["LumpedPort"],
        "driven": config["Solver"]["Driven"],
        "dielectric_rows": len(
            config["Boundaries"].get("Postprocessing", {}).get("Dielectric", [])
        ),
    }
    handoff = sim.prepare_handoff(
        profile="direct-local",
        executable=args.palace,
        resources={
            "command_style": "wrapper",
            "processes": args.processes,
            "threads": 1,
        },
        # The Palace wrapper finds mpirun on PATH.
        setup_commands=(f'export PATH={shlex.quote(args.mpi_bin)}:"$PATH"',),
    )
    record["handoff"] = {
        "run_dir": str(handoff.run_dir),
        "handoff_id": handoff.handoff_id,
        "script": str(handoff.script_path),
    }
    _write(output / "prepare.json", record)
    print(json.dumps(record, indent=1))


def _settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "case": CASE,
        "route": args.route,
        "order": args.order,
        "start_hz": START_HZ,
        "stop_hz": STOP_HZ,
        "points": args.points,
        "adaptive_tolerance": args.adaptive_tolerance,
        "full_solve_hz": args.full_solve_hz or [],
        "bands_um": [0.5, 1.0] if args.bands else [],
        "processes": args.processes,
    }


# --------------------------------------------------------------------------- execute


def execute(output: Path) -> None:
    """Run the handoff's own script and record its exit status and wall time."""
    handoff = _read(output / "prepare.json")["handoff"]
    started = time.perf_counter()
    completed = subprocess.run(
        ["bash", Path(handoff["script"]).name], cwd=handoff["run_dir"], check=False
    )
    record = {
        "exit_status": completed.returncode,
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    _write(output / "execute.json", record)
    print(json.dumps(record, indent=1))


# --------------------------------------------------------------------------- readback


def readback(output: Path) -> None:
    """Read the run back through ``driven_result()``, write its Touchstone file, and record the checks."""
    handoff = _read(output / "prepare.json")["handoff"]
    run_dir = Path(handoff["run_dir"])
    result = resolve_palace_result(
        run_dir, expected_handoff_id=handoff["handoff_id"]
    ).driven_result()
    touchstone = result.write_touchstone(
        output / f"{OUTPUT_PREFIX}.s{len(result.ports)}p"
    )
    record = {
        "ports": [
            {
                "number": number,
                "name": port.name,
                "sheet_id": port.sheet_id,
                "impedance_ohm": port.impedance_ohm,
            }
            for number, port in enumerate(result.ports, start=1)
        ],
        "touchstone_reads_back_equal": _reads_back_equal(touchstone, result),
        "checks": _checks(result),
        "surface_epr": _surface_epr(result),
        "adaptive_sampling": [
            {
                "excited_port": sampling.excited_port,
                "samples": sampling.samples,
                "error": sampling.error,
                "sampled_frequencies_ghz": [
                    f / 1e9 for f in sampling.sampled_frequencies_hz
                ],
            }
            for sampling in result.adaptive_sampling
        ],
        "palace": _palace_facts(run_dir),
    }
    _write(output / "report.json", record)
    print(json.dumps(record, indent=1))


def _reads_back_equal(path: Path, result: PalaceDrivenResult) -> bool:
    """Whether the file, read by a reader of its own, holds the record's numbers exactly (check A1)."""
    values: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith(("!", "#")):
            values.extend(float(value) for value in line.split())
    # Touchstone 1.1 writes a two-port's row as S11, S21, S12, S22.
    order = ((0, 0), (1, 0), (0, 1), (1, 1))
    expected: list[float] = []
    for frequency, magnitude, phase in zip(
        result.frequencies_hz, result.magnitude_db, result.phase_deg, strict=True
    ):
        expected.append(frequency / 1e9)
        for o, e in order:
            expected.extend((magnitude[o][e], phase[o][e]))
    return values == expected


def _checks(result: PalaceDrivenResult) -> dict[str, Any]:
    """The two-port checks A2 to A5 of the plan's section 6, and arg S21 at fact 11's frequencies."""
    s11, s21, s12, s22 = (result.s(o, e) for o, e in ((1, 1), (2, 1), (1, 2), (2, 2)))
    frequencies_ghz = [f / 1e9 for f in result.frequencies_hz]
    s21_db = [20.0 * math.log10(abs(value)) for value in s21]
    s11_db = [20.0 * math.log10(abs(value)) for value in s11]
    phase = {
        f"{target:g}": result.phase_deg[k][1][0]
        for target in PHASE_FREQUENCIES_GHZ
        for k, frequency in enumerate(frequencies_ghz)
        if math.isclose(frequency, target, rel_tol=1e-9)
    }
    return {
        "frequencies": len(frequencies_ghz),
        "a2_min_s21_db": min(s21_db),
        "a2_max_s11_db": max(s11_db),
        "s21_db_at_ends": [s21_db[0], s21_db[-1]],
        "s11_db_at_ends": [s11_db[0], s11_db[-1]],
        "arg_s21_deg": phase,
        "arg_s11_deg_at_start": result.phase_deg[0][0][0],
        "a3_max_abs_s12_minus_s21": max(
            abs(a - b) for a, b in zip(s12, s21, strict=True)
        ),
        "a4_max_abs_one_minus_power": max(
            abs(1.0 - abs(a) ** 2 - abs(b) ** 2) for a, b in zip(s11, s21, strict=True)
        ),
        "a5_max_abs_s11_minus_s22": max(
            abs(a - b) for a, b in zip(s11, s22, strict=True)
        ),
    }


def _surface_epr(result: PalaceDrivenResult) -> dict[str, Any]:
    """Check A8: the records per excited port and frequency, their rows, and each margin row against its baseline."""
    ratios = [
        margin.participation / baseline.participation
        for record in result.surface_epr
        for margin in record.margins
        for baseline in record.surfaces
        if baseline.index == margin.baseline_index
    ]
    first = result.surface_epr[0] if result.surface_epr else None
    return {
        "records": len(result.surface_epr),
        "expected_records": len(result.ports) * len(result.frequencies_hz),
        "surface_rows": sorted({len(record.surfaces) for record in result.surface_epr}),
        "margin_rows": sorted({len(record.margins) for record in result.surface_epr}),
        "max_margin_over_baseline": max(ratios) if ratios else None,
        "first": None
        if first is None
        else {
            "excited_port": first.excited_port,
            "frequency_ghz": first.frequency_hz / 1e9,
            "participation": {row.name: row.participation for row in first.surfaces},
        },
    }


def _palace_facts(run_dir: Path) -> dict[str, Any]:
    """What Palace reports of its own run in ``palace.json``: build, size, time and memory."""
    facts = _read(run_dir / "results" / "palace" / "palace.json")
    problem = facts["Problem"]
    return {
        "git_tag": facts["GitTag"],
        "mpi_size": problem["MPISize"],
        "mesh_elements": problem["MeshElements"],
        "degrees_of_freedom": problem["DegreesOfFreedom"],
        "total_seconds": facts["ElapsedTime"]["Durations"]["Total"],
        "peak_node_memory_mb": facts["PeakNodeMemoryMegabytes"]["Max"],
    }


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, record: dict[str, Any]) -> None:
    path.write_text(json.dumps(record, indent=1, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
