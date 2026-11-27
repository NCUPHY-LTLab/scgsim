"""Run the Driven sampling study of issue 307 on the hanger resonator through the public SCGSim API.

Every run is built from ``hanger_layout.hanger_plan()`` on Route B at FEM order 2 with no mesh refinement, and is
one entry of ``RUNS``: an Eigenmode reference, then Driven sweeps that differ in one setting each.  The Driven
frequencies around the resonance come from the Eigenmode run's f0 and Q, so ``eigenmode`` runs first.  Stages,
each a subcommand, run in order on one study folder:

* ``prepare STUDY RUN``  ``simulation_drc``, ``mesh()``, ``write_config()`` and ``prepare_handoff()`` into
  ``STUDY/RUN/run``; the ``add-to-prom`` run adds one ``AddToPROM`` point to the written ``config.json`` before
  the handoff seals it, the only edit of a file SCGSim wrote.
* ``execute STUDY RUN``  ``bash run_palace.sh``, the script the handoff wrote.
* ``readback STUDY RUN``  ``resolve_palace_result(...)``: ``driven_result()`` or, for the Eigenmode run, the
  modes, the ports' Q and where each mode's electric energy is; with the cost and the timing Palace reports.
  A refusal is recorded as it is raised.
* ``fit STUDY``  the S21 notch of every read-back Driven run fitted with the hanger model of Probst et al.
  (2015) (``notch_fit``), compared with the Eigenmode mode, the full solves and ``adaptive``; writes
  ``STUDY/study.json`` and prints the two tables.

It records numbers and judges none: every agreement tolerance is the Human's.  The Palace executable and the
folder holding the MPI launcher Palace was built with come from ``PALACE_EXECUTABLE`` and ``PALACE_MPI_BIN``;
nothing host-specific lives in this file.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from scgsim.palace import (
    DrivenSim,
    EigenmodeSim,
    Linear,
    Points,
    resolve_palace_result,
)

# `python -I` does not put this folder on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hanger_layout import hanger_plan  # noqa: E402
from notch_fit import band_delay, fit_notch  # noqa: E402
from palace_log import log_timing  # noqa: E402

BAND_HZ = (4e9, 9e9)
BAND_POINTS = 501  # 10 MHz apart
# The dense window: +-20 linewidths (f0 / Q) of the Eigenmode mode, 50 points per linewidth.
WINDOW_LINEWIDTHS = 20.0
WINDOW_POINTS = 2001
# The full solves: 9 frequencies of the window's grid, 0.75 linewidth apart, across +-3 linewidths.
FULL_SOLVE_STEP_LINEWIDTHS = 0.75
FULL_SOLVE_COUNT = 9
EIGEN_MODES = 4
EIGEN_TARGET_HZ = 6.5e9  # below the 7.38 GHz the resonator's length estimates
PROCESSES = 8
# Band points this many linewidths from f0 or more carry no resonance: their phase gives the line's delay.
DELAY_EXCLUSION_LINEWIDTHS = 100.0
# Only the meander is above this y (um): the feedline, the launchers and the coupling straight are below it.
MEANDER_Y_UM = 200.0
# The runs every run's S21 is compared with, where both hold a frequency.
REFERENCES = ("full-solves", "full-solves-linear-1e-9", "adaptive")
# Palace prints a frequency to 9 significant digits, 10 Hz between 1 and 10 GHz: two read-back frequencies
# this close are the same one.
SAME_FREQUENCY_HZ = 10.0


@dataclass(frozen=True)
class Run:
    """One run of the study: its problem, its frequencies and the setting it varies."""

    problem: Literal["Eigenmode", "Driven"]
    # "band+window": the 4-9 GHz band and the dense window; "full-solves": the 9 points; "uniform": one Linear.
    sweep: Literal["band+window", "full-solves", "uniform"] | None = None
    uniform_points: int | None = None
    adaptive_tolerance: float | None = None
    adaptive_max_samples: int | None = None
    surface_epr: bool = True
    add_to_prom_at_f0: bool = False
    linear_tolerance: float | None = None  # Solver.Linear.Tol; None keeps SCGSim's 1e-6
    eigen_tolerance: float | None = (
        None  # Solver.Eigenmode.Tol; None keeps SCGSim's 1e-6
    )


RUNS: dict[str, Run] = {
    "eigenmode": Run("Eigenmode"),
    # The Eigenmode Q is 20 % above the notch's Q_l: the same with a tighter linear solve, then also a tighter
    # eigensolve.
    "eigenmode-linear-1e-9": Run("Eigenmode", linear_tolerance=1e-9),
    "eigenmode-linear-1e-9-eigen-1e-9": Run(
        "Eigenmode", linear_tolerance=1e-9, eigen_tolerance=1e-9
    ),
    "adaptive": Run("Driven", "band+window", adaptive_tolerance=1e-3),
    "add-to-prom": Run(
        "Driven", "band+window", adaptive_tolerance=1e-3, add_to_prom_at_f0=True
    ),
    "tolerance-1e-5": Run("Driven", "band+window", adaptive_tolerance=1e-5),
    # The 1e-5 run reached Palace's default cap of 20 samples on port 1, so the same with a cap of 40.
    "tolerance-1e-5-max-40": Run(
        "Driven", "band+window", adaptive_tolerance=1e-5, adaptive_max_samples=40
    ),
    "no-surface-epr": Run(
        "Driven", "band+window", adaptive_tolerance=1e-3, surface_epr=False
    ),
    "full-solves": Run("Driven", "full-solves"),
    # The adaptive runs differ from the full solves near f0: the same full solves with a tighter linear solve.
    "full-solves-linear-1e-9": Run("Driven", "full-solves", linear_tolerance=1e-9),
    "uniform-25000": Run(
        "Driven", "uniform", uniform_points=25000, adaptive_tolerance=1e-3
    ),
    "uniform-2500": Run(
        "Driven", "uniform", uniform_points=2500, adaptive_tolerance=1e-3
    ),
}


def main() -> None:
    """Parse the stage and run it."""
    args = _parse_arguments()
    study = Path(args.study_dir).resolve()
    if args.stage == "fit":
        fit(study)
        return
    if args.run not in RUNS:
        raise SystemExit(f"RUN must be one of {', '.join(RUNS)}")
    if args.stage == "prepare":
        prepare(study, args.run, args.palace, args.mpi_bin)
    elif args.stage == "execute":
        execute(study / args.run)
    else:
        readback(study / args.run)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("prepare", "execute", "readback", "fit"))
    parser.add_argument("study_dir")
    parser.add_argument("run", nargs="?", help=f"one of {', '.join(RUNS)}")
    parser.add_argument("--palace", default=os.environ.get("PALACE_EXECUTABLE"))
    parser.add_argument("--mpi-bin", default=os.environ.get("PALACE_MPI_BIN"))
    return parser.parse_args()


# --------------------------------------------------------------------------- prepare


def prepare(study: Path, name: str, palace: str | None, mpi_bin: str | None) -> None:
    """Check, mesh and configure one run, and write its handoff into ``STUDY/RUN/run``."""
    if not palace or not mpi_bin:
        raise SystemExit(
            "--palace and --mpi-bin (or PALACE_EXECUTABLE, PALACE_MPI_BIN) are required"
        )
    run = RUNS[name]
    folder = study / name
    folder.mkdir(parents=True, exist_ok=True)
    plan = hanger_plan(surface_epr=run.surface_epr)
    simulation_class = EigenmodeSim if run.problem == "Eigenmode" else DrivenSim
    record: dict[str, Any] = {"run": name, "settings": asdict(run)}

    started = time.perf_counter()
    report = simulation_class.simulation_drc(plan)
    record["simulation_drc"] = {
        "geometry_passed": report.geometry.passed,
        "meshing_passed": report.meshing.passed,
        "lumped_elements": list(report.lumped_elements),
        "seconds": round(time.perf_counter() - started, 2),
    }
    sim = simulation_class()
    sim.set_plan(plan.prepare())
    sim.set_output_dir(folder / "run")
    if isinstance(sim, EigenmodeSim):
        # Every mode is saved as a ParaView field, so readback can tell where its energy is.
        sim.set_eigenmode(
            num_modes=EIGEN_MODES,
            target=EIGEN_TARGET_HZ,
            tolerance=1e-6 if run.eigen_tolerance is None else run.eigen_tolerance,
            save=EIGEN_MODES,
        )
        sim.set_numerical(
            order=2,
            amr_max_passes=0,
            output_paraview=True,
            tolerance=1e-6 if run.linear_tolerance is None else run.linear_tolerance,
        )
    else:
        sim.set_driven(
            _frequencies(run, study),
            adaptive_tolerance=run.adaptive_tolerance,
            adaptive_max_samples=run.adaptive_max_samples,
        )
        if run.linear_tolerance is not None:
            sim.set_numerical(tolerance=run.linear_tolerance)
    started = time.perf_counter()
    sim.mesh()
    summary = sim.mesh_summary()
    record["mesh"] = {
        "seconds": round(time.perf_counter() - started, 2),
        "tetrahedra": summary["meshing"]["statistics"]["tetrahedral_topology"],
        "dof_estimate": summary["dof_estimate"],
    }
    config_path = sim.write_config()
    if run.add_to_prom_at_f0:
        record["config_edit"] = _add_to_prom(config_path, _reference(study)[0])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    record["config"] = {
        "solver": config["Solver"],
        "lumped_ports": config["Boundaries"]["LumpedPort"],
        "dielectric_rows": len(
            config["Boundaries"].get("Postprocessing", {}).get("Dielectric", [])
        ),
    }
    record["handoff"] = _handoff(sim, palace, mpi_bin)
    _write(folder / "prepare.json", record)
    print(json.dumps(record, indent=1, default=str))


def _frequencies(run: Run, study: Path) -> list[Linear | Points]:
    """The ranges of a Driven run, the window and the full solves placed from the Eigenmode run's f0 and Q."""
    if run.sweep == "uniform":
        assert run.uniform_points is not None
        return [Linear(*BAND_HZ, points=run.uniform_points)]
    f0_hz, linewidth_hz = _reference(study)
    low, high = _window_hz(f0_hz, linewidth_hz)
    if run.sweep == "band+window":
        return [
            Linear(*BAND_HZ, points=BAND_POINTS),
            Linear(low, high, points=WINDOW_POINTS),
        ]
    return [Points(_full_solve_frequencies_hz(low, high))]


def _reference(study: Path) -> tuple[float, float]:
    """The Eigenmode resonator mode's f0 and linewidth f0 / Q, in Hz."""
    mode = _read(study / "eigenmode" / "report.json")["resonator_mode"]
    return mode["frequency_hz"], mode["frequency_hz"] / mode["q"]


def _window_hz(f0_hz: float, linewidth_hz: float) -> tuple[float, float]:
    return (
        f0_hz - WINDOW_LINEWIDTHS * linewidth_hz,
        f0_hz + WINDOW_LINEWIDTHS * linewidth_hz,
    )


def _full_solve_frequencies_hz(low_hz: float, high_hz: float) -> list[float]:
    """Nine points of the window's own grid, as Palace computes it, ``FULL_SOLVE_STEP_LINEWIDTHS`` apart.

    Being points of the window, the adaptive runs hold the same frequencies, so they compare without interpolation.
    """
    steps_per_linewidth = (WINDOW_POINTS - 1) / (2.0 * WINDOW_LINEWIDTHS)
    centre = (WINDOW_POINTS - 1) // 2
    low_ghz, high_ghz = low_hz / 1e9, high_hz / 1e9
    frequencies = []
    for m in range(-(FULL_SOLVE_COUNT // 2), FULL_SOLVE_COUNT // 2 + 1):
        offset = math.copysign(
            math.floor(abs(m) * FULL_SOLVE_STEP_LINEWIDTHS * steps_per_linewidth + 0.5),
            m,
        )
        i = centre + int(offset)
        frequencies.append(
            (low_ghz + (i / (WINDOW_POINTS - 1)) * (high_ghz - low_ghz)) * 1e9
        )
    return frequencies


def _add_to_prom(config_path: Path, f0_hz: float) -> dict[str, Any]:
    """Add a ``Point`` at f0 with ``AddToPROM`` to ``Solver.Driven.Samples``: a full solve SCGSim does not expose."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    entry = {"Type": "Point", "Freq": [f0_hz / 1e9], "AddToPROM": True}
    config["Solver"]["Driven"]["Samples"].append(entry)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return {
        "file": config_path.name,
        "appended_to": "Solver.Driven.Samples",
        "entry": entry,
    }


def _handoff(
    sim: DrivenSim | EigenmodeSim, palace: str, mpi_bin: str
) -> dict[str, Any]:
    handoff = sim.prepare_handoff(
        profile="direct-local",
        executable=palace,
        resources={"command_style": "wrapper", "processes": PROCESSES, "threads": 1},
        # The Palace wrapper finds mpirun on PATH.
        setup_commands=(f'export PATH={shlex.quote(mpi_bin)}:"$PATH"',),
    )
    return {
        "run_dir": str(handoff.run_dir),
        "handoff_id": handoff.handoff_id,
        "script": str(handoff.script_path),
    }


# --------------------------------------------------------------------------- execute


def execute(folder: Path) -> None:
    """Run the handoff's own script and record its exit status and wall time."""
    handoff = _read(folder / "prepare.json")["handoff"]
    started = time.perf_counter()
    completed = subprocess.run(
        ["bash", Path(handoff["script"]).name], cwd=handoff["run_dir"], check=False
    )
    record = {
        "exit_status": completed.returncode,
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    _write(folder / "execute.json", record)
    print(json.dumps(record, indent=1))


# --------------------------------------------------------------------------- readback


def readback(folder: Path) -> None:
    """Resolve the returned run and record its result, its cost and the timing in Palace's log.

    A refusal of ``resolve_palace_result`` is recorded as it is raised.  A Driven run is then not read further;
    an Eigenmode run's modes are read from Palace's own files, as ``_eigenmode_readback`` says.
    """
    prepared = _read(folder / "prepare.json")
    run_dir = Path(prepared["handoff"]["run_dir"])
    receipt = _read(run_dir / "metadata" / "palace_returned_run_receipt.json")
    record: dict[str, Any] = {
        "run": prepared["run"],
        "tetrahedra": prepared["mesh"]["tetrahedra"]["tetrahedra"],
        "palace": _palace_facts(run_dir),
        "log": log_timing(
            (run_dir / receipt["log"]["path"]).read_text(encoding="utf-8")
        ),
    }
    try:
        resolved = resolve_palace_result(
            run_dir, expected_handoff_id=prepared["handoff"]["handoff_id"]
        )
        record["resolve"] = {"status": "resolved"}
    except ValueError as error:
        resolved = None
        record["resolve"] = {"refused": type(error).__name__, "message": str(error)}
    if prepared["settings"]["problem"] == "Eigenmode":
        record.update(_eigenmode_readback(resolved, run_dir))
    elif resolved is not None:
        record.update(_driven_readback(resolved, folder))
    _write(folder / "report.json", record)
    print(json.dumps(record, indent=1, default=str))


def _driven_readback(resolved: Any, folder: Path) -> dict[str, Any]:
    """``driven_result()``: its frequencies and sampling in ``report.json``, S21 and S11 in ``s21.json``.

    A refusal is recorded as it is raised, with nothing read in its place.
    """
    try:
        result = resolved.driven_result()
    except (ValueError, TypeError) as error:
        return {
            "driven_result": {"refused": type(error).__name__, "message": str(error)}
        }
    result.write_touchstone(folder / "hanger.s2p")
    s21, s11 = result.s(2, 1), result.s(1, 1)
    _write(
        folder / "s21.json",
        {
            "frequencies_hz": list(result.frequencies_hz),
            "s21": [[value.real, value.imag] for value in s21],
            "s11": [[value.real, value.imag] for value in s11],
        },
    )
    return {
        "driven_result": {
            "frequencies": len(result.frequencies_hz),
            "first_last_hz": [result.frequencies_hz[0], result.frequencies_hz[-1]],
            "sweep": result.sweep,
            "surface_epr_records": len(result.surface_epr),
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
        }
    }


def _eigenmode_readback(resolved: Any, run_dir: Path) -> dict[str, Any]:
    """The modes, the ports' Q per mode and where each mode's electric energy is; the resonator mode among them.

    The modes are read from Palace's ``eig.csv`` and ``port-Q.csv``, the same values with or without a resolved
    run, so a run ``resolve_palace_result`` refuses still gives the reference.  The resonator mode is the one with
    the largest share of its electric energy above ``MEANDER_Y_UM``, where only the meander is: read from the
    field, not from its frequency.
    """
    if resolved is None:
        epr = {"not_read": "resolve_palace_result refused the run"}
    else:
        try:
            resolved.epr_result()
            epr = {"status": "read"}
        except (ValueError, TypeError) as error:
            epr = {"refused": type(error).__name__, "message": str(error)}
    results = run_dir / "results" / "palace"
    modes = _csv_rows(results / "eig.csv")
    localization = _energy_localization(results / "paraview", len(modes))
    resonator = max(localization, key=lambda mode: mode["meander_share"])
    row = modes[resonator["mode"] - 1]
    frequency_hz = row["Re{f} (GHz)"] * 1e9
    return {
        "epr_result": epr,
        "modes": modes,
        "port_q": _csv_rows(results / "port-Q.csv")
        if (results / "port-Q.csv").is_file()
        else None,
        "energy_localization": localization,
        "resonator_mode": {
            "mode": resonator["mode"],
            "frequency_hz": frequency_hz,
            "imag_hz": row["Im{f} (GHz)"] * 1e9,
            "q": frequency_hz / (2.0 * row["Im{f} (GHz)"] * 1e9),
            "q_in_eig_csv": row.get("Q"),
            "meander_share": resonator["meander_share"],
        },
    }


def _energy_localization(paraview: Path, modes: int) -> list[dict[str, Any]]:
    """Each mode's share of electric energy above ``MEANDER_Y_UM`` and above the feedline's ground strip (16 um).

    Read from Palace's ParaView output of the electric energy density ``U_e``: mode m is cycle m, each cycle one
    file per rank of second-order Lagrange tetrahedra (4 corners, then 6 edge midpoints), integrated cell by
    cell with the exact rule for a quadratic, ``V (-sum(corners) / 20 + sum(midpoints) / 5)``.
    """
    # pyvista is needed only here, for the Eigenmode fields.
    import pyvista as pv  # noqa: PLC0415

    shares = []
    for mode in range(1, modes + 1):
        inside_meander = above_strip = total = 0.0
        for part in sorted(
            (paraview / "eigenmode" / f"Cycle{mode:06d}").glob("proc*.vtu")
        ):
            grid = pv.read(part)
            if not np.all(grid.celltypes == pv.CellType.LAGRANGE_TETRAHEDRON):
                raise ValueError(
                    f"{part} holds cells other than second-order tetrahedra"
                )
            cells = np.asarray(grid.cell_connectivity).reshape(-1, 10)
            points = np.asarray(grid.points)[cells]
            density = np.asarray(grid.point_data["U_e"])[cells]
            edges = points[:, 1:4] - points[:, :1]
            volume = np.abs(np.linalg.det(edges)) / 6.0
            energy = volume * (
                -density[:, :4].sum(1) / 20.0 + density[:, 4:].sum(1) / 5.0
            )
            y = points[:, :4, 1].mean(1)
            total += float(energy.sum())
            inside_meander += float(energy[y > MEANDER_Y_UM].sum())
            above_strip += float(energy[y > 16.0].sum())
        shares.append(
            {
                "mode": mode,
                "meander_share": inside_meander / total,
                "above_ground_strip_share": above_strip / total,
            }
        )
    return shares


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
        "durations_seconds": facts["ElapsedTime"]["Durations"],
        "peak_node_memory_mb": facts["PeakNodeMemoryMegabytes"]["Max"],
    }


def _csv_rows(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    headers = [header.strip() for header in rows[0]]
    return [
        {header: float(value) for header, value in zip(headers, row, strict=True)}
        for row in rows[1:]
    ]


# --------------------------------------------------------------------------- fit


def fit(study: Path) -> None:
    """Fit the notch of every read-back Driven run, compare it with the references, and print the two tables."""
    eigen = _read(study / "eigenmode" / "report.json")
    f0_hz, linewidth_hz = _reference(study)
    delay = {}
    runs: dict[str, dict[str, Any]] = {}
    for name in RUNS:
        read = _run_s21(study / name) if RUNS[name].problem == "Driven" else None
        if read is None:
            continue
        frequencies, s21 = read
        near = np.abs(frequencies - f0_hz) <= DELAY_EXCLUSION_LINEWIDTHS * linewidth_hz
        delay[name] = band_delay(frequencies, s21, near)
        runs[name] = {"frequencies": frequencies, "s21": s21}
    reference_delay = delay.get("adaptive")
    fits = {}
    for name, data in runs.items():
        tau = delay[name] if delay[name] is not None else reference_delay
        fits[name] = _fit_run(
            data["frequencies"], data["s21"], f0_hz, linewidth_hz, tau
        )
        fits[name]["delay_from"] = name if delay[name] is not None else "adaptive"
        fits[name]["s21_from"] = (
            "driven_result()"
            if (study / name / "s21.json").is_file()
            else "port-S.csv; driven_result() refused the run"
        )
    record = {
        "eigenmode": {**eigen["resonator_mode"], **_external_q(eigen)},
        "fits": fits,
        "against": {
            reference: _against(runs, reference, keep_rows=reference != "adaptive")
            for reference in REFERENCES
        },
    }
    _write(study / "study.json", record)
    print(json.dumps(record, indent=1, default=str))
    print(
        _runs_table(study), _eigenmode_table(study), _results_table(record), sep="\n\n"
    )


def _run_s21(folder: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """A Driven run's frequencies (Hz) and S21: as ``driven_result()`` gave them, or else from Palace's own table.

    A run ``driven_result()`` refused, an adaptive sweep that reached its cap, is read from ``port-S.csv`` so its
    S21 can be compared; ``fit`` labels it.  None for a run not yet returned.
    """
    if (folder / "s21.json").is_file():
        data = _read(folder / "s21.json")
        return np.asarray(data["frequencies_hz"]), np.asarray(
            [complex(*pair) for pair in data["s21"]]
        )
    table = folder / "run" / "results" / "palace" / "port-S.csv"
    if not (folder / "report.json").is_file() or not table.is_file():
        return None
    rows = _csv_rows(table)
    magnitude = np.asarray([row["|S[2][1]| (dB)"] for row in rows])
    phase = np.radians([row["arg(S[2][1]) (deg.)"] for row in rows])
    return np.asarray([row["f (GHz)"] * 1e9 for row in rows]), 10.0 ** (
        magnitude / 20.0
    ) * np.exp(1j * phase)


def _external_q(eigen: dict[str, Any]) -> dict[str, Any]:
    """The resonator mode's ``Q_ext`` of each port from ``port-Q.csv``, and the two together, 1 / sum(1 / Q_ext)."""
    row = eigen["port_q"][eigen["resonator_mode"]["mode"] - 1]
    q_external = [row["Q_ext[1]"], row["Q_ext[2]"]]
    return {
        "q_external": q_external,
        "q_external_together": 1.0 / sum(1.0 / q for q in q_external),
    }


def _fit_run(
    frequencies: np.ndarray,
    s21: np.ndarray,
    f0_hz: float,
    linewidth_hz: float,
    tau: float | None,
) -> dict[str, Any]:
    """The hanger fit on the window's points, the delay fixed at ``tau``; with no delay known, none is reported."""
    if tau is None:
        return {"fitted": False, "reason": "no band to take the delay from"}
    low, high = _window_hz(f0_hz, linewidth_hz)
    inside = (frequencies >= low - SAME_FREQUENCY_HZ) & (
        frequencies <= high + SAME_FREQUENCY_HZ
    )
    f, z = (
        frequencies[inside],
        s21[inside] * np.exp(2j * math.pi * frequencies[inside] * tau),
    )
    fitted = fit_notch(f, z, f0_hz, f0_hz / linewidth_hz)
    fitted.update(
        {
            "delay_s": tau,
            "points": int(inside.sum()),
            "min_s21_db": float(20 * np.log10(np.abs(z).min())),
        }
    )
    return fitted


def _against(
    runs: dict[str, dict[str, Any]], reference_name: str, *, keep_rows: bool
) -> dict[str, Any]:
    """Each run's S21 against ``reference_name``'s at the frequencies both hold (within 10 Hz): the largest
    ``abs(S21 - S21_reference)`` over all of them and over the dense window's, and with ``keep_rows`` each one.
    """
    reference = runs.get(reference_name)
    if reference is None:
        return {}
    compared = {}
    for name, data in runs.items():
        if name == reference_name:
            continue
        k = np.searchsorted(data["frequencies"], reference["frequencies"])
        k = np.clip(k, 1, len(data["frequencies"]) - 1)
        nearer = np.where(
            np.abs(data["frequencies"][k - 1] - reference["frequencies"])
            < np.abs(data["frequencies"][k] - reference["frequencies"]),
            k - 1,
            k,
        )
        shared = (
            np.abs(data["frequencies"][nearer] - reference["frequencies"])
            <= SAME_FREQUENCY_HZ
        )
        if not shared.any():
            continue
        difference = np.abs(data["s21"][nearer] - reference["s21"])[shared]
        frequencies = reference["frequencies"][shared]
        compared[name] = {
            "frequencies": int(shared.sum()),
            "max_abs_difference": float(difference.max()),
            "at_hz": float(frequencies[int(np.argmax(difference))]),
        }
        if keep_rows:
            compared[name]["rows"] = [
                {"frequency_hz": float(f), "abs_difference": float(d)}
                for f, d in zip(frequencies, difference, strict=True)
            ]
    return compared


def _runs_table(study: Path) -> str:
    """The runs table: settings, size, samples per port, offline and online time, total time and peak memory."""
    lines = [
        "| Run | Settings | Tetrahedra | Unknowns | Samples per port | Offline (s) | Online per frequency per port (s) | Total (s) | Peak memory (MB) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, run in RUNS.items():
        path = study / name / "report.json"
        if not path.is_file():
            continue
        report = _read(path)
        sampling = report["log"]["adaptive_sampling"]
        timing = report["log"]
        per_frequency = ", ".join(
            f"{sweep['seconds_per_frequency']:.3g}" for sweep in timing["excitations"]
        )
        lines.append(
            f"| {name} | {_settings_text(run)} | {report['tetrahedra']:,} | "
            f"{report['palace']['degrees_of_freedom']:,} | "
            f"{', '.join(_samples_text(s) for s in sampling) or '-'} | "
            f"{timing['offline_seconds'] if timing['offline_seconds'] is not None else '-'} | "
            f"{per_frequency or '-'} | {report['palace']['total_seconds']:.1f} | "
            f"{report['palace']['peak_node_memory_mb']:,.0f} |"
        )
    return "\n".join(lines)


def _samples_text(sampling: dict[str, Any]) -> str:
    """A port's sample count, with its last error when the sweep reached its cap before it converged."""
    if sampling["converged"]:
        return str(sampling["samples"])
    return f"{sampling['samples']} (cap; error {sampling['error']:.3g})"


def _settings_text(run: Run) -> str:
    if run.problem == "Eigenmode":
        parts = [f"Eigenmode, {EIGEN_MODES} modes from {EIGEN_TARGET_HZ / 1e9:g} GHz"]
        if run.linear_tolerance is not None:
            parts.append(f"linear tol {run.linear_tolerance:g}")
        if run.eigen_tolerance is not None:
            parts.append(f"eigen tol {run.eigen_tolerance:g}")
        return ", ".join(parts)
    sweep = {
        "band+window": f"{BAND_POINTS} band + {WINDOW_POINTS} window",
        "full-solves": f"{FULL_SOLVE_COUNT} full solves",
        "uniform": f"{run.uniform_points} uniform",
    }[run.sweep or "uniform"]
    parts = [sweep]
    if run.adaptive_tolerance is not None:
        parts.append(f"adaptive {run.adaptive_tolerance:g}")
    if run.adaptive_max_samples is not None:
        parts.append(f"max samples {run.adaptive_max_samples}")
    if run.add_to_prom_at_f0:
        parts.append("AddToPROM at f0")
    if run.linear_tolerance is not None:
        parts.append(f"linear tol {run.linear_tolerance:g}")
    if not run.surface_epr:
        parts.append("no Surface EPR")
    return ", ".join(parts)


def _eigenmode_table(study: Path) -> str:
    """Each Eigenmode run's resonator mode: f0, Q, the ports' Q_ext and kappa_ext, and its linear solves."""
    lines = [
        "| Run | f0 (GHz) | Q = Re f / (2 Im f) | Q_ext[1] | Q_ext[2] | Q_ext together | kappa_ext[1], [2] (GHz) "
        "| Linear solves | GMRES iterations (per solve) | Total (s) | Peak memory (MB) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, run in RUNS.items():
        path = study / name / "report.json"
        if run.problem != "Eigenmode" or not path.is_file():
            continue
        report = _read(path)
        mode = report["resonator_mode"]
        external = _external_q(report)
        row = report["port_q"][mode["mode"] - 1]
        log = report["log"]
        lines.append(
            f"| {name} | {mode['frequency_hz'] / 1e9:.9f} | {mode['q']:.6g} | "
            f"{external['q_external'][0]:.6g} | {external['q_external'][1]:.6g} | "
            f"{external['q_external_together']:.6g} | "
            f"{row['κ_ext[1] (GHz)']:+.4e}, {row['κ_ext[2] (GHz)']:+.4e} | {log.get('gmres_solves')} | "
            f"{log.get('gmres_iterations')} ({log.get('gmres_iterations_range')}) | "
            f"{report['palace']['total_seconds']:.1f} | {report['palace']['peak_node_memory_mb']:,.0f} |"
        )
    return "\n".join(lines)


def _results_table(record: dict[str, Any]) -> str:
    """The results table: the fitted notch of each run against the Eigenmode mode."""
    eigen = record["eigenmode"]
    lines = [
        f"Eigenmode: f0 = {eigen['frequency_hz'] / 1e9:.9f} GHz, Q = Re f / (2 Im f) = {eigen['q']:.6g}; "
        f"port-Q: Q_ext = {eigen['q_external'][0]:.6g} and {eigen['q_external'][1]:.6g}, together "
        f"{eigen['q_external_together']:.6g}",
        "",
        "| Run | Points fitted | rms residual | f0 (GHz) | Q_l | abs(Q_c) | Q_i | f0 - f0_eig (Hz) | "
        "Q_l / Q_eig - 1 | "
        + " | ".join(f"max abs(S21 - S21 of {reference})" for reference in REFERENCES)
        + " |",
        "|---|---|---|---|---|---|---|---|---|" + "---|" * len(REFERENCES),
    ]
    for run, fitted in record["fits"].items():
        # A run driven_result() refused is read from Palace's table: the table says so.
        name = run if fitted["s21_from"] == "driven_result()" else f"{run} (port-S.csv)"
        differences = [
            record["against"][reference].get(run, {}).get("max_abs_difference")
            for reference in REFERENCES
        ]
        compared = (
            " | ".join(
                "-" if value is None else f"{value:.3e}" for value in differences
            )
            + " |"
        )
        if not fitted.get("fitted"):
            lines.append(
                f"| {name} | {fitted.get('reason', 'not fitted')} | | | | | | | | {compared}"
            )
            continue
        lines.append(
            f"| {name} | {fitted['points']} | {fitted['rms_residual']:.2e} | "
            f"{fitted['f0_hz'] / 1e9:.9f} | {fitted['q_loaded']:.6g} | "
            f"{fitted['q_coupling_abs']:.6g} | {fitted['q_internal']:.4g} | "
            f"{fitted['f0_hz'] - eigen['frequency_hz']:.4g} | "
            f"{fitted['q_loaded'] / eigen['q'] - 1:.3e} | {compared}"
        )
    return "\n".join(lines)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, record: dict[str, Any]) -> None:
    path.write_text(json.dumps(record, indent=1, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
