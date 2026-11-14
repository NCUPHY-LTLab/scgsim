"""Run folders with a stand-in Palace, and the real run script run on them: for the convergence stop tests.

``prepared_run`` writes the config and handoff of a declared six-pass run whose ``palace`` is
``palace_stand_in_solver``; ``start_script`` and ``finish_script`` run the generated script as a person or a
scheduler would, with a ``python3`` that sees only the standard library, and leave none of its processes.

The declared run settles at pass 4 of the 6 its limit allows (``amr_max_passes=5``): every criterion holds at
passes 3 and 4, two in a row.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from palace_convergence_support import (
    eigenmode,
    electrostatic,
    write_eigenmode_pass,
    write_electrostatic_pass,
)
from palace_stand_in_solver import (
    end_left_over,
    runner_processes,
    write_stand_in,
)

from scgsim.contracts.palace_convergence import ConvergenceDecision
from scgsim.contracts.palace_run_folder import (
    CONVERGENCE_DECISION,
    INDEX_MAP,
    read_json,
    solver_log,
)
from scgsim.palace import (
    Capacitance,
    ErrorIndicator,
    ModeFrequency,
    SurfaceEpr,
)
from scgsim.palace.results.tables import count_index_entries

SETTLES_AT_4 = [
    (100.0, 300.0),
    (101.0, 300.06),
    (101.05, 300.07),
    (101.06, 300.07),
    (101.06, 300.07),
    (101.06, 300.07),
]
SETTLES_AT_5 = [
    (100.0, 300.0),
    (101.0, 300.06),
    (103.0, 300.5),
    (103.05, 300.51),
    (103.06, 300.51),
    (103.06, 300.51),
]
"""Every criterion holds at passes 4 and 5: converged at pass 5, the last before the run's final solve."""
NEVER_SETTLES = [(100.0 + 2 * k, 300.0 + k) for k in range(6)]
NORMS = [0.2, 0.04, 0.03, 0.02, 0.02, 0.02]
DECLARED = (
    Capacitance.diagonal().within(percent=0.1),
    SurfaceEpr.each("MA", "MS", margin_um=1.0).within(percent=2),
    ErrorIndicator().below(0.05),
)
LONG_S = 60.0
"""A pause no test waits out: the runner stops the stand-in during it, or the test ends it."""


def isolated_python(directory: Path) -> Path:
    """A ``python3`` on PATH that sees only the standard library, as on a solver host."""
    directory.mkdir(parents=True, exist_ok=True)
    wrapper = directory / "python3"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" -I -S "$@"\n')
    wrapper.chmod(0o755)
    srun = directory / "srun"
    srun.write_text('#!/bin/sh\nexec "$@"\n')
    srun.chmod(0o755)
    return directory


def prepared_run(  # noqa: PLR0913 -- one run folder, each switch independent
    tmp_path: Path,
    *,
    problem: str = "Electrostatic",
    values: list | None = None,
    refine_s: list[float] | None = None,
    solve_s: list[float] | None = None,
    declare: bool = True,
    profile: str = "direct-local",
    style: str = "binary",
    fail_after: int | None = None,
    ignore_sigterm: bool = False,
    behind_wrapper: bool = False,
    shutdown_s: float = 0.0,
    finish_s: float = 0.0,
    ends_itself_on_decision: bool = False,
) -> SimpleNamespace:
    """Write the config and handoff of a six-pass run with its stand-in Palace; nothing runs yet."""
    if problem == "Electrostatic":
        sim = electrostatic(tmp_path / "model")
        values = values or SETTLES_AT_4
        criteria = DECLARED
    else:
        sim = eigenmode(tmp_path / "model", num_modes=1)
        values = values or [(5.0,), (5.2,), (5.2001,), (5.2001,), (5.2001,), (5.2001,)]
        criteria = (ModeFrequency().within(percent=0.1), ErrorIndicator().below(0.05))
    sim.set_numerical(
        amr_max_passes=len(values) - 1,
        **({} if problem == "Electrostatic" else {"amr_update_fraction": 0.7}),
    )
    if declare:
        sim.set_convergence(*criteria, minimum_converged_passes=2)
    sim.write_config()
    counts = count_index_entries(read_json(sim.output_dir / INDEX_MAP))
    write = (
        write_electrostatic_pass if problem == "Electrostatic" else write_eigenmode_pass
    )
    passes = []
    for number, value in enumerate(values, start=1):
        folder = tmp_path / "solve" / f"pass{number}"
        write(folder, number, counts, value, NORMS[number - 1])
        passes.append(
            {
                "tables": str(folder),
                "norm": NORMS[number - 1],
                "solve_s": (solve_s or [0.05] * len(values))[number - 1],
                "refine_s": (refine_s or [0.05] * len(values))[number - 1],
            }
        )
    pids = tmp_path / "pids.json"
    palace = write_stand_in(
        tmp_path / "bin" / "palace",
        problem=problem,
        passes=passes,
        pids=pids,
        fail_after=fail_after,
        ignore_sigterm=ignore_sigterm,
        behind_wrapper=behind_wrapper,
        shutdown_s=shutdown_s,
        finish_s=finish_s,
        ends_itself_on_decision=ends_itself_on_decision,
    )
    resources = (
        {
            "processes": 1,
            "threads": 1,
            **({"command_style": style} if style != "binary" else {}),
        }
        if profile == "direct-local"
        else {"ntasks": 1, "cpus_per_task": 1, "nodes": 1}
    )
    plan = sim.prepare_handoff(
        profile=profile, executable=str(palace), resources=resources
    )
    return SimpleNamespace(
        root=plan.run_dir,
        plan=plan,
        handoff_id=plan.handoff_id,
        pids_path=pids,
        python=isolated_python(tmp_path / "bin"),
        profile=profile,
    )


def start_script(run: SimpleNamespace) -> subprocess.Popen:
    """Start the run script as a person or a scheduler would: in its own session, the log where it goes."""
    environment = {
        **os.environ,
        "PATH": f"{run.python}{os.pathsep}{os.environ['PATH']}",
    }
    if run.profile == "direct-local":
        output = subprocess.PIPE
    else:
        environment.update(
            {
                "SLURM_SUBMIT_DIR": str(run.root),
                "SLURM_JOB_ID": "4242",
                "SLURM_JOB_NAME": "stop",
            }
        )
        # Slurm writes the job's output to the log the script's directives name.
        output = (run.root / solver_log("4242")).open("ab")
    return subprocess.Popen(
        ["bash", str(run.plan.script_path)],
        stdout=output,
        stderr=subprocess.STDOUT,
        env=environment,
        start_new_session=True,
    )


def finish_script(
    run: SimpleNamespace, process: subprocess.Popen, timeout_s: float = 60.0
) -> int:
    """Wait for the run script; whatever the test saw, leave none of the run's processes behind."""
    try:
        process.wait(timeout=timeout_s)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        if run.pids_path.is_file():
            end_left_over(json.loads(run.pids_path.read_text()))
    run.log = (run.root / log_name(run)).read_text(encoding="utf-8")
    return process.returncode


def log_name(run: SimpleNamespace) -> str:
    """The solver log of the run, as its profile names it."""
    return solver_log("manual" if run.profile == "direct-local" else "4242")


def run_script(run: SimpleNamespace) -> int:
    """Run the generated script to its end; return its exit code."""
    return finish_script(run, start_script(run))


def decision_of(run: SimpleNamespace) -> ConvergenceDecision:
    """The decision record the runner left in the run folder."""
    return ConvergenceDecision.from_payload(read_json(run.root / CONVERGENCE_DECISION))


def rewrite_json(path: Path, change) -> None:
    """Apply ``change`` to the JSON object in ``path`` and write it back."""
    payload = read_json(path)
    change(payload)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def runner_processes_left(run: SimpleNamespace, timeout_s: float = 10.0) -> list[int]:
    """The runner and guard processes of the run still there after ``timeout_s``."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and runner_processes(run.root):
        time.sleep(0.05)
    return runner_processes(run.root)


def wait_for(path: Path, timeout_s: float = 20.0) -> None:
    """Wait until ``path`` exists; fail the test after ``timeout_s``."""
    deadline = time.monotonic() + timeout_s
    while not path.exists():
        assert time.monotonic() < deadline, f"{path} did not appear"
        time.sleep(0.05)
