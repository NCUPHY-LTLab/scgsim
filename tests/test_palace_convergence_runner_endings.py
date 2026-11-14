"""A declared run ended in another way than by the stop at convergence, with a stand-in Palace.

The solver fails, is ended from outside or ignores the stop; the runner is told to end, is ended outright, or
its guard cannot be started.  In each case the receipt says how the run ended and no solver, runner or guard
process is left.  The run folders and the stand-in are those of ``palace_stop_support``; the run script, the
runner and the recorder are the real ones.  No solver runs.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys

import pytest
from palace_stand_in_solver import (
    read_pids,
    wait_until_gone,
)
from palace_stop_support import (
    LONG_S,
    decision_of,
    finish_script,
    prepared_run,
    run_script,
    runner_processes_left,
    start_script,
    wait_for,
)

from scgsim.contracts.palace_run_folder import (
    CONVERGENCE_DECISION,
    CONVERGENCE_RUNNER,
    OUTPUT_DIR,
    RETURNED_RECEIPT,
    read_json,
)
from scgsim.palace import (
    inspect_run_trustworthiness,
    resolve_palace_result,
)

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="the runner's stop is checked through /proc",
)


def test_a_solver_that_fails_is_failed(tmp_path):
    """The solver exits 2 after pass 2: no stop, ``failed``, the exit code kept, nothing left."""
    run = prepared_run(tmp_path, fail_after=2)
    assert run_script(run) == 2
    decision = decision_of(run)
    assert decision.ending == "solver_ended" and decision.solver.exit_code == 2
    receipt = read_json(run.root / RETURNED_RECEIPT)
    assert (receipt["status"], receipt["ending"], receipt["solver_exit_code"]) == (
        "failed",
        "solver_ended",
        2,
    )
    assert wait_until_gone(read_pids(run.pids_path)) == []
    with pytest.raises(ValueError, match="status must be completed"):
        resolve_palace_result(run.root, expected_handoff_id=run.handoff_id)


def test_a_solver_killed_from_outside_is_failed_not_a_stop(tmp_path):
    """SIGKILL to the solver before convergence: the runner did not stop it, so the run is ``failed``."""
    run = prepared_run(tmp_path, refine_s=[LONG_S] * 6)
    process = start_script(run)
    pids = read_pids(run.pids_path)
    wait_for(run.root / OUTPUT_DIR / "iteration1")
    os.kill(pids["solver"], signal.SIGKILL)
    assert finish_script(run, process) == 137
    decision = decision_of(run)
    assert (decision.ending, decision.stop, decision.solver.exit_code) == (
        "solver_ended",
        None,
        137,
    )
    receipt = read_json(run.root / RETURNED_RECEIPT)
    assert (receipt["status"], receipt["ending"]) == ("failed", "solver_ended")
    # The rank the killed launcher left is ended with the rest of the solver's group.
    assert wait_until_gone(pids) == []


def test_a_solver_that_ignores_sigterm_is_killed_after_the_wait(tmp_path):
    """SIGTERM is ignored: after the wait the runner sends SIGKILL, records it, and the run is ``failed``."""
    run = prepared_run(
        tmp_path, ignore_sigterm=True, refine_s=[0.05, 0.05, 0.05, LONG_S, 0.05, 0.05]
    )
    runner = run.root / CONVERGENCE_RUNNER
    # The real runner with its wait shortened, so the test does not wait the minute out.
    runner.write_text(
        runner.read_text().replace("\nSTOP_WAIT_S = 60.0\n", "\nSTOP_WAIT_S = 1.0\n"),
        encoding="utf-8",
    )
    assert run_script(run) != 0
    decision = decision_of(run)
    assert decision.ending == "stop_escalated"
    assert decision.stop.escalated_utc is not None
    assert decision.solver.exit_code == 128 + signal.SIGKILL
    assert read_json(run.root / RETURNED_RECEIPT)["status"] == "failed"
    assert "sending SIGKILL to its session" in run.log
    assert wait_until_gone(read_pids(run.pids_path)) == []
    failure = inspect_run_trustworthiness(run.root).failure
    assert failure is not None and failure.category == "convergence_stop"
    assert "did not end on SIGTERM" in failure.summary


def test_a_runner_told_to_end_ends_the_solver_first(tmp_path):
    """SIGTERM to the runner during a solve: it stops the solver, records why, and leaves nothing."""
    run = prepared_run(tmp_path, refine_s=[LONG_S] * 6)
    process = start_script(run)
    pids = read_pids(run.pids_path)
    wait_for(run.root / OUTPUT_DIR / "iteration1")
    os.kill(pids["runner"], signal.SIGTERM)
    assert finish_script(run, process) != 0
    decision = decision_of(run)
    assert decision.ending == "runner_terminated"
    assert decision.stop.reason == "runner_terminated"
    assert read_json(run.root / RETURNED_RECEIPT)["status"] == "failed"
    assert wait_until_gone(pids) == []
    failure = inspect_run_trustworthiness(run.root).failure
    assert failure is not None and failure.category == "convergence_stop"


def test_a_runner_told_to_end_while_it_stops_at_convergence_leaves_the_stop_standing(
    tmp_path,
):
    """SIGTERM to the runner while Palace is ending after the stop at convergence: the stop stands.

    The run had converged and the runner had already stopped Palace for it, which is what the record says;
    ``runner_terminated`` is for a runner that ended Palace because it was told to end.
    """
    run = prepared_run(
        tmp_path,
        refine_s=[0.05, 0.05, 0.05, LONG_S, 0.05, 0.05],
        behind_wrapper=True,
        shutdown_s=2.0,
    )
    process = start_script(run)
    pids = read_pids(run.pids_path)
    wait_for(run.root / CONVERGENCE_DECISION)
    os.kill(pids["runner"], signal.SIGTERM)
    assert finish_script(run, process) == 0, run.log
    decision = decision_of(run)
    assert decision.ending == "stopped_at_convergence"
    assert decision.stop is not None and decision.stop.reason == "converged"
    receipt = read_json(run.root / RETURNED_RECEIPT)
    assert (receipt["status"], receipt["ending"]) == (
        "completed",
        "stopped_at_convergence",
    )
    launcher_ended = float(run.pids_path.with_name("launcher_ended").read_text())
    assert (run.root / RETURNED_RECEIPT).stat().st_mtime >= launcher_ended


_NO_GUARD = """
import os, runpy, subprocess, sys

def no_process_left():
    raise OSError(11, "no process left")

def solver(*_arguments, **_options):
    print("SOLVER STARTED")

def pipes():
    held = []
    for descriptor in os.listdir("/proc/self/fd"):
        try:
            link = os.readlink("/proc/self/fd/" + descriptor)
        except OSError:
            continue  # the listing's own descriptor
        if link.startswith("pipe:"):
            held.append((descriptor, link))
    return sorted(held)

os.fork = no_process_left
subprocess.Popen = solver
root, runner = sys.argv[1:]
sys.argv = ["runner", root, "--", "true"]
before = pipes()
try:
    runpy.run_path(os.path.join(root, runner), run_name="__main__")
except OSError as error:
    print("REFUSED:", error.strerror, "| pipes as before:", pipes() == before)
"""


def test_no_solver_is_started_when_its_guard_cannot_be(tmp_path):
    """The guard is started first: when it cannot be, no solver is started and its pipe is closed again.

    The copied runner is run with a ``fork`` that fails, as on a host with no process left to give.
    """
    run = prepared_run(tmp_path)
    done = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            _NO_GUARD,
            str(run.root),
            str(CONVERGENCE_RUNNER),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.stdout == "REFUSED: no process left | pipes as before: True\n", (
        done.stdout + done.stderr
    )


@pytest.mark.parametrize(
    "behind_wrapper", [False, True], ids=["launcher", "wrapper-script"]
)
def test_a_runner_killed_outright_still_leaves_no_solver(tmp_path, behind_wrapper):
    """SIGKILL to the runner: its guard ends the solver's session, and nothing is left.

    Behind a wrapper script, as Palace's own ``palace`` is, the launcher is not the solver's first process:
    only a signal to the whole session ends it and its rank.
    """
    run = prepared_run(tmp_path, refine_s=[LONG_S] * 6, behind_wrapper=behind_wrapper)
    process = start_script(run)
    pids = read_pids(run.pids_path)
    wait_for(run.root / OUTPUT_DIR / "iteration1")
    os.kill(pids["runner"], signal.SIGKILL)
    assert finish_script(run, process) != 0
    assert wait_until_gone(pids) == []
    assert runner_processes_left(run) == []
    receipt = read_json(run.root / RETURNED_RECEIPT)
    assert (receipt["status"], receipt["ending"]) == ("failed", "not_recorded")


def test_a_script_killed_with_its_whole_group_still_leaves_no_solver(tmp_path):
    """SIGKILL to the script's process group ends the script, ``tee`` and the runner at once.

    The solver is in a session of its own, so that signal does not reach it; the guard, in a session of its
    own too, ends it.  The script wrote no receipt.
    """
    run = prepared_run(tmp_path, refine_s=[LONG_S] * 6, behind_wrapper=True)
    process = start_script(run)
    pids = read_pids(run.pids_path)
    wait_for(run.root / OUTPUT_DIR / "iteration1")
    os.killpg(process.pid, signal.SIGKILL)
    # Looked at before ``finish_script``, which ends whatever a run left behind.
    left = wait_until_gone(pids)
    assert finish_script(run, process) == -signal.SIGKILL
    assert left == []
    assert runner_processes_left(run) == []
    assert not (run.root / RETURNED_RECEIPT).exists()
