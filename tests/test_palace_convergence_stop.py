"""The stop at convergence: the real run script, convergence runner and recorder, with a stand-in Palace.

Each run folder is the one ``write_config()`` and ``prepare_handoff()`` write for a real Route B SGB build
(``palace_convergence_support``).  The generated run script is executed for real with a ``python3`` that sees
only the standard library; the ``palace`` it starts is ``palace_stand_in_solver``, which writes its passes the
way Palace 0.18.1 was recorded doing (tables at the top level, the move into ``iteration<k>/``, the links, the
rewritten ``palace.json``, the pause for refinement, the log lines) and answers SIGTERM as a launcher does.
The runner and the recorder are the copies the handoff put in the run folder.  No solver runs.

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

import pytest
from palace_convergence_support import (
    eigenmode,
    electrostatic,
    write_eigenmode_pass,
    write_electrostatic_pass,
)
from palace_stand_in_solver import (
    end_left_over,
    read_pids,
    wait_until_gone,
    write_stand_in,
)

from scgsim.contracts.palace_convergence import ConvergenceDecision
from scgsim.contracts.palace_run_folder import (
    CONVERGENCE_DECISION,
    CONVERGENCE_RUNNER,
    INDEX_MAP,
    OUTPUT_DIR,
    RESOURCE_RECORD,
    RETURNED_RECEIPT,
    RUN_METADATA,
    read_json,
    record_for_path,
    solver_log,
)
from scgsim.palace import (
    Capacitance,
    ErrorIndicator,
    ModeFrequency,
    SurfaceEpr,
    inspect_run_trustworthiness,
    resolve_palace_result,
)
from scgsim.palace.reports.convergence import verdict_card, verdict_sentence
from scgsim.palace.results.tables import count_index_entries

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="the runner's stop is checked through /proc and Linux's parent-death signal",
)

_SETTLES_AT_4 = [
    (100.0, 300.0),
    (101.0, 300.06),
    (101.05, 300.07),
    (101.06, 300.07),
    (101.06, 300.07),
    (101.06, 300.07),
]
_NEVER_SETTLES = [(100.0 + 2 * k, 300.0 + k) for k in range(6)]
_NORMS = [0.2, 0.04, 0.03, 0.02, 0.02, 0.02]
_DECLARED = (
    Capacitance.diagonal().within(relative=1e-3),
    SurfaceEpr.each("MA", "MS", margin_um=1.0).within(relative=2e-2),
    ErrorIndicator().below(0.05),
)
_LONG_S = 60.0
"""A pause no test waits out: the runner stops the stand-in during it, or the test ends it."""


def _isolated_python(directory: Path) -> Path:
    """A ``python3`` on PATH that sees only the standard library, as on a solver host."""
    directory.mkdir(parents=True, exist_ok=True)
    wrapper = directory / "python3"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" -I -S "$@"\n')
    wrapper.chmod(0o755)
    srun = directory / "srun"
    srun.write_text('#!/bin/sh\nexec "$@"\n')
    srun.chmod(0o755)
    return directory


def _prepared(  # noqa: PLR0913 -- one run folder, each switch independent
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
) -> SimpleNamespace:
    """Write the config and handoff of a six-pass run with its stand-in Palace; nothing runs yet."""
    if problem == "Electrostatic":
        sim = electrostatic(tmp_path / "model")
        values = values or _SETTLES_AT_4
        criteria = _DECLARED
    else:
        sim = eigenmode(tmp_path / "model", num_modes=1)
        values = values or [(5.0,), (5.2,), (5.2001,), (5.2001,), (5.2001,), (5.2001,)]
        criteria = (ModeFrequency().within(relative=1e-3), ErrorIndicator().below(0.05))
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
        write(folder, number, counts, value, _NORMS[number - 1])
        passes.append(
            {
                "tables": str(folder),
                "norm": _NORMS[number - 1],
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
        python=_isolated_python(tmp_path / "bin"),
        profile=profile,
    )


def _start(run: SimpleNamespace) -> subprocess.Popen:
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


def _finish(
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
    run.log = (run.root / _log_name(run)).read_text(encoding="utf-8")
    return process.returncode


def _log_name(run: SimpleNamespace) -> str:
    return solver_log("manual" if run.profile == "direct-local" else "4242")


def _run(run: SimpleNamespace) -> int:
    return _finish(run, _start(run))


def _decision(run: SimpleNamespace) -> ConvergenceDecision:
    return ConvergenceDecision.from_payload(read_json(run.root / CONVERGENCE_DECISION))


def _rewrite_json(path: Path, change) -> None:
    payload = read_json(path)
    change(payload)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@pytest.fixture(scope="module")
def stopped(tmp_path_factory):
    """One declared Electrostatic run stopped at convergence, after its pass 4 of 6."""
    run = _prepared(
        tmp_path_factory.mktemp("stopped"),
        refine_s=[0.05, 0.05, 0.05, _LONG_S, 0.05, 0.05],
    )
    run.exit_code = _run(run)
    return run


def test_the_runner_stops_palace_after_the_first_converged_pass(stopped):
    """Converged at pass 4: the decision is written, the solver's session gets SIGTERM, nothing is left."""
    assert stopped.exit_code == 0, stopped.log
    decision = _decision(stopped)
    assert decision.ending == "stopped_at_convergence"
    assert decision.stop is not None
    assert (decision.stop.reason, decision.stop.after_pass) == ("converged", 4)
    assert decision.stop.escalated_utc is None
    assert decision.solver is not None and decision.solver.exit_code == 1
    assert [item["number"] for item in decision.checked["passes"]] == [1, 2, 3, 4]
    assert decision.checked["verdict"]["state"] == "converged"
    assert decision.checked["verdict"]["pass_number"] == 4
    assert sorted(decision.seen_utc) == ["1", "2", "3", "4"]
    results = stopped.root / OUTPUT_DIR
    assert not (results / "iteration5").exists()
    assert (results / "terminal-C.csv").is_symlink()
    assert wait_until_gone(read_pids(stopped.pids_path)) == []
    assert "Adaptive mesh refinement (AMR) iteration 4:" in stopped.log
    assert (
        "SCGSim convergence runner: converged at pass 4; stopping the solver after pass 4"
        in stopped.log
    )
    assert "Proceeding with solve/estimate iteration 5..." not in stopped.log


def test_the_receipt_states_the_stop_and_keeps_the_solver_exit_code(stopped):
    """``completed`` by a stop: the ending, the solver's real exit code 1, the decision bound by its hash."""
    receipt = read_json(stopped.root / RETURNED_RECEIPT)
    assert receipt["schema"] == "palace-returned-run-receipt.v2"
    assert receipt["schema_version"] == 2
    assert receipt["status"] == "completed"
    assert receipt["ending"] == "stopped_at_convergence"
    assert (
        receipt["exit_code"],
        receipt["solver_exit_code"],
        receipt["tee_exit_code"],
    ) == (1, 1, 0)
    assert receipt["convergence_decision"] == record_for_path(
        stopped.root, CONVERGENCE_DECISION
    )
    paths = [record["path"] for record in receipt["output_files"]]
    assert f"{OUTPUT_DIR}/iteration4/terminal-C.csv" in paths
    assert all(record["present"] for record in receipt["output_files"])


def test_result_reading_accepts_the_stop_and_reads_the_last_pass(stopped):
    """The result is pass 4, read from ``iteration4/``: its tables and its own ``palace.json``."""
    result = resolve_palace_result(stopped.root, expected_handoff_id=stopped.handoff_id)
    assert result.status == "completed"
    assert result.returned_receipt.ending == "stopped_at_convergence"
    assert (result.solver_end.ending, result.solver_end.stopped_after) == (
        "stopped_at_convergence",
        4,
    )
    assert result.solver_end.passes_allowed == 6
    assert result.convergence.verdict.pass_number == 4
    assert (
        result.tables["terminal-C"].path
        == stopped.root / OUTPUT_DIR / "iteration4/terminal-C.csv"
    )
    # The top-level palace.json already describes the next mesh (16000 unknowns); the result's is pass 4's.
    assert (
        read_json(stopped.root / OUTPUT_DIR / "palace.json")["Problem"][
            "DegreesOfFreedom"
        ]
        == 16000
    )
    assert result.cost.problem_degrees_of_freedom == 4000


def test_the_report_says_where_it_stopped_and_what_was_not_run(stopped):
    """The verdict sentence, the card and the trust report's completeness for a stopped run."""
    result = resolve_palace_result(stopped.root, expected_handoff_id=stopped.handoff_id)
    sentence = verdict_sentence(result.convergence, result.solver_end)
    assert sentence.startswith(
        "Converged at pass 4, of the 4 passes read (passes 1 to 4): every criterion held at passes 3 to 4"
    )
    assert sentence.endswith(
        "SCGSim stopped Palace after pass 4: 2 of the 6 passes its limit allows were not run."
    )
    assert verdict_card(result.convergence, result.solver_end) == (
        "converged at pass 4, stopped after pass 4"
    )
    trust = result.show_run_trustworthiness()
    assert trust.completeness == "complete"
    assert trust.selection.reason == "last_pass_of_a_stopped_run"
    assert trust.selection.selected_source == "iteration4"
    assert "Stopped at convergence" in trust.identity_html()
    assert "none: stopped at convergence" in trust.identity_html()
    inspected = inspect_run_trustworthiness(stopped.root)
    assert inspected.completeness == "complete"
    assert verdict_sentence(inspected.convergence, inspected.solver_end) == sentence
    # The benchmark and cost views use the passes that exist: pass 4 is the last solve, its cost the result's.
    benchmark = result.show_simulation_benchmark().data
    assert benchmark["cost"]["problem_degrees_of_freedom"] == 4000
    assert [
        record["native_iteration"] for record in benchmark["pass_cost_records"]
    ] == [1, 2, 3, 4]
    assert benchmark["pass_cost_records"][-1]["degrees_of_freedom"] == 4000


@pytest.mark.parametrize(
    ("profile", "style"),
    [("direct-local", "wrapper"), ("slurm-single-node", "binary")],
)
def test_every_profile_and_command_style_stops_the_same_way(tmp_path, profile, style):
    """The wrapper's ``-np``/``-nt`` command and the Slurm script's ``srun`` command run through the runner."""
    run = _prepared(
        tmp_path,
        profile=profile,
        style=style,
        refine_s=[0.05, 0.05, 0.05, _LONG_S, 0.05, 0.05],
    )
    script = run.plan.script_path.read_text()
    assert f'python3 {CONVERGENCE_RUNNER} "$(pwd)" -- ' in script
    assert _run(run) == 0, run.log
    assert _decision(run).ending == "stopped_at_convergence"
    assert wait_until_gone(read_pids(run.pids_path)) == []
    result = resolve_palace_result(run.root, expected_handoff_id=run.handoff_id)
    assert result.solver_end.stopped_after == 4


def test_an_eigenmode_run_stopped_during_the_next_solve_reads_its_last_pass(tmp_path):
    """Stopped while the next solve has started ``eig.csv`` with its header: the result is still pass 4."""
    run = _prepared(
        tmp_path,
        problem="Eigenmode",
        solve_s=[0.05, 0.05, 0.05, 0.05, _LONG_S, 0.05],
    )
    assert _run(run) == 0, run.log
    results = run.root / OUTPUT_DIR
    assert not (results / "eig.csv").is_symlink()
    assert len((results / "eig.csv").read_text().splitlines()) == 1
    result = resolve_palace_result(run.root, expected_handoff_id=run.handoff_id)
    assert result.solver_end.stopped_after == 4
    assert result.convergence.verdict.pass_number == 4
    assert result.tables["eig"].path == results / "iteration4/eig.csv"
    assert any(
        "solve under way" in item.reason for item in result.convergence.set_aside
    )


def test_a_run_that_does_not_converge_ends_at_its_limit(tmp_path):
    """No stop: Palace runs every pass its limit allows, the receipt is completed, and the report says so."""
    run = _prepared(tmp_path, values=_NEVER_SETTLES)
    assert _run(run) == 0, run.log
    decision = _decision(run)
    assert decision.ending == "solver_ended" and decision.stop is None
    receipt = read_json(run.root / RETURNED_RECEIPT)
    assert (receipt["status"], receipt["ending"], receipt["solver_exit_code"]) == (
        "completed",
        "solver_ended",
        0,
    )
    result = resolve_palace_result(run.root, expected_handoff_id=run.handoff_id)
    assert result.convergence.verdict.state == "not_converged"
    assert [item.number for item in result.convergence.passes] == [1, 2, 3, 4, 5, 6]
    sentence = verdict_sentence(result.convergence, result.solver_end)
    assert sentence.startswith(
        "Not converged in the 6 passes read (passes 1 to 6). At pass 6:"
    )
    assert sentence.endswith("Palace ended by itself at its limit of 6 passes.")
    assert result.tables["terminal-C"].path == run.root / OUTPUT_DIR / "terminal-C.csv"


def test_a_solver_that_fails_is_failed(tmp_path):
    """The solver exits 2 after pass 2: no stop, ``failed``, the exit code kept, nothing left."""
    run = _prepared(tmp_path, fail_after=2)
    assert _run(run) == 2
    decision = _decision(run)
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
    run = _prepared(tmp_path, refine_s=[_LONG_S] * 6)
    process = _start(run)
    pids = read_pids(run.pids_path)
    _wait_for(run.root / OUTPUT_DIR / "iteration1")
    os.kill(pids["solver"], signal.SIGKILL)
    assert _finish(run, process) == 137
    decision = _decision(run)
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
    run = _prepared(
        tmp_path, ignore_sigterm=True, refine_s=[0.05, 0.05, 0.05, _LONG_S, 0.05, 0.05]
    )
    runner = run.root / CONVERGENCE_RUNNER
    # The real runner with its wait shortened, so the test does not wait the minute out.
    runner.write_text(
        runner.read_text().replace("\nSTOP_WAIT_S = 60.0\n", "\nSTOP_WAIT_S = 1.0\n"),
        encoding="utf-8",
    )
    assert _run(run) != 0
    decision = _decision(run)
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
    run = _prepared(tmp_path, refine_s=[_LONG_S] * 6)
    process = _start(run)
    pids = read_pids(run.pids_path)
    _wait_for(run.root / OUTPUT_DIR / "iteration1")
    os.kill(pids["runner"], signal.SIGTERM)
    assert _finish(run, process) != 0
    decision = _decision(run)
    assert decision.ending == "runner_terminated"
    assert decision.stop.reason == "runner_terminated"
    assert read_json(run.root / RETURNED_RECEIPT)["status"] == "failed"
    assert wait_until_gone(pids) == []
    failure = inspect_run_trustworthiness(run.root).failure
    assert failure is not None and failure.category == "convergence_stop"


def test_a_runner_killed_outright_still_leaves_no_solver(tmp_path):
    """SIGKILL to the runner: Linux ends the solver through its parent-death signal; nothing is left."""
    run = _prepared(tmp_path, refine_s=[_LONG_S] * 6)
    process = _start(run)
    pids = read_pids(run.pids_path)
    _wait_for(run.root / OUTPUT_DIR / "iteration1")
    os.kill(pids["runner"], signal.SIGKILL)
    assert _finish(run, process) != 0
    assert wait_until_gone(pids) == []
    receipt = read_json(run.root / RETURNED_RECEIPT)
    assert (receipt["status"], receipt["ending"]) == ("failed", "not_recorded")


def _wait_for(path: Path, timeout_s: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not path.exists():
        assert time.monotonic() < deadline, f"{path} did not appear"
        time.sleep(0.05)


# ---- Result reading refuses what disagrees ----------------------------------------------------------------


def test_a_decision_record_edited_after_the_run_is_refused(tmp_path):
    """Edited alone, its hash no longer matches the receipt; edited with the receipt, the recompute differs."""
    run = _prepared(tmp_path, refine_s=[0.05, 0.05, 0.05, _LONG_S, 0.05, 0.05])
    assert _run(run) == 0, run.log
    path = run.root / CONVERGENCE_DECISION

    def change_a_reading(payload):
        payload["passes"][3]["criteria"][0]["values"][0]["value"] = 1.2e-13

    _rewrite_json(path, change_a_reading)
    with pytest.raises(ValueError, match="changed after .* was written"):
        resolve_palace_result(run.root, expected_handoff_id=run.handoff_id)
    _rewrite_json(
        run.root / RETURNED_RECEIPT,
        lambda receipt: receipt.update(
            convergence_decision=record_for_path(run.root, CONVERGENCE_DECISION)
        ),
    )
    with pytest.raises(
        ValueError,
        match=r"does not match the passes read again from results/palace: pass 4, C diagonal, "
        r"C\[pad\]\[pad\], value: recorded 1\.2e-13, read 1\.0106e-13",
    ):
        resolve_palace_result(run.root, expected_handoff_id=run.handoff_id)


def test_a_stopped_folder_with_a_receipt_forged_to_say_palace_ended_by_itself_is_refused(
    tmp_path,
):
    """Exit codes rewritten to 0 and the ending to ``solver_ended``: the top level of links says otherwise."""
    run = _prepared(tmp_path, refine_s=[0.05, 0.05, 0.05, _LONG_S, 0.05, 0.05])
    assert _run(run) == 0, run.log

    def ended_by_itself(record):
        record.update(exit_code=0, solver_exit_code=0, ending="solver_ended")

    _rewrite_json(run.root / RETURNED_RECEIPT, ended_by_itself)
    for record in (RUN_METADATA, RESOURCE_RECORD):
        _rewrite_json(run.root / record, lambda payload: payload.update(exit_code=0))
    with pytest.raises(
        ValueError,
        match=r"is the folder of a solver ended during its adaptive loop \(the final result's tables are "
        r"links into iteration4\), and metadata/palace_returned_run_receipt.json says the solver ended by itself",
    ):
        resolve_palace_result(run.root, expected_handoff_id=run.handoff_id)


def test_final_tables_of_its_own_with_a_receipt_saying_stopped_are_refused(tmp_path):
    """A run that ended by itself, its receipt forged to say it was stopped: the top-level solve says otherwise."""
    run = _prepared(tmp_path, values=_NEVER_SETTLES)
    assert _run(run) == 0, run.log
    _rewrite_json(
        run.root / RETURNED_RECEIPT,
        lambda receipt: receipt.update(ending="stopped_at_convergence"),
    )
    with pytest.raises(ValueError, match="holds a solve of its own at its top level"):
        resolve_palace_result(run.root, expected_handoff_id=run.handoff_id)


def test_a_plan_without_its_decision_record_is_refused(tmp_path):
    """The receipt of this version binds the decision of a declared run; without it the run is not read."""
    run = _prepared(tmp_path, values=_NEVER_SETTLES)
    assert _run(run) == 0, run.log
    (run.root / CONVERGENCE_DECISION).unlink()
    with pytest.raises(
        FileNotFoundError, match="is missing, and the run has a convergence plan"
    ):
        resolve_palace_result(run.root, expected_handoff_id=run.handoff_id)


def test_a_decision_record_without_a_plan_is_refused(tmp_path):
    """A run with no declaration has no runner, so a decision record in its folder is refused."""
    run = _prepared(tmp_path, declare=False)
    assert _run(run) == 0, run.log
    (run.root / CONVERGENCE_DECISION).write_text("{}", encoding="utf-8")
    with pytest.raises(
        ValueError, match="sealed no convergence plan: no runner ran for this handoff"
    ):
        resolve_palace_result(run.root, expected_handoff_id=run.handoff_id)


def test_a_receipt_an_earlier_scgsim_wrote_is_refused_with_how_to_read_it(stopped):
    """Version 1 receipts are not read: the refusal names the commit that wrote it as the way to read it."""
    path = stopped.root / RETURNED_RECEIPT
    original = path.read_text(encoding="utf-8")
    try:
        _rewrite_json(
            path,
            lambda receipt: receipt.update(
                schema="palace-returned-run-receipt.v1", schema_version=1
            ),
        )
        with pytest.raises(ValueError, match="was written by an earlier SCGSim"):
            resolve_palace_result(stopped.root, expected_handoff_id=stopped.handoff_id)
        with pytest.raises(
            ValueError, match="read this result with the SCGSim commit that wrote it"
        ):
            inspect_run_trustworthiness(stopped.root)
    finally:
        path.write_text(original, encoding="utf-8")


def test_without_a_declaration_palace_runs_as_before(tmp_path):
    """No plan: no runner in the folder, the script starts Palace itself, the receipt has no decision."""
    run = _prepared(tmp_path, declare=False)
    script = run.plan.script_path.read_text()
    assert CONVERGENCE_RUNNER not in script and CONVERGENCE_DECISION not in script
    assert (
        '\n"$PALACE_EXECUTABLE" "$PALACE_CONFIG" 2>&1 | tee -a "$PALACE_LOG"\n'
        in script
    )
    assert script.endswith(
        'if [ "$RECEIPT_EXIT" -ne 0 ]; then exit "$RECEIPT_EXIT"; fi\nexit "$PIPELINE_EXIT_CODE"\n'
    )
    assert not (run.root / CONVERGENCE_RUNNER).exists()
    assert _run(run) == 0, run.log
    receipt = read_json(run.root / RETURNED_RECEIPT)
    assert (receipt["status"], receipt["ending"], receipt["convergence_decision"]) == (
        "completed",
        "solver_ended",
        None,
    )
    assert not (run.root / CONVERGENCE_DECISION).exists()
    result = resolve_palace_result(run.root, expected_handoff_id=run.handoff_id)
    assert result.convergence.verdict.state == "not_declared"
    assert verdict_sentence(result.convergence, result.solver_end) == (
        "Convergence is not declared for this run."
    )
