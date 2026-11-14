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

import shutil
import signal
import sys

import pytest
from palace_stand_in_solver import (
    read_pids,
    wait_until_gone,
)
from palace_stop_support import (
    LONG_S,
    NEVER_SETTLES,
    SETTLES_AT_5,
    decision_of,
    prepared_run,
    rewrite_json,
    run_script,
    runner_processes_left,
)

from scgsim.contracts.palace_run_folder import (
    CONVERGENCE_DECISION,
    CONVERGENCE_RUNNER,
    OUTPUT_DIR,
    RESOURCE_RECORD,
    RETURNED_RECEIPT,
    RUN_METADATA,
    read_json,
    record_for_path,
)
from scgsim.palace import (
    inspect_run_trustworthiness,
    resolve_palace_result,
)
from scgsim.palace.reports.convergence import verdict_card, verdict_sentence

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="the runner's stop is checked through /proc",
)


@pytest.fixture(scope="module")
def stopped(tmp_path_factory):
    """One declared Electrostatic run stopped at convergence, after its pass 4 of 6."""
    run = prepared_run(
        tmp_path_factory.mktemp("stopped"),
        refine_s=[0.05, 0.05, 0.05, LONG_S, 0.05, 0.05],
    )
    run.exit_code = run_script(run)
    return run


def test_the_runner_stops_palace_after_the_first_converged_pass(stopped):
    """Converged at pass 4: the decision is written, the solver's session gets SIGTERM, nothing is left."""
    assert stopped.exit_code == 0, stopped.log
    decision = decision_of(stopped)
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
    assert runner_processes_left(stopped) == []
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
    run = prepared_run(
        tmp_path,
        profile=profile,
        style=style,
        refine_s=[0.05, 0.05, 0.05, LONG_S, 0.05, 0.05],
    )
    script = run.plan.script_path.read_text()
    assert f'python3 {CONVERGENCE_RUNNER} "$(pwd)" -- ' in script
    assert run_script(run) == 0, run.log
    assert decision_of(run).ending == "stopped_at_convergence"
    assert wait_until_gone(read_pids(run.pids_path)) == []
    result = resolve_palace_result(run.root, expected_handoff_id=run.handoff_id)
    assert result.solver_end.stopped_after == 4


def test_an_eigenmode_run_stopped_during_the_next_solve_reads_its_last_pass(tmp_path):
    """Stopped while the next solve has started ``eig.csv`` with its header: the result is still pass 4."""
    run = prepared_run(
        tmp_path,
        problem="Eigenmode",
        solve_s=[0.05, 0.05, 0.05, 0.05, LONG_S, 0.05],
    )
    assert run_script(run) == 0, run.log
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


def test_a_stop_waits_for_the_launcher_behind_a_wrapper_script(tmp_path):
    """Behind a wrapper script the solver's first process ends at once on SIGTERM; its launcher takes longer.

    Palace's own ``palace`` is such a script.  The runner waits until the launcher has brought its ranks
    down itself, so the decision record and the receipt are written after the solver's last process ended;
    the exit code the receipt keeps is the wrapper's, ended by the signal.
    """
    run = prepared_run(
        tmp_path,
        refine_s=[0.05, 0.05, 0.05, LONG_S, 0.05, 0.05],
        behind_wrapper=True,
        shutdown_s=1.0,
    )
    assert run_script(run) == 0, run.log
    launcher_ended = float(run.pids_path.with_name("launcher_ended").read_text())
    decision = decision_of(run)
    assert decision.ending == "stopped_at_convergence"
    assert decision.stop is not None and decision.stop.escalated_utc is None
    assert decision.solver is not None
    assert decision.solver.exit_code == 128 + signal.SIGTERM
    assert (run.root / CONVERGENCE_DECISION).stat().st_mtime >= launcher_ended
    assert (run.root / RETURNED_RECEIPT).stat().st_mtime >= launcher_ended
    assert wait_until_gone(read_pids(run.pids_path)) == []
    resolved = resolve_palace_result(run.root, expected_handoff_id=run.handoff_id)
    assert resolved.convergence.verdict.state == "converged"


def test_a_pass_shorter_than_the_looks_is_read_and_the_stop_comes_after_it(tmp_path):
    """The stop is requested one to two seconds after a pass is complete, so a shorter pass still runs.

    Pass 5 completes a tenth of a second after pass 4, at which the run has converged, and pass 6 takes
    long.  The runner counts both passes at the same look: the verdict is still pass 4, the record says one
    pass was read after it, Palace is stopped after pass 5, and the result is pass 5's tables.
    """
    run = prepared_run(tmp_path, refine_s=[0.05, 0.05, 0.05, 0.05, LONG_S, 0.05])
    assert run_script(run) == 0, run.log
    decision = decision_of(run)
    assert decision.ending == "stopped_at_convergence"
    assert decision.stop is not None and decision.stop.after_pass == 5
    assert decision.checked["verdict"]["pass_number"] == 4
    assert not (run.root / OUTPUT_DIR / "iteration6").exists()
    result = resolve_palace_result(run.root, expected_handoff_id=run.handoff_id)
    verdict = result.convergence.verdict
    assert (verdict.state, verdict.pass_number, verdict.passes_read_after) == (
        "converged",
        4,
        1,
    )
    sentence = verdict_sentence(result.convergence, result.solver_end)
    assert "1 more pass was read after it." in sentence
    assert "SCGSim stopped Palace after pass 5" in sentence
    assert (
        result.tables["terminal-C"].path
        == run.root / OUTPUT_DIR / "iteration5" / "terminal-C.csv"
    )


def test_a_run_that_ends_before_the_first_look_is_read_as_it_ended(tmp_path):
    """Every pass takes a hundredth of a second: Palace has ended before the runner looked, so nothing is stopped.

    The run is complete as Palace ended it; the verdict says at which pass it had converged and how many
    passes came after.
    """
    run = prepared_run(tmp_path, solve_s=[0.01] * 6, refine_s=[0.01] * 6)
    assert run_script(run) == 0, run.log
    decision = decision_of(run)
    assert decision.ending == "solver_ended" and decision.stop is None
    receipt = read_json(run.root / RETURNED_RECEIPT)
    assert (receipt["status"], receipt["ending"], receipt["solver_exit_code"]) == (
        "completed",
        "solver_ended",
        0,
    )
    result = resolve_palace_result(run.root, expected_handoff_id=run.handoff_id)
    verdict = result.convergence.verdict
    assert (verdict.state, verdict.pass_number, verdict.passes_read_after) == (
        "converged",
        4,
        2,
    )
    sentence = verdict_sentence(result.convergence, result.solver_end)
    assert "2 more passes were read after it." in sentence
    assert sentence.endswith("Palace ended by itself at its limit of 6 passes.")
    assert result.tables["terminal-C"].path == run.root / OUTPUT_DIR / "terminal-C.csv"


def test_a_solver_that_ends_by_itself_as_the_runner_decides_ended_by_itself(tmp_path):
    """The run converges at pass 5 and the solver completes its last solve and ends as the runner decides.

    The stop found nothing to end: the solver exits with 0 and leaves its last solve at the top level, so
    the run is recorded as ended by itself, complete, and read as such.
    """
    run = prepared_run(tmp_path, values=SETTLES_AT_5, ends_itself_on_decision=True)
    assert run_script(run) == 0, run.log
    decision = decision_of(run)
    assert decision.ending == "solver_ended" and decision.stop is None
    assert decision.solver is not None and decision.solver.exit_code == 0
    receipt = read_json(run.root / RETURNED_RECEIPT)
    assert (receipt["status"], receipt["ending"], receipt["solver_exit_code"]) == (
        "completed",
        "solver_ended",
        0,
    )
    result = resolve_palace_result(run.root, expected_handoff_id=run.handoff_id)
    verdict = result.convergence.verdict
    assert (verdict.state, verdict.pass_number, verdict.passes_read_after) == (
        "converged",
        5,
        1,
    )
    assert verdict_sentence(result.convergence, result.solver_end).endswith(
        "Palace ended by itself at its limit of 6 passes."
    )
    assert result.tables["terminal-C"].path == run.root / OUTPUT_DIR / "terminal-C.csv"


def test_a_stop_after_the_last_solve_is_complete_gives_that_solve_as_the_result(
    tmp_path,
):
    """The run converges at pass 5, and the stop reaches the solver when its last solve is already complete.

    The stop ended the solver, so the run was stopped at convergence; every pass its limit allows had run,
    and the result is the last one, at the top level.
    """
    run = prepared_run(tmp_path, values=SETTLES_AT_5, finish_s=LONG_S)
    assert run_script(run) == 0, run.log
    decision = decision_of(run)
    assert decision.ending == "stopped_at_convergence"
    assert decision.stop is not None and decision.stop.after_pass == 5
    assert decision.solver is not None and decision.solver.exit_code == 1
    assert read_json(run.root / RETURNED_RECEIPT)["status"] == "completed"
    result = resolve_palace_result(run.root, expected_handoff_id=run.handoff_id)
    verdict = result.convergence.verdict
    assert (verdict.state, verdict.pass_number, verdict.passes_read_after) == (
        "converged",
        5,
        1,
    )
    sentence = verdict_sentence(result.convergence, result.solver_end)
    assert sentence.endswith("SCGSim stopped Palace after pass 5.")
    assert result.tables["terminal-C"].path == run.root / OUTPUT_DIR / "terminal-C.csv"
    assert result.show_run_trustworthiness().completeness == "complete"


def test_a_run_that_does_not_converge_ends_at_its_limit(tmp_path):
    """No stop: Palace runs every pass its limit allows, the receipt is completed, and the report says so."""
    run = prepared_run(tmp_path, values=NEVER_SETTLES)
    assert run_script(run) == 0, run.log
    decision = decision_of(run)
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


# ---- Result reading refuses what disagrees ----------------------------------------------------------------


def test_a_decision_record_edited_after_the_run_is_refused(tmp_path):
    """Edited alone, its hash no longer matches the receipt; edited with the receipt, the recompute differs."""
    run = prepared_run(tmp_path, refine_s=[0.05, 0.05, 0.05, LONG_S, 0.05, 0.05])
    assert run_script(run) == 0, run.log
    path = run.root / CONVERGENCE_DECISION

    def change_a_reading(payload):
        payload["passes"][3]["criteria"][0]["values"][0]["value"] = 1.2e-13

    rewrite_json(path, change_a_reading)
    with pytest.raises(ValueError, match="changed after .* was written"):
        resolve_palace_result(run.root, expected_handoff_id=run.handoff_id)
    rewrite_json(
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


def test_a_decision_record_whose_exit_code_differs_from_the_receipt_is_refused(
    stopped, tmp_path
):
    """The decision's exit code edited, and the receipt's binding of the decision updated to match the edit.

    The readings and the ending still agree; the two records now give two accounts of how the solver
    exited, and result reading refuses them.
    """
    root = tmp_path / "run"
    shutil.copytree(stopped.root, root, symlinks=True)
    shutil.copy(stopped.root.parent / "run.tar.gz", tmp_path / "run.tar.gz")
    rewrite_json(
        root / CONVERGENCE_DECISION,
        lambda decision: decision["solver"].update(exit_code=137),
    )
    rewrite_json(
        root / RETURNED_RECEIPT,
        lambda receipt: receipt.update(
            convergence_decision=record_for_path(root, CONVERGENCE_DECISION)
        ),
    )
    with pytest.raises(
        ValueError,
        match=r"records the solver's exit code 137, and .* states 1",
    ):
        resolve_palace_result(root, expected_handoff_id=stopped.handoff_id)


def test_a_stopped_folder_with_a_receipt_forged_to_say_palace_ended_by_itself_is_refused(
    tmp_path,
):
    """Exit codes rewritten to 0 and the ending to ``solver_ended``: the top level of links says otherwise."""
    run = prepared_run(tmp_path, refine_s=[0.05, 0.05, 0.05, LONG_S, 0.05, 0.05])
    assert run_script(run) == 0, run.log

    def ended_by_itself(record):
        record.update(exit_code=0, solver_exit_code=0, ending="solver_ended")

    rewrite_json(run.root / RETURNED_RECEIPT, ended_by_itself)
    for record in (RUN_METADATA, RESOURCE_RECORD):
        rewrite_json(run.root / record, lambda payload: payload.update(exit_code=0))
    with pytest.raises(
        ValueError,
        match=r"is the folder of a solver ended during its adaptive loop \(the final result's tables are "
        r"links into iteration4\), and metadata/palace_returned_run_receipt.json says the solver ended by itself",
    ):
        resolve_palace_result(run.root, expected_handoff_id=run.handoff_id)


def test_final_tables_of_its_own_with_a_receipt_saying_stopped_are_refused(tmp_path):
    """A run that ended by itself, its receipt forged to say it was stopped: the top-level solve says otherwise."""
    run = prepared_run(tmp_path, values=NEVER_SETTLES)
    assert run_script(run) == 0, run.log
    rewrite_json(
        run.root / RETURNED_RECEIPT,
        lambda receipt: receipt.update(ending="stopped_at_convergence"),
    )
    with pytest.raises(
        ValueError,
        match="records the ending 'solver_ended', and .* states 'stopped_at_convergence'",
    ):
        resolve_palace_result(run.root, expected_handoff_id=run.handoff_id)


def test_a_plan_without_its_decision_record_is_refused(tmp_path):
    """The receipt of this version binds the decision of a declared run; without it the run is not read."""
    run = prepared_run(tmp_path, values=NEVER_SETTLES)
    assert run_script(run) == 0, run.log
    (run.root / CONVERGENCE_DECISION).unlink()
    with pytest.raises(
        FileNotFoundError, match="is missing, and the run has a convergence plan"
    ):
        resolve_palace_result(run.root, expected_handoff_id=run.handoff_id)


def test_a_decision_record_without_a_plan_is_refused(tmp_path):
    """A run with no declaration has no runner, so a decision record in its folder is refused."""
    run = prepared_run(tmp_path, declare=False)
    assert run_script(run) == 0, run.log
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
        rewrite_json(
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
    run = prepared_run(tmp_path, declare=False)
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
    assert run_script(run) == 0, run.log
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
