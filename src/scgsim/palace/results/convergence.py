"""Read the declared convergence of a returned run: its sealed plan, the check of its passes, and how it ended.

``read_plan`` reads ``metadata/palace_convergence_plan.json`` only when the handoff sealed it, refuses a plan
that is present but unsealed or changed since, and returns None for a run that declared nothing, before any
pass is read.  ``check_passes`` runs ``scgsim.contracts.palace_convergence``'s check on the passes its
``returned_passes`` finds in ``results/palace``, the enumeration ``collect_amr_passes`` uses too; the check
reads every number from the pass folders' tables itself.  Without a plan nothing in ``results/palace`` is
read but its top level, so a run without a declaration costs what it did before.

``returned_convergence`` is what result reading checks besides: the check computed again from the returned
files must be the one the runner recorded in its decision record, the decision must be the one the receipt
binds, and the results folder must hold what the receipt's ending says the solver left: a solve of its own
at the top level when it ended by itself, none when the runner stopped it.  It also says where the run's
result is: the top level, or the last pass of a run stopped at convergence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scgsim.contracts.palace_convergence import (
    ConvergenceCheck,
    ConvergenceDecision,
    ConvergencePlan,
    check_convergence,
    check_disagreement,
    completed_stop_problem,
    recorded_check,
    returned_passes,
    top_level_state,
)
from scgsim.contracts.palace_run_folder import (
    CONVERGENCE_DECISION,
    CONVERGENCE_PLAN,
    HANDOFF_METADATA,
    OUTPUT_DIR,
    RETURNED_RECEIPT,
    RunEnding,
    read_json,
    sha256,
)
from scgsim.palace.run.archive import RETURNED_RUN_FOLDER
from scgsim.refusal import Declared


@dataclass(frozen=True)
class SolverEnd:
    """How a returned run's solver ended, as the report says it beside the verdict.

    ``ending`` is the receipt's (``RunEnding``), None for a folder without a receipt.  ``stopped_after`` is the
    pass after which the runner stopped the solver, None when it did not.  ``passes_allowed`` is the number of
    solves the config's pass limit allows: ``amr_max_passes`` (Palace's ``MaxIts``) plus the first solve.
    """

    ending: RunEnding | None
    stopped_after: int | None
    passes_allowed: int | None


@dataclass(frozen=True)
class ReturnedConvergence:
    """The convergence of a returned run checked against its receipt, and where its result tables are.

    ``result_folder`` is relative to the run folder: ``results/palace`` for a solver that ended by itself, the
    folder of the last pass for one stopped at convergence.
    """

    check: ConvergenceCheck
    decision: ConvergenceDecision | None
    end: SolverEnd
    result_folder: str


def read_plan(
    root: Path, handoff: dict[str, Any], problem: str
) -> ConvergencePlan | None:
    """The sealed convergence plan of the run in ``root``, or None when the run declared none.

    Raises:
        ValueError: the plan is present but the handoff did not seal it, it changed since, the handoff
            seals a plan the folder lacks, or the plan is for another problem or of another schema.
    """
    plan_path = root / CONVERGENCE_PLAN
    sealed = _sealed_plan(handoff)
    if sealed is None and not plan_path.is_file():
        return None
    _check_sealed(plan_path, sealed)
    plan = ConvergencePlan.from_payload(read_json(plan_path))
    if plan.problem != problem:
        raise ValueError(
            Declared(f"{CONVERGENCE_PLAN} in {root}", RETURNED_RUN_FOLDER).refusal(
                f"is a plan for {plan.problem}, and the run is {problem}"
            )
        )
    return plan


def check_passes(plan: ConvergencePlan | None, root: Path) -> ConvergenceCheck:
    """The check of ``plan`` on the passes of the run folder ``root``, or ``not declared`` without a plan."""
    if plan is None:
        return ConvergenceCheck.not_declared()
    returned = returned_passes(root / OUTPUT_DIR)
    return check_convergence(plan, returned.passes, returned.set_aside)


def returned_convergence(
    root: Path,
    handoff: dict[str, Any],
    problem: str,
    receipt: dict[str, Any],
    passes_allowed: int | None,
) -> ReturnedConvergence:
    """The convergence of a completed run, after checking it against the receipt, the decision and the files.

    Raises:
        ValueError: the folder's top level contradicts the receipt's ending, a decision record is present
            without a plan or missing with one, it is not the record the receipt binds or is for another plan,
            its passes or verdict differ from those read from the returned files again, or it does not
            record a completed stop although the receipt says the solver was stopped at convergence.
    """
    ending = receipt.get("ending")
    _check_top_level(root, ending)
    plan = read_plan(root, handoff, problem)
    if plan is None:
        _check_undeclared(root, receipt, ending)
        return ReturnedConvergence(
            ConvergenceCheck.not_declared(),
            None,
            SolverEnd(ending, None, passes_allowed),
            OUTPUT_DIR,
        )
    decision = _bound_decision(root, receipt, handoff)
    returned = returned_passes(root / OUTPUT_DIR)
    check = check_convergence(plan, returned.passes, returned.set_aside)
    disagreement = check_disagreement(decision.checked, recorded_check(check, root))
    if disagreement is not None:
        raise ValueError(
            Declared(f"{CONVERGENCE_DECISION} in {root}", RETURNED_RUN_FOLDER).refusal(
                f"does not match the passes read again from {OUTPUT_DIR}: {disagreement}. The runner "
                "recorded what it read during the solve; the returned files or the record changed since"
            )
        )
    if decision.ending != ending:
        raise ValueError(
            Declared(f"{CONVERGENCE_DECISION} in {root}", RETURNED_RUN_FOLDER).refusal(
                f"records the ending {decision.ending!r}, and {RETURNED_RECEIPT} states {ending!r}"
            )
        )
    recorded_exit = decision.solver.exit_code if decision.solver is not None else None
    if recorded_exit != receipt.get("solver_exit_code"):
        raise ValueError(
            Declared(f"{CONVERGENCE_DECISION} in {root}", RETURNED_RUN_FOLDER).refusal(
                f"records the solver's exit code {recorded_exit!r}, and {RETURNED_RECEIPT} states "
                f"{receipt.get('solver_exit_code')!r}"
            )
        )
    if ending == "stopped_at_convergence":
        problem_text = completed_stop_problem(decision)
        if problem_text is not None:
            raise ValueError(
                Declared(
                    f"{CONVERGENCE_DECISION} in {root}", RETURNED_RUN_FOLDER
                ).refusal(f"does not record a completed stop: {problem_text}")
            )
        result_folder = decision.result_folder or OUTPUT_DIR
    else:
        if returned.final != "pass":
            raise ValueError(
                Declared(f"{OUTPUT_DIR} in {root}", RETURNED_RUN_FOLDER).refusal(
                    f"holds no solve of its own at its top level ({returned.final}: "
                    f"{returned.final_reason}), and {RETURNED_RECEIPT} says the solver ended by itself"
                )
            )
        result_folder = OUTPUT_DIR
    stopped_after = decision.stop.after_pass if decision.stop is not None else None
    return ReturnedConvergence(
        check,
        decision,
        SolverEnd(ending, stopped_after, passes_allowed),
        result_folder,
    )


def passes_allowed(config: dict[str, Any] | None) -> int | None:
    """The solves the config's pass limit allows: Palace's ``Model.Refinement.MaxIts`` plus the first solve."""
    model = config.get("Model") if isinstance(config, dict) else None
    refinement = model.get("Refinement") if isinstance(model, dict) else None
    limit = refinement.get("MaxIts") if isinstance(refinement, dict) else None
    return limit + 1 if type(limit) is int and limit >= 0 else None


def inspected_end(
    root: Path, receipt: dict[str, Any] | None, passes_allowed: int | None
) -> SolverEnd:
    """How the solver of a returned folder ended, for the trust report of a folder that may be partial.

    The ending is the receipt's; the pass the runner stopped after is the decision record's, read only when
    the receipt binds it and its hash still matches.  Nothing is refused here: ``resolve_palace_result``
    checks the record.
    """
    ending = receipt.get("ending") if receipt is not None else None
    record = receipt.get("convergence_decision") if receipt is not None else None
    path = root / CONVERGENCE_DECISION
    stopped_after = None
    if (
        isinstance(record, dict)
        and path.is_file()
        and record.get("sha256") == sha256(path)
    ):
        try:
            stop = ConvergenceDecision.from_payload(read_json(path)).stop
        except (TypeError, ValueError):
            stop = None
        stopped_after = stop.after_pass if stop is not None else None
    return SolverEnd(
        ending if isinstance(ending, str) else None,  # type: ignore[arg-type]
        stopped_after,
        passes_allowed,
    )


def _check_top_level(root: Path, ending: Any) -> None:
    """Refuse a top level that contradicts the receipt's ending, for every run with or without a plan."""
    results = root / OUTPUT_DIR
    state, reason = top_level_state(results)
    if ending == "solver_ended" and state in ("linked", "unfinished"):
        raise ValueError(
            Declared(f"{results}", RETURNED_RUN_FOLDER).refusal(
                f"is the folder of a solver ended during its adaptive loop ({reason}), and "
                f"{RETURNED_RECEIPT} says the solver ended by itself"
            )
        )


def _check_undeclared(root: Path, receipt: dict[str, Any], ending: Any) -> None:
    """A run without a plan has no decision record and its solver ended by itself."""
    if (root / CONVERGENCE_DECISION).exists() or receipt.get(
        "convergence_decision"
    ) is not None:
        raise ValueError(
            Declared(f"{CONVERGENCE_DECISION} in {root}", RETURNED_RUN_FOLDER).refusal(
                f"is present, and {HANDOFF_METADATA} sealed no convergence plan: no runner ran for this "
                "handoff"
            )
        )
    if ending != "solver_ended":
        raise ValueError(
            Declared(f"{RETURNED_RECEIPT} in {root}", RETURNED_RUN_FOLDER).refusal(
                f"states the ending {ending!r}, which only a run with a convergence plan can have"
            )
        )


def _bound_decision(
    root: Path, receipt: dict[str, Any], handoff: dict[str, Any]
) -> ConvergenceDecision:
    """The runner's decision record, which the receipt binds by hash, for the plan the handoff sealed."""
    path = root / CONVERGENCE_DECISION
    record = receipt.get("convergence_decision")
    subject = Declared(f"{CONVERGENCE_DECISION} in {root}", RETURNED_RUN_FOLDER)
    if not path.is_file():
        raise FileNotFoundError(
            subject.refusal(
                f"is missing, and the run has a convergence plan: the runner writes it, and {RETURNED_RECEIPT} "
                "of this version binds it"
            )
        )
    if not isinstance(record, dict) or record.get("present") is not True:
        raise ValueError(
            subject.refusal(f"is not the decision record {RETURNED_RECEIPT} binds")
        )
    if (
        record.get("sha256") != sha256(path)
        or record.get("bytes") != path.stat().st_size
    ):
        raise ValueError(
            subject.refusal(
                f"changed after {RETURNED_RECEIPT} was written: its size or SHA-256 is not the one the "
                "receipt binds"
            )
        )
    decision = ConvergenceDecision.from_payload(read_json(path))
    sealed = _sealed_plan(handoff)
    if sealed is None or decision.plan_sha256 != sealed.get("sha256"):
        raise ValueError(
            subject.refusal(
                f"is for another plan than the one {HANDOFF_METADATA} sealed (plan SHA-256 "
                f"{decision.plan_sha256})"
            )
        )
    return decision


def _sealed_plan(handoff: dict[str, Any]) -> dict[str, Any] | None:
    """The handoff's hash record of the plan, None when it sealed none."""
    hashes = handoff.get("hashes")
    return next(
        (
            entry
            for entry in (hashes if isinstance(hashes, list) else ())
            if isinstance(entry, dict) and entry.get("path") == CONVERGENCE_PLAN
        ),
        None,
    )


def _check_sealed(plan_path: Path, sealed: dict[str, Any] | None) -> None:
    """Refuse a plan the handoff did not seal, one it sealed that is gone, and one changed since."""
    if sealed is None:
        raise ValueError(
            Declared(f"{plan_path}", RETURNED_RUN_FOLDER).refusal(
                f"is not among the inputs {HANDOFF_METADATA} sealed, so it is not the plan this run was "
                "handed off with"
            )
        )
    if not plan_path.is_file():
        raise FileNotFoundError(
            Declared(f"{plan_path}", RETURNED_RUN_FOLDER).refusal(
                f"is missing, and {HANDOFF_METADATA} sealed it"
            )
        )
    if sha256(plan_path) != sealed.get("sha256"):
        raise ValueError(
            Declared(f"{plan_path}", RETURNED_RUN_FOLDER).refusal(
                f"changed after the handoff: its SHA-256 is not the one {HANDOFF_METADATA} sealed"
            )
        )


__all__ = [
    "ReturnedConvergence",
    "SolverEnd",
    "check_passes",
    "inspected_end",
    "passes_allowed",
    "read_plan",
    "returned_convergence",
]
