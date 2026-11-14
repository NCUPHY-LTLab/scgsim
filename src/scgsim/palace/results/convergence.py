"""Read the declared convergence of a returned run: its bound plan, checked on the passes the folder holds.

``read_convergence`` reads ``metadata/palace_convergence_plan.json`` only when the handoff sealed it, refuses a
plan that is present but unsealed or changed since, and runs ``scgsim.contracts.palace_convergence``'s
check on the passes in the order ``collect_amr_passes`` gives them.  The check reads every number from the
pass folders' tables itself; the snapshots give only the pass numbers and their folders.  A run folder
without a plan has convergence ``not declared``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from scgsim.contracts.palace_convergence import (
    ConvergenceCheck,
    ConvergencePlan,
    PassFolder,
    check_convergence,
)
from scgsim.contracts.palace_run_folder import (
    CONVERGENCE_PLAN,
    HANDOFF_METADATA,
    read_json,
    sha256,
)
from scgsim.palace.results.inspection_records import AmrPassSnapshot
from scgsim.palace.run.archive import RETURNED_RUN_FOLDER
from scgsim.refusal import Declared


def read_convergence(
    root: Path,
    handoff: dict[str, Any],
    problem: str,
    passes: Sequence[AmrPassSnapshot],
) -> ConvergenceCheck:
    """The convergence check of the run in ``root`` on ``passes``, or ``not declared`` without a plan.

    Raises:
        ValueError: the plan is present but the handoff did not seal it, it changed since, the handoff
            seals a plan the folder lacks, or the plan is for another problem or of another schema.
    """
    plan_path = root / CONVERGENCE_PLAN
    sealed = _sealed_plan(handoff)
    if sealed is None and not plan_path.is_file():
        return ConvergenceCheck.not_declared()
    _check_sealed(plan_path, sealed)
    plan = ConvergencePlan.from_payload(read_json(plan_path))
    if plan.problem != problem:
        raise ValueError(
            Declared(f"{CONVERGENCE_PLAN} in {root}", RETURNED_RUN_FOLDER).refusal(
                f"is a plan for {plan.problem}, and the run is {problem}"
            )
        )
    return check_convergence(
        plan,
        [PassFolder(snapshot.pass_number, snapshot.path) for snapshot in passes],
    )


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


__all__ = ["read_convergence"]
