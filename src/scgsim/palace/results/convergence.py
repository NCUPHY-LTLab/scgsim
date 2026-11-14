"""Read the declared convergence of a returned run: its sealed plan, then the check of the passes it holds.

``read_plan`` reads ``metadata/palace_convergence_plan.json`` only when the handoff sealed it, refuses a plan
that is present but unsealed or changed since, and returns None for a run that declared nothing, before any
pass is read.  ``check_passes`` runs ``scgsim.contracts.palace_convergence``'s check on the passes its
``returned_passes`` finds in ``results/palace``, the enumeration ``collect_amr_passes`` uses too; the check
reads every number from the pass folders' tables itself.  Without a plan nothing in ``results/palace`` is
read, so a run without a declaration costs what it did before.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scgsim.contracts.palace_convergence import (
    ConvergenceCheck,
    ConvergencePlan,
    check_convergence,
    returned_passes,
)
from scgsim.contracts.palace_run_folder import (
    CONVERGENCE_PLAN,
    HANDOFF_METADATA,
    OUTPUT_DIR,
    read_json,
    sha256,
)
from scgsim.palace.run.archive import RETURNED_RUN_FOLDER
from scgsim.refusal import Declared


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


__all__ = ["check_passes", "read_plan"]
