"""Verify report-readable receipts and classify attempted runs and selected snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple

from scgsim.contracts.palace_run_folder import (
    NATIVE_METADATA_NAME,
    confined_path,
    palace_problem,
    table_name,
)
from scgsim.palace.results.artifacts import (
    FileHash,
    compute_hash_entry,
    extract_hash_map,
)
from scgsim.palace.results.inspection_records import (
    CollectedSnapshots,
    FailureCategory,
    PalaceFailureDiagnosis,
    PalaceResultSelection,
)
from scgsim.refusal import refusal


def validate_inspection_receipt(
    root: Path,
    handoff: dict[str, Any],
    receipt: dict[str, Any] | None,
) -> frozenset[str]:
    """Verify available input, output and log identities before selecting report snapshots."""
    if receipt is None:
        return frozenset()
    _validate_inspection_identity(handoff, receipt)

    input_entries = receipt.get("input_hashes")
    handoff_entries = handoff.get("hashes")
    if not isinstance(input_entries, list) or not isinstance(handoff_entries, list):
        raise TypeError(
            refusal(
                "returned receipt and handoff input hashes must be lists",
                found={
                    "input_hashes": type(input_entries).__name__,
                    "hashes": type(handoff_entries).__name__,
                },
            )
        )
    receipt_inputs = extract_hash_map(input_entries)
    handoff_inputs = extract_hash_map(handoff_entries)
    if receipt_inputs != handoff_inputs:
        raise ValueError(
            refusal(
                "returned receipt input hashes do not match handoff metadata",
                found=receipt_inputs,
                expected=handoff_inputs,
            )
        )
    for relative, expected in receipt_inputs.items():
        observed = compute_hash_entry(confined_path(root, relative))
        if observed != expected:
            raise ValueError(f"returned receipt input hash mismatch for {relative}.")

    output_entries = receipt.get("output_files")
    if not isinstance(output_entries, list):
        raise TypeError(
            refusal(
                "returned receipt output_files must be a list", found=output_entries
            )
        )
    verified: set[str] = set()
    seen: set[str] = set()
    for index, entry in enumerate(output_entries):
        relative = _verify_receipt_file_record(root, entry, f"output_files[{index}]")
        if relative in seen:
            raise ValueError(f"returned receipt contains duplicate path {relative!r}.")
        seen.add(relative)
        if entry["present"] is True:
            verified.add(relative)

    log = receipt.get("log")
    if not isinstance(log, dict):
        raise TypeError(refusal("returned receipt log must be a mapping", found=log))
    log_relative = _verify_receipt_file_record(root, log, "log")
    if log["present"] is True:
        verified.add(log_relative)
    return frozenset(verified)


def _verify_receipt_file_record(root: Path, entry: Any, label: str) -> str:
    """Check one recorded artifact, keeping absent and present files distinct."""
    if not isinstance(entry, dict):
        raise TypeError(f"returned receipt {label} must be a mapping.")
    relative = entry.get("path")
    present = entry.get("present")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"returned receipt {label} path must be non-empty.")
    if not isinstance(present, bool):
        raise TypeError(f"returned receipt {label} present must be bool.")
    observed = compute_hash_entry(confined_path(root, relative))
    if present:
        size = entry.get("bytes")
        sha256 = entry.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"returned receipt {label} bytes must be non-negative.")
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise ValueError(f"returned receipt {label} sha256 must have 64 digits.")
        if observed != FileHash(size, sha256):
            raise ValueError(f"returned receipt output hash mismatch for {relative}.")
    elif entry.get("bytes") is not None or entry.get("sha256") is not None:
        raise ValueError(f"returned receipt missing {label} must not carry a hash.")
    elif observed != FileHash(None, None):
        raise ValueError(f"returned receipt missing output now exists: {relative}.")
    return relative


def _receipt_exit_code(receipt: dict[str, Any], field: str) -> int:
    """Read an integer exit code while rejecting boolean receipt values."""
    value = receipt.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"returned receipt {field} must be an integer.")
    return value


@dataclass(frozen=True)
class _ExecutionState:
    """Which stage a returned run reached and the exit codes the receipt records for it."""

    stage: Literal["config_compatibility", "solver"] | None
    solver_invoked: bool | None
    preflight_exit_code: int | None
    solver_exit_code: int | None
    exit_code: int
    tee_exit_code: int


def _receipt_execution_state(receipt: dict[str, Any]) -> _ExecutionState:
    """Interpret receipt execution stages without conflating failed preflight and solver invocation."""
    exit_code = _receipt_exit_code(receipt, "exit_code")
    tee_exit_code = _receipt_exit_code(receipt, "tee_exit_code")
    stage = receipt.get("execution_stage")
    if stage is None:
        return _ExecutionState(
            None,
            None,
            None,
            _receipt_exit_code(receipt, "solver_exit_code"),
            exit_code,
            tee_exit_code,
        )
    if stage not in {"config_compatibility", "solver"}:
        raise ValueError(
            f"returned receipt execution_stage is invalid: {stage!r}; the recorder writes "
            "'config_compatibility' or 'solver'."
        )
    solver_invoked = receipt.get("solver_invoked")
    if not isinstance(solver_invoked, bool):
        raise TypeError(
            f"returned receipt solver_invoked must be boolean; got {solver_invoked!r}."
        )
    preflight_exit_code = _receipt_exit_code(receipt, "preflight_exit_code")
    solver_exit_code = receipt.get("solver_exit_code")
    if stage == "config_compatibility":
        if solver_invoked or solver_exit_code is not None:
            raise ValueError(
                "config compatibility failure must record solver not invoked: the receipt records "
                f"solver_invoked={solver_invoked!r}, solver_exit_code={solver_exit_code!r}."
            )
    else:
        if not solver_invoked or preflight_exit_code != 0:
            raise ValueError(
                "solver execution requires a successful config compatibility preflight: the receipt "
                f"records solver_invoked={solver_invoked!r}, preflight_exit_code={preflight_exit_code!r}."
            )
        solver_exit_code = _receipt_exit_code(receipt, "solver_exit_code")
    return _ExecutionState(
        stage,
        solver_invoked,
        preflight_exit_code,
        solver_exit_code,
        exit_code,
        tee_exit_code,
    )


def result_selection(
    *,
    root: Path,
    problem: str,
    collected: CollectedSnapshots,
    receipt_paths: frozenset[str],
) -> PalaceResultSelection:
    """Choose final or iteration data from completion state and receipt-bound readable snapshots."""
    if not collected.passes:
        return PalaceResultSelection(
            final_snapshot_status=collected.final_snapshot_status,
            selected_source=None,
            selected_path=None,
            selected_pass_index=None,
            reason="no_complete_snapshot",
            integrity="unavailable",
        )
    selected = collected.passes[-1]
    reason = (
        "final_snapshot"
        if selected.source == "final"
        else "latest_complete_iteration_after_failed_attempt"
    )
    artifact_paths = _snapshot_artifact_paths(selected.path, problem)
    relative_paths = {path.relative_to(root).as_posix() for path in artifact_paths}
    integrity: Literal["receipt_bound", "observed_unsealed"] = (
        "receipt_bound"
        if relative_paths and relative_paths.issubset(receipt_paths)
        else "observed_unsealed"
    )
    return PalaceResultSelection(
        final_snapshot_status=collected.final_snapshot_status,
        selected_source=selected.source,
        selected_path=selected.path.relative_to(root),
        selected_pass_index=selected.pass_index,
        reason=reason,
        integrity=integrity,
    )


def _snapshot_artifact_paths(path: Path, problem: str) -> tuple[Path, ...]:
    """Enumerate the problem-specific files that establish one complete result snapshot."""
    names = (
        *(table_name(family) for family in palace_problem(problem).snapshot_tables),
        NATIVE_METADATA_NAME,
    )
    return tuple(candidate for name in names if (candidate := path / name).is_file())


def failure_diagnosis(
    root: Path,
    receipt: dict[str, Any] | None,
    receipt_paths: frozenset[str],
) -> PalaceFailureDiagnosis | None:
    """Describe the failed attempt from receipt status and verified log evidence."""
    if receipt is None or receipt.get("status") != "failed":
        return None
    state = _receipt_execution_state(receipt)
    evidence: list[dict[str, Any]] = [
        {
            "source": "returned_receipt",
            "exit_code": state.exit_code,
            "execution_stage": state.stage,
            "solver_invoked": state.solver_invoked,
            "preflight_exit_code": state.preflight_exit_code,
            "solver_exit_code": state.solver_exit_code,
            "tee_exit_code": state.tee_exit_code,
        }
    ]
    oom_evidence = (
        None
        if state.stage == "config_compatibility"
        else _slurm_oom_evidence(root, receipt, receipt_paths)
    )
    cause = _failure_cause(state, oom_evidence is not None)
    if cause.category == "out_of_memory" and oom_evidence is not None:
        evidence.append(oom_evidence)
    return PalaceFailureDiagnosis(
        category=cause.category,
        exit_code=state.exit_code,
        execution_stage=state.stage,
        solver_invoked=state.solver_invoked,
        preflight_exit_code=state.preflight_exit_code,
        solver_exit_code=state.solver_exit_code,
        tee_exit_code=state.tee_exit_code,
        summary=cause.summary,
        evidence=tuple(evidence),
    )


class _FailureCause(NamedTuple):
    """The category a failed run falls in and the sentence that tells a person why."""

    category: FailureCategory
    summary: str


def _failure_cause(state: _ExecutionState, scheduler_oom: bool) -> _FailureCause:
    """Classify a failure: the preflight first, then scheduler memory, a kill, the solver, the capture."""
    if state.stage == "config_compatibility" and state.preflight_exit_code != 0:
        return _FailureCause(
            "config_compatibility",
            "Palace config compatibility preflight exited with status "
            f"{state.preflight_exit_code}; the solver was not invoked.",
        )
    if state.stage == "config_compatibility" and state.tee_exit_code != 0:
        return _FailureCause(
            "output_capture_error",
            "Config compatibility output capture exited with status "
            f"{state.tee_exit_code}; the solver was not invoked.",
        )
    if state.stage == "config_compatibility":
        return _FailureCause(
            "unknown",
            "The run failed during config compatibility preflight; the solver "
            "was not invoked.",
        )
    if scheduler_oom:
        return _FailureCause("out_of_memory", "Slurm reported an out-of-memory event.")
    if state.solver_exit_code == 137 or state.exit_code == 137:
        return _FailureCause(
            "signal_killed",
            "The solver was killed with exit 137; out of memory is possible but "
            "not confirmed by scheduler evidence.",
        )
    if state.solver_exit_code != 0:
        return _FailureCause(
            "solver_error",
            f"The Palace solver exited with status {state.solver_exit_code}.",
        )
    if state.tee_exit_code != 0:
        return _FailureCause(
            "output_capture_error",
            f"Solver output capture exited with status {state.tee_exit_code}.",
        )
    return _FailureCause(
        "unknown", "The returned receipt reports failure without a classified cause."
    )


def _slurm_oom_evidence(
    root: Path,
    receipt: dict[str, Any],
    receipt_paths: frozenset[str],
) -> dict[str, Any] | None:
    """Find scheduler out-of-memory markers only in receipt-bound log files."""
    log = receipt.get("log")
    if not isinstance(log, dict):
        return None
    relative = log.get("path")
    if not isinstance(relative, str) or relative not in receipt_paths:
        return None
    path = confined_path(root, relative)
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            lowered = line.lower()
            if "slurm" not in lowered:
                continue
            if (
                "oom_kill event" in lowered
                or "oom-kill event" in lowered
                or "out_of_memory" in lowered
            ):
                return {
                    "source": "hash_verified_log",
                    "path": relative,
                    "marker": "slurm_out_of_memory",
                    "sha256": log.get("sha256"),
                }
    return None


def selection_payload(selection: PalaceResultSelection) -> dict[str, Any]:
    """Serialize snapshot choice and integrity fields for the report metadata."""
    return {
        "final_snapshot_status": selection.final_snapshot_status,
        "selected_source": selection.selected_source,
        "selected_path": (
            None if selection.selected_path is None else str(selection.selected_path)
        ),
        "selected_pass_index": selection.selected_pass_index,
        "reason": selection.reason,
        "integrity": selection.integrity,
    }


def failure_payload(
    failure: PalaceFailureDiagnosis | None,
) -> dict[str, Any] | None:
    """Serialize an available failure diagnosis without manufacturing absent evidence."""
    if failure is None:
        return None
    return {
        "category": failure.category,
        "exit_code": failure.exit_code,
        "execution_stage": failure.execution_stage,
        "solver_invoked": failure.solver_invoked,
        "preflight_exit_code": failure.preflight_exit_code,
        "solver_exit_code": failure.solver_exit_code,
        "tee_exit_code": failure.tee_exit_code,
        "summary": failure.summary,
        "evidence": failure.evidence,
    }


def _validate_inspection_identity(handoff: Any, receipt: Any) -> None:
    """Match receipt route, problem, handoff and thin-film identity before inspecting files."""
    for field in ("handoff_id", "route", "problem"):
        value = receipt.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"returned receipt {field} must be a non-empty string.")
        if field in handoff and handoff.get(field) != value:
            raise ValueError(
                refusal(
                    f"returned receipt {field} does not match handoff metadata",
                    found=value,
                    expected=handoff.get(field),
                )
            )
    if receipt.get("route_a_thin_film") != handoff.get("route_a_thin_film"):
        raise ValueError(
            refusal(
                "returned receipt Route-A thin-film identity mismatch",
                found=receipt.get("route_a_thin_film"),
                expected=handoff.get("route_a_thin_film"),
            )
        )
    if not isinstance(receipt.get("status"), str) or not receipt["status"]:
        raise ValueError(
            refusal(
                "returned receipt status must be a non-empty string",
                found=receipt.get("status"),
            )
        )
    _receipt_execution_state(receipt)
