"""Verify report-readable receipts and classify attempted runs and selected snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from scgsim.palace.results.artifacts import (
    compute_hash_entry,
    confined_path,
    extract_hash_map,
)
from scgsim.palace.results.report_models import (
    CollectedSnapshots,
    PalaceFailureDiagnosis,
    PalaceResultSelection,
)


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
        raise TypeError("returned receipt and handoff input hashes must be lists.")
    receipt_inputs = extract_hash_map(input_entries)
    handoff_inputs = extract_hash_map(handoff_entries)
    if receipt_inputs != handoff_inputs:
        raise ValueError("returned receipt input hashes do not match handoff metadata.")
    for relative, expected in receipt_inputs.items():
        observed = compute_hash_entry(confined_path(root, relative))
        if observed != expected:
            raise ValueError(f"returned receipt input hash mismatch for {relative}.")

    output_entries = receipt.get("output_files")
    if not isinstance(output_entries, list):
        raise TypeError("returned receipt output_files must be a list.")
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
        raise TypeError("returned receipt log must be a mapping.")
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
        if observed != (size, sha256):
            raise ValueError(f"returned receipt output hash mismatch for {relative}.")
    elif entry.get("bytes") is not None or entry.get("sha256") is not None:
        raise ValueError(f"returned receipt missing {label} must not carry a hash.")
    elif observed != (None, None):
        raise ValueError(f"returned receipt missing output now exists: {relative}.")
    return relative


def _receipt_exit_code(receipt: dict[str, Any], field: str) -> int:
    """Read an integer exit code while rejecting boolean receipt values."""
    value = receipt.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"returned receipt {field} must be an integer.")
    return value


def _receipt_execution_state(
    receipt: dict[str, Any],
) -> tuple[
    Literal["config_compatibility", "solver"] | None,
    bool | None,
    int | None,
    int | None,
    int,
    int,
]:
    """Interpret receipt execution stages without conflating failed preflight and solver invocation."""
    exit_code = _receipt_exit_code(receipt, "exit_code")
    tee_exit_code = _receipt_exit_code(receipt, "tee_exit_code")
    stage = receipt.get("execution_stage")
    if stage is None:
        return (
            None,
            None,
            None,
            _receipt_exit_code(receipt, "solver_exit_code"),
            exit_code,
            tee_exit_code,
        )
    if stage not in {"config_compatibility", "solver"}:
        raise ValueError("returned receipt execution_stage is invalid.")
    solver_invoked = receipt.get("solver_invoked")
    if not isinstance(solver_invoked, bool):
        raise TypeError("returned receipt solver_invoked must be boolean.")
    preflight_exit_code = _receipt_exit_code(receipt, "preflight_exit_code")
    solver_exit_code = receipt.get("solver_exit_code")
    if stage == "config_compatibility":
        if solver_invoked or solver_exit_code is not None:
            raise ValueError(
                "config compatibility failure must record solver not invoked."
            )
    else:
        if not solver_invoked or preflight_exit_code != 0:
            raise ValueError(
                "solver execution requires a successful config compatibility preflight."
            )
        solver_exit_code = _receipt_exit_code(receipt, "solver_exit_code")
    return (
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
        (
            "eig.csv",
            "port-EPR.csv",
            "surface-Q.csv",
            "error-indicators.csv",
            "palace.json",
        )
        if problem == "Eigenmode"
        else ("terminal-C.csv", "surface-Q.csv", "error-indicators.csv", "palace.json")
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
    (
        execution_stage,
        solver_invoked,
        preflight_exit,
        solver_exit,
        exit_code,
        tee_exit,
    ) = _receipt_execution_state(receipt)
    evidence: list[dict[str, Any]] = [
        {
            "source": "returned_receipt",
            "exit_code": exit_code,
            "execution_stage": execution_stage,
            "solver_invoked": solver_invoked,
            "preflight_exit_code": preflight_exit,
            "solver_exit_code": solver_exit,
            "tee_exit_code": tee_exit,
        }
    ]
    oom_evidence = (
        None
        if execution_stage == "config_compatibility"
        else _slurm_oom_evidence(root, receipt, receipt_paths)
    )
    if execution_stage == "config_compatibility" and preflight_exit != 0:
        category = "config_compatibility"
        summary = (
            "Palace config compatibility preflight exited with status "
            f"{preflight_exit}; the solver was not invoked."
        )
    elif execution_stage == "config_compatibility" and tee_exit != 0:
        category = "output_capture_error"
        summary = (
            "Config compatibility output capture exited with status "
            f"{tee_exit}; the solver was not invoked."
        )
    elif execution_stage == "config_compatibility":
        category = "unknown"
        summary = (
            "The run failed during config compatibility preflight; the solver "
            "was not invoked."
        )
    elif oom_evidence is not None:
        evidence.append(oom_evidence)
        category = "out_of_memory"
        summary = "Slurm reported an out-of-memory event."
    elif solver_exit == 137 or exit_code == 137:
        category = "signal_killed"
        summary = (
            "The solver was killed with exit 137; out of memory is possible but "
            "not confirmed by scheduler evidence."
        )
    elif solver_exit != 0:
        category = "solver_error"
        summary = f"The Palace solver exited with status {solver_exit}."
    elif tee_exit != 0:
        category = "output_capture_error"
        summary = f"Solver output capture exited with status {tee_exit}."
    else:
        category = "unknown"
        summary = "The returned receipt reports failure without a classified cause."
    return PalaceFailureDiagnosis(
        category=category,
        exit_code=exit_code,
        execution_stage=execution_stage,
        solver_invoked=solver_invoked,
        preflight_exit_code=preflight_exit,
        solver_exit_code=solver_exit,
        tee_exit_code=tee_exit,
        summary=summary,
        evidence=tuple(evidence),
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
                f"returned receipt {field} does not match handoff metadata."
            )
    if receipt.get("route_a_thin_film") != handoff.get("route_a_thin_film"):
        raise ValueError("returned receipt Route-A thin-film identity mismatch.")
    if not isinstance(receipt.get("status"), str) or not receipt["status"]:
        raise ValueError("returned receipt status must be a non-empty string.")
    _receipt_execution_state(receipt)
