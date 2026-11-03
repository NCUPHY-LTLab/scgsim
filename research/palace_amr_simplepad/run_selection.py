"""Select prepared run pairs and verify their sealed input files."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contract import CASES


@dataclass(frozen=True)
class RunSelection:
    """Cases that may run and cases skipped after incomplete route preparation."""

    selected_cases: tuple[dict[str, Any], ...]
    skipped_cases: tuple[dict[str, Any], ...]


def select_run_cases(
    manifest: dict[str, Any], *, prepared_routes_only: bool
) -> RunSelection:
    """Select all sealed cases or complete pairs from a failed preparation."""
    if not prepared_routes_only:
        require_prepared_manifest(manifest)
        return RunSelection(tuple(manifest["cases"]), ())
    if manifest.get("status") == "prepared":
        require_prepared_manifest(manifest)
        return RunSelection(tuple(manifest["cases"]), ())
    if manifest.get("status") != "preparation_failed":
        raise ValueError(
            "prepared-routes-only requires a prepared or failed-preparation study"
        )

    expected_route_ids = list(dict.fromkeys(case.route_id for case in CASES))
    cases = _require_case_plan(manifest.get("cases", []))
    route_states = _require_route_plan(manifest.get("routes", []), expected_route_ids)
    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for route_id in expected_route_ids:
        route_cases = [case for case in cases if case["route_id"] == route_id]
        route_selected, route_skipped = _select_route_pair(
            route_id, route_cases, route_states[route_id]
        )
        selected.extend(route_selected)
        skipped.extend(route_skipped)

    if not selected:
        raise ValueError(
            "prepared-routes-only requires at least one complete route pair"
        )
    _require_unique_handoff_ids(selected)
    return RunSelection(tuple(selected), tuple(skipped))


def _require_case_plan(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Require the exact ordered source and refinement definitions."""
    actual_contracts = [
        (
            case.get("case_id"),
            case.get("route_id"),
            case.get("representation"),
            case.get("approximation"),
            case.get("amr_nonconformal"),
        )
        for case in cases
    ]
    expected_contracts = [
        (
            case.case_id,
            case.route_id,
            case.representation,
            case.approximation,
            case.nonconformal,
        )
        for case in CASES
    ]
    if actual_contracts != expected_contracts:
        raise ValueError("partial run requires the exact ordered four-case plan")
    return cases


def _require_route_plan(
    routes: list[dict[str, Any]], expected_route_ids: list[str]
) -> dict[str, dict[str, Any]]:
    """Require one route state for each route in the prepared case plan."""
    if [route.get("route_id") for route in routes] != expected_route_ids:
        raise ValueError("partial run requires the exact ordered route plan")
    return {route["route_id"]: route for route in routes}


def _select_route_pair(
    route_id: str, route_cases: list[dict[str, Any]], route: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select a fully sealed pair or classify both cases as route failures."""
    sealed = [
        case for case in route_cases if case.get("preparation_status") == "sealed"
    ]
    if len(sealed) == 1:
        raise ValueError(f"prepared-routes-only rejects half-sealed route {route_id}")
    if len(sealed) == 2:
        if route.get("preparation_status") != "completed":
            raise ValueError(f"sealed route does not have completed state: {route_id}")
        return sealed, []
    if any(case.get("preparation_status") != "failed" for case in route_cases):
        raise ValueError(f"unsealed route lacks recorded failure: {route_id}")
    if route.get("preparation_status") != "failed":
        raise ValueError(f"unsealed route does not have failed state: {route_id}")
    return [], route_cases


def _require_unique_handoff_ids(cases: list[dict[str, Any]]) -> None:
    """Require each selected case to retain its independent handoff identity."""
    handoff_ids = [case.get("handoff_id") for case in cases]
    if any(not isinstance(identity, str) or not identity for identity in handoff_ids):
        raise ValueError("prepared route cases must retain handoff identities")
    if len(set(handoff_ids)) != len(handoff_ids):
        raise ValueError("prepared route cases must have unique handoff identities")


def record_preparation_skips(skipped_cases: tuple[dict[str, Any], ...]) -> int:
    """Mark unsealed cases as unrun without inventing process receipts."""
    for case in skipped_cases:
        case["run_status"] = "not_run_preparation_failed"
        case["run_exit_code"] = None
    return len(skipped_cases)


def verify_case_input_hashes(
    study_dir: Path, selected_cases: tuple[dict[str, Any], ...]
) -> None:
    """Verify selected config and mesh bytes and each route's shared mesh identity."""
    route_meshes: dict[str, set[str]] = {}
    for case in selected_cases:
        run_dir = study_path(study_dir, case["run_dir"])
        for input_name, file_name in (
            ("config", "config.json"),
            ("mesh", "palace.msh"),
        ):
            expected = case.get(f"{input_name}_sha256")
            if not isinstance(expected, str) or not expected:
                raise ValueError(f"prepared case has no sealed {input_name} hash")
            actual = sha256_file(run_dir / file_name)
            if actual != expected:
                raise ValueError(
                    f"prepared case input hash changed: {case['case_id']} {input_name}"
                )
            if input_name == "mesh":
                route_meshes.setdefault(case["route_id"], set()).add(actual)
    for route_id, digests in route_meshes.items():
        if len(digests) != 1:
            raise ValueError(f"prepared route cases do not share one mesh: {route_id}")


def require_prepared_manifest(manifest: dict[str, Any]) -> None:
    """Fail closed unless exactly the four independently sealed cases are present."""
    if manifest.get("status") != "prepared":
        raise ValueError("run requires a completely prepared study")
    cases = manifest.get("cases", [])
    if any(case.get("preparation_status") != "sealed" for case in cases):
        raise ValueError("run requires every planned case to be sealed")
    actual_ids = [case.get("case_id") for case in cases]
    if actual_ids != [case.case_id for case in CASES]:
        raise ValueError("prepared study does not contain the ordered four-case set")
    handoff_ids = [case.get("handoff_id") for case in cases]
    if any(not isinstance(identity, str) or not identity for identity in handoff_ids):
        raise ValueError("prepared cases must retain their handoff identities")
    if len(set(handoff_ids)) != len(CASES):
        raise ValueError("each prepared case must have an independent handoff identity")


def study_path(study_dir: Path, relative_path: str) -> Path:
    """Resolve one manifest path while keeping it inside the selected study."""
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("study manifest paths must be non-empty relative paths")
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise ValueError("study manifest paths must be relative")
    root = study_dir.resolve()
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("study manifest path escapes the selected study directory")
    return resolved


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one existing file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
