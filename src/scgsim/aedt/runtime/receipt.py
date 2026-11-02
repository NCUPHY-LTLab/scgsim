"""Record native results and released project identities in the run receipt."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from scgsim.aedt.io import file_sha256
from scgsim.aedt.runtime.handoff_validation import contained
from scgsim.aedt.specs.hfss import (
    HfssDrivenGeometrySpec,
    HfssDrivenSpec,
    HfssEigenmodeSpec,
    HfssEprAnalysisSpec,
    HfssEprSpec,
)
from scgsim.aedt.specs.parse import AedtSpec
from scgsim.aedt.specs.q2d import Q2dSpec
from scgsim.aedt.specs.q3d import Q3dSpec


def record_result_before_release(
    receipt: dict[str, Any], result: dict[str, Any], spec: AedtSpec
) -> None:
    """Copy returned facts before releasing the one transaction-owned Desktop."""

    if isinstance(spec, Q3dSpec) and "termination" in result:
        for name in (
            "outputs",
            "connected",
            "project",
            "materials",
            "region",
            "nets",
            "setup",
            "save",
            "solver_invoked",
            "termination",
        ):
            receipt[name] = result[name]
    elif (
        isinstance(
            spec,
            (
                Q3dSpec,
                Q2dSpec,
                HfssDrivenGeometrySpec,
                HfssDrivenSpec,
                HfssEigenmodeSpec,
            ),
        )
        and result.get("solver_invoked") is False
    ):
        for name in (
            "outputs",
            "connected",
            "project",
            "materials",
            "region",
            "setup",
            "save",
            "solver_invoked",
        ):
            receipt[name] = result[name]
        for name in ("nets", "ports", "conductors", "mesh", "geometry"):
            if name in result:
                receipt[name] = result[name]
    elif isinstance(spec, (HfssEprAnalysisSpec, HfssEprSpec)):
        _record_epr_before_release(receipt, result)
    else:
        receipt["save"] = result["save"]
        if "geometry" in result:
            receipt["geometry"] = result["geometry"]
        if "benchmark" in result:
            receipt["benchmark"] = result["benchmark"]


def _record_epr_before_release(receipt: dict[str, Any], result: dict[str, Any]) -> None:
    receipt["outputs"] = result["outputs"]
    receipt["connected"] = result["connected"]
    receipt["project"] = result["project"]
    receipt["geometry"] = result["geometry"]
    receipt["setup"] = result["setup"]
    receipt["expressions"] = result.get("expressions", [])
    receipt["cache"] = result.get("cache", {"status": "not_applicable"})
    receipt["timings"] = {**receipt.get("timings", {}), **result.get("timings", {})}
    receipt["solver_invoked"] = result["solver_invoked"]
    receipt["workflow_status"] = result["workflow_status"]
    if "convergence" in result:
        receipt["convergence"] = result["convergence"]
    if "result_readback" in result:
        receipt["result_readback"] = result["result_readback"]
    if "result" in result:
        receipt["epr_result"] = result["result"]
    if "saved_field_evidence" in result:
        receipt["saved_field_evidence"] = result["saved_field_evidence"]
    if "benchmark" in result:
        receipt["benchmark"] = result["benchmark"]


def record_epr_project_after_release(
    run_dir: Path, receipt: dict[str, Any], result: dict[str, Any]
) -> bool:
    """Hash a released saved project; return false only when it is missing."""

    project_relative = result["project"]
    project_path = contained(run_dir, project_relative)
    if not project_path.is_file():
        return False
    project_sha256 = file_sha256(project_path)
    result["outputs"] = {
        project_relative: project_sha256,
        **{
            key: value
            for key, value in result["outputs"].items()
            if key != project_relative
        },
    }
    result["save"] = {
        "ok": True,
        "project_sha256": project_sha256,
        "identity_stage": "after_owned_desktop_release",
    }
    receipt["outputs"] = result["outputs"]
    receipt["save"] = result["save"]
    return True


def record_ordinary_result_after_release(
    receipt: dict[str, Any], result: dict[str, Any]
) -> None:
    """Copy ordinary HFSS/Q3D/Q2D export facts after Desktop release."""

    receipt["outputs"] = result["outputs"]
    receipt["connected"] = result["connected"]
    receipt["project"] = result["project"]
    receipt["ports"] = result.get("ports", [])
    receipt["nets"] = result.get("nets", [])
    receipt["conductors"] = result.get("conductors", [])
    receipt["mesh"] = result.get("mesh", {})
    receipt["materials"] = result["materials"]
    receipt["region"] = result["region"]
    receipt["result_readback"] = result["result_readback"]
    receipt["setup"] = result.get("setup")
    if "stop_control" in result:
        receipt["stop_control"] = result["stop_control"]
    if "convergence" in result:
        receipt["convergence"] = result["convergence"]


def read_physics_warnings(run_dir: Path) -> dict[str, Any]:
    """Preserve HFSS terminal-mode diagnostics without treating them as gates."""
    path = run_dir / "batch.log"
    if not path.is_file():
        return {"batch_log": "batch.log", "present": False, "physics_warnings": []}
    warnings: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        conductor = re.search(
            r"Port '([^']+)' has (\d+) signal conductors with (\d+) terminals", raw_line
        )
        slow_mode = re.search(
            r"Port ([^ ]+) supports an additional propagating and/or slowly decaying mode",
            raw_line,
        )
        if conductor:
            port, signals, terminals = conductor.groups()
            detail = {
                "kind": "signal_conductors_per_terminal",
                "port": port,
                "signal_conductors": int(signals),
                "terminals": int(terminals),
            }
        elif slow_mode:
            detail = {
                "kind": "additional_propagating_or_slow_mode",
                "port": slow_mode.group(1),
            }
        else:
            continue
        key = (detail["kind"], detail["port"])
        record = warnings.setdefault(key, {**detail, "occurrences": 0})
        record["occurrences"] += 1
    return {
        "batch_log": "batch.log",
        "sha256": file_sha256(path),
        "present": True,
        "physics_warnings": [warnings[key] for key in sorted(warnings)],
    }
