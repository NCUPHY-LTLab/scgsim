"""Write what one AEDT execution did into its receipt.

The receipt is the durable record of a run; result resolution reads it later. This
module decides which facts are copied and when. Whether a run counts as successful
is decided by ``transaction.py``.

Results are recorded in two stages around Desktop release. Facts returned by the
solver family are copied first, so they survive a failed release. The saved project
is only hashed after release, because that is when AEDT has finished with the file.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from scgsim.aedt._io import file_sha256
from scgsim.aedt.results.provenance import RECEIPT_V3
from scgsim.aedt.runtime.handoff_checks import VerifiedHandoff, contained_path
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

_MODEL_ONLY_SPECS = (
    Q2dSpec,
    Q3dSpec,
    HfssDrivenGeometrySpec,
    HfssDrivenSpec,
    HfssEigenmodeSpec,
)
_MODEL_ONLY_REQUIRED_FIELDS = (
    "outputs",
    "connected",
    "project",
    "materials",
    "region",
    "setup",
    "save",
    "solver_invoked",
)
_MODEL_ONLY_OPTIONAL_FIELDS = ("nets", "ports", "conductors", "mesh", "geometry")
_EPR_OPTIONAL_FIELDS = (
    "convergence",
    "result_readback",
    "saved_field_evidence",
    "benchmark",
)

SIGNAL_CONDUCTORS_WARNING = re.compile(
    r"Port '([^']+)' has (\d+) signal conductors with (\d+) terminals"
)
SLOW_MODE_WARNING = re.compile(
    r"Port ([^ ]+) supports an additional propagating and/or slowly decaying mode"
)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def mark_running(
    handoff: VerifiedHandoff,
    *,
    prepare_only: bool,
    analyze_epr: bool,
    resources_overridden: bool,
) -> None:
    """Turn the prepared receipt into a running one, keeping the prepared identity."""
    receipt = handoff.receipt
    cohort = handoff.cohort
    receipt.update(
        {
            "schema_version": RECEIPT_V3,
            "expected_receipt_schema": RECEIPT_V3,
            "preparation_cohort": cohort["preparation_cohort"],
            "prepared_runtime_source": cohort["prepared_runtime_source"],
            "prepared_receipt_sha256": cohort["prepared_receipt_sha256"],
            "verified_prepared_manifest_sha256": cohort[
                "verified_prepared_manifest_sha256"
            ],
            "status": "running",
            "started_at_utc": utc_now(),
            "save": {"ok": False},
            "release": {"ok": False},
        }
    )
    resources = handoff.resources
    if resources_overridden:
        request_source = "cli_override"
    elif resources is not None:
        request_source = "handoff"
    else:
        request_source = "omitted"
    if prepare_only:
        resource_status = "not_used_prepare_only"
    elif analyze_epr:
        resource_status = "not_applicable_saved_analysis"
    else:
        resource_status = "not_started"
    receipt["resources"] = {
        "requested": resources.to_payload() if resources else None,
        "request_source": request_source,
        "status": resource_status,
    }
    if prepare_only:
        receipt["solver_invoked"] = False
        receipt["model_stage"] = (
            "model_only" if handoff.spec.run_control is None else "model_with_setup"
        )


def record_result_before_release(
    receipt: dict[str, Any], result: dict[str, Any], spec: AedtSpec
) -> None:
    """Copy returned facts before releasing the one transaction-owned Desktop."""
    if isinstance(spec, _MODEL_ONLY_SPECS) and result.get("solver_invoked") is False:
        for name in _MODEL_ONLY_REQUIRED_FIELDS:
            receipt[name] = result[name]
        for name in _MODEL_ONLY_OPTIONAL_FIELDS:
            if name in result:
                receipt[name] = result[name]
    elif isinstance(spec, (HfssEprAnalysisSpec, HfssEprSpec)):
        _record_epr_result(receipt, result)
    else:
        receipt["save"] = result["save"]
        if "geometry" in result:
            receipt["geometry"] = result["geometry"]
        if "benchmark" in result:
            receipt["benchmark"] = result["benchmark"]


def _record_epr_result(receipt: dict[str, Any], result: dict[str, Any]) -> None:
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
    # The receipt field is "epr_result"; the family returns it as "result".
    if "result" in result:
        receipt["epr_result"] = result["result"]
    for name in _EPR_OPTIONAL_FIELDS:
        if name in result:
            receipt[name] = result[name]


def record_saved_project(
    run_dir: Path, receipt: dict[str, Any], result: dict[str, Any]
) -> bool:
    """Hash a released saved project; return false only when it is missing."""
    project_relative = result["project"]
    project_path = contained_path(run_dir, project_relative)
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


def record_solved_result(receipt: dict[str, Any], result: dict[str, Any]) -> None:
    """Copy ordinary HFSS, Q3D or Q2D export facts after Desktop release."""
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
    if "convergence" in result:
        receipt["convergence"] = result["convergence"]


def read_physics_warnings(run_dir: Path) -> dict[str, Any]:
    """Preserve HFSS terminal-mode diagnostics without treating them as gates."""
    path = run_dir / "batch.log"
    if not path.is_file():
        return {"batch_log": "batch.log", "present": False, "physics_warnings": []}
    warnings: dict[tuple[str, str], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        detail = _physics_warning(line)
        if detail is None:
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


def _physics_warning(line: str) -> dict[str, Any] | None:
    if conductor := SIGNAL_CONDUCTORS_WARNING.search(line):
        port, signals, terminals = conductor.groups()
        return {
            "kind": "signal_conductors_per_terminal",
            "port": port,
            "signal_conductors": int(signals),
            "terminals": int(terminals),
        }
    if slow_mode := SLOW_MODE_WARNING.search(line):
        return {
            "kind": "additional_propagating_or_slow_mode",
            "port": slow_mode.group(1),
        }
    return None
