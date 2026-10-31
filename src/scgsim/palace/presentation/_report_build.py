"""Collect what a Palace trust report shows from a run directory.

``_trust_report_fields`` reads a returned run, complete or not, and returns the
values ``PalaceTrustReport`` is made of. It trusts nothing it cannot cross-check: the
mesh manifest and handoff must agree on the Route-A thin-film identity, and a run that
declares surface masks must pass the same mask checks the resolver applies.

Not here: the report classes and their display (``reports.py``) and the figures
(``_figures.py``). The values come back as a dict instead of a ``PalaceTrustReport``
so this module does not import the class that imports it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, TypedDict

from ..mesh.summary import mesh_summary
from ..results._artifact_files import _read_json
from ..results._config_checks import _validate_surface_mask_config
from ..results._index_map import _validate_index_entries
from ..results._inspection import (
    _failure_diagnosis,
    _result_selection,
    _validate_inspection_receipt,
)
from ..results._surface_epr_reader import (
    _read_surface_bindings,
    _read_surface_mask_bindings,
)
from ..results.report_data import (
    AmrPassSnapshot,
    PalaceFailureDiagnosis,
    PalaceResultSelection,
    _as_int,
    _collect_amr_passes,
    _declares_surface_masks,
    _mapping_max,
    _read_optional_json,
)
from ..results.resolve import ResolvedPalaceResult
from ._figures import ReportTheme, _checked_theme


class TrustIdentity(TypedDict):
    """The short run identity shown at the top of the trust report."""

    handoff_id: str
    handoff_id_short: str | None
    resources: str | None
    receipt: str | None
    git_tag: str | None


class TrustReportFields(TypedDict):
    """The constructor arguments of ``PalaceTrustReport``."""

    run_dir: Path
    problem: str
    route: str
    profile: str
    completeness: Literal["complete", "partial"]
    latest_source: str
    selection: PalaceResultSelection
    failure: PalaceFailureDiagnosis | None
    identity: TrustIdentity
    passes: tuple[AmrPassSnapshot, ...]
    amr_tolerance: float | None
    amr_max_passes: int | None
    durations: dict[str, float]
    cost: dict[str, Any]
    provenance: dict[str, Any]
    theme: ReportTheme


# Handoff keys the report's provenance section repeats when the handoff has them.
_PROVENANCE_KEYS = (
    "source_revisions",
    "palace_identity",
    "hashes",
    "requested_resources",
    "resolved_resources",
    "route_a_thin_film",
)


def _trust_report_fields(
    root: Path,
    *,
    resolved: ResolvedPalaceResult | None = None,
    theme: ReportTheme = "light",
) -> TrustReportFields:
    handoff, mesh_manifest = _read_handoff_and_mesh_manifest(root)
    problem = str(handoff.get("problem") or (resolved.problem if resolved else ""))
    route = str(handoff.get("route") or (resolved.route if resolved else ""))
    profile = str(handoff.get("profile") or "")
    if problem not in {"Eigenmode", "Electrostatic"}:
        raise ValueError(f"unsupported problem {problem!r} for trustworthiness report.")

    config = _read_optional_json(root / "config.json")
    receipt = _read_optional_json(
        root / "metadata" / "palace_returned_run_receipt.json"
    )
    receipt_paths = _validate_inspection_receipt(root, handoff, receipt)
    refinement = _refinement(config)
    index_map = (
        resolved.provenance.index_map
        if resolved is not None
        else _read_optional_json(root / "metadata" / "palace_index_map.json")
    )
    _check_declared_surface_masks(config, index_map)
    collected = _collect_amr_passes(
        root,
        problem,
        _read_surface_bindings(index_map),
        _read_surface_mask_bindings(index_map),
        failed_attempt=receipt is not None and receipt.get("status") == "failed",
    )
    selection = _result_selection(
        root=root,
        problem=problem,
        collected=collected,
        receipt_paths=receipt_paths,
    )
    # "complete" needs a completed receipt and the final solver snapshot as the shown one.
    parent_complete = (
        _receipt_status(receipt) == "completed"
        and collected.final_snapshot_status == "readable"
        and selection.selected_source == "final"
    )
    failure = _failure_diagnosis(root, receipt, receipt_paths)
    palace_payload = _latest_palace_json(collected.passes, resolved)
    cost = _cost_cards(palace_payload, resolved)
    return {
        "run_dir": root,
        "problem": problem,
        "route": route,
        "profile": profile,
        "completeness": "complete" if parent_complete else "partial",
        "latest_source": selection.selected_source or "none",
        "selection": selection,
        "failure": failure,
        "identity": _trust_identity(handoff, receipt, resolved, cost),
        "passes": tuple(collected.passes),
        "amr_tolerance": refinement.get("Tol"),
        "amr_max_passes": _as_int(refinement.get("MaxIts")),
        "durations": _durations(palace_payload),
        "cost": cost,
        "provenance": _report_provenance(
            handoff, receipt, config, mesh_manifest, problem
        ),
        "theme": _checked_theme(theme),
    }


def _read_handoff_and_mesh_manifest(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    handoff_path = root / "metadata" / "palace_handoff_metadata.json"
    if not handoff_path.is_file():
        raise FileNotFoundError(f"required artifact missing: {handoff_path}")
    handoff = _read_json(handoff_path)
    mesh_manifest = _read_optional_json(root / "metadata" / "mesh_manifest.json")
    mesh_thin_film = (
        mesh_manifest.get("route_a_thin_film")
        if isinstance(mesh_manifest, dict)
        else None
    )
    if mesh_thin_film != handoff.get("route_a_thin_film"):
        raise ValueError(
            "mesh manifest and handoff Route-A thin-film identity mismatch."
        )
    return handoff, mesh_manifest


def _check_declared_surface_masks(
    config: dict[str, Any] | None, index_map: dict[str, Any] | None
) -> None:
    if not _declares_surface_masks(config, index_map):
        return
    if not isinstance(config, dict) or not isinstance(index_map, dict):
        raise ValueError("declared surface masks require config and index evidence.")
    _validate_index_entries(index_map)
    _validate_surface_mask_config(config, index_map)


def _receipt_status(receipt: dict[str, Any] | None) -> Any:
    return receipt.get("status") if receipt is not None else None


def _trust_identity(
    handoff: dict[str, Any],
    receipt: dict[str, Any] | None,
    resolved: ResolvedPalaceResult | None,
    cost: dict[str, Any],
) -> TrustIdentity:
    mpi = cost.get("mpi_size")
    omp = cost.get("openmp_threads")
    resources = (
        f"{mpi} MPI × {omp} OMP" if mpi is not None and omp is not None else None
    )
    receipt_summary = None
    if receipt is not None:
        receipt_summary = (
            f"{receipt.get('status', 'unknown')} / exit {receipt.get('exit_code', '?')}"
        )
    elif resolved is not None:
        receipt_summary = f"{resolved.returned_receipt.status} / exit {resolved.returned_receipt.exit_code}"
    handoff_id = str(handoff.get("handoff_id") or "")
    return {
        "handoff_id": handoff_id,
        "handoff_id_short": handoff_id[:12] if handoff_id else None,
        "resources": resources,
        "receipt": receipt_summary,
        "git_tag": cost.get("git_tag"),
    }


def _report_provenance(
    handoff: dict[str, Any],
    receipt: dict[str, Any] | None,
    config: dict[str, Any] | None,
    mesh_manifest: dict[str, Any] | None,
    problem: str,
) -> dict[str, Any]:
    provenance = {
        key: handoff.get(key)
        for key in _PROVENANCE_KEYS
        if handoff.get(key) is not None
    }
    hashes = {entry["path"]: entry["sha256"] for entry in (handoff.get("hashes") or ())}
    solver = config.get("Solver", {}) if config is not None else {}
    fem_order = solver.get("Order")
    provenance["mesh_summary"] = mesh_summary(
        mesh_manifest if mesh_manifest is not None else {},
        problem=problem,
        fem_order=fem_order,
        mesh_sha256=hashes.get("palace.msh"),
        manifest_sha256=hashes.get("metadata/mesh_manifest.json"),
    )
    if fem_order is None and problem == "Eigenmode":
        provenance["mesh_summary"]["dof_estimate"]["reason"] = "FEM order not recorded"
    if receipt is not None:
        provenance["returned_receipt"] = receipt
    return provenance


def _resolved_benchmark_data(result: ResolvedPalaceResult) -> dict[str, Any]:
    return {
        "cost": {
            "problem_degrees_of_freedom": result.cost.problem_degrees_of_freedom,
            "mesh_elements": result.cost.mesh_elements,
            "mpi_size": result.cost.mpi_size,
            "openmp_threads": result.cost.openmp_threads,
            "peak_memory_megabytes": result.cost.peak_memory_megabytes,
            "peak_node_memory_megabytes": result.cost.peak_node_memory_megabytes,
            "linear_solver": result.cost.linear_solver,
            "git_tag": result.cost.git_tag,
        },
        "performance": {
            "counts": result.performance.counts,
            "durations": result.performance.durations,
        },
        "resources": {
            "requested_resources": result.provenance.resource_record.get(
                "requested_resources", {}
            ),
            "resolved_resources": result.provenance.resource_record.get(
                "resolved_resources", {}
            ),
        },
        "performance_metadata": {
            "route": result.route,
            "problem": result.problem,
            "status": result.status,
            "has_returned_outputs": result.has_returned_outputs,
            "receipt_status": result.returned_receipt.status,
            "receipt_exit_code": result.returned_receipt.exit_code,
        },
        "mesh_summary": result.mesh_summary(),
        "error_indicators": {
            "rows": len(result.tables["error-indicators"].rows),
            "headers": result.tables["error-indicators"].headers,
        },
    }


def _refinement(config: dict[str, Any] | None) -> dict[str, Any]:
    if not config:
        return {}
    model = config.get("Model")
    if not isinstance(model, dict):
        return {}
    refinement = model.get("Refinement")
    return refinement if isinstance(refinement, dict) else {}


def _latest_palace_json(
    passes: Sequence[AmrPassSnapshot], resolved: ResolvedPalaceResult | None
) -> dict[str, Any] | None:
    if resolved is not None:
        return resolved.provenance.palace_json
    for pass_ in reversed(passes):
        payload = _read_optional_json(pass_.path / "palace.json")
        if payload is not None:
            return payload
    return None


def _durations(palace_payload: dict[str, Any] | None) -> dict[str, float]:
    if not palace_payload:
        return {}
    elapsed = palace_payload.get("ElapsedTime")
    if not isinstance(elapsed, dict):
        return {}
    raw = elapsed.get("Durations")
    if not isinstance(raw, dict):
        return {}
    return {
        str(key): float(value)
        for key, value in raw.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _cost_cards(
    palace_payload: dict[str, Any] | None, resolved: ResolvedPalaceResult | None
) -> dict[str, Any]:
    if resolved is not None:
        return {
            "problem_degrees_of_freedom": resolved.cost.problem_degrees_of_freedom,
            "mesh_elements": resolved.cost.mesh_elements,
            "mpi_size": resolved.cost.mpi_size,
            "openmp_threads": resolved.cost.openmp_threads,
            "peak_memory_megabytes": _mapping_max(resolved.cost.peak_memory_megabytes),
            "peak_node_memory_megabytes": _mapping_max(
                resolved.cost.peak_node_memory_megabytes
            ),
            "git_tag": resolved.cost.git_tag,
        }
    if not palace_payload:
        return {}
    problem_block = palace_payload.get("Problem")
    problem_block = problem_block if isinstance(problem_block, dict) else {}
    return {
        "problem_degrees_of_freedom": _as_int(problem_block.get("DegreesOfFreedom")),
        "mesh_elements": _as_int(problem_block.get("MeshElements")),
        "mpi_size": _as_int(problem_block.get("MPISize")),
        "openmp_threads": _as_int(problem_block.get("OpenMPThreads")),
        "peak_memory_megabytes": _mapping_max(
            palace_payload.get("PeakMemoryMegabytes")
        ),
        "peak_node_memory_megabytes": _mapping_max(
            palace_payload.get("PeakNodeMemoryMegabytes")
        ),
        "git_tag": palace_payload.get("GitTag")
        if isinstance(palace_payload.get("GitTag"), str)
        else None,
    }
