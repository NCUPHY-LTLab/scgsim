"""Build Palace trust reports and expose the resolved-run display entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from scgsim.palace.mesh.summary import mesh_summary
from scgsim.palace.presentation.observations import (
    configured_refinement,
    cost_cards,
    latest_palace_json,
    recorded_durations,
)
from scgsim.palace.presentation.plotting import (
    ReportTheme,
    checked_show_details,
    checked_theme,
)
from scgsim.palace.results.artifacts import read_json
from scgsim.palace.results.index_contract import validate_index_entries
from scgsim.palace.results.inspection import (
    failure_diagnosis,
    result_selection,
    validate_inspection_receipt,
)
from scgsim.palace.results.models import ResolvedRunData
from scgsim.palace.results.observations import as_int, read_optional_json
from scgsim.palace.results.report_models import (
    AmrPassSnapshot,
    PalaceFailureDiagnosis,
    PalaceResultSelection,
    PassCostRecord,
)
from scgsim.palace.results.snapshots import (
    collect_amr_passes,
    declares_surface_masks,
)
from scgsim.palace.results.surface_contract import validate_surface_mask_config
from scgsim.palace.results.surface_readers import (
    read_surface_bindings,
    read_surface_mask_bindings,
)

from .trust import PalaceTrustReport as PalaceTrustReport
from .views import PhysicsQuantitiesReport as PhysicsQuantitiesReport
from .views import SimulationBenchmarkReport as SimulationBenchmarkReport


def inspect_run_trustworthiness(
    run_dir: str | Path, *, theme: ReportTheme = "light"
) -> PalaceTrustReport:
    """Build the trustworthiness report from a package folder.

    This path is for complete or incomplete returned runs. It does not replace
    ``resolve_palace_result`` identity verification for a finished package.
    ``theme`` is ``light`` (default) or ``dark`` and applies only to Plotly
    figures.
    """

    root = Path(run_dir).expanduser().resolve()
    return _build_trust_report(root, theme=theme)


def show_run_trustworthiness(
    result: ResolvedRunData,
    *,
    theme: ReportTheme = "light",
    show_details: bool = False,
) -> PalaceTrustReport:
    """Build the resolved-run identity, AMR and failure view with the requested theme."""
    if not isinstance(result, ResolvedRunData):
        raise TypeError("resolved result report requires ResolvedPalaceResult.")
    report = _build_trust_report(result.run_dir, resolved=result, theme=theme)
    return report.show_run_trustworthiness(theme=theme, show_details=show_details)


def show_physics_quantities(
    result: ResolvedRunData,
    *,
    theme: ReportTheme = "light",
    ranking_limit: int | None = 20,
) -> PhysicsQuantitiesReport:
    """Return the latest-readable physical series after building their trust context."""
    report = show_run_trustworthiness(result, theme=theme)
    return report.show_physics_quantities(
        theme=theme,
        ranking_limit=ranking_limit,
    )


def show_all_results(
    result: ResolvedRunData,
    *,
    theme: ReportTheme = "light",
    ranking_limit: int | None = 20,
    show_details: bool = False,
) -> None:
    """Display trust, benchmark, and physics in the Human-defined order."""

    if not isinstance(result, ResolvedRunData):
        raise TypeError("resolved result report requires ResolvedPalaceResult.")

    # Notebook display is optional for detached result processing.

    from IPython.display import display  # noqa: PLC0415

    trust = show_run_trustworthiness(
        result,
        theme=theme,
        show_details=show_details,
    )
    display(trust)
    display(
        SimulationBenchmarkReport(
            trust=trust,
            data=_resolved_benchmark_data(result),
            show_details=checked_show_details(show_details),
        )
    )
    display(
        trust.show_physics_quantities(
            theme=theme,
            ranking_limit=ranking_limit,
        )
    )


def show_simulation_benchmark(
    result: ResolvedRunData,
    *,
    show_details: bool = False,
) -> SimulationBenchmarkReport:
    """Return the readable benchmark and its complete machine data."""

    if not isinstance(result, ResolvedRunData):
        raise TypeError("resolved result report requires ResolvedPalaceResult.")

    return SimulationBenchmarkReport(
        trust=show_run_trustworthiness(result),
        data=_resolved_benchmark_data(result),
        show_details=checked_show_details(show_details),
    )


def _resolved_benchmark_data(result: ResolvedRunData) -> dict[str, Any]:
    """Detach final and attempted-run costs, timings, resources and selection metadata."""
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


def _build_trust_report(
    root: Path,
    *,
    resolved: ResolvedRunData | None = None,
    theme: ReportTheme = "light",
) -> PalaceTrustReport:
    """Bind source identity, readable snapshots and failure state into the Palace trust view."""
    handoff_path = root / "metadata" / "palace_handoff_metadata.json"
    if not handoff_path.is_file():
        raise FileNotFoundError(f"required artifact missing: {handoff_path}")
    handoff = read_json(handoff_path)
    mesh_manifest = read_optional_json(root / "metadata" / "mesh_manifest.json")
    mesh_thin_film = (
        mesh_manifest.get("route_a_thin_film")
        if isinstance(mesh_manifest, dict)
        else None
    )
    if mesh_thin_film != handoff.get("route_a_thin_film"):
        raise ValueError(
            "mesh manifest and handoff Route-A thin-film identity mismatch."
        )
    problem = str(handoff.get("problem") or (resolved.problem if resolved else ""))
    route = str(handoff.get("route") or (resolved.route if resolved else ""))
    profile = str(handoff.get("profile") or "")
    if problem not in {"Eigenmode", "Electrostatic"}:
        raise ValueError(f"unsupported problem {problem!r} for trustworthiness report.")

    config = read_optional_json(root / "config.json")
    receipt = read_optional_json(root / "metadata" / "palace_returned_run_receipt.json")
    receipt_paths = validate_inspection_receipt(root, handoff, receipt)
    refinement = configured_refinement(config)
    index_map = (
        resolved.provenance.index_map
        if resolved is not None
        else read_optional_json(root / "metadata" / "palace_index_map.json")
    )
    if declares_surface_masks(config, index_map):
        if not isinstance(config, dict) or not isinstance(index_map, dict):
            raise ValueError(
                "declared surface masks require config and index evidence."
            )
        validate_index_entries(index_map)
        validate_surface_mask_config(config, index_map)
    surface_bindings = read_surface_bindings(index_map)
    surface_mask_bindings = read_surface_mask_bindings(index_map)
    collected = collect_amr_passes(
        root,
        problem,
        surface_bindings,
        surface_mask_bindings,
        failed_attempt=receipt is not None and receipt.get("status") == "failed",
    )
    passes = collected.passes
    receipt_status = receipt.get("status") if receipt is not None else None
    selection = result_selection(
        root=root,
        problem=problem,
        collected=collected,
        receipt_paths=receipt_paths,
    )
    parent_complete = (
        receipt_status == "completed"
        and collected.final_snapshot_status == "readable"
        and selection.selected_source == "final"
    )
    completeness: Literal["complete", "partial"] = (
        "complete" if parent_complete else "partial"
    )
    latest_source = selection.selected_source or "none"
    failure = failure_diagnosis(root, receipt, receipt_paths)
    palace_payload = latest_palace_json(passes, resolved)
    durations = recorded_durations(palace_payload)
    cost = cost_cards(palace_payload, resolved)
    identity = _report_identity(cost, handoff, receipt, resolved)
    provenance = _report_provenance(config, handoff, mesh_manifest, problem, receipt)
    return PalaceTrustReport(
        run_dir=root,
        problem=problem,
        route=route,
        profile=profile,
        completeness=completeness,
        latest_source=latest_source,
        selection=selection,
        failure=failure,
        identity=identity,
        passes=tuple(passes),
        amr_tolerance=refinement.get("Tol"),
        amr_max_passes=as_int(refinement.get("MaxIts")),
        durations=durations,
        cost=cost,
        provenance=provenance,
        theme=checked_theme(theme),
    )


__all__ = [
    "AmrPassSnapshot",
    "PalaceFailureDiagnosis",
    "PalaceResultSelection",
    "PalaceTrustReport",
    "PassCostRecord",
    "PhysicsQuantitiesReport",
    "inspect_run_trustworthiness",
]


def _report_provenance(
    config: dict[str, Any] | None,
    handoff: dict[str, Any],
    mesh_manifest: dict[str, Any] | None,
    problem: str,
    receipt: dict[str, Any] | None,
) -> dict[str, Any]:
    """Preserve source/resource identities and detached mesh observations for report inspection."""
    provenance = {
        key: handoff.get(key)
        for key in (
            "source_revisions",
            "palace_identity",
            "hashes",
            "requested_resources",
            "resolved_resources",
            "route_a_thin_film",
        )
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


def _report_identity(
    cost: dict[str, Any],
    handoff: dict[str, Any],
    receipt: dict[str, Any] | None,
    resolved: ResolvedRunData | None,
) -> dict[str, Any]:
    """Project handoff, receipt status, solver tag and parallel resources into identity cards."""
    mpi = cost.get("mpi_size")
    omp = cost.get("openmp_threads")
    resources = (
        f"{mpi} MPI × {omp} OMP" if mpi is not None and omp is not None else None
    )
    receipt_status = None
    if receipt is not None:
        receipt_status = (
            f"{receipt.get('status', 'unknown')} / exit {receipt.get('exit_code', '?')}"
        )
    elif resolved is not None:
        receipt_status = f"{resolved.returned_receipt.status} / exit {resolved.returned_receipt.exit_code}"
    handoff_id = str(handoff.get("handoff_id") or "")
    identity = {
        "handoff_id": handoff_id,
        "handoff_id_short": handoff_id[:12] if handoff_id else None,
        "resources": resources,
        "receipt": receipt_status,
        "git_tag": cost.get("git_tag"),
    }
    return identity
