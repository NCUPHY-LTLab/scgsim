"""Build Palace trust reports and expose the resolved-run display entry points."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from scgsim.contracts.palace_run_folder import (
    CONFIG,
    HANDOFF_METADATA,
    INDEX_MAP,
    MESH,
    MESH_MANIFEST,
    PROBLEMS,
    RETURNED_RECEIPT,
    palace_problem,
    read_json,
)
from scgsim.palace.mesh.summary import mesh_summary
from scgsim.palace.reports.observations import (
    CostCards,
    configured_refinement,
    cost_cards,
    latest_palace_json,
    recorded_durations,
)
from scgsim.palace.reports.plotting import (
    ReportTheme,
    checked_show_details,
    checked_theme,
)
from scgsim.palace.reports.trust import PalaceTrustReport, ReportIdentity
from scgsim.palace.reports.views import (
    PhysicsQuantitiesReport,
    SimulationBenchmarkReport,
)
from scgsim.palace.results.artifacts import (
    as_int,
    read_optional_json,
)
from scgsim.palace.results.index_contract import (
    require_index_map_schema,
    require_manifest_schema,
    validate_index_entries,
)
from scgsim.palace.results.inspection import (
    failure_diagnosis,
    result_selection,
    validate_inspection_receipt,
)
from scgsim.palace.results.inspection_records import (
    CollectedSnapshots,
    PalaceResultSelection,
    PassCostRecord,
)
from scgsim.palace.results.resolved_records import ResolvedRunData
from scgsim.palace.results.snapshots import collect_amr_passes
from scgsim.palace.results.surface_contract import validate_surface_epr_config
from scgsim.palace.results.surface_readers import (
    read_surface_bindings,
    read_surface_margin_bindings,
)
from scgsim.palace.results.tables import configured_mode_count


def inspect_run_trustworthiness(
    run_dir: str | Path, *, theme: ReportTheme = "light"
) -> PalaceTrustReport:
    """Build the trustworthiness report from a returned run folder.

    This path is for complete or incomplete returned runs. It does not replace
    ``resolve_palace_result`` identity verification for a finished run folder.
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
    _require_resolved(result, "show_run_trustworthiness")
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

    _require_resolved(result, "show_physics_quantities")

    from IPython.display import display  # noqa: PLC0415 -- IPython is optional; detached result reading needs no notebook runtime.

    trust = show_run_trustworthiness(
        result,
        theme=theme,
        show_details=show_details,
    )
    display(trust)
    pass_cost_records = trust.pass_cost_records()
    display(
        SimulationBenchmarkReport(
            trust=trust,
            data=_resolved_benchmark_data(result, pass_cost_records),
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

    _require_resolved(result, "show_simulation_benchmark")

    trust = show_run_trustworthiness(result)
    pass_cost_records = trust.pass_cost_records()
    return SimulationBenchmarkReport(
        trust=trust,
        data=_resolved_benchmark_data(result, pass_cost_records),
        show_details=checked_show_details(show_details),
    )


def _require_resolved(result: object, view: str) -> None:
    """Refuse a view of anything but a resolved run, naming the view and what was passed."""
    if not isinstance(result, ResolvedRunData):
        raise TypeError(
            f"{view} requires the ResolvedPalaceResult that resolve_palace_result() returns; "
            f"got {type(result).__name__}."
        )


def _resolved_benchmark_data(
    result: ResolvedRunData,
    pass_cost_records: tuple[PassCostRecord, ...],
) -> dict[str, Any]:
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
        "pass_cost_records": [asdict(record) for record in pass_cost_records],
    }


def _build_trust_report(
    root: Path,
    *,
    resolved: ResolvedRunData | None = None,
    theme: ReportTheme = "light",
) -> PalaceTrustReport:
    """Bind source identity, readable snapshots and failure state into the Palace trust view."""
    try:
        run = _InspectedRun.read(root, resolved)
    except (FileNotFoundError, TypeError, ValueError) as error:
        # Each check names what disagrees; every one of them is acted on the same way.
        error.add_note(
            f"Run folder {root}: the trust report reads the folder Palace returned, complete or partial, as "
            "the run left it. Fetch the folder again from the host that ran Palace."
        )
        raise
    collected = collect_amr_passes(
        root,
        run.problem,
        read_surface_bindings(run.index_map),
        read_surface_margin_bindings(run.index_map),
        failed_attempt=run.receipt is not None
        and run.receipt.get("status") == "failed",
        mode_count=configured_mode_count(run.config),
    )
    selection = result_selection(
        root=root,
        problem=run.problem,
        collected=collected,
        receipt_paths=run.receipt_paths,
    )
    failure = failure_diagnosis(root, run.receipt, run.receipt_paths)
    palace_payload = latest_palace_json(collected.passes, resolved)
    durations = recorded_durations(palace_payload)
    cost = cost_cards(palace_payload, resolved)
    refinement = configured_refinement(run.config)
    return PalaceTrustReport(
        run_dir=root,
        problem=run.problem,
        route=run.route,
        profile=run.profile,
        completeness=_completeness(run.receipt, collected, selection),
        latest_source=selection.selected_source or "none",
        selection=selection,
        failure=failure,
        identity=_report_identity(cost, run.handoff, run.receipt, resolved),
        passes=tuple(collected.passes),
        amr_tolerance=refinement.get("Tol"),
        amr_max_passes=as_int(refinement.get("MaxIts")),
        durations=durations,
        cost=cost,
        provenance=_report_provenance(
            run.config, run.handoff, run.mesh_manifest, run.problem, run.receipt
        ),
        theme=checked_theme(theme),
    )


@dataclass(frozen=True)
class _InspectedRun:
    """The records of a returned run folder that a trust report reads, each checked as it is read.

    A run that failed or stopped early may lack the mesh manifest, the config, the receipt or the index
    map; each is then None.
    """

    handoff: dict[str, Any]
    mesh_manifest: dict[str, Any] | None
    problem: str
    route: str
    profile: str
    config: dict[str, Any] | None
    receipt: dict[str, Any] | None
    receipt_paths: frozenset[str]
    index_map: dict[str, Any] | None

    @classmethod
    def read(cls, root: Path, resolved: ResolvedRunData | None) -> _InspectedRun:
        """Read the handoff first, then each record a partial run may lack, refusing what disagrees."""
        handoff_path = root / HANDOFF_METADATA
        if not handoff_path.is_file():
            raise FileNotFoundError(f"required artifact missing: {handoff_path}")
        handoff = read_json(handoff_path)
        mesh_manifest = read_optional_json(root / MESH_MANIFEST)
        if mesh_manifest is not None:
            require_manifest_schema(mesh_manifest)
        mesh_thin_film = (
            mesh_manifest.get("route_a_thin_film")
            if isinstance(mesh_manifest, dict)
            else None
        )
        if mesh_thin_film != handoff.get("route_a_thin_film"):
            raise ValueError(
                "mesh manifest and handoff Route-A thin-film identity mismatch: "
                f"{MESH_MANIFEST} records {mesh_thin_film!r}, {HANDOFF_METADATA} records "
                f"{handoff.get('route_a_thin_film')!r} in {root}."
            )
        problem = str(handoff.get("problem") or (resolved.problem if resolved else ""))
        route = str(handoff.get("route") or (resolved.route if resolved else ""))
        profile = str(handoff.get("profile") or "")
        if problem not in PROBLEMS:
            raise ValueError(
                f"unsupported problem {problem!r} for trustworthiness report "
                f"({HANDOFF_METADATA} in {root}); SCGSim writes {sorted(PROBLEMS)}."
            )
        config = read_optional_json(root / CONFIG)
        receipt = read_optional_json(root / RETURNED_RECEIPT)
        receipt_paths = validate_inspection_receipt(root, handoff, receipt)
        index_map = (
            resolved.provenance.index_map
            if resolved is not None
            else read_optional_json(root / INDEX_MAP)
        )
        if index_map is not None:
            require_index_map_schema(index_map)
            validate_index_entries(index_map)
            if config is not None:
                validate_surface_epr_config(config, index_map)
        return cls(
            handoff,
            mesh_manifest,
            problem,
            route,
            profile,
            config,
            receipt,
            receipt_paths,
            index_map,
        )


def _completeness(
    receipt: dict[str, Any] | None,
    collected: CollectedSnapshots,
    selection: PalaceResultSelection,
) -> Literal["complete", "partial"]:
    """Complete only when the receipt says completed and the final tables are the readable selection."""
    receipt_status = receipt.get("status") if receipt is not None else None
    parent_complete = (
        receipt_status == "completed"
        and collected.final_snapshot_status == "readable"
        and selection.selected_source == "final"
    )
    return "complete" if parent_complete else "partial"


__all__ = [
    "inspect_run_trustworthiness",
    "show_all_results",
    "show_physics_quantities",
    "show_run_trustworthiness",
    "show_simulation_benchmark",
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
        mesh_sha256=hashes.get(MESH),
        manifest_sha256=hashes.get(MESH_MANIFEST),
    )
    if fem_order is None and palace_problem(problem).fem_space == "ND":
        provenance["mesh_summary"]["dof_estimate"]["reason"] = "FEM order not recorded"
    if receipt is not None:
        provenance["returned_receipt"] = receipt
    return provenance


def _report_identity(
    cost: CostCards,
    handoff: dict[str, Any],
    receipt: dict[str, Any] | None,
    resolved: ResolvedRunData | None,
) -> ReportIdentity:
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
    return {
        "handoff_id": handoff_id,
        "handoff_id_short": handoff_id[:12] if handoff_id else None,
        "resources": resources,
        "receipt": receipt_status,
        "git_tag": cost.get("git_tag"),
    }
