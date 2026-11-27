"""Build Palace trust reports and expose the resolved-run display entry points."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

from scgsim.contracts.palace_convergence import returned_passes
from scgsim.contracts.palace_run_folder import (
    CONFIG,
    DRIVEN,
    HANDOFF_METADATA,
    INDEX_MAP,
    MESH_MANIFEST,
    NATIVE_METADATA,
    OUTPUT_DIR,
    PROBLEMS,
    RETURNED_RECEIPT,
    read_json,
)
from scgsim.palace.mesh.summary import recorded_mesh_summary
from scgsim.palace.reports.driven import DrivenPhysicsReport, DrivenTrustReport
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
from scgsim.palace.reports.trust import PalaceTrustReport
from scgsim.palace.reports.trust_records import (
    ReportIdentity,
    ReportProvenance,
)
from scgsim.palace.reports.views import (
    PhysicsQuantitiesReport,
    ResolvedBenchmarkData,
    SimulationBenchmarkReport,
)
from scgsim.palace.results.artifacts import (
    as_int,
    read_optional_json,
)
from scgsim.palace.results.convergence import (
    check_passes,
    inspected_end,
    passes_allowed,
    read_plan,
)
from scgsim.palace.results.driven import driven_result
from scgsim.palace.results.driven_records import PalaceDrivenResult
from scgsim.palace.results.driven_tables import requested_sweep
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
from scgsim.palace.run.archive import RETURNED_RUN_FOLDER
from scgsim.refusal import Declared

_WHERE_REPORT_VIEW = "the call to the report view"


def inspect_run_trustworthiness(
    run_dir: str | Path, *, theme: ReportTheme = "light"
) -> PalaceTrustReport | DrivenTrustReport:
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
) -> PalaceTrustReport | DrivenTrustReport:
    """Build the resolved-run identity, convergence and failure view with the requested theme."""
    _require_resolved(result, "show_run_trustworthiness")
    report = _build_trust_report(result.run_dir, resolved=result, theme=theme)
    return report.show_run_trustworthiness(theme=theme, show_details=show_details)


def show_physics_quantities(
    result: ResolvedRunData,
    *,
    theme: ReportTheme = "light",
    ranking_limit: int | None = 20,
) -> PhysicsQuantitiesReport | DrivenPhysicsReport:
    """Return the latest-readable physical series after building their trust context.

    ``ranking_limit`` caps the surfaces ranked for each mode or excitation; a Driven run's view ranks none.
    """
    return _physics(show_run_trustworthiness(result, theme=theme), theme, ranking_limit)


def show_all_results(
    result: ResolvedRunData,
    *,
    theme: ReportTheme = "light",
    ranking_limit: int | None = 20,
    show_details: bool = False,
) -> None:
    """Display the views in the order a reader asks: which run and did it converge, its numbers, its cost."""

    _require_resolved(result, "show_physics_quantities")

    from IPython.display import display  # noqa: PLC0415 -- IPython is optional; detached result reading needs no notebook runtime.

    trust = show_run_trustworthiness(
        result,
        theme=theme,
        show_details=show_details,
    )
    display(trust)
    display(_physics(trust, theme, ranking_limit))
    if isinstance(trust, DrivenTrustReport):
        display(trust.show_simulation_benchmark(theme=theme, show_details=show_details))
        return
    pass_cost_records = trust.pass_cost_records()
    display(
        SimulationBenchmarkReport(
            trust=trust,
            data=_resolved_benchmark_data(result, pass_cost_records),
            show_details=checked_show_details(show_details),
        )
    )


def show_simulation_benchmark(
    result: ResolvedRunData,
    *,
    theme: ReportTheme = "light",
    show_details: bool = False,
) -> SimulationBenchmarkReport:
    """Return the readable benchmark, its figures in ``theme``, and its complete machine data."""

    _require_resolved(result, "show_simulation_benchmark")

    trust = show_run_trustworthiness(result, theme=theme)
    # A Driven trust report holds the cost of its full solves and builds its own benchmark.
    if isinstance(trust, DrivenTrustReport):
        return trust.show_simulation_benchmark(theme=theme, show_details=show_details)
    pass_cost_records = trust.pass_cost_records()
    return SimulationBenchmarkReport(
        trust=trust,
        data=_resolved_benchmark_data(result, pass_cost_records),
        show_details=checked_show_details(show_details),
    )


def _physics(
    trust: PalaceTrustReport | DrivenTrustReport,
    theme: ReportTheme,
    ranking_limit: int | None,
) -> PhysicsQuantitiesReport | DrivenPhysicsReport:
    """The physics view of ``trust``: a Driven run's S-parameters and Surface EPR rank no surface."""
    if isinstance(trust, DrivenTrustReport):
        return trust.show_physics_quantities(theme=theme)
    return trust.show_physics_quantities(theme=theme, ranking_limit=ranking_limit)


def _require_resolved(result: object, view: str) -> None:
    """Refuse a view of anything but a resolved run, naming the view and what was passed."""
    if not isinstance(result, ResolvedRunData):
        raise TypeError(
            Declared(f"{view}", _WHERE_REPORT_VIEW).refusal(
                f"requires the ResolvedPalaceResult that resolve_palace_result() returns; got {type(result).__name__}"
            )
        )


def _resolved_benchmark_data(
    result: ResolvedRunData,
    pass_cost_records: tuple[PassCostRecord, ...],
) -> ResolvedBenchmarkData:
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
) -> PalaceTrustReport | DrivenTrustReport:
    """Bind source identity, readable snapshots and failure state into the Palace trust view.

    A Driven run has its own report (``_driven_trust_report``): the AMR snapshots below read Electrostatic and
    Eigenmode tables.
    """
    try:
        run = _InspectedRun.read(root, resolved)
    except (FileNotFoundError, TypeError, ValueError) as error:
        # Each check names what disagrees; every one of them is acted on the same way.
        error.add_note(
            f"Run folder {root}: the trust report reads the folder Palace returned, complete or partial, as "
            "the run left it. Fetch the folder again from the host that ran Palace."
        )
        raise
    if run.problem == DRIVEN.name:
        return _driven_trust_report(root, run, resolved, theme)
    collected = collect_amr_passes(
        root,
        run.problem,
        read_surface_bindings(run.index_map),
        read_surface_margin_bindings(run.index_map),
        mode_count=configured_mode_count(run.config),
    )
    solver_end = (
        resolved.solver_end
        if resolved is not None
        else inspected_end(root, run.receipt, passes_allowed(run.config))
    )
    selection = result_selection(
        root=root,
        problem=run.problem,
        collected=collected,
        receipt_paths=run.receipt_paths,
        stopped_at_convergence=solver_end.ending == "stopped_at_convergence",
    )
    failure = failure_diagnosis(root, run.receipt, run.receipt_paths)
    palace_payload = latest_palace_json(collected.passes, resolved)
    durations = recorded_durations(palace_payload)
    cost = cost_cards(palace_payload, resolved)
    refinement = configured_refinement(run.config)
    # A resolved run was checked on the same passes; a partial one is checked on the passes it holds.
    convergence = (
        resolved.convergence
        if resolved is not None
        else check_passes(read_plan(root, run.handoff, run.problem), root)
    )
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
        convergence=convergence,
        solver_end=solver_end,
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
            raise FileNotFoundError(
                Declared("required artifact", RETURNED_RUN_FOLDER).refusal(
                    f"missing: {handoff_path}"
                )
            )
        handoff = read_json(handoff_path)
        mesh_manifest = _read_mesh_manifest(root, handoff)
        problem = str(handoff.get("problem") or (resolved.problem if resolved else ""))
        route = str(handoff.get("route") or (resolved.route if resolved else ""))
        profile = str(handoff.get("profile") or "")
        if problem not in PROBLEMS:
            raise ValueError(
                Declared(
                    f"problem {problem!r} of {HANDOFF_METADATA} in {root}",
                    RETURNED_RUN_FOLDER,
                ).refusal(f"is not one SCGSim writes: {sorted(PROBLEMS)}")
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


def _read_mesh_manifest(root: Path, handoff: dict[str, Any]) -> dict[str, Any] | None:
    """The run's mesh manifest, if it wrote one, with the Route-A thin-film identity the handoff records."""
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
            Declared(
                "mesh manifest and handoff Route-A thin-film identity",
                RETURNED_RUN_FOLDER,
            ).refusal(
                f"mismatch: {MESH_MANIFEST} records {mesh_thin_film!r}, {HANDOFF_METADATA} records {handoff.get('route_a_thin_film')!r} in {root}"
            )
        )
    return mesh_manifest


def _completeness(
    receipt: dict[str, Any] | None,
    collected: CollectedSnapshots,
    selection: PalaceResultSelection,
) -> Literal["complete", "partial"]:
    """Complete when the receipt says completed and the run's result is the readable selection.

    The result is the final tables, or for a solver stopped at convergence its last pass, bound by the receipt.
    """
    receipt_status = receipt.get("status") if receipt is not None else None
    final = (
        collected.final_snapshot_status == "readable"
        and selection.selected_source == "final"
    )
    stopped = (
        selection.reason == "last_pass_of_a_stopped_run"
        and selection.integrity == "receipt_bound"
    )
    return (
        "complete"
        if receipt_status == "completed" and (final or stopped)
        else "partial"
    )


def _driven_trust_report(
    root: Path,
    run: _InspectedRun,
    resolved: ResolvedRunData | None,
    theme: ReportTheme,
) -> DrivenTrustReport:
    """The trust report of a Driven run: identity, failure, sweep and cost, and a resolved run's S-parameters.

    A resolved run is complete, and its result is its ``driven_result()``; a run ``driven_result()`` refuses,
    an adaptive sweep that reached its maximum for a port say, is still reported, with the refusal in place of
    the S-parameters.  A folder read as it came back is complete when the receipt says completed and the top
    level of ``results/palace`` is a solve of its own; its S-parameters are not read.
    """
    palace_json = (
        resolved.provenance.palace_json
        if resolved is not None
        else read_optional_json(root / NATIVE_METADATA)
    )
    returned = returned_passes(root / OUTPUT_DIR)
    status = run.receipt.get("status") if run.receipt is not None else None
    detached, refusal = _detached_result(resolved)
    cost = cost_cards(palace_json, resolved)
    return DrivenTrustReport(
        run_dir=root,
        route=run.route,
        profile=run.profile,
        completeness="complete"
        if resolved is not None or (status == "completed" and returned.final == "pass")
        else "partial",
        failure=failure_diagnosis(root, run.receipt, run.receipt_paths),
        identity=_report_identity(cost, run.handoff, run.receipt, resolved),
        passes_recorded=len(returned.passes),
        sweep=None if run.config is None else requested_sweep(run.config),
        finite_element_order=_finite_element_order(run.config),
        full_solves=_full_solves(
            palace_json.get("LinearSolver") if palace_json is not None else None
        ),
        cost=cost,
        durations=recorded_durations(palace_json),
        provenance=_report_provenance(
            run.config, run.handoff, run.mesh_manifest, run.problem, run.receipt
        ),
        convergence=resolved.convergence
        if resolved is not None
        else check_passes(read_plan(root, run.handoff, run.problem), root),
        result=detached,
        readback_refusal=refusal,
        solver_end=None if resolved is None else resolved.solver_end,
        theme=checked_theme(theme),
    )


def _detached_result(
    resolved: ResolvedRunData | None,
) -> tuple[PalaceDrivenResult | None, str | None]:
    """A resolved run's ``driven_result()``, or why it refused the run; neither for a folder not resolved."""
    if resolved is None:
        return None, None
    try:
        return driven_result(resolved), None
    except ValueError as error:
        return None, str(error)


def _finite_element_order(config: Mapping[str, Any] | None) -> int | None:
    """The FEM order the config solves at (``Solver.Order``); None when the folder has no config."""
    solver = config.get("Solver") if config is not None else None
    return as_int(solver.get("Order")) if isinstance(solver, Mapping) else None


def _full_solves(linear_solver: object) -> int | None:
    """Palace's count of linear solves (``LinearSolver.TotalSolves``), one per excited port and solved frequency."""
    return (
        as_int(linear_solver.get("TotalSolves"))
        if isinstance(linear_solver, Mapping)
        else None
    )


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
) -> ReportProvenance:
    """Preserve source/resource identities and detached mesh observations for report inspection."""
    recorded = {
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
    # The handoff is read as JSON, so its fields are typed by the record they fill.
    provenance = cast(ReportProvenance, recorded)
    provenance["mesh_summary"] = recorded_mesh_summary(
        mesh_manifest, handoff=handoff, config=config, problem=problem
    )
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
            f"{receipt.get('status', 'unknown')}, {receipt.get('ending', 'ending unknown')} "
            f"/ exit {receipt.get('exit_code', '?')}"
        )
    elif resolved is not None:
        receipt_status = (
            f"{resolved.returned_receipt.status}, {resolved.returned_receipt.ending} "
            f"/ exit {resolved.returned_receipt.exit_code}"
        )
    handoff_id = str(handoff.get("handoff_id") or "")
    return {
        "handoff_id": handoff_id,
        "handoff_id_short": handoff_id[:12] if handoff_id else None,
        "resources": resources,
        "receipt": receipt_status,
        "git_tag": cost.get("git_tag"),
    }
