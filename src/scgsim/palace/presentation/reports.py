"""Palace-specific report projections, displays, and detached figures.

Trust answers whether a run is usable from identity and AMR evidence. Latest
physical interpretation and simulation cost remain explicit separate calls.
"""

from __future__ import annotations

import html
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from ..results._inspection import _failure_payload, _selection_payload
from ..results._surface_epr_reader import (
    _read_surface_mask_bindings,
    _surface_mask_snapshots,
    _surface_snapshots,
)
from ..results.report_data import (
    AmrPassSnapshot,
    PalaceFailureDiagnosis,
    PalaceResultSelection,
    PassCostRecord,
    SurfaceEprSeriesSnapshot,
    SurfaceMaskEprSeriesSnapshot,
    _read_optional_json,
)
from ..results.resolve import ResolvedPalaceResult
from ._benchmark import (
    benchmark_cards_html,
    benchmark_pass_figures,
    benchmark_pass_table_html,
    benchmark_time_figure,
    pass_cost_records,
)
from ._figures import (
    _PLOTLY_CONFIG,
    _SURFACE_TYPES,
    ReportTheme,
    _capacitance_traces,
    _checked_theme,
    _delta_traces,
    _line_figure,
    _mode_traces,
    _series_label,
    _surface_mask_participation_figure,
    _surface_percentage_figure,
    _surface_ranking_figure,
    _surface_total_figure,
    _traces_are_finite,
    _yaxis_type,
)
from ._html import (
    _fmt,
    _mesh_summary_html,
    _native_positive_infinity_q_notice,
    _surface_loss_html,
    _surface_mask_html,
    card_html,
)
from ._report_build import (
    TrustIdentity,
    _cost_cards,
    _durations,
    _resolved_benchmark_data,
    _trust_report_fields,
)

_FREQ_HEADER = "Re{f} (GHz)"
_SKIP_EIG_COLUMNS = {"Error (Bkwd.)", "Error (Abs.)"}
_ERROR_INDICATOR_TRACES = ("Norm", "Maximum", "Mean")


@dataclass(frozen=True)
class PalaceTrustReport:
    """Run-identity, AMR evidence, and cost for one Palace package folder."""

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
    theme: ReportTheme = "light"
    show_details: bool = False

    def mesh_summary(self) -> dict[str, Any]:
        """Read the same detached mesh authority used by resolved results."""
        import copy

        return copy.deepcopy(self.provenance["mesh_summary"])

    def with_theme(self, theme: ReportTheme) -> PalaceTrustReport:
        """Return this report with the plot theme changed (``light`` or ``dark``)."""
        checked = _checked_theme(theme)
        return self if self.theme == checked else replace(self, theme=checked)

    def show_run_trustworthiness(
        self, *, theme: ReportTheme = "light", show_details: bool = False
    ) -> PalaceTrustReport:
        """Return this complete or partial trust report with the selected theme."""

        report = self.with_theme(theme)
        checked = _checked_show_details(show_details)
        return (
            report
            if report.show_details == checked
            else replace(report, show_details=checked)
        )

    def show_simulation_benchmark(
        self, *, show_details: bool = False
    ) -> SimulationBenchmarkReport:
        """Return the readable benchmark while retaining every machine field."""

        attempted = _read_optional_json(self.run_dir / "results/palace/palace.json")
        data = {
            "cost": dict(self.cost),
            "mesh_summary": self.mesh_summary(),
            "performance": {"counts": {}, "durations": dict(self.durations)},
            "attempted_run": {
                "cost": _cost_cards(attempted, None),
                "durations": _durations(attempted),
            },
            "selected_snapshot": {
                "source": self.selection.selected_source,
                "cost": dict(self.cost),
                "durations": dict(self.durations),
            },
            "resources": {
                "requested_resources": self.provenance.get("requested_resources", {}),
                "resolved_resources": self.provenance.get("resolved_resources", {}),
            },
            "performance_metadata": {
                "route": self.route,
                "problem": self.problem,
                "status": self.identity.get("receipt"),
                "completeness": self.completeness,
                "latest_source": self.latest_source,
                "selection": _selection_payload(self.selection),
                "failure": _failure_payload(self.failure),
            },
        }
        return SimulationBenchmarkReport(
            trust=self,
            data=data,
            show_details=_checked_show_details(show_details),
        )

    def show_all_results(
        self,
        *,
        theme: ReportTheme = "light",
        ranking_limit: int | None = 20,
        show_details: bool = False,
    ) -> None:
        """Display trust, benchmark, and available physics in the shared order."""

        from IPython.display import display

        trust = self.show_run_trustworthiness(
            theme=theme,
            show_details=show_details,
        )
        display(trust)
        display(trust.show_simulation_benchmark(show_details=show_details))
        display(
            trust.show_physics_quantities(
                theme=theme,
                ranking_limit=ranking_limit,
            )
        )

    def _ipython_display_(self) -> None:
        from IPython.display import HTML, display

        display(HTML(self._identity_html()))
        items = self._convergence_items()
        if len(self.passes) >= 2:
            display(HTML(self._convergence_heading_html()))
        for item in items:
            if isinstance(item, str):
                display(HTML(item))
            else:
                _show_figure(item)
        for item in self._surface_convergence_items():
            if isinstance(item, str):
                display(HTML(item))
            else:
                _show_figure(item)
        if self.show_details:
            display(HTML(self._provenance_html()))

    def show_physics_quantities(
        self,
        *,
        theme: ReportTheme = "light",
        ranking_limit: int | None = 20,
    ) -> PhysicsQuantitiesReport:
        """Return latest readable physics without redrawing convergence."""

        return PhysicsQuantitiesReport(
            self.with_theme(theme),
            _checked_ranking_limit(ranking_limit),
        )

    def _identity_html(self) -> str:
        cards = (
            ("Problem", self.problem),
            ("Route", self.route),
            ("Profile", self.profile),
            ("Completeness", self.completeness),
            ("Latest snapshot", self.latest_source),
            ("Final snapshot", self.selection.final_snapshot_status),
            ("Selection integrity", self.selection.integrity),
            ("AMR passes recorded", str(len(self.passes))),
            ("AMR MaxIts", _fmt(self.amr_max_passes)),
            ("AMR Tol", _fmt(self.amr_tolerance)),
            ("MPI × OpenMP", self.identity.get("resources")),
            ("Receipt", self.identity.get("receipt")),
            ("Palace git", self.identity.get("git_tag")),
            ("Handoff id", self.identity.get("handoff_id_short")),
        )
        items = "".join(
            card_html(label, value) for label, value in cards if value not in {None, ""}
        )
        return (
            "<section><h3>Run Identity</h3>"
            "<p style='opacity:0.75;margin-top:0'>Package folder and solver identity. "
            "This is not a physics result.</p>"
            f"<div style='display:flex;flex-wrap:wrap'>{items}</div>"
            f"{self._run_state_html()}</section>"
        )

    def _run_state_html(self) -> str:
        notices: list[str] = []
        if self.failure is not None:
            notices.append(
                "<div style='border-left:4px solid #cf222e;background:rgba(207,34,46,0.10);"
                "padding:0.65rem 0.8rem;margin:0.7rem 0'>"
                f"<strong>Run failed — {html.escape(self.failure.category.replace('_', ' '))}</strong><br>"
                f"{html.escape(self.failure.summary)}<br>"
                f"Stage: {_fmt(self.failure.execution_stage)}; preflight: "
                f"{_fmt(self.failure.preflight_exit_code)}; solver invoked: "
                f"{_fmt(self.failure.solver_invoked)}; exit: "
                f"{_fmt(self.failure.exit_code)}; solver: "
                f"{_fmt(self.failure.solver_exit_code)}; output capture: "
                f"{_fmt(self.failure.tee_exit_code)}."
                "<details><summary>Failure evidence</summary><code>"
                f"{html.escape(json.dumps(self.failure.evidence, sort_keys=True))}"
                "</code></details></div>"
            )
        if self.selection.reason == "latest_complete_iteration_after_failed_attempt":
            source = self.selection.selected_source or "unavailable"
            path = self.selection.selected_path
            notices.append(
                "<div style='border-left:4px solid #bf8700;background:rgba(191,135,0,0.10);"
                "padding:0.65rem 0.8rem;margin:0.7rem 0'>"
                "<strong>Fallback result selected</strong><br>"
                f"Using {html.escape(source)} as partial evidence; the attempted final "
                f"snapshot is {html.escape(self.selection.final_snapshot_status)}. "
                f"Integrity: {html.escape(self.selection.integrity)}."
                + (
                    f"<br><code>{html.escape(str(path))}</code>"
                    if path is not None
                    else ""
                )
                + "</div>"
            )
        elif self.selection.reason == "no_complete_snapshot":
            notices.append(
                "<div style='border-left:4px solid #bf8700;background:rgba(191,135,0,0.10);"
                "padding:0.65rem 0.8rem;margin:0.7rem 0'>"
                "<strong>No complete result snapshot is available.</strong></div>"
            )
        return "".join(notices)

    def _convergence_items(self) -> list[Any]:
        if len(self.passes) < 2:
            items: list[Any] = [self._convergence_status_html()]
            q_traces = _mode_traces(
                self.passes,
                lambda pass_: (
                    None if pass_.eig_columns is None else pass_.eig_columns.get("Q")
                ),
            )
            q_notice = _native_positive_infinity_q_notice(q_traces)
            if q_notice is not None:
                items.append(q_notice)
            return items
        xs = tuple(pass_.pass_index for pass_ in self.passes)
        items: list[Any] = []
        for header in _union_mapping_keys(self.passes, "eig_columns"):
            if header in _SKIP_EIG_COLUMNS:
                continue
            traces = _mode_traces(
                self.passes,
                lambda pass_, column=header: (
                    None if pass_.eig_columns is None else pass_.eig_columns.get(column)
                ),
            )
            if header == "Q" and not _traces_are_finite(traces):
                items.append(
                    _native_positive_infinity_q_notice(traces)
                    or "<p style='opacity:0.75'>A finite Q convergence plot is "
                    "unavailable because the Q trace contains non-finite or "
                    "unavailable data.</p>"
                )
                continue
            figure = _line_figure(
                title=f"{header} vs AMR pass",
                xlabel="AMR pass",
                ylabel=header,
                xs=xs,
                traces=traces,
                yaxis_type=_yaxis_type(header),
                theme=self.theme,
            )
            if figure is not None:
                items.append(figure)
            if header == _FREQ_HEADER:
                delta = _line_figure(
                    title=f"|Δ {header}| vs AMR pass",
                    xlabel="AMR pass",
                    ylabel=f"|Δ {header}|",
                    xs=xs[1:],
                    traces=_delta_traces(traces),
                    yaxis_type="log",
                    theme=self.theme,
                )
                if delta is not None:
                    items.append(delta)
        for header in _union_mapping_keys(self.passes, "port_epr"):
            traces = _mode_traces(
                self.passes,
                lambda pass_, column=header: (
                    None if pass_.port_epr is None else pass_.port_epr.get(column)
                ),
            )
            figure = _line_figure(
                title=f"Port EPR {header} vs AMR pass",
                xlabel="AMR pass",
                ylabel=header,
                xs=xs,
                traces=traces,
                theme=self.theme,
            )
            if figure is not None:
                items.append(figure)
        for label, traces in _capacitance_traces(self.passes):
            figure = _line_figure(
                title=f"{label} vs AMR pass",
                xlabel="AMR pass",
                ylabel="fF",
                xs=xs,
                traces=traces,
                theme=self.theme,
            )
            if figure is not None:
                items.append(figure)
            if label == "Maxwell C_ij":
                continue
            delta = _line_figure(
                title=f"|Δ {label}| vs AMR pass",
                xlabel="AMR pass",
                ylabel="|Δ| (fF)",
                xs=xs[1:],
                traces=_delta_traces(traces),
                yaxis_type="log",
                theme=self.theme,
            )
            if delta is not None:
                items.append(delta)
        traces = [
            (
                name,
                [
                    None
                    if pass_.error_indicators is None
                    else pass_.error_indicators.get(name)
                    for pass_ in self.passes
                ],
            )
            for name in _ERROR_INDICATOR_TRACES
        ]
        hline = None
        if self.amr_tolerance is not None:
            hline = (self.amr_tolerance, "configured AMR Tol")
        figure = _line_figure(
            title="AMR error indicator vs AMR pass",
            xlabel="AMR pass",
            ylabel="error indicator",
            xs=xs,
            traces=traces,
            yaxis_type="log",
            hline=hline,
            theme=self.theme,
        )
        if figure is not None:
            items.append(figure)
        return items

    def _convergence_heading_html(self) -> str:
        return (
            "<section><h3>Numerical Evidence</h3>"
            "<p style='opacity:0.75;margin-top:0'>Each readable AMR scalar is its own "
            "figure. AMR error Norm, Maximum, and Mean share one log plot because they "
            "are the same local indicator. Minimum is omitted. The dashed line is the "
            "configured Palace AMR Tol, not a newly invented acceptance Gate. "
            "Surface participation convergence follows below; latest ranking and "
            "loss interpretation remain a separate physics call.</p>"
            "</section>"
        )

    def _convergence_status_html(self) -> str:
        latest = self.passes[-1] if self.passes else None
        quantity = _scalar_physics(latest, self.problem) if latest is not None else None
        if self.problem == "Eigenmode" and quantity is not None:
            quantity_line = f"Latest Re(f): {quantity:.6g} GHz"
        elif quantity is not None:
            quantity_line = f"Latest max |C_ii|: {quantity * 1e15:.6g} fF"
        else:
            quantity_line = "Latest physics quantity: unavailable"
        error_line = (
            f"Estimated error indicator (Norm): {_fmt(latest.error_norm)}"
            if latest is not None
            else "Estimated error indicator: unavailable"
        )
        body = (
            f"<p>{html.escape(quantity_line)}</p>"
            f"<p>{html.escape(error_line)}</p>"
            f"<p>Configured AMR Tol: {html.escape(_fmt(self.amr_tolerance))}</p>"
            f"<p>Configured MaxIts: {html.escape(_fmt(self.amr_max_passes))}</p>"
            f"<p>Latest snapshot: {html.escape(self.latest_source)}</p>"
        )
        title = (
            "No solver snapshots in results/palace"
            if not self.passes
            else "AMR stopped after the initial solve"
        )
        return (
            "<section><h3>Numerical Evidence</h3>"
            "<div style='border:1px solid var(--border,#d0d7de);border-radius:8px;"
            f"padding:0.9rem 1rem'><strong>{html.escape(title)}</strong>{body}</div>"
            "<p style='opacity:0.75'>A trend plot is omitted because a single pass "
            "cannot show adaptation.</p></section>"
        )

    def _surface_convergence_items(self) -> list[Any]:
        snapshots = _surface_snapshots(self.passes)
        if not snapshots:
            return [
                (
                    "<section><h3>Surface-EPR Numerical Convergence</h3>"
                    "<p style='opacity:0.75'>MA/MS/SA participation convergence is "
                    "unavailable because no readable structured surface-Q snapshot "
                    "was returned.</p></section>"
                )
            ]
        items: list[Any] = [
            (
                "<section><h3>Surface-EPR Numerical Convergence</h3>"
                "<p style='opacity:0.75;margin-top:0'>Each mode or excitation is kept "
                "separate. Percentages use the MA+MS+SA sum from the same source and "
                "AMR pass.</p></section>"
            )
        ]
        for series_index in sorted({item.series_index for item in snapshots}):
            series = tuple(
                item for item in snapshots if item.series_index == series_index
            )
            items.append(f"<h4>{html.escape(_series_label(series[-1]))}</h4>")
            total = _surface_total_figure(series, self.theme)
            if total is not None:
                items.append(total)
            items.append(_surface_percentage_figure(series, self.theme))
        return items

    def pass_cost_records(self) -> tuple[PassCostRecord, ...]:
        """One cost record per AMR pass: degrees of freedom, mesh elements, time and memory."""
        return pass_cost_records(self.passes)

    def _provenance_html(self) -> str:
        payload = html.escape(json.dumps(self.provenance, indent=2, sort_keys=True))
        return (
            "<section><h3>Provenance</h3>"
            "<p style='opacity:0.75;margin-top:0'>Exact source, Palace, input, and "
            "returned-run identities recorded by the handoff package.</p>"
            f"<pre style='white-space:pre-wrap'>{payload}</pre></section>"
        )


@dataclass(frozen=True)
class SimulationBenchmarkReport:
    """Human-readable simulation cost with complete machine data in ``data``."""

    trust: PalaceTrustReport
    data: dict[str, Any]
    show_details: bool = False

    def _ipython_display_(self) -> None:
        from IPython.display import HTML, display

        display(HTML(benchmark_cards_html(self.trust)))
        display(
            HTML(_mesh_summary_html(self.data["mesh_summary"], card_html=card_html))
        )
        timing = benchmark_time_figure(self.trust)
        if isinstance(timing, str):
            display(HTML(timing))
        else:
            _show_figure(timing)
        display(HTML(benchmark_pass_table_html(self.trust)))
        for figure in benchmark_pass_figures(self.trust):
            _show_figure(figure)
        if self.show_details:
            payload = html.escape(json.dumps(self.data, indent=2, sort_keys=True))
            display(
                HTML(
                    "<section><h4>Benchmark metadata</h4>"
                    "<p style='opacity:0.75;margin-top:0'>Complete machine-readable "
                    "cost, timing, resource, and run-state fields.</p>"
                    f"<pre style='white-space:pre-wrap'>{payload}</pre></section>"
                )
            )


@dataclass(frozen=True)
class PhysicsQuantitiesReport:
    """Latest readable physical quantities built from structured solver data."""

    trust: PalaceTrustReport
    ranking_limit: int | None = 20

    @property
    def snapshots(self) -> tuple[SurfaceEprSeriesSnapshot, ...]:
        """Baseline surface-EPR snapshots, one per AMR pass and mode or excitation."""
        return _surface_snapshots(self.trust.passes)

    @property
    def masked_snapshots(self) -> tuple[SurfaceMaskEprSeriesSnapshot, ...]:
        """All readable native mask passes; ``snapshots`` remains baseline-only."""
        return _surface_mask_snapshots(self.trust.passes)

    @property
    def masks_configured(self) -> bool:
        """Whether the run's index map declares any masked surface-EPR rows."""
        index_map = _read_optional_json(
            self.trust.run_dir / "metadata" / "palace_index_map.json"
        )
        return bool(_read_surface_mask_bindings(index_map))

    def _ipython_display_(self) -> None:
        from IPython.display import HTML, display

        display(HTML(self._heading_html()))
        snapshots = self.snapshots
        if not snapshots:
            display(
                HTML(
                    "<p style='opacity:0.75'>Surface-EPR physics is unavailable: "
                    "no readable surface-Q snapshot could be bound to structured "
                    "index-map semantics.</p>"
                )
            )
            return
        for series_index in sorted({item.series_index for item in snapshots}):
            series = tuple(
                item for item in snapshots if item.series_index == series_index
            )
            latest = series[-1]
            label = _series_label(latest)
            display(HTML(f"<h4>{html.escape(label)}</h4>"))
            ranking = _surface_ranking_figure(
                latest, self.ranking_limit, self.trust.theme
            )
            if ranking is not None:
                _show_figure(ranking)
            display(HTML(_surface_loss_html(latest, self.trust.problem)))
            mask_latest = _latest_mask_snapshot(
                self.masked_snapshots,
                pass_index=latest.pass_index,
                source=latest.source,
                series_index=latest.series_index,
            )
            if mask_latest is not None:
                for interface_type in _SURFACE_TYPES:
                    figure = _surface_mask_participation_figure(
                        mask_latest,
                        latest,
                        interface_type,
                        self.trust.theme,
                    )
                    if figure is not None:
                        _show_figure(figure)
                display(HTML(_surface_mask_html(mask_latest)))
            elif self.masks_configured:
                display(
                    HTML(
                        "<p style='opacity:0.75'>Native masked Surface EPR is "
                        "unavailable for the selected source/pass; baseline "
                        "Surface-EPR remains separately readable.</p>"
                    )
                )

    def _heading_html(self) -> str:
        state = (
            "complete returned run"
            if self.trust.completeness == "complete"
            else "partial / convergence not established"
        )
        limit = (
            "all surfaces" if self.ranking_limit is None else str(self.ranking_limit)
        )
        latest_source = self.trust.selection.selected_source or "unavailable"
        return (
            "<section><h3>Physics Quantities</h3>"
            "<p style='opacity:0.75;margin-top:0'>Latest-readable individual-surface "
            "participation and loss interpretation are shown only after run identity "
            "and numerical evidence. Modes and excitations remain separate. "
            f"Run state: {html.escape(state)}; latest readable snapshot: "
            f"{html.escape(latest_source)}; ranking limit: "
            f"{html.escape(limit)}; snapshot integrity: "
            f"{html.escape(self.trust.selection.integrity)}. Complete bound records "
            "remain available through "
            "<code>snapshots</code>.</p></section>"
        )


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


def _build_trust_report(
    root: Path,
    *,
    resolved: ResolvedPalaceResult | None = None,
    theme: ReportTheme = "light",
) -> PalaceTrustReport:
    return PalaceTrustReport(
        **_trust_report_fields(root, resolved=resolved, theme=theme)
    )


def _show_run_trustworthiness(
    result: ResolvedPalaceResult,
    *,
    theme: ReportTheme = "light",
    show_details: bool = False,
) -> PalaceTrustReport:
    if not isinstance(result, ResolvedPalaceResult):
        raise TypeError("resolved result report requires ResolvedPalaceResult.")
    report = _build_trust_report(result.run_dir, resolved=result, theme=theme)
    return report.show_run_trustworthiness(theme=theme, show_details=show_details)


def _show_physics_quantities(
    result: ResolvedPalaceResult,
    *,
    theme: ReportTheme = "light",
    ranking_limit: int | None = 20,
) -> PhysicsQuantitiesReport:
    report = _show_run_trustworthiness(result, theme=theme)
    return report.show_physics_quantities(
        theme=theme,
        ranking_limit=ranking_limit,
    )


def _show_all_results(
    result: ResolvedPalaceResult,
    *,
    theme: ReportTheme = "light",
    ranking_limit: int | None = 20,
    show_details: bool = False,
) -> None:
    """Display trust, benchmark, and physics in the Human-defined order."""

    if not isinstance(result, ResolvedPalaceResult):
        raise TypeError("resolved result report requires ResolvedPalaceResult.")

    from IPython.display import display

    trust = _show_run_trustworthiness(
        result,
        theme=theme,
        show_details=show_details,
    )
    display(trust)
    display(
        SimulationBenchmarkReport(
            trust=trust,
            data=_resolved_benchmark_data(result),
            show_details=_checked_show_details(show_details),
        )
    )
    display(
        trust.show_physics_quantities(
            theme=theme,
            ranking_limit=ranking_limit,
        )
    )


def _show_simulation_benchmark(
    result: ResolvedPalaceResult,
    *,
    show_details: bool = False,
) -> SimulationBenchmarkReport:
    """Return the readable benchmark and its complete machine data."""

    if not isinstance(result, ResolvedPalaceResult):
        raise TypeError("resolved result report requires ResolvedPalaceResult.")

    return SimulationBenchmarkReport(
        trust=_show_run_trustworthiness(result),
        data=_resolved_benchmark_data(result),
        show_details=_checked_show_details(show_details),
    )


def _latest_mask_snapshot(
    snapshots: Sequence[SurfaceMaskEprSeriesSnapshot],
    *,
    pass_index: int,
    source: str,
    series_index: int,
) -> SurfaceMaskEprSeriesSnapshot | None:
    return next(
        (
            snapshot
            for snapshot in reversed(snapshots)
            if snapshot.pass_index == pass_index
            and snapshot.source == source
            and snapshot.series_index == series_index
        ),
        None,
    )


def _scalar_physics(pass_: AmrPassSnapshot | None, problem: str) -> float | None:
    if pass_ is None:
        return None
    if problem == "Eigenmode":
        if not pass_.frequencies_ghz:
            return None
        return pass_.frequencies_ghz[0]
    if pass_.capacitance_matrix_f is None:
        return None
    return _max_abs_diagonal(pass_.capacitance_matrix_f)


def _max_abs_diagonal(matrix: Sequence[Sequence[float]]) -> float:
    values = []
    for index, row in enumerate(matrix):
        if index < len(row):
            values.append(abs(row[index]))
    return max(values) if values else 0.0


def _union_mapping_keys(
    passes: Sequence[AmrPassSnapshot], attribute: str
) -> tuple[str, ...]:
    keys: list[str] = []
    seen: set[str] = set()
    for pass_ in passes:
        mapping = getattr(pass_, attribute)
        if not isinstance(mapping, dict):
            continue
        for key in mapping:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    return tuple(keys)


def _show_figure(fig: Any) -> None:
    fig.show(config=_PLOTLY_CONFIG)


def _checked_ranking_limit(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("ranking_limit must be a positive integer or None.")
    return value


def _checked_show_details(value: bool) -> bool:
    if not isinstance(value, bool):
        raise TypeError("show_details must be a bool.")
    return value


__all__ = [
    "AmrPassSnapshot",
    "PalaceFailureDiagnosis",
    "PalaceResultSelection",
    "PalaceTrustReport",
    "PassCostRecord",
    "PhysicsQuantitiesReport",
    "inspect_run_trustworthiness",
]
