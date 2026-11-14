"""Palace run trust view: identity, convergence and cost display methods."""

from __future__ import annotations

import copy
import html
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

from scgsim.contracts.palace_convergence import ConvergenceCheck
from scgsim.contracts.palace_run_folder import (
    NATIVE_METADATA,
    OUTPUT_DIR,
    palace_problem,
)
from scgsim.palace.mesh.summary import MeshSummary
from scgsim.palace.reports.convergence import (
    declared_items,
    declared_threshold,
    pass_limit_text,
    solved_elements,
    verdict_card,
)
from scgsim.palace.reports.observations import (
    ERROR_INDICATOR_TRACES,
    FREQ_HEADER,
    SKIP_EIG_COLUMNS,
    CostCards,
    capacitance_traces,
    cost_cards,
    delta_traces,
    elapsed_pass_cost,
    mode_traces,
    native_positive_infinity_q_notice,
    recorded_durations,
    scalar_physics,
    traces_are_finite,
    union_mapping_keys,
)
from scgsim.palace.reports.plotting import (
    COLORWAY,
    PLOT_TICK_FONT,
    FigureStyle,
    LinePlot,
    PlotTheme,
    ReferenceLine,
    ReportTheme,
    Trace,
    axis_title,
    checked_ranking_limit,
    checked_show_details,
    checked_theme,
    fmt,
    fmt_mb,
    fmt_seconds,
    html_cell,
    plotly,
    show_figure,
    theme_tokens,
    yaxis_type,
)
from scgsim.palace.reports.surface_figures import (
    series_label,
    surface_percentage_figure,
    surface_total_figure,
)
from scgsim.palace.reports.trust_records import ReportIdentity, ReportProvenance
from scgsim.palace.results.artifacts import read_optional_json
from scgsim.palace.results.convergence import SolverEnd
from scgsim.palace.results.inspection import (
    failure_payload,
    selection_payload,
)
from scgsim.palace.results.inspection_records import (
    AmrPassSnapshot,
    PalaceFailureDiagnosis,
    PalaceResultSelection,
    PassCostRecord,
    pass_number,
)
from scgsim.palace.results.snapshots import (
    surface_snapshots,
)

from .views import (
    InspectedBenchmarkData,
    PhysicsQuantitiesReport,
    SimulationBenchmarkReport,
)

_DURATION_BARS = (
    "MeshPreprocessing",
    "Setup",
    "Construction",
    "OperatorConstruction",
    "Preconditioner",
    "LinearSolve",
    "EigenvalueSolve",
    "Div.-FreeProjection",
    "Solve",
    "Adaptation",
    "Rebalancing",
    "Postprocessing",
    "Paraview",
    "DiskIO",
)


@dataclass(frozen=True)
class PalaceTrustReport:
    """Run-identity, AMR evidence, and cost for one returned Palace run folder."""

    run_dir: Path
    problem: str
    route: str
    profile: str
    completeness: Literal["complete", "partial"]
    latest_source: str
    selection: PalaceResultSelection
    failure: PalaceFailureDiagnosis | None
    identity: ReportIdentity
    passes: tuple[AmrPassSnapshot, ...]
    amr_tolerance: float | None
    amr_max_passes: int | None
    durations: dict[str, float]
    cost: CostCards
    provenance: ReportProvenance
    convergence: ConvergenceCheck
    solver_end: SolverEnd | None = None
    theme: ReportTheme = "light"
    show_details: bool = False

    def mesh_summary(self) -> MeshSummary:
        """Read the same detached mesh authority used by resolved results."""

        return copy.deepcopy(self.provenance["mesh_summary"])

    def with_theme(self, theme: ReportTheme) -> PalaceTrustReport:
        """Return this report with validated display colors and unchanged source data."""
        checked = checked_theme(theme)
        return self if self.theme == checked else replace(self, theme=checked)

    def show_run_trustworthiness(
        self, *, theme: ReportTheme = "light", show_details: bool = False
    ) -> PalaceTrustReport:
        """Return this complete or partial trust report with the selected theme."""

        report = self.with_theme(theme)
        checked = checked_show_details(show_details)
        return (
            report
            if report.show_details == checked
            else replace(report, show_details=checked)
        )

    def show_simulation_benchmark(
        self,
        *,
        show_details: bool = False,
    ) -> SimulationBenchmarkReport:
        """Return the readable benchmark while retaining every machine field."""
        attempted = read_optional_json(self.run_dir / NATIVE_METADATA)
        pass_cost_records = self.pass_cost_records()
        data: InspectedBenchmarkData = {
            "cost": CostCards(**self.cost),
            "mesh_summary": self.mesh_summary(),
            "performance": {"counts": {}, "durations": dict(self.durations)},
            "attempted_run": {
                "cost": cost_cards(attempted, None),
                "durations": recorded_durations(attempted),
            },
            "selected_snapshot": {
                "source": self.selection.selected_source,
                "cost": CostCards(**self.cost),
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
                "selection": selection_payload(self.selection),
                "failure": failure_payload(self.failure),
            },
            "pass_cost_records": [asdict(record) for record in pass_cost_records],
        }
        return SimulationBenchmarkReport(
            trust=self,
            data=data,
            show_details=checked_show_details(show_details),
        )

    def show_all_results(
        self,
        *,
        theme: ReportTheme = "light",
        ranking_limit: int | None = 20,
        show_details: bool = False,
    ) -> None:
        """Display trust, benchmark, and available physics in the shared order."""

        from IPython.display import display  # noqa: PLC0415 -- IPython is optional; detached result reading needs no notebook runtime.

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

    def _tokens(self) -> PlotTheme:
        return theme_tokens(self.theme)

    def card_html(self, label: str, value: Any) -> str:
        """Escape and render one identity or cost label/value card."""
        return (
            "<div style='min-width:11rem;flex:1 1 11rem;border:1px solid var(--border,#d0d7de);"
            "border-radius:8px;padding:0.6rem 0.75rem;margin:0.25rem;'>"
            f"<div style='font-size:0.75rem;opacity:0.7'>{html.escape(str(label))}</div>"
            f"<div style='font-size:0.95rem;font-weight:600'>{html.escape(str(value))}</div>"
            "</div>"
        )

    def _ipython_display_(self) -> None:
        from IPython.display import HTML, display  # noqa: PLC0415 -- IPython is optional; detached result reading needs no notebook runtime.

        display(HTML(self.identity_html()))
        items = self._convergence_items()
        if len(self.passes) >= 2:
            display(HTML(self._convergence_heading_html()))
        items = [
            *declared_items(
                self.convergence,
                self.theme,
                self.solver_end,
                solved_elements=solved_elements(self.passes),
            ),
            *items,
        ]
        for item in items:
            if isinstance(item, str):
                display(HTML(item))
            else:
                show_figure(item)
        for item in self._surface_convergence_items():
            if isinstance(item, str):
                display(HTML(item))
            else:
                show_figure(item)
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
            checked_ranking_limit(ranking_limit),
        )

    def identity_html(self) -> str:
        """Render run identity together with failure and snapshot-selection notices."""
        cards = (
            ("Problem", self.problem),
            ("Route", self.route),
            ("Profile", self.profile),
            ("Completeness", self.completeness),
            ("Latest snapshot", self.latest_source),
            ("Final snapshot", self._final_snapshot_text()),
            ("Selection integrity", self.selection.integrity),
            ("AMR passes recorded", str(len(self.passes))),
            ("AMR MaxIts", pass_limit_text(self.amr_max_passes)),
            ("AMR Tol", self._tolerance_text()),
            ("Convergence", verdict_card(self.convergence, self.solver_end)),
            ("MPI × OpenMP", self.identity.get("resources")),
            ("Receipt", self.identity.get("receipt")),
            ("Palace git", self.identity.get("git_tag")),
            ("Handoff id", self.identity.get("handoff_id_short")),
        )
        items = "".join(
            self.card_html(str(label), value)
            for label, value in cards
            if value not in {None, ""}
        )
        return (
            "<section><h3>Run Identity</h3>"
            "<p style='opacity:0.75;margin-top:0'>Run folder and solver identity. "
            "This is not a physics result.</p>"
            f"<div style='display:flex;flex-wrap:wrap'>{items}</div>"
            f"{self._run_state_html()}</section>"
        )

    def _final_snapshot_text(self) -> str:
        """The top level's state; a run stopped at convergence has no final solve there, by design."""
        if self.selection.reason == "last_pass_of_a_stopped_run":
            return "none: stopped at convergence"
        return self.selection.final_snapshot_status

    def _run_state_html(self) -> str:
        notices: list[str] = []
        if self.selection.reason == "last_pass_of_a_stopped_run":
            notices.append(
                "<div style='border-left:4px solid #1a7f37;background:rgba(26,127,55,0.10);"
                "padding:0.65rem 0.8rem;margin:0.7rem 0'>"
                "<strong>Stopped at convergence</strong><br>"
                f"The result is the last pass Palace completed, "
                f"{html.escape(self.selection.selected_source or 'unavailable')}: the convergence runner "
                "stopped Palace after it, so the top level holds no final solve. Integrity: "
                f"{html.escape(self.selection.integrity)}.</div>"
            )
        if self.failure is not None:
            notices.append(
                "<div style='border-left:4px solid #cf222e;background:rgba(207,34,46,0.10);"
                "padding:0.65rem 0.8rem;margin:0.7rem 0'>"
                f"<strong>Run failed — {html.escape(self.failure.category.replace('_', ' '))}</strong><br>"
                f"{html.escape(self.failure.summary)}<br>"
                f"Stage: {fmt(self.failure.execution_stage)}; preflight: "
                f"{fmt(self.failure.preflight_exit_code)}; solver invoked: "
                f"{fmt(self.failure.solver_invoked)}; exit: "
                f"{fmt(self.failure.exit_code)}; solver: "
                f"{fmt(self.failure.solver_exit_code)}; output capture: "
                f"{fmt(self.failure.tee_exit_code)}."
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
            q_traces = mode_traces(
                self.passes,
                lambda pass_: (
                    None if pass_.eig_columns is None else pass_.eig_columns.get("Q")
                ),
            )
            q_notice = native_positive_infinity_q_notice(q_traces)
            if q_notice is not None:
                items.append(q_notice)
            return items
        xs = tuple(pass_.pass_number for pass_ in self.passes)
        items: list[Any] = []
        items.extend(self._eigenmode_convergence_items(xs))
        items.extend(self._port_epr_figures(xs))
        items.extend(self._capacitance_figures(xs))
        items.extend(self._error_indicator_figures(xs))
        return items

    def _port_epr_figures(self, xs: tuple[int, ...]) -> list[Any]:
        """One figure per recorded port-EPR column, each mode a trace across the AMR passes."""
        items: list[Any] = []
        for header in union_mapping_keys(self.passes, "port_epr"):
            traces = mode_traces(
                self.passes,
                lambda pass_, column=header: (
                    None if pass_.port_epr is None else pass_.port_epr.get(column)
                ),
            )
            figure = LinePlot(
                title=f"Port EPR {header} vs AMR pass",
                xlabel="AMR pass",
                ylabel=header,
                xs=xs,
                traces=traces,
                theme=self.theme,
            ).figure()
            if figure is not None:
                items.append(figure)
        return items

    def _capacitance_figures(self, xs: tuple[int, ...]) -> list[Any]:
        """The Maxwell capacitance coefficients across the AMR passes, with their pass-to-pass changes."""
        items: list[Any] = []
        for label, traces in capacitance_traces(self.passes):
            figure = LinePlot(
                title=f"{label} vs AMR pass",
                xlabel="AMR pass",
                ylabel="fF",
                xs=xs,
                traces=traces,
                theme=self.theme,
            ).figure()
            if figure is not None:
                items.append(figure)
            if label == "Maxwell C_ij":
                continue
            delta = LinePlot(
                title=f"|Δ {label}| vs AMR pass",
                xlabel="AMR pass",
                ylabel="|Δ| (fF)",
                xs=xs[1:],
                traces=delta_traces(traces),
                yaxis_type="log",
                theme=self.theme,
            ).figure()
            if delta is not None:
                items.append(delta)
        return items

    def _error_indicator_figures(self, xs: tuple[int, ...]) -> list[Any]:
        """The AMR error indicators across the passes, against the configured tolerance."""
        items: list[Any] = []
        traces = [
            Trace(
                name,
                [
                    None
                    if pass_.error_indicators is None
                    else pass_.error_indicators.get(name)
                    for pass_ in self.passes
                ],
            )
            for name in ERROR_INDICATOR_TRACES
        ]
        hline = None
        # With a declaration the configured Tol only keeps Palace from stopping; the line is the declared one.
        declared = declared_threshold(self.convergence, "error_indicator")
        if declared is not None:
            hline = ReferenceLine(declared, "declared threshold")
        elif self.convergence.plan is None and self.amr_tolerance is not None:
            hline = ReferenceLine(self.amr_tolerance, "configured AMR Tol")
        figure = LinePlot(
            title="AMR error indicator vs AMR pass",
            xlabel="AMR pass",
            ylabel="error indicator",
            xs=xs,
            traces=traces,
            yaxis_type="log",
            hline=hline,
            theme=self.theme,
        ).figure()
        if figure is not None:
            items.append(figure)
        return items

    def _convergence_heading_html(self) -> str:
        dashed_line = (
            "The dashed line is the configured Palace AMR Tol, not a newly invented acceptance Gate. "
            if self.convergence.plan is None
            else "A dashed line is the threshold declared with set_convergence(). "
        )
        return (
            "<section><h3>Numerical Evidence</h3>"
            "<p style='opacity:0.75;margin-top:0'>Each readable AMR scalar is its own "
            "figure. AMR error Norm, Maximum, and Mean share one log plot because they "
            "are the same local indicator. Minimum is omitted. "
            f"{dashed_line}"
            "Surface participation convergence follows below; latest ranking and "
            "loss interpretation remain a separate physics call.</p>"
            "</section>"
        )

    def _tolerance_text(self) -> str:
        """Palace's AMR Tol, or that the declared criteria end the run in its place."""
        if self.convergence.plan is not None:
            return "set aside: convergence declared"
        return fmt(self.amr_tolerance)

    def _convergence_status_html(self) -> str:
        latest = self.passes[-1] if self.passes else None
        quantity = scalar_physics(latest, self.problem) if latest is not None else None
        if palace_problem(self.problem).solves_modes and quantity is not None:
            quantity_line = f"Latest Re(f): {quantity:.6g} GHz"
        elif quantity is not None:
            quantity_line = f"Latest max |C_ii|: {quantity * 1e15:.6g} fF"
        else:
            quantity_line = "Latest physics quantity: unavailable"
        error_line = (
            f"Estimated error indicator (Norm): {fmt(latest.error_norm)}"
            if latest is not None
            else "Estimated error indicator: unavailable"
        )
        body = (
            f"<p>{html.escape(quantity_line)}</p>"
            f"<p>{html.escape(error_line)}</p>"
            f"<p>Configured AMR Tol: {html.escape(self._tolerance_text())}</p>"
            f"<p>Configured MaxIts: {html.escape(fmt(self.amr_max_passes))}</p>"
            f"<p>Latest snapshot: {html.escape(self.latest_source)}</p>"
        )
        title = (
            f"No solver snapshots in {OUTPUT_DIR}"
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
        snapshots = surface_snapshots(self.passes)
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
            items.append(f"<h4>{html.escape(series_label(series[-1]))}</h4>")
            total = surface_total_figure(series, self.theme)
            if total is not None:
                items.append(total)
            items.append(surface_percentage_figure(series, self.theme))
        return items

    def benchmark_cards_html(self) -> str:
        """Render resource, solver-size and memory observations without adding missing values."""
        cards = (
            ("Total wall", fmt_seconds(self.durations.get("Total"))),
            ("Peak node memory", fmt_mb(self.cost.get("peak_node_memory_megabytes"))),
            ("Peak rank memory", fmt_mb(self.cost.get("peak_memory_megabytes"))),
            ("DOFs", fmt(self.cost.get("problem_degrees_of_freedom"))),
            ("Mesh elements", fmt(self.cost.get("mesh_elements"))),
            ("MPI × OpenMP", self.identity.get("resources")),
        )
        items = "".join(
            self.card_html(label, value)
            for label, value in cards
            if value not in {None, ""}
        )
        return (
            "<section><h3>Simulation Benchmark</h3>"
            "<p style='opacity:0.75;margin-top:0'>Cost and timing. This does not "
            "decide whether the physics is correct. Palace timers can overlap; they "
            "are not a partition of wall time.</p>"
            f"<div style='display:flex;flex-wrap:wrap'>{items}</div></section>"
        )

    def benchmark_time_figure(self) -> Any:
        """Plot recorded elapsed categories separately from mesh and convergence evidence."""
        labels: list[str] = []
        values: list[float] = []
        for name in _DURATION_BARS:
            duration = self.durations.get(name)
            if duration is None or duration <= 0:
                continue
            labels.append(name)
            values.append(duration)
        if not labels:
            return (
                "<p style='opacity:0.75'>Palace elapsed-time timers are unavailable "
                "in this run folder.</p>"
            )
        tokens = self._tokens()
        go = plotly()
        fig = go.Figure(
            data=[
                go.Bar(
                    x=values,
                    y=labels,
                    orientation="h",
                    marker={"color": COLORWAY[0], "opacity": 0.88},
                    text=[fmt_seconds(value) for value in values],
                    textposition="outside",
                    textfont={"color": tokens.muted, "size": PLOT_TICK_FONT},
                    hovertemplate="%{x:.3g} s<extra>%{y}</extra>",
                )
            ]
        )
        FigureStyle(
            title="Elapsed time by Palace timer",
            height=max(320, 42 * len(labels) + 110),
            margin={"l": 196, "r": 88, "t": 56, "b": 52},
            hovermode="closest",
            showlegend=False,
            theme=self.theme,
        ).apply(fig)
        fig.update_xaxes(title=axis_title("seconds", tokens), rangemode="tozero")
        fig.update_yaxes(categoryorder="total ascending")
        return fig

    def pass_cost_records(self) -> tuple[PassCostRecord, ...]:
        """Detach pass costs while only differencing adjacent native iterations."""
        records: list[PassCostRecord] = []
        previous_iteration: int | None = None
        previous_elapsed: float | None = None
        for pass_ in self.passes:
            iteration = pass_.native_iteration
            elapsed_pass, elapsed_reason = elapsed_pass_cost(
                iteration,
                pass_.elapsed_total_s,
                previous_iteration,
                previous_elapsed,
            )
            if iteration is None:
                previous_iteration = None
                previous_elapsed = None
            else:
                previous_iteration = iteration
                previous_elapsed = pass_.elapsed_total_s
            seconds_per_million = None
            if (
                elapsed_pass is not None
                and pass_.degrees_of_freedom is not None
                and pass_.degrees_of_freedom > 0
            ):
                seconds_per_million = elapsed_pass / (pass_.degrees_of_freedom / 1e6)
            records.append(
                PassCostRecord(
                    pass_index=pass_.pass_index,
                    source=pass_.source,
                    native_iteration=iteration,
                    degrees_of_freedom=pass_.degrees_of_freedom,
                    mesh_elements=pass_.mesh_elements,
                    elapsed_cumulative_s=pass_.elapsed_total_s,
                    elapsed_pass_s=elapsed_pass,
                    seconds_per_million_dof=seconds_per_million,
                    peak_node_memory_mb=pass_.peak_node_memory_mb,
                    elapsed_pass_reason=elapsed_reason,
                )
            )
        return tuple(records)

    def benchmark_pass_table_html(self) -> str:
        """Render pass-local mesh, solver-size, elapsed-time and memory observations."""
        records = self.pass_cost_records()
        if not records:
            return (
                "<p style='opacity:0.75'>No AMR pass cost snapshots are available.</p>"
            )
        header = (
            "<tr>"
            + "".join(
                html_cell("th", name)
                for name in (
                    "Pass",
                    "Snapshot",
                    "Native iteration",
                    "DOFs",
                    "Mesh elems",
                    "This pass",
                    "Pass-time status",
                    "Cumulative",
                    "s / MDoF",
                    "Peak node mem",
                )
            )
            + "</tr>"
        )
        rows = []
        for record in records:
            rows.append(
                "<tr>"
                + html_cell("td", str(pass_number(record.pass_index)))
                + html_cell("td", record.source)
                + html_cell("td", fmt(record.native_iteration))
                + html_cell("td", fmt(record.degrees_of_freedom))
                + html_cell("td", fmt(record.mesh_elements))
                + html_cell("td", fmt_seconds(record.elapsed_pass_s))
                + html_cell("td", record.elapsed_pass_reason or "available")
                + html_cell("td", fmt_seconds(record.elapsed_cumulative_s))
                + html_cell("td", fmt(record.seconds_per_million_dof))
                + html_cell("td", fmt_mb(record.peak_node_memory_mb))
                + "</tr>"
            )
        return (
            "<section><h4>Cost by AMR pass</h4>"
            "<p style='opacity:0.75;margin-top:0'>Palace <code>ElapsedTime.Total</code> "
            "is cumulative; this-pass time is the difference between snapshots. "
            "These rows are the per-run cost records to accumulate later. "
            "They do not decide physics correctness.</p>"
            "<table style='border-collapse:collapse;font-size:0.9rem'>"
            f"<thead>{header}</thead><tbody>{''.join(rows)}</tbody></table></section>"
        )

    def benchmark_pass_figures(self) -> list[Any]:
        """Plot the available per-pass solver sizes, timing and memory records."""
        records = self.pass_cost_records()
        if len(records) < 2:
            return []
        figures: list[Any] = []
        this_pass = [record.elapsed_pass_s for record in records]
        dofs = [record.degrees_of_freedom for record in records]
        labels = [record.source for record in records]
        dof_cost = LinePlot(
            title="This-pass wall time vs DOFs",
            xlabel="DOFs",
            ylabel="seconds this pass",
            xs=dofs,
            traces=[Trace("this pass", this_pass)],
            point_labels=labels,
            theme=self.theme,
        ).figure()
        if dof_cost is not None:
            figures.append(dof_cost)
        return figures

    def _provenance_html(self) -> str:
        payload = html.escape(json.dumps(self.provenance, indent=2, sort_keys=True))
        return (
            "<section><h3>Provenance</h3>"
            "<p style='opacity:0.75;margin-top:0'>Exact source, Palace, input, and "
            "returned-run identities recorded in the run folder.</p>"
            f"<pre style='white-space:pre-wrap'>{payload}</pre></section>"
        )

    def _eigenmode_convergence_items(self, xs: tuple[int, ...]) -> list[Any]:
        """Build native modal traces and frequency deltas in header order."""
        items: list[Any] = []
        for header in union_mapping_keys(self.passes, "eig_columns"):
            if header in SKIP_EIG_COLUMNS:
                continue
            traces = mode_traces(
                self.passes,
                lambda pass_, column=header: (
                    None if pass_.eig_columns is None else pass_.eig_columns.get(column)
                ),
            )
            if header == "Q" and not traces_are_finite(traces):
                items.append(
                    native_positive_infinity_q_notice(traces)
                    or "<p style='opacity:0.75'>A finite Q convergence plot is "
                    "unavailable because the Q trace contains non-finite or "
                    "unavailable data.</p>"
                )
                continue
            figure = LinePlot(
                title=f"{header} vs AMR pass",
                xlabel="AMR pass",
                ylabel=header,
                xs=xs,
                traces=traces,
                yaxis_type=yaxis_type(header),
                theme=self.theme,
            ).figure()
            if figure is not None:
                items.append(figure)
            if header == FREQ_HEADER:
                delta = LinePlot(
                    title=f"|Δ {header}| vs AMR pass",
                    xlabel="AMR pass",
                    ylabel=f"|Δ {header}|",
                    xs=xs[1:],
                    traces=delta_traces(traces),
                    yaxis_type="log",
                    theme=self.theme,
                ).figure()
                if delta is not None:
                    items.append(delta)
        return items
