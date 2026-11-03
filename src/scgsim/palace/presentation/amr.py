"""Render detached Palace AMR analysis without reading or changing solver state."""

from __future__ import annotations

import html
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from scgsim.palace.presentation.plotting import (
    COLORWAY,
    FigureStyle,
    ReportTheme,
    axis_title,
    checked_theme,
    plotly,
    show_figure,
    theme_tokens,
)
from scgsim.palace.results.amr_models import AmrMetric, PalaceAmrAnalysis


@dataclass(frozen=True)
class AmrFigureData:
    """Native coordinates, units, and explicit missing points for one AMR plot."""

    title: str
    xlabel: str
    ylabel: str
    xs: Sequence[int | None]
    traces: Sequence[tuple[str, Sequence[float | int | None]]]


def show_amr_analysis(
    analysis: PalaceAmrAnalysis,
    *,
    theme: ReportTheme = "light",
) -> PalaceAmrReport:
    """Return a detached display wrapper without recomputing the supplied analysis."""
    if not isinstance(analysis, PalaceAmrAnalysis):
        raise TypeError("AMR report requires a PalaceAmrAnalysis instance.")
    return PalaceAmrReport(analysis=analysis, theme=checked_theme(theme))


@dataclass(frozen=True)
class PalaceAmrReport:
    """A themed rendering wrapper retaining the exact detached AMR analysis."""

    analysis: PalaceAmrAnalysis
    theme: ReportTheme = "light"

    @property
    def data(self) -> dict[str, Any]:
        """Return the complete JSON-compatible analysis payload under its report key."""
        return {"amr_analysis": self.analysis.to_payload()}

    def to_payload(self) -> dict[str, Any]:
        """Return detached analysis data suitable for serialization."""
        return self.data

    @property
    def figures(self) -> dict[str, Any]:
        """Build lazily rendered AMR figures keyed by their distinct metrics."""
        return _amr_figures(self.analysis, self.theme)

    def tables_html(self) -> str:
        """Render exact per-solve and per-transition observations with availability reasons."""
        return _analysis_tables_html(self.analysis)

    def _ipython_display_(self) -> None:
        from IPython.display import HTML, display  # noqa: PLC0415 -- Notebook display is optional.  # pyright: ignore[reportMissingImports]

        display(HTML(self.tables_html()))
        for figure in self.figures.values():
            show_figure(figure)


def _amr_figures(
    analysis: PalaceAmrAnalysis,
    theme: ReportTheme,
) -> dict[str, Any]:
    """Render compatible metric families while retaining explicit gaps in each trace."""
    figures = _transition_figures(analysis, theme)
    figures.update(_pass_figures(analysis, theme))
    return figures


def _transition_figures(
    analysis: PalaceAmrAnalysis,
    theme: ReportTheme,
) -> dict[str, Any]:
    """Build figures from native refinement-transition observations."""
    figures: dict[str, Any] = {}
    transitions = analysis.transitions
    xs = [item.after_iteration for item in transitions]
    _add_transition_figure(
        figures,
        "marking_and_coverage",
        AmrFigureData(
            "Marked-cell fraction and squared-indicator coverage",
            "Native Palace iteration",
            "fraction of cells / squared indicator",
            xs,
            (
                (
                    "Marked-cell fraction",
                    [_available_value(item.marked_fraction) for item in transitions],
                ),
                (
                    "Squared-indicator coverage",
                    [
                        _available_value(item.marked_squared_indicator_coverage)
                        for item in transitions
                    ],
                ),
            ),
        ),
        theme,
    )
    _add_transition_figure(
        figures,
        "closure_amplification",
        AmrFigureData(
            "Closure and new-cell amplification",
            "Native Palace iteration",
            "ratio",
            xs,
            (
                (
                    "Closure parents / marked parents",
                    [
                        _available_value(item.closure_amplification)
                        for item in transitions
                    ],
                ),
                (
                    "New cells / marked parents",
                    [
                        _available_value(item.new_cell_amplification)
                        for item in transitions
                    ],
                ),
            ),
        ),
        theme,
    )
    _add_transition_figure(
        figures,
        "extra_closure_parents",
        AmrFigureData(
            "Extra closure parents (P − M)",
            "Native Palace iteration",
            "parent cells",
            xs,
            (
                (
                    "Extra parents",
                    [
                        _available_value(item.closure_extra_parents)
                        for item in transitions
                    ],
                ),
            ),
        ),
        theme,
    )
    _add_transition_figure(
        figures,
        "dof_growth_ratio",
        AmrFigureData(
            "True DOF growth ratio",
            "Native Palace iteration",
            "ratio",
            xs,
            (
                (
                    "DOF growth ratio",
                    [_available_value(item.dof_growth_ratio) for item in transitions],
                ),
            ),
        ),
        theme,
    )
    _add_transition_figure(
        figures,
        "dof_increase",
        AmrFigureData(
            "True DOF increase",
            "Native Palace iteration",
            "degrees of freedom",
            xs,
            (
                (
                    "DOF increase",
                    [_available_value(item.dof_increase) for item in transitions],
                ),
            ),
        ),
        theme,
    )
    return figures


def _pass_figures(
    analysis: PalaceAmrAnalysis,
    theme: ReportTheme,
) -> dict[str, Any]:
    """Build figures from genuine native solves and their capacitance comparisons."""
    figures: dict[str, Any] = {}
    transitions = analysis.transitions
    pass_xs = [item.native_iteration for item in analysis.passes]
    _add_pass_figure(
        figures,
        "native_zz_norm",
        AmrFigureData(
            "Native ZZ estimator norm",
            "Native Palace iteration",
            "native ZZ norm",
            pass_xs,
            [
                (
                    "native ZZ norm",
                    [_available_value(item.native_zz_norm) for item in analysis.passes],
                )
            ],
        ),
        theme,
    )
    _add_pass_figure(
        figures,
        "adjacent_capacitance_change",
        AmrFigureData(
            "Adjacent-pass capacitance change",
            "Native Palace iteration",
            "relative Frobenius change",
            [item.after_iteration for item in transitions],
            [
                (
                    "relative Frobenius change",
                    [
                        _available_value(item.adjacent_capacitance.relative_frobenius)
                        for item in transitions
                    ],
                )
            ],
        ),
        theme,
    )
    reference_values = [item.reference_capacitance for item in analysis.passes]
    if any(value is not None for value in reference_values):
        _add_pass_figure(
            figures,
            "explicit_reference_capacitance_error",
            AmrFigureData(
                "Capacitance difference from explicit reference",
                "Native Palace iteration",
                "relative Frobenius difference",
                pass_xs,
                [
                    (
                        "relative Frobenius difference",
                        [
                            None
                            if value is None
                            else _available_value(value.relative_frobenius)
                            for value in reference_values
                        ],
                    )
                ],
            ),
            theme,
        )
    return figures


def _add_transition_figure(
    figures: dict[str, Any],
    key: str,
    data: AmrFigureData,
    theme: ReportTheme,
) -> None:
    """Append one multi-series transition figure with native iteration coordinates."""
    _add_figure(figures, key, data, theme)


def _add_pass_figure(
    figures: dict[str, Any],
    key: str,
    data: AmrFigureData,
    theme: ReportTheme,
) -> None:
    """Append one pass figure while keeping unavailable observations as gaps."""
    _add_figure(figures, key, data, theme)


def _add_figure(
    figures: dict[str, Any],
    key: str,
    data: AmrFigureData,
    theme: ReportTheme,
) -> None:
    """Append one fully described AMR figure when at least one value is observed."""
    figure = _gap_preserving_figure(data, theme)
    if figure is not None:
        figures[key] = figure


def _gap_preserving_figure(
    data: AmrFigureData,
    theme: ReportTheme,
) -> Any | None:
    """Create a styled Plotly figure without joining across unavailable values."""
    if not any(value is not None for _, values in data.traces for value in values):
        return None
    tokens = theme_tokens(theme)
    go = plotly()
    figure = go.Figure()
    FigureStyle(title=data.title, theme=theme).apply(figure)
    for index, (name, values) in enumerate(data.traces):
        plot_xs, positioned_values = _native_plot_points(data.xs, values)
        figure.add_trace(
            go.Scatter(
                x=plot_xs,
                y=positioned_values,
                mode="lines+markers",
                name=name,
                connectgaps=False,
                line={"width": 3, "color": COLORWAY[index % len(COLORWAY)]},
                marker={"size": 9},
                hovertemplate="iteration %{x}<br>%{y:.6g}<extra>%{fullData.name}</extra>",
            )
        )
    figure.update_xaxes(title=axis_title(data.xlabel, tokens))
    figure.update_yaxes(title=axis_title(data.ylabel, tokens))
    return figure


def _native_plot_points(
    iterations: Sequence[int | None],
    values: Sequence[float | int | None],
) -> tuple[list[int | None], list[float | int | None]]:
    """Add null separators for skipped native iterations without making observations."""
    plot_iterations: list[int | None] = []
    plot_values: list[float | int | None] = []
    previous_iteration: int | None = None
    for iteration, value in zip(iterations, values, strict=False):
        if (
            previous_iteration is not None
            and iteration is not None
            and iteration > previous_iteration + 1
        ):
            for missing_iteration in range(previous_iteration + 1, iteration):
                plot_iterations.append(missing_iteration)
                plot_values.append(None)
        plot_iterations.append(iteration)
        plot_values.append(None if iteration is None else value)
        previous_iteration = iteration
    return plot_iterations, plot_values


def _analysis_tables_html(analysis: PalaceAmrAnalysis) -> str:
    """Present source identities, raw capacitance, solve state and metric availability."""
    pass_rows: list[str] = []
    for pass_analysis in analysis.passes:
        raw_capacitance = json.dumps(
            {
                "terminal_ids": pass_analysis.terminal_ids,
                "values_f": pass_analysis.capacitance_matrix_f,
            }
        )
        zz = pass_analysis.native_zz_norm
        solve_warning = pass_analysis.linear_solve_reason or ""
        pass_rows.append(
            "<tr>"
            + _cell(pass_analysis.native_iteration)
            + _cell(pass_analysis.source)
            + _cell(pass_analysis.linear_solve_state)
            + _cell(solve_warning)
            + _cell(_metric_text(zz))
            + _cell(raw_capacitance)
            + "</tr>"
        )
    transition_rows = []
    for transition in analysis.transitions:
        metrics = (
            ("marked fraction", transition.marked_fraction),
            ("indicator coverage", transition.marked_squared_indicator_coverage),
            ("closure amplification", transition.closure_amplification),
            ("extra closure parents", transition.closure_extra_parents),
            ("new-cell amplification", transition.new_cell_amplification),
            ("DOF growth ratio", transition.dof_growth_ratio),
            ("DOF increase", transition.dof_increase),
            ("adjacent C change", transition.adjacent_capacitance.relative_frobenius),
        )
        reason = transition.reason or ""
        metric_summary = "; ".join(
            f"{name}: {_metric_text(metric)}" for name, metric in metrics
        )
        transition_rows.append(
            "<tr>"
            + _cell(transition.before_iteration)
            + _cell(transition.after_iteration)
            + _cell(transition.availability)
            + _cell(reason)
            + _cell(metric_summary)
            + "</tr>"
        )
    return (
        "<section><h3>Adaptive Mesh Refinement Analysis</h3>"
        f"<p>Problem: {html.escape(analysis.problem)}; run status: "
        f"{html.escape(analysis.run_status)}; integrity: {html.escape(analysis.integrity)}.</p>"
        "<h4>Native solves</h4><table><thead><tr>"
        "<th>Native iteration</th><th>Source</th><th>Linear solve state</th>"
        "<th>Validity warning</th><th>Native ZZ norm</th><th>Raw C (F)</th>"
        f"</tr></thead><tbody>{''.join(pass_rows)}</tbody></table>"
        "<h4>Native transitions</h4><table><thead><tr>"
        "<th>Before iteration</th><th>After iteration</th><th>Availability</th>"
        "<th>Reason</th><th>Observed metrics (units in figures)</th>"
        f"</tr></thead><tbody>{''.join(transition_rows)}</tbody></table></section>"
    )


def _metric_text(metric: AmrMetric) -> str:
    """Show an available value with its declared unit or its explicit missing reason."""
    if metric.availability == "unavailable":
        return f"unavailable ({metric.reason or 'reason not recorded'})"
    return f"{metric.value} {metric.units}"


def _available_value(metric: AmrMetric) -> float | int | None:
    """Keep unavailable measurements as explicit Plotly gaps."""
    if metric.availability == "unavailable":
        return None
    return metric.value


def _cell(value: Any) -> str:
    """Escape one value for inclusion in a report table cell."""
    return f"<td>{html.escape(str(value))}</td>"
