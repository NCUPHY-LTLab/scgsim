"""The cost and timing sections of a Palace run report.

Benchmark means cost, not correctness: wall time, memory, degrees of freedom (DOFs)
and how they grow over adaptive mesh refinement (AMR) passes. Each function takes a
``PalaceTrustReport`` and returns an HTML string or a figure. Display order and the
report classes are in ``reports.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ..results.report_data import AmrPassSnapshot, PassCostRecord
from ._figures import (
    COLORWAY,
    PLOT_TICK_FONT,
    axis_title,
    import_plotly,
    line_figure,
    style_figure,
    theme_tokens,
)
from ._html import card_html, fmt, fmt_mb, fmt_seconds, html_cell

if TYPE_CHECKING:
    from .reports import PalaceTrustReport

# Palace timers shown as bars, in display order. Palace timers can overlap, so the
# bars are not a partition of the total wall time.
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


def benchmark_cards_html(report: PalaceTrustReport) -> str:
    cards = (
        ("Total wall", fmt_seconds(report.durations.get("Total"))),
        ("Peak node memory", fmt_mb(report.cost.get("peak_node_memory_megabytes"))),
        ("Peak rank memory", fmt_mb(report.cost.get("peak_memory_megabytes"))),
        ("DOFs", fmt(report.cost.get("problem_degrees_of_freedom"))),
        ("Mesh elements", fmt(report.cost.get("mesh_elements"))),
        ("MPI × OpenMP", report.identity.get("resources")),
    )
    items = "".join(
        card_html(label, value) for label, value in cards if value not in {None, ""}
    )
    return (
        "<section><h3>Simulation Benchmark</h3>"
        "<p style='opacity:0.75;margin-top:0'>Cost and timing. This does not "
        "decide whether the physics is correct. Palace timers can overlap; they "
        "are not a partition of wall time.</p>"
        f"<div style='display:flex;flex-wrap:wrap'>{items}</div></section>"
    )


def benchmark_time_figure(report: PalaceTrustReport) -> Any:
    """A bar figure of the Palace timers, or an HTML notice when there are none."""
    labels: list[str] = []
    values: list[float] = []
    for name in _DURATION_BARS:
        duration = report.durations.get(name)
        if duration is None or duration <= 0:
            continue
        labels.append(name)
        values.append(duration)
    if not labels:
        return (
            "<p style='opacity:0.75'>Palace elapsed-time timers are unavailable "
            "in this package.</p>"
        )
    tokens = theme_tokens(report.theme)
    go = import_plotly()
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
    style_figure(
        fig,
        title="Elapsed time by Palace timer",
        height=max(320, 42 * len(labels) + 110),
        margin={"l": 196, "r": 88, "t": 56, "b": 52},
        hovermode="closest",
        showlegend=False,
        theme=report.theme,
    )
    fig.update_xaxes(title=axis_title("seconds", tokens), rangemode="tozero")
    fig.update_yaxes(categoryorder="total ascending")
    return fig


def pass_cost_records(passes: Sequence[AmrPassSnapshot]) -> tuple[PassCostRecord, ...]:
    """One cost record per AMR pass.

    Palace's ``ElapsedTime.Total`` is cumulative, so the time of one pass is the
    difference to the previous pass that reported a time.
    """
    records: list[PassCostRecord] = []
    previous_elapsed: float | None = None
    for pass_ in passes:
        elapsed_pass = None
        if pass_.elapsed_total_s is not None:
            if previous_elapsed is None:
                elapsed_pass = pass_.elapsed_total_s
            else:
                elapsed_pass = pass_.elapsed_total_s - previous_elapsed
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
                degrees_of_freedom=pass_.degrees_of_freedom,
                mesh_elements=pass_.mesh_elements,
                elapsed_cumulative_s=pass_.elapsed_total_s,
                elapsed_pass_s=elapsed_pass,
                seconds_per_million_dof=seconds_per_million,
                peak_node_memory_mb=pass_.peak_node_memory_mb,
            )
        )
    return tuple(records)


def benchmark_pass_table_html(report: PalaceTrustReport) -> str:
    records = pass_cost_records(report.passes)
    if not records:
        return "<p style='opacity:0.75'>No AMR pass cost snapshots are available.</p>"
    header = (
        "<tr>"
        + "".join(
            html_cell("th", name)
            for name in (
                "Pass",
                "Snapshot",
                "DOFs",
                "Mesh elems",
                "This pass",
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
            + html_cell("td", str(record.pass_index))
            + html_cell("td", record.source)
            + html_cell("td", fmt(record.degrees_of_freedom))
            + html_cell("td", fmt(record.mesh_elements))
            + html_cell("td", fmt_seconds(record.elapsed_pass_s))
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


def benchmark_pass_figures(report: PalaceTrustReport) -> list[Any]:
    records = pass_cost_records(report.passes)
    if len(records) < 2:
        return []
    figures: list[Any] = []
    dof_cost = line_figure(
        title="This-pass wall time vs DOFs",
        xlabel="DOFs",
        ylabel="seconds this pass",
        xs=[record.degrees_of_freedom for record in records],
        traces=[("this pass", [record.elapsed_pass_s for record in records])],
        point_labels=[record.source for record in records],
        theme=report.theme,
    )
    if dof_cost is not None:
        figures.append(dof_cost)
    return figures
