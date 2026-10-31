"""Notebook display of recorded AEDT benchmark evidence.

Owns: turning the recorded native profile tree into table rows, series and Plotly
figures. Does not own: collecting the profile or attributing memory
(``scgsim.aedt._benchmark``).
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Any

from scgsim.aedt._benchmark import attribute_profile_memory
from scgsim.presentation.notebook import (
    cards,
    checked_theme,
    details,
    json_details,
    section,
    style_figure,
    table,
)

_TIME_METRICS = ("real_seconds", "elapsed_seconds")


@dataclass(frozen=True)
class _Position:
    """Where the walk over the native profile tree currently is."""

    profile: str
    process_group: str
    branch: str
    pass_id: int | None = None
    inside_adaptive_meshing: bool = False


def _benchmark_view(data: dict[str, Any], *, show_details: bool) -> dict[str, Any]:
    """Project the recorded native profile tree into display-only rows and series."""

    benchmark = data["benchmark"]
    if benchmark["status"] != "complete":
        return {"table_rows": [], "adaptive_series": [], "sweep_series": []}
    rows, pass_rows = _profile_rows(benchmark["profiles"], data["mode"])
    summary_rows = pass_rows + [
        row
        for row in rows
        if row["pass"] is None and row["scope"] in {"sweep", "subproblem"}
    ]
    table_rows = rows if show_details else (summary_rows or rows)
    adaptive_series: list[dict[str, Any]] = []
    sweep_series: list[dict[str, Any]] = []
    if table_rows:
        adaptive_series = _adaptive_series(rows, benchmark["profiles"])
        sweep_series = _sweep_series(rows, benchmark["profiles"])
    return {
        "table_rows": table_rows,
        "adaptive_series": adaptive_series,
        "sweep_series": sweep_series,
    }


def _profile_rows(
    profiles: list[dict[str, Any]], mode: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Flatten every profile's stage tree into (all rows, adaptive-pass rows)."""
    rows: list[dict[str, Any]] = []
    pass_rows: list[dict[str, Any]] = []
    for profile in profiles:
        for group in profile["process_groups"]:
            start = _Position(profile["native_setup_profile"], group["name"], "stage")
            _visit_stage(group, start, mode, rows, pass_rows)
    return rows, pass_rows


def _visit_stage(
    node: dict[str, Any],
    where: _Position,
    mode: str,
    rows: list[dict[str, Any]],
    pass_rows: list[dict[str, Any]],
) -> None:
    name = str(node["name"])
    pass_id = node.get("adaptive_pass", where.pass_id)
    inside_adaptive_meshing = (
        where.inside_adaptive_meshing or name == "Adaptive Meshing Group"
    )
    native_pass_group = False
    if "adaptive_pass" not in node and mode == "q3d" and inside_adaptive_meshing:
        match = re.fullmatch(r"Pass (\d+) Group", name)
        if match is not None:
            # This literal native group owns its descendant stages, not all problems.
            pass_id = int(match[1])
            native_pass_group = True
    kind = _stage_scope(name, pass_id, mode, where.branch)
    if node.get("metrics"):
        rows.append(
            {
                "profile": where.profile,
                "process_group": where.process_group,
                "scope": kind,
                "stage": name,
                "pass": pass_id,
                "frequency_native": node.get("native_properties", {}).get("Frequency"),
                **node["metrics"],
            }
        )
    if "adaptive_pass" in node or native_pass_group:
        peak = node.get("stage_memory_peak", {})
        pass_rows.append(
            {
                "profile": where.profile,
                "process_group": where.process_group,
                "scope": "adaptive",
                "stage": name,
                "pass": pass_id,
                "memory_native": peak.get("memory_native"),
                "memory_scope": peak.get("scope"),
                **node["metrics"],
            }
        )
    below = _Position(
        where.profile, where.process_group, kind, pass_id, inside_adaptive_meshing
    )
    for child in node.get("children", []):
        _visit_stage(child, below, mode, rows, pass_rows)


def _stage_scope(name: str, pass_id: int | None, mode: str, branch: str) -> str:
    """Label a stage adaptive, sweep or subproblem; otherwise inherit the parent's."""
    if pass_id is not None:
        return "adaptive"
    if "sweep" in name.lower():
        return "sweep"
    if mode in {"q2d", "q3d"}:
        return "subproblem"
    return branch


def _reports_metric(row: dict[str, Any], metric: str) -> bool:
    """Elapsed time is plotted only for rows that have no real time to plot."""
    return metric in row and (metric == "real_seconds" or "real_seconds" not in row)


def _adaptive_series(
    rows: list[dict[str, Any]], profiles: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """One time-versus-pass line for each stage of each process group."""
    series: list[dict[str, Any]] = []
    for profile in profiles:
        for group in profile["process_groups"]:
            stages = {
                row["stage"]
                for row in rows
                if row["profile"] == profile["native_setup_profile"]
                and row["process_group"] == group["name"]
                and row["scope"] == "adaptive"
                and row["pass"] is not None
                and ("real_seconds" in row or "elapsed_seconds" in row)
            }
            for stage in sorted(stages):
                for metric in _TIME_METRICS:
                    selected = [
                        row
                        for row in rows
                        if row["profile"] == profile["native_setup_profile"]
                        and row["process_group"] == group["name"]
                        and row["stage"] == stage
                        and row["pass"] is not None
                        and _reports_metric(row, metric)
                    ]
                    if selected:
                        series.append(
                            {
                                "x": [row["pass"] for row in selected],
                                "y": [row[metric] for row in selected],
                                "name": f"{group['name']} / {stage} ({metric.removesuffix('_seconds')})",
                            }
                        )
    return series


def _sweep_series(
    rows: list[dict[str, Any]], profiles: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """One time-versus-frequency line for each process group."""
    series: list[dict[str, Any]] = []
    for profile in profiles:
        for group in profile["process_groups"]:
            for metric in _TIME_METRICS:
                selected = [
                    row
                    for row in rows
                    if row["profile"] == profile["native_setup_profile"]
                    and row["process_group"] == group["name"]
                    and row["scope"] == "sweep"
                    and row["frequency_native"] is not None
                    and _reports_metric(row, metric)
                ]
                if selected:
                    series.append(
                        {
                            "x": [row["frequency_native"] for row in selected],
                            "y": [row[metric] for row in selected],
                            "name": f"{group['name']} ({metric.removesuffix('_seconds')})",
                        }
                    )
    return series


def _legend_label(name: str) -> str:
    """Insert a visual break without changing the recorded benchmark label."""

    return html.escape(name).replace(" / ", " /<br> ", 1)


def _style_benchmark_figure(figure: Any, *, theme: str) -> Any:
    """Keep each complete trace label visible below the native Plotly chart."""

    legend_rows = len(figure.data)
    legend_extra = 36 * legend_rows
    style_figure(figure, theme=theme, height=460 + legend_extra)
    figure.update_layout(
        legend={
            "orientation": "v",
            "x": 0,
            "xanchor": "left",
            "y": -0.22,
            "yanchor": "top",
        },
        margin={"b": 115 + legend_extra},
        hoverlabel={"namelength": -1},
    )
    return figure


def display_benchmark(
    data: dict[str, Any], *, show_details: bool, theme: str = "light"
) -> None:
    """Display recorded benchmark evidence; preserve the table without Plotly."""

    checked_theme(theme)
    data = {**data, "benchmark": attribute_profile_memory(data["benchmark"])}
    from IPython.display import HTML, display

    benchmark = data["benchmark"]
    display(
        HTML(
            section(
                "AEDT simulation benchmark",
                cards(
                    (
                        ("Analysis", data["mode"]),
                        ("Native profile", benchmark["status"]),
                        ("SCGSim execution (s)", data["execution_seconds"]),
                        ("Project bytes", data["project_bytes"]),
                        ("Primary CSV bytes", data["primary_csv_bytes"]),
                    )
                ),
                theme=theme,
            )
        )
    )
    if benchmark["status"] != "complete":
        if benchmark.get("reason"):
            display(
                HTML(
                    section(
                        "Native profile unavailable",
                        f"<p>{html.escape(str(benchmark['reason']))}</p>",
                        theme=theme,
                    )
                )
            )
        return
    view = _benchmark_view(data, show_details=show_details)
    if view["table_rows"]:
        _display_stage_table(view["table_rows"], show_details, theme)
        _display_time_figures(view, theme)
    if show_details:
        display(
            HTML(
                section(
                    "Native benchmark details",
                    json_details("Recorded native profile", benchmark, opened=True),
                    theme=theme,
                )
            )
        )


def _display_stage_table(
    table_rows: list[dict[str, Any]], show_details: bool, theme: str
) -> None:
    from IPython.display import HTML, display

    headings = (
        "profile",
        "process_group",
        "scope",
        "pass",
        "stage",
        "real_seconds",
        "elapsed_seconds",
        "cpu_seconds",
        "memory_native",
        "memory_scope",
        "tetrahedra",
        "solved_elements",
        "elements",
        "linear_matrix_size",
    )
    body = table(
        headings, [tuple(row.get(key) for key in headings) for row in table_rows]
    )
    display(
        HTML(
            section(
                "Native stage and pass evidence",
                details(
                    "Native profile table",
                    body,
                    opened=show_details or len(table_rows) <= 20,
                ),
                theme=theme,
            )
        )
    )


def _display_time_figures(view: dict[str, Any], theme: str) -> None:
    """Plot time per adaptive pass and per sweep frequency, if Plotly is installed."""
    from IPython.display import HTML, display

    try:
        import plotly.graph_objects as go
    except ImportError:
        display(
            HTML(
                section(
                    "Benchmark figures unavailable",
                    "<p>Plotly is unavailable; the native pass table remains available.</p>",
                    theme=theme,
                )
            )
        )
        return
    for series_list, x_title in (
        (view["adaptive_series"], "Native adaptive pass"),
        (view["sweep_series"], "Native sweep frequency"),
    ):
        figure = go.Figure()
        for series in series_list:
            figure.add_scatter(
                x=series["x"],
                y=series["y"],
                mode="lines+markers",
                name=_legend_label(series["name"]),
            )
        if figure.data:
            figure.update_layout(
                xaxis_title=x_title,
                yaxis_title="Native reported time (s; real/elapsed labelled)",
            )
            display(_style_benchmark_figure(figure, theme=theme))
