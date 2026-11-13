"""Display a recorded AEDT benchmark as a table and Plotly figures."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Any

from scgsim.aedt.benchmark_data import attribute_profile_memory
from scgsim.presentation.notebook import (
    cards,
    checked_theme,
    details,
    json_details,
    section,
    style_figure,
    table,
)


@dataclass(frozen=True)
class _BenchmarkBranch:
    """Traversal context inherited by one native profile-tree branch."""

    scope: str
    pass_id: int | None = None
    inside_adaptive_meshing: bool = False


@dataclass
class _BenchmarkGroupProjection:
    """Collect display rows for one recorded native process group."""

    data: dict[str, Any]
    profile: str
    group: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    pass_rows: list[dict[str, Any]] = field(default_factory=list)

    def visit(self, node: dict[str, Any], branch: _BenchmarkBranch) -> None:
        """Project one node, then pass its derived scope to child nodes."""
        name = str(node["name"])
        pass_id = node.get("adaptive_pass", branch.pass_id)
        inside_adaptive = (
            branch.inside_adaptive_meshing or name == "Adaptive Meshing Group"
        )
        native_pass_group = self._native_pass_group(node, name, inside_adaptive)
        if native_pass_group is not None:
            pass_id = native_pass_group
        scope = self._node_scope(name, pass_id, branch.scope)
        self._record_metrics(node, name, scope, pass_id)
        self._record_pass_peak(node, name, pass_id, native_pass_group is not None)
        child_branch = _BenchmarkBranch(scope, pass_id, inside_adaptive)
        for child in node.get("children", []):
            self.visit(child, child_branch)

    def _native_pass_group(
        self, node: dict[str, Any], name: str, inside_adaptive: bool
    ) -> int | None:
        if "adaptive_pass" in node or self.data["mode"] != "q3d" or not inside_adaptive:
            return None
        match = re.fullmatch(r"Pass (\d+) Group", name)
        if match is None:
            return None
        # This literal native group owns its descendants, not every Q3D problem.
        return int(match[1])

    def _node_scope(self, name: str, pass_id: int | None, parent_scope: str) -> str:
        if pass_id is not None:
            return "adaptive"
        if "sweep" in name.lower():
            return "sweep"
        if self.data["mode"] in {"q2d", "q3d"}:
            return "subproblem"
        return parent_scope

    def _record_metrics(
        self, node: dict[str, Any], name: str, scope: str, pass_id: int | None
    ) -> None:
        if node.get("metrics"):
            self.rows.append(
                {
                    "profile": self.profile,
                    "process_group": self.group,
                    "scope": scope,
                    "stage": name,
                    "pass": pass_id,
                    "frequency_native": node.get("native_properties", {}).get(
                        "Frequency"
                    ),
                    **node["metrics"],
                }
            )

    def _record_pass_peak(
        self,
        node: dict[str, Any],
        name: str,
        pass_id: int | None,
        native_pass_group: bool,
    ) -> None:
        if "adaptive_pass" not in node and not native_pass_group:
            return
        peak = node.get("stage_memory_peak", {})
        self.pass_rows.append(
            {
                "profile": self.profile,
                "process_group": self.group,
                "scope": "adaptive",
                "stage": name,
                "pass": pass_id,
                "memory_native": peak.get("memory_native"),
                "memory_scope": peak.get("scope"),
                **node["metrics"],
            }
        )


def _benchmark_view(data: dict[str, Any], *, show_details: bool) -> dict[str, Any]:
    """Project the recorded native profile tree into display-only rows and series."""
    benchmark = data["benchmark"]
    if benchmark["status"] != "complete":
        return {"table_rows": [], "adaptive_series": [], "sweep_series": []}
    rows, pass_rows = _profile_rows(data, benchmark["profiles"])
    summary_rows = pass_rows + [
        row
        for row in rows
        if row["pass"] is None and row["scope"] in {"sweep", "subproblem"}
    ]
    table_rows = rows if show_details else (summary_rows or rows)
    adaptive_series, sweep_series = (
        _profile_series(rows, benchmark["profiles"]) if table_rows else ([], [])
    )
    return {
        "table_rows": table_rows,
        "adaptive_series": adaptive_series,
        "sweep_series": sweep_series,
    }


def _profile_rows(
    data: dict[str, Any], profiles: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect all profile rows in recorded profile and group order."""
    rows: list[dict[str, Any]] = []
    pass_rows: list[dict[str, Any]] = []
    for profile in profiles:
        for group in profile["process_groups"]:
            projection = _BenchmarkGroupProjection(
                data, profile["native_setup_profile"], group["name"]
            )
            projection.visit(group, _BenchmarkBranch("stage"))
            rows.extend(projection.rows)
            pass_rows.extend(projection.pass_rows)
    return rows, pass_rows


def _profile_series(
    rows: list[dict[str, Any]], profiles: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build adaptive-pass and frequency-sweep plotting projections."""
    adaptive = _adaptive_series(rows, profiles)
    sweeps = _sweep_series(rows, profiles)
    return adaptive, sweeps


def _adaptive_series(
    rows: list[dict[str, Any]], profiles: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Select the recorded timing series for each native adaptive stage."""
    series: list[dict[str, Any]] = []
    for profile in profiles:
        for group in profile["process_groups"]:
            group_rows = [
                row
                for row in rows
                if row["profile"] == profile["native_setup_profile"]
                and row["process_group"] == group["name"]
            ]
            stages = {
                row["stage"]
                for row in group_rows
                if row["scope"] == "adaptive"
                and row["pass"] is not None
                and ("real_seconds" in row or "elapsed_seconds" in row)
            }
            for stage in sorted(stages):
                for metric in ("real_seconds", "elapsed_seconds"):
                    selected = [
                        row
                        for row in group_rows
                        if row["stage"] == stage
                        and row["pass"] is not None
                        and metric in row
                        and (metric == "real_seconds" or "real_seconds" not in row)
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
    """Select native sweep timing values paired with recorded frequencies."""
    series: list[dict[str, Any]] = []
    for profile in profiles:
        for group in profile["process_groups"]:
            group_rows = [
                row
                for row in rows
                if row["profile"] == profile["native_setup_profile"]
                and row["process_group"] == group["name"]
            ]
            for metric in ("real_seconds", "elapsed_seconds"):
                selected = [
                    row
                    for row in group_rows
                    if row["scope"] == "sweep"
                    and row["frequency_native"] is not None
                    and metric in row
                    and (metric == "real_seconds" or "real_seconds" not in row)
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
    from IPython.display import HTML, display  # noqa: PLC0415 - notebook output is optional

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
    table_rows = view["table_rows"]
    if table_rows:
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
        try:
            import plotly.graph_objects as go  # noqa: PLC0415 - Plotly is an optional plot extra
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
        else:
            figure = go.Figure()
            for series in view["adaptive_series"]:
                figure.add_scatter(
                    x=series["x"],
                    y=series["y"],
                    mode="lines+markers",
                    name=_legend_label(series["name"]),
                )
            if figure.data:
                figure.update_layout(
                    xaxis_title="Native adaptive pass",
                    yaxis_title="Native reported time (s; real/elapsed labelled)",
                )
                display(_style_benchmark_figure(figure, theme=theme))
            sweep = go.Figure()
            for series in view["sweep_series"]:
                sweep.add_scatter(
                    x=series["x"],
                    y=series["y"],
                    mode="lines+markers",
                    name=_legend_label(series["name"]),
                )
            if sweep.data:
                sweep.update_layout(
                    xaxis_title="Native sweep frequency",
                    yaxis_title="Native reported time (s; real/elapsed labelled)",
                )
                display(_style_benchmark_figure(sweep, theme=theme))
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
