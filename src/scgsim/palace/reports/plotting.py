"""Shared Palace figure styling, themes and notebook display adapters."""

from __future__ import annotations

import html
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

COLORWAY = (
    "#56B4E9",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#0072B2",
    "#D55E00",
    "#F0E442",
)


_PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


_PLOT_HEIGHT = 460


_PLOT_FONT = 16


_PLOT_TITLE_FONT = 18


_PLOT_AXIS_FONT = 16


PLOT_TICK_FONT = 14


_PLOT_LEGEND_FONT = 15


ReportTheme = Literal["light", "dark"]


@dataclass(frozen=True)
class PlotTheme:
    """Resolved colors shared by Palace figure backgrounds, labels, grids and hover cards."""

    paper_bgcolor: str
    plot_bgcolor: str
    font: str
    muted: str
    grid: str
    hover_bg: str
    hover_border: str
    hover_font: str


_THEMES = {
    "light": PlotTheme(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f6f8fa",
        font="#1f2328",
        muted="#59636e",
        grid="rgba(31, 35, 40, 0.12)",
        hover_bg="#ffffff",
        hover_border="#d0d7de",
        hover_font="#1f2328",
    ),
    "dark": PlotTheme(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font="#e6edf3",
        muted="#8b949e",
        grid="rgba(139, 148, 158, 0.22)",
        hover_bg="#161b22",
        hover_border="#30363d",
        hover_font="#e6edf3",
    ),
}


def yaxis_type(name: str) -> str:
    """Choose the logarithmic or linear scale for a named physical quantity."""
    lowered = name.lower()
    if any(
        marker in lowered
        for marker in ("error", "q", "norm", "minimum", "maximum", "mean")
    ):
        return "log"
    return "linear"


@dataclass(frozen=True)
class LinePlot:
    """Trace data and axis/theme choices that travel together into a convergence figure."""

    title: str
    xlabel: str
    ylabel: str
    xs: Sequence[int | float | None]
    traces: Sequence[tuple[str, Sequence[float | None]]]
    point_labels: Sequence[str] | None = None
    yaxis_type: str = "linear"
    hline: tuple[float, str] | None = None
    theme: ReportTheme = "light"

    def figure(self) -> Any | None:
        """Render available paired points, labels and an optional configured reference line."""
        tokens = theme_tokens(self.theme)
        go = plotly()
        fig = go.Figure()
        # A responsive width does not reclaim the fixed gutter for an outside legend.
        FigureStyle(
            title=self.title,
            margin={"l": 84, "r": 36, "t": 116, "b": 64},
            theme=self.theme,
        ).apply(fig)
        fig.update_layout(
            legend={
                "orientation": "h",
                "x": 0,
                "xanchor": "left",
                "y": 1.04,
                "yanchor": "bottom",
            },
        )
        plotted = False
        for name, ys in self.traces:
            plot_ys = _positive_or_none(ys) if self.yaxis_type == "log" else list(ys)
            pairs: list[tuple[int | float, float]] = []
            labels: list[str] = []
            for index, (x_value, y_value) in enumerate(
                zip(self.xs, plot_ys, strict=False)
            ):
                if x_value is None or y_value is None:
                    continue
                pairs.append((x_value, y_value))
                if self.point_labels is not None and index < len(self.point_labels):
                    labels.append(self.point_labels[index])
            if not pairs:
                continue
            plot_x = [item[0] for item in pairs]
            plot_y = [item[1] for item in pairs]
            trace: dict[str, Any] = {
                "x": plot_x,
                "y": plot_y,
                "mode": "lines+markers+text" if labels else "lines+markers",
                "name": name,
                "line": {"width": 3},
                "marker": {"size": 10, "line": {"width": 0}},
                "hovertemplate": "%{y:.6g}<extra>%{fullData.name}</extra>",
            }
            if labels:
                trace["text"] = labels
                trace["textposition"] = "top center"
                trace["textfont"] = {"size": PLOT_TICK_FONT, "color": tokens.muted}
            fig.add_trace(go.Scatter(**trace))
            plotted = True
        if not plotted:
            return None
        if self.hline is not None:
            fig.add_hline(
                y=self.hline[0],
                line_dash="dot",
                line_color=tokens.muted,
                line_width=1.5,
                annotation_text=self.hline[1],
                annotation_font={"size": PLOT_TICK_FONT, "color": tokens.muted},
                annotation_position="top right",
            )
        fig.update_xaxes(title=axis_title(self.xlabel, tokens))
        fig.update_yaxes(title=axis_title(self.ylabel, tokens), type=self.yaxis_type)
        _maybe_integer_xticks(fig, self.xs)
        return fig


@dataclass(frozen=True)
class FigureStyle:
    """Reusable Palace colors, layout, legend and hover choices for one figure."""

    title: str
    height: int = _PLOT_HEIGHT
    margin: dict[str, int] | None = None
    hovermode: str = "x unified"
    showlegend: bool = True
    theme: ReportTheme = "light"

    def apply(self, fig: Any) -> None:
        """Apply the selected Palace layout and theme to an existing Plotly figure."""
        margin = self.margin
        tokens = theme_tokens(self.theme)
        if margin is None:
            margin = (
                {"l": 84, "r": 168, "t": 72, "b": 64}
                if self.showlegend
                else {"l": 72, "r": 36, "t": 72, "b": 64}
            )
        fig.update_layout(
            template="none",
            paper_bgcolor=tokens.paper_bgcolor,
            plot_bgcolor=tokens.plot_bgcolor,
            font={
                "family": "ui-sans-serif, system-ui, sans-serif",
                "size": _PLOT_FONT,
                "color": tokens.font,
            },
            title={
                "text": self.title,
                "font": {"size": _PLOT_TITLE_FONT, "color": tokens.font},
                "x": 0,
                "xanchor": "left",
                "y": 0.98,
                "yanchor": "top",
                "pad": {"t": 12, "b": 12, "l": 16, "r": 0},
            },
            colorway=list(COLORWAY),
            hovermode=self.hovermode,
            hoverlabel={
                "bgcolor": tokens.hover_bg,
                "bordercolor": tokens.hover_border,
                "font": {"color": tokens.hover_font, "size": PLOT_TICK_FONT},
            },
            legend={
                "orientation": "v",
                "xref": "paper",
                "yref": "paper",
                "yanchor": "top",
                "y": 1,
                "xanchor": "left",
                "x": 1.02,
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "font": {"size": _PLOT_LEGEND_FONT, "color": tokens.font},
                "tracegroupgap": 6,
                "itemsizing": "constant",
                "itemwidth": 36,
            },
            height=self.height,
            margin=margin,
            showlegend=self.showlegend,
        )
        axis = {
            "showgrid": True,
            "gridcolor": tokens.grid,
            "zeroline": False,
            "showline": False,
            "ticks": "",
            "automargin": False,
            "tickfont": {"size": PLOT_TICK_FONT, "color": tokens.muted},
        }
        fig.update_xaxes(**axis)
        fig.update_yaxes(**axis)


def axis_title(text: str, tokens: PlotTheme) -> dict[str, Any]:
    """Style an axis label using the same resolved theme as its figure."""
    return {
        "text": text,
        "font": {"size": _PLOT_AXIS_FONT, "color": tokens.font},
        "standoff": 8,
    }


def _maybe_integer_xticks(fig: Any, xs: Sequence[int | float | None]) -> None:
    """Use integer ticks when the recorded horizontal values are all integral."""
    numeric = [float(value) for value in xs if isinstance(value, (int, float))]
    if not numeric or not all(value.is_integer() for value in numeric):
        return
    span = max(numeric) - min(numeric)
    if span <= 24:
        fig.update_xaxes(dtick=1)


def show_figure(fig: Any) -> None:
    """Display a detached Plotly figure with the shared Palace interaction configuration."""
    fig.show(config=_PLOTLY_CONFIG)


def plotly() -> Any:
    """Load optional Plotly rendering only when a caller requests a figure."""
    try:
        import plotly.graph_objects as go  # noqa: PLC0415 -- Defer heavy Plotly loading until a figure is requested.
    except ImportError as exc:
        raise RuntimeError(
            "plotly is not available; install plotly to render Palace trustworthiness figures."
        ) from exc
    return go


def html_cell(tag: str, text: str) -> str:
    """Escape a table cell value while retaining the requested HTML cell tag."""
    return (
        f"<{tag} style='border:1px solid var(--border,#d0d7de);"
        f"padding:0.35rem 0.65rem;text-align:left'>{html.escape(text)}</{tag}>"
    )


def checked_theme(theme: str) -> ReportTheme:
    """Accept the supported light/dark display choices without changing report data."""
    if theme == "light" or theme == "dark":
        return theme
    raise ValueError("theme must be 'light' or 'dark'.")


def checked_ranking_limit(value: int | None) -> int | None:
    """Require an optional positive integer limit for individual-surface ranking."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("ranking_limit must be a positive integer or None.")
    return value


def checked_show_details(value: bool) -> bool:
    """Require an explicit boolean choice for expanded report metadata."""
    if not isinstance(value, bool):
        raise TypeError("show_details must be a bool.")
    return value


def theme_tokens(theme: str) -> PlotTheme:
    """Resolve the supported theme into the shared figure color palette."""
    return _THEMES[checked_theme(theme)]


def fmt(value: Any) -> str:
    """Format readable numerical card values while preserving unavailable observations."""
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def fmt_seconds(value: Any) -> str:
    """Format elapsed seconds as seconds, or minutes for durations of at least 120 seconds."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "unavailable"
    seconds = float(value)
    if seconds >= 120:
        return f"{seconds / 60:.2f} min"
    return f"{seconds:.3g} s"


def _positive_or_none(values: Sequence[float | None]) -> list[float | None]:
    """Retain positive values for logarithmic plots and mask other observations."""
    return [None if value is None or value <= 0 else value for value in values]


def fmt_mb(value: Any) -> str:
    """Format recorded memory with the report's MiB/GiB labels and 1024 scaling."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "unavailable"
    mb = float(value)
    if mb >= 1024:
        return f"{mb / 1024:.2f} GiB"
    return f"{mb:.3g} MiB"


def mesh_summary_html(
    summary: Mapping[str, Any], *, card_html: Callable[[str, Any], str]
) -> str:
    """Keep the meshing record (recipe, mesher, quality) distinct from native solver measurements."""
    if summary["status"] == "not_recorded":
        return "<section><h4>Geometry mesh</h4><p>Mesh observations were not recorded in this manifest.</p></section>"
    meshing = summary["meshing"]
    stats, recipe = meshing["statistics"], meshing["recipe"]
    items = "".join(card_html(label, value) for label, value in _mesh_cards(summary))
    rows = _region_rows(stats["physical_regions"])
    header = "".join(html_cell("th", label) for label in _REGION_COLUMNS)
    types = ", ".join(
        f"{record['name']}: {record['count']}" for record in stats["element_types"]
    )
    parameters = ", ".join(
        f"{key}={value}" for key, value in recipe["parameters"].items()
    )
    return (
        "<section><h4>Geometry mesh</h4>"
        f'<div style="display:flex;flex-wrap:wrap">{items}</div>'
        "<p>The ND estimate uses the configured FEM order and tetrahedral primary-corner topology before boundary constraints; it is not actual solver DOF. Electrostatic H1 estimation is unavailable. Quality is measured on the linear mesh, before any elevation to the geometry order.</p>"
        "<details><summary>Physical regions and mesh discretization</summary>"
        f"<p>Gmsh {html.escape(str(meshing['gmsh_version']))}. {html.escape(types)}.</p>"
        f"<p>Recipe {html.escape(recipe['name'])} v{html.escape(str(recipe['version']))}: {html.escape(parameters or 'no parameters')}.</p>"
        "<p>Edge statistics count unique PRIMARY CORNER CHORDS within each region, not curved arc lengths. Point regions have no edges.</p>"
        f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        "<p>Full controls, repair policy, the quality report hash and the provenance remain available in mesh_summary() and report data.</p>"
        "</details></section>"
    )


def _mesh_cards(summary: Mapping[str, Any]) -> list[tuple[str, str]]:
    """The (label, value) cards of a recorded mesh: recipe, orders, sizes, algorithms, quality and repair."""
    meshing = summary["meshing"]
    estimate = summary["dof_estimate"]
    stats, mesher = meshing["statistics"], meshing["mesher"]
    linear = meshing["linear_mesh"]
    recipe, budgets = meshing["recipe"], meshing["budgets"]
    return [
        ("Sizing recipe", f"{recipe['name']} v{recipe['version']}"),
        (
            "Geometry / FEM order",
            f"{mesher['geometry_order']} / {fmt(summary['finite_element_order'])}",
        ),
        (
            "Native nodes / elements",
            f"{stats['node_count']} / {stats['element_count']}",
        ),
        (
            "3D / 2D algorithm",
            f"{mesher['algorithm_3d']} / {fmt(mesher['algorithm_2d'] or 'Gmsh default')}",
        ),
        (
            "HighOrder optimization",
            "applied"
            if mesher["high_order_optimize"] and mesher["geometry_order"] > 1
            else "not applied",
        ),
        (
            "Quality threshold (min SICN)",
            f"{fmt(meshing['threshold'])} ({meshing['quality_measure']})",
        ),
        (
            "Min SICN / elements below threshold",
            f"{fmt(linear['min_quality'])} / {linear['bad_count']}",
        ),
        (
            "Repair loop",
            f"{meshing['stop']} after {meshing['rounds']} of {budgets['max_rounds']} rounds",
        ),
        (
            "Tetrahedra / budget",
            f"{linear['tetrahedra']} / {budgets['max_tetrahedra']}",
        ),
        ("ND topological DOF estimate", fmt(estimate.get("value"))),
    ]


_REGION_COLUMNS = (
    "Region",
    "Dim",
    "Elements",
    "Gmsh type: count",
    "Edges",
    "Min",
    "Mean",
    "Max",
    "Unit",
)


def _region_rows(regions: Sequence[Mapping[str, Any]]) -> list[str]:
    """One table row per physical region: its element counts and its corner-chord edge statistics."""
    rows = []
    for region in regions:
        chords = region.get("corner_chord_edge_lengths")
        values = [
            region["name"],
            region["dimension"],
            region["element_count"],
            ", ".join(
                f"{code}: {count}" for code, count in region["element_types"].items()
            ),
        ]
        values.extend(
            [
                chords["count"],
                *(fmt(chords[key]) for key in ("minimum", "mean", "maximum")),
                chords["unit"],
            ]
            if chords is not None
            else ["not recorded"] * 5
        )
        rows.append(
            "<tr>" + "".join(html_cell("td", str(value)) for value in values) + "</tr>"
        )
    return rows
