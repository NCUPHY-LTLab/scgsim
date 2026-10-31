"""Plotly figures and the plot theme for Palace run reports.

Everything here turns already-verified numbers into figures: the theme tokens, the
line figures for convergence traces, and the surface-participation figures. Nothing
reads a run directory or decides whether a run is trustworthy; that is
``reports.py``, which imports these builders.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ..results.report_data import (
    AmrPassSnapshot,
    SurfaceEprSeriesSnapshot,
    SurfaceMaskEprRecord,
    SurfaceMaskEprSeriesSnapshot,
)

_SURFACE_TYPES = ("MA", "MS", "SA")


_COLORWAY = (
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


_PLOT_TICK_FONT = 14


_PLOT_LEGEND_FONT = 15


ReportTheme = Literal["light", "dark"]


@dataclass(frozen=True)
class _PlotTheme:
    paper_bgcolor: str
    plot_bgcolor: str
    font: str
    muted: str
    grid: str
    hover_bg: str
    hover_border: str
    hover_font: str


_THEMES = {
    "light": _PlotTheme(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f6f8fa",
        font="#1f2328",
        muted="#59636e",
        grid="rgba(31, 35, 40, 0.12)",
        hover_bg="#ffffff",
        hover_border="#d0d7de",
        hover_font="#1f2328",
    ),
    "dark": _PlotTheme(
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


def _surface_totals(snapshot: SurfaceEprSeriesSnapshot) -> dict[str, float]:
    return {
        interface_type: sum(
            record.participation
            for record in snapshot.records
            if record.interface_type == interface_type
        )
        for interface_type in _SURFACE_TYPES
    }


def _surface_total_figure(
    snapshots: Sequence[SurfaceEprSeriesSnapshot], theme: ReportTheme
) -> Any | None:
    return _line_figure(
        title=f"{_series_label(snapshots[-1])}: MA/MS/SA participation vs AMR pass",
        xlabel="AMR pass",
        ylabel="total participation",
        xs=[snapshot.pass_index for snapshot in snapshots],
        traces=[
            (
                interface_type,
                [_surface_totals(snapshot)[interface_type] for snapshot in snapshots],
            )
            for interface_type in _SURFACE_TYPES
        ],
        theme=theme,
    )


def _surface_mask_participation_figure(
    snapshot: SurfaceMaskEprSeriesSnapshot,
    baseline: SurfaceEprSeriesSnapshot,
    interface_type: str,
    theme: ReportTheme,
) -> Any | None:
    records = [
        record for record in snapshot.records if record.interface_type == interface_type
    ]
    if not records:
        return None
    baseline_by_index = {record.index: record for record in baseline.records}
    go = _plotly()
    figure = go.Figure()
    grouped: dict[int, list[SurfaceMaskEprRecord]] = {}
    for record in records:
        grouped.setdefault(record.baseline_index, []).append(record)
    for color_index, (baseline_index, series) in enumerate(sorted(grouped.items())):
        ordered = sorted(series, key=lambda item: item.margin_index)
        first = ordered[0]
        label = f"#{baseline_index} {first.face_kind} · {first.owner_semantic_ids[0]}"
        figure.add_scatter(
            x=[record.margin_um for record in ordered],
            y=[record.participation for record in ordered],
            mode="lines+markers",
            name=label,
            line={"color": _COLORWAY[color_index % len(_COLORWAY)]},
            customdata=[
                [
                    record.index,
                    record.margin_index,
                    record.energy_j,
                    (
                        "unavailable / +inf"
                        if math.isinf(record.quality_factor)
                        else f"{record.quality_factor:.6g}"
                    ),
                    record.contribution_status,
                    record.retained_area_status,
                    record.surface_id,
                    ", ".join(record.owner_semantic_ids),
                ]
                for record in ordered
            ],
            hovertemplate=(
                "margin=%{x:.6g} µm<br>participation=%{y:.6g}"
                "<br>masked index=%{customdata[0]}<br>ordinal=%{customdata[1]}"
                "<br>energy=%{customdata[2]:.6g} J<br>native Q=%{customdata[3]}"
                "<br>contribution=%{customdata[4]}<br>retained area=%{customdata[5]}"
                "<br>surface=%{customdata[6]}<br>owners=%{customdata[7]}"
                "<extra>%{fullData.name}</extra>"
            ),
        )
        baseline_record = baseline_by_index.get(baseline_index)
        if baseline_record is not None:
            figure.add_scatter(
                x=[0.0],
                y=[baseline_record.participation],
                mode="markers",
                name=f"baseline {label}",
                marker={
                    "symbol": "diamond",
                    "size": 9,
                    "color": _COLORWAY[color_index % len(_COLORWAY)],
                },
                hovertemplate="baseline participation=%{y:.6g}<extra>%{fullData.name}</extra>",
            )
    _style_figure(
        figure,
        title=(
            f"{_series_label(snapshot)}: {interface_type} native Inset-mask participation "
            f"vs margin ({snapshot.source})"
        ),
        height=460,
        margin={"l": 78, "r": 220, "t": 72, "b": 64},
        hovermode="closest",
        showlegend=True,
        theme=theme,
    )
    figure.update_xaxes(title="Inset margin (µm)", rangemode="tozero")
    figure.update_yaxes(title="participation", rangemode="tozero")
    figure.update_layout(legend={"x": 1.02, "xanchor": "left", "y": 1})
    return figure


def _surface_percentage_figure(
    snapshots: Sequence[SurfaceEprSeriesSnapshot], theme: ReportTheme
) -> Any | str:
    totals = [_surface_totals(snapshot) for snapshot in snapshots]
    denominators = [sum(values.values()) for values in totals]
    figure = _line_figure(
        title=f"{_series_label(snapshots[-1])}: normalized MA/MS/SA participation",
        xlabel="AMR pass",
        ylabel="share of MA+MS+SA (%)",
        xs=[snapshot.pass_index for snapshot in snapshots],
        traces=[
            (
                interface_type,
                [
                    None
                    if denominator <= 0
                    else 100.0 * values[interface_type] / denominator
                    for values, denominator in zip(totals, denominators, strict=True)
                ],
            )
            for interface_type in _SURFACE_TYPES
        ],
        theme=theme,
    )
    if figure is None:
        return (
            "<p style='opacity:0.75'>Normalized MA/MS/SA percentages are unavailable "
            "because their same-snapshot denominator is zero.</p>"
        )
    figure.update_yaxes(range=[0, 100])
    return figure


def _surface_ranking_figure(
    snapshot: SurfaceEprSeriesSnapshot,
    ranking_limit: int | None,
    theme: ReportTheme,
) -> Any | None:
    records = sorted(
        snapshot.records, key=lambda record: (-record.participation, record.index)
    )
    visible = records if ranking_limit is None else records[:ranking_limit]
    if not visible:
        return None
    tokens = _theme_tokens(theme)
    go = _plotly()
    labels = [
        f"#{record.index} {record.interface_type} · {record.face_kind}"
        f"<br>{record.owner_semantic_ids[0]}"
        for record in visible
    ]
    custom = [
        [
            record.surface_id,
            ", ".join(record.owner_semantic_ids),
            record.face_kind,
            record.net_id or "unassigned",
            record.equipotential_id or "unassigned",
            _source_provenance_label(
                record.source_provenance,
                interface_type=record.interface_type,
            ),
        ]
        for record in visible
    ]
    fig = go.Figure(
        data=[
            go.Bar(
                x=[record.participation for record in visible],
                y=labels,
                orientation="h",
                marker={
                    "color": [
                        _COLORWAY[_SURFACE_TYPES.index(record.interface_type)]
                        for record in visible
                    ]
                },
                customdata=custom,
                hovertemplate=(
                    "%{x:.6g}<br>surface=%{customdata[0]}"
                    "<br>owners=%{customdata[1]}<br>face=%{customdata[2]}"
                    "<br>net=%{customdata[3]}<br>equipotential=%{customdata[4]}"
                    "<br>source=%{customdata[5]}<extra>%{y}</extra>"
                ),
            )
        ]
    )
    count = len(snapshot.records)
    shown = len(visible)
    _style_figure(
        fig,
        title=(
            f"{_series_label(snapshot)}: latest surface participation ranking "
            f"({snapshot.source}; {shown} of {count})"
        ),
        height=max(420, 48 * shown + 150),
        margin={"l": 240, "r": 52, "t": 72, "b": 64},
        hovermode="closest",
        showlegend=False,
        theme=theme,
    )
    fig.update_layout(bargap=0.38)
    fig.update_xaxes(title=_axis_title("participation", tokens), rangemode="tozero")
    fig.update_yaxes(autorange="reversed", automargin=True)
    return fig


def _series_label(
    snapshot: SurfaceEprSeriesSnapshot | SurfaceMaskEprSeriesSnapshot,
) -> str:
    noun = "Mode" if snapshot.series_kind == "mode" else "Excitation"
    return f"{noun} {snapshot.series_index}"


def _source_provenance_label(
    provenance: dict[str, Any],
    *,
    interface_type: str | None = None,
) -> str:
    record_ids = provenance.get("source_record_ids")
    label = "structured provenance retained"
    if (
        isinstance(record_ids, list)
        and record_ids
        and all(isinstance(record_id, str) for record_id in record_ids)
    ):
        suffix = f" (+{len(record_ids) - 1})" if len(record_ids) > 1 else ""
        label = f"{record_ids[0]}{suffix}"
    contributions = _surface_contribution_records(
        provenance,
        interface_type=interface_type,
    )
    if not contributions:
        return label
    sides = tuple(
        dict.fromkeys(
            str(record["side"])
            for record in contributions
            if isinstance(record.get("side"), str)
        )
    )
    suffix = f"; {len(contributions)} contribution(s)"
    if sides:
        suffix += f" ({'/'.join(sides)})"
    return f"{label}{suffix}"


def _surface_contribution_records(
    provenance: Mapping[str, Any],
    *,
    interface_type: str | None = None,
) -> tuple[Mapping[str, Any], ...]:
    records: list[Mapping[str, Any]] = []
    ledger = provenance.get("surface_contribution_ledger")
    if isinstance(ledger, (list, tuple)):
        records.extend(record for record in ledger if isinstance(record, Mapping))
    else:
        sources = provenance.get("sources", ())
        if isinstance(sources, (list, tuple)):
            for source in sources:
                if isinstance(source, Mapping):
                    records.extend(_surface_contribution_records(source))
    deduplicated: dict[str, Mapping[str, Any]] = {}
    identities: dict[str, str] = {}
    for record in records:
        contribution_id = record.get("contribution_id")
        if not isinstance(contribution_id, str) or not contribution_id:
            continue
        identity = _serialized_report_contribution(record)
        previous = identities.setdefault(contribution_id, identity)
        if previous != identity:
            raise ValueError(
                f"conflicting repeated surface contribution {contribution_id!r}"
            )
        deduplicated.setdefault(contribution_id, record)
    return tuple(
        record
        for record in deduplicated.values()
        if interface_type is None or record.get("classification") == interface_type
    )


def _serialized_report_contribution(record: Mapping[str, Any]) -> str:
    """Canonicalize one complete serialized EvidenceResult transport record."""
    return json.dumps(
        dict(record),
        sort_keys=True,
        separators=(",", ":"),
    )


def _mode_traces(
    passes: Sequence[AmrPassSnapshot],
    getter: Callable[[AmrPassSnapshot], Sequence[float] | None],
) -> list[tuple[str, list[float | None]]]:
    mode_count = max((len(getter(pass_) or ()) for pass_ in passes), default=0)
    traces: list[tuple[str, list[float | None]]] = []
    for mode in range(mode_count):
        values = []
        for pass_ in passes:
            column = getter(pass_)
            if column is None or mode >= len(column):
                values.append(None)
            else:
                values.append(column[mode])
        traces.append((f"Mode {mode + 1}", values))
    return traces


def _traces_are_finite(traces: Sequence[tuple[str, Sequence[float | None]]]) -> bool:
    numbers = [
        value
        for _name, ys in traces
        for value in ys
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return bool(numbers) and all(math.isfinite(value) for value in numbers)


def _delta_traces(
    traces: Sequence[tuple[str, Sequence[float | None]]],
) -> list[tuple[str, list[float | None]]]:
    deltas: list[tuple[str, list[float | None]]] = []
    for name, ys in traces:
        series: list[float | None] = []
        previous: float | None = None
        for value in ys:
            if previous is None or value is None:
                series.append(None)
            else:
                series.append(abs(value - previous))
            if value is not None:
                previous = value
        deltas.append((name, series[1:]))
    return deltas


def _capacitance_traces(
    passes: Sequence[AmrPassSnapshot],
) -> list[tuple[str, list[tuple[str, list[float | None]]]]]:
    rows = max(
        (
            len(pass_.capacitance_matrix_f)
            for pass_ in passes
            if pass_.capacitance_matrix_f
        ),
        default=0,
    )
    cols = max(
        (
            len(row)
            for pass_ in passes
            if pass_.capacitance_matrix_f
            for row in pass_.capacitance_matrix_f
        ),
        default=0,
    )
    if rows == 0 or cols == 0:
        return []
    all_traces: list[tuple[str, list[float | None]]] = []
    per_element: list[tuple[str, list[tuple[str, list[float | None]]]]] = []
    for row in range(rows):
        for col in range(cols):
            name = f"C[{row + 1}][{col + 1}]"
            values = []
            for pass_ in passes:
                matrix = pass_.capacitance_matrix_f
                if matrix is None or row >= len(matrix) or col >= len(matrix[row]):
                    values.append(None)
                else:
                    values.append(matrix[row][col] * 1e15)
            trace = (name, values)
            all_traces.append(trace)
            per_element.append((name, [trace]))
    if len(all_traces) > 9:
        return [("Maxwell C_ij", all_traces)]
    return per_element


def _line_figure(
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    xs: Sequence[int | float | None],
    traces: Sequence[tuple[str, Sequence[float | None]]],
    point_labels: Sequence[str] | None = None,
    yaxis_type: str = "linear",
    hline: tuple[float, str] | None = None,
    theme: ReportTheme = "light",
) -> Any | None:
    tokens = _theme_tokens(theme)
    go = _plotly()
    fig = go.Figure()
    # A responsive width does not reclaim the fixed gutter for an outside legend.
    _style_figure(
        fig,
        title=title,
        margin={"l": 84, "r": 36, "t": 116, "b": 64},
        theme=theme,
    )
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
    for name, ys in traces:
        plot_ys = _positive_or_none(ys) if yaxis_type == "log" else list(ys)
        pairs: list[tuple[int | float, float]] = []
        labels: list[str] = []
        for index, (x_value, y_value) in enumerate(zip(xs, plot_ys, strict=False)):
            if x_value is None or y_value is None:
                continue
            pairs.append((x_value, y_value))
            if point_labels is not None and index < len(point_labels):
                labels.append(point_labels[index])
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
            trace["textfont"] = {"size": _PLOT_TICK_FONT, "color": tokens.muted}
        fig.add_trace(go.Scatter(**trace))
        plotted = True
    if not plotted:
        return None
    if hline is not None:
        fig.add_hline(
            y=hline[0],
            line_dash="dot",
            line_color=tokens.muted,
            line_width=1.5,
            annotation_text=hline[1],
            annotation_font={"size": _PLOT_TICK_FONT, "color": tokens.muted},
            annotation_position="top right",
        )
    fig.update_xaxes(title=_axis_title(xlabel, tokens))
    fig.update_yaxes(title=_axis_title(ylabel, tokens), type=yaxis_type)
    _maybe_integer_xticks(fig, xs)
    return fig


def _style_figure(
    fig: Any,
    *,
    title: str,
    height: int = _PLOT_HEIGHT,
    margin: dict[str, int] | None = None,
    hovermode: str = "x unified",
    showlegend: bool = True,
    theme: ReportTheme = "light",
) -> None:
    tokens = _theme_tokens(theme)
    if margin is None:
        margin = (
            {"l": 84, "r": 168, "t": 72, "b": 64}
            if showlegend
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
            "text": title,
            "font": {"size": _PLOT_TITLE_FONT, "color": tokens.font},
            "x": 0,
            "xanchor": "left",
            "y": 0.98,
            "yanchor": "top",
            "pad": {"t": 12, "b": 12, "l": 16, "r": 0},
        },
        colorway=list(_COLORWAY),
        hovermode=hovermode,
        hoverlabel={
            "bgcolor": tokens.hover_bg,
            "bordercolor": tokens.hover_border,
            "font": {"color": tokens.hover_font, "size": _PLOT_TICK_FONT},
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
        height=height,
        margin=margin,
        showlegend=showlegend,
    )
    axis = {
        "showgrid": True,
        "gridcolor": tokens.grid,
        "zeroline": False,
        "showline": False,
        "ticks": "",
        "automargin": False,
        "tickfont": {"size": _PLOT_TICK_FONT, "color": tokens.muted},
    }
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)


def _axis_title(text: str, tokens: _PlotTheme) -> dict[str, Any]:
    return {
        "text": text,
        "font": {"size": _PLOT_AXIS_FONT, "color": tokens.font},
        "standoff": 8,
    }


def _maybe_integer_xticks(fig: Any, xs: Sequence[int | float | None]) -> None:
    numeric = [float(value) for value in xs if isinstance(value, (int, float))]
    if not numeric or not all(value.is_integer() for value in numeric):
        return
    span = max(numeric) - min(numeric)
    if span <= 24:
        fig.update_xaxes(dtick=1)


def _plotly() -> Any:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError(
            "plotly is not available; install plotly to render Palace trustworthiness figures."
        ) from exc
    return go


def _checked_theme(theme: str) -> ReportTheme:
    if theme == "light" or theme == "dark":
        return theme
    raise ValueError("theme must be 'light' or 'dark'.")


def _theme_tokens(theme: str) -> _PlotTheme:
    return _THEMES[_checked_theme(theme)]


def _positive_or_none(values: Sequence[float | None]) -> list[float | None]:
    return [None if value is None or value <= 0 else value for value in values]


def _yaxis_type(name: str) -> str:
    lowered = name.lower()
    if any(
        marker in lowered
        for marker in ("error", "q", "norm", "minimum", "maximum", "mean")
    ):
        return "log"
    return "linear"
