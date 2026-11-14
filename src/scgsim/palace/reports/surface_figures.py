"""Render surface participation, margin rows and loss views from detached records."""

from __future__ import annotations

import html
import math
from collections.abc import Sequence
from typing import Any

from scgsim.contracts.palace_run_folder import palace_problem
from scgsim.palace.reports.plotting import (
    COLORWAY,
    FigureStyle,
    LinePlot,
    ReportTheme,
    Trace,
    axis_title,
    fmt,
    fmt_seconds,
    plotly,
    theme_tokens,
)
from scgsim.palace.results.inspection_records import (
    SurfaceEprRecord,
    SurfaceEprSeriesSnapshot,
    SurfaceMarginEprRecord,
    SurfaceMarginEprSeriesSnapshot,
    pass_number,
)

SURFACE_TYPES = ("MA", "MS", "SA")


def _surface_totals(snapshot: SurfaceEprSeriesSnapshot) -> dict[str, float]:
    """Sum available participation by native interface type for one mode or excitation."""
    return {
        interface_type: sum(
            record.participation
            for record in snapshot.records
            if record.interface_type == interface_type
        )
        for interface_type in SURFACE_TYPES
    }


def surface_total_figure(
    snapshots: Sequence[SurfaceEprSeriesSnapshot], theme: ReportTheme
) -> Any | None:
    """Plot total MA/MS/SA participation across the recorded surface series."""
    return LinePlot(
        title=f"{series_label(snapshots[-1])}: MA/MS/SA participation vs AMR pass",
        xlabel="AMR pass",
        ylabel="total participation",
        xs=[pass_number(snapshot.pass_index) for snapshot in snapshots],
        traces=[
            Trace(
                interface_type,
                [_surface_totals(snapshot)[interface_type] for snapshot in snapshots],
            )
            for interface_type in SURFACE_TYPES
        ],
        theme=theme,
    ).figure()


def latest_margin_snapshot(
    snapshots: Sequence[SurfaceMarginEprSeriesSnapshot],
    *,
    pass_index: int,
    source: str,
    series_index: int,
) -> SurfaceMarginEprSeriesSnapshot | None:
    """Select the margin series bound to one exact pass, source and mode."""
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


def surface_margin_participation_figure(
    snapshot: SurfaceMarginEprSeriesSnapshot,
    baseline: SurfaceEprSeriesSnapshot,
    interface_type: str,
    theme: ReportTheme,
) -> Any | None:
    """Compare margin-row participation against the whole-surface row of the same interface."""
    records = [
        record for record in snapshot.records if record.interface_type == interface_type
    ]
    if not records:
        return None
    baseline_by_index = {record.index: record for record in baseline.records}
    go = plotly()
    figure = go.Figure()
    grouped: dict[int, list[SurfaceMarginEprRecord]] = {}
    for record in records:
        grouped.setdefault(record.baseline_index, []).append(record)
    for color_index, (baseline_index, series) in enumerate(sorted(grouped.items())):
        ordered = sorted(series, key=lambda item: item.margin_index)
        first = ordered[0]
        label = f"#{baseline_index} {'/'.join(first.face_kinds)} · {first.entities[0]}"
        figure.add_scatter(
            x=[record.margin_um for record in ordered],
            y=[record.participation for record in ordered],
            mode="lines+markers",
            name=label,
            line={"color": COLORWAY[color_index % len(COLORWAY)]},
            customdata=[
                [
                    record.index,
                    record.margin_index,
                    ", ".join(str(band) for band in record.bands),
                    (
                        "unavailable / +inf"
                        if math.isinf(record.quality_factor)
                        else f"{record.quality_factor:.6g}"
                    ),
                    record.name,
                    ", ".join(record.entities),
                ]
                for record in ordered
            ],
            hovertemplate=(
                "margin=%{x:.6g} µm<br>participation=%{y:.6g}"
                "<br>margin index=%{customdata[0]}<br>ordinal=%{customdata[1]}"
                "<br>bands=%{customdata[2]}<br>native Q=%{customdata[3]}"
                "<br>surface=%{customdata[4]}<br>entities=%{customdata[5]}"
                "<extra>%{fullData.name}</extra>"
            ),
        )
        baseline_record = baseline_by_index.get(baseline_index)
        if baseline_record is not None:
            figure.add_scatter(
                x=[0.0],
                y=[baseline_record.participation],
                mode="markers",
                name=f"whole surface {label}",
                marker={
                    "symbol": "diamond",
                    "size": 9,
                    "color": COLORWAY[color_index % len(COLORWAY)],
                },
                hovertemplate="whole-surface participation=%{y:.6g}<extra>%{fullData.name}</extra>",
            )
    FigureStyle(
        title=f"{series_label(snapshot)}: {interface_type} margin-row participation "
        f"vs margin ({snapshot.source})",
        height=460,
        margin={"l": 78, "r": 220, "t": 72, "b": 64},
        hovermode="closest",
        showlegend=True,
        theme=theme,
    ).apply(figure)
    figure.update_xaxes(title="Margin from the gap edge (µm)", rangemode="tozero")
    figure.update_yaxes(title="participation", rangemode="tozero")
    figure.update_layout(legend={"x": 1.02, "xanchor": "left", "y": 1})
    return figure


def surface_margin_html(snapshot: SurfaceMarginEprSeriesSnapshot) -> str:
    """Explain what a margin row integrates and where its records stay available."""
    del snapshot  # the explanation holds for every margin snapshot
    return (
        "<p style='opacity:0.75'>A margin row is the union of the SGB bands from its margin "
        "inwards, measured from the gap edge, listed as plain attributes of one Palace "
        "dielectric row. It is an alternative to its whole-surface row, never an addition. "
        "Its records remain available in <code>margin_snapshots</code>.</p>"
    )


def surface_percentage_figure(
    snapshots: Sequence[SurfaceEprSeriesSnapshot], theme: ReportTheme
) -> Any | str:
    """Plot each interface contribution as a share of available surface participation."""
    totals = [_surface_totals(snapshot) for snapshot in snapshots]
    denominators = [sum(values.values()) for values in totals]
    figure = LinePlot(
        title=f"{series_label(snapshots[-1])}: normalized MA/MS/SA participation",
        xlabel="AMR pass",
        ylabel="share of MA+MS+SA (%)",
        xs=[pass_number(snapshot.pass_index) for snapshot in snapshots],
        traces=[
            Trace(
                interface_type,
                [
                    None
                    if denominator <= 0
                    else 100.0 * values[interface_type] / denominator
                    for values, denominator in zip(totals, denominators, strict=True)
                ],
            )
            for interface_type in SURFACE_TYPES
        ],
        theme=theme,
    ).figure()
    if figure is None:
        return (
            "<p style='opacity:0.75'>Normalized MA/MS/SA percentages are unavailable "
            "because their same-snapshot denominator is zero.</p>"
        )
    figure.update_yaxes(range=[0, 100])
    return figure


def surface_ranking_figure(
    snapshot: SurfaceEprSeriesSnapshot,
    ranking_limit: int | None,
    theme: ReportTheme,
) -> Any | None:
    """Rank individual surface rows while retaining readable Entity and row labels."""
    records = sorted(
        snapshot.records, key=lambda record: (-record.participation, record.index)
    )
    visible = records if ranking_limit is None else records[:ranking_limit]
    if not visible:
        return None
    tokens = theme_tokens(theme)
    go = plotly()
    labels = [
        f"#{record.index} {record.interface_type} · {'/'.join(record.face_kinds)}"
        f"<br>{record.entities[0]}"
        for record in visible
    ]
    custom = [
        [
            record.name,
            ", ".join(record.entities),
            ", ".join(record.face_kinds),
            ", ".join(record.nets) or "unassigned",
            ", ".join(record.rows),
            _lineage_label(record),
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
                        COLORWAY[SURFACE_TYPES.index(record.interface_type)]
                        for record in visible
                    ]
                },
                customdata=custom,
                hovertemplate=(
                    "%{x:.6g}<br>surface=%{customdata[0]}"
                    "<br>entities=%{customdata[1]}<br>face=%{customdata[2]}"
                    "<br>net=%{customdata[3]}<br>rows=%{customdata[4]}"
                    "<br>lineage=%{customdata[5]}<extra>%{y}</extra>"
                ),
            )
        ]
    )
    count = len(snapshot.records)
    shown = len(visible)
    FigureStyle(
        title=f"{series_label(snapshot)}: latest surface participation ranking "
        f"({snapshot.source}; {shown} of {count})",
        height=max(420, 48 * shown + 150),
        margin={"l": 240, "r": 52, "t": 72, "b": 64},
        hovermode="closest",
        showlegend=False,
        theme=theme,
    ).apply(fig)
    fig.update_layout(bargap=0.38)
    fig.update_xaxes(title=axis_title("participation", tokens), rangemode="tozero")
    fig.update_yaxes(autorange="reversed", automargin=True)
    return fig


def surface_loss_html(snapshot: SurfaceEprSeriesSnapshot, problem: str) -> str:
    """Explain available participation-weighted loss and the recorded loss status."""
    if snapshot.loss_status == "available":
        q_value = fmt(snapshot.quality_factor_total)
        detail = "Q_total uses 1 / Σ(p_i tanδ_i); individual Q values are never summed."
    elif snapshot.loss_status == "unavailable_missing":
        q_value = "unavailable"
        detail = "At least one structured surface loss tangent is unavailable."
    else:
        q_value = "unavailable / non-finite"
        detail = (
            "The configured surface-loss sum is zero, so native +inf Q is not "
            "converted to zero. Participation remains available."
        )
    cards = [
        ("Latest source", snapshot.source),
        ("Surface-loss Q_total", q_value),
    ]
    if palace_problem(problem).solves_modes:
        cards.append(
            (
                "Surface-loss T1",
                fmt_seconds(snapshot.t1_seconds)
                if snapshot.t1_seconds is not None
                else "unavailable / non-finite",
            )
        )
    items = "".join(
        "<div style='min-width:11rem;flex:1 1 11rem;border:1px solid "
        "var(--border,#d0d7de);border-radius:8px;padding:0.6rem 0.75rem;margin:0.25rem'>"
        f"<div style='font-size:0.75rem;opacity:0.7'>{html.escape(label)}</div>"
        f"<div style='font-size:0.95rem;font-weight:600'>{html.escape(value)}</div></div>"
        for label, value in cards
    )
    return (
        f"<div style='display:flex;flex-wrap:wrap'>{items}</div>"
        f"<p style='opacity:0.75'>{html.escape(detail)}</p>"
    )


def series_label(
    snapshot: SurfaceEprSeriesSnapshot | SurfaceMarginEprSeriesSnapshot,
) -> str:
    """Label a solver mode or electrostatic excitation without merging their series."""
    noun = "Mode" if snapshot.series_kind == "mode" else "Excitation"
    return f"{noun} {snapshot.series_index}"


def _lineage_label(record: SurfaceEprRecord) -> str:
    """Which Entities share the faces of a row, as recorded attribution (it weights nothing)."""
    sharing = sorted(
        {entity for owners in record.lineage.values() for entity in owners}
    )
    return f"{len(record.lineage)} face(s) shared among {', '.join(sharing) or 'no Entity'}"
