"""Render surface participation, native margins and loss views from detached records."""

from __future__ import annotations

import html
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from scgsim.palace.presentation.plotting import (
    COLORWAY,
    FigureStyle,
    LinePlot,
    ReportTheme,
    axis_title,
    fmt,
    fmt_seconds,
    plotly,
    theme_tokens,
)
from scgsim.palace.results.report_models import (
    SurfaceEprSeriesSnapshot,
    SurfaceMaskEprRecord,
    SurfaceMaskEprSeriesSnapshot,
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
        xs=[snapshot.pass_index for snapshot in snapshots],
        traces=[
            (
                interface_type,
                [_surface_totals(snapshot)[interface_type] for snapshot in snapshots],
            )
            for interface_type in SURFACE_TYPES
        ],
        theme=theme,
    ).figure()


def latest_mask_snapshot(
    snapshots: Sequence[SurfaceMaskEprSeriesSnapshot],
    *,
    pass_index: int,
    source: str,
    series_index: int,
) -> SurfaceMaskEprSeriesSnapshot | None:
    """Select the native-mask series bound to one exact pass, source and mode."""
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


def surface_mask_participation_figure(
    snapshot: SurfaceMaskEprSeriesSnapshot,
    baseline: SurfaceEprSeriesSnapshot,
    interface_type: str,
    theme: ReportTheme,
) -> Any | None:
    """Compare recorded Inset participation against the matching unmasked interface baseline."""
    records = [
        record for record in snapshot.records if record.interface_type == interface_type
    ]
    if not records:
        return None
    baseline_by_index = {record.index: record for record in baseline.records}
    go = plotly()
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
            line={"color": COLORWAY[color_index % len(COLORWAY)]},
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
                    "color": COLORWAY[color_index % len(COLORWAY)],
                },
                hovertemplate="baseline participation=%{y:.6g}<extra>%{fullData.name}</extra>",
            )
    FigureStyle(
        title=f"{series_label(snapshot)}: {interface_type} native Inset-mask participation "
        f"vs margin ({snapshot.source})",
        height=460,
        margin={"l": 78, "r": 220, "t": 72, "b": 64},
        hovermode="closest",
        showlegend=True,
        theme=theme,
    ).apply(figure)
    figure.update_xaxes(title="Inset margin (µm)", rangemode="tozero")
    figure.update_yaxes(title="participation", rangemode="tozero")
    figure.update_layout(legend={"x": 1.02, "xanchor": "left", "y": 1})
    return figure


def surface_mask_html(snapshot: SurfaceMaskEprSeriesSnapshot) -> str:
    """Render exact native mask margins and their source-bound contributions."""
    zeroes = sum(
        record.contribution_status == "zero_contribution" for record in snapshot.records
    )
    return (
        "<p style='opacity:0.75'>Native mask Q and energy remain available in "
        "<code>masked_snapshots</code>. The native Inset output does not retain "
        "surface area support; zero participation or energy is reported as "
        "<code>zero_contribution</code>, not as proof of excluded geometry. "
        f"This snapshot contains {zeroes} zero-contribution mask record(s).</p>"
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
    """Rank individual source surfaces while retaining readable provenance labels."""
    records = sorted(
        snapshot.records, key=lambda record: (-record.participation, record.index)
    )
    visible = records if ranking_limit is None else records[:ranking_limit]
    if not visible:
        return None
    tokens = theme_tokens(theme)
    go = plotly()
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
                        COLORWAY[SURFACE_TYPES.index(record.interface_type)]
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
    if problem == "Eigenmode":
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
    snapshot: SurfaceEprSeriesSnapshot | SurfaceMaskEprSeriesSnapshot,
) -> str:
    """Label a solver mode or electrostatic excitation without merging their series."""
    noun = "Mode" if snapshot.series_kind == "mode" else "Excitation"
    return f"{noun} {snapshot.series_index}"


def _source_provenance_label(
    provenance: dict[str, Any],
    *,
    interface_type: str | None = None,
) -> str:
    """Summarize recorded conductor and source-surface identities for hover inspection."""
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
    """Expose recorded source contribution records without inferring new ownership."""
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
