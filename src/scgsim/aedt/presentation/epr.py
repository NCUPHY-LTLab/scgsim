"""EPR plots and notebook views for a single AEDT eigenmode result.

Owns: the two public figures, ``plot_epr_result`` and ``show_epr``.
Does not own: the row projections they draw (``epr_history.py``, ``show_view`` below)
or label formatting (``epr_surfaces.py``).
"""

from __future__ import annotations

import html
import textwrap
from typing import Any

from scgsim.aedt.epr.models import EprResult, detached
from scgsim.aedt.presentation.epr_history import history_view
from scgsim.aedt.presentation.epr_surfaces import (
    native_reference_table_data,
    ranking_height,
    surface_hover,
    surface_key,
    surface_metadata,
    surface_tick_label,
    usable_frequency,
)
from scgsim.presentation.notebook import INTERFACE_COLORS, checked_theme, style_figure


def plot_epr_result(
    result: EprResult,
    *,
    mode: int | None = None,
    native_pass: int | None = None,
    show_convergence: bool = True,
    theme: str = "light",
) -> Any:
    """Render exact selected-pass values and independent adaptive history axes."""

    checked_theme(theme)
    _check_plot_arguments(result, mode, native_pass, show_convergence)
    from plotly import graph_objects as go
    from plotly.subplots import make_subplots

    view = history_view(
        result, mode=mode, native_pass=native_pass, show_convergence=show_convergence
    )
    convergence_view = view["convergence_view"]
    titles = (
        (
            "Selected-pass participation",
            "Frequency by pass",
            "Participation by pass",
            "Energy balance by pass",
        )
        if convergence_view
        else ("Selected-pass participation",)
    )
    figure = make_subplots(
        rows=len(titles),
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.09 if convergence_view else 0.12,
        subplot_titles=titles,
    )
    _add_selected_pass_bars(figure, go, view)
    if convergence_view:
        _add_convergence_lines(figure, go, view)
    figure.update_layout(
        barmode="group",
        title=f"AEDT EPR — {view['result_kind'].replace('_', ' ')}",
        meta={
            "schema_version": "scgsim.aedt.epr-plot.v1",
            "result_kind": view["result_kind"],
            "selected_mode": mode,
            "selected_native_pass": view["native_pass"],
            "show_convergence": show_convergence,
            "unavailable": view["unavailable"],
        },
    )
    _title_plot_axes(figure, view)
    if view["unavailable"]:
        figure.add_annotation(
            text="<br>".join(html.escape(message) for message in view["unavailable"]),
            xref="paper",
            yref="paper",
            x=0.0,
            y=1.0,
            xanchor="left",
            yanchor="bottom",
            showarrow=False,
        )
    style_figure(figure, theme=theme, height=1320 if convergence_view else 460)
    figure.update_layout(
        margin={"l": 165, "r": 20, "t": 65, "b": 210 if convergence_view else 110}
    )
    return figure


def _check_plot_arguments(
    result: EprResult,
    mode: int | None,
    native_pass: int | None,
    show_convergence: bool,
) -> None:
    if not isinstance(result, EprResult):
        raise TypeError("result must be EprResult")
    if mode is not None and (type(mode) is not int or mode <= 0):
        raise ValueError("mode must be a positive integer or None")
    if native_pass is not None and (type(native_pass) is not int or native_pass <= 0):
        raise ValueError("native_pass must be a positive integer or None")
    if native_pass is not None and result.result_kind != "adaptive_history":
        raise ValueError("native_pass selection requires adaptive-history results")
    if type(show_convergence) is not bool:
        raise TypeError("show_convergence must be bool")


def _add_selected_pass_bars(figure: Any, go: Any, view: dict[str, Any]) -> None:
    """Top panel: one bar group per mode at the selected pass."""
    ticks: list[str] = []
    tick_labels: list[str] = []
    for bar in view["bars"]:
        keys = [f"{bar['name']}:{key}" for key in bar["keys"]]
        ticks.extend(keys)
        tick_labels.extend(bar["labels"])
        figure.add_trace(
            go.Bar(
                name=bar["name"],
                x=keys,
                y=bar["values"],
                marker_color=[
                    INTERFACE_COLORS.get(kind.removeprefix("surface:"), "#CC79A7")
                    for kind in bar["kinds"]
                ],
                customdata=bar["labels"],
                hovertemplate="%{customdata}<br>p=%{y:.6g}<extra>%{fullData.name}</extra>",
            ),
            row=1,
            col=1,
        )
    figure.update_xaxes(
        tickmode="array", tickvals=ticks, ticktext=tick_labels, row=1, col=1
    )


def _add_convergence_lines(figure: Any, go: Any, view: dict[str, Any]) -> None:
    """Lower panels: frequency (row 2), participation (row 3), energy (row 4) by pass."""
    # Each quantity keeps one dash and marker across modes, so it stays
    # recognisable without relying on colour alone.
    owner_styles: dict[str, int] = {}
    dashes = ("solid", "dash", "dot", "dashdot")
    symbols = ("circle", "diamond", "square", "cross")
    for line in view["lines"]:
        kind = line["kind"]
        row_number = 2 if kind == "frequency" else 4 if kind == "energy" else 3
        style_index = owner_styles.setdefault(line["identity"], len(owner_styles))
        color = (
            INTERFACE_COLORS.get(line.get("interface_kind"), "#CC79A7")
            if kind == "surface"
            else "#0072B2"
            if kind == "frequency"
            else "#D55E00"
            if kind.startswith("junction")
            else "#E69F00"
        )
        figure.add_trace(
            go.Scatter(
                name=line["name"],
                x=line["x"],
                y=line["y"],
                mode="lines+markers",
                connectgaps=False,
                line={"color": color, "dash": dashes[style_index % len(dashes)]},
                marker={"symbol": symbols[style_index % len(symbols)]},
                customdata=[line.get("hover", line["identity"])] * len(line["x"]),
                meta=line.get("source_metadata", {"identity": line["identity"]}),
                hovertemplate="%{fullData.name}<br>pass %{x}<br>value %{y:.6g}<br>"
                "%{customdata}<extra></extra>",
            ),
            row=row_number,
            col=1,
        )


def _title_plot_axes(figure: Any, view: dict[str, Any]) -> None:
    figure.update_xaxes(title_text="Contribution", row=1, col=1)
    figure.update_yaxes(title_text="Participation", row=1, col=1)
    if view["convergence_view"]:
        for row_number in (2, 3, 4):
            figure.update_xaxes(
                title_text="Adaptive pass",
                tickmode="array",
                tickvals=list(range(1, view["native_pass"] + 1)),
                row=row_number,
                col=1,
            )
        figure.update_yaxes(title_text="Frequency (Hz)", row=2, col=1)
        figure.update_yaxes(title_text="Participation", row=3, col=1)
        figure.update_yaxes(
            title_text="Magnetic+inductive / normalization", row=4, col=1
        )


def show_view(
    result: EprResult, *, mode: int | None, native_pass: int | None
) -> dict[str, Any]:
    """Select the one row to show and precompute its coverage counts."""
    if not isinstance(result, EprResult):
        raise TypeError("result must be EprResult")
    selected = [
        row
        for row in result.rows
        if (mode is None or row["mode"] == mode)
        and (native_pass is None or row.get("native_pass") == native_pass)
    ]
    if len(selected) != 1:
        raise ValueError("show_epr requires one exact mode/pass selection")
    row = selected[0]
    has_native_references = "native_surface_references" in row
    native_references = row.get("native_surface_references")
    if row["status"] != "complete":
        frequency = usable_frequency(row.get("frequency_hz"))
        return {
            "row": row,
            "complete": False,
            "has_native_references": has_native_references,
            "native_references": native_references,
            "frequency_text": (
                f"; verified frequency {frequency:.6g} Hz"
                if frequency is not None
                else ""
            ),
        }
    surfaces = row["surface_contributions"]
    domains = row["electric_domains"]
    junctions = row["junctions"]
    baselines = [
        item for item in surfaces if item.get("evaluation_kind") == "unmasked_baseline"
    ]
    known_baselines = [
        item["inverse_q"] for item in baselines if item.get("inverse_q") is not None
    ]
    return {
        "row": row,
        "complete": True,
        "has_native_references": has_native_references,
        "native_references": native_references,
        "surfaces": surfaces,
        "domains": domains,
        "junctions": junctions,
        "bulk_participations": [
            item["energy_j"] / row["normalization_energy_j"] for item in domains
        ],
        "coverage": {
            "surface_rows": len(surfaces),
            "bindings": sum(
                len(item.get("members", item.get("binding_ids", ())))
                for item in surfaces
            ),
            "bulk_domains": len(domains),
            "junctions": len(junctions),
            "known_baseline_subtotal": (
                f"{sum(known_baselines):.6g}" if known_baselines else "unavailable"
            ),
            "unknown_baseline_losses": sum(
                item.get("inverse_q") is None for item in baselines
            ),
        },
    }


def show_epr(
    result: EprResult,
    *,
    mode: int | None = None,
    native_pass: int | None = None,
    theme: str = "light",
) -> Any:
    """Show one exact pass/mode with raw native references kept separate."""

    checked_theme(theme)
    view = show_view(result, mode=mode, native_pass=native_pass)
    row = view["row"]
    from plotly import graph_objects as go
    from plotly.subplots import make_subplots

    has_native_references = view["has_native_references"]
    subplot_titles = ("Surface", "Bulk", "Junction")
    specs = None
    if has_native_references:
        subplot_titles += (
            "Native-face raw references · not participation or loss · support may differ",
        )
        specs = [
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "table"}],
        ]
    figure = make_subplots(
        rows=len(subplot_titles),
        cols=1,
        subplot_titles=subplot_titles,
        specs=specs,
    )
    top_margin = 65
    if view["complete"]:
        _add_surface_panel(figure, go, view)
        _add_bulk_and_junction_panels(figure, go, view)
        top_margin = _add_coverage_note(figure, view["coverage"])
    else:
        figure.add_annotation(
            text=f"Partial native result{view['frequency_text']}; EPR quantities unavailable",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
    if has_native_references:
        _add_native_reference_panel(figure, go, view["native_references"])
    title = f"EPR mode {row['mode']}" + (
        f" pass {row['native_pass']}" if "native_pass" in row else " saved field"
    )
    figure.update_layout(
        title={
            "text": title,
            "x": 0.02,
            "xanchor": "left",
            "y": 0.99,
            "yanchor": "top",
            "yref": "container",
        }
        if view["complete"]
        else title,
        showlegend=True,
        meta={
            "surface_identities": [
                surface_key(item) for item in view.get("surfaces", ())
            ],
            "mode": row["mode"],
            "native_pass": row.get("native_pass"),
            **(
                {"native_surface_references": detached(view["native_references"])}
                if has_native_references
                else {}
            ),
        },
    )
    for axis in ("xaxis", "xaxis2", "xaxis3"):
        figure.layout[axis].title = "participation"
    style_figure(
        figure,
        theme=theme,
        height=_show_height(view, top_margin, has_native_references),
    )
    figure.update_layout(margin={"l": 180, "r": 20, "t": top_margin, "b": 125})
    figure.update_yaxes(automargin=False, tickfont={"size": 12}, row=1, col=1)
    return figure


def _show_height(
    view: dict[str, Any], top_margin: int, has_native_references: bool
) -> int:
    table_extra = 230 if has_native_references else 0
    if not view["complete"]:
        return 900 + table_extra
    return (
        max(
            900 + top_margin - 65,
            ranking_height(len(view.get("surfaces", ()))) + 320 + top_margin - 65,
        )
        + table_extra
    )


def _add_surface_panel(figure: Any, go: Any, view: dict[str, Any]) -> None:
    """Panel 1: surface participation, largest first."""
    surfaces = sorted(view["surfaces"], key=lambda item: -float(item["participation"]))
    surface_keys = [surface_key(item) for item in surfaces]
    surface_labels = [surface_tick_label(item) for item in surfaces]
    figure.add_trace(
        go.Bar(
            x=[item["participation"] for item in surfaces],
            y=surface_keys,
            orientation="h",
            name="surface p",
            marker_color=[
                INTERFACE_COLORS.get(item["interface_kind"], "#CC79A7")
                for item in surfaces
            ],
            customdata=[surface_hover(item) for item in surfaces],
            meta=[surface_metadata(item) for item in surfaces],
            hovertemplate="p=%{x:.6g}<br>%{customdata}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.update_yaxes(
        tickmode="array",
        tickvals=surface_keys,
        ticktext=surface_labels,
        autorange="reversed",
        row=1,
        col=1,
    )


def _add_bulk_and_junction_panels(figure: Any, go: Any, view: dict[str, Any]) -> None:
    """Panel 2: bulk domains. Panel 3: junction inductive and capacitive participation."""
    domains = view["domains"]
    figure.add_trace(
        go.Bar(
            x=view["bulk_participations"],
            y=[html.escape(str(item["domain_id"])) for item in domains],
            orientation="h",
            name="bulk p",
        ),
        row=2,
        col=1,
    )
    junctions = view["junctions"]
    for field, name in (
        ("inductive_participation", "junction pL"),
        ("capacitive_participation", "junction pC"),
    ):
        figure.add_trace(
            go.Bar(
                x=[item[field] for item in junctions],
                y=[html.escape(str(item["junction_id"])) for item in junctions],
                orientation="h",
                name=name,
            ),
            row=3,
            col=1,
        )


def _add_coverage_note(figure: Any, coverage: dict[str, Any]) -> int:
    """Add the coverage summary above the plot; return the top margin it needs."""
    coverage_text = (
        f"Coverage: {coverage['surface_rows']} surface rows / "
        f"{coverage['bindings']} bindings; "
        f"{coverage['bulk_domains']} bulk domains; {coverage['junctions']} junctions. "
        f"Known baseline 1/Q subtotal: {coverage['known_baseline_subtotal']}; "
        f"{coverage['unknown_baseline_losses']} baseline losses unknown. "
        "Sidewalls uncomputed; requested margins are alternatives."
    )
    coverage_lines = textwrap.wrap(coverage_text, width=32, break_long_words=True)
    top_margin = max(240, 135 + 14 * len(coverage_lines))
    figure.add_annotation(
        text="<br>".join(html.escape(line) for line in coverage_lines),
        xref="paper",
        yref="paper",
        x=0,
        y=1,
        yshift=top_margin - 110,
        xanchor="left",
        yanchor="top",
        align="left",
        font={"size": 11},
        showarrow=False,
    )
    return top_margin


def _add_native_reference_panel(figure: Any, go: Any, native_references: Any) -> None:
    """Panel 4: raw native-face references, shown as recorded."""
    headers, native_rows = native_reference_table_data(native_references or ())
    if native_rows:
        columns = [list(column) for column in zip(*native_rows, strict=False)]
    else:
        columns = [[] for _ in headers]
    figure.add_trace(
        go.Table(
            header={"values": list(headers), "align": "left"},
            cells={"values": columns, "align": "left"},
        ),
        row=4,
        col=1,
    )
    if not native_rows:
        empty_note = (
            "The recorded native reference value is null."
            if native_references is None
            else "The recorded native reference sequence is empty."
        )
        figure.add_annotation(
            text=html.escape(empty_note),
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.08,
            showarrow=False,
        )
