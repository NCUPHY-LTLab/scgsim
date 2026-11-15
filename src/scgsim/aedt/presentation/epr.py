"""Plot (``plot_epr_result``) and show (``show_epr``) an EPR result in a notebook."""

from __future__ import annotations

import html
import textwrap
from typing import Any

from scgsim.aedt.data import detached
from scgsim.aedt.epr.models import EprResult
from scgsim.aedt.presentation.epr_projection import (
    history_view,
    native_reference_table_data,
    ranking_height,
    saved_solution_versions_text,
    show_view,
    surface_hover,
    surface_key,
    surface_metadata,
    surface_tick_label,
)
from scgsim.presentation.notebook import (
    INTERFACE_COLORS,
    checked_theme,
    style_figure,
)


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
    from plotly import graph_objects as go  # noqa: PLC0415 - Plotly is an optional plot extra
    from plotly.subplots import make_subplots  # noqa: PLC0415 - Plotly is an optional plot extra

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
    if convergence_view:
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
    figure.update_xaxes(title_text="Contribution", row=1, col=1)
    figure.update_yaxes(title_text="Participation", row=1, col=1)
    if convergence_view:
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
    from plotly import graph_objects as go  # noqa: PLC0415 - Plotly is an optional plot extra
    from plotly.subplots import make_subplots  # noqa: PLC0415 - Plotly is an optional plot extra

    figure = _show_epr_figure(view, result, theme, go, make_subplots)
    return figure


def _show_epr_figure(
    view: dict[str, Any],
    result: EprResult,
    theme: str,
    go: Any,
    make_subplots: Any,
) -> Any:
    """Assemble the selected-pass display from detached projection records."""
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
    if view["complete"]:
        top_margin = _add_complete_show_traces(figure, go, view)
    else:
        top_margin = 65
        figure.add_annotation(
            text=f"Partial native result{view['frequency_text']}; EPR quantities unavailable",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
    if has_native_references:
        _add_native_reference_table(figure, go, view)
    _style_show_epr_figure(figure, view, result, theme, top_margin)
    return figure


def _add_complete_show_traces(figure: Any, go: Any, view: dict[str, Any]) -> int:
    """Add complete-row participation traces and coverage annotation."""
    surfaces = sorted(view["surfaces"], key=lambda item: -float(item["participation"]))
    surface_keys = [surface_key(item) for item in surfaces]
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
        ticktext=[surface_tick_label(item) for item in surfaces],
        autorange="reversed",
        row=1,
        col=1,
    )
    _add_bulk_and_junction_traces(figure, go, view)
    coverage = view["coverage"]
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


def _add_bulk_and_junction_traces(figure: Any, go: Any, view: dict[str, Any]) -> None:
    """Add the independent bulk, inductive, and capacitive participation bars."""
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
    for field_name, trace_name in (
        ("inductive_participation", "junction pL"),
        ("capacitive_participation", "junction pC"),
    ):
        figure.add_trace(
            go.Bar(
                x=[item[field_name] for item in junctions],
                y=[html.escape(str(item["junction_id"])) for item in junctions],
                orientation="h",
                name=trace_name,
            ),
            row=3,
            col=1,
        )


def _add_native_reference_table(figure: Any, go: Any, view: dict[str, Any]) -> None:
    """Add raw native-face references without deriving participation or loss."""
    native_references = view["native_references"]
    headers, native_rows = native_reference_table_data(native_references or ())
    columns = (
        [list(column) for column in zip(*native_rows, strict=True)]
        if native_rows
        else [[] for _ in headers]
    )
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


def _style_show_epr_figure(
    figure: Any,
    view: dict[str, Any],
    result: EprResult,
    theme: str,
    top_margin: int,
) -> None:
    """Set selected-row identity, axes, and existing responsive plot sizing."""
    row = view["row"]
    has_native_references = view["has_native_references"]
    title = f"EPR mode {row['mode']}" + (
        f" pass {row['native_pass']}" if "native_pass" in row else " saved field"
    )
    if result.result_kind == "saved_field":
        title += (
            "<br><sup>" + saved_solution_versions_text(result.provenance) + "</sup>"
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
        height=max(
            900 + top_margin - 65,
            ranking_height(len(view.get("surfaces", ()))) + 320 + top_margin - 65,
        )
        + (230 if has_native_references else 0)
        if view["complete"]
        else 900 + (230 if has_native_references else 0),
    )
    figure.update_layout(margin={"l": 180, "r": 20, "t": top_margin, "b": 125})
    figure.update_yaxes(automargin=False, tickfont={"size": 12}, row=1, col=1)
