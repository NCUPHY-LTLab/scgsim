"""Notebook display of a completed, verified AEDT run.

Owns: the section order of ``display_resolved_run`` and the per-pass, per-margin and
final-pass figures and tables. Does not own: verifying the run (``scgsim.aedt.results``)
or the shared EPR projections (``epr.py``, ``epr_history.py``, ``epr_surfaces.py``).
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from typing import Any

from scgsim.aedt.epr.models import EprResult, detached
from scgsim.aedt.presentation.benchmark import display_benchmark
from scgsim.aedt.presentation.epr import show_view
from scgsim.aedt.presentation.epr_history import history_view
from scgsim.aedt.presentation.epr_surfaces import (
    native_reference_html,
    ranking_height,
    surface_hover,
    surface_key,
    surface_label,
    surface_metadata,
    surface_tick_label,
    usable_frequency,
)
from scgsim.presentation.notebook import (
    INTERFACE_COLORS,
    cards,
    checked_theme,
    details,
    json_details,
    section,
    style_figure,
    table,
)


def _history_figures(result: EprResult, *, theme: str) -> tuple[Any, ...]:
    """Separate the existing pass projections without changing eligibility."""

    from plotly import graph_objects as go

    view = history_view(result, mode=None, native_pass=None, show_convergence=True)
    figures = []
    for kind, selected_mode, selected_interface, title, y_title in _history_groups(
        view
    ):
        figure = go.Figure()
        seen: dict[str, int] = {}
        for line in view["lines"]:
            if not _line_belongs(line, kind, selected_mode, selected_interface):
                continue
            index = seen.setdefault(line["identity"], len(seen))
            _add_history_line(figure, line, kind, index)
        figure.update_layout(
            title=title,
            xaxis_title="Adaptive pass",
            yaxis_title=y_title,
            meta={
                "requested_modes": list(result.provenance["requested_modes"]),
                "observed_pass": view["native_pass"],
                "kind": kind,
                "mode": selected_mode,
                "interface_kind": selected_interface,
            },
        )
        figure.update_xaxes(
            tickmode="array", tickvals=list(range(1, view["native_pass"] + 1))
        )
        legend_rows = len(figure.data)
        style_figure(figure, theme=theme, height=max(460, 440 + 28 * legend_rows))
        figure.update_layout(
            margin={"l": 70, "r": 20, "t": 65, "b": max(115, 75 + 29 * legend_rows)}
        )
        figures.append(figure)
    return tuple(figures)


def _history_groups(view: dict[str, Any]) -> list[tuple[Any, ...]]:
    """One figure each: frequency, surfaces per (mode, interface), junctions, energy."""
    surface_groups = sorted(
        {
            (line["mode"], line["interface_kind"])
            for line in view["lines"]
            if line["kind"] == "surface"
        },
        key=lambda item: (
            item[0],
            ("MA", "MS", "SA").index(item[1]) if item[1] in {"MA", "MS", "SA"} else 3,
            item[1],
        ),
    )
    return [
        ("frequency", None, None, "Frequency by adaptive pass", "Frequency (Hz)"),
        *(
            (
                "surface",
                mode,
                interface,
                f"Surface participation · mode {mode} / {interface}",
                "Participation",
            )
            for mode, interface in surface_groups
        ),
        (
            "junction",
            None,
            None,
            "Junction participation by adaptive pass",
            "Participation",
        ),
        (
            "energy",
            None,
            None,
            "Magnetic and inductive energy balance",
            "Balance / normalization",
        ),
    ]


def _line_belongs(
    line: dict[str, Any],
    kind: str,
    selected_mode: int | None,
    selected_interface: str | None,
) -> bool:
    if not (
        line["kind"] == kind
        or kind == "junction"
        and line["kind"].startswith("junction")
    ):
        return False
    return selected_mode is None or (
        line["mode"] == selected_mode
        and line.get("interface_kind") == selected_interface
    )


def _add_history_line(figure: Any, line: dict[str, Any], kind: str, index: int) -> None:
    color = (
        INTERFACE_COLORS.get(line.get("interface_kind"), "#CC79A7")
        if kind == "surface"
        else "#0072B2"
        if kind == "frequency"
        else "#D55E00"
        if kind == "junction"
        else "#E69F00"
    )
    figure.add_scatter(
        x=line["x"],
        y=line["y"],
        name=line["name"],
        mode="lines+markers",
        connectgaps=False,
        line={
            "color": color,
            "dash": ("solid", "dash", "dot", "dashdot")[index % 4],
        },
        marker={"symbol": ("circle", "diamond", "square", "cross")[index % 4]},
        customdata=[line.get("hover", line["identity"])] * len(line["x"]),
        meta=line.get("source_metadata", {"identity": line["identity"]}),
        hovertemplate="%{fullData.name}<br>pass %{x}<br>value %{y:.6g}<br>"
        "%{customdata}<extra></extra>",
    )


def _margin_figure(row: Any, *, theme: str) -> Any | None:
    """Keep requested margins within exact source/film lineages; baseline is a marker."""

    from plotly import graph_objects as go

    surfaces = row["surface_contributions"]
    if not surfaces:
        return None
    groups: dict[str, list[Any]] = {}
    for item in surfaces:
        members = item.get("members", item.get("binding_ids", ()))
        lineage = json.dumps(
            {
                "owner_id": item.get("owner_id"),
                "interface_kind": item["interface_kind"],
                "field_side": item.get("field_side"),
                "members": detached(members),
                "assumptions": detached(item.get("assumptions")),
            },
            sort_keys=True,
        )
        groups.setdefault(lineage, []).append(item)
    figure = go.Figure()
    for index, items in enumerate(groups.values()):
        first = items[0]
        color = INTERFACE_COLORS.get(first["interface_kind"], "#CC79A7")
        requested = sorted(
            (
                item
                for item in items
                if item.get("evaluation_kind") != "unmasked_baseline"
            ),
            key=lambda item: item["margin_um"],
        )
        baseline = [
            item for item in items if item.get("evaluation_kind") == "unmasked_baseline"
        ]
        label = surface_label(first).split(" · ")[0] + f" · {first['interface_kind']}"
        for evaluation, values, mode in (
            ("Requested", requested, "lines+markers"),
            ("Unmasked", baseline, "markers"),
        ):
            if not values:
                continue
            figure.add_scatter(
                x=[item["margin_um"] for item in values],
                y=[item["participation"] for item in values],
                mode=mode,
                name=f"{label} · {evaluation}",
                connectgaps=False,
                line={
                    "color": color,
                    "dash": ("solid", "dash", "dot", "dashdot")[index % 4],
                },
                marker={
                    "color": color,
                    "symbol": ("circle", "diamond", "square", "cross")[index % 4],
                },
                customdata=[surface_hover(item) for item in values],
                meta=[surface_metadata(item) for item in values],
                hovertemplate="%{fullData.name}<br>margin %{x:g} µm<br>p=%{y:.6g}<br>"
                "%{customdata}<extra></extra>",
            )
    figure.update_layout(
        title=f"Mode {row['mode']} requested margins and unmasked baseline",
        xaxis_title="Requested margin (µm)",
        yaxis_title="Surface participation",
        meta={"mode": row["mode"], "native_pass": row.get("native_pass")},
    )
    return style_figure(figure, theme=theme)


def _final_quantity_figures(row: Any, *, theme: str) -> tuple[Any, ...]:
    """Present each final-pass quantity independently without recombination."""

    from plotly import graph_objects as go

    surfaces = sorted(
        row["surface_contributions"], key=lambda item: -float(item["participation"])
    )
    surface_keys = [surface_key(item) for item in surfaces]
    surface = go.Figure(
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
        )
    )
    surface.update_yaxes(
        tickmode="array",
        tickvals=surface_keys,
        ticktext=[surface_tick_label(item) for item in surfaces],
        autorange="reversed",
    )
    surface.update_layout(
        title=f"Mode {row['mode']} surface owners · pass {row['native_pass']}",
        xaxis_title="Surface participation",
    )
    style_figure(surface, theme=theme, height=ranking_height(len(surfaces)))
    surface.update_layout(
        showlegend=False, margin={"l": 180, "r": 20, "t": 65, "b": 75}
    )
    surface.update_yaxes(automargin=False, tickfont={"size": 12})
    figures = [surface]
    bulk = go.Figure(
        go.Bar(
            x=[
                item["energy_j"] / row["normalization_energy_j"]
                for item in row["electric_domains"]
            ],
            y=[html.escape(str(item["domain_id"])) for item in row["electric_domains"]],
            orientation="h",
            name="bulk p",
            marker_color="#0072B2",
        )
    )
    bulk.update_layout(
        title=f"Mode {row['mode']} bulk domains · pass {row['native_pass']}",
        xaxis_title="Bulk participation",
    )
    style_figure(bulk, theme=theme, height=ranking_height(len(row["electric_domains"])))
    bulk.update_layout(showlegend=False, margin={"l": 140, "r": 20, "t": 65, "b": 75})
    figures.append(bulk)
    for field, label, color in (
        ("inductive_participation", "junction pL", "#D55E00"),
        ("capacitive_participation", "junction pC", "#CC79A7"),
    ):
        junction = go.Figure(
            go.Bar(
                x=[item[field] for item in row["junctions"]],
                y=[html.escape(str(item["junction_id"])) for item in row["junctions"]],
                orientation="h",
                name=label,
                marker_color=color,
            )
        )
        junction.update_layout(
            title=f"Mode {row['mode']} {label} · pass {row['native_pass']}",
            xaxis_title=label,
        )
        style_figure(
            junction, theme=theme, height=ranking_height(len(row["junctions"]))
        )
        junction.update_layout(
            showlegend=False, margin={"l": 140, "r": 20, "t": 65, "b": 75}
        )
        figures.append(junction)
    return tuple(figures)


def _physics_tables(
    result: EprResult, row: Any, *, show_details: bool, plotly_available: bool
) -> str:
    view = show_view(result, mode=row["mode"], native_pass=row["native_pass"])
    coverage = view["coverage"]
    surfaces = sorted(view["surfaces"], key=lambda item: -float(item["participation"]))
    surface_rows = [
        (
            item.get("owner_id", item.get("contribution_id")),
            item["interface_kind"],
            "Unmasked"
            if item.get("evaluation_kind") == "unmasked_baseline"
            else "Requested",
            item["margin_um"],
            item["participation"],
            item.get("assumptions", {}).get("loss_tangent"),
            item.get("inverse_q"),
            item.get("group_id"),
        )
        for item in surfaces
    ]
    surface_table = table(
        (
            "Owner",
            "Interface",
            "Evaluation",
            "Margin (µm)",
            "p",
            "Loss tangent",
            "1/Q",
            "Group",
        ),
        surface_rows,
    )
    bulk_table = table(
        ("Domain", "Energy (J)", "Participation"),
        [
            (item["domain_id"], item["energy_j"], participation)
            for item, participation in zip(
                view["domains"], view["bulk_participations"], strict=False
            )
        ],
    )
    junction_table = table(
        ("Junction", "pL", "pC"),
        [
            (
                item["junction_id"],
                item["inductive_participation"],
                item["capacitive_participation"],
            )
            for item in view["junctions"]
        ],
    )
    return (
        cards(
            (
                ("Surface rows", coverage["surface_rows"]),
                ("Source bindings", coverage["bindings"]),
                ("Bulk domains", coverage["bulk_domains"]),
                ("Junctions", coverage["junctions"]),
                ("Known baseline 1/Q subtotal", coverage["known_baseline_subtotal"]),
                ("Unknown baseline loss", coverage["unknown_baseline_losses"]),
            )
        )
        + "<p>Sidewalls uncomputed; requested margins are alternative evaluations. "
        "No total device Q or T1 is inferred.</p>"
        + native_reference_html(row, opened=show_details)
        + details(
            "Surface owner, interface, margin, and loss table",
            surface_table,
            opened=show_details or not plotly_available or len(surfaces) <= 12,
        )
        + details(
            "Bulk participation table",
            bulk_table,
            opened=show_details or not plotly_available,
        )
        + details(
            "Junction inductive and capacitive table",
            junction_table,
            opened=show_details or not plotly_available,
        )
    )


@dataclass(frozen=True)
class _DisplayOptions:
    show_details: bool
    theme: str


def display_resolved_run(
    resolved: Any, *, show_details: bool, theme: str = "light"
) -> None:
    """Show verified identity, numerical evidence, cost, and final-pass physics."""

    checked_theme(theme)
    # Fail before reading the run if IPython is missing.
    from IPython.display import display as _  # noqa: F401

    rows = resolved.physics_results()
    benchmark = resolved.simulation_benchmark()
    result = resolved.epr_result()
    final_pass = (
        result.provenance["solver_last_completed_pass"] if result is not None else None
    )
    modes = result.provenance["requested_modes"] if result is not None else None

    _display_run_summary(resolved, modes, final_pass, theme)
    _display_lumped_elements(resolved, show_details, theme)
    _display_primary_results(rows, show_details, theme)
    plotly_available = _display_history(result, theme)
    display_benchmark(benchmark, show_details=show_details, theme=theme)
    if result is None:
        _show_section(
            "AEDT physics", "<p>Adaptive EPR was not requested for this run.</p>", theme
        )
    else:
        for mode in modes:
            plotly_available = _display_mode_physics(
                result,
                mode,
                final_pass,
                _DisplayOptions(show_details, theme),
                plotly_available=plotly_available,
            )
    _display_identity(resolved, result, show_details, theme)


def _show_section(title: str, body: str, theme: str) -> None:
    from IPython.display import HTML, display

    display(HTML(section(title, body, theme=theme)))


def _pass_cards(resolved: Any, final_pass: Any) -> tuple[tuple[str, Any], ...]:
    if resolved.mode != "q3d":
        return (("Observed final pass", final_pass),)
    # Independent native problems need their own pass attribution, never a maximum.
    histories = resolved.convergence or {}
    return tuple(
        (label, histories[problem].get("final_pass"))
        for problem, label in (
            ("capacitance", "Capacitance observed final pass"),
            ("ac_rl", "AC/RL observed final pass"),
        )
        if problem in histories
    ) or (("Q3D observed final pass", None),)


def _display_run_summary(
    resolved: Any, modes: Any, final_pass: Any, theme: str
) -> None:
    _show_section(
        "AEDT completed run",
        cards(
            (
                ("Backend", "AEDT"),
                ("Project", resolved.project_path.name),
                ("Analysis", resolved.mode),
                ("Setup", resolved.setup_name),
                (
                    "Requested modes",
                    ", ".join(map(str, modes))
                    if modes is not None
                    else "not requested",
                ),
                *_pass_cards(resolved, final_pass),
                ("Completion", "completed"),
            )
        ),
        theme,
    )


def _lumped_treatment(request: Any) -> str:
    """Describe how the author asked a lumped boundary to be treated."""
    if "topology" in request:
        return f"{request['topology']} R={request['resistance_ohm']} ohm; L={request['inductance_h']} H; C={request['capacitance_f']} F (None disabled)"
    return f"Lumped Terminal {request['impedance_ohm']} ohm; renormalize={request['renormalize']}; deembed={request['deembed_um']} um"


def _display_lumped_elements(resolved: Any, show_details: bool, theme: str) -> None:
    lumped = resolved.lumped_boundaries
    if not lumped:
        return
    summaries = []
    for record in lumped:
        request = record["requested"]
        summaries.append(
            (
                record.get("support_id", request.get("support_id")),
                record["boundary"],
                _lumped_treatment(request),
            )
        )
    _show_section(
        "Source-bound lumped elements",
        table(("Support", "Native boundary", "Authored treatment"), summaries)
        + json_details(
            "Source and native assignment evidence",
            {"boundaries": list(lumped)},
            opened=show_details,
        ),
        theme,
    )


def _display_primary_results(rows: Any, show_details: bool, theme: str) -> None:
    headings = tuple(rows[0]) if rows else ()
    primary_table = (
        table(headings, [tuple(row.get(key, "") for key in headings) for row in rows])
        if headings
        else "<p>No primary result rows were recorded.</p>"
    )
    _show_section(
        "AEDT numerical evidence",
        f"<p>{len(rows)} primary result rows verified against the completed receipt.</p>"
        + details(
            "Primary result table",
            primary_table,
            opened=show_details or len(rows) <= 20,
        ),
        theme,
    )


def _display_history(result: EprResult | None, theme: str) -> bool:
    """Show the per-pass figures; return False when Plotly is not installed."""
    if result is None:
        return True
    from IPython.display import display

    try:
        for figure in _history_figures(result, theme=theme):
            display(figure)
    except ImportError:
        _show_section(
            "Numerical figures unavailable",
            "<p>Plotly is unavailable; recorded frequencies and complete-row "
            "quantities remain in the tables below.</p>",
            theme,
        )
        return False
    return True


def _display_mode_physics(
    result: EprResult,
    mode: int,
    final_pass: Any,
    options: _DisplayOptions,
    *,
    plotly_available: bool,
) -> bool:
    """Show one mode's final-pass section; return whether Plotly is still usable."""
    from IPython.display import display

    show_details = options.show_details
    theme = options.theme

    title = f"Mode {mode} · observed pass {final_pass}"
    selected = [
        row
        for row in result.rows
        if row["mode"] == mode and row["native_pass"] == final_pass
    ]
    if not selected:
        _show_section(
            title,
            "<p>The requested final-pass EPR row is missing. Earlier passes are not substituted.</p>",
            theme,
        )
        return plotly_available
    if len(selected) != 1:
        raise RuntimeError("adaptive EPR contains duplicate mode/pass rows")
    row = selected[0]
    if row["status"] != "complete":
        frequency = usable_frequency(row.get("frequency_hz"))
        reason = row.get("combination_error", "incomplete native evidence")
        _show_section(
            title,
            cards((("Status", "partial"), ("Frequency (Hz)", frequency)))
            + "<p>Surface, bulk, and junction EPR quantities are unavailable: "
            + html.escape(str(reason))
            + ".</p>"
            + native_reference_html(row, opened=show_details),
            theme,
        )
        return plotly_available
    _show_section(
        title,
        _physics_tables(
            result, row, show_details=show_details, plotly_available=plotly_available
        ),
        theme,
    )
    if not plotly_available:
        return plotly_available
    try:
        for figure in _final_quantity_figures(row, theme=theme):
            display(figure)
        margin = _margin_figure(row, theme=theme)
        if margin is not None:
            display(margin)
    except ImportError:
        _show_section(
            "Physics figures unavailable",
            "<p>Plotly is unavailable; final-pass tables remain available.</p>",
            theme,
        )
        return False
    return True


def _display_identity(
    resolved: Any, result: EprResult | None, show_details: bool, theme: str
) -> None:
    """Show the hashes that tie this display to the verified spec, project and receipt."""
    project_sha = dict(resolved.output_hashes)[resolved.project_path.name]
    source_sha = (
        result.provenance["analysis_source_sha256"] if result is not None else None
    )
    receipt = resolved.receipt_path.relative_to(
        resolved.receipt_path.parent.parent
    ).as_posix()
    identity = {
        "receipt": receipt,
        "spec_sha256": resolved.spec_sha256,
        "project_sha256": project_sha,
        "analysis_source_sha256": source_sha,
    }
    _show_section(
        "AEDT source and receipt details",
        json_details(
            "Recorded source and receipt identity", identity, opened=show_details
        ),
        theme,
    )
