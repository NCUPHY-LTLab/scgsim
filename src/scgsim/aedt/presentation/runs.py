"""Resolved AEDT run result presentation only."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from scgsim.aedt.epr.models import EprResult, detached
from scgsim.aedt.presentation.benchmark import display_benchmark
from scgsim.aedt.presentation.epr_projection import (
    history_view,
    native_reference_html,
    ranking_height,
    saved_solution_version_text,
    show_view,
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

    from plotly import graph_objects as go  # noqa: PLC0415 - Plotly is an optional plot extra

    view = history_view(result, mode=None, native_pass=None, show_convergence=True)
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
    groups = [
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
    figures = []
    for kind, selected_mode, selected_interface, title, y_title in groups:
        figure = go.Figure()
        seen: dict[str, int] = {}
        for line in view["lines"]:
            if not (
                line["kind"] == kind
                or kind == "junction"
                and line["kind"].startswith("junction")
            ):
                continue
            if selected_mode is not None and (
                line["mode"] != selected_mode
                or line.get("interface_kind") != selected_interface
            ):
                continue
            index = seen.setdefault(line["identity"], len(seen))
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
        legend_rows = len(tuple(figure.data))
        style_figure(figure, theme=theme, height=max(460, 440 + 28 * legend_rows))
        figure.update_layout(
            margin={"l": 70, "r": 20, "t": 65, "b": max(115, 75 + 29 * legend_rows)}
        )
        figures.append(figure)
    return tuple(figures)


def _margin_figure(row: Any, *, theme: str) -> Any | None:
    """Keep requested margins within exact source/film lineages; baseline is a marker."""

    from plotly import graph_objects as go  # noqa: PLC0415 - Plotly is an optional plot extra

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

    from plotly import graph_objects as go  # noqa: PLC0415 - Plotly is an optional plot extra

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
                view["domains"], view["bulk_participations"], strict=True
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


def display_resolved_run(
    resolved: Any, *, show_details: bool, theme: str = "light"
) -> None:
    """Show detached numerical, physics, and receipt projections in order."""
    checked_theme(theme)
    from IPython.display import HTML, display  # noqa: PLC0415 - notebook output is optional

    rows = resolved.physics_results()
    benchmark = resolved.simulation_benchmark()
    result = resolved.epr_result()
    final_pass = (
        result.provenance["solver_last_completed_pass"] if result is not None else None
    )
    modes = result.provenance["requested_modes"] if result is not None else None
    view = _ResolvedRunDisplay(
        resolved,
        show_details,
        theme,
        HTML,
        display,
        rows,
        benchmark,
        result,
        final_pass,
        modes,
    )
    view.display()


@dataclass
class _ResolvedRunDisplay:
    """Carry one detached result and notebook context through display stages."""

    resolved: Any
    show_details: bool
    theme: str
    html_type: Any
    notebook_display: Any
    rows: list[dict[str, Any]]
    benchmark: dict[str, Any]
    result: EprResult | None
    final_pass: Any
    modes: Any
    plotly_available: bool = True

    def display(self) -> None:
        """Render run identity, tables, plots, physics, then receipt hashes."""
        self.display_identity()
        self.display_lumped_boundaries()
        self.display_numerical_results()
        self.display_history()
        display_benchmark(
            self.benchmark, show_details=self.show_details, theme=self.theme
        )
        self.display_physics()
        self.display_source_identity()

    def display_identity(self) -> None:
        """Show run identity and captured release/pass labels."""
        resolved = self.resolved
        version_cards = (
            (
                "Requested AEDT version",
                resolved.requested_aedt_version or "not recorded",
            ),
            (
                "Connected AEDT version",
                resolved.connected_aedt_version or "not recorded",
            ),
        )
        if resolved.epr_requested:
            version_cards = (
                (
                    "Saved-solution AEDT version",
                    saved_solution_version_text(
                        {
                            "saved_solution_aedt_version": resolved.saved_solution_aedt_version,
                            "requested_aedt_version": resolved.requested_aedt_version,
                            "connected_aedt_version": resolved.connected_aedt_version,
                        }
                    ),
                ),
                *version_cards,
            )
        pass_cards = (("Observed final pass", self.final_pass),)
        if resolved.mode == "q3d":
            # Independent native problems need their own pass attribution, never a maximum.
            histories = resolved.convergence or {}
            pass_cards = tuple(
                (label, histories[problem].get("final_pass"))
                for problem, label in (
                    ("capacitance", "Capacitance observed final pass"),
                    ("ac_rl", "AC/RL observed final pass"),
                )
                if problem in histories
            ) or (("Q3D observed final pass", None),)
        self.notebook_display(
            self.html_type(
                section(
                    "AEDT completed run",
                    cards(
                        (
                            ("Backend", "AEDT"),
                            ("Project", resolved.project_path.name),
                            ("Analysis", resolved.mode),
                            ("Setup", resolved.setup_name),
                            *version_cards,
                            (
                                "Requested modes",
                                ", ".join(map(str, self.modes))
                                if self.modes is not None
                                else "not requested",
                            ),
                            *pass_cards,
                            ("Completion", "completed"),
                        )
                    ),
                    theme=self.theme,
                )
            )
        )

    def display_lumped_boundaries(self) -> None:
        """Show captured requested treatments beside native boundary evidence."""
        lumped = self.resolved.lumped_boundaries
        if not lumped:
            return
        summaries = []
        for record in lumped:
            request = record["requested"]
            if "topology" in request:
                treatment = f"{request['topology']} R={request['resistance_ohm']} ohm; L={request['inductance_h']} H; C={request['capacitance_f']} F (None disabled)"
            else:
                treatment = f"Lumped Terminal {request['impedance_ohm']} ohm; renormalize={request['renormalize']}; deembed={request['deembed_um']} um"
            summaries.append(
                (
                    record.get("support_id", request.get("support_id")),
                    record["boundary"],
                    treatment,
                )
            )
        self.notebook_display(
            self.html_type(
                section(
                    "Source-bound lumped elements",
                    table(
                        ("Support", "Native boundary", "Authored treatment"), summaries
                    )
                    + json_details(
                        "Source and native assignment evidence",
                        {"boundaries": list(lumped)},
                        opened=self.show_details,
                    ),
                    theme=self.theme,
                )
            )
        )

    def display_numerical_results(self) -> None:
        """Show verified primary numerical rows."""
        headings = tuple(self.rows[0]) if self.rows else ()
        primary_table = (
            table(
                headings,
                [tuple(row.get(key, "") for key in headings) for row in self.rows],
            )
            if headings
            else "<p>No primary result rows were recorded.</p>"
        )
        self.notebook_display(
            self.html_type(
                section(
                    "AEDT numerical evidence",
                    f"<p>{len(self.rows)} primary result rows verified against the completed receipt.</p>"
                    + details(
                        "Primary result table",
                        primary_table,
                        opened=self.show_details or len(self.rows) <= 20,
                    ),
                    theme=self.theme,
                )
            )
        )

    def display_history(self) -> None:
        """Show adaptive figures when the optional Plotly extra is available."""
        if self.result is None:
            return
        try:
            for figure in _history_figures(self.result, theme=self.theme):
                self.notebook_display(figure)
        except ImportError:
            self.plotly_available = False
            self.notebook_display(
                self.html_type(
                    section(
                        "Numerical figures unavailable",
                        "<p>Plotly is unavailable; recorded frequencies and complete-row quantities remain in the tables below.</p>",
                        theme=self.theme,
                    )
                )
            )

    def display_physics(self) -> None:
        """Show exact final-pass EPR rows and their separate display projections."""
        if self.result is None:
            self.notebook_display(
                self.html_type(
                    section(
                        "AEDT physics",
                        "<p>Adaptive EPR was not requested for this run.</p>",
                        theme=self.theme,
                    )
                )
            )
            return
        for mode in self.modes:
            self._display_mode(mode)

    def _display_mode(self, mode: int) -> None:
        """Render one requested mode at the observed final pass only."""
        result = self.result
        assert result is not None
        selected = [
            row
            for row in result.rows
            if row["mode"] == mode and row["native_pass"] == self.final_pass
        ]
        if not selected:
            self.notebook_display(
                self.html_type(
                    section(
                        f"Mode {mode} · observed pass {self.final_pass}",
                        "<p>The requested final-pass EPR row is missing. Earlier passes are not substituted.</p>",
                        theme=self.theme,
                    )
                )
            )
            return
        if len(selected) != 1:
            raise RuntimeError("adaptive EPR contains duplicate mode/pass rows")
        row = selected[0]
        if row["status"] != "complete":
            self._display_partial_mode(mode, row)
            return
        self.notebook_display(
            self.html_type(
                section(
                    f"Mode {mode} · observed pass {self.final_pass}",
                    _physics_tables(
                        result,
                        row,
                        show_details=self.show_details,
                        plotly_available=self.plotly_available,
                    ),
                    theme=self.theme,
                )
            )
        )
        if self.plotly_available:
            self._display_mode_figures(row)

    def _display_partial_mode(self, mode: int, row: Mapping[str, Any]) -> None:
        """Show a partial row's status and any frequency/reference facts."""
        frequency = usable_frequency(row.get("frequency_hz"))
        self.notebook_display(
            self.html_type(
                section(
                    f"Mode {mode} · observed pass {self.final_pass}",
                    cards((("Status", "partial"), ("Frequency (Hz)", frequency)))
                    + "<p>Surface, bulk, and junction EPR quantities are unavailable: "
                    + html.escape(
                        str(row.get("combination_error", "incomplete native evidence"))
                    )
                    + ".</p>"
                    + native_reference_html(row, opened=self.show_details),
                    theme=self.theme,
                )
            )
        )

    def _display_mode_figures(self, row: Mapping[str, Any]) -> None:
        """Show final-quantity and requested-margin figures independently."""
        try:
            for figure in _final_quantity_figures(row, theme=self.theme):
                self.notebook_display(figure)
            margin = _margin_figure(row, theme=self.theme)
            if margin is not None:
                self.notebook_display(margin)
        except ImportError:
            self.plotly_available = False
            self.notebook_display(
                self.html_type(
                    section(
                        "Physics figures unavailable",
                        "<p>Plotly is unavailable; final-pass tables remain available.</p>",
                        theme=self.theme,
                    )
                )
            )

    def display_source_identity(self) -> None:
        """Show receipt and source hashes captured during result resolution."""
        resolved = self.resolved
        result = self.result
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
        self.notebook_display(
            self.html_type(
                section(
                    "AEDT source and receipt details",
                    json_details(
                        "Recorded source and receipt identity",
                        identity,
                        opened=self.show_details,
                    ),
                    theme=self.theme,
                )
            )
        )
