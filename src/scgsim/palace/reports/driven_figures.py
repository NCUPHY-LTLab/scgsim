"""The figures of a Palace Driven run's physics view: its S-parameters and Surface EPR against frequency.

Each figure is drawn from the detached ``PalaceDrivenResult`` and reads no file.  |S| is drawn in two figures,
reflection (S11, S22, ...) apart from transmission (S21, S12, ...): a matched line reflects at about -40 dB and
transmits at about -0.001 dB, so on one axis the transmission would be a flat line at 0 dB.  The phase of every
entry shares one figure, in degrees as the solver printed it.  Surface EPR is one figure per interface type, MA
(metal-air), MS (metal-substrate) and SA (substrate-air), since their participations differ by orders of
magnitude; with two excited ports or more, each port is one group of lines, the whole surfaces and each inset.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

from scgsim.palace.reports.plotting import (
    COLORWAY,
    FigureStyle,
    ReportTheme,
    axis_title,
    plotly,
    theme_tokens,
)
from scgsim.palace.reports.surface_figures import SURFACE_TYPES
from scgsim.palace.results.driven_records import DrivenSurfaceEpr, PalaceDrivenResult
from scgsim.palace.results.inspection_records import SurfaceMarginEprRecord
from scgsim.semantics.s_parameters import SParameters

# The line styles in the order they are given out.  An S-matrix entry takes its excited port's style: S12
# (port 2 excited) is dashed over S21, so the two stay visible where a reciprocal network draws them on top
# of each other.  An inset takes the style after the whole surfaces', in its port's colour.
_DASHES = ("solid", "dash", "dot", "dashdot", "longdash", "longdashdot")
# Markers show where a short sweep was evaluated; on a long one they would hide the line.
_MARKED_POINTS = 21


class _Line(NamedTuple):
    """One line of a frequency figure: its legend label, its values at each frequency and how it is drawn.

    ``group`` is the legend group the line belongs to, its title shown once over the group's lines; None
    leaves the line ungrouped.
    """

    label: str
    frequencies_ghz: Sequence[float]
    values: Sequence[float | None]
    color: str
    dash: str = "solid"
    group: str | None = None


def s_parameter_figures(result: SParameters, theme: ReportTheme) -> list[Any]:
    """Reflection, transmission (from two ports on) and phase against frequency, one line per S-matrix entry.

    The entries are named as the record numbers its ports: ``S21`` is output port 2 with port 1 excited.
    """
    count = len(result.ports)
    frequencies = [frequency / 1e9 for frequency in result.frequencies_hz]
    # Column by column, as Touchstone orders a two-port line: S11, S21, S12, S22.
    entries = [
        (output, excited)
        for excited in range(1, count + 1)
        for output in range(1, count + 1)
    ]
    reflection = [entry for entry in entries if entry[0] == entry[1]]
    transmission = [entry for entry in entries if entry[0] != entry[1]]
    groups = [
        ("Reflection", "Magnitude (dB)", " dB", result.magnitude_db, reflection),
        ("Transmission", "Magnitude (dB)", " dB", result.magnitude_db, transmission),
        ("Phase", "Phase (degrees)", "°", result.phase_deg, entries),
    ]
    colors = {
        entry: COLORWAY[index % len(COLORWAY)] for index, entry in enumerate(entries)
    }
    return [
        _frequency_figure(
            title,
            ylabel,
            [
                _Line(
                    entry_name(output, excited, count),
                    frequencies,
                    [matrix[output - 1][excited - 1] for matrix in matrices],
                    colors[(output, excited)],
                    _DASHES[(excited - 1) % len(_DASHES)],
                )
                for output, excited in members
            ],
            theme,
            unit=unit,
        )
        for title, ylabel, unit, matrices, members in groups
        if members
    ]


def entry_name(output: int, excited: int, ports: int) -> str:
    """``S21`` for output port 2 with port 1 excited; ``S10,2`` once a port number has two digits."""
    separator = "," if ports >= 10 else ""
    return f"S{output}{separator}{excited}"


def surface_epr_figures(result: PalaceDrivenResult, theme: ReportTheme) -> list[Any]:
    """One figure per interface type the run's rows have: its participation against frequency.

    A line is the type's whole surfaces, the sum of its whole-surface rows, or one inset: the sum of its
    margin rows at that margin, the part of the surfaces that far or farther from the gap edge.  Margin rows
    are alternatives to the whole-surface rows, so no line adds them up.
    """
    if not result.surface_epr:
        return []
    ports = sorted({record.excited_port for record in result.surface_epr})
    margins = sorted(
        {row.margin_um for record in result.surface_epr for row in record.margins}
    )
    figures: list[Any] = []
    for interface_type in SURFACE_TYPES:
        if not any(
            row.interface_type == interface_type
            for record in result.surface_epr
            for row in record.surfaces
        ):
            continue
        lines = [
            line
            for index, port in enumerate(ports)
            for line in _port_lines(
                [
                    record
                    for record in result.surface_epr
                    if record.excited_port == port
                ],
                interface_type,
                margins,
                color=COLORWAY[index % len(COLORWAY)],
                group=f"Port {port} driven" if len(ports) > 1 else None,
            )
        ]
        figure = _frequency_figure(
            f"{interface_type} participation", "Participation", lines, theme
        )
        # A participation is a fraction near 1e-6 to 1e-4: scientific ticks, not SI prefixes that read as units,
        # and a left margin wide enough for them beside the axis title.
        figure.update_yaxes(exponentformat="e")
        figure.update_layout(margin={"l": 116})
        figures.append(figure)
    return figures


def _port_lines(
    records: Sequence[DrivenSurfaceEpr],
    interface_type: str,
    margins: Sequence[float],
    *,
    color: str,
    group: str | None,
) -> list[_Line]:
    """The lines of one excited port: its whole surfaces of ``interface_type``, then each inset it records."""
    frequencies = [record.frequency_hz / 1e9 for record in records]
    whole = _Line(
        "whole surface",
        frequencies,
        [
            sum(
                row.participation
                for row in record.surfaces
                if row.interface_type == interface_type
            )
            for record in records
        ],
        color,
        group=group,
    )
    lines = [whole]
    # Each margin keeps its dash in every port's group, so one inset reads the same across the figure.
    for index, margin in enumerate(margins):
        values = [
            _inset_total(record.margins, interface_type, margin) for record in records
        ]
        if any(value is not None for value in values):
            lines.append(
                _Line(
                    f"inset {margin:g} µm",
                    frequencies,
                    values,
                    color,
                    _DASHES[(index + 1) % len(_DASHES)],
                    group,
                )
            )
    return lines


def _inset_total(
    rows: Sequence[SurfaceMarginEprRecord], interface_type: str, margin_um: float
) -> float | None:
    """The participation of one type's margin rows at one margin; None when the record has no such row."""
    values = [
        row.participation
        for row in rows
        if row.interface_type == interface_type and row.margin_um == margin_um
    ]
    return sum(values) if values else None


def _frequency_figure(
    title: str,
    ylabel: str,
    lines: Sequence[_Line],
    theme: ReportTheme,
    *,
    unit: str = "",
) -> Any:
    """``lines`` against frequency in GHz, the legend above the plot and below the title.

    ``unit`` follows each value the hover shows; the hover names a grouped line with its group.
    """
    tokens = theme_tokens(theme)
    go = plotly()
    figure = go.Figure()
    # The title sits above a legend of one or two rows, as in the convergence figures; one line needs no legend.
    legend = len(lines) > 1
    FigureStyle(
        title=title,
        margin={"l": 84, "r": 36, "t": 116 if legend else 72, "b": 64},
        showlegend=legend,
        theme=theme,
    ).apply(figure)
    figure.update_layout(
        legend={
            "orientation": "h",
            "x": 0,
            "xanchor": "left",
            "y": 1.04,
            "yanchor": "bottom",
        },
    )
    for line in lines:
        name = line.label if line.group is None else f"{line.group}, {line.label}"
        trace: dict[str, Any] = {
            "x": list(line.frequencies_ghz),
            "y": list(line.values),
            "mode": "lines+markers"
            if len(line.frequencies_ghz) <= _MARKED_POINTS
            else "lines",
            "name": line.label,
            "line": {"width": 2.5, "color": line.color, "dash": line.dash},
            "marker": {"size": 7, "color": line.color},
            "hovertemplate": f"%{{y:.6g}}{unit}<extra>{name}</extra>",
        }
        if line.group is not None:
            trace["legendgroup"] = line.group
            trace["legendgrouptitle"] = {"text": line.group}
        figure.add_trace(go.Scatter(**trace))
    figure.update_xaxes(title=axis_title("Frequency (GHz)", tokens))
    figure.update_yaxes(title=axis_title(ylabel, tokens))
    return figure
