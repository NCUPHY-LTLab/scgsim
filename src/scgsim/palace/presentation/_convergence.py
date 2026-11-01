"""Numerical-evidence figures: one scalar per figure against the AMR pass number.

Owns: choosing which quantities of an AMR (adaptive mesh refinement) run get a trend
figure and how each is drawn.  Does not own: displaying the figures (``reports.py``),
the single-pass status card, or surface-EPR convergence.

Each figure is one readable scalar, so a reader can see which quantity moved.  Items
are Plotly figures, or an HTML string where a figure cannot be drawn and the reader
needs to know why.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ..results.report_data import (
    _ERROR_INDICATOR_TRACES,
    _FREQ_HEADER,
    _SKIP_EIG_COLUMNS,
    AmrPassSnapshot,
)
from ._figures import (
    ReportTheme,
    _capacitance_traces,
    _delta_traces,
    _line_figure,
    _mode_traces,
    _traces_are_finite,
    _yaxis_type,
)
from ._html import _native_positive_infinity_q_notice


def convergence_trend_items(
    passes: Sequence[AmrPassSnapshot],
    *,
    amr_tolerance: float | None,
    theme: ReportTheme,
) -> list[Any]:
    """Trend figures for two or more AMR passes, in display order."""
    xs = tuple(pass_.pass_index for pass_ in passes)
    items: list[Any] = []
    items.extend(_eigenvalue_items(passes, xs, theme))
    items.extend(_port_epr_items(passes, xs, theme))
    items.extend(_capacitance_items(passes, xs, theme))
    error_figure = _error_indicator_figure(passes, xs, amr_tolerance, theme)
    if error_figure is not None:
        items.append(error_figure)
    return items


def single_pass_q_notice(passes: Sequence[AmrPassSnapshot]) -> str | None:
    """Explain a Q of positive infinity when only one pass exists to plot."""
    q_traces = _mode_traces(passes, _column_reader("eig_columns", "Q"))
    return _native_positive_infinity_q_notice(q_traces)


def _eigenvalue_items(
    passes: Sequence[AmrPassSnapshot], xs: tuple[int, ...], theme: ReportTheme
) -> list[Any]:
    items: list[Any] = []
    for header in _union_mapping_keys(passes, "eig_columns"):
        if header in _SKIP_EIG_COLUMNS:
            continue
        traces = _mode_traces(passes, _column_reader("eig_columns", header))
        if header == "Q" and not _traces_are_finite(traces):
            items.append(
                _native_positive_infinity_q_notice(traces)
                or "<p style='opacity:0.75'>A finite Q convergence plot is "
                "unavailable because the Q trace contains non-finite or "
                "unavailable data.</p>"
            )
            continue
        figure = _line_figure(
            title=f"{header} vs AMR pass",
            xlabel="AMR pass",
            ylabel=header,
            xs=xs,
            traces=traces,
            yaxis_type=_yaxis_type(header),
            theme=theme,
        )
        if figure is not None:
            items.append(figure)
        if header == _FREQ_HEADER:
            items.extend(
                _delta_items(f"|Δ {header}|", f"|Δ {header}|", traces, xs, theme)
            )
    return items


def _port_epr_items(
    passes: Sequence[AmrPassSnapshot], xs: tuple[int, ...], theme: ReportTheme
) -> list[Any]:
    items: list[Any] = []
    for header in _union_mapping_keys(passes, "port_epr"):
        traces = _mode_traces(passes, _column_reader("port_epr", header))
        figure = _line_figure(
            title=f"Port EPR {header} vs AMR pass",
            xlabel="AMR pass",
            ylabel=header,
            xs=xs,
            traces=traces,
            theme=theme,
        )
        if figure is not None:
            items.append(figure)
    return items


def _capacitance_items(
    passes: Sequence[AmrPassSnapshot], xs: tuple[int, ...], theme: ReportTheme
) -> list[Any]:
    items: list[Any] = []
    for label, traces in _capacitance_traces(passes):
        figure = _line_figure(
            title=f"{label} vs AMR pass",
            xlabel="AMR pass",
            ylabel="fF",
            xs=xs,
            traces=traces,
            theme=theme,
        )
        if figure is not None:
            items.append(figure)
        if label == "Maxwell C_ij":
            continue
        items.extend(_delta_items(f"|Δ {label}|", "|Δ| (fF)", traces, xs, theme))
    return items


def _delta_items(
    title_quantity: str,
    ylabel: str,
    traces: Any,
    xs: tuple[int, ...],
    theme: ReportTheme,
) -> list[Any]:
    """Pass-to-pass change on a log axis, so convergence shows as a falling line."""
    figure = _line_figure(
        title=f"{title_quantity} vs AMR pass",
        xlabel="AMR pass",
        ylabel=ylabel,
        xs=xs[1:],
        traces=_delta_traces(traces),
        yaxis_type="log",
        theme=theme,
    )
    return [] if figure is None else [figure]


def _error_indicator_figure(
    passes: Sequence[AmrPassSnapshot],
    xs: tuple[int, ...],
    amr_tolerance: float | None,
    theme: ReportTheme,
) -> Any | None:
    """Norm, Maximum and Mean share one log plot because they are the same indicator."""
    traces = [
        (
            name,
            [
                None
                if pass_.error_indicators is None
                else pass_.error_indicators.get(name)
                for pass_ in passes
            ],
        )
        for name in _ERROR_INDICATOR_TRACES
    ]
    hline = None
    if amr_tolerance is not None:
        hline = (amr_tolerance, "configured AMR Tol")
    return _line_figure(
        title="AMR error indicator vs AMR pass",
        xlabel="AMR pass",
        ylabel="error indicator",
        xs=xs,
        traces=traces,
        yaxis_type="log",
        hline=hline,
        theme=theme,
    )


def _column_reader(
    attribute: str, column: str
) -> Callable[[AmrPassSnapshot], Any | None]:
    """Reader for one named column of a pass's ``eig_columns`` or ``port_epr``."""

    def read(pass_: AmrPassSnapshot) -> Any | None:
        mapping = getattr(pass_, attribute)
        return None if mapping is None else mapping.get(column)

    return read


def _union_mapping_keys(
    passes: Sequence[AmrPassSnapshot], attribute: str
) -> tuple[str, ...]:
    """Keys of ``attribute`` across all passes, in first-seen order."""
    keys: list[str] = []
    seen: set[str] = set()
    for pass_ in passes:
        mapping = getattr(pass_, attribute)
        if not isinstance(mapping, dict):
            continue
        for key in mapping:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    return tuple(keys)
