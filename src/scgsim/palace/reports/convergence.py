"""The declared convergence as the trust report shows it: the verdict, the table of passes, the change figures.

Everything here is read from a ``ConvergenceCheck`` (``scgsim.contracts.palace_convergence``); nothing is
computed again.  ``verdict_sentence`` says what a reader can act on, ``pass_table_html`` shows every criterion
at every pass by its worst number, and ``change_figures`` plots what each criterion measured against the
threshold the user declared.
"""

from __future__ import annotations

import html
from typing import Any

from scgsim.contracts.palace_convergence import (
    ConvergenceCheck,
    CriterionAtPass,
    PlannedCriterion,
    ValueAtPass,
)
from scgsim.palace.reports.plotting import (
    LinePlot,
    ReferenceLine,
    ReportTheme,
    Trace,
    html_cell,
)


def declared_items(check: ConvergenceCheck, theme: ReportTheme) -> list[Any]:
    """The declared convergence as the trust report opens its numerical evidence; nothing when undeclared.

    The verdict comes first, then the table of passes, then each criterion's figure.
    """
    if check.plan is None:
        return []
    return [
        "<section><h4>Declared convergence</h4>"
        f"<p><strong>{html.escape(verdict_sentence(check))}</strong></p>"
        f"{pass_table_html(check)}</section>",
        *change_figures(check, theme),
    ]


def declared_threshold(check: ConvergenceCheck, quantity: str) -> float | None:
    """The threshold declared for ``quantity`` (``error_indicator``), None when no criterion is on it."""
    if check.plan is None:
        return None
    return next(
        (item.threshold for item in check.plan.criteria if item.quantity == quantity),
        None,
    )


def verdict_card(check: ConvergenceCheck) -> str:
    """The verdict in a few words, as the run-identity card shows it."""
    verdict = check.verdict
    if verdict.state == "converged":
        return f"converged at pass {verdict.pass_number}"
    if verdict.state == "not_converged":
        return f"not converged in {verdict.passes_run} passes"
    return "not declared"


def verdict_sentence(check: ConvergenceCheck) -> str:
    """The verdict as a sentence: at which pass, which criterion held last, or what still fails and by how much."""
    verdict = check.verdict
    plan = check.plan
    if plan is None:
        return "Convergence is not declared for this run."
    in_a_row = plan.minimum_converged_passes
    if verdict.state == "converged" and verdict.pass_number is not None:
        first = verdict.pass_number - in_a_row + 1
        held = (
            f"at pass {first}"
            if in_a_row == 1
            else f"at passes {first} to {verdict.pass_number}, {in_a_row} in a row"
        )
        after = verdict.passes_after_convergence
        ran_after = (
            "No pass ran after it."
            if not after
            else f"{after} more pass{'es' if after > 1 else ''} ran after it, up to amr_max_passes."
        )
        return (
            f"Converged at pass {verdict.pass_number} of the {verdict.passes_run} passes run: every criterion "
            f"held {held}. Last to hold: {', '.join(verdict.last_to_hold)}, from pass "
            f"{verdict.last_to_hold_from}. {ran_after}"
        )
    if not check.passes:
        return "Not converged: the run folder holds no pass to check."
    last = check.passes[-1]
    if not verdict.failing:
        return (
            f"Not converged within the {verdict.passes_run} passes run: every criterion holds at pass "
            f"{last.number}, for {last.streak} of the {in_a_row} passes in a row the declaration needs."
        )
    failing = "; ".join(_failure(item) for item in verdict.failing)
    return f"Not converged within the {verdict.passes_run} passes run. At pass {last.number}: {failing}."


def _failure(item: CriterionAtPass) -> str:
    """One failing criterion: its worst number, what it measured, and the threshold."""
    worst = item.worst
    criterion = item.criterion
    named = (
        item.criterion.label
        if len(item.values) == 1 and worst.label == criterion.label
        else f"{criterion.label} ({worst.label})"
    )
    if worst.measured is None:
        return f"{named}: unavailable, {worst.unavailable}"
    times = worst.measured / criterion.threshold
    return (
        f"{named}: {_measured_words(criterion)} {_number(worst.measured, criterion)}, "
        f"{times:.3g} times the declared {_threshold_words(criterion)}"
    )


def pass_table_html(check: ConvergenceCheck) -> str:
    """One row per criterion, one column per pass: whether it holds, by its worst number."""
    if check.plan is None or not check.passes:
        return ""
    header = (
        "<tr>"
        + html_cell("th", "Criterion")
        + html_cell("th", "Declared")
        + "".join(html_cell("th", f"Pass {item.number}") for item in check.passes)
        + "</tr>"
    )
    rows = [
        "<tr>"
        + html_cell("td", criterion.label)
        + html_cell("td", _threshold_words(criterion))
        + "".join(_cell(check_pass.criteria[index]) for check_pass in check.passes)
        + "</tr>"
        for index, criterion in enumerate(check.plan.criteria)
    ]
    rows.append(
        "<tr>"
        + html_cell("td", "Every criterion")
        + html_cell("td", f"{check.plan.minimum_converged_passes} pass(es) in a row")
        + "".join(
            html_cell(
                "td",
                f"holds, {item.streak} in a row" if item.holds else "fails",
            )
            for item in check.passes
        )
        + "</tr>"
    )
    return (
        "<table style='border-collapse:collapse;font-size:0.9rem'>"
        f"<thead>{header}</thead><tbody>{''.join(rows)}</tbody></table>"
        "<p style='opacity:0.75'>Each cell is the criterion's worst number at that pass: what it measured "
        "and, for a group, which number. A change criterion cannot hold at pass 1, which has no pass "
        "before it.</p>"
    )


def _cell(item: CriterionAtPass) -> str:
    worst = item.worst
    mark = "holds" if item.holds else "fails"
    measured = (
        "unavailable"
        if worst.measured is None
        else _number(worst.measured, item.criterion)
    )
    which = "" if len(item.values) == 1 else f" ({worst.label})"
    title = html.escape(worst.unavailable or "", quote=True)
    cell = html_cell("td", f"{mark}: {measured}{which}")
    return cell.replace("<td ", f"<td title='{title}' ", 1) if title else cell


def change_figures(check: ConvergenceCheck, theme: ReportTheme) -> list[Any]:
    """Per criterion, what it measured at each pass on a log axis, with the declared threshold as the line."""
    if check.plan is None or not check.passes:
        return []
    xs = [item.number for item in check.passes]
    figures: list[Any] = []
    for index, criterion in enumerate(check.plan.criteria):
        at_passes = [item.criteria[index] for item in check.passes]
        traces = [
            Trace(
                planned.label,
                [_measured(at.values[value_index]) for at in at_passes],
            )
            for value_index, planned in enumerate(criterion.values)
        ]
        figure = LinePlot(
            title=f"{criterion.label}: {_measured_words(criterion)} vs pass",
            xlabel="pass",
            ylabel=_axis_words(criterion),
            xs=xs,
            traces=traces,
            yaxis_type="log",
            hline=ReferenceLine(criterion.threshold, "declared threshold"),
            theme=theme,
        ).figure()
        if figure is not None:
            figures.append(figure)
    return figures


def _measured(value: ValueAtPass) -> float | None:
    return value.measured


def _measured_words(criterion: PlannedCriterion) -> str:
    if criterion.test == "relative_change":
        return "relative change"
    if criterion.test == "absolute_change":
        return "change"
    return "value"


def _axis_words(criterion: PlannedCriterion) -> str:
    if criterion.test == "relative_change":
        return "|q_k - q_(k-1)| / |q_k|"
    unit = "" if criterion.unit == "1" else f" ({criterion.unit})"
    if criterion.test == "absolute_change":
        return f"|q_k - q_(k-1)|{unit}"
    return f"value{unit}"


def _threshold_words(criterion: PlannedCriterion) -> str:
    """The declared threshold as a reader says it: ``relative change at most 0.001``."""
    return f"{_measured_words(criterion)} at most {_number(criterion.threshold, criterion)}"


def _number(value: float, criterion: PlannedCriterion) -> str:
    """A measured number or threshold, in the criterion's unit unless it is relative or unitless."""
    unit = (
        ""
        if criterion.test == "relative_change" or criterion.unit == "1"
        else f" {criterion.unit}"
    )
    return f"{value:.3g}{unit}"


__all__ = [
    "change_figures",
    "declared_items",
    "declared_threshold",
    "pass_table_html",
    "verdict_card",
    "verdict_sentence",
]
