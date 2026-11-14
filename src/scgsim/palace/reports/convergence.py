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
        return f"not converged in {_passes(verdict.passes_read)} read"
    return "not declared"


def verdict_sentence(check: ConvergenceCheck) -> str:
    """The verdict as a sentence: at which pass, which criterion held last, or what still fails and by how much.

    It speaks of the passes read: every pass of a returned run, only some of a partly returned one.  A pass
    is named by Palace's number, counted from the solve on the input mesh.
    """
    verdict = check.verdict
    plan = check.plan
    if plan is None:
        return "Convergence is not declared for this run."
    if not check.passes:
        return "Not converged: the run folder holds no pass to check."
    read = _passes_read(check)
    in_a_row = plan.minimum_converged_passes
    if verdict.state == "converged" and verdict.pass_number is not None:
        first = verdict.pass_number - in_a_row + 1
        held = (
            f"at pass {first}"
            if in_a_row == 1
            else f"at passes {first} to {verdict.pass_number}, {in_a_row} in a row"
        )
        after = verdict.passes_read_after or 0
        read_after = (
            "No pass was read after it."
            if not after
            else f"{after} more {'pass was' if after == 1 else 'passes were'} read after it."
        )
        return (
            f"Converged at pass {verdict.pass_number}, of the {read}: every criterion held {held}. "
            f"Last to hold: {', '.join(verdict.last_to_hold)}, from pass {verdict.last_to_hold_from}. "
            f"{read_after}{_reading_notes(check, verdict.pass_number)}"
        )
    last = check.passes[-1]
    notes = _reading_notes(check, None)
    if not verdict.failing:
        return (
            f"Not converged in the {read}: every criterion holds at pass {last.number}, for "
            f"{last.streak} of the {in_a_row} passes in a row the declaration needs.{notes}"
        )
    failing = "; ".join(_failure(item) for item in verdict.failing)
    return f"Not converged in the {read}. At pass {last.number}: {failing}.{notes}"


def _reading_notes(check: ConvergenceCheck, converged_at: int | None) -> str:
    """What the passes read leave out: passes not returned, and passes whose number is not established.

    A pass that was not read could have converged the run earlier, so the verdict says so; a pass whose
    number is not established is in no comparison, and the verdict says why.
    """
    established = {
        item.number for item in check.passes if item.identity_problem is None
    }
    last = converged_at if converged_at is not None else max(established, default=0)
    read_numbers = {item.number for item in check.passes}
    missing = [number for number in range(1, last) if number not in read_numbers]
    notes = ""
    if missing:
        names = _listed(missing)
        notes += (
            f" {'Pass' if len(missing) == 1 else 'Passes'} {names} "
            f"{'was' if len(missing) == 1 else 'were'} not read"
            + (
                ", so the run may have converged earlier."
                if converged_at is not None
                else "."
            )
        )
    for item in check.passes:
        if item.identity_problem is not None:
            notes += (
                f" {item.folder.name} is in no comparison: {item.identity_problem}."
            )
    return notes


def _listed(numbers: list[int]) -> str:
    """``2``, ``2 and 3`` or ``1, 2 and 3``."""
    if len(numbers) == 1:
        return str(numbers[0])
    return ", ".join(str(number) for number in numbers[:-1]) + f" and {numbers[-1]}"


def _passes_read(check: ConvergenceCheck) -> str:
    """``4 passes read (passes 1 to 4)``, or ``3 passes read (passes 1, 4 and 5)`` when some are missing."""
    numbers = [item.number for item in check.passes]
    if len(numbers) == 1:
        which = f"pass {numbers[0]}"
    elif numbers == list(range(numbers[0], numbers[0] + len(numbers))):
        which = f"passes {numbers[0]} to {numbers[-1]}"
    else:
        which = (
            "passes " + ", ".join(str(n) for n in numbers[:-1]) + f" and {numbers[-1]}"
        )
    return f"{_passes(len(numbers))} read ({which})"


def _passes(count: int) -> str:
    return f"{count} pass" if count == 1 else f"{count} passes"


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
        + "".join(
            html_cell(
                "th",
                f"Pass {item.number}" + ("" if item.identity_problem is None else "?"),
            )
            for item in check.passes
        )
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
        + html_cell("td", _in_a_row(check.plan.minimum_converged_passes))
        + "".join(
            html_cell(
                "td",
                "not counted: number not established"
                if item.identity_problem is not None
                else f"holds, {item.streak} in a row"
                if item.holds
                else "fails",
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
    """Whether the criterion holds, by its worst number; why that number is missing when it is."""
    worst = item.worst
    mark = "holds" if item.holds else "fails"
    if worst.measured is None:
        # Pass 1 of a change criterion has the same reason for every number: say it once.
        shared = len({value.unavailable for value in item.values}) == 1
        which = "" if shared or len(item.values) == 1 else f" ({worst.label})"
        return html_cell("td", f"{mark}: {worst.unavailable}{which}")
    which = "" if len(item.values) == 1 else f" ({worst.label})"
    return html_cell("td", f"{mark}: {_number(worst.measured, item.criterion)}{which}")


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
            title=f"{criterion.label}: {_measured_words(criterion)} vs AMR pass",
            xlabel="AMR pass",
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


def _in_a_row(passes: int) -> str:
    return "1 pass" if passes == 1 else f"{passes} passes in a row"


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
