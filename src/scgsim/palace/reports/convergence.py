"""The declared convergence as the trust report shows it: the verdict, the table of passes, the change figures.

Everything here is read from a ``ConvergenceCheck`` (``scgsim.contracts.palace_convergence``) and the
``SolverEnd`` of the run; nothing is computed again.  ``verdict_sentence`` says what a reader can act on: the
verdict, then how Palace ended (stopped by SCGSim after a pass, with the passes of the limit not run, or at its
own end).  ``pass_table_html`` shows every criterion at every pass by its worst number, and ``change_figures``
plots what each criterion measured against the threshold the user declared.
The numbers are shown as the AEDT side of SCGSim names them, so a reader compares the two solvers in one
vocabulary: a change is a percentage of the value, called ``Delta %`` for a capacitance (Q3D's matrix delta),
``Max Delta Freq. %`` for a mode frequency (HFSS Eigenmode) and ``Max Percent Delta`` for a participation
(HFSS's expression convergence); the table has a row per pass with its solved elements, like AEDT's.  The
declaration, the plan and the check use the same percentage.
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from typing import Any

from scgsim.contracts.palace_convergence import (
    ConvergenceCheck,
    CriterionAtPass,
    PassCheck,
    PlannedCriterion,
)
from scgsim.palace.reports.plotting import (
    LinePlot,
    ReferenceLine,
    ReportTheme,
    Trace,
    fmt,
    html_cell,
)
from scgsim.palace.results.convergence import SolverEnd
from scgsim.palace.results.inspection_records import AmrPassSnapshot


def declared_items(
    check: ConvergenceCheck,
    theme: ReportTheme,
    end: SolverEnd | None = None,
    *,
    solved_elements: Mapping[int, int] | None = None,
) -> list[Any]:
    """The declared convergence as the trust report opens its numerical evidence; nothing when undeclared.

    The verdict comes first, then the table of passes, then each criterion's figure.  ``solved_elements``
    gives the mesh elements of each pass by its number, for the table.
    """
    if check.plan is None:
        return []
    return [
        "<section><h4>Declared convergence</h4>"
        f"<p><strong>{html.escape(verdict_sentence(check, end))}</strong></p>"
        f"{pass_table_html(check, solved_elements)}</section>",
        *change_figures(check, theme),
    ]


def solved_elements(passes: Sequence[AmrPassSnapshot]) -> dict[int, int]:
    """The mesh elements of each pass read, by the pass's number, for the table's ``Solved elements``."""
    return {
        item.pass_number: item.mesh_elements
        for item in passes
        if item.mesh_elements is not None
    }


def pass_limit_text(amr_max_passes: int | None) -> str:
    """The configured ``MaxIts`` with the passes it allows: N refinements are N + 1 solves."""
    if amr_max_passes is None:
        return fmt(amr_max_passes)
    allowed = amr_max_passes + 1
    return f"{amr_max_passes} ({allowed} pass{'' if allowed == 1 else 'es'})"


def declared_threshold(check: ConvergenceCheck, quantity: str) -> float | None:
    """The threshold declared for ``quantity`` (``error_indicator``), None when no criterion is on it."""
    if check.plan is None:
        return None
    return next(
        (item.threshold for item in check.plan.criteria if item.quantity == quantity),
        None,
    )


def verdict_card(check: ConvergenceCheck, end: SolverEnd | None = None) -> str:
    """The verdict in a few words, as the run-identity card shows it."""
    verdict = check.verdict
    if verdict.state == "converged":
        stopped = _stopped_after(end)
        return f"converged at pass {verdict.pass_number}" + (
            "" if stopped is None else f", stopped after pass {stopped}"
        )
    if verdict.state == "not_converged":
        return f"not converged in {_passes(verdict.passes_read)} read"
    return "not declared"


def verdict_sentence(check: ConvergenceCheck, end: SolverEnd | None = None) -> str:
    """The verdict as a sentence: at which pass, which criterion held last, or what still fails and by how much.

    It speaks of the passes read: every pass of a returned run, only some of a partly returned one.  A pass
    is named by Palace's number, counted from the solve on the input mesh.  With the run's ``end`` it then
    says how Palace ended (``ending_sentence``).
    """
    return _verdict_words(check) + ending_sentence(check, end)


def ending_sentence(check: ConvergenceCheck, end: SolverEnd | None) -> str:
    """How Palace ended, after the verdict: stopped after a pass, with the passes of the limit not run, or by itself.

    Empty without a receipt to say it, or for a run that declared nothing.
    """
    if end is None or end.ending is None or check.plan is None:
        return ""
    last = check.passes[-1].number if check.passes else None
    allowed = end.passes_allowed
    stopped = _stopped_after(end)
    if stopped is not None:
        not_run = None if allowed is None or last is None else allowed - last
        return f" SCGSim stopped Palace after pass {stopped}" + (
            "."
            if not not_run or allowed is None
            else f": {not_run} of the {_passes(allowed)} its limit allows "
            f"{'was' if not_run == 1 else 'were'} not run."
        )
    if end.ending == "solver_ended":
        if allowed is not None and last == allowed:
            return f" Palace ended by itself at its limit of {_passes(allowed)}."
        return f" Palace ended by itself after pass {last}."
    return f" The solver's ending: {end.ending}."


def _stopped_after(end: SolverEnd | None) -> int | None:
    """The pass after which SCGSim stopped Palace at convergence, None for any other ending."""
    if end is None or end.ending != "stopped_at_convergence":
        return None
    return end.stopped_after


def _verdict_words(check: ConvergenceCheck) -> str:
    """The verdict part of ``verdict_sentence``."""
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
            f"Last to hold: {'; '.join(verdict.last_to_hold)}, from pass {verdict.last_to_hold_from}. "
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
    """What the passes read leave out: passes not returned, unreadable passes, and folders set aside.

    A pass that was not read could have converged the run earlier, so the verdict says so; a pass two
    folders record with different tables is in no comparison, and a folder that is no pass (its number not
    established, or a solve under way) is named with why.
    """
    read_numbers = {item.number for item in check.passes}
    last = converged_at if converged_at is not None else max(read_numbers, default=0)
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
                f" Pass {item.number} is in no comparison: {item.identity_problem}."
            )
    for folder in check.set_aside:
        notes += f" Not read as a pass: {folder.reason}."
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
        else f"{criterion.label} ({_within(worst.label, criterion.label)})"
    )
    if worst.measured is None:
        return f"{named}: not measured, {worst.unavailable}"
    times = worst.measured / criterion.threshold
    return (
        f"{named}: {_measured_words(criterion)} {_number(worst.measured, criterion)}, "
        f"above the declared {_number(criterion.threshold, criterion)} ({times:.3g} times it)"
    )


def _within(value_label: str, criterion_label: str) -> str:
    """A number's label without the criterion's label it starts with: ``excitation signal``."""
    return value_label.removeprefix(f"{criterion_label}, ")


def pass_table_html(
    check: ConvergenceCheck, solved_elements: Mapping[int, int] | None = None
) -> str:
    """One row per pass, as AEDT's convergence table has it: its solved elements and each criterion's worst number.

    A column is a criterion, headed by its label and what was declared; the last column says whether every
    criterion held and for how many passes in a row.
    """
    if check.plan is None or not check.passes:
        return ""
    elements = solved_elements or {}
    header = (
        "<tr>"
        + html_cell("th", "Pass")
        + html_cell("th", "Solved elements")
        + "".join(_criterion_heading(criterion) for criterion in check.plan.criteria)
        + _heading("Every criterion", _in_a_row(check.plan.minimum_converged_passes))
        + "</tr>"
    )
    rows = [
        "<tr>"
        + html_cell(
            "td", f"{item.number}" + ("" if item.identity_problem is None else "?")
        )
        + html_cell(
            "td", f"{elements[item.number]:,}" if item.number in elements else ""
        )
        + "".join(_cell(at_pass) for at_pass in item.criteria)
        + html_cell("td", _every_criterion(item))
        + "</tr>"
        for item in check.passes
    ]
    return (
        "<table style='border-collapse:collapse;font-size:0.9rem'>"
        f"<thead>{header}</thead><tbody>{''.join(rows)}</tbody></table>"
        "<p style='opacity:0.75'>Each cell is the criterion's worst number at that pass and, for a group, "
        "which number; a change is a percentage of the value. N/A: a change needs the pass before it, so "
        "pass 1 has none.</p>"
    )


def _heading(label: str, declared: str) -> str:
    """A column heading in two lines: what the column is, and what was declared for it."""
    return (
        "<th style='border:1px solid var(--border,#d0d7de);padding:0.35rem 0.65rem;text-align:left'>"
        f"{html.escape(label)}<br><span style='font-weight:normal;opacity:0.75'>"
        f"{html.escape(declared)}</span></th>"
    )


def _criterion_heading(criterion: PlannedCriterion) -> str:
    return _heading(criterion.label, _threshold_words(criterion))


def _every_criterion(item: PassCheck) -> str:
    """Whether every criterion held at the pass, and for how many passes in a row."""
    if item.identity_problem is not None:
        return "not counted: its folders differ"
    if item.holds:
        return f"holds, {item.streak} in a row"
    if any(value.measured is None for at in item.criteria for value in at.values):
        return "not yet"
    return "fails"


def _cell(item: CriterionAtPass) -> str:
    """The criterion's worst number at a pass and whether it holds; why it is missing when it is."""
    worst = item.worst
    which = (
        ""
        if len(item.values) == 1
        else f" ({_within(worst.label, item.criterion.label)})"
    )
    if worst.measured is None:
        if worst.unavailable == _NO_PASS_BEFORE:
            return html_cell("td", "N/A")
        # One reason for every number of the criterion is said once.
        shared = len({value.unavailable for value in item.values}) == 1
        return html_cell(
            "td", f"not measured: {worst.unavailable}{'' if shared else which}"
        )
    mark = "✓" if item.holds else "✗"
    return html_cell("td", f"{_number(worst.measured, item.criterion)} {mark}{which}")


def change_figures(check: ConvergenceCheck, theme: ReportTheme) -> list[Any]:
    """What the criteria measured at each pass on a log axis, with the declared threshold as the line.

    Criteria that measure one quantity the same way against one threshold share a figure, so the three kinds
    of one ``SurfaceEpr.each(...).within(...)`` are three lines under their threshold, not three figures.
    """
    if check.plan is None or not check.passes:
        return []
    criteria = check.plan.criteria
    xs = [item.number for item in check.passes]
    figures: list[Any] = []
    for group in _sharing_a_figure(criteria):
        first = criteria[group[0]]
        traces = [
            Trace(
                planned.label,
                [
                    item.criteria[index].values[position].measured
                    for item in check.passes
                ],
            )
            for index in group
            for position, planned in enumerate(criteria[index].values)
        ]
        label = first.label if len(group) == 1 else _QUANTITY_NAMES[first.quantity]
        figure = LinePlot(
            title=f"{label}: {_measured_words(first)} vs AMR pass",
            xlabel="AMR pass",
            ylabel=_axis_words(first),
            xs=xs,
            traces=traces,
            yaxis_type="log",
            hline=ReferenceLine(
                first.threshold, f"declared: at most {_number(first.threshold, first)}"
            ),
            theme=theme,
        ).figure()
        if figure is not None:
            figures.append(figure)
    return figures


def _sharing_a_figure(criteria: Sequence[PlannedCriterion]) -> list[list[int]]:
    """The criteria's positions, grouped by what they measure and against which threshold, in declared order."""
    groups: dict[tuple[str, str, float, str], list[int]] = {}
    for index, criterion in enumerate(criteria):
        key = (criterion.quantity, criterion.test, criterion.threshold, criterion.unit)
        groups.setdefault(key, []).append(index)
    return list(groups.values())


_QUANTITY_NAMES = {
    "capacitance": "Capacitance",
    "mode_frequency": "Mode frequency",
    "surface_epr": "Surface participation",
    "error_indicator": "Error indicator",
}
"""What a figure of several criteria on one quantity is called."""


def _in_a_row(passes: int) -> str:
    return "1 pass in a row" if passes == 1 else f"{passes} passes in a row"


_NO_PASS_BEFORE = "no pass before it"
"""The contract's reason for a change that pass 1 cannot have."""

_DELTA_NAMES = {
    "capacitance": "Delta %",
    "mode_frequency": "Max Delta Freq. %",
    "surface_epr": "Max Percent Delta",
}
"""A relative change by the name the AEDT side gives the same measure of that quantity."""


def _measured_words(criterion: PlannedCriterion) -> str:
    if criterion.test == "percent_change":
        return _DELTA_NAMES.get(criterion.quantity, "Delta %")
    if criterion.test == "absolute_change":
        return "Max Delta"
    return "value"


def _axis_words(criterion: PlannedCriterion) -> str:
    """What the figure's vertical axis is, in the words of the table."""
    if criterion.test == "percent_change":
        return "change from the pass before (%)"
    unit = "" if criterion.unit == "1" else f" ({criterion.unit})"
    if criterion.test == "absolute_change":
        return f"change from the pass before{unit}"
    return f"value{unit}"


def _threshold_words(criterion: PlannedCriterion) -> str:
    """The declared threshold as a reader says it: ``Delta % at most 0.1``."""
    return f"{_measured_words(criterion)} at most {_number(criterion.threshold, criterion)}"


def _number(value: float, criterion: PlannedCriterion) -> str:
    """A measured number or threshold as shown, in the criterion's unit unless it is relative or unitless."""
    unit = (
        ""
        if criterion.test == "percent_change" or criterion.unit == "1"
        else f" {criterion.unit}"
    )
    return f"{value:.3g}{unit}"


__all__ = [
    "change_figures",
    "declared_items",
    "declared_threshold",
    "pass_limit_text",
    "pass_table_html",
    "solved_elements",
    "verdict_card",
    "verdict_sentence",
]
