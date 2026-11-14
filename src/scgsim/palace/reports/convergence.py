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
import itertools
from collections.abc import Mapping, Sequence
from typing import Any

from scgsim.contracts.palace_convergence import (
    ConvergenceCheck,
    CriterionAtPass,
    PassCheck,
    PlannedCriterion,
    PlannedValue,
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


def declared_html(
    check: ConvergenceCheck,
    end: SolverEnd | None = None,
    *,
    solved_elements: Mapping[int, int] | None = None,
    show_details: bool = False,
) -> list[str]:
    """The table of the declared criteria at each pass; nothing when undeclared.

    ``solved_elements`` gives the mesh elements of each pass by its number.  With ``show_details`` the
    verdict comes first as a sentence; the table alone already shows which pass every criterion held at.
    """
    if check.plan is None:
        return []
    sentence = (
        f"<p><strong>{html.escape(verdict_sentence(check, end))}</strong></p>"
        if show_details
        else ""
    )
    return [f"<section>{sentence}{pass_table_html(check, solved_elements)}</section>"]


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


def threshold_line(check: ConvergenceCheck, quantity: str) -> ReferenceLine | None:
    """The line of the threshold declared for ``quantity`` (``error_indicator``), None when none is."""
    if check.plan is None:
        return None
    return next(
        (
            ReferenceLine(item.threshold, _threshold_words(item))
            for item in check.plan.criteria
            if item.quantity == quantity
        ),
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
        criterion.name
        if len(item.values) == 1 and worst.label == criterion.name
        else f"{criterion.name} ({_within(worst.label, criterion.name)})"
    )
    if worst.measured is None:
        return f"{named}: not measured, {worst.unavailable}"
    times = worst.measured / criterion.threshold
    measured = "changed by" if criterion.measures_change else "value"
    return (
        f"{named}: {measured} {_number(worst.measured, criterion)}, "
        f"above the declared {_number(criterion.threshold, criterion)} ({times:.3g} times it)"
    )


def _within(value_label: str, criterion_name: str) -> str:
    """A number's label without the criterion's name it starts with: ``excitation signal``."""
    return value_label.removeprefix(f"{criterion_name}, ")


def pass_table_html(
    check: ConvergenceCheck, solved_elements: Mapping[int, int] | None = None
) -> str:
    """One row per pass, as AEDT's convergence table has it: its solved elements and each criterion's number.

    A column is a criterion, headed by its name and its threshold, so each number below is read against the
    threshold above it.  Criteria of one group stand under the group's name, said once.  The last column
    says whether every criterion held and for how many passes in a row.
    """
    if check.plan is None or not check.passes:
        return ""
    criteria = check.plan.criteria
    elements = solved_elements or {}
    grouped = any(criterion.group for criterion in criteria)
    rows_tall = 2 if grouped else 1
    top, below = _criteria_headings(criteria, rows_tall)
    header = (
        "<tr>"
        + _heading("Pass", rows=rows_tall)
        + _heading("Solved elements", rows=rows_tall)
        + top
        + _heading(
            "Every criterion",
            _in_a_row(check.plan.minimum_converged_passes),
            rows=rows_tall,
        )
        + "</tr>"
        + (f"<tr>{below}</tr>" if grouped else "")
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
    worst = (
        " A criterion that reads several numbers shows its worst one."
        if any(len(criterion.values) > 1 for criterion in criteria)
        else ""
    )
    return (
        "<table style='border-collapse:collapse;font-size:0.9rem'>"
        f"<thead>{header}</thead><tbody>{''.join(rows)}</tbody></table>"
        "<p style='opacity:0.75'>A change is measured from the pass before, so pass 1 has none (N/A)."
        f"{worst}</p>"
    )


def _criteria_headings(
    criteria: Sequence[PlannedCriterion], rows_tall: int
) -> tuple[str, str]:
    """The two heading rows: a group's name once over its criteria, a criterion alone over both rows."""
    top: list[str] = []
    below: list[str] = []
    for group, members_of in itertools.groupby(criteria, key=lambda item: item.group):
        members = list(members_of)
        if not group:
            top.extend(
                _heading(item.label, _threshold_words(item), rows=rows_tall)
                for item in members
            )
            continue
        shared = len({(item.test, item.threshold) for item in members}) == 1
        top.append(
            _heading(
                group,
                _threshold_words(members[0]) if shared else "",
                columns=len(members),
            )
        )
        below.extend(
            _heading(item.label, "" if shared else _threshold_words(item))
            for item in members
        )
    return "".join(top), "".join(below)


def _heading(label: str, declared: str = "", *, rows: int = 1, columns: int = 1) -> str:
    """A column heading: what the column is and, on a second line, what was declared for it."""
    second = (
        f"<br><span style='font-weight:normal;opacity:0.75'>{html.escape(declared)}</span>"
        if declared
        else ""
    )
    return (
        f"<th rowspan='{rows}' colspan='{columns}' style='border:1px solid var(--border,#d0d7de);"
        f"padding:0.35rem 0.65rem;text-align:left'>{html.escape(label)}{second}</th>"
    )


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
        else f" ({_within(worst.label, item.criterion.name)})"
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


def change_figures(
    check: ConvergenceCheck, theme: ReportTheme, quantity: str
) -> list[Any]:
    """What the change criteria on ``quantity`` measured at each pass, on a log axis, with their thresholds.

    The criteria of one group share a figure, a line each (MA, MS and SA at one inset); a criterion that
    stands alone has its own, a line per number it reads.
    """
    if check.plan is None or not check.passes:
        return []
    criteria = check.plan.criteria
    xs = [item.number for item in check.passes]
    figures: list[Any] = []
    for members in _sharing_a_figure(criteria, quantity):
        first = criteria[members[0]]
        traces = [
            Trace(
                _line_name(criteria[index], planned),
                [
                    item.criteria[index].values[position].measured
                    for item in check.passes
                ],
            )
            for index in members
            for position, planned in enumerate(criteria[index].values)
        ]
        figure = LinePlot(
            title=f"{_figure_subject(first)} vs AMR pass",
            xlabel="AMR pass",
            ylabel=_axis_words(first),
            xs=xs,
            traces=traces,
            yaxis_type="log",
            hlines=_threshold_lines([criteria[index] for index in members]),
            theme=theme,
        ).figure()
        if figure is not None:
            figures.append(figure)
    return figures


def _sharing_a_figure(
    criteria: Sequence[PlannedCriterion], quantity: str
) -> list[list[int]]:
    """The positions of the change criteria on ``quantity``, by group and by what they measure, in order."""
    groups: dict[tuple[str, str, str], list[int]] = {}
    for index, criterion in enumerate(criteria):
        if criterion.quantity != quantity or not criterion.measures_change:
            continue
        key = (criterion.group or criterion.name, criterion.test, criterion.unit)
        groups.setdefault(key, []).append(index)
    return list(groups.values())


def _figure_subject(criterion: PlannedCriterion) -> str:
    """What a change figure is of: its group's measure, or what its axis says for a criterion alone."""
    if criterion.group:
        return f"{criterion.group} {_measured_words(criterion)}"
    return _axis_words(criterion)


def _line_name(criterion: PlannedCriterion, value: PlannedValue) -> str:
    """A line's name in the legend: the number's own label, or within a group the criterion's and which number."""
    if not criterion.group:
        return value.label
    if len(criterion.values) == 1:
        return criterion.label
    return f"{criterion.label}, {_within(value.label, criterion.name)}"


def _threshold_lines(criteria: Sequence[PlannedCriterion]) -> list[ReferenceLine]:
    """One line per threshold of the figure's criteria, named by its criterion when they differ."""
    thresholds = {criterion.threshold for criterion in criteria}
    if len(thresholds) == 1:
        return [ReferenceLine(criteria[0].threshold, _threshold_words(criteria[0]))]
    return [
        ReferenceLine(
            criterion.threshold, f"{criterion.label} {_threshold_words(criterion)}"
        )
        for criterion in criteria
    ]


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
    """The figure's vertical axis: whose change it is and the name of the measure, ``C diagonal Delta %``."""
    subject = {
        "capacitance": f"{criterion.label} ",
        "surface_epr": "Surface EPR ",
    }.get(criterion.quantity, "")
    unit = (
        ""
        if criterion.test == "percent_change" or criterion.unit == "1"
        else f" ({criterion.unit})"
    )
    return f"{subject}{_measured_words(criterion)}{unit}"


def _threshold_words(criterion: PlannedCriterion) -> str:
    """The declared threshold as a heading states it: ``≤ 0.5 %``, ``change ≤ 0.001 GHz``, ``≤ 0.045``."""
    bound = f"≤ {_number(criterion.threshold, criterion)}"
    return f"change {bound}" if criterion.test == "absolute_change" else bound


def _number(value: float, criterion: PlannedCriterion) -> str:
    """A measured number or threshold as shown: a percentage, or in the criterion's unit unless unitless."""
    if criterion.test == "percent_change":
        return f"{value:.3g} %"
    unit = "" if criterion.unit == "1" else f" {criterion.unit}"
    return f"{value:.3g}{unit}"


__all__ = [
    "change_figures",
    "declared_html",
    "pass_limit_text",
    "pass_table_html",
    "solved_elements",
    "threshold_line",
    "verdict_card",
    "verdict_sentence",
]
