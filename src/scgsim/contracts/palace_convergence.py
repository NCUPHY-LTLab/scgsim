"""The convergence plan of a Palace run and the one check that decides, pass by pass, whether the run converged.

``set_convergence`` declares criteria; ``write_config()`` turns them into a ``ConvergencePlan``: for every
criterion, which numbers to read from which result table of a pass, how to compare them and against what
threshold.  The plan is written to the run folder (``metadata/palace_convergence_plan.json``) and bound into
the handoff.  ``check_convergence`` executes the plan on the pass folders of a returned run and is the only
code that decides "converged": result reading and the trust report call it, and nothing else computes a
change or compares one with a threshold.

Passes are Palace's solves.  Pass 1 is the solve on the input mesh, pass k the solve after k - 1
refinements; Palace writes pass k to ``results/palace/iteration<k>`` and the last solve to
``results/palace``.  A change at pass k compares pass k with pass k - 1, so a change criterion cannot hold
at pass 1.  A criterion holds at a pass when every number it reads holds; the run has converged at pass k
when every criterion holds at each of the ``minimum_converged_passes`` passes ending at k.

The module imports only the standard library and has no ``__future__`` import, so it can be copied into a
run folder and run there without SCGSim, as the receipt recorder is.
"""

import csv
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

PLAN_SCHEMA = "scgsim.palace.convergence-plan"
PLAN_SCHEMA_VERSION = 1

Test = Literal["relative_change", "absolute_change", "at_most"]
"""How a criterion compares a number with its threshold.

``relative_change``: ``|q_k - q_(k-1)| / |q_k|``.  When both are zero the change is zero; when only ``q_k`` is
zero the relative change is undefined and the criterion does not hold.  ``absolute_change``:
``|q_k - q_(k-1)|`` in the unit of the number.  ``at_most``: the number itself.  Each holds when what it
measures is at most the threshold.
"""
TESTS: tuple[Test, ...] = ("relative_change", "absolute_change", "at_most")

Quantity = Literal["capacitance", "mode_frequency", "surface_epr", "error_indicator"]
QUANTITIES: tuple[Quantity, ...] = (
    "capacitance",
    "mode_frequency",
    "surface_epr",
    "error_indicator",
)

_WHERE_PLAN = (
    "the run folder's convergence plan: write it again with set_convergence() and write_config(), "
    "or fetch the returned run again"
)


def _refused(subject: str, problem: str) -> str:
    """The sentence ``scgsim.refusal.refused`` words, written here because a run folder has no SCGSim."""
    return f"{subject}: {problem}. Change it in {_WHERE_PLAN}."


# ---- The plan --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TableRow:
    """The row of a result table a number is read from: the row whose index column ``column`` is ``index``.

    ``column`` is ``i`` (an excitation of an Electrostatic table) or ``m`` (a mode of an Eigenmode table).
    """

    column: str
    index: int


@dataclass(frozen=True)
class PlannedValue:
    """One number a criterion reads from each pass: the sum of ``columns`` on ``row`` of ``table``.

    ``table`` is the file name of the table inside a pass folder (``terminal-C.csv``).  ``row`` is None for a
    table with exactly one row (``error-indicators.csv``).  A capacitance or a frequency is one column; the
    participation of an interface kind is the sum of the columns of its Surface-EPR rows.
    """

    label: str
    table: str
    row: TableRow | None
    columns: tuple[str, ...]

    def to_payload(self) -> dict[str, Any]:
        """The value as the plan file holds it."""
        return {
            "label": self.label,
            "table": self.table,
            "row": None
            if self.row is None
            else {"column": self.row.column, "index": self.row.index},
            "columns": list(self.columns),
            "combine": "sum",
        }


@dataclass(frozen=True)
class PlannedCriterion:
    """One declared criterion: the numbers it reads, its test and its threshold.

    ``label`` is how the report names the criterion; ``unit`` is the unit the numbers are stored in
    (``F``, ``GHz``, or ``1`` for a participation or Palace's error indicator), which is the unit of an
    ``absolute_change`` or ``at_most`` threshold.
    """

    label: str
    quantity: Quantity
    test: Test
    threshold: float
    unit: str
    values: tuple[PlannedValue, ...]

    @property
    def measures_change(self) -> bool:
        """Whether the criterion compares a pass with the one before it, so it cannot hold at pass 1."""
        return self.test != "at_most"

    def to_payload(self) -> dict[str, Any]:
        """The criterion as the plan file holds it."""
        return {
            "label": self.label,
            "quantity": self.quantity,
            "test": self.test,
            "threshold": self.threshold,
            "unit": self.unit,
            "values": [value.to_payload() for value in self.values],
        }


@dataclass(frozen=True)
class ConvergencePlan:
    """Everything the check needs: the criteria of one run and how many passes in a row must hold."""

    problem: str
    criteria: tuple[PlannedCriterion, ...]
    minimum_converged_passes: int

    def to_payload(self) -> dict[str, Any]:
        """The plan as ``metadata/palace_convergence_plan.json`` holds it."""
        return {
            "schema": PLAN_SCHEMA,
            "schema_version": PLAN_SCHEMA_VERSION,
            "problem": self.problem,
            "minimum_converged_passes": self.minimum_converged_passes,
            "criteria": [criterion.to_payload() for criterion in self.criteria],
        }

    @classmethod
    def from_payload(cls, payload: Any) -> "ConvergencePlan":
        """Read a plan file's payload; one of another schema, or with a field out of shape, is refused.

        Raises:
            ValueError: the schema or a field is not what this version writes.
        """
        plan = _mapping(payload, "the convergence plan")
        if (
            plan.get("schema") != PLAN_SCHEMA
            or plan.get("schema_version") != PLAN_SCHEMA_VERSION
        ):
            raise ValueError(
                _refused(
                    "the convergence plan",
                    f"is schema {plan.get('schema')!r} version {plan.get('schema_version')!r}; this SCGSim "
                    f"reads {PLAN_SCHEMA!r} version {PLAN_SCHEMA_VERSION}",
                )
            )
        criteria = tuple(
            _criterion(item) for item in _list(plan.get("criteria"), "criteria")
        )
        if not criteria:
            raise ValueError(_refused("the convergence plan", "has no criterion"))
        return cls(
            problem=_text(plan.get("problem"), "problem"),
            criteria=criteria,
            minimum_converged_passes=_positive_int(
                plan.get("minimum_converged_passes"), "minimum_converged_passes"
            ),
        )


def _criterion(payload: Any) -> PlannedCriterion:
    criterion = _mapping(payload, "a criterion of the convergence plan")
    label = _text(criterion.get("label"), "criterion label")
    quantity = criterion.get("quantity")
    test = criterion.get("test")
    if quantity not in QUANTITIES or test not in TESTS:
        raise ValueError(
            _refused(
                f"criterion {label!r} of the convergence plan",
                f"has quantity {quantity!r} and test {test!r}; expected one of {QUANTITIES} and one of {TESTS}",
            )
        )
    threshold: Any = criterion.get("threshold")
    if not _positive_finite(threshold):
        raise ValueError(
            _refused(
                f"criterion {label!r} of the convergence plan",
                f"has threshold {threshold!r}; it must be a positive finite number",
            )
        )
    values = tuple(
        _value(item) for item in _list(criterion.get("values"), f"{label} values")
    )
    if not values:
        raise ValueError(
            _refused(f"criterion {label!r} of the convergence plan", "reads no number")
        )
    return PlannedCriterion(
        label=label,
        quantity=quantity,
        test=test,
        threshold=float(threshold),
        unit=_text(criterion.get("unit"), f"{label} unit"),
        values=values,
    )


def _value(payload: Any) -> PlannedValue:
    value = _mapping(payload, "a value of the convergence plan")
    label = _text(value.get("label"), "value label")
    if value.get("combine") != "sum":
        raise ValueError(
            _refused(
                f"value {label!r} of the convergence plan",
                f"combines its columns by {value.get('combine')!r}; this version sums them",
            )
        )
    columns = tuple(
        _text(column, f"{label} column")
        for column in _list(value.get("columns"), f"{label} columns")
    )
    if not columns:
        raise ValueError(
            _refused(f"value {label!r} of the convergence plan", "names no column")
        )
    row = value.get("row")
    return PlannedValue(
        label=label,
        table=_text(value.get("table"), f"{label} table"),
        row=None
        if row is None
        else TableRow(
            column=_text(_mapping(row, f"{label} row").get("column"), f"{label} row"),
            index=_positive_int(row.get("index"), f"{label} row index"),
        ),
        columns=columns,
    )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(_refused(name, f"must be a JSON object, not {value!r}"))
    return value


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(_refused(name, f"must be a JSON list, not {value!r}"))
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(_refused(name, f"must be non-empty text, not {value!r}"))
    return value


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(
            _refused(name, f"must be an integer of at least 1, not {value!r}")
        )
    return value


def _positive_finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


# ---- The check -------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PassFolder:
    """One pass of a run: its number, the folder holding its tables, and whether that number is established.

    ``number`` is the pass the folder records about itself (Palace's ``Problem.Iteration``: 1 is the solve on
    the input mesh).  ``identity_problem`` says why it could not be established (no readable record, or one
    that disagrees with the folder name); such a pass is read, but no change is measured to or from it and it
    is in no run of passes in a row.
    """

    number: int
    path: Path
    identity_problem: str | None = None


@dataclass(frozen=True)
class ValueAtPass:
    """One number of a criterion at one pass: the number, what the test measured, and whether it holds.

    ``measured`` is the change from the previous pass for a change test and the number itself for
    ``at_most``.  ``unavailable`` says why ``value`` or ``measured`` is None.
    """

    label: str
    value: float | None
    measured: float | None
    holds: bool
    unavailable: str | None = None


@dataclass(frozen=True)
class CriterionAtPass:
    """One criterion at one pass: each of its numbers, and whether all of them hold."""

    criterion: PlannedCriterion
    values: tuple[ValueAtPass, ...]
    holds: bool

    @property
    def worst(self) -> ValueAtPass:
        """The number farthest from holding: an unavailable one first, then the largest measured one."""
        return max(
            self.values,
            key=lambda value: math.inf if value.measured is None else value.measured,
        )


@dataclass(frozen=True)
class PassCheck:
    """Every criterion at one pass; ``streak`` is how many passes in a row, ending here, all criteria held.

    ``identity_problem`` says why the pass's number is not established (see ``PassFolder``), or that two
    folders record the same pass; such a pass has a streak of 0 and its neighbours measure no change to it.
    """

    number: int
    folder: Path
    criteria: tuple[CriterionAtPass, ...]
    holds: bool
    streak: int
    identity_problem: str | None = None


@dataclass(frozen=True)
class ConvergenceVerdict:
    """What the passes decide.

    ``converged``: every criterion held for ``minimum_converged_passes`` passes in a row, first at Palace's
    pass ``pass_number``; ``last_to_hold`` names the criteria whose run of holding passes began last, at
    pass ``last_to_hold_from``.  ``not_converged``: no pass reached that; ``failing`` are the criteria that do
    not hold at the last pass read (empty when all hold there but not yet for enough passes in a row).
    ``not_declared``: the run has no plan.

    ``passes_read`` counts the pass folders the check read, and ``passes_read_after`` those of them after the
    converged pass, by position among the passes read.  Both differ from Palace's pass numbers when a folder
    returned only some passes (passes 1, 4 and 5 are three passes read).
    """

    state: Literal["converged", "not_converged", "not_declared"]
    passes_read: int
    pass_number: int | None = None
    passes_read_after: int | None = None
    last_to_hold: tuple[str, ...] = ()
    last_to_hold_from: int | None = None
    failing: tuple[CriterionAtPass, ...] = ()


@dataclass(frozen=True)
class ConvergenceCheck:
    """The plan of a run, the check of every pass read, and the verdict; ``plan`` is None when nothing was declared."""

    plan: ConvergencePlan | None
    passes: tuple[PassCheck, ...]
    verdict: ConvergenceVerdict

    @classmethod
    def not_declared(cls) -> "ConvergenceCheck":
        """The check of a run whose folder has no convergence plan."""
        return cls(None, (), ConvergenceVerdict("not_declared", passes_read=0))


def check_convergence(
    plan: ConvergencePlan, passes: Sequence[PassFolder]
) -> ConvergenceCheck:
    """Read the planned numbers of every pass, apply each criterion, and decide the verdict.

    ``passes`` are in order of their numbers.  A change is measured only between two passes whose numbers are
    established and consecutive: otherwise it is unavailable, with the reason, the change criterion does not
    hold there, and the run of passes in a row starts again.  A value criterion needs no neighbour.  Two
    folders that record the same pass make both unreadable, since which is that solve cannot be told.
    """
    problems = _identity_problems(passes)
    readings = [
        _read_pass(plan, folder.path)
        if problem is None or folder.identity_problem == problem
        else _unavailable(plan, problem)
        for folder, problem in zip(passes, problems, strict=True)
    ]
    checks: list[PassCheck] = []
    for position, folder in enumerate(passes):
        previous = _previous(passes, problems, readings, position)
        criteria = tuple(
            _criterion_at_pass(
                criterion,
                readings[position][index],
                previous if isinstance(previous, str) else previous[index],
            )
            for index, criterion in enumerate(plan.criteria)
        )
        holds = all(item.holds for item in criteria)
        continues = (
            bool(checks)
            and checks[-1].identity_problem is None
            and checks[-1].number == folder.number - 1
        )
        streak = (
            0
            if not holds or problems[position] is not None
            else checks[-1].streak + 1
            if continues
            else 1
        )
        checks.append(
            PassCheck(
                folder.number, folder.path, criteria, holds, streak, problems[position]
            )
        )
    return ConvergenceCheck(plan, tuple(checks), _verdict(plan, checks))


def _identity_problems(passes: Sequence[PassFolder]) -> list[str | None]:
    """Why each pass's number is not established: its own record, or that another folder records it too."""
    problems: list[str | None] = []
    for folder in passes:
        twins = [item.path.name for item in passes if item.number == folder.number]
        if folder.identity_problem is not None:
            problems.append(folder.identity_problem)
        elif len(twins) > 1:
            problems.append(
                f"pass {folder.number} is recorded by {len(twins)} folders ({', '.join(twins)})"
            )
        else:
            problems.append(None)
    return problems


def _unavailable(plan: ConvergencePlan, reason: str) -> list[list["_Reading"]]:
    """Every planned number of a pass that cannot be read, each with ``reason``."""
    return [
        [(None, reason) for _value in criterion.values] for criterion in plan.criteria
    ]


def _previous(
    passes: Sequence[PassFolder],
    problems: list[str | None],
    readings: list[list[list["_Reading"]]],
    position: int,
) -> list[list["_Reading"]] | str:
    """The readings of the solve immediately before this pass, or why there is none to compare with."""
    number = passes[position].number
    if problems[position] is not None:
        return f"no previous pass to compare with: {problems[position]}"
    if number == 1:
        return "no previous pass to compare with"
    if position == 0 or passes[position - 1].number != number - 1:
        return f"no previous pass to compare with: pass {number - 1} was not read"
    if problems[position - 1] is not None:
        return f"no previous pass to compare with: {problems[position - 1]}"
    return readings[position - 1]


_Reading = tuple[float | None, str | None]
"""One number read from a pass: the number, or None with why it could not be read."""


def _read_pass(plan: ConvergencePlan, folder: Path) -> list[list[_Reading]]:
    """Every planned number of one pass, by criterion and value; each table is read once."""
    tables: dict[str, _Table | str] = {}
    readings: list[list[_Reading]] = []
    for criterion in plan.criteria:
        row: list[_Reading] = []
        for value in criterion.values:
            if value.table not in tables:
                tables[value.table] = _read_table(folder / value.table)
            table = tables[value.table]
            row.append((None, table) if isinstance(table, str) else table.sum(value))
        readings.append(row)
    return readings


@dataclass(frozen=True)
class _Table:
    """A result table of one pass: its headers and its rows of text cells, as Palace wrote them."""

    name: str
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]

    def sum(self, value: PlannedValue) -> _Reading:
        """The sum of ``value``'s columns on its row, or None with why it cannot be read."""
        cells = self._row(value.row)
        if isinstance(cells, str):
            return None, cells
        total = 0.0
        for column in value.columns:
            if column not in self.headers:
                return None, f"{self.name} has no column {column!r}"
            number = _number(cells[self.headers.index(column)])
            if number is None:
                return None, f"{self.name} {column!r} is not a finite number"
            total += number
        return total, None

    def _row(self, row: TableRow | None) -> tuple[str, ...] | str:
        if row is None:
            if len(self.rows) != 1:
                return f"{self.name} has {len(self.rows)} rows where one is expected"
            return self.rows[0]
        if row.column not in self.headers:
            return f"{self.name} has no index column {row.column!r}"
        position = self.headers.index(row.column)
        matching = [
            cells for cells in self.rows if _number(cells[position]) == row.index
        ]
        if not matching:
            return f"{self.name} has no row {row.column} = {row.index}"
        if len(matching) > 1:
            # Which of them Palace meant cannot be told, so neither is read.
            return f"{self.name} has {len(matching)} rows {row.column} = {row.index}, so the row is ambiguous"
        return matching[0]


def _read_table(path: Path) -> _Table | str:
    """A pass's table as text cells, or why it cannot be read; headers and cells lose Palace's padding."""
    if not path.is_file():
        return f"{path.name} is missing"
    with path.open(encoding="utf-8", newline="") as handle:
        lines = [
            tuple(cell.strip() for cell in line)
            for line in csv.reader(handle)
            if any(cell.strip() for cell in line)
        ]
    if not lines:
        return f"{path.name} is empty"
    headers, rows = lines[0], tuple(lines[1:])
    repeated = sorted({header for header in headers if headers.count(header) > 1})
    if repeated:
        # A repeated column makes every cell under that name ambiguous; the whole table is not read.
        return (
            f"{path.name} repeats the column(s) {repeated}, so its cells are ambiguous"
        )
    if any(len(cells) != len(headers) for cells in rows):
        return f"{path.name} has a row whose cells do not match its headers"
    return _Table(path.name, headers, rows)


def _number(cell: str) -> float | None:
    try:
        number = float(cell)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _criterion_at_pass(
    criterion: PlannedCriterion,
    readings: list[_Reading],
    previous: list[_Reading] | str,
) -> CriterionAtPass:
    """``previous`` is the readings of the pass before, or why there is none to compare with."""
    values = tuple(
        _value_at_pass(
            criterion,
            planned.label,
            reading,
            previous if isinstance(previous, str) else previous[index],
        )
        for index, (planned, reading) in enumerate(
            zip(criterion.values, readings, strict=True)
        )
    )
    return CriterionAtPass(criterion, values, all(value.holds for value in values))


def _value_at_pass(
    criterion: PlannedCriterion,
    label: str,
    reading: _Reading,
    previous: _Reading | str,
) -> ValueAtPass:
    value, unavailable = reading
    if value is None:
        return ValueAtPass(label, None, None, False, unavailable)
    if criterion.test == "at_most":
        return ValueAtPass(label, value, value, value <= criterion.threshold)
    if isinstance(previous, str):
        return ValueAtPass(label, value, None, False, previous)
    earlier, earlier_unavailable = previous
    if earlier is None:
        return ValueAtPass(
            label, value, None, False, f"previous pass: {earlier_unavailable}"
        )
    change = abs(value - earlier)
    if criterion.test == "relative_change":
        if value == 0.0:
            if earlier != 0.0:
                return ValueAtPass(
                    label,
                    value,
                    None,
                    False,
                    "relative change undefined: the value is zero",
                )
            # Both passes read zero: nothing changed.
            return ValueAtPass(label, value, 0.0, True)
        change /= abs(value)
    return ValueAtPass(label, value, change, change <= criterion.threshold)


def _verdict(plan: ConvergencePlan, checks: list[PassCheck]) -> ConvergenceVerdict:
    """Converged at the first pass whose streak reaches ``minimum_converged_passes``; otherwise what fails last."""
    converged_at = next(
        (
            position
            for position, check in enumerate(checks)
            if check.streak >= plan.minimum_converged_passes
        ),
        None,
    )
    if converged_at is None:
        return ConvergenceVerdict(
            "not_converged",
            passes_read=len(checks),
            failing=tuple(
                item
                for item in (checks[-1].criteria if checks else ())
                if not item.holds
            ),
        )
    starts = [
        _holding_since(checks, converged_at, index)
        for index in range(len(plan.criteria))
    ]
    latest = max(starts)
    return ConvergenceVerdict(
        "converged",
        passes_read=len(checks),
        pass_number=checks[converged_at].number,
        passes_read_after=len(checks) - 1 - converged_at,
        last_to_hold=tuple(
            criterion.label
            for criterion, start in zip(plan.criteria, starts, strict=True)
            if start == latest
        ),
        last_to_hold_from=latest,
    )


def _holding_since(checks: list[PassCheck], position: int, criterion: int) -> int:
    """The first pass of the unbroken run of passes, ending at ``position``, at which ``criterion`` holds."""
    start = position
    while (
        start > 0
        and checks[start - 1].identity_problem is None
        and checks[start - 1].number == checks[start].number - 1
        and checks[start - 1].criteria[criterion].holds
    ):
        start -= 1
    return checks[start].number
