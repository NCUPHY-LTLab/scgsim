# long-file: the enumeration of passes and the check travel to the solver host as one standalone script.

"""The convergence plan of a Palace run, the passes a returned run holds, and the one check on them.

``set_convergence`` declares criteria; ``write_config()`` turns them into a ``ConvergencePlan``: for every
criterion, which numbers to read from which result table of a pass, how to compare them and against what
threshold.  The plan is written to the run folder (``metadata/palace_convergence_plan.json``) and bound into
the handoff.  ``returned_passes`` is the one enumeration of the passes of a ``results/palace`` folder, by
the identity of its files; ``collect_amr_passes`` and result reading both use it.  ``check_convergence``
executes the plan on those passes and is the only code that decides "converged": nothing else computes a
change or compares one with a threshold.

Passes are Palace's solves.  Pass 1 is the solve on the input mesh, pass k the solve after k - 1
refinements; Palace writes pass k to ``results/palace/iteration<k>`` and the last solve to
``results/palace``.  A change at pass k compares pass k with pass k - 1, so a change criterion cannot hold
at pass 1.  A criterion holds at a pass when every number it reads holds; the run has converged at pass k
when every criterion holds at each of the ``minimum_converged_passes`` passes ending at k.

The convergence runner (``scgsim.palace.run.convergence_runner``) runs the same check on the solver host
after each pass and stops the solver once it has converged; it records what it read and how the solver ended
in a ``ConvergenceDecision``, which the receipt binds and result reading recomputes (``check_disagreement``).

The module imports only the standard library and the run-folder contract, and has no ``__future__`` import,
so it can be copied into a run folder and run there without SCGSim, as the receipt recorder is.
"""

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from scgsim.contracts.palace_run_folder import RunEnding

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
    """One pass of a run: its number, the folder holding its tables, and whether its readings can be used.

    ``number`` is the pass the folder records about itself (Palace's ``Problem.Iteration``: 1 is the solve on
    the input mesh), established by ``returned_passes``.  ``identity_problem`` says why the pass cannot be read
    (two folders record it with different tables); none of its readings is then used, no change is measured to
    or from it, and it is in no run of passes in a row.
    """

    number: int
    path: Path
    identity_problem: str | None = None


# ---- The passes of a returned results folder ---------------------------------------------------------------
#
# Palace 0.18.1 writes the tables of the solve under way at the top of ``results/palace``.  After solve k,
# when it refines again with ``SaveAdaptIterations``, it moves those tables into ``iteration<k>/`` (zero-padded
# to the width of ``MaxIts``), leaves a symbolic link to each at the top level, and writes a new ``palace.json``
# there; ``iteration<k>/palace.json`` records ``Problem.Iteration = k``.  The next solve replaces the links one
# by one, starting some tables with their header only.  The last solve of a run that ends by itself stays at
# the top level as regular files, and its ``palace.json`` records its own iteration.  So which pass a folder
# holds is decided by what its files are, never by its place in a listing.


@dataclass(frozen=True)
class SetAside:
    """A result folder that is not read as a pass, and why (a number not established, an unfinished solve)."""

    folder: Path
    reason: str


@dataclass(frozen=True)
class ReturnedPasses:
    """The passes a ``results/palace`` folder holds, in order of their numbers, and what it holds besides.

    ``final`` says what the top level is: ``pass`` (a solve of its own), ``copy`` (regular files equal to the
    iteration of the pass it records, as a tool that follows links leaves a stopped run), ``linked`` (links
    into an iteration, so no solve of its own), ``unfinished`` (a solve under way: links and new files mixed,
    or a table with its header only), ``missing`` (no table) or ``set_aside`` (a number not established, or
    a table that cannot be read).
    """

    passes: tuple[PassFolder, ...]
    set_aside: tuple[SetAside, ...]
    final: Literal["pass", "copy", "linked", "unfinished", "missing", "set_aside"]
    final_reason: str | None = None


_METADATA = "palace.json"
_ITERATION_PREFIX = "iteration"


def returned_passes(results: Path) -> ReturnedPasses:
    """Enumerate the passes of ``results`` (a run's ``results/palace``) by the identity of their files.

    * ``iteration<k>/`` is pass k when its ``palace.json`` records ``Problem.Iteration`` k and its tables are
      regular files; otherwise it is set aside with the reason.
    * The top level is a pass of its own only when its ``palace.json`` and every table are regular files
      with data rows; its number is the iteration it records.  Tables that are links into ``iteration<k>/``
      are that iteration's files.  Links mixed with regular files, or a table with only its header, are a
      solve under way.
    * Two folders that record the same pass are one pass when they hold the same tables, byte for byte;
      otherwise the pass is kept with an ``identity_problem`` and none of its readings is used.

    There is no other numbering: a pass's number is recorded and consistent, or the folder is set aside.
    A folder or a table that cannot be read sets that folder aside, or leaves the pass with an
    ``identity_problem`` when it is needed to tell two folders apart; the other folders are read all the
    same.  Only a results folder that cannot be listed itself is set aside whole.
    """
    try:
        return _returned_passes(results)
    except OSError as error:
        reason = _unread(results.name, error)
        return ReturnedPasses((), (SetAside(results, reason),), "set_aside", reason)


def _unread(name: str, error: OSError) -> str:
    """Why a folder or a table could not be read: its name and the system's reason."""
    return f"{name} cannot be read ({error.strerror or error})"


def _returned_passes(results: Path) -> ReturnedPasses:
    candidates: list[tuple[int, Path]] = []
    set_aside: list[SetAside] = []
    iterations = sorted(
        (number, child)
        for child in (results.iterdir() if results.is_dir() else ())
        if child.is_dir() and (number := _folder_iteration(child.name)) is not None
    )
    for number, folder in iterations:
        try:
            reason = _iteration_problem(folder, number)
        except OSError as error:
            # This folder only: its record or its listing cannot be read, the others can.
            reason = _unread(folder.name, error)
        if reason is None:
            candidates.append((number, folder))
        else:
            set_aside.append(SetAside(folder, reason))
    try:
        final, final_reason, recorded = _top_level(results)
    except OSError as error:
        final, final_reason, recorded = (
            "set_aside",
            f"the final result cannot be read ({error.strerror or error})",
            None,
        )
    if final == "pass" and recorded is not None:
        candidates.append((recorded, results))
    elif final in ("unfinished", "set_aside") and final_reason is not None:
        set_aside.append(SetAside(results, final_reason))
    passes, copied = _merge(candidates)
    return ReturnedPasses(
        tuple(passes),
        tuple(set_aside),
        "copy" if final == "pass" and copied else final,
        final_reason,
    )


def _folder_iteration(name: str) -> int | None:
    """The ``N`` of a folder named ``iteration<N>``, None for any other name."""
    number = name.removeprefix(_ITERATION_PREFIX)
    return (
        int(number) if name.startswith(_ITERATION_PREFIX) and number.isdigit() else None
    )


def _tables(folder: Path) -> list[Path]:
    """The tables of a result folder: its ``.csv`` files, links included."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.suffix == ".csv" and (path.is_file() or path.is_symlink())
    )


def _recorded_iteration(folder: Path) -> int | str:
    """The ``Problem.Iteration`` a folder's own ``palace.json`` records, or why there is none."""
    metadata = folder / _METADATA
    if metadata.is_symlink() or not metadata.is_file():
        return f"has no {_METADATA} of its own"
    try:
        payload = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return f"has a {_METADATA} that cannot be read"
    problem = payload.get("Problem") if isinstance(payload, dict) else None
    recorded = problem.get("Iteration") if isinstance(problem, dict) else None
    if type(recorded) is not int or recorded < 1:
        return f"has a {_METADATA} that records no Problem.Iteration"
    return recorded


def _iteration_problem(folder: Path, number: int) -> str | None:
    """Why ``iteration<number>/`` is not pass ``number``, or None when it is."""
    recorded = _recorded_iteration(folder)
    if isinstance(recorded, str):
        return f"{folder.name} {recorded}"
    if recorded != number:
        return f"{folder.name} records Problem.Iteration {recorded}, not {number}"
    linked = [path.name for path in _tables(folder) if path.is_symlink()]
    if linked:
        return f"{folder.name} holds links, not tables of its own: {linked}"
    return None


def _top_level(
    results: Path,
) -> tuple[
    Literal["pass", "linked", "unfinished", "missing", "set_aside"],
    str | None,
    int | None,
]:
    """What the top level of ``results`` is, why when it is no pass, and the pass it records when it is."""
    tables = _tables(results) if results.is_dir() else []
    if not tables:
        return "missing", "the final result holds no table", None
    links = [path for path in tables if path.is_symlink()]
    if links:
        targets = {_link_iteration(results, path) for path in links}
        if len(links) == len(tables) and len(targets) == 1 and None not in targets:
            (target,) = targets
            return (
                "linked",
                f"the final result's tables are links into iteration{target}",
                None,
            )
        return (
            "unfinished",
            "the final result is a solve under way: its tables mix links into iteration folders with new files",
            None,
        )
    contents = {path.name: _table_bytes(path) for path in tables}
    unreadable = [content for content in contents.values() if isinstance(content, str)]
    if unreadable:
        return (
            "set_aside",
            f"the final result holds a table that cannot be read: {'; '.join(unreadable)}",
            None,
        )
    header_only = [
        name
        for name, content in contents.items()
        if isinstance(content, bytes) and _data_rows(content) == 0
    ]
    if header_only:
        return (
            "unfinished",
            f"the final result is a solve under way: {header_only} hold only their header",
            None,
        )
    recorded = _recorded_iteration(results)
    if isinstance(recorded, str):
        return "set_aside", f"the final result {recorded}", None
    return "pass", None, recorded


def _link_iteration(results: Path, link: Path) -> int | None:
    """The ``N`` of the ``iteration<N>/`` a top-level link resolves into, None when it points elsewhere."""
    try:
        target = link.resolve(strict=True)
        relative = target.relative_to(results.resolve())
    except (OSError, ValueError):
        return None
    return _folder_iteration(relative.parts[0]) if len(relative.parts) == 2 else None


def _table_bytes(path: Path) -> bytes | str:
    """A table's bytes, or why they cannot be read."""
    try:
        return path.read_bytes()
    except OSError as error:
        return _unread(path.name, error)


def _data_rows(table: bytes) -> int:
    """How many non-empty lines a table has after its header."""
    return max(sum(1 for line in table.splitlines() if line.strip()) - 1, 0)


def _merge(candidates: list[tuple[int, Path]]) -> tuple[list[PassFolder], bool]:
    """One pass per recorded number; whether the top level was a copy of an iteration it repeats."""
    passes: list[PassFolder] = []
    copied = False
    for number in sorted({n for n, _ in candidates}):
        # Iteration folders first: a copied top level's palace.json describes the solve after it.
        folders = sorted(
            (path for n, path in candidates if n == number),
            key=lambda path: _folder_iteration(path.name) is None,
        )
        if len(folders) == 1:
            passes.append(PassFolder(number, folders[0]))
            continue
        try:
            differing, unreadable = _differing_tables(folders)
        except OSError as error:
            differing, unreadable = [], [_unread(f"pass {number}", error)]
        if differing or unreadable:
            names = " and ".join(_source_name(path) for path in folders)
            problem = (
                f"whose tables cannot be compared: {'; '.join(unreadable)}"
                if unreadable
                else f"whose tables differ: {differing}"
            )
            passes.append(
                PassFolder(
                    number,
                    folders[0],
                    f"pass {number} is recorded by {names}, {problem}",
                )
            )
        else:
            passes.append(PassFolder(number, folders[0]))
            copied = copied or any(
                _folder_iteration(path.name) is None for path in folders
            )
    return passes, copied


def _source_name(folder: Path) -> str:
    return (
        folder.name
        if _folder_iteration(folder.name) is not None
        else "the final result"
    )


def _differing_tables(folders: list[Path]) -> tuple[list[str], list[str]]:
    """The tables not the same bytes in every one of ``folders`` (or held by only some), and those unreadable.

    The second list says, per unreadable table, the folder and why.
    """
    names = sorted({path.name for folder in folders for path in _tables(folder)})
    differing: list[str] = []
    unreadable: list[str] = []
    for name in names:
        contents = [
            _table_bytes(folder / name) if (folder / name).is_file() else None
            for folder in folders
        ]
        reasons = [
            f"{_source_name(folder)}: {content}"
            for folder, content in zip(folders, contents, strict=True)
            if isinstance(content, str)
        ]
        if reasons:
            unreadable.extend(reasons)
        elif len(set(contents)) != 1:
            differing.append(name)
    return differing, unreadable


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

    ``identity_problem`` is the ``PassFolder``'s: such a pass has a streak of 0 and its neighbours measure no
    change to it.
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
    """The plan of a run, the check of every pass read, and the verdict; ``plan`` is None when nothing was declared.

    ``set_aside`` are the result folders that were not read as passes, each with why.
    """

    plan: ConvergencePlan | None
    passes: tuple[PassCheck, ...]
    verdict: ConvergenceVerdict
    set_aside: tuple[SetAside, ...] = ()

    @classmethod
    def not_declared(cls) -> "ConvergenceCheck":
        """The check of a run whose folder has no convergence plan."""
        return cls(None, (), ConvergenceVerdict("not_declared", passes_read=0))


def check_convergence(
    plan: ConvergencePlan,
    passes: Sequence[PassFolder],
    set_aside: Sequence[SetAside] = (),
) -> ConvergenceCheck:
    """Read the planned numbers of every pass, apply each criterion, and decide the verdict.

    ``passes`` are ``returned_passes``'s, in order of their numbers; ``set_aside`` its folders that are no
    pass, kept for the record.  A change is measured only between two readable passes whose numbers are
    consecutive: otherwise it is unavailable, with the reason, the change criterion does not hold there, and
    the run of passes in a row starts again.  A value criterion needs no neighbour.
    """
    problems = [folder.identity_problem for folder in passes]
    readings = [
        _read_pass(plan, folder.path)
        if folder.identity_problem is None
        else _unavailable(plan, folder.identity_problem)
        for folder in passes
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
    return ConvergenceCheck(
        plan, tuple(checks), _verdict(plan, checks), tuple(set_aside)
    )


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
        return "no pass before it"
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
    content = _table_bytes(path)
    if isinstance(content, str):
        return content
    try:
        lines = [
            tuple(cell.strip() for cell in line)
            for line in csv.reader(content.decode("utf-8").splitlines())
            if any(cell.strip() for cell in line)
        ]
    except (UnicodeDecodeError, csv.Error) as error:
        return f"{path.name} cannot be read as a table ({error})"
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


def top_level_state(
    results: Path,
) -> tuple[Literal["pass", "linked", "unfinished", "missing", "set_aside"], str | None]:
    """What the top level of ``results`` holds, and why when it is no pass, without reading the iterations.

    ``pass``: the last solve of a run that ended by itself.  ``linked`` or ``unfinished``: the top level of a
    solver that was ended during its adaptive loop.  Result reading compares it with how the receipt says the
    solver ended; ``returned_passes`` gives the same answer, and says besides whether a top level is a copy.
    """
    state, reason, _recorded = _top_level(results)
    return state, reason


# ---- The decision record -----------------------------------------------------------------------------------
#
# The runner writes ``metadata/palace_convergence_decision.json`` twice: once when the verdict is converged,
# before it signals the solver, and again when the solver has ended, with the passes the results folder then
# holds and the solver's exit code.  A run that ends without a stop gets the second record only.

DECISION_SCHEMA = "scgsim.palace.convergence-decision"
DECISION_SCHEMA_VERSION = 1

_WHERE_DECISION = (
    "the run folder's convergence decision: the runner writes it during the solve; run the handoff again, "
    "or fetch the returned run again"
)


def _decision_refused(subject: str, problem: str) -> str:
    """The sentence ``scgsim.refusal.refused`` words, for the decision record."""
    return f"{subject}: {problem}. Change it in {_WHERE_DECISION}."


StopReason = Literal["converged", "runner_terminated"]
"""Why the runner signalled the solver: the verdict was converged, or the runner itself was told to end."""


@dataclass(frozen=True)
class SolverStop:
    """The runner's stop of the solver: why, after which pass, and when it signalled the solver's session.

    ``after_pass`` is the last pass the runner had read when it decided, None when it had read none.
    ``escalated_utc`` is when it sent SIGKILL because the solver had not ended within its wait after SIGTERM.
    """

    reason: StopReason
    after_pass: int | None
    requested_utc: str
    escalated_utc: str | None = None

    def to_payload(self) -> dict[str, Any]:
        """The stop as the decision record holds it."""
        return {
            "reason": self.reason,
            "after_pass": self.after_pass,
            "signal": "SIGTERM",
            "requested_utc": self.requested_utc,
            "escalated_utc": self.escalated_utc,
        }


@dataclass(frozen=True)
class SolverExit:
    """How the solver ended as the runner saw it: its exit code as a shell reports it (128 + N for signal N)."""

    exit_code: int
    ended_utc: str


@dataclass(frozen=True)
class ConvergenceDecision:
    """What the runner read, what the check decided and how the solver ended, as the decision record holds it.

    ``checked`` is ``recorded_check`` of the check the runner ran last: its passes with their readings, the
    folders set aside and the verdict.  ``seen_utc`` maps each pass number to when the runner first read it.
    ``solver`` is None in the record written before the stop.  ``look_interval_s`` and ``stop_wait_s`` are the
    runner's operating parameters, kept for the reader of the record.
    """

    plan_sha256: str
    started_utc: str
    look_interval_s: float
    stop_wait_s: float
    checked: dict[str, Any]
    seen_utc: dict[str, str]
    stop: SolverStop | None
    solver: SolverExit | None

    @property
    def ending(self) -> RunEnding | None:
        """How the solver ended (``RunEnding``); None while the record is the one written before the stop."""
        if self.solver is None:
            return None
        if self.stop is None:
            return "solver_ended"
        if self.stop.escalated_utc is not None:
            return "stop_escalated"
        if self.stop.reason == "runner_terminated":
            return "runner_terminated"
        return "stopped_at_convergence"

    @property
    def result_folder(self) -> str | None:
        """The folder of the last pass recorded, relative to the run folder: a stopped run's result."""
        passes = self.checked.get("passes") or []
        return passes[-1]["folder"] if passes else None

    def to_payload(self) -> dict[str, Any]:
        """The decision as ``metadata/palace_convergence_decision.json`` holds it."""
        return {
            "schema": DECISION_SCHEMA,
            "schema_version": DECISION_SCHEMA_VERSION,
            "plan_sha256": self.plan_sha256,
            "runner": {
                "look_interval_s": self.look_interval_s,
                "stop_wait_s": self.stop_wait_s,
            },
            "started_utc": self.started_utc,
            **self.checked,
            "seen_utc": dict(self.seen_utc),
            "stop": None if self.stop is None else self.stop.to_payload(),
            "solver": None
            if self.solver is None
            else {
                "exit_code": self.solver.exit_code,
                "ended_utc": self.solver.ended_utc,
            },
        }

    @classmethod
    def from_payload(cls, payload: Any) -> "ConvergenceDecision":
        """Read a decision record; one of another schema, or with a field out of shape, is refused.

        Raises:
            ValueError: the schema or a field is not what this version writes.
        """
        record = _decision_mapping(payload, "the convergence decision")
        if (
            record.get("schema") != DECISION_SCHEMA
            or record.get("schema_version") != DECISION_SCHEMA_VERSION
        ):
            raise ValueError(
                _decision_refused(
                    "the convergence decision",
                    f"is schema {record.get('schema')!r} version {record.get('schema_version')!r}; this "
                    f"SCGSim reads {DECISION_SCHEMA!r} version {DECISION_SCHEMA_VERSION}",
                )
            )
        runner = _decision_mapping(record.get("runner"), "runner")
        return cls(
            plan_sha256=_decision_text(record.get("plan_sha256"), "plan_sha256"),
            started_utc=_decision_text(record.get("started_utc"), "started_utc"),
            look_interval_s=_decision_seconds(
                runner.get("look_interval_s"), "look_interval_s"
            ),
            stop_wait_s=_decision_seconds(runner.get("stop_wait_s"), "stop_wait_s"),
            checked={
                "passes": _decision_list(record.get("passes"), "passes"),
                "set_aside": _decision_list(record.get("set_aside"), "set_aside"),
                "verdict": dict(_decision_mapping(record.get("verdict"), "verdict")),
            },
            seen_utc={
                str(key): _decision_text(value, f"seen_utc {key}")
                for key, value in _decision_mapping(
                    record.get("seen_utc"), "seen_utc"
                ).items()
            },
            stop=_solver_stop(record.get("stop")),
            solver=_solver_exit(record.get("solver")),
        )


def _solver_stop(payload: Any) -> SolverStop | None:
    if payload is None:
        return None
    stop = _decision_mapping(payload, "stop")
    reason = stop.get("reason")
    if (
        reason not in ("converged", "runner_terminated")
        or stop.get("signal") != "SIGTERM"
    ):
        raise ValueError(
            _decision_refused(
                "stop of the convergence decision",
                f"has reason {reason!r} and signal {stop.get('signal')!r}; expected 'converged' or "
                "'runner_terminated', and 'SIGTERM'",
            )
        )
    after = stop.get("after_pass")
    escalated = stop.get("escalated_utc")
    return SolverStop(
        reason=reason,
        after_pass=None if after is None else _positive_int(after, "stop after_pass"),
        requested_utc=_decision_text(stop.get("requested_utc"), "stop requested_utc"),
        escalated_utc=None
        if escalated is None
        else _decision_text(escalated, "stop escalated_utc"),
    )


def _solver_exit(payload: Any) -> SolverExit | None:
    if payload is None:
        return None
    solver = _decision_mapping(payload, "solver")
    code = solver.get("exit_code")
    if type(code) is not int or code < 0:
        raise ValueError(
            _decision_refused(
                "solver exit_code of the convergence decision",
                f"must be an integer of at least 0, not {code!r}",
            )
        )
    return SolverExit(code, _decision_text(solver.get("ended_utc"), "solver ended_utc"))


def _decision_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(
            _decision_refused(name, f"must be a JSON object, not {value!r}")
        )
    return value


def _decision_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(_decision_refused(name, f"must be a JSON list, not {value!r}"))
    return value


def _decision_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(
            _decision_refused(name, f"must be non-empty text, not {value!r}")
        )
    return value


def _decision_seconds(value: Any, name: str) -> float:
    if not _positive_finite(value):
        raise ValueError(
            _decision_refused(
                name, f"must be a positive number of seconds, not {value!r}"
            )
        )
    return float(value)


def recorded_check(check: ConvergenceCheck, root: Path) -> dict[str, Any]:
    """The check as the decision record holds it: every pass with its readings, the folders set aside, the verdict.

    Folders are relative to the run folder ``root``.  Result reading builds the same record from the returned
    files and compares the two (``check_disagreement``), so this is the one form both sides write.
    """
    verdict = check.verdict
    return {
        "passes": [
            {
                "number": item.number,
                "folder": item.folder.relative_to(root).as_posix(),
                "holds": item.holds,
                "streak": item.streak,
                "identity_problem": item.identity_problem,
                "criteria": [
                    {
                        "label": criterion.criterion.label,
                        "holds": criterion.holds,
                        "values": [
                            {
                                "label": value.label,
                                "value": value.value,
                                "measured": value.measured,
                                "holds": value.holds,
                                "unavailable": value.unavailable,
                            }
                            for value in criterion.values
                        ],
                    }
                    for criterion in item.criteria
                ],
            }
            for item in check.passes
        ],
        "set_aside": [
            {
                "folder": folder.folder.relative_to(root).as_posix(),
                "reason": folder.reason,
            }
            for folder in check.set_aside
        ],
        "verdict": {
            "state": verdict.state,
            "pass_number": verdict.pass_number,
            "passes_read": verdict.passes_read,
            "passes_read_after": verdict.passes_read_after,
            "last_to_hold": list(verdict.last_to_hold),
            "last_to_hold_from": verdict.last_to_hold_from,
            "failing": [item.criterion.label for item in verdict.failing],
        },
    }


def check_disagreement(
    recorded: Mapping[str, Any], read: Mapping[str, Any]
) -> str | None:
    """Where a recorded check and the check read again from the files first differ; None when they agree.

    Both are ``recorded_check`` forms.  The answer names the part (the verdict, a pass, a criterion of it, a
    number of that criterion) and both values, e.g. ``pass 3, C diagonal, C[pad][pad], value: recorded
    1.01e-13, read 1.02e-13``.
    """
    return _first_difference(dict(recorded), dict(read), "")


def _first_difference(recorded: Any, read: Any, where: str) -> str | None:
    if isinstance(recorded, dict) and isinstance(read, dict):
        if set(recorded) != set(read):
            return f"{where or 'the record'}: recorded fields {sorted(recorded)}, read {sorted(read)}"
        for key in recorded:
            found = _first_difference(recorded[key], read[key], _joined(where, key))
            if found is not None:
                return found
        return None
    if isinstance(recorded, list) and isinstance(read, list):
        recorded_names = [_item_name(item) for item in recorded]
        read_names = [_item_name(item) for item in read]
        if len(recorded) != len(read) or recorded_names != read_names:
            return (
                f"{where}: recorded {_names_listed(recorded, recorded_names)}, read "
                f"{_names_listed(read, read_names)}"
            )
        # A list's items name themselves (``pass 3``, a label), so the list's own key is dropped.
        within = where.rpartition(", ")[0] if where.endswith(_LIST_KEYS) else where
        for name, left, right in zip(recorded_names, recorded, read, strict=True):
            found = _first_difference(left, right, _joined(within, name))
            if found is not None:
                return found
        return None
    if recorded != read or type(recorded) is not type(read):
        return f"{where}: recorded {recorded!r}, read {read!r}"
    return None


_LIST_KEYS = ("passes", "criteria", "values")
"""The record keys whose items are named in a disagreement by their own number or label."""


def _item_name(item: Any) -> str:
    """How a list item is named in a disagreement: ``pass 3``, a criterion's or a number's label, a folder."""
    if isinstance(item, dict):
        if "number" in item:
            return f"pass {item['number']}"
        if "label" in item:
            return str(item["label"])
        if "folder" in item:
            return str(item["folder"])
    return repr(item)


def _names_listed(items: list[Any], names: list[str]) -> str:
    return "nothing" if not items else ", ".join(names)


def _joined(where: str, part: str) -> str:
    """``pass 3, C diagonal``: the parts of a disagreement's place, outermost first."""
    return part if not where else f"{where}, {part}"


def completed_stop_problem(decision: ConvergenceDecision) -> str | None:
    """Why ``decision`` does not record a completed stop at convergence; None when it does.

    A completed stop: the runner stopped the solver with SIGTERM because the verdict was converged, the solver
    ended within the wait, the verdict of the passes recorded is converged at a pass no later than the one the
    runner stopped after, and the last pass recorded, the run's result, is readable and no earlier.
    """
    if decision.ending != "stopped_at_convergence":
        return (
            f"the solver's ending is {decision.ending!r}, not 'stopped_at_convergence'"
        )
    verdict = decision.checked["verdict"]
    if verdict.get("state") != "converged":
        return f"the verdict of the passes recorded is {verdict.get('state')!r}, not 'converged'"
    after = decision.stop.after_pass if decision.stop is not None else None
    converged_at = verdict.get("pass_number")
    if after is None or not isinstance(converged_at, int) or converged_at > after:
        return (
            f"the runner stopped the solver after pass {after}, and the passes recorded converged at "
            f"pass {converged_at}"
        )
    last = decision.checked["passes"][-1]
    if last.get("number", 0) < after or last.get("identity_problem") is not None:
        return (
            f"the last pass recorded, pass {last.get('number')}, is not a readable pass from pass {after} "
            f"on: {last.get('identity_problem') or 'it is earlier'}"
        )
    return None
