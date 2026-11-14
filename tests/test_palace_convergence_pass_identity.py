"""Which pass each folder of a returned ``results/palace`` is, by what its files are, and what the check reports.

The folders are written with the structure Palace 0.18.1 leaves (seen in real run folders):

* after solve k, when it refines again, Palace moves the top-level tables into ``iteration<k>/``, leaves a
  symbolic link to each at the top level and writes a new top-level ``palace.json``; ``iteration<k>/palace.json``
  records ``Problem.Iteration = k``;
* the next solve replaces the links one by one, starting some tables (``eig.csv``, ``port-EPR.csv``) with their
  header only;
* the last solve of a run that ends by itself stays at the top level as regular files recording its iteration.

Tables are in Palace's padded CSV format, links are real ``os.symlink`` links.  The passes come from
``returned_passes`` (``check_passes``) and ``collect_amr_passes``, the one enumeration both use.  No solver runs.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from scgsim.contracts.palace_convergence import (
    ConvergenceCheck,
    ConvergencePlan,
    PlannedCriterion,
    PlannedValue,
    TableRow,
    returned_passes,
)
from scgsim.palace.reports.convergence import pass_table_html, verdict_sentence
from scgsim.palace.results.convergence import check_passes
from scgsim.palace.results.snapshots import collect_amr_passes

_DIAGONAL = PlannedCriterion(
    "C diagonal",
    "capacitance",
    "relative_change",
    0.001,
    "F",
    (PlannedValue("C[a][a]", "terminal-C.csv", TableRow("i", 1), ("C[i][1] (F)",)),),
)
_INDICATOR = PlannedCriterion(
    "error indicator (Norm)",
    "error_indicator",
    "at_most",
    0.05,
    "1",
    (PlannedValue("error indicator (Norm)", "error-indicators.csv", None, ("Norm",)),),
)
_MA = PlannedCriterion(
    "MA participation",
    "surface_epr",
    "relative_change",
    0.02,
    "1",
    (
        PlannedValue(
            "MA participation, excitation a",
            "surface-Q.csv",
            TableRow("i", 1),
            ("p_surf[1]",),
        ),
    ),
)
_FREQUENCY = PlannedCriterion(
    "frequency, every mode",
    "mode_frequency",
    "relative_change",
    1e-5,
    "GHz",
    (PlannedValue("frequency, mode 1", "eig.csv", TableRow("m", 1), ("Re{f} (GHz)",)),),
)
_EIG_HEADER = "        m,                Re{f} (GHz),                Im{f} (GHz),                          Q,              Error (Bkwd.),               Error (Abs.)\n"
_PORT_HEADER = "        m,                       p[1]\n"


def _cell(value: float) -> str:
    return f"{value:+27.12e}"


def _electrostatic(  # noqa: PLR0913 -- one solve and the four values a case varies
    folder: Path,
    iteration: object,
    *,
    capacitance: float = 8.934231401794e-14,
    norm: float = 0.01,
    participation: float = 1.0e-3,
    dofs: int = 1000,
) -> None:
    """One Electrostatic solve as Palace writes it; ``iteration`` is its ``Problem.Iteration`` (None: none)."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "terminal-C.csv").write_text(
        f"        i,{'C[i][1] (F)':>27}\n 1.00e+00,{_cell(capacitance)}\n",
        encoding="utf-8",
    )
    (folder / "error-indicators.csv").write_text(
        f"{'Norm':>27},{'Minimum':>27},{'Maximum':>27},{'Mean':>27}\n"
        f"{_cell(norm)},{_cell(1e-10)},{_cell(norm)},{_cell(norm / 10)}\n",
        encoding="utf-8",
    )
    (folder / "surface-Q.csv").write_text(
        f"        i,{'p_surf[1]':>27},{'Q_surf[1]':>27}\n"
        f" 1.00e+00,{_cell(participation)},{'+inf':>27}\n",
        encoding="utf-8",
    )
    _metadata(folder, iteration, dofs)


def _eigenmode(
    folder: Path, iteration: object, *, frequency: float = 5.121852257155
) -> None:
    """One Eigenmode solve as Palace writes it."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "eig.csv").write_text(
        _EIG_HEADER
        + f" 1.00e+00,{_cell(frequency)},{_cell(6.1e-7)},{_cell(4.17e6)},{_cell(5.7e-11)},{_cell(4.4e-5)}\n",
        encoding="utf-8",
    )
    (folder / "port-EPR.csv").write_text(
        _PORT_HEADER + f" 1.00e+00,{_cell(-0.9939)}\n", encoding="utf-8"
    )
    (folder / "error-indicators.csv").write_text(
        f"{'Norm':>27},{'Minimum':>27},{'Maximum':>27},{'Mean':>27}\n"
        f"{_cell(0.01)},{_cell(1e-10)},{_cell(0.01)},{_cell(0.001)}\n",
        encoding="utf-8",
    )
    _metadata(folder, iteration, 1000)


def _metadata(folder: Path, iteration: object, dofs: int) -> None:
    if iteration is None:
        return
    problem: dict[str, object] = {"DegreesOfFreedom": dofs, "MeshElements": dofs // 10}
    if iteration != "absent":
        problem["Iteration"] = iteration
    (folder / "palace.json").write_text(
        json.dumps({"Problem": problem}), encoding="utf-8"
    )


def _link_top_level(results: Path, iteration: str, *, next_dofs: int = 5000) -> None:
    """Palace's move after solve k: links at the top level into ``iteration<k>/``, a new ``palace.json``."""
    for table in sorted((results / iteration).glob("*.csv")):
        os.symlink(f"{iteration}/{table.name}", results / table.name)
    recorded = json.loads(
        (results / iteration / "palace.json").read_text(encoding="utf-8")
    )
    recorded["Problem"]["DegreesOfFreedom"] = next_dofs
    (results / "palace.json").write_text(json.dumps(recorded), encoding="utf-8")


def _results(tmp_path: Path) -> Path:
    return tmp_path / "results/palace"


def _check(
    tmp_path: Path,
    *criteria: PlannedCriterion,
    in_a_row: int = 1,
    problem: str = "Electrostatic",
) -> ConvergenceCheck:
    plan = ConvergencePlan(problem, criteria or (_DIAGONAL,), in_a_row)
    return check_passes(plan, tmp_path)


def _numbers(check: ConvergenceCheck) -> list[int]:
    return [item.number for item in check.passes]


def _reasons(check: ConvergenceCheck, number: int) -> set[str | None]:
    item = next(item for item in check.passes if item.number == number)
    return {
        value.unavailable for criterion in item.criteria for value in criterion.values
    }


def _snapshots(tmp_path: Path, problem: str = "Electrostatic"):
    return collect_amr_passes(tmp_path, problem, None, None)


# ---- Layouts Palace itself leaves ---------------------------------------------------------------------------


def test_a_run_that_ended_by_itself_holds_every_solve(tmp_path):
    """iteration1..3 and a regular top level recording 4: four passes, the top level read as pass 4."""
    results = _results(tmp_path)
    for number in (1, 2, 3):
        _electrostatic(results / f"iteration{number}", number)
    _electrostatic(results, 4)
    returned = returned_passes(results)
    assert returned.final == "pass"
    assert [(item.number, item.path.name) for item in returned.passes] == [
        (1, "iteration1"),
        (2, "iteration2"),
        (3, "iteration3"),
        (4, "palace"),
    ]
    check = _check(tmp_path)
    assert (check.verdict.state, check.verdict.pass_number) == ("converged", 2)
    collected = _snapshots(tmp_path)
    assert [item.pass_number for item in collected.passes] == [1, 2, 3, 4]
    assert collected.passes[-1].source == "final"
    assert collected.final_snapshot_status == "readable"


def test_an_electrostatic_run_stopped_after_a_pass_holds_no_solve_at_the_top(tmp_path):
    """Every top-level table a link into iteration2: the top level is pass 2's files, not a pass of its own."""
    results = _results(tmp_path)
    _electrostatic(results / "iteration1", 1)
    _electrostatic(results / "iteration2", 2)
    _link_top_level(results, "iteration2")
    returned = returned_passes(results)
    assert returned.final == "linked"
    assert (
        returned.final_reason == "the final result's tables are links into iteration2"
    )
    assert returned.set_aside == ()
    check = _check(tmp_path)
    assert _numbers(check) == [1, 2]
    assert (check.verdict.state, check.verdict.pass_number) == ("converged", 2)
    collected = _snapshots(tmp_path)
    assert [item.source for item in collected.passes] == ["iteration1", "iteration2"]
    # The top-level palace.json already counts the next mesh; the iteration's own record is the one read.
    assert collected.passes[-1].degrees_of_freedom == 1000
    assert collected.final_snapshot_status == "unreadable"


def test_an_electrostatic_run_stopped_during_a_solve_leaves_an_unfinished_top_level(
    tmp_path,
):
    """Links into iteration2 mixed with the next solve's new files: a solve under way, set aside."""
    results = _results(tmp_path)
    _electrostatic(results / "iteration1", 1)
    _electrostatic(results / "iteration2", 2)
    _link_top_level(results, "iteration2")
    (results / "surface-Q.csv").unlink()
    (results / "surface-Q.csv").write_text(
        f"        i,{'p_surf[1]':>27},{'Q_surf[1]':>27}\n 1.00e+00,{_cell(2e-3)},{'+inf':>27}\n",
        encoding="utf-8",
    )
    check = _check(tmp_path, _DIAGONAL, _MA)
    assert _numbers(check) == [1, 2]
    assert [item.reason for item in check.set_aside] == [
        "the final result is a solve under way: its tables mix links into iteration folders with new files"
    ]
    assert (
        "Not read as a pass: the final result is a solve under way"
        in verdict_sentence(check)
    )
    assert _snapshots(tmp_path).final_snapshot_status == "unreadable"


def test_an_eigenmode_run_stopped_during_a_solve_has_header_only_tables(tmp_path):
    """The next solve has replaced eig.csv and port-EPR.csv by their headers: unfinished, never pass 3."""
    results = _results(tmp_path)
    _eigenmode(results / "iteration1", 1)
    _eigenmode(results / "iteration2", 2)
    _link_top_level(results, "iteration2")
    for name, header in (("eig.csv", _EIG_HEADER), ("port-EPR.csv", _PORT_HEADER)):
        (results / name).unlink()
        (results / name).write_text(header, encoding="utf-8")
    shutil.rmtree(results / "iteration1")
    for link in [path for path in results.glob("*.csv") if path.is_symlink()]:
        link.unlink()
    returned = returned_passes(results)
    assert returned.final == "unfinished"
    assert returned.final_reason == (
        "the final result is a solve under way: ['eig.csv', 'port-EPR.csv'] hold only their header"
    )
    check = _check(tmp_path, _FREQUENCY, problem="Eigenmode")
    assert _numbers(check) == [2]
    assert check.verdict.state == "not_converged"


def test_an_eigenmode_run_stopped_after_a_pass_reads_its_iterations(tmp_path):
    """Eigenmode's linked top level, as Palace leaves it right after pass 2."""
    results = _results(tmp_path)
    _eigenmode(results / "iteration1", 1)
    _eigenmode(results / "iteration2", 2)
    _link_top_level(results, "iteration2")
    check = _check(tmp_path, _FREQUENCY, problem="Eigenmode")
    assert _numbers(check) == [1, 2]
    assert (check.verdict.state, check.verdict.pass_number) == ("converged", 2)
    assert returned_passes(results).final == "linked"


def test_a_stopped_run_copied_without_links_is_its_iterations_once(tmp_path):
    """A tool that follows links copies the linked tables: equal bytes, so pass 2 once, from iteration2."""
    results = _results(tmp_path)
    _electrostatic(results / "iteration1", 1)
    _electrostatic(results / "iteration2", 2)
    _link_top_level(results, "iteration2")
    for link in [path for path in results.glob("*.csv") if path.is_symlink()]:
        target = link.resolve()
        link.unlink()
        shutil.copyfile(target, link)
    returned = returned_passes(results)
    assert returned.final == "copy"
    assert [(item.number, item.path.name) for item in returned.passes] == [
        (1, "iteration1"),
        (2, "iteration2"),
    ]
    check = _check(tmp_path)
    assert (check.verdict.state, check.verdict.pass_number) == ("converged", 2)
    collected = _snapshots(tmp_path)
    assert collected.passes[-1].source == "iteration2"
    assert collected.passes[-1].degrees_of_freedom == 1000


# ---- Review round 3: B5 and B6 -------------------------------------------------------------------------------


def test_a_final_result_beside_a_misnamed_iteration_is_not_compared_with_it(tmp_path):
    """B5: iteration2 records pass 3, like the final result; iteration2 is set aside, pass 3 has no neighbour."""
    results = _results(tmp_path)
    _electrostatic(results / "iteration1", 1)
    _electrostatic(results / "iteration2", 3)
    _electrostatic(results, 3)
    check = _check(tmp_path)
    assert _numbers(check) == [1, 3]
    assert [item.reason for item in check.set_aside] == [
        "iteration2 records Problem.Iteration 3, not 2"
    ]
    assert _reasons(check, 3) == {
        "no previous pass to compare with: pass 2 was not read"
    }
    assert check.verdict.state == "not_converged"


@pytest.mark.parametrize(
    ("conflict", "criteria", "differing"),
    [
        ({"norm": 0.1}, (_INDICATOR,), "['error-indicators.csv']"),
        ({"participation": 5.0e-3}, (_MA,), "['surface-Q.csv']"),
    ],
    ids=["indicator", "whole-surface participation"],
)
def test_two_readings_of_one_pass_that_conflict_are_not_used(
    tmp_path, conflict, criteria, differing
):
    """B6: iteration2 and the final result both record pass 2 with regular files that differ."""
    results = _results(tmp_path)
    _electrostatic(results / "iteration1", 1, norm=0.2, participation=2e-3)
    _electrostatic(results / "iteration2", 2, norm=0.01)
    _electrostatic(results, 2, **conflict)
    check = _check(tmp_path, *criteria)
    assert _numbers(check) == [1, 2]
    second = check.passes[1]
    reason = f"pass 2 is recorded by iteration2 and the final result, whose tables differ: {differing}"
    assert second.identity_problem == reason
    assert not second.holds and second.streak == 0
    assert _reasons(check, 2) == {reason}
    assert check.verdict.state == "not_converged"
    assert "Pass 2?" in pass_table_html(check)
    assert [item.pass_number for item in _snapshots(tmp_path).passes] == [1]


# ---- The family of round 2 under this rule ------------------------------------------------------------------


def test_the_final_result_without_the_iterations_before_it_is_not_compared(tmp_path):
    """iteration1 and the final result of pass 4: passes 1 and 4, no comparison between them."""
    results = _results(tmp_path)
    _electrostatic(results / "iteration1", 1)
    _electrostatic(results, 4)
    check = _check(tmp_path)
    assert _numbers(check) == [1, 4]
    assert _reasons(check, 4) == {
        "no previous pass to compare with: pass 3 was not read"
    }
    assert check.verdict.state == "not_converged"
    assert "Passes 2 and 3 were not read." in verdict_sentence(check)


def test_a_missing_iteration_in_the_middle_breaks_the_comparison(tmp_path):
    """Passes 1, 3, 4: pass 3 has no neighbour; 4 compares with 3."""
    results = _results(tmp_path)
    _electrostatic(results / "iteration1", 1)
    _electrostatic(results / "iteration3", 3)
    _electrostatic(results, 4)
    check = _check(tmp_path, in_a_row=2)
    assert [item.streak for item in check.passes] == [0, 0, 1]
    assert check.verdict.state == "not_converged"
    check = _check(tmp_path)
    assert (check.verdict.state, check.verdict.pass_number) == ("converged", 4)
    assert verdict_sentence(check).endswith(
        "Pass 2 was not read, so the run may have converged earlier."
    )


def test_the_first_iterations_missing_start_the_comparisons_later(tmp_path):
    """Passes 3, 4, 5: real comparisons; the verdict says passes 1 and 2 were not read."""
    results = _results(tmp_path)
    _electrostatic(results / "iteration3", 3)
    _electrostatic(results / "iteration4", 4)
    _electrostatic(results, 5)
    check = _check(tmp_path, in_a_row=2)
    assert (check.verdict.state, check.verdict.pass_number) == ("converged", 5)
    assert "Passes 1 and 2 were not read, so the run may have converged earlier." in (
        verdict_sentence(check)
    )


def test_two_iteration_folders_naming_one_pass(tmp_path):
    """iteration2 and iteration02 record pass 2: one pass when their tables are equal, unreadable otherwise."""
    results = _results(tmp_path)
    _electrostatic(results / "iteration1", 1)
    _electrostatic(results / "iteration2", 2)
    _electrostatic(results / "iteration02", 2)
    _electrostatic(results, 3)
    equal = returned_passes(results)
    assert [item.number for item in equal.passes] == [1, 2, 3]
    assert equal.passes[1].identity_problem is None
    assert equal.final == "pass"
    _electrostatic(results / "iteration02", 2, capacitance=9e-14)
    check = _check(tmp_path, in_a_row=2)
    assert check.passes[1].identity_problem == (
        "pass 2 is recorded by iteration02 and iteration2, whose tables differ: ['terminal-C.csv']"
    )
    assert check.verdict.state == "not_converged"


@pytest.mark.parametrize(
    ("record", "reason"),
    [
        (None, "iteration2 has no palace.json of its own"),
        ("absent", "iteration2 has a palace.json that records no Problem.Iteration"),
        ("2", "iteration2 has a palace.json that records no Problem.Iteration"),
        (0, "iteration2 has a palace.json that records no Problem.Iteration"),
        (3, "iteration2 records Problem.Iteration 3, not 2"),
    ],
    ids=[
        "no palace.json",
        "no Iteration",
        "not an integer",
        "not positive",
        "disagrees with name",
    ],
)
def test_a_folder_whose_pass_is_not_established_is_not_read_as_a_pass(
    tmp_path, record, reason
):
    """iteration2 is set aside with the reason; pass 3 has no neighbour; no number is invented for it."""
    results = _results(tmp_path)
    _electrostatic(results / "iteration1", 1)
    _electrostatic(results / "iteration2", record)
    _electrostatic(results / "iteration3", 3)
    _electrostatic(results, 4)
    check = _check(tmp_path, _DIAGONAL, _INDICATOR)
    assert _numbers(check) == [1, 3, 4]
    assert [item.reason for item in check.set_aside] == [reason]
    assert _reasons(check, 3) - {None} == {
        "no previous pass to compare with: pass 2 was not read"
    }
    assert (check.verdict.state, check.verdict.pass_number) == ("converged", 4)
    assert f"Not read as a pass: {reason}." in verdict_sentence(check)


def test_a_final_result_that_records_no_iteration_is_set_aside(tmp_path):
    """Regular top-level tables but no Problem.Iteration: no pass number is established for it."""
    results = _results(tmp_path)
    _electrostatic(results / "iteration1", 1)
    _electrostatic(results, "absent")
    returned = returned_passes(results)
    assert returned.final == "set_aside"
    assert [item.reason for item in returned.set_aside] == [
        "the final result has a palace.json that records no Problem.Iteration"
    ]
    assert [item.number for item in returned.passes] == [1]


def test_a_table_missing_in_one_pass_makes_that_pass_and_the_next_change_unavailable(
    tmp_path,
):
    """iteration2 lacks terminal-C.csv: its number is unavailable, and pass 3 has nothing to compare with."""
    results = _results(tmp_path)
    for number in (1, 2, 3):
        _electrostatic(results / f"iteration{number}", number)
    _electrostatic(results, 4)
    (results / "iteration2" / "terminal-C.csv").unlink()
    check = _check(tmp_path, in_a_row=2)
    assert _reasons(check, 2) == {"terminal-C.csv is missing"}
    assert _reasons(check, 3) == {"previous pass: terminal-C.csv is missing"}
    assert [item.streak for item in check.passes] == [0, 0, 0, 1]
    assert check.verdict.state == "not_converged"


def test_without_the_final_result_the_iterations_are_the_passes_read(tmp_path):
    """No top-level table: the verdict is over the iterations read, and names them."""
    results = _results(tmp_path)
    for number in (1, 2, 3):
        _electrostatic(results / f"iteration{number}", number)
    check = _check(tmp_path, in_a_row=2)
    assert _numbers(check) == [1, 2, 3]
    assert returned_passes(results).final == "missing"
    assert verdict_sentence(check).startswith(
        "Converged at pass 3, of the 3 passes read (passes 1 to 3)"
    )
    assert _snapshots(tmp_path).final_snapshot_status == "missing"
