"""Which pass each returned folder is, and what the convergence check reports when the folder set is incomplete.

Every case is a ``results/palace`` written in Palace's formats: ``iteration<N>`` folders and the final result,
each with ``terminal-C.csv``, ``error-indicators.csv`` and a ``palace.json`` that records the pass as Palace
0.18.1 does (``Problem.Iteration``: 1 for the solve on the input mesh).  The passes come from
``collect_amr_passes``, the one numbering rule, and the check from ``check_passes``, as result reading runs
them.  The capacitance is the same in every pass, so a complete folder converges at pass 2; none of these
partial or malformed folders may report a convergence that rests on a comparison it could not make.
No solver runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scgsim.contracts.palace_convergence import (
    ConvergenceCheck,
    ConvergencePlan,
    PlannedCriterion,
    PlannedValue,
    TableRow,
)
from scgsim.palace.reports.convergence import verdict_sentence
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


def _solve(folder: Path, iteration: object, *, capacitance: float = 1e-13) -> None:
    """One solve's tables; ``iteration`` is what its ``palace.json`` records (None: no ``palace.json``)."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "terminal-C.csv").write_text(
        f"i,C[i][1] (F)\n1,{capacitance:+.12e}\n", encoding="utf-8"
    )
    (folder / "error-indicators.csv").write_text(
        "Norm,Minimum,Maximum,Mean\n0.01,1e-9,0.01,0.001\n", encoding="utf-8"
    )
    if iteration is not None:
        (folder / "palace.json").write_text(
            json.dumps({"Problem": {"Iteration": iteration}}), encoding="utf-8"
        )


def _check(
    root: Path, *criteria: PlannedCriterion, in_a_row: int = 1
) -> ConvergenceCheck:
    plan = ConvergencePlan("Electrostatic", criteria or (_DIAGONAL,), in_a_row)
    passes = collect_amr_passes(
        root, "Electrostatic", None, None, failed_attempt=False
    ).passes
    return check_passes(plan, passes)


def _results(tmp_path: Path) -> Path:
    return tmp_path / "results/palace"


def _reasons(check: ConvergenceCheck, number: int) -> set[str | None]:
    item = next(item for item in check.passes if item.number == number)
    return {
        value.unavailable for criterion in item.criteria for value in criterion.values
    }


def test_a_complete_folder_converges_where_the_change_first_holds(tmp_path):
    """The reference: passes 1 to 4 read, unchanged from pass 2 on."""
    results = _results(tmp_path)
    for number in (1, 2, 3):
        _solve(results / f"iteration{number}", number)
    _solve(results, 4)
    check = _check(tmp_path)
    assert [item.number for item in check.passes] == [1, 2, 3, 4]
    assert (check.verdict.state, check.verdict.pass_number) == ("converged", 2)


def test_the_final_result_without_the_iterations_before_it_is_not_compared(tmp_path):
    """The reviewer's case: iteration1 and the final result of pass 4 are not neighbours (B4)."""
    results = _results(tmp_path)
    _solve(results / "iteration1", 1)
    _solve(results, 4)
    snapshots = collect_amr_passes(
        tmp_path, "Electrostatic", None, None, failed_attempt=False
    ).passes
    assert [item.pass_number for item in snapshots] == [1, 4]
    check = _check(tmp_path)
    assert check.verdict.state == "not_converged"
    assert _reasons(check, 4) == {
        "no previous pass to compare with: pass 3 was not read"
    }
    assert "Passes 2 and 3 were not read." in verdict_sentence(check)


def test_a_missing_iteration_in_the_middle_breaks_the_comparison(tmp_path):
    """Passes 1, 3, 4: pass 3 has no neighbour; 4 compares with 3, so it alone can hold."""
    results = _results(tmp_path)
    _solve(results / "iteration1", 1)
    _solve(results / "iteration3", 3)
    _solve(results, 4)
    check = _check(tmp_path, in_a_row=2)
    assert [item.streak for item in check.passes] == [0, 0, 1]
    assert _reasons(check, 3) == {
        "no previous pass to compare with: pass 2 was not read"
    }
    assert check.verdict.state == "not_converged"
    check = _check(tmp_path)
    assert (check.verdict.state, check.verdict.pass_number) == ("converged", 4)
    assert verdict_sentence(check).endswith(
        "Pass 2 was not read, so the run may have converged earlier."
    )


def test_the_first_iterations_missing_start_the_comparisons_later(tmp_path):
    """Passes 3, 4, 5: the comparisons 3-4 and 4-5 are real; the verdict says passes 1 and 2 were not read."""
    results = _results(tmp_path)
    _solve(results / "iteration3", 3)
    _solve(results / "iteration4", 4)
    _solve(results, 5)
    check = _check(tmp_path, in_a_row=2)
    assert [item.number for item in check.passes] == [3, 4, 5]
    assert (check.verdict.state, check.verdict.pass_number) == ("converged", 5)
    assert "Passes 1 and 2 were not read, so the run may have converged earlier." in (
        verdict_sentence(check)
    )


def test_a_final_result_that_repeats_the_last_iteration_is_one_pass(tmp_path):
    """The existing rule: the same recorded pass and the same physics is the same solve, read once."""
    results = _results(tmp_path)
    _solve(results / "iteration1", 1)
    _solve(results / "iteration2", 2)
    _solve(results, 2)
    check = _check(tmp_path)
    assert [item.number for item in check.passes] == [1, 2]
    assert (check.verdict.state, check.verdict.pass_number) == ("converged", 2)


def test_two_folders_recording_one_pass_with_different_numbers_are_both_unreadable(
    tmp_path,
):
    """The final result records pass 2 like iteration2 but differs from it: neither is read as pass 2."""
    results = _results(tmp_path)
    _solve(results / "iteration1", 1)
    _solve(results / "iteration2", 2)
    _solve(results, 2, capacitance=2e-13)
    check = _check(tmp_path, _INDICATOR)
    assert [item.number for item in check.passes] == [1, 2, 2]
    twins = check.passes[1:]
    assert all(not item.holds and item.streak == 0 for item in twins)
    assert _reasons(check, 2) == {
        "pass 2 is recorded by 2 folders (iteration2, palace)"
    }
    assert (check.verdict.state, check.verdict.pass_number) == ("converged", 1)


def test_two_iteration_folders_naming_one_pass_are_both_unreadable(tmp_path):
    """``iteration2`` and ``iteration02`` both record pass 2."""
    results = _results(tmp_path)
    _solve(results / "iteration1", 1)
    _solve(results / "iteration2", 2)
    _solve(results / "iteration02", 2)
    _solve(results, 3)
    check = _check(tmp_path, in_a_row=2)
    assert [item.number for item in check.passes].count(2) == 2
    assert {item.identity_problem for item in check.passes if item.number == 2} == {
        "pass 2 is recorded by 2 folders (iteration02, iteration2)"
    }
    assert check.verdict.state == "not_converged"


@pytest.mark.parametrize(
    ("record", "reason"),
    [
        (None, "iteration2 has no palace.json"),
        ("2", "iteration2: its palace.json records no Problem.Iteration"),
        (0, "iteration2: its palace.json records no Problem.Iteration"),
        (3, "iteration2 records Problem.Iteration 3, not 2"),
    ],
    ids=["no palace.json", "not an integer", "not positive", "disagrees with name"],
)
def test_a_pass_whose_number_is_not_established_is_in_no_comparison(
    tmp_path, record, reason
):
    """iteration2 keeps its place but no change is measured to or from it, and it is in no run in a row."""
    results = _results(tmp_path)
    _solve(results / "iteration1", 1)
    _solve(results / "iteration2", record)
    _solve(results / "iteration3", 3)
    _solve(results, 4)
    check = _check(tmp_path, _DIAGONAL, _INDICATOR)
    second, third = check.passes[1], check.passes[2]
    assert second.identity_problem == reason
    assert second.streak == 0
    assert second.criteria[1].holds  # a value criterion needs no neighbour
    assert second.criteria[0].values[0].unavailable == (
        f"no previous pass to compare with: {reason}"
    )
    assert third.criteria[0].values[0].unavailable == (
        f"no previous pass to compare with: {reason}"
    )
    assert (check.verdict.state, check.verdict.pass_number) == ("converged", 4)
    assert f"iteration2 is in no comparison: {reason}." in verdict_sentence(check)


def test_a_table_missing_in_one_pass_makes_that_pass_and_the_next_change_unavailable(
    tmp_path,
):
    """iteration2 lacks terminal-C.csv: its number is unavailable, and pass 3 has nothing to compare with."""
    results = _results(tmp_path)
    for number in (1, 2, 3):
        _solve(results / f"iteration{number}", number)
    _solve(results, 4)
    (results / "iteration2" / "terminal-C.csv").unlink()
    check = _check(tmp_path, in_a_row=2)
    assert _reasons(check, 2) == {"terminal-C.csv is missing"}
    assert _reasons(check, 3) == {"previous pass: terminal-C.csv is missing"}
    assert [item.streak for item in check.passes] == [0, 0, 0, 1]
    assert check.verdict.state == "not_converged"


def test_without_the_final_result_the_iterations_are_the_passes_read(tmp_path):
    """The final result is not there: the verdict is over the iterations read, and names them."""
    results = _results(tmp_path)
    for number in (1, 2, 3):
        _solve(results / f"iteration{number}", number)
    check = _check(tmp_path, in_a_row=2)
    assert [item.number for item in check.passes] == [1, 2, 3]
    assert (check.verdict.state, check.verdict.pass_number) == ("converged", 3)
    assert verdict_sentence(check).startswith(
        "Converged at pass 3, of the 3 passes read (passes 1 to 3)"
    )
