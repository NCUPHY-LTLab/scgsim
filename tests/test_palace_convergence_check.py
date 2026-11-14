"""The convergence check on pass folders written in Palace's CSV formats: values, changes, holds and the verdict.

Every table here has the headers Palace writes (the ones the result readers validate); no solver runs.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

from scgsim.contracts import palace_convergence
from scgsim.contracts.palace_convergence import (
    ConvergencePlan,
    PassFolder,
    PlannedCriterion,
    PlannedValue,
    TableRow,
    check_convergence,
)

_TERMINAL_C = "i,C[i][1] (F),C[i][2] (F)\n"
_EIG = "m,Re{f} (GHz),Im{f} (GHz),Q,Error (Bkwd.),Error (Abs.)\n"
_ERROR = "Norm,Minimum,Maximum,Mean\n"


def _capacitance_pass(folder: Path, c11: float, c12: float, c22: float) -> Path:
    """A pass folder with a two-terminal Maxwell matrix, padded as Palace pads its columns."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "terminal-C.csv").write_text(
        _TERMINAL_C
        + f"   1.000000000e+00, {c11:+.9e}, {c12:+.9e}\n"
        + f"   2.000000000e+00, {c12:+.9e}, {c22:+.9e}\n",
        encoding="utf-8",
    )
    return folder


def _error_pass(folder: Path, norm: float) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "error-indicators.csv").write_text(
        _ERROR + f"{norm:+.9e},1e-9,{norm:+.9e},{norm / 10:+.9e}\n", encoding="utf-8"
    )
    return folder


def _criterion(
    label: str, values: tuple[PlannedValue, ...], test: str, threshold: float
) -> PlannedCriterion:
    quantity = {
        "terminal-C.csv": "capacitance",
        "eig.csv": "mode_frequency",
        "surface-Q.csv": "surface_epr",
        "error-indicators.csv": "error_indicator",
    }[values[0].table]
    unit = {"capacitance": "F", "mode_frequency": "GHz"}.get(quantity, "1")
    return PlannedCriterion(label, quantity, test, threshold, unit, values)  # type: ignore[arg-type]


def _diagonal(test: str = "percent_change", threshold: float = 0.1) -> PlannedCriterion:
    return _criterion(
        "C diagonal",
        (
            PlannedValue(
                "C[a][a]", "terminal-C.csv", TableRow("i", 1), ("C[i][1] (F)",)
            ),
            PlannedValue(
                "C[b][b]", "terminal-C.csv", TableRow("i", 2), ("C[i][2] (F)",)
            ),
        ),
        test,
        threshold,
    )


def _error(threshold: float) -> PlannedCriterion:
    return _criterion(
        "error indicator (Norm)",
        (
            PlannedValue(
                "error indicator (Norm)", "error-indicators.csv", None, ("Norm",)
            ),
        ),
        "at_most",
        threshold,
    )


def _plan(*criteria: PlannedCriterion, in_a_row: int = 1) -> ConvergencePlan:
    return ConvergencePlan("Electrostatic", criteria, in_a_row)


def _passes(
    tmp_path: Path, matrices: list[tuple[float, float, float]]
) -> list[PassFolder]:
    """Iteration folders for every matrix but the last, which is the final result folder."""
    folders = []
    for number, matrix in enumerate(matrices, start=1):
        last = number == len(matrices)
        folder = tmp_path / (
            "results/palace" if last else f"results/palace/iteration{number}"
        )
        folders.append(PassFolder(number, _capacitance_pass(folder, *matrix)))
    return folders


def test_a_percent_change_compares_each_pass_with_the_one_before(tmp_path):
    """The change at pass k is |C_k - C_(k-1)| / |C_k| for every element; pass 1 has none."""
    passes = _passes(
        tmp_path,
        [
            (100e-15, -5e-15, 80e-15),
            (101e-15, -5e-15, 80.4e-15),
            (101.05e-15, -5e-15, 80.41e-15),
        ],
    )
    check = check_convergence(_plan(_diagonal()), passes)
    first, second, third = (item.criteria[0] for item in check.passes)
    assert [value.measured for value in first.values] == [None, None]
    assert first.values[0].unavailable == "no pass before it"
    assert not first.holds
    assert second.values[0].measured == pytest.approx(100 * 1e-15 / 101e-15)
    assert second.worst.label == "C[a][a]"
    assert not second.holds
    assert third.values[0].measured == pytest.approx(100 * 0.05e-15 / 101.05e-15)
    assert third.holds
    assert check.verdict.state == "converged"
    assert check.verdict.pass_number == 3
    assert check.verdict.passes_read_after == 0


def test_an_absolute_change_is_in_the_unit_of_the_number(tmp_path):
    """``absolute_change`` bounds |C_k - C_(k-1)| in farads."""
    passes = _passes(tmp_path, [(100e-15, 0.0, 80e-15), (100.2e-15, 0.0, 80.1e-15)])
    check = check_convergence(_plan(_diagonal("absolute_change", 0.15e-15)), passes)
    second = check.passes[1].criteria[0]
    assert [value.measured for value in second.values] == pytest.approx(
        [0.2e-15, 0.1e-15]
    )
    assert [value.holds for value in second.values] == [False, True]
    assert not second.holds


def test_a_zero_value_holds_only_when_the_previous_one_is_zero_too(tmp_path):
    """Both zero: no change.  Only the new one zero: the percentage is undefined and fails."""
    off_diagonal = _criterion(
        "C[a][b]",
        (
            PlannedValue(
                "C[a][b]", "terminal-C.csv", TableRow("i", 1), ("C[i][2] (F)",)
            ),
        ),
        "percent_change",
        0.1,
    )
    passes = _passes(
        tmp_path, [(1e-15, 2e-18, 1e-15), (1e-15, 0.0, 1e-15), (1e-15, 0.0, 1e-15)]
    )
    check = check_convergence(_plan(off_diagonal), passes)
    undefined, unchanged = check.passes[1].criteria[0], check.passes[2].criteria[0]
    assert not undefined.holds
    assert (
        undefined.values[0].unavailable == "percent change undefined: the value is zero"
    )
    assert unchanged.holds
    assert unchanged.values[0].measured == 0.0


def test_minimum_converged_passes_counts_passes_in_a_row(tmp_path):
    """Converged at the first pass that ends a run of holding passes as long as declared."""
    folder = tmp_path / "results/palace"
    norms = [0.5, 0.01, 0.2, 0.01, 0.005, 0.004]
    passes = [
        PassFolder(number, _error_pass(folder / f"iteration{number}", norm))
        for number, norm in enumerate(norms, start=1)
    ]
    check = check_convergence(_plan(_error(0.02), in_a_row=2), passes)
    assert [item.streak for item in check.passes] == [0, 1, 0, 1, 2, 3]
    assert check.verdict.state == "converged"
    assert check.verdict.pass_number == 5
    assert check.verdict.passes_read_after == 1
    assert check.verdict.last_to_hold == ("error indicator (Norm)",)
    assert check.verdict.last_to_hold_from == 4


def test_the_last_criterion_to_hold_is_the_one_whose_run_began_last(tmp_path):
    """With two criteria, the verdict names the one that joined last and the pass it joined at."""
    matrices = [
        (100e-15, 0.0, 80e-15),
        (101e-15, 0.0, 80.01e-15),
        (101.01e-15, 0.0, 80.02e-15),
    ]
    passes = _passes(tmp_path, matrices)
    for number, norm in ((1, 0.1), (2, 0.01), (3, 0.01)):
        _error_pass(passes[number - 1].path, norm)
    check = check_convergence(_plan(_error(0.05), _diagonal()), passes)
    assert check.verdict.pass_number == 3
    assert check.verdict.last_to_hold == ("C diagonal",)
    assert check.verdict.last_to_hold_from == 3


def test_a_run_that_never_holds_names_what_fails_at_its_last_pass(tmp_path):
    """Not converged: the failing criteria at the last pass, each with its worst number."""
    passes = _passes(
        tmp_path,
        [(100e-15, 0.0, 80e-15), (101e-15, 0.0, 80e-15), (102e-15, 0.0, 80e-15)],
    )
    check = check_convergence(_plan(_diagonal()), passes)
    assert check.verdict.state == "not_converged"
    assert check.verdict.passes_read == 3
    (failing,) = check.verdict.failing
    assert failing.worst.label == "C[a][a]"
    assert failing.worst.measured == pytest.approx(100 * 1e-15 / 102e-15)


def test_a_missing_or_unreadable_number_does_not_hold(tmp_path):
    """A missing table, row or column is reported as unavailable and fails its criterion."""
    passes = _passes(tmp_path, [(1e-15, 0.0, 1e-15), (1e-15, 0.0, 1e-15)])
    surface = _criterion(
        "MA participation",
        (
            PlannedValue(
                "MA, excitation a", "surface-Q.csv", TableRow("i", 1), ("p_surf[1]",)
            ),
        ),
        "percent_change",
        1.0,
    )
    beyond = _criterion(
        "C[c][c]",
        (
            PlannedValue(
                "C[c][c]", "terminal-C.csv", TableRow("i", 3), ("C[i][3] (F)",)
            ),
        ),
        "percent_change",
        1.0,
    )
    check = check_convergence(_plan(surface, beyond), passes)
    missing, absent_row = check.passes[1].criteria
    assert missing.values[0].unavailable == "surface-Q.csv is missing"
    assert absent_row.values[0].unavailable == "terminal-C.csv has no row i = 3"
    assert not missing.holds and not absent_row.holds
    assert check.verdict.state == "not_converged"


def test_a_gap_in_the_passes_breaks_the_comparison_and_the_streak(tmp_path):
    """A pass whose previous pass was not read has no change to measure."""
    passes = _passes(tmp_path, [(1e-15, 0.0, 1e-15)] * 3)
    check = check_convergence(_plan(_diagonal()), [passes[0], passes[2]])
    assert check.passes[1].criteria[0].values[0].unavailable == (
        "no previous pass to compare with: pass 2 was not read"
    )
    assert check.verdict.state == "not_converged"


def test_a_participation_is_the_sum_of_its_columns(tmp_path):
    """The columns of one interface kind on one row are added; the other columns are not."""
    folder = tmp_path / "results/palace"
    folder.mkdir(parents=True)
    (folder / "surface-Q.csv").write_text(
        "m,p_surf[1],Q_surf[1],p_surf[2],Q_surf[2],p_surf[3],Q_surf[3]\n"
        "1,1.0e-3,+inf,2.0e-3,+inf,4.0e-3,+inf\n"
        "2,1.5e-3,+inf,2.5e-3,+inf,4.5e-3,+inf\n",
        encoding="utf-8",
    )
    ma = _criterion(
        "MA participation",
        (
            PlannedValue(
                "MA, mode 2",
                "surface-Q.csv",
                TableRow("m", 2),
                ("p_surf[1]", "p_surf[3]"),
            ),
        ),
        "at_most",
        1.0,
    )
    check = check_convergence(
        ConvergencePlan("Eigenmode", (ma,), 1), [PassFolder(1, folder)]
    )
    assert check.passes[0].criteria[0].values[0].value == pytest.approx(6.0e-3)


def test_a_frequency_is_read_by_mode_number(tmp_path):
    """Rows are found by their ``m``, not by their position."""
    folder = tmp_path / "results/palace"
    folder.mkdir(parents=True)
    (folder / "eig.csv").write_text(
        _EIG + "2,6.1,0.0,+inf,1e-12,1e-12\n1,5.2,0.0,+inf,1e-12,1e-12\n",
        encoding="utf-8",
    )
    frequency = _criterion(
        "frequency, mode 1",
        (
            PlannedValue(
                "frequency, mode 1", "eig.csv", TableRow("m", 1), ("Re{f} (GHz)",)
            ),
        ),
        "at_most",
        10.0,
    )
    check = check_convergence(
        ConvergencePlan("Eigenmode", (frequency,), 1), [PassFolder(1, folder)]
    )
    assert check.passes[0].criteria[0].values[0].value == 5.2


def test_no_pass_is_not_converged():
    """A folder with no pass read has no verdict but ``not_converged`` over zero passes."""
    check = check_convergence(_plan(_error(0.1)), [])
    assert check.verdict.state == "not_converged"
    assert check.verdict.passes_read == 0


def test_the_plan_payload_round_trips_and_names_its_schema():
    """``from_payload`` reads back what ``to_payload`` writes, with the schema name and version."""
    plan = _plan(_diagonal(), _error(0.02), in_a_row=2)
    payload = json.loads(json.dumps(plan.to_payload()))
    assert payload["schema"] == "scgsim.palace.convergence-plan"
    assert payload["schema_version"] == 1
    assert payload["criteria"][0]["values"][0]["combine"] == "sum"
    assert ConvergencePlan.from_payload(payload) == plan


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"schema_version": 2}, "this SCGSim reads"),
        ({"criteria": []}, "has no criterion"),
        ({"minimum_converged_passes": 0}, "integer of at least 1"),
    ],
)
def test_a_plan_of_another_shape_is_refused(change, message):
    """A plan this version does not write is refused, naming what to change."""
    payload = {**_plan(_error(0.02)).to_payload(), **change}
    with pytest.raises(ValueError, match=message):
        ConvergencePlan.from_payload(payload)


def test_a_criterion_with_an_unknown_test_or_threshold_is_refused():
    """The test must be one the check knows and the threshold positive and finite."""
    payload = _plan(_error(0.02)).to_payload()
    payload["criteria"][0]["test"] = "below"
    with pytest.raises(ValueError, match="quantity 'error_indicator' and test 'below'"):
        ConvergencePlan.from_payload(payload)
    payload["criteria"][0]["test"] = "at_most"
    payload["criteria"][0]["threshold"] = float("inf")
    with pytest.raises(ValueError, match="positive finite number"):
        ConvergencePlan.from_payload(payload)


def test_the_check_imports_only_the_standard_library():
    """The module can be copied into a run folder: no SCGSim import but the run-folder contract, no ``__future__``."""
    tree = ast.parse(Path(palace_convergence.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            if node.module == "scgsim.contracts.palace_run_folder":
                continue  # embedded in the same copy (scgsim.palace.run.standalone_copy)
            imported.add((node.module or "").split(".")[0])
    assert imported <= set(sys.stdlib_module_names)
    assert "__future__" not in imported


def test_two_rows_with_one_index_make_the_number_unavailable(tmp_path):
    """A second row ``i = 1`` is not ignored: the reading is ambiguous and never holds (round 1, B2)."""
    folders = _passes(tmp_path, [(1e-15, 0.0, 1e-15), (1e-15, 0.0, 1e-15)])
    table = folders[1].path / "terminal-C.csv"
    table.write_text(
        table.read_text(encoding="utf-8") + "1,9.0e-15,0.0\n", encoding="utf-8"
    )
    check = check_convergence(_plan(_diagonal()), folders)
    first = check.passes[1].criteria[0].values[0]
    assert first.value is None and not first.holds
    assert (
        first.unavailable == "terminal-C.csv has 2 rows i = 1, so the row is ambiguous"
    )
    assert check.verdict.state == "not_converged"


def test_a_repeated_column_makes_the_table_unavailable(tmp_path):
    """A table that repeats a header gives no number at all, with the reason (round 1, B2)."""
    folders = _passes(tmp_path, [(1e-15, 0.0, 1e-15), (1e-15, 0.0, 1e-15)])
    (folders[1].path / "terminal-C.csv").write_text(
        "i,C[i][1] (F),C[i][1] (F)\n1,1e-15,2e-15\n2,0.0,1e-15\n", encoding="utf-8"
    )
    check = check_convergence(_plan(_diagonal()), folders)
    values = check.passes[1].criteria[0].values
    assert all(value.value is None and not value.holds for value in values)
    assert values[0].unavailable == (
        "terminal-C.csv repeats the column(s) ['C[i][1] (F)'], so its cells are ambiguous"
    )


def test_passes_after_convergence_are_counted_among_the_passes_read(tmp_path):
    """Passes 1, 4 and 5 are three passes read; what comes after convergence is counted among them (B3).

    Palace's numbers would give 3 - 5 = -2 and 3 - 4 = -1 passes after.
    """
    folder = tmp_path / "results/palace"
    passes = [
        PassFolder(number, _error_pass(folder / f"iteration{number}", norm))
        for number, norm in ((1, 0.5), (4, 0.01), (5, 0.01))
    ]
    check = check_convergence(_plan(_error(0.02), in_a_row=2), passes)
    verdict = check.verdict
    assert (verdict.pass_number, verdict.passes_read, verdict.passes_read_after) == (
        5,
        3,
        0,
    )
    check = check_convergence(_plan(_error(0.02), in_a_row=1), passes)
    verdict = check.verdict
    assert (verdict.pass_number, verdict.passes_read, verdict.passes_read_after) == (
        4,
        3,
        1,
    )
