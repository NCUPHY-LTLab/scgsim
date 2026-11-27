"""Check Maximum Delta S against native-shaped Palace Driven pass tables."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from scgsim.contracts.palace_convergence import (
    ConvergencePlan,
    PassFolder,
    PlannedCriterion,
    PlannedValue,
    check_convergence,
    recorded_check,
)

_ENTRIES = ("S[1][1]", "S[2][1]", "S[1][2]", "S[2][2]")
_COLUMNS = tuple((f"|{entry}| (dB)", f"arg({entry}) (deg.)") for entry in _ENTRIES)
_HEADER = "f (GHz), " + ", ".join(column for pair in _COLUMNS for column in pair)
_ROWS = "5,0,0,-inf,0,-inf,0,0,0\n6,0,0,-inf,0,-inf,0,0,0\n"


def _plan(threshold: float = 0.5, streak: int = 1) -> ConvergencePlan:
    values = tuple(
        PlannedValue(entry, "port-S.csv", None, columns, "polar_db_deg", 2)
        for entry, columns in zip(_ENTRIES, _COLUMNS, strict=True)
    )
    criterion = PlannedCriterion(
        "Maximum Delta S", "s_matrix", "absolute_change", threshold, "1", values
    )
    return ConvergencePlan("Driven", (criterion,), streak)


def _make_pass(
    root: Path, number: int, rows: str = _ROWS, header: str = _HEADER
) -> PassFolder:
    folder = root / f"iteration{number}"
    folder.mkdir()
    (folder / "port-S.csv").write_text(f"{header}\n{rows}", encoding="utf-8")
    return PassFolder(number, folder)


def test_delta_s_reads_phase_and_takes_the_worst_frequency_and_matrix_entry(tmp_path):
    """A phase change at the second frequency can exceed a magnitude change in another entry."""
    previous = _make_pass(
        tmp_path,
        1,
        "5,0,0,-inf,0,-inf,0,0,0\n6,0,0,-6.020599913279624,0,-inf,0,0,0\n",
    )
    current = _make_pass(
        tmp_path,
        2,
        "5.000000000e+00,0,0,-inf,0,-20,0,0,0\n6,0,0,-6.020599913279624,90,-inf,0,0,0\n",
    )
    check = check_convergence(_plan(), [previous, current])
    second = check.passes[1].criteria[0]
    assert [value.measured for value in second.values] == pytest.approx(
        [0.0, math.sqrt(0.5), 0.1, 0.0]
    )
    assert second.worst.label == "S[2][1]"
    assert second.worst.value == pytest.approx(0.5)
    assert not second.holds
    assert check.verdict.state == "not_converged"
    json.dumps(recorded_check(check, tmp_path), allow_nan=False)


def test_zero_amplitude_at_the_threshold_requires_another_pass(tmp_path):
    """Negative infinity dB is zero and a change equal to the threshold has not converged."""
    first = _make_pass(tmp_path, 1)
    second = _make_pass(tmp_path, 2, _ROWS.replace("5,0,0", "5,-inf,180"))
    third = _make_pass(tmp_path, 3, _ROWS.replace("5,0,0", "5,-inf,-90"))
    fourth = _make_pass(tmp_path, 4, _ROWS.replace("5,0,0", "5,-inf,0"))
    check = check_convergence(
        _plan(threshold=1, streak=2), [first, second, third, fourth]
    )
    assert [item.streak for item in check.passes] == [0, 0, 1, 2]
    assert check.passes[1].criteria[0].worst.measured == 1
    assert check.passes[2].criteria[0].worst.measured == 0
    assert check.passes[2].criteria[0].values[1].value == 0
    assert check.verdict.pass_number == 4


def test_delta_s_compares_phases_across_the_angle_wrap(tmp_path):
    """The same scattering amplitude may have phases of +180 or -180 degrees."""
    first = _make_pass(tmp_path, 1, _ROWS.replace("5,0,0", "5,0,180"))
    second = _make_pass(tmp_path, 2, _ROWS.replace("5,0,0", "5,0,-180"))
    check = check_convergence(_plan(threshold=1e-17), [first, second])
    assert check.passes[1].criteria[0].worst.measured == 0
    assert check.verdict.pass_number == 2


@pytest.mark.parametrize(
    ("rows", "reason"),
    [
        (_ROWS.splitlines()[0], "has 1 frequency rows where 2 are expected"),
        (_ROWS + _ROWS, "has 4 frequency rows where 2 are expected"),
        (_ROWS.replace("6,", "5,"), "frequencies must be strictly increasing"),
        (_ROWS.replace("6,", "4,"), "frequencies must be strictly increasing"),
        (_ROWS.replace("6,", "0,"), "is not a positive finite frequency"),
        (_ROWS.replace("6,", "inf,"), "is not a positive finite frequency"),
        (_ROWS.replace("6,", "six,"), "is not a positive finite frequency"),
        (_ROWS.replace("-inf,0", "-inf,NaN"), "phase is not a finite number"),
        (_ROWS.replace("-inf,0", "+inf,0"), "magnitude in dB is not a finite number"),
    ],
)
def test_invalid_sweep_data_cannot_hold(tmp_path, rows, reason):
    """Every frequency and both components must be readable."""
    check = check_convergence(
        _plan(), [_make_pass(tmp_path, 1), _make_pass(tmp_path, 2, rows)]
    )
    worst = check.passes[1].criteria[0].worst
    assert worst.value is None
    assert worst.measured is None
    assert reason in worst.unavailable
    assert check.verdict.state == "not_converged"


def test_missing_phase_column_is_unavailable(tmp_path):
    """Both native columns must be present for each matrix entry."""
    check = check_convergence(
        _plan(),
        [
            _make_pass(tmp_path, 1),
            _make_pass(
                tmp_path, 2, header=_HEADER.replace("arg(S[1][1]) (deg.)", "other")
            ),
        ],
    )
    assert check.passes[1].criteria[0].worst.unavailable == (
        "port-S.csv has no column 'arg(S[1][1]) (deg.)'"
    )


def test_equal_row_counts_do_not_make_different_frequency_grids_comparable(tmp_path):
    """Measured frequencies must identify the same points, even below float precision."""
    first = _make_pass(tmp_path, 1)
    second = _make_pass(tmp_path, 2, _ROWS.replace("6,", "6.0000000000000001,"))
    check = check_convergence(_plan(), [first, second])
    worst = check.passes[1].criteria[0].worst
    assert worst.value == 1
    assert worst.measured is None
    assert (
        worst.unavailable
        == "port-S.csv 'f (GHz)' frequencies differ from the previous pass"
    )
    assert check.verdict.state == "not_converged"


def test_delta_s_compares_only_readable_consecutive_passes(tmp_path):
    """The first pass, a gap, and a previous incomplete sweep each prevent a change."""
    first = _make_pass(tmp_path, 1)
    third = _make_pass(tmp_path, 3, _ROWS.splitlines()[0])
    fourth = _make_pass(tmp_path, 4)
    fifth = _make_pass(tmp_path, 5)
    check = check_convergence(_plan(), [first, third, fourth, fifth])
    values = [item.criteria[0].worst for item in check.passes]
    assert values[0].unavailable == "no pass before it"
    assert values[2].unavailable == (
        "previous pass: port-S.csv has 1 frequency rows where 2 are expected"
    )
    assert values[3].measured == 0
    assert check.verdict.pass_number == 5

    check = check_convergence(_plan(), [first, fourth, fifth])
    assert check.passes[1].criteria[0].worst.unavailable == (
        "no previous pass to compare with: pass 3 was not read"
    )


def test_delta_s_plan_round_trips_with_the_existing_schema_version():
    """The PlannedValue holds the whole sweep of one matrix entry."""
    plan = _plan()
    payload = json.loads(json.dumps(plan.to_payload()))
    assert payload["schema_version"] == 2
    assert payload["criteria"][0]["values"][0] == {
        "label": "S[1][1]",
        "table": "port-S.csv",
        "row": None,
        "columns": ["|S[1][1]| (dB)", "arg(S[1][1]) (deg.)"],
        "combine": "polar_db_deg",
        "expected_rows": 2,
    }
    assert ConvergencePlan.from_payload(payload) == plan


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        (
            {"row": {"column": "f (GHz)", "index": 5}},
            "must read the full port-S.csv frequency sweep",
        ),
        ({"expected_rows": 0}, "expected_rows.*integer of at least 1"),
        ({"columns": list(reversed(_COLUMNS[0]))}, "native dB and degree columns"),
        ({"columns": [_COLUMNS[0][0], _COLUMNS[1][1]]}, "same S-parameter"),
    ],
)
def test_delta_s_plan_refuses_partial_or_invalid_sweeps(changes, reason):
    """A sealed S-parameter value must describe exactly one native column pair."""
    payload = _plan().to_payload()
    payload["criteria"][0]["values"][0].update(changes)
    with pytest.raises(ValueError, match=reason):
        ConvergencePlan.from_payload(payload)
