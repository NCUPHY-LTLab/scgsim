"""Stock Palace reports every converged mode; the readers keep the requested ones."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from scgsim.palace import inspect_run_trustworthiness
from scgsim.palace.results.snapshots import collect_amr_passes
from scgsim.palace.results.tables import configured_mode_count, read_result_table

# One lumped port, one energy domain and one Dielectric row, as Palace heads each table.
_HEADERS = {
    "eig": "m,Re{f} (GHz),Im{f} (GHz),Q,Error (Bkwd.),Error (Abs.)",
    "port-EPR": "m,p[1]",
    "port-I": "m,Re{I[1]} (A),Im{I[1]} (A)",
    "port-V": "m,Re{V[1]} (V),Im{V[1]} (V)",
    "domain-E": (
        "m,E_elec (J),E_mag (J),E_cap (J),E_ind (J),"
        "E_elec[1] (J),p_elec[1],E_mag[1] (J),p_mag[1]"
    ),
    "surface-Q": "m,p_surf[1],Q_surf[1]",
}
_COUNTS = {"lumped_port": 1, "domain": 1, "surface": 1, "terminal": 1}
_FAMILIES = sorted(_HEADERS)


def _config(modes: int) -> dict:
    return {"Solver": {"Eigenmode": {"N": modes}}}


def _value(mode: float, column: int) -> float:
    """A distinct finite value for one cell, so a kept row is known to be its own."""
    return mode + 0.125 * column


def _write(
    folder: Path,
    family: str,
    modes: Sequence[float],
    *,
    index: str = "m",
    spoiled: str | None = None,
) -> Path:
    """Write one table with a row per listed mode index, in Palace's number format.

    ``spoiled`` replaces every value of the last row, for a row nobody requested.
    """
    names = _HEADERS[family].split(",")
    lines = [",".join([index, *names[1:]])]
    for mode in modes:
        cells = [f"{_value(mode, column):+.12e}" for column in range(1, len(names))]
        lines.append(",".join([f"{mode:.2e}", *cells]))
    if spoiled is not None:
        lines[-1] = ",".join([f"{modes[-1]:.2e}", *[spoiled] * (len(names) - 1)])
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{family}.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("requested", [1, 2, 3])
@pytest.mark.parametrize("family", _FAMILIES)
def test_a_table_of_every_converged_mode_keeps_the_requested_modes(
    tmp_path, family, requested
):
    """Three converged modes for N requested ones give rows 1..N with their own values."""
    path = _write(tmp_path, family, [1, 2, 3])
    table = read_result_table(path, "Eigenmode", _COUNTS, _config(requested))
    assert [row["m"] for row in table.rows] == list(range(1, requested + 1))
    for row in table.rows:
        values = [row[name] for name in table.headers[1:]]
        assert values == [_value(row["m"], k) for k in range(1, len(table.headers))]


@pytest.mark.parametrize("family", _FAMILIES)
def test_fewer_modes_than_requested_are_refused(tmp_path, family):
    """A solve that converged two of three requested modes is not a complete result."""
    path = _write(tmp_path, family, [1, 2])
    with pytest.raises(ValueError, match=r"must contain 3 .*rows; found 2"):
        read_result_table(path, "Eigenmode", _COUNTS, _config(3))


@pytest.mark.parametrize(
    "modes",
    [[1, 3, 2], [1, 1, 2], [2, 3, 4], [1, 2, 4], [1, 2, 1, 2]],
    ids=["swapped", "repeated", "shifted", "skipped", "concatenated"],
)
@pytest.mark.parametrize("family", _FAMILIES)
def test_a_mode_index_that_is_not_one_to_k_is_refused(tmp_path, family, modes):
    """Extra rows are dropped only from a table whose whole index is 1..K in order."""
    path = _write(tmp_path, family, modes)
    with pytest.raises(ValueError, match=r"must contain 2 .*rows; found"):
        read_result_table(path, "Eigenmode", _COUNTS, _config(2))


@pytest.mark.parametrize("family", _FAMILIES)
def test_an_unrequested_mode_is_not_part_of_the_result(tmp_path, family):
    """Values Palace wrote for a mode nobody requested cannot refuse the requested ones."""
    path = _write(tmp_path, family, [1, 2, 3], spoiled="NaN")
    table = read_result_table(path, "Eigenmode", _COUNTS, _config(2))
    assert [row["m"] for row in table.rows] == [1, 2]
    with pytest.raises(ValueError, match="non-finite"):
        read_result_table(path, "Eigenmode", _COUNTS, _config(3))


def test_electrostatic_tables_keep_their_exact_row_count(tmp_path):
    """Only Eigenmode mode tables may hold more rows than the config asks for."""
    path = _write(tmp_path, "surface-Q", [1, 2], index="i")
    with pytest.raises(ValueError, match="must contain 1 rows; found 2"):
        read_result_table(path, "Electrostatic", _COUNTS, {})


def test_the_error_indicator_table_keeps_exactly_one_row(tmp_path):
    """The AMR indicator table is not a mode table, so a second row is still refused."""
    path = tmp_path / "error-indicators.csv"
    path.write_text(
        "Norm,Minimum,Maximum,Mean\n1e-2,1e-9,1e-4,1e-5\n2e-2,2e-9,2e-4,2e-5\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly one error indicator row"):
        read_result_table(path, "Eigenmode", _COUNTS, _config(1))


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (_config(2), 2),
        (None, None),
        ({}, None),
        ({"Solver": {"Electrostatic": {}}}, None),
        (_config(0), None),
        ({"Solver": {"Eigenmode": {"N": True}}}, None),
        ({"Solver": {"Eigenmode": {"N": "2"}}}, None),
    ],
)
def test_the_requested_mode_count_of_a_config_an_inspection_may_lack(config, expected):
    """A missing or malformed config gives no count instead of a guessed one."""
    assert configured_mode_count(config) == expected


def _eigenmode_run(root: Path, modes: Sequence[float]) -> None:
    """A returned Eigenmode folder with one archived AMR pass and the final pass."""
    results = root / "results" / "palace"
    for iteration, folder in ((1, results / "iteration1"), (2, results)):
        _write(folder, "eig", [mode + 0.5 * iteration for mode in modes])
        _reindex(folder / "eig.csv", modes)
        _write(folder, "port-EPR", modes)
        (folder / "palace.json").write_text(
            json.dumps({"Problem": {"Iteration": iteration}}), encoding="utf-8"
        )


def _reindex(path: Path, modes: Sequence[float]) -> None:
    """Restore the mode index of a table whose values were shifted to differ by pass."""
    header, *rows = path.read_text(encoding="utf-8").splitlines()
    kept = [
        ",".join([f"{mode:.2e}", *row.split(",")[1:]])
        for mode, row in zip(modes, rows, strict=True)
    ]
    path.write_text("\n".join([header, *kept]) + "\n", encoding="utf-8")


def test_each_amr_pass_keeps_the_requested_modes_when_the_count_is_known(tmp_path):
    """Archived and final passes both report N modes of the K that Palace wrote."""
    _eigenmode_run(tmp_path, [1, 2, 3])
    collected = collect_amr_passes(tmp_path, "Eigenmode", None, None, mode_count=1)
    assert [item.source for item in collected.passes] == ["iteration1", "final"]
    for iteration, item in enumerate(collected.passes, start=1):
        assert item.frequencies_ghz == (_value(1 + 0.5 * iteration, 1),)
        assert {len(trace) for trace in item.eig_columns.values()} == {1}
        assert item.port_epr == {"p[1]": (_value(1, 1),)}


def test_an_amr_pass_read_without_a_config_reports_the_rows_it_finds(tmp_path):
    """An inspection with no mode count keeps showing every row of the table."""
    _eigenmode_run(tmp_path, [1, 2, 3])
    collected = collect_amr_passes(tmp_path, "Eigenmode", None, None)
    assert [len(item.frequencies_ghz) for item in collected.passes] == [3, 3]


_BAD_INDEXES = {
    "swapped": [1, 3, 2],
    "reversed": [2, 1],
    "repeated": [1, 1, 2],
    "shifted": [2, 3],
    "insufficient": [1],
}


@pytest.mark.parametrize("modes", _BAD_INDEXES.values(), ids=_BAD_INDEXES.keys())
@pytest.mark.parametrize("family", ["eig", "port-EPR"])
def test_an_amr_pass_without_the_requested_modes_in_order_is_unreadable(
    tmp_path, family, modes
):
    """Either table alone makes its pass unreadable; nothing is read by row position."""
    _eigenmode_run(tmp_path, [1, 2, 3])
    results = tmp_path / "results" / "palace"
    _write(results / "iteration1", family, modes)
    collected = collect_amr_passes(tmp_path, "Eigenmode", None, None, mode_count=2)
    assert [item.source for item in collected.passes] == ["final"]
    assert collected.final_snapshot_status == "readable"
    _write(results, family, modes)
    collected = collect_amr_passes(tmp_path, "Eigenmode", None, None, mode_count=2)
    assert collected.passes == ()
    assert collected.final_snapshot_status == "unreadable"


def test_the_trust_report_of_a_run_with_extra_modes_shows_the_requested_mode(tmp_path):
    """The public inspector reads the run's own config for the requested count."""
    _eigenmode_run(tmp_path, [1, 2, 3])
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "palace_handoff_metadata.json").write_text(
        json.dumps(
            {
                "problem": "Eigenmode",
                "route": "A",
                "profile": "synthetic-extra-eigenpairs-test-only",
                "route_a_thin_film": None,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "config.json").write_text(json.dumps(_config(1)), encoding="utf-8")
    report = inspect_run_trustworthiness(tmp_path)
    assert [len(item.frequencies_ghz) for item in report.passes] == [1, 1]
