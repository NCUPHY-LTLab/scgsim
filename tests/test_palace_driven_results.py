"""Reading a returned Palace Driven run back: the strict tables, the S-parameters, Surface EPR and adaptive sampling.

The tables, ``palace.json`` and logs are the real outputs of stock Palace 0.18.1 in issue 187's E1
(``palace_driven_e1_support``): an order-1 uniform sweep of 5 points and an order-2 adaptive sweep of 81, on the
launcher-to-launcher layout.  Each refusal is shown on a copy of a real file with one change.  The run folder's
identity checks are stood in for here; ``test_palace_driven_launchers`` (PDK) runs ``resolve_palace_result`` on
a run folder of the same config.
"""

from __future__ import annotations

import cmath
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
from palace_driven_e1_support import (
    ADAPTIVE_RUN,
    LOG,
    UNIFORM_RUN,
    copied_run,
    fixture_result,
    read_json,
    read_touchstone,
)

from scgsim.contracts.palace_index_map import LUMPED_PORT_SECTION
from scgsim.contracts.palace_run_folder import CONFIG, INDEX_MAP, output_table
from scgsim.palace import (
    PalaceDrivenResult,
    SParameters,
    epr_result,
)

PORTS = ("chip/o_lumped_left", "chip/o_lumped_right")
SHEETS = tuple(f"PORT_SHEET__{name}__0000" for name in PORTS)
POINTS = {UNIFORM_RUN: 5, ADAPTIVE_RUN: 81}


def _result(tmp_path: Path, run: str) -> PalaceDrivenResult:
    return fixture_result(copied_run(run, tmp_path)).driven_result()


def _cells(root: Path, family: str) -> list[list[str]]:
    """The cells of one native table, the header row first, without Palace's padding."""
    text = (root / output_table(family)).read_text(encoding="utf-8")
    return [[cell.strip() for cell in line.split(",")] for line in text.splitlines()]


def _write_cells(root: Path, family: str, rows: list[list[str]]) -> None:
    (root / output_table(family)).write_text(
        "".join(",".join(row) + "\n" for row in rows), encoding="utf-8"
    )


def _rows(root: Path, family: str) -> list[dict[str, float]]:
    header, *rows = _cells(root, family)
    return [dict(zip(header, map(float, row), strict=True)) for row in rows]


# ---- The tables, at Palace 0.18.1's headers ----------------------------------------------------------------


@pytest.mark.parametrize("run", [UNIFORM_RUN, ADAPTIVE_RUN])
def test_the_real_tables_of_both_runs_pass_their_validators(tmp_path, run):
    """Every table stock Palace wrote is read, one row per requested frequency."""
    result = fixture_result(copied_run(run, tmp_path))
    assert set(result.tables) == {
        "port-S",
        "port-I",
        "port-V",
        "domain-E",
        "surface-Q",
        "error-indicators",
    }
    assert {
        len(table.rows)
        for name, table in result.tables.items()
        if name != "error-indicators"
    } == {POINTS[run]}


def _drop_columns(names):
    def change(rows):
        keep = [i for i, name in enumerate(rows[0]) if not names(name)]
        return [[row[i] for i in keep] for row in rows]

    return change


def _set_cell(row: int, column: int, value: str):
    def change(rows):
        rows[row][column] = value
        return rows

    return change


def _swap_rows(first: int, second: int):
    def change(rows):
        rows[first], rows[second] = rows[second], rows[first]
        return rows

    return change


@pytest.mark.parametrize(
    ("family", "change", "match"),
    [
        pytest.param(
            "port-S",
            _drop_columns(lambda name: name == "arg(S[2][1]) (deg.)"),
            r"are not the 9 columns stock Palace 0\.18\.1 writes .* column 5",
            id="a missing column",
        ),
        pytest.param(
            "surface-Q",
            _drop_columns(lambda name: name.endswith("][2]")),
            r"are not the 29 columns stock Palace 0\.18\.1 writes for this Driven run \(15 found\)",
            id="a missing excitation block",
        ),
        pytest.param(
            "port-S",
            _swap_rows(2, 3),
            r"port-S\.csv row 2: is not at the requested frequency .* found 6\.0, expected 4\.0",
            id="reordered rows",
        ),
        pytest.param(
            "domain-E",
            _set_cell(2, 0, "4.00000100e+00"),
            r"domain-E\.csv row 2: is not at the requested frequency",
            id="a shifted frequency",
        ),
        pytest.param(
            "port-V",
            lambda rows: rows[:-1],
            r"has 4 frequency rows, and config\.json requests 5 frequencies",
            id="a missing row",
        ),
        pytest.param("port-I", _set_cell(3, 4, "nan"), "is non-finite", id="a NaN"),
        pytest.param(
            "port-S", _set_cell(1, 3, "-inf"), "is non-finite", id="an infinite |S|"
        ),
        pytest.param(
            "surface-Q",
            _set_cell(1, 1, "+inf"),
            "only a positive solver-native Q",
            id="an infinite participation",
        ),
    ],
)
def test_a_table_palace_did_not_write_this_way_is_refused(
    tmp_path, family, change, match
):
    """One change to a real table, each a way a returned table can be wrong, is refused before it is read."""
    root = copied_run(UNIFORM_RUN, tmp_path)
    _write_cells(root, family, change(_cells(root, family)))
    with pytest.raises(ValueError, match=match):
        fixture_result(root)


def test_a_lossless_film_s_infinite_q_is_read(tmp_path):
    """Palace writes Q_surf as +inf for a row with no loss tangent; the participation beside it is read."""
    root = copied_run(UNIFORM_RUN, tmp_path)
    _write_cells(root, "surface-Q", _set_cell(1, 2, "+inf")(_cells(root, "surface-Q")))
    record = fixture_result(root).driven_result().surface_epr[0].surfaces[0]
    assert record.quality_factor == math.inf
    assert record.participation == _rows(root, "surface-Q")[0]["p_surf[1][1]"]


def _rewrite_config(root: Path, change) -> None:
    config = read_json(root, CONFIG)
    change(config)
    (root / CONFIG).write_text(json.dumps(config), encoding="utf-8")


def test_an_unexcited_lumped_element_is_refused_by_name(tmp_path):
    """Palace's columns for a run with an unexcited element are not verified, so its S-matrix is not read."""
    root = copied_run(UNIFORM_RUN, tmp_path)
    _rewrite_config(
        root,
        lambda config: config["Boundaries"]["LumpedPort"][1].update(Excitation=False),
    )
    with pytest.raises(
        NotImplementedError, match="LumpedPort Index 2: is an unexcited lumped port"
    ):
        fixture_result(root)


def test_an_excitation_that_is_not_the_port_s_index_is_refused(tmp_path):
    """Excitation e is read as port e driven, so a config that excites a port by another index is refused."""
    root = copied_run(UNIFORM_RUN, tmp_path)
    _rewrite_config(
        root, lambda config: config["Boundaries"]["LumpedPort"][1].update(Excitation=1)
    )
    with pytest.raises(ValueError, match="must be the port's own Index"):
        fixture_result(root)


# ---- The S-parameters --------------------------------------------------------------------------------------


@pytest.mark.parametrize("run", [UNIFORM_RUN, ADAPTIVE_RUN])
def test_the_ports_are_the_plan_s_in_the_index_map_s_order(tmp_path, run):
    """A9: port i of the index map (port i of the plan) is port i of the record and of its Touchstone file."""
    root = copied_run(run, tmp_path)
    result = fixture_result(root).driven_result()
    entries = sorted(
        (
            e
            for e in read_json(root, INDEX_MAP)["entries"]
            if e["section"] == LUMPED_PORT_SECTION
        ),
        key=lambda entry: entry["index"],
    )
    assert [entry["port_name"] for entry in entries] == list(PORTS)
    assert [(p.name, p.sheet_id, p.impedance_ohm) for p in result.ports] == [
        (name, sheet, 50.0) for name, sheet in zip(PORTS, SHEETS, strict=True)
    ]
    assert isinstance(result, SParameters)
    assert result.reference_ohm == 50.0
    lines = (
        result.write_touchstone(tmp_path / "run.s2p")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert [line for line in lines if line.startswith("! Port")] == [
        f"! Port {number}: {name} ({sheet}), 50 ohm"
        for number, (name, sheet) in enumerate(zip(PORTS, SHEETS, strict=True), start=1)
    ]


@pytest.mark.parametrize("run", [UNIFORM_RUN, ADAPTIVE_RUN])
def test_each_entry_is_the_cell_its_ports_name(tmp_path, run):
    """``magnitude_db[k][o-1][e-1]`` is the cell under ``|S[o][e]| (dB)``: columns are bound by name."""
    root = copied_run(run, tmp_path)
    result = fixture_result(root).driven_result()
    rows = _rows(root, "port-S")
    assert result.frequencies_hz == tuple(row["f (GHz)"] * 1e9 for row in rows)
    for k, row in enumerate(rows):
        for o in (1, 2):
            for e in (1, 2):
                assert (
                    result.magnitude_db[k][o - 1][e - 1] == row[f"|S[{o}][{e}]| (dB)"]
                )
                assert (
                    result.phase_deg[k][o - 1][e - 1] == row[f"arg(S[{o}][{e}]) (deg.)"]
                )
    # S12 and S21 differ in the last printed digits, so a transposed read would show.
    assert result.magnitude_db[0][1][0] != result.magnitude_db[0][0][1]


def test_the_order_2_run_is_the_matched_lossless_line_of_the_plan(tmp_path):
    """r3 reproduces facts 11 and A2 to A5 of the 2026-11-20 plan through ``s(output, excited)``."""
    result = _result(tmp_path, ADAPTIVE_RUN)
    s11, s21, s12, s22 = (result.s(o, e) for o, e in ((1, 1), (2, 1), (1, 2), (2, 2)))
    at = {round(f / 1e9, 6): k for k, f in enumerate(result.frequencies_hz)}

    def decibel(value: complex) -> float:
        return 20 * math.log10(abs(value))

    assert decibel(s21[0]) == pytest.approx(-0.0003893001159898, rel=1e-9)
    assert min(decibel(value) for value in s21) > -0.1
    assert max(decibel(value) for value in s11) < -29.0
    assert [
        round(math.degrees(cmath.phase(s21[at[f]])), 4)
        for f in (2.0, 4.0, 6.0, 8.0, 10.0)
    ] == [
        -11.118,
        -22.2353,
        -33.352,
        -44.4691,
        -55.5891,
    ]
    assert max(abs(a - b) for a, b in zip(s12, s21, strict=True)) < 1e-13
    assert max(abs(a - b) for a, b in zip(s11, s22, strict=True)) < 2e-4
    assert (
        max(abs(1 - abs(a) ** 2 - abs(b) ** 2) for a, b in zip(s11, s21, strict=True))
        < 3e-4
    )


def test_the_touchstone_of_the_uniform_run_is_palace_s_numbers_in_s2p_order(tmp_path):
    """A1: the ``.s2p`` holds Palace's printed |S| (dB) and arg S (degrees) digit for digit, S11 S21 S12 S22."""
    path = _result(tmp_path, UNIFORM_RUN).write_touchstone(tmp_path / "r1.s2p")
    assert path.read_text(encoding="utf-8") == (
        "! S-parameters written by SCGSim; ports in the order the plan declares them\n"
        "! Port 1: chip/o_lumped_left (PORT_SHEET__chip/o_lumped_left__0000), 50 ohm\n"
        "! Port 2: chip/o_lumped_right (PORT_SHEET__chip/o_lumped_right__0000), 50 ohm\n"
        "# GHz S DB R 50\n"
        "2 -34.90319393692 -101.456506768 -0.001505138226289 -11.51197408445 "
        "-0.001476373052404 -11.51204409239 -34.90275477063 -101.5774804455\n"
        "4 -28.91833207664 -112.9366937128 -0.005835729964721 -23.02180828905 "
        "-0.005835833868773 -23.02180786656 -28.91834103975 -113.1445590174\n"
        "6 -25.45707795998 -124.4006399887 -0.01295957970547 -34.52731749003 "
        "-0.01295968790981 -34.52731688463 -25.45709823885 -124.7067181315\n"
        "8 -23.04605208905 -135.8599723231 -0.02260972315064 -46.02654624287 "
        "-0.02260995990993 -46.0265440945 -23.04608661718 -136.2576322487\n"
        "10 -21.22586741596 -147.3148357858 -0.03448934096034 -57.51771973049 "
        "-0.03449045888929 -57.51772129722 -21.22593333355 -147.7957335561\n"
    )


def test_the_touchstone_of_the_adaptive_run_reads_back_as_its_record(tmp_path):
    """A1 on the 81-point run: the file read back gives the record's frequencies and values exactly."""
    result = _result(tmp_path, ADAPTIVE_RUN)
    option, frequencies, values = read_touchstone(
        result.write_touchstone(tmp_path / "r3.s2p")
    )
    assert option == ["#", "GHz", "S", "DB", "R", "50"]
    assert frequencies == [f / 1e9 for f in result.frequencies_hz]
    order = ((0, 0), (1, 0), (0, 1), (1, 1))
    assert values == [
        [
            part[o][e]
            for o, e in order
            for part in (result.magnitude_db[k], result.phase_deg[k])
        ]
        for k in range(81)
    ]


@pytest.mark.parametrize("run", [UNIFORM_RUN, ADAPTIVE_RUN])
def test_the_sweep_and_the_provenance_name_the_request_and_the_files(tmp_path, run):
    """``sweep`` is the request as ``set_driven`` set it; ``provenance`` lists the returned files read, with hashes."""
    result = _result(tmp_path, run)
    adaptive = run == ADAPTIVE_RUN
    assert dict(result.sweep) == {
        "kind": "adaptive" if adaptive else "uniform",
        "start_hz": 2e9,
        "stop_hz": 10e9,
        "points": POINTS[run],
        "adaptive_tolerance": 1e-3 if adaptive else None,
        "adaptive_max_samples": None,
    }
    assert result.provenance["solver"] == "Palace"
    assert result.provenance["solver_git_tag"] == "0dc74cd"
    assert [item["path"] for item in result.provenance["source_files"]] == [
        output_table("port-S"),
        output_table("surface-Q"),
        *([LOG] if adaptive else []),
    ]


# ---- Surface EPR -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("run", [UNIFORM_RUN, ADAPTIVE_RUN])
def test_the_surface_epr_is_each_row_s_participation_per_excited_port_and_frequency(
    tmp_path, run
):
    """A8: every configured row, for both ports driven, at every frequency, bound as Eigenmode binds it."""
    root = copied_run(run, tmp_path)
    result = fixture_result(root).driven_result()
    rows = _rows(root, "surface-Q")
    assert [(item.excited_port, item.frequency_hz) for item in result.surface_epr] == [
        (port, f) for port in (1, 2) for f in result.frequencies_hz
    ]
    first = result.surface_epr[0]
    assert [(r.index, r.interface_type) for r in first.surfaces] == [
        (1, "MA"),
        (2, "MA"),
        (3, "MA"),
        (4, "MA"),
        (5, "MS"),
        (6, "MS"),
        (7, "SA"),
    ]
    assert first.surfaces[6].name == "SA::D0_SUBSTRATE|VACUUM_REGION__interface"
    assert first.margins == ()
    for item in result.surface_epr:
        row = rows[result.frequencies_hz.index(item.frequency_hz)]
        for record in item.surfaces:
            assert (
                record.participation
                == row[f"p_surf[{record.index}][{item.excited_port}]"]
            )
            assert (
                record.quality_factor
                == row[f"Q_surf[{record.index}][{item.excited_port}]"]
            )


# ---- The adaptive sweep in Palace's log --------------------------------------------------------------------


def test_the_adaptive_sampling_of_each_port_is_read_from_the_log(tmp_path):
    """r3's log: 4 samples for each port, at 2, 10, 6.142 and 3.744 GHz; a uniform sweep has none."""
    result = _result(tmp_path, ADAPTIVE_RUN)
    assert [(a.excited_port, a.samples, a.error) for a in result.adaptive_sampling] == [
        (1, 4, 1.220e-06),
        (2, 4, 4.566e-06),
    ]
    assert result.adaptive_sampling[0].sampled_frequencies_hz == (
        2e9,
        10e9,
        6.142e9,
        3.744e9,
    )
    assert _result(tmp_path / "uniform", UNIFORM_RUN).adaptive_sampling == ()


def _rewrite_log(root: Path, change) -> None:
    path = root / LOG
    path.write_text(change(path.read_text(encoding="utf-8")), encoding="utf-8")


def _second(old: str, new: str):
    """Replace the second occurrence of ``old``: the log line of excitation 2."""

    def change(text: str) -> str:
        first = text.index(old) + len(old)
        return text[:first] + text[first:].replace(old, new, 1)

    return change


def test_a_sweep_that_reached_its_maximum_for_a_port_is_refused_naming_it(tmp_path):
    """The convergence of the adaptive sweep is only in the log: "reached maximum" refuses the run."""
    root = copied_run(ADAPTIVE_RUN, tmp_path)
    _rewrite_log(
        root,
        _second(
            "Adaptive sampling converged with 4 frequency samples:",
            "Adaptive sampling reached maximum 20 frequency samples:",
        ),
    )
    with pytest.raises(ValueError) as refusal:
        fixture_result(root).driven_result()
    message = str(refusal.value)
    assert "the adaptive sweep with port 2 (chip/o_lumped_right) excited" in message
    assert "reached its maximum of 20 frequency samples" in message
    assert "adaptive_max_samples" in message
    assert "Change it in the call to DrivenSim.set_driven" in message


def test_a_log_that_does_not_record_a_port_s_sampling_is_refused(tmp_path):
    """Without the status line of a port, whether its sweep converged is unknown, so the run is refused."""
    root = copied_run(ADAPTIVE_RUN, tmp_path)
    _rewrite_log(
        root, _second("Adaptive sampling converged with", "Adaptive sampling done with")
    )
    with pytest.raises(
        ValueError, match="does not record how the adaptive sweep converged with port 2"
    ):
        fixture_result(root).driven_result()


# ---- What does not read a Driven run -----------------------------------------------------------------------


def test_epr_result_of_a_driven_run_points_to_its_surface_epr(tmp_path):
    """``epr_result`` reads Eigenmode tables; for a Driven run it names where the Surface EPR is."""
    result = fixture_result(copied_run(UNIFORM_RUN, tmp_path))
    with pytest.raises(TypeError, match=r"driven_result\(\)\.surface_epr"):
        result.epr_result()
    with pytest.raises(TypeError, match=r"driven_result\(\)\.surface_epr"):
        epr_result(result)


def test_driven_result_of_another_problem_is_refused(tmp_path):
    """``driven_result`` reads Driven tables only and names what an Eigenmode run yields instead."""
    result = replace(
        fixture_result(copied_run(UNIFORM_RUN, tmp_path)), problem="Eigenmode"
    )
    with pytest.raises(
        TypeError, match="driven_result requires a resolved Palace Driven run"
    ):
        result.driven_result()
