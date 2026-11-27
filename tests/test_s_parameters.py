"""``SParameters``: the record every driven solver returns, ``s(output, excited)``, and its Touchstone 1.1 file.

The records here are built by hand with values chosen so that every entry differs, so an entry read from the
wrong place shows.  The record a Palace run builds, from the real port-S table of issue 187's E1, is in
``test_palace_driven_results``.
"""

from __future__ import annotations

import cmath
import math

import pytest
from palace_driven_runs_support import read_touchstone

from scgsim.semantics.s_parameters import SParameterPort, SParameters


def _port(number: int, impedance: float = 50.0) -> SParameterPort:
    return SParameterPort(
        f"chip/p{number}", f"PORT_SHEET__chip/p{number}__0000", impedance
    )


def _record(ports: int, *, impedances: tuple[float, ...] | None = None) -> SParameters:
    """``ports`` ports at 4 and 5 GHz; |S[o][e]| is -(10 o + e + k) dB and arg S[o][e] is 10 o + e + k degrees."""
    impedances = impedances or (50.0,) * ports
    return SParameters(
        ports=[
            _port(n, z) for n, z in zip(range(1, ports + 1), impedances, strict=True)
        ],
        frequencies_hz=[4e9, 5e9],
        magnitude_db=[
            [
                [-(10.0 * o + e + k) for e in range(1, ports + 1)]
                for o in range(1, ports + 1)
            ]
            for k in range(2)
        ],
        phase_deg=[
            [
                [10.0 * o + e + k for e in range(1, ports + 1)]
                for o in range(1, ports + 1)
            ]
            for k in range(2)
        ],
        sweep={
            "sampling": "uniform",
            "ranges": [
                {"kind": "linear", "start_hz": 4e9, "stop_hz": 5e9, "points": 2}
            ],
            "frequency_count": 2,
        },
        provenance={"solver": "test"},
    )


def test_s_of_output_and_excited_is_the_entry_s21_names():
    """``s(2, 1)`` is S21: output port 2 with port 1 excited, at every frequency, as a complex number."""
    record = _record(2)
    s21 = record.s(2, 1)
    assert len(s21) == 2
    for k, value in enumerate(s21):
        assert abs(value) == pytest.approx(10 ** (-(21 + k) / 20), rel=1e-14)
        assert math.degrees(cmath.phase(value)) == pytest.approx(21 + k, rel=1e-14)
    assert record.s(1, 2)[0] != record.s(2, 1)[0]


def test_s_refuses_a_port_number_the_record_does_not_have():
    """Port numbers count from 1 to N, as S21 is written; 0, N + 1 and a bool are refused."""
    record = _record(2)
    for output, excited in ((0, 1), (3, 1), (1, True)):
        with pytest.raises(ValueError, match="must be a port number from 1 to 2"):
            record.s(output, excited)


def test_the_two_port_touchstone_is_s11_s21_s12_s22_in_ghz_db_and_degrees(tmp_path):
    """Touchstone 1.1's one exception to row order: a two-port line is S11, S21, S12, S22."""
    path = _record(2).write_touchstone(tmp_path / "line.s2p")
    assert path.read_text(encoding="utf-8") == (
        "! S-parameters written by SCGSim; ports in the order the plan declares them\n"
        "! Port 1: chip/p1 (PORT_SHEET__chip/p1__0000), 50 ohm\n"
        "! Port 2: chip/p2 (PORT_SHEET__chip/p2__0000), 50 ohm\n"
        "# GHz S DB R 50\n"
        "4 -11 11 -21 21 -12 12 -22 22\n"
        "5 -12 12 -22 22 -13 13 -23 23\n"
    )


def test_a_three_port_touchstone_writes_one_matrix_row_per_line(tmp_path):
    """From three ports on, each matrix row starts a line; only the first line carries the frequency."""
    path = _record(3).write_touchstone(tmp_path / "tee.s3p")
    data = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line[0] not in "!#"
    ]
    assert data[:3] == [
        "4 -11 11 -12 12 -13 13",
        "-21 21 -22 22 -23 23",
        "-31 31 -32 32 -33 33",
    ]


def test_a_matrix_row_of_more_than_four_pairs_continues_on_the_next_line(tmp_path):
    """Touchstone 1.1 puts at most four pairs on a line: a five-port row takes two lines."""
    path = _record(5).write_touchstone(tmp_path / "five.s5p")
    data = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line[0] not in "!#"
    ]
    assert data[:2] == ["4 -11 11 -12 12 -13 13 -14 14", "-15 15"]
    assert len(data) == 2 * 5 * 2


def test_the_touchstone_reads_back_as_the_record(tmp_path):
    """Every number is written as the shortest text of its float, so reading the file gives the record's."""
    record = SParameters(
        ports=[_port(1, 25.0)],
        frequencies_hz=[2.1e9, 3.3e9],
        magnitude_db=[[[-0.1 / 3]], [[-34.90319393692]]],
        phase_deg=[[[math.pi]], [[-101.456506768]]],
        sweep={},
        provenance={},
    )
    option, frequencies, values = read_touchstone(
        record.write_touchstone(tmp_path / "pad.s1p")
    )
    assert option == ["#", "GHz", "S", "DB", "R", "25"]
    assert frequencies == [f / 1e9 for f in record.frequencies_hz]
    assert values == [
        [record.magnitude_db[k][0][0], record.phase_deg[k][0][0]] for k in range(2)
    ]


def test_ports_of_different_impedance_are_refused_by_touchstone(tmp_path):
    """Touchstone 1.1 has one reference impedance; the record keeps each port's (#187 decision 10)."""
    record = _record(2, impedances=(50.0, 25.0))
    assert record.reference_impedances_ohm == (50.0, 25.0)
    with pytest.raises(ValueError, match="have different impedances") as refusal:
        record.write_touchstone(tmp_path / "line.s2p")
    assert "'chip/p1': 50.0, 'chip/p2': 25.0" in str(refusal.value)
    assert not (tmp_path / "line.s2p").exists()


def test_a_path_without_the_port_count_suffix_is_refused(tmp_path):
    """A Touchstone reader takes the port count from the suffix, so another suffix is refused."""
    for name in ("line.s1p", "line.txt", "line"):
        with pytest.raises(ValueError, match=r"must end in \.s2p"):
            _record(2).write_touchstone(tmp_path / name)


@pytest.mark.parametrize(
    ("change", "error", "match"),
    [
        ({"ports": []}, ValueError, "ports must be at least one SParameterPort"),
        ({"ports": [_port(1), _port(1)]}, ValueError, "each named once"),
        ({"frequencies_hz": [5e9, 4e9]}, ValueError, "above 0 Hz and increasing"),
        ({"frequencies_hz": [0.0, 4e9]}, ValueError, "above 0 Hz and increasing"),
        (
            {"magnitude_db": [[[0.0, 0.0], [0.0, 0.0]]]},
            ValueError,
            "2 matrices of 2 x 2",
        ),
        (
            {"phase_deg": [[[0.0, math.nan], [0.0, 0.0]]] * 2},
            ValueError,
            "phase_deg must be finite",
        ),
        ({"sweep": None}, TypeError, "sweep must be a mapping"),
    ],
)
def test_a_record_that_is_not_an_s_matrix_is_refused(change, error, match):
    """A record built by hand is checked: ports, increasing frequencies, one N x N matrix per frequency."""
    fields = {
        "ports": [_port(1), _port(2)],
        "frequencies_hz": [4e9, 5e9],
        "magnitude_db": [[[0.0, 0.0], [0.0, 0.0]]] * 2,
        "phase_deg": [[[0.0, 0.0], [0.0, 0.0]]] * 2,
        "sweep": {},
        "provenance": {},
    }
    with pytest.raises(error, match=match):
        SParameters(**(fields | change))


def test_a_port_needs_a_positive_impedance():
    """A port's impedance is its waves' reference, so it must be above zero."""
    with pytest.raises(ValueError, match="impedance_ohm must be above 0 ohm"):
        SParameterPort("chip/p1", "PORT_SHEET__chip/p1__0000", 0.0)


@pytest.mark.parametrize("field", ["name", "sheet_id"])
@pytest.mark.parametrize("label", ["chip/p1\n# MHz Y MA R 1", "chip/p1\r1 0 0", ""])
def test_a_port_label_must_be_nonempty_text_on_one_line(field, label):
    """The Touchstone file names each port on one comment line, so a label with a line break is refused."""
    labels = {"name": "chip/p1", "sheet_id": "PORT_SHEET__chip/p1__0000"} | {
        field: label
    }
    with pytest.raises(ValueError, match=f"{field} must be nonempty text on one line"):
        SParameterPort(labels["name"], labels["sheet_id"], 50.0)
