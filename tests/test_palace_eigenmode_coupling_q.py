"""An Eigenmode run's port tables follow each lumped port's R and L, and each mode carries its ports' coupling Q.

Palace 0.18.1 writes ``port-EPR`` only for ports with an inductance and ``port-Q`` only for ports with a
resistance (``palace/models/postoperatorcsv.cpp``).  The tables here are written in Palace's own layout, with
the values of a recorded hanger-resonator run (issue 311), and read through the readers ``resolve`` uses.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scgsim.contracts.palace_convergence import ConvergenceCheck
from scgsim.contracts.palace_index_map import ENERGY_SECTION, LUMPED_PORT_SECTION
from scgsim.contracts.palace_run_folder import EIGENMODE
from scgsim.palace.results.convergence import SolverEnd
from scgsim.palace.results.epr import epr_result
from scgsim.palace.results.epr_records import PalaceEprResult
from scgsim.palace.results.resolved_records import ResolvedRunData
from scgsim.palace.results.tables import read_result_table
from scgsim.palace.run.returned_receipt import iteration_output_paths

LEFT, RIGHT = "chip/o_lumped_left", "chip/o_lumped_right"
RESISTOR = {"R": 50.0}
INDUCTOR = {"L": 10e-9}
PORT_TABLES = {"port-EPR", "port-Q", "port-I", "port-V"}

# The hanger run's mode 1 and 2 at linear tolerance 1e-6: both feedline ports 50 ohm, port 1's Direction +x
# and port 2's -x, so their coupling rates have opposite signs and their Q_ext do not.
_EIG = (
    "        m,                Re{f} (GHz),                Im{f} (GHz),                          Q,"
    "              Error (Bkwd.),               Error (Abs.)\n"
    " 1.00e+00,        +7.356647803532e+00,        +2.637241083276e-04,        +1.394762097199e+04,"
    "        +1.586939413578e-09,        +1.702902841067e-02\n"
    " 2.00e+00,        +1.643820173418e+01,        +1.620212803789e-02,        +5.072855146083e+02,"
    "        +3.291579345047e-08,        +3.532108559705e-01\n"
)
_PORT_Q_BOTH = (
    "        m,                   Q_ext[1],             κ_ext[1] (GHz),                   Q_ext[2],"
    "             κ_ext[2] (GHz)\n"
    " 1.00e+00,        +2.254852585888e+04,        -3.262584813559e-04,        +2.248447301351e+04,"
    "        +3.271879131485e-04\n"
    " 2.00e+00,        +1.077222528090e+03,        -1.525980129967e-02,        +1.047128330681e+03,"
    "        +1.569836404244e-02\n"
)
# Port 2 alone is resistive; mode 2 does not load it, which Palace writes as an infinite Q and a zero rate.
_PORT_Q_SECOND = (
    "        m,                   Q_ext[2],             κ_ext[2] (GHz)\n"
    " 1.00e+00,        +2.248447301351e+04,        -3.271879131485e-04\n"
    " 2.00e+00,                       +inf,        +0.000000000000e+00\n"
)
_PORT_EPR_FIRST = (
    "        m,                       p[1]\n"
    " 1.00e+00,        -9.940000000000e-01\n"
    " 2.00e+00,        +1.250000000000e-01\n"
)
_DOMAIN_E = (
    "        m,                 E_elec (J),                  E_mag (J),                  E_cap (J),"
    "                  E_ind (J),              E_elec[1] (J),                 p_elec[1],"
    "               E_mag[1] (J),                  p_mag[1]\n"
    " 1.00e+00,        +1.000000000000e+00,        +1.000000000000e+00,        +0.000000000000e+00,"
    "        +0.000000000000e+00,        +9.207327809000e-01,        +9.207327809000e-01,"
    "        +5.000000000000e-01,        +5.000000000000e-01\n"
    " 2.00e+00,        +1.000000000000e+00,        +1.000000000000e+00,        +0.000000000000e+00,"
    "        +0.000000000000e+00,        +9.000000000000e-01,        +9.000000000000e-01,"
    "        +5.000000000000e-01,        +5.000000000000e-01\n"
)


def _ports(*circuits: dict[str, float]) -> list[dict]:
    """``Boundaries.LumpedPort`` rows as ``write_config`` writes them: port n is ``Index`` n."""
    return [
        {"Index": number, "Excitation": False, "Attributes": [10 + number], **circuit}
        for number, circuit in enumerate(circuits, start=1)
    ]


@pytest.mark.parametrize(
    ("circuits", "written"),
    [
        ((RESISTOR, RESISTOR), {"port-Q", "port-I", "port-V"}),
        ((INDUCTOR,), {"port-EPR", "port-I", "port-V"}),
        ((INDUCTOR, RESISTOR), PORT_TABLES),
        (({"R": 50.0, "L": 10e-9},), PORT_TABLES),
        (({"C": 1e-15},), {"port-I", "port-V"}),
        ((), set()),
    ],
    ids=["R only", "L only", "L and R", "one RL port", "C only", "no port"],
)
def test_the_port_tables_follow_each_ports_r_and_l(circuits, written):
    """A port table is written when a port gives its key a nonzero value, not whenever ports exist."""
    families = EIGENMODE.written_families(lumped_ports=_ports(*circuits), surfaces=True)
    assert set(families) & PORT_TABLES == written
    assert {"eig", "domain-E", "surface-Q", "error-indicators"} <= set(families)


def test_each_port_table_has_a_column_for_the_ports_it_covers():
    """``port-EPR`` covers the inductive port 1, ``port-Q`` the resistive port 2, ``port-I`` both."""
    ports = _ports(INDUCTOR, RESISTOR)
    assert EIGENMODE.port_columns("port-EPR", ports) == (1,)
    assert EIGENMODE.port_columns("port-Q", ports) == (2,)
    assert EIGENMODE.port_columns("port-I", ports) == (1, 2)


def test_the_receipt_records_port_q_where_palace_wrote_it(tmp_path):
    """A pass of a run whose ports are 50 ohm loads is recorded with its ``port-Q``; it has no ``port-EPR``."""
    folder = tmp_path / "results" / "palace" / "iteration1"
    folder.mkdir(parents=True)
    for name in ("eig.csv", "port-Q.csv", "port-EPR.csv", "palace.json"):
        (folder / name).write_text("", encoding="utf-8")
    config = {"Boundaries": {"LumpedPort": _ports(RESISTOR, RESISTOR)}}
    assert iteration_output_paths(tmp_path, "Eigenmode", config=config) == [
        "results/palace/iteration1/eig.csv",
        "results/palace/iteration1/port-Q.csv",
        "results/palace/iteration1/palace.json",
    ]


def _run(tmp_path: Path, tables: dict[str, str], ports: list[dict]) -> ResolvedRunData:
    """A resolved two-mode Eigenmode run with these Palace tables, read and validated as ``resolve`` reads them."""
    config = {
        "Model": {"L0": 1e-6},
        "Solver": {"Eigenmode": {"N": 2}},
        "Boundaries": {"LumpedPort": ports},
    }
    entries = [
        {
            "section": LUMPED_PORT_SECTION,
            "index": port["Index"],
            "port_name": name,
            "metadata": {"host_entity_ids": ["chip/SIGNAL", "chip/GROUND"]},
        }
        for port, name in zip(ports, (LEFT, RIGHT), strict=False)
    ]
    entries.append(
        {
            "section": ENERGY_SECTION,
            "index": 1,
            "entry_name": "D0_SUBSTRATE",
            "metadata": {"material": {"material_id": "silicon"}},
        }
    )
    counts = {"lumped_port": len(ports), "domain": 1, "surface": 0, "terminal": 0}
    read = {}
    for family, text in {"eig": _EIG, "domain-E": _DOMAIN_E, **tables}.items():
        path = tmp_path / f"{family}.csv"
        path.write_text(text, encoding="utf-8")
        read[family] = read_result_table(path, "Eigenmode", counts, config)
    receipt = SimpleNamespace(
        handoff_id="h" * 8,
        schema="receipt",
        timestamp_utc="2026-01-01T00:00:00Z",
        solver_identity={},
        output_files=(),
        input_hashes=(),
    )
    return ResolvedRunData(
        run_dir=tmp_path,
        problem="Eigenmode",
        route="B",
        has_returned_outputs=True,
        status="completed",
        performance=None,
        cost=None,
        tables=read,
        returned_receipt=receipt,
        provenance=SimpleNamespace(
            index_map={"entries": entries}, config=config, palace_json={}
        ),
        convergence=ConvergenceCheck.not_declared(),
        solver_end=SolverEnd("solver_ended", None, None),
    )


def test_a_run_with_only_resistive_ports_carries_each_ports_coupling_q(tmp_path):
    """Both 50 ohm ports are in ``external`` with 1/Q_ext, positive though port 1's κ_ext is negative."""
    run = _run(tmp_path, {"port-Q": _PORT_Q_BOTH}, _ports(RESISTOR, RESISTOR))
    first, second = epr_result(run).rows
    assert first["junction"] == second["junction"] == ()
    assert [dict(item) for item in first["external"]] == [
        {"port_name": LEFT, "inverse_q": 1 / 2.254852585888e04},
        {"port_name": RIGHT, "inverse_q": 1 / 2.248447301351e04},
    ]
    assert [item["inverse_q"] for item in second["external"]] == [
        1 / 1.077222528090e03,
        1 / 1.047128330681e03,
    ]


def test_a_run_with_an_inductive_and_a_resistive_port_carries_both(tmp_path):
    """Port 1's participation comes from ``port-EPR`` and port 2's coupling Q from ``port-Q``, each by its Index."""
    run = _run(
        tmp_path,
        {"port-EPR": _PORT_EPR_FIRST, "port-Q": _PORT_Q_SECOND},
        _ports(INDUCTOR, RESISTOR),
    )
    first, second = epr_result(run).rows
    (junction,) = first["junction"]
    assert (junction["port_name"], junction["participation"]) == (LEFT, -0.994)
    assert [dict(item) for item in first["external"]] == [
        {"port_name": RIGHT, "inverse_q": 1 / 2.248447301351e04}
    ]
    assert [dict(item) for item in second["external"]] == [
        {"port_name": RIGHT, "inverse_q": 0.0}
    ]


def test_a_run_without_a_resistive_port_has_no_external_loss(tmp_path):
    """An inductive port alone gives ``external`` an empty list, and the result is saved and read back with it."""
    run = _run(tmp_path, {"port-EPR": _PORT_EPR_FIRST}, _ports(INDUCTOR))
    result = epr_result(run)
    assert [row["external"] for row in result.rows] == [(), ()]
    again = PalaceEprResult.from_payload(json.loads(json.dumps(result.to_payload())))
    assert again.rows == result.rows


def test_a_port_q_table_without_a_resistive_port_is_refused(tmp_path):
    """A ``port-Q`` that the config's ports cannot have produced is not read as one."""
    with pytest.raises(
        ValueError, match="no lumped port of the config has a column in port-Q"
    ):
        _run(
            tmp_path,
            {"port-EPR": _PORT_EPR_FIRST, "port-Q": _PORT_Q_SECOND},
            _ports(INDUCTOR),
        )


def test_a_saved_result_of_the_previous_schema_is_refused(tmp_path):
    """A v2 result has no ``external`` group, so it is refused rather than read as having no coupling loss."""
    run = _run(tmp_path, {"port-Q": _PORT_Q_BOTH}, _ports(RESISTOR, RESISTOR))
    payload = {
        **epr_result(run).to_payload(),
        "schema_version": "scgsim.palace.epr-result.v2",
    }
    with pytest.raises(ValueError, match="epr-result.v2"):
        PalaceEprResult.from_payload(payload)
