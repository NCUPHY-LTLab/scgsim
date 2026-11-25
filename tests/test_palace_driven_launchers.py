"""``DrivenSim`` on the launcher-to-launcher layout of issue 187: two ports in the plan's order, in either drawn sense.

The layout is composed from OrcaPen SC PDK cells (``palace_driven_support``), so these tests run in CI's
pdk-tests job.  ``mesh()`` builds the real SGB model on Routes A and B and stands in only for Gmsh; the
config is written from that model's physical groups.  The same layout, meshed and solved on stock Palace
0.18.1 through these calls, is the run recorded on issue 187.
"""

from __future__ import annotations

import json

import pytest
from palace_driven_support import launcher_plan, mesh_without_gmsh

from scgsim.contracts.palace_index_map import LUMPED_PORT_SECTION
from scgsim.contracts.palace_run_folder import INDEX_MAP
from scgsim.geometry import LumpedElement, LumpedPort
from scgsim.palace import DrivenSim

pytestmark = pytest.mark.pdk


@pytest.fixture(autouse=True)
def _pdk_installed():
    """Skip a selected test without the PDK; the public suite deselects these tests, so nothing skips there."""
    pytest.importorskip("gdsfactory", reason="the launcher layout needs gdsfactory")
    pytest.importorskip(
        "orcapen_sc_pdk", reason="public OrcaPen SC PDK is not installed"
    )


# Both ports point from the pad to the ground: -x at the left launcher, +x at the right one.
PAD_TO_GROUND = {1: [-1.0, 0.0, 0.0], 2: [1.0, 0.0, 0.0]}


def configured(monkeypatch, tmp_path, plan, *, route):
    """``write_config()`` of ``plan`` on ``route`` with a uniform 2-10 GHz sweep, and the run folder."""
    mesh_without_gmsh(monkeypatch)
    sim = DrivenSim()
    sim.set_plan(plan.prepare())
    sim.set_output_dir(tmp_path / f"run-{route}")
    sim.set_route(route)
    sim.set_driven(start=2e9, stop=10e9, points=5)
    sim.mesh()
    config = json.loads(sim.write_config().read_text(encoding="utf-8"))
    return config, sim.output_dir


@pytest.mark.parametrize("route", ["A", "B"])
def test_the_ports_are_the_s_matrix_in_the_plan_order(monkeypatch, tmp_path, route):
    """Port 1 is the left launcher, as declared first: ``Index``, ``Excitation`` and the index map agree."""
    config, run = configured(monkeypatch, tmp_path, launcher_plan(), route=route)
    rows = config["Boundaries"]["LumpedPort"]
    assert [(row["Index"], row["Excitation"], row["R"]) for row in rows] == [
        (1, 1, 50.0),
        (2, 2, 50.0),
    ]
    entries = [
        entry
        for entry in json.loads((run / INDEX_MAP).read_text())["entries"]
        if entry["section"] == LUMPED_PORT_SECTION
    ]
    assert [(entry["index"], entry["port_name"]) for entry in entries] == [
        (1, "chip/o_lumped_left"),
        (2, "chip/o_lumped_right"),
    ]
    assert [row["Attributes"] for row in rows] == [
        entry["attributes"] for entry in entries
    ]


@pytest.mark.parametrize("route", ["A", "B"])
@pytest.mark.parametrize("right_sense", ["pdk", "reversed"])
def test_each_port_points_from_its_pad_to_the_ground_in_either_drawn_sense(
    monkeypatch, tmp_path, route, right_sense
):
    """The right sheet drawn either way gives the same ``Direction``: the sign follows the declaration, not the drawing."""
    plan = launcher_plan(right_sense=right_sense)
    config, _ = configured(monkeypatch, tmp_path, plan, route=route)
    assert {
        row["Index"]: row["Direction"] for row in config["Boundaries"]["LumpedPort"]
    } == PAD_TO_GROUND
    authored = {
        record["name"]: list(record["direction"])
        for record in plan.prepare().stack["metadata"]["port_sheet_source_layers"]
    }
    # The PDK draws both sheets from the ground to the pad; "reversed" draws the right one from its pad.
    assert authored == {
        "chip/o_lumped_left": [1.0, 0.0, 0.0],
        "chip/o_lumped_right": [1.0 if right_sense == "reversed" else -1.0, 0.0, 0.0],
    }


def test_a_lumped_element_is_a_not_excited_port_with_its_circuit(monkeypatch, tmp_path):
    """Beside a port, an inductance on the right sheet is written with ``L``, not excited, in its authored direction."""
    plan = launcher_plan(
        lumped_elements=[
            LumpedElement("chip/o_lumped_right", inductance_h=1e-8),
            LumpedPort(
                "chip/o_lumped_left", reference_net="ground", impedance_ohm=50.0
            ),
        ]
    )
    config, _ = configured(monkeypatch, tmp_path, plan, route="B")
    rows = config["Boundaries"]["LumpedPort"]
    assert [
        (row["Index"], row["Excitation"], row["Direction"], row.get("R"), row.get("L"))
        for row in rows
    ] == [
        (1, 1, [-1.0, 0.0, 0.0], 50.0, None),
        (2, False, [-1.0, 0.0, 0.0], None, 1e-8),
    ]
    assert DrivenSim.simulation_drc(plan).lumped_elements == (
        "P1 chip/o_lumped_left: LumpedPort, R 50.0 ohm, excitation 1: port 1 of the S-matrix, "
        "driven from chip/SIGNAL to chip/GROUND",
        "P2 chip/o_lumped_right: LumpedPort, L 1e-08 H, not excited",
    )


@pytest.mark.parametrize("route", ["A", "B"])
def test_simulation_drc_passes_the_launcher_layout_with_its_bands(route):
    """Both sheets land 10 grid steps into the pad and the ground, which both stages accept, bands included."""
    report = DrivenSim.simulation_drc(
        launcher_plan(surface_epr=True, bands=True), route=route
    )
    assert report.geometry.passed
    assert report.meshing.passed
    assert [line.split(":")[0] for line in report.lumped_elements] == [
        "P1 chip/o_lumped_left",
        "P2 chip/o_lumped_right",
    ]
