"""``DrivenSim`` on the launcher-to-launcher layout of issue 187: two ports in the plan's order, in either drawn sense.

The layout is the roster's, composed from OrcaPen SC PDK cells (``scripts/roster_cases.py``), so these tests run in CI's
pdk-tests job.  ``mesh()`` builds the real SGB model on Routes A and B and stands in only for Gmsh; the
config is written from that model's physical groups.  The same layout, meshed and solved on stock Palace
0.18.1 through these calls, is the run recorded on issue 187.

The readback tests run the run script ``prepare_handoff()`` writes for each fixture run's config; the
``palace`` it starts is a stand-in that returns that run's real outputs (``palace_driven_runs_support``), and the
copied recorder writes the receipt, so ``resolve_palace_result`` checks a run folder with every identity a real
return has.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pytest
from palace_driven_runs_support import (
    ADAPTIVE_RUN,
    FIXTURES,
    INDUCTOR_RUN,
    LOG,
    RESISTOR_ADAPTIVE_RUN,
    RESISTOR_RUN,
    UNIFORM_RUN,
    copied_run,
    fixture_result,
    read_touchstone,
)
from palace_driven_support import launcher_plan, mesh_without_gmsh

from scgsim.contracts.palace_index_map import LUMPED_PORT_SECTION
from scgsim.contracts.palace_run_folder import CONFIG, INDEX_MAP, OUTPUT_DIR, read_json
from scgsim.geometry import LumpedElement, LumpedPort
from scgsim.palace import (
    DrivenPhysicsReport,
    DrivenSim,
    DrivenTrustReport,
    inspect_run_trustworthiness,
    resolve_palace_result,
)
from scgsim.semantics.s_parameters import SParameters

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


# ---- Reading the run back --------------------------------------------------------------------------------

_UNIFORM_SWEEP = {"start": 2e9, "stop": 10e9, "points": 5}
_LEFT_PORT = LumpedPort(
    "chip/o_lumped_left", reference_net="ground", impedance_ohm=50.0
)
# Each fixture run as it was prepared: its right sheet (None: the 50 ohm port of launcher_plan), FEM order and
# sweep, through set_lumped_elements, set_numerical and set_driven.
_RUN_SETTINGS = {
    UNIFORM_RUN: (None, 1, _UNIFORM_SWEEP),
    ADAPTIVE_RUN: (
        None,
        2,
        {"start": 2e9, "stop": 10e9, "points": 81, "adaptive_tolerance": 1e-3},
    ),
    RESISTOR_RUN: (
        LumpedElement("chip/o_lumped_right", resistance_ohm=50.0),
        1,
        _UNIFORM_SWEEP,
    ),
    INDUCTOR_RUN: (
        LumpedElement("chip/o_lumped_right", inductance_h=10e-9),
        1,
        _UNIFORM_SWEEP,
    ),
    RESISTOR_ADAPTIVE_RUN: (
        LumpedElement("chip/o_lumped_right", resistance_ohm=50.0),
        1,
        {"start": 2e9, "stop": 10e9, "points": 11, "adaptive_tolerance": 1e-3},
    ),
}
# Touchstone 1.1's order of the (output, excited) pairs of one frequency, for one and two ports.
_TOUCHSTONE_ORDER = {1: ((0, 0),), 2: ((0, 0), (1, 0), (0, 1), (1, 1))}
_STAND_IN_PALACE = """#!/usr/bin/env bash
# A stand-in for Palace: it reports the supported release, accepts the dry run, and returns a fixture run's outputs.
if [ "$1" = "--version" ]; then
  printf 'Palace version: 0.18.1\\nSchema version: 1-7-0\\n'
  exit 0
fi
if [ "$1" = "--dry-run" ]; then
  exit 0
fi
cp "{returned}/{output}"/* {output}/
cat "{returned}/{log}"
"""


def _returned(monkeypatch, tmp_path: Path, run: str):
    """The fixture run's folder as DrivenSim prepares it and its run script returns it, with the run's outputs."""
    right, order, sweep = _RUN_SETTINGS[run]
    mesh_without_gmsh(monkeypatch)
    sim = DrivenSim()
    elements = None if right is None else [_LEFT_PORT, right]
    plan = launcher_plan(surface_epr=True, bands=False, lumped_elements=elements)
    sim.set_plan(plan.prepare())
    sim.set_output_dir(tmp_path / run)
    sim.set_driven(**sweep)
    sim.set_numerical(order=order)
    sim.mesh()
    sim.write_config()
    palace = tmp_path / "bin" / "palace"
    palace.parent.mkdir()
    palace.write_text(
        _STAND_IN_PALACE.format(returned=FIXTURES / run, output=OUTPUT_DIR, log=LOG),
        encoding="utf-8",
    )
    palace.chmod(0o755)
    python = tmp_path / "bin" / "python3"
    # A python3 that sees only the standard library, as on a solver host.
    python.write_text(
        f'#!/bin/sh\nexec "{sys.executable}" -I -S "$@"\n', encoding="utf-8"
    )
    python.chmod(0o755)
    handoff = sim.prepare_handoff(
        profile="direct-local",
        executable=str(palace),
        resources={"processes": 1, "threads": 1},
    )
    completed = subprocess.run(
        ["bash", str(handoff.script_path)],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{python.parent}{os.pathsep}{os.environ['PATH']}"},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return handoff


@pytest.mark.parametrize("run", list(_RUN_SETTINGS))
def test_a_returned_driven_run_resolves_to_its_s_parameters(monkeypatch, tmp_path, run):
    """A1 and A9: ``resolve_palace_result(...).driven_result()`` on each fixture run's config, ports in the plan's order.

    The config and the index map ``DrivenSim`` writes equal the run's, so the returned tables are the ones this
    config asked for.  The result is the one the fixture's tables give, and its Touchstone file reads back as its
    record.  With the right sheet an unexcited element, the plan has one port.
    """
    handoff = _returned(monkeypatch, tmp_path, run)
    for relative in (CONFIG, INDEX_MAP):
        assert read_json(handoff.run_dir / relative) == read_json(
            FIXTURES / run / relative
        )
    result = resolve_palace_result(
        handoff.run_dir, expected_handoff_id=handoff.handoff_id
    ).driven_result()
    expected = fixture_result(copied_run(run, tmp_path / "fixture")).driven_result()
    for field in fields(result):
        if field.name != "provenance":
            assert getattr(result, field.name) == getattr(expected, field.name), (
                field.name
            )
    assert result.provenance["handoff_id"] == handoff.handoff_id
    ports = ["chip/o_lumped_left"] + (
        ["chip/o_lumped_right"] if _RUN_SETTINGS[run][0] is None else []
    )
    assert [port.name for port in result.ports] == ports
    option, frequencies, values = read_touchstone(
        result.write_touchstone(tmp_path / f"{run}.s{len(ports)}p")
    )
    assert option == ["#", "GHz", "S", "DB", "R", "50"]
    assert frequencies == [f / 1e9 for f in result.frequencies_hz]
    order = _TOUCHSTONE_ORDER[len(ports)]
    assert values == [
        [part[o][e] for o, e in order for part in (magnitude, phase)]
        for magnitude, phase in zip(result.magnitude_db, result.phase_deg, strict=True)
    ]
    assert isinstance(result, SParameters)


def test_the_views_of_a_resolved_driven_run_are_its_own(monkeypatch, tmp_path):
    """The really resolved run's views: its trust report, its S-parameters and its benchmark, complete.

    ``inspect_run_trustworthiness`` reads the same folder as it came back: complete, with the same sweep and
    full solves, and no S-parameters.  ``epr_result()`` still points to ``driven_result().surface_epr``.
    """
    handoff = _returned(monkeypatch, tmp_path, UNIFORM_RUN)
    resolved = resolve_palace_result(
        handoff.run_dir, expected_handoff_id=handoff.handoff_id
    )
    trust = resolved.show_run_trustworthiness()
    assert isinstance(trust, DrivenTrustReport)
    assert (trust.completeness, trust.profile, trust.full_solves) == (
        "complete",
        "direct-local",
        10,
    )
    assert trust.result == resolved.driven_result()
    assert isinstance(resolved.show_physics_quantities(), DrivenPhysicsReport)
    assert resolved.show_simulation_benchmark().data["full_solves"] == 10
    inspected = inspect_run_trustworthiness(handoff.run_dir)
    assert (inspected.completeness, inspected.sweep, inspected.full_solves) == (
        "complete",
        trust.sweep,
        10,
    )
    assert inspected.result is None
    with pytest.raises(TypeError, match=r"driven_result\(\)\.surface_epr"):
        resolved.epr_result()
