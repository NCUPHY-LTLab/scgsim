"""``DrivenSim``: the plan's ports as Palace Driven excitations, the frequency sweep, and what it refuses.

The layout is real: two pads joined by one authored port sheet (the stand-in PDK of ``test_simulation_drc``),
built by SGB on Routes A and B.  ``mesh()`` runs its preparation, SGB build and meshing-stage check for real;
only Gmsh is stood in for (``palace_driven_support.mesh_without_gmsh``), so ``write_config()`` builds from the
physical groups of the real SGB model.  No Palace runs here: the runs of this config on stock Palace 0.18.1
are recorded on issue 187.  The launcher-to-launcher layout, two ports and both drawn senses, is in
``test_palace_driven_launchers`` (PDK).
"""

from __future__ import annotations

import json

import pytest
from palace_driven_support import mesh_without_gmsh
from test_simulation_drc import BANDED, layout_plan, two_pads

from scgsim.contracts.palace_run_folder import (
    HANDOFF_METADATA,
    PORT_INFORMATION,
    palace_problem,
)
from scgsim.geometry import LumpedElement, LumpedPort
from scgsim.palace import (
    DrivenSim,
    EigenmodeSim,
    ErrorIndicator,
)

NETS = {"pad": ("chip/pad",), "electrode": ("chip/electrode",)}
# The stand-in PDK's materials with the constants a config reads.
MATERIALS = {
    "PEC": {"kind": "conductor"},
    "vacuum": {"kind": "vacuum", "permittivity": 1.0, "loss_tangent": 0.0},
    "Si": {"kind": "dielectric", "permittivity": 11.45, "loss_tangent": 1e-6},
}
# The sheet is drawn from the pad to the electrode: its authored direction is +x.
AUTHORED_DIRECTION = [1.0, 0.0, 0.0]
UNIFORM = {"start": 4e9, "stop": 8e9, "points": 5}


def driven_plan(monkeypatch, *, reference_net="electrode", **options):
    """The two pads with their port sheet declared as one 50 ohm port, the other pad its reference."""
    return layout_plan(
        two_pads(monkeypatch, gap_um=10.0, port=True),
        material_records=MATERIALS,
        nets=NETS,
        lumped_elements=[
            LumpedPort("chip/P1", reference_net=reference_net, impedance_ohm=50.0)
        ],
        **options,
    )


def meshed(monkeypatch, tmp_path, plan, *, route="B", simulation=DrivenSim):
    """A ``simulation`` of ``plan`` after ``mesh()`` on ``route``, Gmsh stood in for."""
    mesh_without_gmsh(monkeypatch)
    sim = simulation()
    sim.set_plan(plan.prepare())
    sim.set_output_dir(tmp_path / f"run-{simulation.__name__}-{route}")
    sim.set_route(route)
    sim.mesh()
    return sim


def written_config(sim, **sweep):
    """``config.json`` as ``write_config()`` writes it after ``set_driven(**sweep)``."""
    sim.set_driven(**(sweep or UNIFORM))
    return json.loads(sim.write_config().read_text(encoding="utf-8"))


# ---- The config ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("route", ["A", "B"])
def test_a_port_is_its_own_excitation_with_its_impedance_on_its_sheet(
    monkeypatch, tmp_path, route
):
    """The port is ``LumpedPort`` 1, excitation 1, R 50 ohm, on the attributes SGB gave its sheet ``P1``."""
    sim = meshed(monkeypatch, tmp_path, driven_plan(monkeypatch), route=route)
    config = written_config(sim)
    sheet = sim._mesh_result.groups["port_surfaces"]["P1"]  # noqa: SLF001 -- the groups the config is built from
    assert config["Problem"]["Type"] == "Driven"
    assert config["Boundaries"]["LumpedPort"] == [
        {
            "Index": 1,
            "Direction": AUTHORED_DIRECTION,
            "Excitation": 1,
            "Attributes": sorted(sheet["attributes"]),
            "R": 50.0,
        }
    ]
    assert not set(sheet["attributes"]) & set(config["Boundaries"]["PEC"]["Attributes"])
    # Palace's natural boundary is the exterior: no boundary row names it.
    assert set(config["Boundaries"]) == {"PEC", "LumpedPort"}
    assert config["Model"]["Refinement"]["MaxIts"] == 0


@pytest.mark.parametrize(
    ("reference_net", "direction"),
    [("electrode", [1.0, 0.0, 0.0]), ("pad", [-1.0, 0.0, 0.0])],
    ids=["signal-at-start", "signal-at-end"],
)
def test_the_direction_points_from_the_signal_to_the_reference(
    monkeypatch, tmp_path, reference_net, direction
):
    """The same sheet, drawn from the pad to the electrode, points the other way when the pad is the reference."""
    plan = driven_plan(monkeypatch, reference_net=reference_net)
    sim = meshed(monkeypatch, tmp_path, plan)
    config = written_config(sim)
    (row,) = config["Boundaries"]["LumpedPort"]
    assert row["Direction"] == direction
    (authored,) = plan.prepare().stack["metadata"]["port_sheet_source_layers"]
    assert list(authored["direction"]) == AUTHORED_DIRECTION
    information = json.loads((sim.output_dir / PORT_INFORMATION).read_text())
    assert information["ports"][0]["direction"] == direction


@pytest.mark.parametrize(
    ("sweep", "block"),
    [
        (
            UNIFORM,
            {
                "Samples": [
                    {"Type": "Linear", "MinFreq": 4.0, "MaxFreq": 8.0, "NSample": 5}
                ]
            },
        ),
        (
            {"start": 2e9, "stop": 10e9, "points": 81, "adaptive_tolerance": 1e-3},
            {
                "Samples": [
                    {"Type": "Linear", "MinFreq": 2.0, "MaxFreq": 10.0, "NSample": 81}
                ],
                "AdaptiveTol": 1e-3,
            },
        ),
        (
            {
                "start": 2.5e9,
                "stop": 7.25e9,
                "points": 20,
                "adaptive_tolerance": 1e-4,
                "adaptive_max_samples": 12,
            },
            {
                "Samples": [
                    {"Type": "Linear", "MinFreq": 2.5, "MaxFreq": 7.25, "NSample": 20}
                ],
                "AdaptiveTol": 1e-4,
                "AdaptiveMaxSamples": 12,
            },
        ),
    ],
    ids=["uniform", "adaptive", "adaptive-capped"],
)
def test_the_sweep_is_one_linear_range_in_ghz(monkeypatch, tmp_path, sweep, block):
    """``set_driven`` in Hz is ``Solver.Driven``'s one ``Linear`` range in GHz; a uniform sweep has no ``AdaptiveTol``."""
    sim = meshed(monkeypatch, tmp_path, driven_plan(monkeypatch))
    assert written_config(sim, **sweep)["Solver"]["Driven"] == block


@pytest.mark.parametrize("route", ["A", "B"])
def test_the_model_and_surface_epr_rows_are_the_ones_eigenmode_writes(
    monkeypatch, tmp_path, route
):
    """The same plan with bands: Driven's PEC, domains and ``Dielectric`` rows are Eigenmode's, row for row."""
    plan = driven_plan(monkeypatch, surface_epr_specs=BANDED)
    driven = written_config(meshed(monkeypatch, tmp_path, plan, route=route))
    eigenmode_sim = meshed(
        monkeypatch, tmp_path, plan, route=route, simulation=EigenmodeSim
    )
    eigenmode_sim.set_numerical(amr_max_passes=0)
    eigenmode = json.loads(eigenmode_sim.write_config().read_text(encoding="utf-8"))
    dielectric = driven["Boundaries"]["Postprocessing"]["Dielectric"]
    assert dielectric
    assert dielectric == eigenmode["Boundaries"]["Postprocessing"]["Dielectric"]
    assert driven["Boundaries"]["PEC"] == eigenmode["Boundaries"]["PEC"]
    assert driven["Domains"] == eigenmode["Domains"]
    assert driven["Model"].get("CrackInternalBoundaryElements") == eigenmode[
        "Model"
    ].get("CrackInternalBoundaryElements")
    # Eigenmode writes the same port as its impedance, not excited, on the same sheet.
    (driven_port,) = driven["Boundaries"]["LumpedPort"]
    (eigenmode_port,) = eigenmode["Boundaries"]["LumpedPort"]
    assert eigenmode_port == {**driven_port, "Excitation": False}


# ---- What it refuses -------------------------------------------------------------------------------------


def test_write_config_needs_the_sweep(monkeypatch, tmp_path):
    """No default band is invented: ``write_config()`` names ``set_driven``."""
    sim = meshed(monkeypatch, tmp_path, driven_plan(monkeypatch))
    with pytest.raises(ValueError, match=r"call set_driven\(start=\.\.\., stop="):
        sim.write_config()


def test_a_plan_without_a_port_is_refused_by_every_call_that_prepares_it(
    monkeypatch, tmp_path
):
    """A plan whose sheet carries an inductance only has nothing to excite; the refusal names the plan's call."""
    plan = layout_plan(
        two_pads(monkeypatch, gap_um=10.0, port=True),
        nets=NETS,
        lumped_elements=[LumpedElement("chip/P1", inductance_h=1e-9)],
    )
    message = r"declares no LumpedPort.*GeometryPlan\.set_lumped_elements"
    with pytest.raises(ValueError, match=message):
        DrivenSim.simulation_drc(plan)
    sim = DrivenSim()
    sim.set_plan(plan.prepare())
    sim.set_output_dir(tmp_path / "run")
    with pytest.raises(ValueError, match=message):
        sim.input_summary()
    with pytest.raises(ValueError, match=message):
        sim.mesh()


@pytest.mark.parametrize(
    ("sweep", "error", "match"),
    [
        ({"start": 0.0, "stop": 1e9, "points": 2}, ValueError, "start must be above 0"),
        (
            {"start": -1e9, "stop": 1e9, "points": 2},
            ValueError,
            "start must be above 0",
        ),
        (
            {"start": 2e9, "stop": 2e9, "points": 2},
            ValueError,
            "stop must be above start",
        ),
        (
            {"start": 2e9, "stop": 1e9, "points": 2},
            ValueError,
            "stop must be above start",
        ),
        (
            {"start": 1e9, "stop": 2e9, "points": 1},
            ValueError,
            "points must be at least 2",
        ),
        (
            {"start": 1e9, "stop": 2e9, "points": 2.0},
            TypeError,
            "points must be an integer",
        ),
        (
            {"start": 1e9, "stop": 2e9, "points": True},
            TypeError,
            "points must be an integer",
        ),
        (
            {"start": "1e9", "stop": 2e9, "points": 2},
            TypeError,
            "start must be a number",
        ),
        (
            {"start": 1e9, "stop": float("inf"), "points": 2},
            ValueError,
            "stop must be finite",
        ),
        (
            {"start": 1e9, "stop": 2e9, "points": 2, "adaptive_tolerance": 0.0},
            ValueError,
            "adaptive_tolerance must be above 0",
        ),
        (
            {"start": 1e9, "stop": 2e9, "points": 2, "adaptive_max_samples": 10},
            ValueError,
            "no adaptive_tolerance turns that sweep on",
        ),
        (
            {
                "start": 1e9,
                "stop": 2e9,
                "points": 2,
                "adaptive_tolerance": 1e-3,
                "adaptive_max_samples": 0,
            },
            ValueError,
            "adaptive_max_samples must be at least 1",
        ),
    ],
)
def test_set_driven_refuses_a_sweep_palace_cannot_sample(sweep, error, match):
    """Each refusal names ``set_driven`` and leaves the earlier sweep as it was."""
    sim = DrivenSim()
    sim.set_driven(**UNIFORM)
    before = sim.sweep
    with pytest.raises(error, match=match) as raised:
        sim.set_driven(**sweep)
    assert "Change it in the call to set_driven" in str(raised.value)
    assert sim.sweep == before


def test_amr_is_off_by_default_and_needs_its_tolerance_when_turned_on():
    """No AMR unless asked (issue 187); an omitted control keeps its value; enabled AMR states its tolerance."""
    sim = DrivenSim()
    assert (
        sim.numerical.amr_max_passes,
        sim.numerical.amr_nonconformal,
        sim.numerical.amr_tolerance,
    ) == (0, True, None)
    sim.set_numerical(order=2)
    assert (sim.numerical.order, sim.numerical.amr_max_passes) == (2, 0)
    sim.set_numerical(amr_max_passes=2)
    with pytest.raises(
        ValueError, match="Enabled Driven AMR requires explicit amr_tolerance"
    ):
        sim.numerical.check_ready_for("Driven")


def test_a_convergence_declaration_is_refused():
    """Driven has no convergence quantity yet, so no criterion is taken; removing a declaration still works."""
    sim = DrivenSim()
    with pytest.raises(ValueError, match="DrivenSim takes no convergence quantity yet"):
        sim.set_convergence(ErrorIndicator().below(0.1))
    sim.set_convergence(None)
    assert sim.convergence is None


# ---- What it reports -------------------------------------------------------------------------------------


def test_simulation_drc_reports_the_port_as_its_excitation(monkeypatch):
    """The check models the plan as ``mesh()`` does and says what Driven makes of the port."""
    report = DrivenSim.simulation_drc(driven_plan(monkeypatch))
    assert report.passed
    assert report.lumped_elements == (
        "P1 chip/P1: LumpedPort, R 50.0 ohm, excitation 1: port 1 of the S-matrix, "
        "driven from chip/pad_0000 to chip/electrode_0000",
    )


def test_input_summary_shows_the_sweep_and_the_port(monkeypatch):
    """The backend block holds the sweep in Hz and the port with what Driven makes of it."""
    sim = DrivenSim()
    sim.set_plan(driven_plan(monkeypatch).prepare())
    assert sim.input_summary().data["backend"]["sweep"] is None
    sim.set_driven(start=2e9, stop=10e9, points=81, adaptive_tolerance=1e-3)
    backend = sim.input_summary().data["backend"]
    assert backend["sweep"] == {
        "kind": "adaptive",
        "start_hz": 2e9,
        "stop_hz": 10e9,
        "points": 81,
        "adaptive_tolerance": 1e-3,
        "adaptive_max_samples": None,
    }
    (port,) = backend["lumped_elements"]
    assert (port["kind"], port["impedance_ohm"], port["signal"], port["reference"]) == (
        "port",
        50.0,
        "chip/pad_0000",
        "chip/electrode_0000",
    )
    assert port["realization"].startswith("LumpedPort, R 50.0 ohm, excitation 1")


# ---- The handoff, and the run folder Palace returns ------------------------------------------------------


def handed_off(monkeypatch, tmp_path):
    """A Driven run folder after ``prepare_handoff()``, on Route B."""
    sim = meshed(monkeypatch, tmp_path, driven_plan(monkeypatch))
    sim.set_driven(**UNIFORM)
    sim.write_config()
    return sim.prepare_handoff(
        profile="direct-local", executable="palace", resources={"processes": 1}
    )


def test_the_handoff_records_driven_and_the_tables_palace_writes(monkeypatch, tmp_path):
    """The handoff names the problem; the recorder requires the six tables stock Palace 0.18.1 wrote (issue 187)."""
    handoff = handed_off(monkeypatch, tmp_path)
    metadata = json.loads((handoff.run_dir / HANDOFF_METADATA).read_text())
    assert metadata["problem"] == "Driven"
    assert palace_problem("Driven").written_families(
        lumped_ports=True, surfaces=True
    ) == ("port-S", "port-I", "port-V", "domain-E", "surface-Q", "error-indicators")
    assert palace_problem("Driven").written_families(
        lumped_ports=True, surfaces=False
    ) == ("port-S", "port-I", "port-V", "domain-E", "error-indicators")
