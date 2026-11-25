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
from itertools import pairwise

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
    Linear,
    Log,
    Points,
)
from scgsim.palace.config.driven_sweep import palace_frequencies_ghz
from scgsim.palace.config.numerical import NumericalControls
from scgsim.palace.results.driven_tables import (
    requested_frequencies_ghz,
    requested_sweep,
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
UNIFORM = {"frequencies": [Linear(4e9, 8e9, points=5)]}


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


# The band 10 MHz apart and a window 10 kHz apart around 7.32 GHz: the example of the S-parameter guide.
BAND_AND_WINDOW = [Linear(4e9, 9e9, points=501), Linear(7.30e9, 7.34e9, points=4001)]


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
            {"frequencies": [Linear(2e9, 10e9, points=81)], "adaptive_tolerance": 1e-3},
            {
                "Samples": [
                    {"Type": "Linear", "MinFreq": 2.0, "MaxFreq": 10.0, "NSample": 81}
                ],
                "AdaptiveTol": 1e-3,
            },
        ),
        (
            {
                "frequencies": [Linear(2.5e9, 7.25e9, points=20)],
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
        (
            {"frequencies": [Log(1e8, 1e10, points=21)]},
            {
                "Samples": [
                    {"Type": "Log", "MinFreq": 0.1, "MaxFreq": 10.0, "NSample": 21}
                ]
            },
        ),
        (
            {"frequencies": [Points([7.3e9])]},
            {"Samples": [{"Type": "Point", "Freq": [7.3]}]},
        ),
        (
            {
                "frequencies": [
                    *BAND_AND_WINDOW,
                    Points([5.5e9, 2.25e9]),
                    Log(1e9, 1e10, points=11),
                ],
                "adaptive_tolerance": 1e-3,
            },
            {
                "Samples": [
                    {"Type": "Linear", "MinFreq": 4.0, "MaxFreq": 9.0, "NSample": 501},
                    {
                        "Type": "Linear",
                        "MinFreq": 7.3,
                        "MaxFreq": 7.34,
                        "NSample": 4001,
                    },
                    {"Type": "Point", "Freq": [5.5, 2.25]},
                    {"Type": "Log", "MinFreq": 1.0, "MaxFreq": 10.0, "NSample": 11},
                ],
                "AdaptiveTol": 1e-3,
            },
        ),
    ],
    ids=["uniform", "adaptive", "adaptive-capped", "log", "one-point", "mixed"],
)
def test_each_range_is_one_samples_entry_in_ghz(monkeypatch, tmp_path, sweep, block):
    """Each range in Hz is one ``Solver.Driven.Samples`` entry in GHz, in the order given; uniform has no ``AdaptiveTol``."""
    sim = meshed(monkeypatch, tmp_path, driven_plan(monkeypatch))
    assert written_config(sim, **sweep)["Solver"]["Driven"] == block


def test_the_written_config_reads_back_as_the_sweep_set(monkeypatch, tmp_path):
    """The read-back's sweep of the config ``write_config()`` wrote is the one set, and so are its frequencies."""
    sim = meshed(monkeypatch, tmp_path, driven_plan(monkeypatch))
    config = written_config(
        sim,
        frequencies=[*BAND_AND_WINDOW, Points([2.25e9]), Log(1e9, 1e10, points=11)],
        adaptive_tolerance=1e-3,
        adaptive_max_samples=30,
    )
    assert requested_sweep(config) == sim.sweep
    assert requested_frequencies_ghz(config) == palace_frequencies_ghz(
        sim.sweep.samples()
    )
    # 4,497 of the band and window, then 2.25 GHz and the log range's 11, none of them on the band's grid.
    assert len(requested_frequencies_ghz(config)) == sim.sweep.frequency_count == 4509


# ---- The frequencies Palace solves at ----------------------------------------------------------------------


def test_a_linear_range_is_evenly_spaced_with_both_ends():
    """``NSample`` frequencies from ``MinFreq`` to ``MaxFreq``, as Palace's ``ConstructLinearRange`` spaces them."""
    entry = Linear(2e9, 10e9, points=5).samples_entry()
    assert palace_frequencies_ghz([entry]) == (2.0, 4.0, 6.0, 8.0, 10.0)


def test_a_log_range_is_evenly_spaced_on_a_log_scale():
    """1 to 10 GHz in 5 points is 10^(k/4) GHz: 1, 10^(1/4), sqrt(10), 10^(3/4) and 10 GHz."""
    entry = Log(1e9, 1e10, points=5).samples_entry()
    assert palace_frequencies_ghz([entry]) == pytest.approx(
        (1.0, 1.77827941003892, 3.16227766016838, 5.62341325190349, 10.0), rel=1e-14
    )
    # A range of two decades in 3 points has the decade between.
    entry = Log(1e8, 1e10, points=3).samples_entry()
    assert palace_frequencies_ghz([entry]) == pytest.approx((0.1, 1.0, 10.0), rel=1e-15)


def test_the_ranges_merge_sorted_and_each_shared_frequency_once():
    """The band's 501 and the window's 4,001 less the five they share: 7.30, 7.31, 7.32, 7.33 and 7.34 GHz.

    Palace sorts every range's frequencies together, whatever the order the ranges are given in.
    """
    band, window = BAND_AND_WINDOW
    merged = palace_frequencies_ghz([window.samples_entry(), band.samples_entry()])
    assert len(merged) == 501 + 4001 - 5
    assert (merged[0], merged[-1]) == (4.0, 9.0)
    assert all(high > low for low, high in pairwise(merged))
    # 7.29 GHz is the band's last below the window, 10 MHz away; the window then steps 10 kHz.
    around = [f for f in merged if 7.28 < f < 7.36]
    assert len(around) == 4001 + 2
    assert around[:3] == pytest.approx([7.29, 7.30, 7.30001], rel=1e-15)
    assert around[-2:] == pytest.approx([7.34, 7.35], rel=1e-15)


def test_frequencies_within_1_hz_of_the_last_kept_are_one():
    """Palace's ``std::unique`` keeps a frequency 1 Hz or more above the last one kept, not the one before.

    0.6 Hz above 7.3 GHz goes; 1.2 Hz above stays, though it is only 0.6 Hz above the one that went.
    """
    entries = [{"Type": "Point", "Freq": [7.3 + 1.2e-9, 7.3, 2.0, 7.3 + 0.6e-9]}]
    assert palace_frequencies_ghz(entries) == (2.0, 7.3, 7.3 + 1.2e-9)


def test_the_fem_order_is_two_unless_another_is_given(monkeypatch, tmp_path):
    """Second-order elements unless another order is given (issue 187), in the config too; Eigenmode keeps first."""
    sim = meshed(monkeypatch, tmp_path, driven_plan(monkeypatch))
    assert written_config(sim)["Solver"]["Order"] == 2
    handed_over = DrivenSim(numerical=NumericalControls(tolerance=1e-8))
    assert handed_over.numerical.order == 2
    first_order = DrivenSim()
    first_order.set_numerical(order=1)
    assert first_order.numerical.order == 1
    assert EigenmodeSim().numerical.order == 1


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
    with pytest.raises(
        ValueError, match=r"call set_driven\(\[Linear\(\.\.\.\), \.\.\.\]\)"
    ):
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
        pytest.param(
            lambda: {"frequencies": []},
            ValueError,
            r"frequencies must be at least one range, not \[\]",
            id="no range",
        ),
        pytest.param(
            lambda: {"frequencies": Linear(4e9, 8e9, points=5)},
            TypeError,
            "frequencies must be a list of Linear, Log and Points ranges",
            id="a range alone",
        ),
        pytest.param(
            lambda: {"frequencies": [Linear(4e9, 8e9, points=5), (4e9, 8e9, 5)]},
            TypeError,
            r"frequencies\[1\] must be a Linear, Log or Points range",
            id="not a range",
        ),
        pytest.param(
            lambda: {"frequencies": [Points([7.3e9, 7.3e9 + 5.0])]},
            ValueError,
            r"7\.3 GHz and 7\.300000005 GHz are 5 Hz apart.* 10 Hz here, so their rows could not be read back",
            id="5 Hz apart",
        ),
        pytest.param(
            lambda: {"frequencies": [*BAND_AND_WINDOW, Points([7.320000005e9])]},
            ValueError,
            r"7\.32 GHz and 7\.320000005 GHz are 5 Hz apart",
            id="across ranges",
        ),
        pytest.param(
            lambda: {"frequencies": [Points([12e9, 12e9 + 80.0])]},
            ValueError,
            r"12 GHz and 12\.00000008 GHz are 80 Hz apart.* 100 Hz here",
            id="80 Hz apart above 10 GHz",
        ),
        pytest.param(
            lambda: {"frequencies": [Points([7.3e9])], "adaptive_tolerance": 1e-3},
            ValueError,
            "an adaptive sweep needs at least 3 distinct frequencies, and the ranges give 1",
            id="adaptive at one frequency",
        ),
        pytest.param(
            lambda: {
                "frequencies": [Linear(7e9, 8e9, points=2)],
                "adaptive_tolerance": 1e-3,
            },
            ValueError,
            "an adaptive sweep needs at least 3 distinct frequencies, and the ranges give 2",
            id="adaptive at two frequencies",
        ),
        pytest.param(
            lambda: {
                "frequencies": [Linear(7e9, 8e9, points=2), Points([7e9 + 0.5])],
                "adaptive_tolerance": 1e-3,
            },
            ValueError,
            "the ranges give 2",
            id="adaptive at three frequencies, two within 1 Hz",
        ),
        pytest.param(
            lambda: {"frequencies": [Linear(0.0, 1e9, points=2)]},
            ValueError,
            "the Linear range: start_hz must be above 0 Hz",
            id="Linear from 0",
        ),
        pytest.param(
            lambda: {"frequencies": [Linear(-1e9, 1e9, points=2)]},
            ValueError,
            "start_hz must be above 0 Hz",
            id="Linear from below 0",
        ),
        pytest.param(
            lambda: {"frequencies": [Linear(2e9, 2e9, points=2)]},
            ValueError,
            "stop_hz must be above start_hz",
            id="Linear of no width",
        ),
        pytest.param(
            lambda: {"frequencies": [Log(2e9, 1e9, points=2)]},
            ValueError,
            "the Log range: stop_hz must be above start_hz",
            id="Log downward",
        ),
        pytest.param(
            lambda: {"frequencies": [Linear(1e9, 2e9, points=1)]},
            ValueError,
            "points must be at least 2",
            id="Linear of one point",
        ),
        pytest.param(
            lambda: {"frequencies": [Log(1e9, 2e9, points=1)]},
            ValueError,
            "the Log range: points must be at least 2",
            id="Log of one point",
        ),
        pytest.param(
            lambda: {"frequencies": [Linear(1e9, 2e9, points=2.0)]},
            TypeError,
            "points must be an integer",
            id="points a float",
        ),
        pytest.param(
            lambda: {"frequencies": [Linear(1e9, 2e9, points=True)]},
            TypeError,
            "points must be an integer",
            id="points a bool",
        ),
        pytest.param(
            lambda: {"frequencies": [Linear("1e9", 2e9, points=2)]},
            TypeError,
            "start_hz must be a number",
            id="start a string",
        ),
        pytest.param(
            lambda: {"frequencies": [Log(1e9, float("inf"), points=2)]},
            ValueError,
            "stop_hz must be finite",
            id="stop infinite",
        ),
        pytest.param(
            lambda: {"frequencies": [Points([])]},
            ValueError,
            "the Points range: frequencies_hz must be at least one frequency",
            id="no point",
        ),
        pytest.param(
            lambda: {"frequencies": [Points(7.3e9)]},
            TypeError,
            "frequencies_hz must be a list of frequencies in Hz",
            id="a point alone",
        ),
        pytest.param(
            lambda: {"frequencies": [Points([7.3e9, -1.0])]},
            ValueError,
            r"frequencies_hz\[1\] must be above 0 Hz",
            id="a negative point",
        ),
        pytest.param(
            lambda: {"frequencies": [Points([float("nan")])]},
            ValueError,
            r"frequencies_hz\[0\] must be finite",
            id="a NaN point",
        ),
        pytest.param(
            lambda: {**UNIFORM, "adaptive_tolerance": 0.0},
            ValueError,
            "adaptive_tolerance must be above 0",
            id="tolerance 0",
        ),
        pytest.param(
            lambda: {**UNIFORM, "adaptive_max_samples": 10},
            ValueError,
            "no adaptive_tolerance turns that sweep on",
            id="cap without tolerance",
        ),
        pytest.param(
            lambda: {**UNIFORM, "adaptive_tolerance": 1e-3, "adaptive_max_samples": 0},
            ValueError,
            "adaptive_max_samples must be at least 1",
            id="cap 0",
        ),
    ],
)
def test_set_driven_refuses_a_sweep_palace_cannot_sample(sweep, error, match):
    """Each refusal names ``set_driven`` and leaves the earlier sweep as it was.

    A range checks itself when it is built, inside the call; the sweep checks the list and what Palace merges.
    """
    sim = DrivenSim()
    sim.set_driven(**UNIFORM)
    before = sim.sweep
    with pytest.raises(error, match=match) as raised:
        sim.set_driven(**sweep())
    assert "Change it in the call to set_driven" in str(raised.value)
    assert sim.sweep == before


def test_a_uniform_sweep_may_solve_one_frequency(monkeypatch, tmp_path):
    """A full solve at one frequency is a sweep of one point; only the adaptive sweep needs three."""
    sim = meshed(monkeypatch, tmp_path, driven_plan(monkeypatch))
    config = written_config(sim, frequencies=[Points([7.3e9])])
    assert config["Solver"]["Driven"] == {"Samples": [{"Type": "Point", "Freq": [7.3]}]}
    assert sim.sweep.frequency_count == 1


def test_amr_is_off_by_default_and_needs_its_tolerance_when_turned_on():
    """No AMR unless asked (issue 187); an omitted control keeps its value; enabled AMR states its tolerance."""
    sim = DrivenSim()
    assert (
        sim.numerical.amr_max_passes,
        sim.numerical.amr_nonconformal,
        sim.numerical.amr_tolerance,
    ) == (0, True, None)
    sim.set_numerical(tolerance=1e-8)
    assert sim.numerical.amr_max_passes == 0
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
    """The backend block holds the sweep's ranges in Hz and its merged count, and the port as Driven makes it."""
    sim = DrivenSim()
    sim.set_plan(driven_plan(monkeypatch).prepare())
    assert sim.input_summary().data["backend"]["sweep"] is None
    sim.set_driven(
        [*BAND_AND_WINDOW, Points([2.25e9]), Log(1e9, 1e10, points=11)],
        adaptive_tolerance=1e-3,
    )
    backend = sim.input_summary().data["backend"]
    assert backend["sweep"] == {
        "sampling": "adaptive",
        "ranges": [
            {"kind": "linear", "start_hz": 4e9, "stop_hz": 9e9, "points": 501},
            {"kind": "linear", "start_hz": 7.30e9, "stop_hz": 7.34e9, "points": 4001},
            {"kind": "points", "frequencies_hz": [2.25e9]},
            {"kind": "log", "start_hz": 1e9, "stop_hz": 1e10, "points": 11},
        ],
        "frequency_count": 4509,
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
