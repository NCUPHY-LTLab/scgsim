"""Each Palace ``simulation_drc`` takes the ``GeometryPlan`` its simulation is given and adds nothing to it (Issue #266).

It refuses anything but a plan, leaves the caller's plan as it was, and on a plan without port sheets the two
Palace classes check one model.  No test here meshes or solves.
"""

from __future__ import annotations

import pytest
from test_simulation_drc import (
    BANDED,
    NO_VACUUM_STACK,
    SIMULATIONS,
    check_sidecars,
    configured,
    layout_plan,
    two_pads,
)


@pytest.mark.parametrize("simulation", SIMULATIONS)
def test_a_plan_without_a_vacuum_is_refused_as_mesh_refuses_it(
    monkeypatch, tmp_path, simulation
):
    """A plan with no vacuum is refused with the error ``mesh()`` raises for it."""
    component = two_pads(monkeypatch, gap_um=10.0)
    plan = layout_plan(component, layer_stack=NO_VACUUM_STACK, vacuum_padding_um=None)
    sim = configured(simulation, plan.prepare())
    sim.set_output_dir(tmp_path / "run")
    with pytest.raises(ValueError, match="has no vacuum region") as expected:
        sim.mesh()
    with pytest.raises(ValueError) as raised:  # noqa: PT011 -- the message is compared whole
        simulation.simulation_drc(plan)
    assert str(raised.value) == str(expected.value)


@pytest.mark.parametrize("simulation", SIMULATIONS)
def test_the_check_leaves_the_callers_plan_as_it_was(monkeypatch, simulation):
    """The plan prepares to the same snapshot after the check as before it: the check changes nothing on it."""
    plan = layout_plan(
        two_pads(monkeypatch, gap_um=10.0, port=True), surface_epr_specs=BANDED
    )
    before = plan.prepare()
    assert simulation.simulation_drc(plan).passed
    after = plan.prepare()
    assert (after.gds_sha256, after.stack, after.surface_epr_specs) == (
        before.gds_sha256,
        before.stack,
        before.surface_epr_specs,
    )


@pytest.mark.parametrize("simulation", SIMULATIONS)
def test_a_first_argument_that_is_not_a_plan_is_refused_with_the_plan_to_build(
    monkeypatch, simulation
):
    """A Component or a prepared snapshot is refused before anything is read, with the plan to build."""
    component = two_pads(monkeypatch, gap_um=10.0)
    for value in (component, layout_plan(component).prepare()):
        with pytest.raises(TypeError) as raised:
            simulation.simulation_drc(value)
        message = str(raised.value)
        assert message.startswith(
            f"the first argument of {simulation.__name__}.simulation_drc: must be the "
            f"GeometryPlan the simulation is given, not {type(value).__name__}; build it as "
            "plan = GeometryPlan(component=..., root_id=..., layer_stack=..., "
            "material_records=..., coupon_padding_um=...); plan.set_nets({...})"
        )
        assert message.endswith(
            f"Change it in the call to {simulation.__name__}.simulation_drc."
        )


def test_without_port_sheets_both_palace_checks_build_one_model(monkeypatch):
    """On a plan with no port sheet, ``EigenmodeSim`` and ``ElectrostaticSim`` check the same SGB model.

    Eigenmode cuts every authored port sheet as a lumped port and Electrostatic cuts none, so without port
    sheets nothing tells their geometry apart: one check stands for the other on the same route and specs.
    """
    component = two_pads(monkeypatch, gap_um=10.0)
    models = [
        check_sidecars(monkeypatch, simulation, component, surface_epr_specs=BANDED)
        for simulation in SIMULATIONS
    ]
    assert models[0] == models[1]


def test_with_a_port_sheet_the_two_palace_checks_build_different_models(monkeypatch):
    """On a plan with a port sheet, Eigenmode cuts the sheet's rectangle as a lumped port; Electrostatic keeps it as drawn.

    Both models carry the sheet's port group, but its faces and edges differ, so a pass of one check does
    not stand for the other.
    """
    component = two_pads(monkeypatch, gap_um=10.0, port=True)
    (eigenmode,), (electrostatic,) = (
        check_sidecars(monkeypatch, simulation, component) for simulation in SIMULATIONS
    )
    for sidecar in (eigenmode, electrostatic):
        assert [name for name in sidecar["group"] if name.startswith("PORT::")] == [
            "PORT::PORT_SHEET__chip/P1__0000"
        ]
    assert eigenmode["face"] != electrostatic["face"]
    assert eigenmode["carrier"] != electrostatic["carrier"]
