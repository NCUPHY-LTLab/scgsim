"""``mesh()`` refuses a model with no vacuum region, before it writes anything (Issue #203).

The vacuum is the plan's: the stack's own, or ``GeometryPlan.set_vacuum_region``.  No test here runs Gmsh or a
solver: where ``mesh()`` is allowed to proceed, its meshing stage is replaced by a marker error.
"""

from __future__ import annotations

import pytest
from test_simulation_drc import (
    LAYER_STACK,
    MATERIALS,
    NO_VACUUM_STACK,
    SIMULATIONS,
    MeshingReached,
    check,
    configured,
    two_pads,
)

import scgsim.palace.mesh.build as build_module
from scgsim.geometry import GeometryPlan
from scgsim.palace import EigenmodeSim

VACUUM_PADDING_UM = (0.0, 0.0, 1000.0)


def mesh_sim(monkeypatch, simulation, layer_stack, output_dir, vacuum_padding_um=None):
    """A ``simulation`` on the two-pad plan of ``layer_stack``, with its output folder set and meshing replaced.

    ``vacuum_padding_um`` goes to the plan's ``set_vacuum_region``; None sets no vacuum.
    """
    plan = GeometryPlan(
        two_pads(monkeypatch, gap_um=10.0, port=True),
        root_id="chip",
        layer_stack=layer_stack,
        material_records=MATERIALS,
        coupon_padding_um=0.0,
    )
    if vacuum_padding_um is not None:
        plan.set_vacuum_region(vacuum_padding_um)
    sim = configured(simulation, plan.prepare())
    sim.set_output_dir(output_dir)

    def meshing_reached(*_args, **_kwargs):
        raise MeshingReached

    monkeypatch.setattr(build_module, "mesh_sgb", meshing_reached)
    return sim


@pytest.mark.parametrize("simulation", SIMULATIONS)
def test_mesh_refuses_a_model_with_no_vacuum(monkeypatch, tmp_path, simulation):
    """No vacuum in the stack and none set on the plan: ``mesh()`` raises and writes no file."""
    output_dir = tmp_path / "run"
    sim = mesh_sim(monkeypatch, simulation, NO_VACUUM_STACK, output_dir)

    with pytest.raises(ValueError, match="the model: has no vacuum region") as raised:
        sim.mesh()

    assert "call plan.set_vacuum_region(...) before prepare()" in str(raised.value)
    assert "sim.set_vacuum_region" not in str(raised.value)
    assert not output_dir.exists() or not any(output_dir.iterdir())


@pytest.mark.parametrize("simulation", SIMULATIONS)
def test_a_vacuum_set_on_the_plan_builds(monkeypatch, tmp_path, simulation):
    """The same layout with ``set_vacuum_region`` on the plan reaches the mesher."""
    sim = mesh_sim(
        monkeypatch, simulation, NO_VACUUM_STACK, tmp_path / "run", VACUUM_PADDING_UM
    )

    with pytest.raises(MeshingReached):
        sim.mesh()


@pytest.mark.parametrize("simulation", SIMULATIONS)
def test_a_vacuum_in_the_plan_builds(monkeypatch, tmp_path, simulation):
    """A plan whose stack carries its own vacuum reaches the mesher."""
    sim = mesh_sim(monkeypatch, simulation, LAYER_STACK, tmp_path / "run")

    with pytest.raises(MeshingReached):
        sim.mesh()


def test_input_summary_refuses_a_model_with_no_vacuum(monkeypatch, tmp_path):
    """``input_summary()`` meets the refusal in the preparation it shares with ``mesh()``."""
    sim = mesh_sim(monkeypatch, EigenmodeSim, NO_VACUUM_STACK, tmp_path / "run")

    with pytest.raises(ValueError, match="the model: has no vacuum region"):
        sim.input_summary()


@pytest.mark.parametrize("simulation", SIMULATIONS)
def test_simulation_drc_adds_no_vacuum_of_its_own(monkeypatch, simulation):
    """``simulation_drc`` on a plan with no vacuum is refused as ``mesh()`` refuses it (Issue #266); with the plan's vacuum it passes."""
    component = two_pads(monkeypatch, gap_um=10.0, port=True)

    with pytest.raises(ValueError, match="the model: has no vacuum region"):
        check(
            simulation, component, layer_stack=NO_VACUUM_STACK, vacuum_padding_um=None
        )
    assert check(simulation, component, layer_stack=NO_VACUUM_STACK).passed
