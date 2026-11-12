"""The lifecycle of both Palace simulations: which call needs which stage, and how far each call falls back.

The rules are the table in ``scgsim.palace.simulation.lifecycle``; these tests drive them through the public
calls of ``EigenmodeSim`` and ``ElectrostaticSim`` and check each refusal names the call that fixes it.
No test here meshes: a stage that needs a mesh is reached by standing in for ``mesh()``'s outputs.
"""

from __future__ import annotations

import inspect

import pytest
from test_palace_mesh_controls import two_pad_snapshot

from scgsim.palace import EigenmodeSim, ElectrostaticSim
from scgsim.palace.simulation.core import PalaceSimulation
from scgsim.palace.simulation.lifecycle import CALLS, Stage

SIMS = [EigenmodeSim, ElectrostaticSim]


def handed_off(sim):
    """``sim`` as if ``mesh()``, ``write_config()`` and ``prepare_handoff()`` had run."""
    sim._mesh_result = object()  # noqa: SLF001 -- stands for mesh()
    sim.config_path = "config.json"  # type: ignore[assignment]
    sim.handoff_plan = object()  # type: ignore[assignment]
    sim._stage = Stage.HANDED_OFF  # noqa: SLF001 -- stands for the three calls
    return sim


@pytest.mark.parametrize("sim_type", SIMS)
def test_every_public_call_that_changes_or_needs_state_is_in_the_table(sim_type):
    """Each public method of the simulation is a row of the lifecycle table, except the modeling check."""
    public = {
        name
        for name, _ in inspect.getmembers(sim_type, inspect.isfunction)
        if not name.startswith("_")
    }
    assert public - set(CALLS) == set()
    assert hasattr(sim_type, "simulation_drc")


@pytest.mark.parametrize("sim_type", SIMS)
@pytest.mark.parametrize(
    ("call", "error", "fix"),
    [
        ("mesh", ValueError, r"call set_plan\(\) before mesh\(\)"),
        ("mesh_summary", RuntimeError, r"call mesh\(\) before mesh_summary\(\)"),
        (
            "write_config",
            ValueError,
            r"call set_plan\(\) and mesh\(\) before write_config\(\)",
        ),
    ],
)
def test_a_call_before_its_stage_names_the_call_that_fixes_it(
    sim_type, call, error, fix
):
    """A new simulation is unbound: each call that needs a later stage refuses with its own message."""
    with pytest.raises(error, match=fix):
        getattr(sim_type(), call)()


@pytest.mark.parametrize("sim_type", SIMS)
def test_a_handoff_before_the_config_names_write_config(sim_type):
    """``prepare_handoff()`` needs ``CONFIG_WRITTEN``."""
    with pytest.raises(
        ValueError,
        match=r"call mesh\(\) and write_config\(\) before prepare_handoff\(\)",
    ):
        sim_type().prepare_handoff(profile="direct-local", executable="palace")


@pytest.mark.parametrize("sim_type", SIMS)
def test_surface_epr_before_a_plan_names_set_plan(sim_type):
    """The specs are checked against the bound plan's materials, so ``set_surface_epr`` needs ``BOUND``."""
    with pytest.raises(
        ValueError, match=r"call set_plan\(\) before set_surface_epr\(\)"
    ):
        sim_type().set_surface_epr(representation="B", specs={})


@pytest.mark.parametrize("sim_type", SIMS)
def test_mesh_without_an_output_folder_names_set_output_dir(sim_type, monkeypatch):
    """A bound simulation still needs its run folder."""
    sim = sim_type()
    sim.set_plan(two_pad_snapshot(monkeypatch))
    with pytest.raises(ValueError, match=r"call set_output_dir\(\) before mesh\(\)"):
        sim.mesh()


def test_only_electrostatic_mesh_requires_surface_epr(monkeypatch, tmp_path):
    """The one rule that differs by problem is a class attribute; Electrostatic refuses before any build."""
    assert ElectrostaticSim.mesh_requires_surface_epr is True
    assert EigenmodeSim.mesh_requires_surface_epr is False
    sim = ElectrostaticSim()
    sim.set_plan(two_pad_snapshot(monkeypatch))
    sim.set_output_dir(tmp_path / "run")
    with pytest.raises(ValueError, match=r"call set_surface_epr\(\) before mesh\(\)"):
        sim.mesh()
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("sim_type", SIMS)
def test_a_geometry_setter_falls_back_to_bound(sim_type, monkeypatch, tmp_path):
    """A value ``mesh()`` reads discards the mesh, config and handoff."""
    sim = sim_type()
    sim.set_plan(two_pad_snapshot(monkeypatch))
    handed_off(sim)
    sim.set_output_dir(tmp_path)
    assert sim._stage is Stage.BOUND  # noqa: SLF001
    assert (sim._mesh_result, sim.config_path, sim.handoff_plan) == (None, None, None)  # noqa: SLF001


@pytest.mark.parametrize(
    ("sim_type", "setter"),
    [
        (EigenmodeSim, lambda sim: sim.set_eigenmode(num_modes=2)),
        (EigenmodeSim, lambda sim: sim.set_epr_request(surface_interfaces=["MA"])),
        (ElectrostaticSim, lambda sim: sim.add_terminal("pad", net_id="pad")),
        (ElectrostaticSim, lambda sim: sim.add_ground(net_id="ground")),
        (ElectrostaticSim, lambda sim: sim.set_electrostatic(save_fields=1)),
        (EigenmodeSim, lambda sim: sim.set_numerical(amr_max_passes=0)),
        (ElectrostaticSim, lambda sim: sim.set_numerical()),
    ],
)
def test_a_problem_setter_keeps_the_mesh(sim_type, setter, monkeypatch):
    """A value only ``write_config()`` reads discards the config and handoff and keeps the mesh."""
    sim = sim_type()
    sim.set_plan(two_pad_snapshot(monkeypatch))
    handed_off(sim)
    setter(sim)
    assert sim._stage is Stage.MESHED  # noqa: SLF001
    assert sim._mesh_result is not None  # noqa: SLF001
    assert (sim.config_path, sim.handoff_plan) == (None, None)


def test_the_shared_core_is_not_a_simulation_of_its_own():
    """``PalaceSimulation`` is abstract: a user builds ``EigenmodeSim`` or ``ElectrostaticSim``."""
    with pytest.raises(TypeError):
        PalaceSimulation()  # type: ignore[abstract]
