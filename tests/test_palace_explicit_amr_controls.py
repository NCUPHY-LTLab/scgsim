"""Explicit case controls through public state and pure config construction.

The structural group records below exercise config encoding, not a native mesh.
"""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scgsim.palace import EigenmodeSim, ElectrostaticSim
from scgsim.palace.preparation.config import (
    build_eigenmode_config,
    configure_numerical_controls,
)


def _config(sim):
    return build_eigenmode_config(
        groups={
            "volumes": {"vacuum": {
                "phys_group": 1,
                "physical_attribute": {"material_ids": ["vacuum"]},
            }},
            "pec_surfaces": {"conductor": {"phys_group": 2}},
        },
        ports=(),
        materials={"vacuum": {
            "kind": "vacuum", "permittivity": 1.0, "loss_tangent": 0.0,
        }},
        surface_epr_specs={},
        numerical=sim.numerical,
        num_modes=sim.num_modes,
        target_hz=sim.target_hz,
        eigenmode_tolerance=sim.eigenmode_tolerance,
        save_fields=sim.save_fields,
        mesh_path=Path("palace.msh"),
    ).config


def test_new_eigenmode_keeps_amr_unspecified_until_case_configuration():
    sim = EigenmodeSim()
    assert sim.numerical["amr_nonconformal"] is True
    assert all(sim.numerical[key] is None for key in (
        "amr_max_passes", "amr_tolerance", "amr_update_fraction",
    ))
    sim.set_mesh(algorithm_3d="HXT")
    sim.set_numerical(order=2)
    with pytest.raises(ValueError, match="explicit amr_max_passes"):
        _config(sim)


@pytest.mark.parametrize("controls", (
    {"amr_max_passes": 3},
    {"amr_max_passes": 3, "amr_tolerance": 0.02},
    {"amr_max_passes": 3, "amr_update_fraction": 0.15},
))
def test_enabled_amr_cannot_generate_config_with_incomplete_case_controls(controls):
    sim = EigenmodeSim()
    sim.set_numerical(**controls)
    with pytest.raises(ValueError, match="explicit amr_tolerance and amr_update_fraction"):
        _config(sim)


def test_complete_case_controls_survive_later_setters_and_json_transport():
    sim = EigenmodeSim()
    sim.set_numerical(
        order=2, amr_max_passes=3, amr_tolerance=0.02,
        amr_update_fraction=0.15, amr_nonconformal=False,
    )
    sim.set_numerical(order=2, tolerance=1e-7)
    sim.set_mesh(threads=2)
    encoded = json.loads(json.dumps(sim.numerical))
    assert configure_numerical_controls(**encoded) == sim.numerical
    assert _config(sim)["Model"]["Refinement"] == {
        "UniformLevels": 0, "MaxIts": 3, "Tol": 0.02,
        "UpdateFraction": 0.15, "Nonconformal": False,
    }
    assert _config(sim)["Solver"]["Order"] == 2


def test_explicit_zero_disables_amr_without_inventing_tolerance_or_fraction():
    sim = EigenmodeSim()
    sim.set_numerical(amr_max_passes=0)
    sim.set_numerical(order=2)
    sim.set_mesh(geometry_order=2)
    assert _config(sim)["Model"]["Refinement"] == {
        "UniformLevels": 0, "MaxIts": 0, "Nonconformal": True,
    }
    encoded = json.loads(json.dumps(sim.numerical))
    assert encoded["amr_max_passes"] == 0
    assert encoded["amr_tolerance"] is None
    assert encoded["amr_update_fraction"] is None
    assert configure_numerical_controls(**encoded) == sim.numerical


@pytest.mark.parametrize("invalid", (
    {"amr_max_passes": -1},
    {"amr_tolerance": 0.0},
    {"amr_update_fraction": 0.0},
    {"amr_nonconformal": "true"},
))
def test_invalid_case_controls_do_not_partially_apply_other_changes(invalid):
    sim = EigenmodeSim()
    sim.set_numerical(amr_max_passes=3, amr_tolerance=0.02, amr_update_fraction=0.15)
    before = deepcopy(sim.numerical)
    with pytest.raises((ValueError, TypeError)):
        sim.set_numerical(order=2, **invalid)
    assert sim.numerical == before


def test_electrostatic_retains_existing_sizing_and_amr_defaults():
    sim = ElectrostaticSim()
    assert sim.numerical == configure_numerical_controls()
    assert sim.numerical["size_policy"] == "scgsim_default"
    assert (sim.numerical["refined_mesh_size"], sim.numerical["max_mesh_size"]) == (5.0, 300.0)
    assert sim.numerical["amr_nonconformal"] is False
    assert sim.numerical["amr_max_passes"] == 0
    assert sim.numerical["amr_tolerance"] == 0.01
    assert sim.numerical["amr_update_fraction"] is None
