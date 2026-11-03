"""Pure configuration coverage for explicit AMR studies; no native solver runs."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from scgsim.palace import EigenmodeSim, ElectrostaticSim
from scgsim.palace.preparation.config import (
    TerminalBinding,
    build_eigenmode_config,
    build_electrostatic_config,
)
from scgsim.palace.preparation.numerical import (
    build_refinement,
    configure_numerical_controls,
    linear_solver,
)

_GROUPS = {
    "volumes": {
        "vacuum": {
            "phys_group": 1,
            "physical_attribute": {"material_ids": ["vacuum"]},
        }
    },
    "boundary_surfaces": {
        "contact": {
            "sgb_record": "final_physical_group",
            "source": "volume_interface",
            "solver_use": "solver_active",
            "interface_type": "MA",
            "conductor_component_id": "metal",
            "net_id": "net",
            "phys_group": 2,
        }
    },
    "pec_surfaces": {"conductor": {"phys_group": 2}},
}
_MATERIALS = {"vacuum": {"kind": "vacuum", "permittivity": 1.0, "loss_tangent": 0.0}}


def _electrostatic_config(sim: ElectrostaticSim) -> dict[str, Any]:
    return build_electrostatic_config(
        groups=_GROUPS,
        terminals=[TerminalBinding(name="terminal", net_id="net")],
        materials=_MATERIALS,
        save_fields=sim.save_fields,
        unassigned_conductor_policy=sim.unassigned_conductor_policy,
        exterior_boundary_policy=sim.exterior_boundary_policy,
        mesh_path=Path("palace.msh"),
        surface_epr_specs={},
        numerical=sim.numerical,
    ).config


def test_electrostatic_defaults_use_cg_boomeramg_without_changing_case_controls() -> (
    None
):
    """Select the electrostatic solver while retaining all other baseline defaults."""
    sim = ElectrostaticSim()
    assert sim.numerical == configure_numerical_controls()
    config = _electrostatic_config(sim)
    assert config["Solver"]["Linear"] == {
        "Type": "BoomerAMG",
        "KSPType": "CG",
        "Tol": 1e-6,
        "MaxIts": 400,
    }
    assert config["Solver"]["Order"] == 1
    assert config["Solver"]["Device"] == "CPU"
    assert config["Solver"]["Electrostatic"]["Save"] == 0
    assert "MaxNCLevels" not in config["Model"]["Refinement"]
    assert "SaveAdaptDiagnostics" not in config["Model"]["Refinement"]


def test_explicit_study_settings_lower_from_public_setters_without_json_edits() -> None:
    """Encode the caller's complete study settings without promoting them to defaults."""
    sim = ElectrostaticSim()
    sim.set_numerical(
        order=2,
        tolerance=1e-6,
        max_iterations=200,
        amr_max_passes=20,
        amr_nonconformal=True,
        amr_tolerance=0.01,
        amr_update_fraction=0.7,
        amr_nc_levels=1,
        save_adapt_iterations=True,
        save_adapt_mesh=True,
        save_adapt_diagnostics=True,
        output_paraview=True,
        output_grid_function=False,
        device="CPU",
    )
    sim.set_electrostatic(save_fields=1)
    config = _electrostatic_config(sim)
    assert config["Model"]["Refinement"] == {
        "UniformLevels": 0,
        "MaxIts": 20,
        "Nonconformal": True,
        "MaxNCLevels": 1,
        "Tol": 0.01,
        "UpdateFraction": 0.7,
        "SaveAdaptIterations": True,
        "SaveAdaptMesh": True,
        "SaveAdaptDiagnostics": True,
    }
    assert config["Solver"] == {
        "Order": 2,
        "Device": "CPU",
        "Linear": {"Type": "BoomerAMG", "KSPType": "CG", "Tol": 1e-6, "MaxIts": 200},
        "Electrostatic": {"Save": 1},
    }
    assert config["Problem"]["OutputFormats"] == {
        "Paraview": True,
        "GridFunction": False,
    }
    assert (
        configure_numerical_controls(**json.loads(json.dumps(sim.numerical)))
        == sim.numerical
    )


def test_eigenmode_solver_defaults_remain_gmres_and_default_preconditioner() -> None:
    """Retain the existing non-electrostatic solver behavior."""
    sim = EigenmodeSim()
    sim.set_numerical(amr_max_passes=0)
    config = build_eigenmode_config(
        groups=_GROUPS,
        ports=(),
        materials=_MATERIALS,
        surface_epr_specs={},
        numerical=sim.numerical,
        num_modes=sim.num_modes,
        target_hz=sim.target_hz,
        eigenmode_tolerance=sim.eigenmode_tolerance,
        save_fields=sim.save_fields,
        mesh_path=Path("palace.msh"),
    ).config
    assert config["Solver"]["Linear"] == {
        "Type": "Default",
        "KSPType": "GMRES",
        "Tol": 1e-6,
        "MaxIts": 400,
    }
    assert linear_solver(configure_numerical_controls()) == config["Solver"]["Linear"]


@pytest.mark.parametrize(
    "ksp_type", ("Default", "CG", "MINRES", "GMRES", "FGMRES", "BiCGSTAB")
)
def test_explicit_krylov_choices_are_not_overwritten(ksp_type: str) -> None:
    """Pass each supported native Krylov selection through unchanged."""
    sim = ElectrostaticSim()
    sim.set_numerical(ksp_type=ksp_type, preconditioner="AMS")
    assert _electrostatic_config(sim)["Solver"]["Linear"]["KSPType"] == ksp_type
    assert _electrostatic_config(sim)["Solver"]["Linear"]["Type"] == "AMS"


def test_direct_mumps_solver_controls_remain_explicit() -> None:
    """Keep the existing direct-solver override fields and one-iteration setting."""
    sim = ElectrostaticSim()
    sim.set_numerical(solver_type="MUMPS")
    assert _electrostatic_config(sim)["Solver"]["Linear"] == {
        "Type": "MUMPS",
        "KSPType": "GMRES",
        "Tol": 1e-6,
        "MaxIts": 1,
        "MGMaxLevels": 1,
        "PCMatReal": False,
        "ComplexCoarseSolve": True,
    }


@pytest.mark.parametrize("sim_type", (ElectrostaticSim, EigenmodeSim))
def test_new_optional_controls_retain_choices_across_later_setters(
    sim_type: Any,
) -> None:
    """Keep explicit native-default Krylov, unlimited NC levels and diagnostics opt-out."""
    sim = sim_type()
    sim.set_numerical(ksp_type="Default", amr_nc_levels=0, save_adapt_diagnostics=False)
    sim.set_numerical(order=2)
    sim.set_mesh(threads=1)
    assert sim.numerical["ksp_type"] == "Default"
    assert build_refinement({**sim.numerical, "amr_max_passes": 0})["MaxNCLevels"] == 0
    assert (
        build_refinement({**sim.numerical, "amr_max_passes": 0})["SaveAdaptDiagnostics"]
        is False
    )


@pytest.mark.parametrize(
    "invalid",
    (
        {"amr_nc_levels": -1},
        {"amr_nc_levels": True},
        {"amr_nc_levels": 1.5},
        {"ksp_type": "unknown"},
        {"save_adapt_diagnostics": 1},
    ),
)
@pytest.mark.parametrize("sim_type", (ElectrostaticSim, EigenmodeSim))
def test_invalid_new_controls_preserve_numerical_state(
    sim_type: Any, invalid: dict[str, Any]
) -> None:
    """Reject invalid declared native settings before applying any accompanying update."""
    sim = sim_type()
    before = deepcopy(sim.numerical)
    with pytest.raises((ValueError, TypeError)):
        sim.set_numerical(order=2, **invalid)
    assert sim.numerical == before


@pytest.mark.parametrize("solver_type", ("MUMPS", "SuperLU", "STRUMPACK"))
def test_existing_direct_solver_keeps_gmres_when_krylov_is_omitted(
    solver_type: str,
) -> None:
    """Preserve the historical Krylov token on explicit direct-solver paths."""
    sim = ElectrostaticSim()
    sim.set_numerical(solver_type=solver_type)
    linear = _electrostatic_config(sim)["Solver"]["Linear"]
    assert linear["Type"] == solver_type
    assert linear["KSPType"] == "GMRES"
    assert linear["MaxIts"] == (1 if solver_type == "MUMPS" else 400)
    sim.set_numerical(solver_type=solver_type, ksp_type="CG")
    assert _electrostatic_config(sim)["Solver"]["Linear"]["KSPType"] == "CG"


@pytest.mark.parametrize("preconditioner", ("AMS", "BoomerAMG"))
def test_existing_explicit_preconditioner_keeps_gmres_when_krylov_is_omitted(
    preconditioner: str,
) -> None:
    """Limit changed implicit defaults to the fully default electrostatic path."""
    sim = ElectrostaticSim()
    sim.set_numerical(preconditioner=preconditioner)
    linear = _electrostatic_config(sim)["Solver"]["Linear"]
    assert linear["Type"] == preconditioner
    assert linear["KSPType"] == "GMRES"
