"""Pure configuration coverage for explicit AMR studies; no native solver runs."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scgsim.palace import EigenmodeSim, ElectrostaticSim
from scgsim.palace.config.config import (
    build_eigenmode_config,
    build_electrostatic_config,
)
from scgsim.palace.config.config_models import (
    ConfigInput,
    EigenmodeSettings,
    ElectrostaticSettings,
    TerminalBinding,
)
from scgsim.palace.config.numerical import (
    NumericalControls,
    build_refinement,
    linear_solver,
)
from scgsim.palace.source.surface_epr import SurfaceEprSpecs

_GROUPS = {
    "volumes": {
        "vacuum": {"attributes": [1], "material": "vacuum", "entities": ["vacuum"]}
    },
    "boundary_surfaces": {},
    "pec_surfaces": {
        "contact": {
            "entity": "metal",
            "entities": ["metal"],
            "net": "net",
            "attributes": [2],
        }
    },
}
_MATERIALS = {"vacuum": {"kind": "vacuum", "permittivity": 1.0, "loss_tangent": 0.0}}


def _common(sim: EigenmodeSim | ElectrostaticSim) -> ConfigInput:
    return ConfigInput(
        problem=sim.problem,
        groups=_GROUPS,
        mesh_path=Path("palace.msh"),
        materials=_MATERIALS,
        surface_epr_specs=SurfaceEprSpecs({}),
        numerical=sim.numerical,
    )


def _electrostatic_config(sim: ElectrostaticSim) -> dict[str, Any]:
    return build_electrostatic_config(
        _common(sim),
        terminals=[TerminalBinding(name="terminal", net_id="net")],
        ground_net_ids=(),
        electrostatic=ElectrostaticSettings(
            sim.save_fields,
            sim.unassigned_conductor_policy,
            sim.exterior_boundary_policy,
        ),
    ).config


def test_electrostatic_defaults_use_cg_boomeramg_without_changing_case_controls() -> (
    None
):
    """Select the electrostatic solver while retaining all other baseline defaults."""
    sim = ElectrostaticSim()
    assert sim.numerical == NumericalControls()
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
        NumericalControls(**json.loads(json.dumps(sim.numerical.to_payload())))
        == sim.numerical
    )


def test_eigenmode_solver_defaults_remain_gmres_and_default_preconditioner() -> None:
    """Retain the existing non-electrostatic solver behavior."""
    sim = EigenmodeSim()
    sim.set_numerical(amr_max_passes=0)
    config = build_eigenmode_config(
        _common(sim), ports=(), eigenmode=EigenmodeSettings()
    ).config
    assert config["Solver"]["Linear"] == {
        "Type": "Default",
        "KSPType": "GMRES",
        "Tol": 1e-6,
        "MaxIts": 400,
    }
    assert linear_solver(NumericalControls()) == config["Solver"]["Linear"]


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
    """Keep explicit native-default Krylov and unlimited NC levels."""
    sim = sim_type()
    sim.set_numerical(ksp_type="Default", amr_nc_levels=0)
    sim.set_numerical(order=2)
    sim.set_mesh(quality_threshold=0.02)
    assert sim.numerical.ksp_type == "Default"
    assert (
        build_refinement(replace(sim.numerical, amr_max_passes=0))["MaxNCLevels"] == 0
    )


@pytest.mark.parametrize(
    "invalid",
    (
        {"amr_nc_levels": -1},
        {"amr_nc_levels": True},
        {"amr_nc_levels": 1.5},
        {"ksp_type": "unknown"},
    ),
)
@pytest.mark.parametrize("sim_type", (ElectrostaticSim, EigenmodeSim))
def test_invalid_new_controls_preserve_numerical_state(
    sim_type: Any, invalid: dict[str, Any]
) -> None:
    """Reject invalid declared native settings before applying any accompanying update."""
    sim = sim_type()
    before = sim.numerical
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
