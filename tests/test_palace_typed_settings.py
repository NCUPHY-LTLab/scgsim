"""The named settings a Palace simulation hands to the config: each checks itself and writes what the dicts wrote."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from scgsim.geometry.surface_epr.specs import normalize_surface_epr_specs
from scgsim.palace import EigenmodeSim, ElectrostaticSim
from scgsim.palace.config.config_models import (
    ConfigInput,
    EigenmodeSettings,
    ElectrostaticSettings,
)
from scgsim.palace.config.index import (
    EnergyEntry,
    EprRequest,
    PortEntry,
    check_epr_request,
)
from scgsim.palace.config.numerical import NumericalControls
from scgsim.palace.source.surface_epr import SurfaceEprSpecs

MATERIALS = {"Si": {"kind": "dielectric", "permittivity": 11.45, "loss_tangent": 1e-6}}
SPECS = {
    "SA": {"film_thickness_m": 3e-9, "permittivity": 11.0, "face_kinds": ["Top"]},
    "MS": {"thickness": 0.003, "material_id": "Si", "inset_margins_um": [0.5, 1]},
    "MA": {
        "film_thickness_m": 2e-9,
        "permittivity": 4,
        "loss_tangent": 0.0,
        "inset_margins_um": [0.5, 1.0],
        "inset_margins_by_entity_um": {"chip/pad_0000": (0.5, 1.0, 1.5)},
    },
}


def test_surface_epr_specs_write_the_normalized_mapping_key_for_key():
    """``to_payload()`` is the shared normalization, in the same key order, so the index map keeps its bytes."""
    expected = normalize_surface_epr_specs(SPECS, materials=MATERIALS)
    payload = SurfaceEprSpecs.normalized(SPECS, materials=MATERIALS).to_payload()
    assert payload == expected
    assert list(payload) == list(expected)
    for interface_type, spec in expected.items():
        assert list(payload[interface_type]) == list(spec)


def test_a_spec_gives_a_conductor_its_own_margins_and_its_native_thickness():
    """A chosen conductor takes its own margins, every other face the type's; thickness is in Model.L0 units."""
    specs = SurfaceEprSpecs.normalized(SPECS, materials=MATERIALS)
    ma = specs.by_type["MA"]
    assert ma.margins_for(["chip/pad_0000"]) == [0.5, 1.0, 1.5]
    assert ma.margins_for(["chip/other"]) == [0.5, 1.0]
    assert ma.native_thickness(1e-6) == pytest.approx(2e-3)
    assert specs.by_type["MS"].native_thickness(1e-6) == 0.003
    assert specs.by_type["MS"].permittivity == 11.45


def test_numerical_controls_write_the_recorded_payload_and_hold_mumps_at_one_iteration():
    """The payload keeps its key order and leaves unset optional keys out; MUMPS records one iteration."""
    controls = NumericalControls(
        solver_type="MUMPS", max_iterations=50, ksp_type="CG", amr_nc_levels=0
    )
    payload = controls.to_payload()
    assert list(payload) == [
        "order",
        "tolerance",
        "max_iterations",
        "solver_type",
        "preconditioner",
        "device",
        "amr_max_passes",
        "amr_nonconformal",
        "amr_tolerance",
        "amr_update_fraction",
        "save_adapt_iterations",
        "estimator_mg",
        "output_paraview",
        "output_grid_function",
        "ksp_type",
        "amr_nc_levels",
    ]
    assert payload["max_iterations"] == 1
    assert NumericalControls(**payload) == controls


@pytest.mark.parametrize(
    ("change", "error", "message"),
    [
        ({"order": 5}, ValueError, r"set_numerical\(order=1..4\)"),
        ({"device": "TPU"}, ValueError, r"unsupported device.*set_numerical"),
        ({"amr_max_passes": True}, TypeError, "amr_max_passes must be an integer"),
        ({"amr_update_fraction": 1.0}, ValueError, "strictly between 0 and 1"),
        ({"output_paraview": "yes"}, TypeError, r"set_numerical\(output_paraview="),
    ],
)
def test_numerical_controls_refuse_where_built_and_name_the_setter(
    change, error, message
):
    """A control is checked when the record is built, also through ``dataclasses.replace``."""
    with pytest.raises(error, match=message):
        replace(NumericalControls(), **change)


@pytest.mark.parametrize("problem", ["Eigenmode", "Electrostatic"])
def test_a_config_input_refuses_an_unchosen_amr_pass_limit(problem):
    """Both problems refuse a config while ``amr_max_passes`` is None, and say which call sets it."""
    with pytest.raises(ValueError, match=r"set_numerical\(amr_max_passes=\.\.\.\)"):
        ConfigInput(
            problem=problem,
            groups={},
            mesh_path=Path("palace.msh"),
            materials={},
            surface_epr_specs=SurfaceEprSpecs({}),
            numerical=NumericalControls(amr_max_passes=None),
        )


def test_problem_settings_refuse_where_built_and_name_the_setter():
    """The mode controls and the boundary policies are checked by their records."""
    with pytest.raises(ValueError, match=r"set_eigenmode\(num_modes=\.\.\.\)"):
        EigenmodeSettings(num_modes=0)
    with pytest.raises(ValueError, match=r"set_eigenmode\(target=\.\.\.\)"):
        EigenmodeSettings(target_hz=-1.0)
    with pytest.raises(ValueError, match="unassigned_conductor_policy='error'"):
        ElectrostaticSettings(exterior_boundary_policy="ground")
    assert EigenmodeSettings(target_hz=5, tolerance=1).target_hz == 5.0


def test_the_setters_store_the_named_values():
    """``set_numerical``, ``set_eigenmode``, ``set_electrostatic`` and ``set_epr_request`` store checked records."""
    sim = EigenmodeSim()
    sim.set_numerical(amr_max_passes=2, amr_tolerance=0.02, amr_update_fraction=0.3)
    sim.set_numerical(order=2)
    assert sim.numerical.amr_max_passes == 2  # Eigenmode keeps omitted AMR controls
    sim.set_eigenmode(num_modes=3, target=4e9)
    assert (sim.num_modes, sim.target_hz) == (3, 4e9)
    sim.set_epr_request(port_names=["P1"])
    assert sim.epr_request == EprRequest(None, None, ("P1",))
    static = ElectrostaticSim()
    static.set_numerical(amr_max_passes=2)
    static.set_numerical(order=2)
    assert static.numerical.amr_max_passes == 0  # Electrostatic resets them to defaults
    with pytest.raises(ValueError, match="exterior ground"):
        static.set_electrostatic(exterior_boundary_policy="ground")


def test_an_epr_request_is_checked_against_the_index_entries():
    """``check_epr_request`` refuses a port or bulk domain the entries do not list, naming what they list."""
    entries = [
        PortEntry(1, "P1", [3], ["P1"], "chip/P1", {}),
        EnergyEntry(
            1, "substrate", [1], {"material_id": "Si", "kind": "dielectric"}, []
        ),
    ]
    check_epr_request(
        EprRequest.selected(
            surface_interfaces=None, bulk_domain_ids=["substrate"], port_names=()
        ),
        entries,
    )
    with pytest.raises(
        ValueError, match=r"unknown port_names binding \['P2'\].*'chip/P1'"
    ):
        check_epr_request(EprRequest(port_names=("P2",)), entries)
    with pytest.raises(ValueError, match="unknown bulk_domain_ids"):
        check_epr_request(EprRequest(bulk_domain_ids=("vacuum",)), entries)
    with pytest.raises(ValueError, match="must not repeat IDs"):
        EprRequest.selected(
            surface_interfaces=["MA", "MA"], bulk_domain_ids=None, port_names=None
        )
    assert list(EprRequest().to_payload()) == [
        "schema_version",
        "surface_interfaces",
        "bulk_domain_ids",
        "port_names",
    ]
