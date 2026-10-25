"""Public secondary-workflow constructors/codecs, with no mesh or native model.

Coordinates are request literals from the public CPW lesson / Q2D export test.
The assertions concern transport of caller values, not scientific predictions.
Palace has no whole-Sim request codec; only its public profile/control codecs are
round-tripped here, without serializing private Sim state.
"""

from __future__ import annotations

import json

import pytest

from scgsim.aedt import (
    MatrixRunControl, PdkMaterial, Q2dConductorSpec, Q2dRectangleSpec, Q2dSpec,
    Q3dBodySpec, Q3dNetSpec, Q3dSpec, parse_aedt_spec,
)
from scgsim.geometry import ApproximationProfile
from scgsim.palace import ElectrostaticSim, EigenmodeSim
from scgsim.palace.preparation.config import configure_numerical_controls


VERSIONS = ("2023.2", "2024.2")


def _materials():
    return {
        "vacuum": PdkMaterial("vacuum", "vacuum", False, "Vacuum"),
        "silicon": PdkMaterial("silicon", "dielectric", False, "Silicon"),
        "metal": PdkMaterial("metal", "superconductor", True, None),
    }


@pytest.mark.parametrize("version", VERSIONS)
def test_q2d_public_request_roundtrip_preserves_caller_values(tmp_path, version):
    control = MatrixRunControl(
        "Setup1", frequency_ghz=6.0, maximum_passes=20,
        convergence_percent=0.1, minimum_passes=1,
        minimum_converged_passes=1, percent_refinement=30.0,
    )
    rectangles = (
        Q2dRectangleSpec("Substrate", (0.0, -10.0), (20.0, 10.0), "silicon"),
        Q2dRectangleSpec("SignalObject", (8.0, 0.0), (4.0, 0.2), "metal"),
        Q2dRectangleSpec("GroundObject", (0.0, 0.0), (4.0, 0.2), "metal"),
    )
    conductors = (
        Q2dConductorSpec("Signal", "SignalLine", ("SignalObject",), 0.2),
        Q2dConductorSpec("Ground", "ReferenceGround", ("GroundObject",), 0.2),
    )
    request = Q2dSpec(
        aedt_version=version, project_name="CodecQ2d", design_name="Q2DDesign1",
        materials=_materials(), vacuum_material_id="vacuum",
        rectangles=rectangles, conductors=conductors, run_control=control,
        region_padding_um=(10.0, 10.0, 10.0, 10.0),
    )
    payload = json.loads(json.dumps(request.to_payload()))
    restored = parse_aedt_spec(payload, base_dir=tmp_path)
    assert isinstance(restored, Q2dSpec)
    assert restored.to_payload() == payload
    assert restored.aedt_version == version
    assert restored.run_control.to_payload() == control.to_payload()
    assert tuple(item.to_payload() for item in restored.rectangles) == tuple(item.to_payload() for item in rectangles)
    assert tuple(item.to_payload() for item in restored.conductors) == tuple(item.to_payload() for item in conductors)


def _q3d_bodies():
    # The public CPW lesson provides the four finite outlines and Z spans in um.
    return (
        Q3dBodySpec("signal", ((-250.0, -5.0), (250.0, -5.0), (250.0, 5.0), (-250.0, 5.0)),
                    (), 0.0, 0.2, "metal", "signal", "Signal"),
        Q3dBodySpec("ground_upper", ((-250.0, 11.0), (250.0, 11.0), (250.0, 91.0), (-250.0, 91.0)),
                    (), 0.0, 0.2, "metal", "ground", "Ground"),
        Q3dBodySpec("ground_lower", ((-250.0, -91.0), (250.0, -91.0), (250.0, -11.0), (-250.0, -11.0)),
                    (), 0.0, 0.2, "metal", "ground", "Ground"),
        Q3dBodySpec("substrate", ((-250.0, -91.0), (250.0, -91.0), (250.0, 91.0), (-250.0, 91.0)),
                    (), -500.0, 0.0, "silicon", "substrate", None),
    )


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("model_only", (False, True))
def test_q3d_public_request_roundtrip_preserves_explicit_stage(tmp_path, version, model_only):
    control = None if model_only else MatrixRunControl(
        "Setup1", frequency_ghz=6.0, maximum_passes=3,
        convergence_percent=1.0, minimum_passes=1,
        minimum_converged_passes=1, percent_refinement=30.0,
    )
    bodies = _q3d_bodies()
    nets = (
        Q3dNetSpec("Signal", "Signal", ("signal",), "signal", "-X", "signal", "+X"),
        Q3dNetSpec("Ground", "Ground", ("ground_upper", "ground_lower")),
    )
    request = Q3dSpec(
        aedt_version=version, modeling="solid", project_name="CodecQ3d",
        design_name="Q3DDesign1", materials=_materials(), vacuum_material_id="vacuum",
        bodies=bodies, nets=nets, run_control=control, solve_ac_rl=True,
        region_padding_um=(0.0, 0.0, 100.0, 100.0, 1000.0, 1000.0),
    )
    payload = json.loads(json.dumps(request.to_payload()))
    restored = parse_aedt_spec(payload, base_dir=tmp_path)
    assert isinstance(restored, Q3dSpec)
    assert restored.to_payload() == payload
    assert (restored.aedt_version, restored.modeling) == (version, "solid")
    assert restored.run_control == control
    assert restored.solve_ac_rl is True
    assert tuple(item.to_payload() for item in restored.bodies) == tuple(item.to_payload() for item in bodies)
    assert tuple(item.to_payload() for item in restored.nets) == tuple(item.to_payload() for item in nets)


def test_palace_public_eigenmode_constructor_and_profile_control_codecs():
    profile = ApproximationProfile("route_b", "v1", {})
    restored_profile = ApproximationProfile.from_payload(json.loads(json.dumps(profile.to_payload())))
    numerical = configure_numerical_controls(
        order=2, tolerance=1e-6, max_iterations=2000,
        refined_mesh_size=10.0, max_mesh_size=200.0, algorithm_3d="Delaunay",
        threads=1, surface_threads=1, geometry_order=1, high_order_optimize=True,
        amr_max_passes=6, amr_nonconformal=False, amr_tolerance=0.02,
        amr_update_fraction=0.15, save_adapt_iterations=True,
        output_paraview=True, output_grid_function=False,
    )
    decoded = json.loads(json.dumps(numerical))
    assert configure_numerical_controls(**decoded) == numerical
    request = EigenmodeSim(route="B", approximation_profile=restored_profile, numerical=decoded)
    request.set_eigenmode(num_modes=2, target=3e9, tolerance=1e-6, save=1)
    assert request.approximation_profile.to_payload() == profile.to_payload()
    assert request.numerical == numerical
    assert (request.num_modes, request.target_hz, request.eigenmode_tolerance, request.save_fields) == (2, 3e9, 1e-6, 1)


def test_palace_public_electrostatic_constructor_preserves_explicit_policies():
    profile = ApproximationProfile("route_b", "v1", {})
    restored = ApproximationProfile.from_payload(json.loads(json.dumps(profile.to_payload())))
    request = ElectrostaticSim(route="B", approximation_profile=restored)
    request.set_electrostatic(save_fields=1, unassigned_conductor_policy="error", exterior_boundary_policy="none")
    assert request.approximation_profile.to_payload() == profile.to_payload()
    assert (request.save_fields, request.unassigned_conductor_policy, request.exterior_boundary_policy) == (1, "error", "none")
