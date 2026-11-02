"""Preserve common OCC lowering's native lifecycle and shell option restoration."""

from __future__ import annotations

import base64
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from test_geometry_context_propagation import conductor_input, conductor_port_input

import scgsim
from scgsim.geometry import ApproximationProfile, interpret_physical_geometry
from scgsim.geometry.compiler import build_route_construction_plan
from scgsim.geometry.construction.gmsh_occ import write_occ_geometry_from_plan
from scgsim.geometry.construction.native import add_surface_first_volume
from scgsim.geometry.models.construction import ConstructionPlanRecord
from scgsim.geometry.models.topology import SurfacePlanRecord


@pytest.mark.parametrize("initialized", (False, True))
@pytest.mark.parametrize(
    "outcome", ("success", "loop_error", "option_error", "checksum_error")
)
def test_occ_lowering_preserves_initialization_and_failure_cleanup(
    initialized, outcome
):
    """Finalize owned native state and preserve the existing caller-owned clear behavior."""
    _run_native_case(outcome, initialized)


@pytest.mark.parametrize("previous", (False, True))
def test_surface_first_failure_restores_the_native_autofix_option(previous):
    """Restore the caller's AutoFix value after a real OCC shell failure."""
    _run_native_case("shell_failure", previous)


def test_port_probe_failure_still_clears_the_disposable_native_mesh():
    """Return the existing failed evidence and clear mesh nodes after missing port ownership."""
    _run_native_case("port_metadata_error", True)


def test_invalid_canonical_plan_fails_before_output_or_native_initialization(tmp_path):
    """Reject missing canonical topology before creating the output directory."""
    gmsh = pytest.importorskip("gmsh")
    initialized = bool(gmsh.isInitialized())
    target = tmp_path / "uncreated" / "native.xao"
    plan = ConstructionPlanRecord(
        "B", surfaces=(SurfacePlanRecord("face", "owner", "interface", {}),)
    )
    with pytest.raises(
        NotImplementedError, match="planned PointPlan/CurvePlan/SurfaceLoop"
    ):
        write_occ_geometry_from_plan(plan, xao_path=target)
    assert not target.parent.exists()
    assert bool(gmsh.isInitialized()) == initialized


def _run_native_case(outcome: str, flag: bool) -> None:
    pytest.importorskip("gmsh")
    environment = dict(os.environ)
    source = str(Path(scgsim.__file__).resolve().parents[1])
    environment["PYTHONPATH"] = os.pathsep.join(
        (source, environment.get("PYTHONPATH", ""))
    )
    completed = subprocess.run(
        [sys.executable, __file__, outcome, str(int(flag))],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def _verify_lowering(outcome: str, initialized: bool) -> None:
    # Gmsh state and environment options belong to this isolated subprocess.
    import gmsh  # noqa: PLC0415 - optional native geometry dependency

    plan = _construction_plan(outcome)
    if initialized:
        gmsh.initialize(readConfigFiles=False)
        gmsh.model.add("caller-model")
        gmsh.model.occ.addPoint(17, 23, 31)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Geometry.OCCAutoFix", 1)
    if outcome == "option_error":
        os.environ["SGB_GMSH_OCC_AUTO_FIX"] = "not-a-number"
    else:
        os.environ.pop("SGB_GMSH_OCC_AUTO_FIX", None)
    try:
        with TemporaryDirectory() as directory:
            target = Path(directory) / "native.xao"
            if outcome == "loop_error":
                with pytest.raises(
                    ValueError, match="could not be lowered to an OCC curve loop"
                ) as error:
                    write_occ_geometry_from_plan(plan, xao_path=target)
                assert error.value.__cause__ is not None
            elif outcome == "option_error":
                with pytest.raises(
                    ValueError, match="could not convert string to float"
                ):
                    write_occ_geometry_from_plan(plan, xao_path=target)
            elif outcome == "checksum_error":
                with pytest.raises(
                    ValueError, match="detached curved boundary XAO identity differs"
                ):
                    write_occ_geometry_from_plan(plan, xao_path=target)
            else:
                lowered = write_occ_geometry_from_plan(plan, xao_path=target)
                assert target.is_file()
                assert lowered.backend_entity_tags
                report = lowered.metadata["engine_gate_gmsh_brep_conformality"]
                if outcome == "port_metadata_error":
                    assert report["status"] == "fail"
                    assert (
                        report["failures"][-1]["code"]
                        == "lumped_port_mesh_generation_failed"
                    )
                    assert len(gmsh.model.mesh.getNodes()[0]) == 0
                else:
                    assert report["status"] == "pass"
            assert target.exists() == (outcome in {"success", "port_metadata_error"})
        assert bool(gmsh.isInitialized()) == initialized
        if initialized:
            assert "caller-model" not in gmsh.model.list()
            assert gmsh.option.getNumber("General.Terminal") == 0
            assert gmsh.option.getNumber("Geometry.OCCAutoFix") == (
                1 if outcome == "option_error" else 0
            )
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()


def _construction_plan(outcome: str) -> ConstructionPlanRecord:
    source, stack = (
        conductor_port_input()
        if outcome == "port_metadata_error"
        else conductor_input()
    )
    effective, _, physical = interpret_physical_geometry(
        source, stack, ApproximationProfile("route_b", "v1", {})
    )
    plan = build_route_construction_plan(
        effective, route="B", representation_context=physical.context
    )
    if outcome == "port_metadata_error":
        plan = replace(
            plan,
            surfaces=tuple(
                replace(
                    surface,
                    metadata={
                        key: value
                        for key, value in surface.metadata.items()
                        if key != "owner_semantic_ids"
                    },
                )
                if surface.surface_role == "lumped_port"
                else surface
                for surface in plan.surfaces
            ),
        )
    elif outcome == "loop_error":
        first = replace(
            plan.surface_loops[0], curve_refs=plan.surface_loops[0].curve_refs[:-1]
        )
        plan = replace(plan, surface_loops=(first, *plan.surface_loops[1:]))
    elif outcome == "checksum_error":
        plan = replace(
            plan,
            metadata={
                **plan.metadata,
                "curved_boundary_xao": base64.b64encode(b"corrupt transport").decode(
                    "ascii"
                ),
                "curved_boundary_xao_sha256": "0" * 64,
            },
        )
    return plan


def _verify_shell_failure(previous: bool) -> None:
    import gmsh  # noqa: PLC0415 - optional native geometry dependency

    gmsh.initialize(readConfigFiles=False)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Geometry.OCCAutoFix", int(previous))
    try:
        with pytest.raises(Exception, match="OpenCASCADE exception"):
            add_surface_first_volume(gmsh, ((),), outer_wire_edges={})
        assert gmsh.option.getNumber("Geometry.OCCAutoFix") == int(previous)
        assert gmsh.isInitialized()
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    if sys.argv[1] == "shell_failure":
        _verify_shell_failure(bool(int(sys.argv[2])))
    else:
        _verify_lowering(sys.argv[1], bool(int(sys.argv[2])))
