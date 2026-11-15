"""Q3D on an imported SGB model: preparation, solving and the returned matrices."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from scgsim.aedt.io import file_sha256
from scgsim.aedt.results.convergence.q3d import read_q3d_convergence
from scgsim.aedt.runtime.benchmark import attach_simulation_benchmark
from scgsim.aedt.runtime.control import Q3dStopControl
from scgsim.aedt.runtime.families.q3d_exports import export_q3d_matrices
from scgsim.aedt.runtime.families.q3d_nets import assign_q3d_nets
from scgsim.aedt.runtime.families.q3d_prepared import PreparedQ3d
from scgsim.aedt.runtime.families.q3d_prepared import (
    prepared_q3d_result as prepared_q3d_result,
)
from scgsim.aedt.runtime.families.q3d_region import (
    read_q3d_region_ground,
    seal_q3d_region,
)
from scgsim.aedt.runtime.families.q3d_setup import read_q3d_setup, setup_q3d
from scgsim.aedt.runtime.families.q3d_termination import termination_q3d_result
from scgsim.aedt.runtime.families.sgb_import import (
    import_sgb_model,
    imported_geometry_record,
)
from scgsim.aedt.runtime.native.common import (
    BoundAedtRequest,
    detached_data,
)
from scgsim.aedt.runtime.native.common import (
    pyaedt_version as _pyaedt_version,
)
from scgsim.aedt.runtime.native.resources import analyze_with_resources
from scgsim.aedt.specs.common import AedtResources
from scgsim.aedt.specs.q3d import Q3dSpec


def run_q3d(  # noqa: PLR0913 -- family runner preserves transaction resource and stop-control keywords.
    Q3d: Any,
    run_dir: Path,
    spec: Q3dSpec,
    resources: AedtResources | None = None,
    resource_evidence: dict[str, Any] | None = None,
    *,
    control_metadata: Path | None = None,
    runtime_source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Use the same preparation stage as diagnostics, then solve and export."""
    if spec.run_control is None:
        raise ValueError("solving requires an explicit complete run_control")
    prepared = prepare_q3d(Q3d, run_dir, spec)
    control = None
    if control_metadata is not None and sys.platform == "linux":
        if runtime_source is None:
            raise RuntimeError(
                "Q3D stop control requires the actual recorder runtime identity"
            )
        control = Q3dStopControl(control_metadata, prepared.app, runtime_source)
    try:
        solve_q3d(prepared, resources, resource_evidence, control=control)
    except Exception as exc:
        if control is None or (
            control.snapshot().get("intent") is None and control.ledger_error is None
        ):
            raise
        return termination_q3d_result(prepared, control.snapshot(), exc)
    if control is not None and control.snapshot().get("intent") is not None:
        return termination_q3d_result(prepared, control.snapshot(), None)
    result = export_q3d(prepared)
    result["stop_control"] = (
        control.snapshot()
        if control is not None
        else {
            "status": "unsupported_platform"
            if sys.platform != "linux"
            else "not_registered"
        }
    )
    return attach_simulation_benchmark(
        result, prepared.app, run_dir, spec.run_control.setup_name
    )


def prepare_q3d(Q3d: Any, run_dir: Path, spec: Q3dSpec) -> PreparedQ3d:
    """Import the request's SGB model, assign its Nets and terminals, set up, save and read back (no solve)."""
    request = BoundAedtRequest.bind(run_dir, spec)
    bound_spec = request.parse()
    if not isinstance(bound_spec, Q3dSpec):
        raise TypeError("bound Q3D request did not retain a Q3D spec")
    run_dir = request.workspace
    spec = bound_spec
    project_path = run_dir / f"{spec.project_name}.aedt"
    app = Q3d(
        project=str(project_path),
        design=spec.design_name,
        new_desktop=False,
        close_on_exit=False,
    )
    app.modeler.model_units = "um"
    geometry = imported_geometry_record(import_sgb_model(app, spec.model, family="q3d"))
    nets = assign_q3d_nets(app, spec)
    region = dict(geometry["region"])
    if spec.grounded_region_net is not None:
        grounded_region = seal_q3d_region(app, spec, region["native_material"])
        region["native_region_object_id"] = grounded_region["native_region_object_id"]
        region["native_bounding_box_um"] = grounded_region["native_bounding_box_um"]
        region["grounded_region"] = grounded_region
    if spec.run_control is not None:
        setup_q3d(app, spec, spec.run_control)
    if not app.save_project() or not project_path.is_file():
        raise RuntimeError("Q3D project was not saved before solve")
    setup = (
        read_q3d_setup(app, spec, spec.run_control)
        if spec.run_control is not None
        else None
    )
    if spec.grounded_region_net is not None:
        region["grounded_region"] = read_q3d_region_ground(app, spec, region)
        region["native_region_object_id"] = region["grounded_region"][
            "native_region_object_id"
        ]
        region["native_bounding_box_um"] = region["grounded_region"][
            "native_bounding_box_um"
        ]
        if app.validate_simple() != 1:
            raise RuntimeError("Q3D grounded Region failed native ValidateDesign")
        region["grounded_region"]["native_design_validation"] = {
            "method": "ValidateDesign",
            "ok": True,
        }
    return PreparedQ3d(app, request, project_path, geometry, region, nets, setup)


def solve_q3d(
    prepared: PreparedQ3d,
    resources: AedtResources | None = None,
    resource_evidence: dict[str, Any] | None = None,
    *,
    control: Q3dStopControl | None = None,
) -> None:
    """Run the one explicit Q3D setup solve."""
    spec = prepared.request.parse()
    if not isinstance(spec, Q3dSpec):
        raise TypeError("bound Q3D request did not retain a Q3D spec")
    if spec.run_control is None:
        raise ValueError("Q3D solving requires an explicit complete run_control")
    if not analyze_with_resources(
        prepared.app,
        spec.run_control.setup_name,
        prepared.request.workspace,
        resources,
        resource_evidence if resource_evidence is not None else {},
        analysis_call=control.analyze if control is not None else None,
    ):
        raise RuntimeError(
            f"Q3D failed to analyze setup {spec.run_control.setup_name!r}"
        )


def export_q3d(prepared: PreparedQ3d) -> dict[str, Any]:
    """Export, perform the final save, and bind convergence readback."""
    run_dir = prepared.request.workspace
    spec = prepared.request.parse()
    if not isinstance(spec, Q3dSpec):
        raise TypeError("bound Q3D request did not retain a Q3D spec")
    outputs, result_readback = export_q3d_matrices(prepared.app, run_dir, spec)
    outputs = detached_data(outputs)
    result_readback = detached_data(result_readback)
    if not prepared.app.save_project() or not prepared.project_path.is_file():
        raise RuntimeError("Q3D project was not saved")
    region = detached_data(prepared.region)
    if spec.grounded_region_net is not None:
        region["grounded_region"] = read_q3d_region_ground(prepared.app, spec, region)
        region["native_region_object_id"] = region["grounded_region"][
            "native_region_object_id"
        ]
        region["native_bounding_box_um"] = region["grounded_region"][
            "native_bounding_box_um"
        ]
    convergence = read_q3d_convergence(run_dir, spec)
    relative = prepared.project_path.relative_to(run_dir).as_posix()
    outputs[relative] = file_sha256(prepared.project_path)
    return {
        "outputs": outputs,
        "connected": {
            "aedt_version": prepared.app.desktop_class.aedt_version_id,
            "pyaedt_version": _pyaedt_version(),
        },
        "project": relative,
        "geometry": detached_data(prepared.geometry),
        "materials": detached_data(prepared.geometry["materials"]),
        "nets": detached_data(prepared.nets),
        "region": region,
        "setup": detached_data(prepared.setup),
        "convergence": convergence,
        "result_readback": result_readback,
        "save": {"ok": True, "project_sha256": outputs[relative]},
    }
