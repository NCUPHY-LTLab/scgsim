"""Own Q3D model preparation, solve and result orchestration."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

from scgsim.aedt.io import file_sha256, read_json
from scgsim.aedt.preparation.cohort import GEOMETRY_SOURCE_PATHS
from scgsim.aedt.results.convergence.q3d import read_q3d_convergence
from scgsim.aedt.runtime.benchmark import attach_simulation_benchmark
from scgsim.aedt.runtime.control import Q3dStopControl
from scgsim.aedt.runtime.families.q3d_exports import export_q3d_matrices
from scgsim.aedt.runtime.families.q3d_prepared import PreparedQ3d
from scgsim.aedt.runtime.families.q3d_prepared import (
    prepared_q3d_result as prepared_q3d_result,
)
from scgsim.aedt.runtime.families.q3d_region import (
    read_q3d_region_ground,
    seal_q3d_region,
)
from scgsim.aedt.runtime.families.q3d_setup import read_q3d_setup, setup_q3d
from scgsim.aedt.runtime.families.q3d_terminals import (
    assign_q3d_nets,
    verify_saved_q3d_sheet_terminals,
)
from scgsim.aedt.runtime.families.q3d_termination import termination_q3d_result
from scgsim.aedt.runtime.native.common import (
    BoundAedtRequest,
    detached_data,
    realize_source_material,
    source_material_properties,
)
from scgsim.aedt.runtime.native.common import (
    create_region as _create_region,
)
from scgsim.aedt.runtime.native.common import (
    pyaedt_version as _pyaedt_version,
)
from scgsim.aedt.runtime.native.q3d_bodies import construct_q3d_bodies
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


def source_material_bindings(
    run_dir: Path, spec: Q3dSpec
) -> dict[str, dict[str, Any]] | None:
    """Select the transaction-verified stack's referenced non-PEC materials.

    Validate all explicit same-name property bindings before writing any native
    material. Optional missing properties do not conflict with supplied values.
    """
    if spec.geometry_source is None:
        return None
    catalog = read_json(run_dir / GEOMETRY_SOURCE_PATHS["stack"])["materials"]
    referenced = {spec.vacuum_material_id} | {
        body.material_id
        for body in spec.bodies
        if not spec.materials[body.material_id].is_superconducting
    }
    bindings: dict[str, dict[str, Any]] = {}
    supplied: dict[str, dict[str, tuple[str, float]]] = {}
    for material_id in sorted(referenced):
        record = catalog[material_id]
        native_name = cast(str, spec.materials[material_id].library_name)
        properties = source_material_properties(record)
        same_name = supplied.setdefault(native_name.casefold(), {})
        for key, value in properties.items():
            previous = same_name.get(key)
            if previous is not None and previous[1] != value:
                raise ValueError(
                    f"Q3D source materials {previous[0]!r} and {material_id!r} "
                    f"bind conflicting {key!r} values to native {native_name!r}"
                )
            same_name[key] = (material_id, value)
        bindings[material_id] = {"record": record, "native_name": native_name}
    return bindings


def prepare_q3d(Q3d: Any, run_dir: Path, spec: Q3dSpec) -> PreparedQ3d:
    """Construct and read back the Q3D model before the explicit solve stage."""
    request = BoundAedtRequest.bind(run_dir, spec)
    bound_spec = request.parse()
    if not isinstance(bound_spec, Q3dSpec):
        raise TypeError("bound Q3D request did not retain a Q3D spec")
    run_dir = request.workspace
    spec = bound_spec
    source_bindings = source_material_bindings(run_dir, spec)
    project_path = run_dir / f"{spec.project_name}.aedt"
    app = Q3d(
        project=str(project_path),
        design=spec.design_name,
        new_desktop=False,
        close_on_exit=False,
    )
    app.modeler.model_units = "um"
    material_realizations = (
        {
            material_id: realize_source_material(
                app, material_id, binding["record"], native_name=binding["native_name"]
            )
            for material_id, binding in source_bindings.items()
        }
        if source_bindings is not None
        else None
    )
    materials = construct_q3d_bodies(
        app, spec, material_realizations=material_realizations
    )
    region = _create_region(app, spec)
    if material_realizations is not None:
        region["source_material_realization"] = material_realizations[
            spec.vacuum_material_id
        ]
    nets = assign_q3d_nets(app, spec)
    if spec.grounded_region_net is not None:
        grounded_region = seal_q3d_region(app, spec)
        region["native_region_object_id"] = grounded_region["native_region_object_id"]
        region["native_bounding_box_um"] = grounded_region["native_bounding_box_um"]
        region["grounded_region"] = grounded_region
    if spec.run_control is not None:
        setup_q3d(app, spec, spec.run_control)
    if not app.save_project() or not project_path.is_file():
        raise RuntimeError("Q3D project was not saved before solve")
    verify_saved_q3d_sheet_terminals(app, nets)
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
    return PreparedQ3d(
        app,
        request,
        project_path,
        materials,
        region,
        nets,
        setup,
    )


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
        "nets": detached_data(prepared.nets),
        "materials": detached_data(prepared.materials),
        "region": region,
        "setup": detached_data(prepared.setup),
        "convergence": convergence,
        "result_readback": result_readback,
        "save": {"ok": True, "project_sha256": outputs[relative]},
    }
