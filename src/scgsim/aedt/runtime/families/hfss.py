"""Own HFSS model preparation, solving and final result orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scgsim.aedt.io import file_sha256
from scgsim.aedt.results.convergence.hfss import read_hfss_convergence
from scgsim.aedt.runtime.benchmark import (
    attach_simulation_benchmark,
)
from scgsim.aedt.runtime.families.hfss_exports import export_hfss_results
from scgsim.aedt.runtime.families.hfss_mesh import assign_mesh
from scgsim.aedt.runtime.families.hfss_ports import (
    assign_body_terminal_ports,
    assign_ports,
    bind_body_terminal_evidence,
    bind_port_evidence,
)
from scgsim.aedt.runtime.families.hfss_requests import HfssModelSpec
from scgsim.aedt.runtime.families.hfss_setup import read_hfss_setup, setup_hfss
from scgsim.aedt.runtime.native.common import (
    BoundAedtRequest,
    detached_data,
)
from scgsim.aedt.runtime.native.common import (
    create_region as _create_region,
)
from scgsim.aedt.runtime.native.common import (
    pyaedt_version as _pyaedt_version,
)
from scgsim.aedt.runtime.native.gds_import import import_and_bind as _import_and_bind
from scgsim.aedt.runtime.native.hfss_geometry import (
    prepare_native_planar_geometry,
)
from scgsim.aedt.runtime.native.resources import analyze_with_resources
from scgsim.aedt.specs.common import (
    AedtResources,
)
from scgsim.aedt.specs.hfss import (
    HfssDrivenGeometrySpec,
    HfssEigenmodeSpec,
)


@dataclass(frozen=True)
class PreparedHfss:
    """Carrier for one prepared HFSS app and its native readback records."""

    app: Any
    request: BoundAedtRequest
    project_path: Path
    materials: list[dict[str, Any]] | dict[str, Any]
    region: dict[str, Any]
    mesh: dict[str, Any]
    ports: list[dict[str, Any]]
    setup: dict[str, Any] | None
    geometry: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.request, BoundAedtRequest):
            raise TypeError("PreparedHfss requires a bound AEDT request")
        for name in ("materials", "region", "mesh", "ports", "setup", "geometry"):
            object.__setattr__(self, name, detached_data(getattr(self, name)))


def run_hfss(
    Hfss: Any,
    run_dir: Path,
    spec: HfssModelSpec,
    resources: AedtResources | None = None,
    resource_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Use the same preparation stage as diagnostics, then solve and export."""
    if spec.run_control is None:
        raise ValueError("solving requires an explicit complete run_control")
    prepared = prepare_hfss(Hfss, run_dir, spec)
    solve_hfss(prepared, resources, resource_evidence)
    result = export_hfss(prepared)
    return attach_simulation_benchmark(
        result, prepared.app, run_dir, spec.run_control.setup_name
    )


def prepare_hfss(Hfss: Any, run_dir: Path, spec: HfssModelSpec) -> PreparedHfss:
    """Construct and read back the HFSS model before the explicit solve stage."""
    request = BoundAedtRequest.bind(run_dir, spec)
    bound_spec = request.parse()
    if not isinstance(bound_spec, HfssModelSpec):
        raise TypeError("bound HFSS request did not retain an HFSS spec")
    run_dir = request.workspace
    spec = bound_spec
    if spec.modeling not in {"solid", "thin_film"}:
        raise ValueError("new HFSS preparation requires explicit modeling")
    project_path = run_dir / f"{spec.project_name}.aedt"
    app = Hfss(
        project=str(project_path),
        design=spec.design_name,
        solution_type={
            "terminal": "DrivenTerminal",
            "modal": "DrivenModal",
            "eigenmode": "Eigenmode",
        }[spec.mode],
        new_desktop=False,
        close_on_exit=False,
    )
    app.modeler.model_units = "um"
    geometry: dict[str, Any] = {}
    if isinstance(spec, HfssDrivenGeometrySpec):
        geometry = prepare_native_planar_geometry(app, spec.geometry)
        materials = geometry["material_readback"]
        region = geometry["closed_enclosure"]
        mesh = {
            "status": "not_requested",
            "reason": "body-first request uses native adaptive mesh",
        }
        ports = assign_body_terminal_ports(app, spec, geometry)
    else:
        materials = _import_and_bind(app, spec)
        region = _create_region(app, spec)
        mesh = assign_mesh(app, spec)
        ports = assign_ports(app, spec)
    if spec.run_control is not None:
        setup_hfss(app, spec)
    if not app.save_project() or not project_path.is_file():
        raise RuntimeError("HFSS project was not saved before native port readback")
    setup = read_hfss_setup(app, spec) if spec.run_control is not None else None
    ports = (
        bind_body_terminal_evidence(app, spec, geometry, ports)
        if isinstance(spec, HfssDrivenGeometrySpec)
        else bind_port_evidence(app, spec, ports)
    )
    return PreparedHfss(
        app,
        request,
        project_path,
        materials,
        region,
        mesh,
        ports,
        setup,
        geometry,
    )


def solve_hfss(
    prepared: PreparedHfss,
    resources: AedtResources | None = None,
    resource_evidence: dict[str, Any] | None = None,
) -> None:
    """Run the one explicit HFSS setup solve."""
    spec = prepared.request.parse()
    if not isinstance(spec, HfssModelSpec):
        raise TypeError("bound HFSS request did not retain an HFSS spec")
    if spec.run_control is None:
        raise ValueError("HFSS solving requires an explicit complete run_control")
    if not analyze_with_resources(
        prepared.app,
        spec.run_control.setup_name,
        prepared.request.workspace,
        resources,
        resource_evidence if resource_evidence is not None else {},
    ):
        raise RuntimeError(
            f"HFSS failed to analyze setup {spec.run_control.setup_name!r}"
        )
    if not prepared.app.save_project() or not prepared.project_path.is_file():
        raise RuntimeError("HFSS solved project save failed")


def export_hfss(prepared: PreparedHfss) -> dict[str, Any]:
    """Export the saved solve and bind convergence readback."""
    run_dir = prepared.request.workspace
    spec = prepared.request.parse()
    if not isinstance(spec, HfssModelSpec):
        raise TypeError("bound HFSS request did not retain an HFSS spec")
    outputs, result_readback = export_hfss_results(
        prepared.app, run_dir, spec, detached_data(prepared.ports)
    )
    outputs = detached_data(outputs)
    result_readback = detached_data(result_readback)
    if isinstance(spec, HfssEigenmodeSpec):
        convergence_path = run_dir / "results/eigenmode/adaptive-convergence.prop"
        returned = prepared.app.export_convergence(
            spec.run_control.setup_name,
            variations="",
            output_file=str(convergence_path),
        )
        if (
            not returned
            or Path(returned).resolve() != convergence_path.resolve()
            or not convergence_path.is_file()
        ):
            raise RuntimeError("HFSS Eigenmode adaptive convergence export is missing")
        outputs[convergence_path.relative_to(run_dir).as_posix()] = file_sha256(
            convergence_path
        )
    saved = bool(prepared.app.save_project())
    if not saved or not prepared.project_path.is_file():
        raise RuntimeError("HFSS project was not saved")
    convergence = read_hfss_convergence(run_dir, spec)
    relative = prepared.project_path.relative_to(run_dir).as_posix()
    outputs[relative] = file_sha256(prepared.project_path)
    return {
        "outputs": outputs,
        "connected": {
            "aedt_version": prepared.app.desktop_class.aedt_version_id,
            "pyaedt_version": _pyaedt_version(),
        },
        "project": relative,
        "ports": detached_data(prepared.ports),
        "mesh": detached_data(prepared.mesh),
        "materials": detached_data(prepared.materials),
        **({"geometry": detached_data(prepared.geometry)} if prepared.geometry else {}),
        "region": detached_data(prepared.region),
        "setup": detached_data(prepared.setup),
        "convergence": convergence,
        "result_readback": result_readback,
        "save": {"ok": True, "project_sha256": outputs[relative]},
    }
