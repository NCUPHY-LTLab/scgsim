"""HFSS Driven Terminal on an imported SGB model: preparation, solving and result export.

:func:`prepare_hfss` imports the request's SGB model into an empty Driven Terminal design
(:func:`scgsim.aedt.runtime.families.sgb_import.import_sgb_model`, which assigns the lumped ports), creates
the setup and sweep when the request has them, saves the project and reads the setup and the ports back.
The geometry record a receipt keeps is :func:`imported_geometry_record`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scgsim.aedt.io import file_sha256
from scgsim.aedt.results.convergence.hfss import read_hfss_convergence
from scgsim.aedt.runtime.benchmark import (
    attach_simulation_benchmark,
)
from scgsim.aedt.runtime.families.hfss_exports import export_hfss_results
from scgsim.aedt.runtime.families.hfss_ports import read_saved_lumped_ports
from scgsim.aedt.runtime.families.hfss_requests import HfssModelSpec
from scgsim.aedt.runtime.families.hfss_setup import read_hfss_setup, setup_hfss
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
from scgsim.aedt.specs.common import (
    AedtResources,
)
from scgsim.contracts.sgb_sidecar import read_sidecar


@dataclass(frozen=True)
class PreparedHfss:
    """Carrier for one prepared HFSS app and its native readback records."""

    app: Any
    request: BoundAedtRequest
    project_path: Path
    geometry: dict[str, Any]
    ports: list[dict[str, Any]]
    setup: dict[str, Any] | None

    def __post_init__(self) -> None:
        if not isinstance(self.request, BoundAedtRequest):
            raise TypeError("PreparedHfss requires a bound AEDT request")
        for name in ("geometry", "ports", "setup"):
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
    """Import the model, assign its ports, set up, save and read back, before the explicit solve stage."""
    request = BoundAedtRequest.bind(run_dir, spec)
    bound = request.parse()
    if not isinstance(bound, HfssModelSpec):
        raise TypeError("bound HFSS request did not retain an HFSS Driven spec")
    project_path = request.workspace / f"{bound.project_name}.aedt"
    app = Hfss(
        project=str(project_path),
        design=bound.design_name,
        solution_type="DrivenTerminal",
        new_desktop=False,
        close_on_exit=False,
    )
    app.modeler.model_units = "um"
    geometry = imported_geometry_record(
        import_sgb_model(app, bound.model, family="hfss")
    )
    if bound.run_control is not None:
        setup_hfss(app, bound)
    if not app.save_project() or not project_path.is_file():
        raise RuntimeError("HFSS project was not saved before native port readback")
    setup = read_hfss_setup(app, bound) if bound.run_control is not None else None
    ports = read_saved_lumped_ports(
        app, geometry["lumped_ports"], read_sidecar(bound.model.path("sidecar"))
    )
    return PreparedHfss(app, request, project_path, geometry, ports, setup)


def solve_hfss(
    prepared: PreparedHfss,
    resources: AedtResources | None = None,
    resource_evidence: dict[str, Any] | None = None,
) -> None:
    """Run the one explicit HFSS setup solve."""
    spec = prepared.request.parse()
    if not isinstance(spec, HfssModelSpec):
        raise TypeError("bound HFSS request did not retain an HFSS Driven spec")
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
        raise TypeError("bound HFSS request did not retain an HFSS Driven spec")
    outputs, result_readback = export_hfss_results(
        prepared.app, run_dir, spec, detached_data(prepared.ports)
    )
    outputs = detached_data(outputs)
    result_readback = detached_data(result_readback)
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
        "materials": detached_data(prepared.geometry["materials"]),
        "region": detached_data(prepared.geometry["region"]),
        "geometry": detached_data(prepared.geometry),
        "setup": detached_data(prepared.setup),
        "convergence": convergence,
        "result_readback": result_readback,
        "save": {"ok": True, "project_sha256": outputs[relative]},
    }
