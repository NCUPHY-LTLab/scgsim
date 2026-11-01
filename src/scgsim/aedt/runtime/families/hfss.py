"""Driven HFSS: mesh, ports, setup, and the prepare/solve/export entry points.

Owns the explicit-port (modal and terminal) HFSS request family. Body-first
Eigenmode lives in ``hfss_body_eigenmode``, port evidence in ``hfss_body_ports`` and
``hfss_readback``, and file export in ``hfss_export``. Generic execution never calls
EPR (energy-participation ratio) workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scgsim.aedt._io import file_sha256
from scgsim.aedt.results.convergence.hfss import read_hfss_convergence
from scgsim.aedt.runtime._text import unstripped_nonempty_text
from scgsim.aedt.runtime.benchmark import (
    attach_simulation_benchmark,
)
from scgsim.aedt.runtime.families.hfss_body_ports import (
    assign_body_terminal_ports,
    bind_body_terminal_evidence,
)
from scgsim.aedt.runtime.families.hfss_export import export
from scgsim.aedt.runtime.families.hfss_readback import (
    bind_port_evidence,
    face_for_side,
    native_mesh_operation_names,
    native_mesh_readback,
    native_modal_oo_readback,
    native_terminal_readback,
    require_native_mesh_properties,
)
from scgsim.aedt.runtime.native.common import (
    BoundAedtRequest,
    analyze_with_resources,
    detached_data,
)
from scgsim.aedt.runtime.native.common import (
    create_region as _create_region,
)
from scgsim.aedt.runtime.native.common import (
    import_and_bind as _import_and_bind,
)
from scgsim.aedt.runtime.native.common import (
    pyaedt_version as _pyaedt_version,
)
from scgsim.aedt.runtime.native.common import (
    saved_setup_properties as _saved_setup_properties,
)
from scgsim.aedt.runtime.native.hfss_eigenmode import (
    create_eigenmode_setup,
    read_eigenmode_setup,
)
from scgsim.aedt.specs.common import (
    REQUIRED_AEDT_VERSION,
    SURFACE_APPROXIMATION_LEVEL,
    AedtResources,
    ModalPort,
    TerminalPort,
)
from scgsim.aedt.specs.hfss import HfssEigenmodeSpec
from scgsim.aedt.specs.hfss_prepared import (
    HfssDrivenGeometrySpec,
)
from scgsim.aedt.specs.parse import HfssSpec


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
    spec: HfssSpec,
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


def prepare_hfss(Hfss: Any, run_dir: Path, spec: HfssSpec) -> PreparedHfss:
    """Construct and read back the HFSS model before the explicit solve stage."""
    request = BoundAedtRequest.bind(run_dir, spec)
    bound_spec = request.parse()
    if not isinstance(bound_spec, HfssSpec):
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
    if app.desktop_class.aedt_version_id != REQUIRED_AEDT_VERSION:
        raise RuntimeError("HFSS did not bind the owned AEDT 2024.2 desktop")
    app.modeler.model_units = "um"
    geometry: dict[str, Any] = {}
    if isinstance(spec, HfssDrivenGeometrySpec):
        from scgsim.aedt.runtime.native.hfss_geometry import (
            prepare_native_planar_geometry,
        )

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
        mesh = _assign_mesh(app, spec)
        ports = _assign_ports(app, spec)
    if spec.run_control is not None:
        _setup(app, spec)
    if not app.save_project() or not project_path.is_file():
        raise RuntimeError("HFSS project was not saved before native port readback")
    setup = _read_hfss_setup(app, spec) if spec.run_control is not None else None
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
    if not isinstance(spec, HfssSpec):
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
    if not isinstance(spec, HfssSpec):
        raise TypeError("bound HFSS request did not retain an HFSS spec")
    outputs, result_readback = export(
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


def _assign_mesh(hfss: Any, spec: HfssSpec) -> dict[str, Any]:
    grounds = [
        item.object_name for item in spec.object_bindings if item.role == "ground"
    ]
    signals = [
        item.object_name for item in spec.object_bindings if item.role == "signal"
    ]
    result: dict[str, Any] = {"surface": {}, "length": {}}
    for role, objects, name in (
        ("signal", signals, "SignalSurfaceApprox9"),
        ("ground", grounds, "GroundSurfaceApprox9"),
    ):
        operation = hfss.mesh.assign_surface_mesh(
            objects, SURFACE_APPROXIMATION_LEVEL, name
        )
        if operation is None or name not in native_mesh_operation_names(hfss):
            raise RuntimeError(f"{role} surface mesh readback failed")
        native = native_mesh_readback(hfss, name, objects)
        require_native_mesh_properties(
            native,
            {
                "Name": name,
                "Type": "Surface Approximation Based",
                "Region": "On Selection",
                "Curved Mesh Approximation Type": "Use Slider",
                "Curved Surface Mesh Resolution": SURFACE_APPROXIMATION_LEVEL,
            },
            f"{role} surface mesh",
        )
        result["surface"][role] = {
            "requested": {
                "name": name,
                "objects": objects,
                "level": SURFACE_APPROXIMATION_LEVEL,
            },
            "native": native,
        }
    if spec.length_mesh is not None:
        for role, objects, name in (
            ("signal", list(spec.length_mesh.signal_objects), "UniformCpwSignalLength"),
            ("ground", list(spec.length_mesh.ground_objects), "UniformCpwGroundLength"),
        ):
            operation = hfss.mesh.assign_length_mesh(
                objects,
                inside_selection=False,
                maximum_length=f"{spec.length_mesh.maximum_length_um:g}um",
                maximum_elements=1_000_000,
                name=name,
            )
            requested = {
                "Objects": objects,
                "MaxLength": f"{spec.length_mesh.maximum_length_um:g}um",
                "NumMaxElem": "1000000",
                "RestrictElem": True,
                "RestrictLength": True,
                "RefineInside": False,
                "Enabled": True,
            }
            if operation is None or name not in native_mesh_operation_names(hfss):
                raise RuntimeError(f"uniform CPW {role} length mesh readback failed")
            native = native_mesh_readback(hfss, name, objects)
            require_native_mesh_properties(
                native,
                {
                    "Name": name,
                    "Type": "Length Based",
                    "Region": "On Selection",
                    "Enabled": True,
                    "Restrict Length": True,
                    "Max Length": f"{spec.length_mesh.maximum_length_um:g}um",
                    "Restrict Max Elems": True,
                    "Max Elems": 1_000_000,
                },
                f"uniform CPW {role} length mesh",
            )
            result["length"][role] = {
                "requested": {"name": name, "properties": requested},
                "native": native,
            }
    return result


def _assign_ports(hfss: Any, spec: HfssSpec) -> list[dict[str, Any]]:
    if isinstance(spec, HfssEigenmodeSpec):
        if hfss.get_oo_name(hfss.odesign, "Excitations"):
            raise RuntimeError("HFSS Eigenmode design must not contain excitations")
        return []
    region = hfss.modeler.get_object_from_name("Region")
    if region is None:
        raise RuntimeError("Region is missing")
    faces = [
        (int(face.id), tuple(float(value) for value in face.center))
        for face in region.faces
    ]
    centers = {face_id: center for face_id, center in faces}
    records: list[dict[str, Any]] = []
    for port, effective_port in zip(spec.ports, spec.effective_ports, strict=True):
        face_id = face_for_side(faces, port.side)
        if isinstance(port, TerminalPort):
            before = set(hfss.oboundary.GetExcitationsOfType("Terminal"))
            boundary = hfss.wave_port(
                face_id,
                reference=list(port.reference_objects),
                name=port.name,
                renormalize=False,
                deembed=f"{port.deembed_um:g}um",
                terminals_rename=False,
            )
            after = set(hfss.oboundary.GetExcitationsOfType("Terminal"))
            names = sorted(after - before)
            if boundary is None or len(names) != 1:
                raise RuntimeError(
                    f"terminal excitation readback failed for {port.name!r}: {names!r}"
                )
            boundary_name = unstripped_nonempty_text(
                getattr(boundary, "name", port.name), "boundary.name"
            )
            native = native_terminal_readback(hfss, boundary_name, names[0], port)
            records.append(
                {
                    "index": port.index,
                    "boundary": boundary_name,
                    "terminal_excitation": names[0],
                    "face_id": face_id,
                    "face_center_um": list(centers[face_id]),
                    "native_terminal_names": names,
                    "requested": {
                        "reference_objects": list(port.reference_objects),
                        "renormalize": False,
                        "deembed_um": port.deembed_um,
                    },
                    "native": native,
                }
            )
        elif isinstance(port, ModalPort):
            if not isinstance(effective_port, ModalPort):
                raise TypeError("modal effective port must preserve ModalPort type")
            before = set(hfss.get_oo_name(hfss.odesign, "Excitations"))
            boundary = hfss.wave_port(
                face_id,
                integration_line=[
                    list(point) for point in effective_port.integration_line_um
                ],
                modes=1,
                impedance=50,
                name=port.name,
                renormalize=False,
                deembed=0,
                characteristic_impedance="Zpi",
            )
            after = set(hfss.get_oo_name(hfss.odesign, "Excitations"))
            names = sorted(after - before)
            if boundary is None or names != [port.name]:
                raise RuntimeError(
                    f"modal excitation readback failed for {port.name!r}: {names!r}"
                )
            native = native_modal_oo_readback(hfss, port.name)
            records.append(
                {
                    "index": port.index,
                    "boundary": port.name,
                    "modal_excitation": port.name,
                    "face_id": face_id,
                    "face_center_um": list(centers[face_id]),
                    "requested": {
                        "integration_line_um": [
                            list(point) for point in port.integration_line_um
                        ],
                        "modes": 1,
                        "renormalize": False,
                        "deembed_um": 0.0,
                        "characteristic_impedance": "Zpi",
                    },
                    "effective": {
                        "integration_line_um": [
                            list(point) for point in effective_port.integration_line_um
                        ]
                    },
                    "native": native,
                }
            )
        else:  # pragma: no cover - HfssDrivenSpec already closes this boundary.
            raise TypeError("unsupported HFSS driven port type")
    return records


def _setup(hfss: Any, spec: HfssSpec) -> None:
    if isinstance(spec, HfssEigenmodeSpec):
        create_eigenmode_setup(hfss, spec.run_control)
        return
    if hfss.setup_names:
        raise RuntimeError("new V1 design must not inherit a setup")
    setup = hfss.create_setup(spec.run_control.setup_name)
    setup.props["SolveType"] = (
        "DrivenTerminal" if spec.mode == "terminal" else "DrivenModal"
    )
    if not setup.enable_adaptive_setup_broadband(
        f"{spec.run_control.sweep.start_ghz}GHz",
        f"{spec.run_control.sweep.stop_ghz}GHz",
        max_passes=spec.run_control.maximum_passes,
        max_delta_s=spec.run_control.maximum_delta_s,
    ):
        raise RuntimeError("adaptive setup initialization failed")
    setup.props["MinimumPasses"] = spec.run_control.minimum_passes
    setup.props["MinimumConvergedPasses"] = spec.run_control.minimum_converged_passes
    setup.props["PercentRefinement"] = spec.run_control.percent_refinement
    if not setup.update():
        raise RuntimeError("HFSS Driven adaptive setup update failed")
    point_count = int(spec.run_control.sweep.points)
    sweep = hfss.create_linear_count_sweep(
        spec.run_control.setup_name,
        "GHz",
        spec.run_control.sweep.start_ghz,
        spec.run_control.sweep.stop_ghz,
        num_of_freq_points=point_count,
        name=spec.run_control.sweep_name,
        save_fields=False,
        sweep_type="Fast",
    )
    if sweep is None:
        raise RuntimeError("Fast sweep creation failed")
    actual = (
        str(sweep.props.get("Type")),
        str(sweep.props.get("RangeStart")),
        str(sweep.props.get("RangeEnd")),
        int(sweep.props.get("RangeCount")),
    )
    expected = (
        "Fast",
        f"{spec.run_control.sweep.start_ghz}GHz",
        f"{spec.run_control.sweep.stop_ghz}GHz",
        point_count,
    )
    if actual != expected:
        raise RuntimeError(f"Fast sweep readback mismatch: {actual!r} != {expected!r}")


def _read_hfss_setup(hfss: Any, spec: HfssSpec) -> dict[str, Any]:
    if isinstance(spec, HfssEigenmodeSpec):
        return read_eigenmode_setup(hfss, spec.run_control)
    raw = _saved_setup_properties(hfss, spec.run_control.setup_name)
    frequencies = raw.get("MultipleAdaptiveFreqsSetup")
    if not isinstance(frequencies, dict):
        raise TypeError("HFSS Driven saved setup lacks broadband frequencies")
    native = {
        "solve_type": raw.get("SolveType"),
        "low_frequency": frequencies.get("Low"),
        "high_frequency": frequencies.get("High"),
        "maximum_delta_s": raw.get("MaxDeltaS"),
        "maximum_passes": raw.get("MaximumPasses"),
        "minimum_passes": raw.get("MinimumPasses"),
        "minimum_converged_passes": raw.get("MinimumConvergedPasses"),
        "percent_refinement": raw.get("PercentRefinement"),
    }
    expected = {
        "solve_type": "Broadband",
        "low_frequency": f"{spec.run_control.sweep.start_ghz:g}GHz",
        "high_frequency": f"{spec.run_control.sweep.stop_ghz:g}GHz",
        "maximum_delta_s": spec.run_control.maximum_delta_s,
        "maximum_passes": spec.run_control.maximum_passes,
        "minimum_passes": spec.run_control.minimum_passes,
        "minimum_converged_passes": spec.run_control.minimum_converged_passes,
        "percent_refinement": spec.run_control.percent_refinement,
    }
    if native != expected:
        raise RuntimeError(f"HFSS saved setup readback mismatch: {native!r}")
    return {"name": spec.run_control.setup_name, "native": native}
