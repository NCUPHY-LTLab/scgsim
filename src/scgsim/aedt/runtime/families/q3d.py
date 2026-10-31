"""Q3D (ANSYS quasi-static 3D extractor) preparation, solve, export and readback.

Owns net assignment, setup and the prepare/solve/export entry points. Region
sealing lives in ``q3d_region``; body construction in ``native.q3d_bodies``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scgsim.aedt._io import file_sha256, read_json, write_csv
from scgsim.aedt._terminal_chain import q3d_terminal_segments
from scgsim.aedt.preparation.cohort import GEOMETRY_SOURCE_PATHS
from scgsim.aedt.results.convergence.q3d import read_q3d_convergence
from scgsim.aedt.results.matrices import parse_matrix_export
from scgsim.aedt.runtime.benchmark import attach_simulation_benchmark
from scgsim.aedt.runtime.families.q3d_region import (
    _read_q3d_region_ground,
    _seal_q3d_region,
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
    pyaedt_version as _pyaedt_version,
)
from scgsim.aedt.runtime.native.common import (
    saved_setup_properties as _saved_setup_properties,
)
from scgsim.aedt.runtime.native.materials import (
    realize_source_material,
    source_material_properties,
)
from scgsim.aedt.runtime.native.q3d_bodies import construct_q3d_bodies
from scgsim.aedt.specs.common import REQUIRED_AEDT_VERSION, AedtResources
from scgsim.aedt.specs.q3d import Q3dSpec

_Q3D_REGION_DIRECTIONS = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")

_Q3D_REGION_SHEET_NAMES = {
    "+X": ("SCGSimRegionGroundPX", "SCGSimRegionGroundPXThinConductor"),
    "-X": ("SCGSimRegionGroundNX", "SCGSimRegionGroundNXThinConductor"),
    "+Y": ("SCGSimRegionGroundPY", "SCGSimRegionGroundPYThinConductor"),
    "-Y": ("SCGSimRegionGroundNY", "SCGSimRegionGroundNYThinConductor"),
    "+Z": ("SCGSimRegionGroundPZ", "SCGSimRegionGroundPZThinConductor"),
    "-Z": ("SCGSimRegionGroundNZ", "SCGSimRegionGroundNZThinConductor"),
}


@dataclass(frozen=True)
class PreparedQ3d:
    """Carrier for one prepared Q3D app and its native readback records."""

    app: Any
    request: BoundAedtRequest
    project_path: Path
    materials: list[dict[str, Any]]
    region: dict[str, Any]
    nets: list[dict[str, Any]]
    setup: dict[str, Any] | None

    def __post_init__(self) -> None:
        if not isinstance(self.request, BoundAedtRequest):
            raise TypeError("PreparedQ3d requires a bound AEDT request")
        for name in ("materials", "region", "nets", "setup"):
            object.__setattr__(self, name, detached_data(getattr(self, name)))


def run_q3d(
    Q3d: Any,
    run_dir: Path,
    spec: Q3dSpec,
    resources: AedtResources | None = None,
    resource_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Use the same preparation stage as diagnostics, then solve and export."""
    if spec.run_control is None:
        raise ValueError("solving requires an explicit complete run_control")
    prepared = prepare_q3d(Q3d, run_dir, spec)
    solve_q3d(prepared, resources, resource_evidence)
    result = export_q3d(prepared)
    return attach_simulation_benchmark(
        result, prepared.app, run_dir, spec.run_control.setup_name
    )


def _source_material_bindings(
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
        native_name = spec.materials[material_id].library_name
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
    source_bindings = _source_material_bindings(run_dir, spec)
    project_path = run_dir / f"{spec.project_name}.aedt"
    app = Q3d(
        project=str(project_path),
        design=spec.design_name,
        new_desktop=False,
        close_on_exit=False,
    )
    if app.desktop_class.aedt_version_id != REQUIRED_AEDT_VERSION:
        raise RuntimeError("Q3D did not bind the owned AEDT 2024.2 desktop")
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
    nets = _assign_q3d_nets(app, spec)
    if spec.grounded_region_net is not None:
        grounded_region = _seal_q3d_region(app, spec)
        region["native_region_object_id"] = grounded_region["native_region_object_id"]
        region["native_bounding_box_um"] = grounded_region["native_bounding_box_um"]
        region["grounded_region"] = grounded_region
    if spec.run_control is not None:
        _setup_q3d(app, spec)
    if not app.save_project() or not project_path.is_file():
        raise RuntimeError("Q3D project was not saved before solve")
    _verify_saved_q3d_sheet_terminals(app, nets)
    setup = _read_q3d_setup(app, spec) if spec.run_control is not None else None
    if spec.grounded_region_net is not None:
        region["grounded_region"] = _read_q3d_region_ground(app, spec, region)
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


def prepared_q3d_result(prepared: PreparedQ3d) -> dict[str, Any]:
    """Detach native preparation facts; the transaction seals the released project."""
    return {
        "outputs": {},
        "connected": {
            "aedt_version": prepared.app.desktop_class.aedt_version_id,
            "pyaedt_version": _pyaedt_version(),
        },
        "project": prepared.project_path.relative_to(
            prepared.request.workspace
        ).as_posix(),
        "materials": detached_data(prepared.materials),
        "region": detached_data(prepared.region),
        "nets": detached_data(prepared.nets),
        "setup": detached_data(prepared.setup),
        "save": {"ok": True},
        "solver_invoked": False,
    }


def solve_q3d(
    prepared: PreparedQ3d,
    resources: AedtResources | None = None,
    resource_evidence: dict[str, Any] | None = None,
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
    outputs, result_readback = _export_q3d(prepared.app, run_dir, spec)
    outputs = detached_data(outputs)
    result_readback = detached_data(result_readback)
    if not prepared.app.save_project() or not prepared.project_path.is_file():
        raise RuntimeError("Q3D project was not saved")
    region = detached_data(prepared.region)
    if spec.grounded_region_net is not None:
        region["grounded_region"] = _read_q3d_region_ground(prepared.app, spec, region)
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


def _assign_sheet_terminal(
    app: Any, spec: Q3dSpec, object_name: str, side: str, name: str, net_name: str
) -> dict[str, Any]:
    """Bind authored exterior segments to actual Sheet edges and native excitation."""
    body = next(body for body in spec.bodies if body.body_id == object_name)
    effective = next(
        record for record in spec.effective_bodies if record["body_id"] == object_name
    )
    segments = q3d_terminal_segments(body, side)
    ring = body.exterior_um
    if ring[0] == ring[-1]:
        ring = ring[:-1]
    obj = app.modeler.get_object_from_name(object_name)
    edge_ids = []
    source_bindings = []
    for index in segments:
        endpoints = {
            (*ring[index], effective["effective_z_min_um"]),
            (*ring[(index + 1) % len(ring)], effective["effective_z_min_um"]),
        }
        matches = [
            edge
            for edge in obj.edges
            if {
                tuple(float(value) for value in vertex.position)
                for vertex in edge.vertices
            }
            == endpoints
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Q3D Sheet terminal source segment {index} has no unique native edge"
            )
        edge_id = int(matches[0].id)
        edge_ids.append(edge_id)
        source_bindings.append(
            {"source_segment_index": index, "native_edge_id": edge_id}
        )
    assignment = [
        f"NAME:{name}",
        "Edges:=",
        edge_ids,
        "Net:=",
        net_name,
        "TerminalType:=",
        "ConstantVoltage",
    ]
    method = (
        app.oboundary.AssignSource
        if name == f"{net_name}Source"
        else app.oboundary.AssignSink
    )
    stage = "assign_source_or_sink"
    native_assignment = None
    try:
        method(assignment)
        stage = "native_assignment_readback"
        native_assignment = app.oboundary.GetExcitationAssignment(name)
        assigned = [int(value) for value in native_assignment]
        if assigned != edge_ids:
            raise RuntimeError(
                f"Q3D Sheet terminal native assignment differs: {name!r}"
            )
    except Exception as exc:  # noqa: BLE001 -- preserve the native primary error.
        exc.add_note(
            f"Q3D Sheet terminal stage={stage}; name={name!r}; net={net_name!r}; "
            f"object={object_name!r}; side={side!r}; source_edges={source_bindings!r}; "
            f"selected_native_edge_ids={edge_ids!r}; native_assignment={native_assignment!r}"
        )
        raise
    return {
        "name": name,
        "object_name": object_name,
        "side": side,
        "native_edge_ids": assigned,
        "source_edges": source_bindings,
        "terminal_type": "ConstantVoltage",
    }


def _verify_saved_q3d_sheet_terminals(app: Any, nets: list[dict[str, Any]]) -> None:
    """Read saved Sheet terminal properties only after the preparation Save succeeds."""
    for net in nets:
        for kind in ("source", "sink"):
            terminal = net.get(kind)
            if terminal is None or "native_edge_ids" not in terminal:
                continue
            try:
                saved = app.design_properties["BoundarySetup"]["Boundaries"][
                    terminal["name"]
                ]
                if (
                    saved.get("Edges") != terminal["native_edge_ids"]
                    or saved.get("Net") != net["name"]
                    or saved.get("TerminalType") != terminal["terminal_type"]
                ):
                    raise RuntimeError(
                        f"Q3D saved Sheet terminal assignment differs: {terminal['name']!r}"
                    )
            except Exception as exc:  # noqa: BLE001 -- preserve failed saved readback.
                exc.add_note(
                    f"Q3D Sheet terminal stage=after_preparation_save_readback; "
                    f"net={net['name']!r}; terminal={terminal!r}"
                )
                raise


def _assign_q3d_nets(app: Any, spec: Q3dSpec) -> list[dict[str, Any]]:
    directions = {"-X": 0, "-Y": 1, "-Z": 2, "+X": 3, "+Y": 4, "+Z": 5}
    records: list[dict[str, Any]] = []
    for net in spec.nets:
        boundary = app.assign_net(list(net.object_names), net.name, net.net_type)
        if boundary is None or net.name not in app.net_names:
            raise RuntimeError(f"Q3D net assignment failed for {net.name!r}")
        expected_ids = [
            int(app.modeler.get_object_from_name(name).id) for name in net.object_names
        ]
        observed_ids = [
            int(value) for value in app.oboundary.GetExcitationAssignment(net.name)
        ]
        if observed_ids != expected_ids:
            raise RuntimeError(f"Q3D native net assignment mismatch for {net.name!r}")
        record: dict[str, Any] = {
            "name": net.name,
            "net_type": net.net_type,
            "object_names": list(net.object_names),
            "native_object_ids": observed_ids,
        }
        if net.net_type == "Signal" and net.source_object is not None:
            source_name = f"{net.name}Source"
            sink_name = f"{net.name}Sink"
            source_direction = directions[net.source_side]
            sink_direction = directions[net.sink_side]
            effective = {item["body_id"]: item for item in spec.effective_bodies}
            for kind, object_name, side, direction, terminal_name in (
                (
                    "source",
                    net.source_object,
                    net.source_side,
                    source_direction,
                    source_name,
                ),
                ("sink", net.sink_object, net.sink_side, sink_direction, sink_name),
            ):
                if effective[object_name]["representation"] == "sheet":
                    record[kind] = _assign_sheet_terminal(
                        app, spec, object_name, side, terminal_name, net.name
                    )
                    continue
                assign = app.source if kind == "source" else app.sink
                boundary = assign(
                    object_name,
                    direction=direction,
                    name=terminal_name,
                    net_name=net.name,
                )
                if boundary is None:
                    raise RuntimeError(f"Q3D {kind} assignment failed for {net.name!r}")
                faces = [
                    int(value)
                    for value in app.oboundary.GetExcitationAssignment(terminal_name)
                ]
                expected = int(app.modeler._get_faceid_on_axis(object_name, direction))  # noqa: SLF001 - PyAEDT has no public API for this
                if faces != [expected]:
                    raise RuntimeError(
                        f"Q3D native {kind} assignment mismatch for {net.name!r}"
                    )
                record[kind] = {
                    "name": terminal_name,
                    "object_name": object_name,
                    "side": side,
                    "native_face_ids": faces,
                }
        records.append(record)
    if app.net_names != [net.name for net in spec.nets]:
        raise RuntimeError("Q3D native net order does not match the structured spec")
    return records


def _setup_q3d(app: Any, spec: Q3dSpec) -> None:
    if app.setup_names:
        raise RuntimeError("new Q3D design must not inherit a setup")
    setup = app.create_setup(spec.run_control.setup_name)
    setup.dc_enabled = False
    setup.ac_rl_enabled = spec.solve_ac_rl
    setup.capacitance_enabled = True
    setup.props["AdaptiveFreq"] = f"{spec.run_control.frequency_ghz:g}GHz"
    setup.props["Cap"]["MaxPass"] = spec.run_control.maximum_passes
    setup.props["Cap"]["MinPass"] = spec.run_control.minimum_passes
    setup.props["Cap"]["MinConvPass"] = spec.run_control.minimum_converged_passes
    setup.props["Cap"]["PerError"] = spec.run_control.convergence_percent
    setup.props["Cap"]["PerRefine"] = spec.run_control.percent_refinement
    if spec.solve_ac_rl:
        setup.props["AC"]["MaxPass"] = spec.run_control.maximum_passes
        setup.props["AC"]["MinPass"] = spec.run_control.minimum_passes
        setup.props["AC"]["MinConvPass"] = spec.run_control.minimum_converged_passes
        setup.props["AC"]["PerError"] = spec.run_control.convergence_percent
        setup.props["AC"]["PerRefine"] = spec.run_control.percent_refinement
    if not setup.update():
        raise RuntimeError("Q3D setup update failed")


def _read_q3d_setup(app: Any, spec: Q3dSpec) -> dict[str, Any]:
    raw = _saved_setup_properties(app, spec.run_control.setup_name)
    cap = raw.get("Cap")
    ac = raw.get("AC")
    if not isinstance(cap, dict) or (spec.solve_ac_rl != isinstance(ac, dict)):
        raise TypeError("Q3D saved setup does not match its capacitance/AC-RL mode")
    if not spec.solve_ac_rl and ("AC" in raw or "DC" in raw):
        raise RuntimeError("Q3D capacitance-only setup saved AC-RL or DC controls")
    native = {
        "adaptive_frequency": raw.get("AdaptiveFreq"),
        "capacitance_maximum_passes": cap.get("MaxPass"),
        "capacitance_minimum_passes": cap.get("MinPass"),
        "capacitance_minimum_converged_passes": cap.get("MinConvPass"),
        "capacitance_convergence_percent": cap.get("PerError"),
        "capacitance_percent_refinement": cap.get("PerRefine"),
        "dc_enabled": "DC" in raw,
    }
    if spec.solve_ac_rl:
        native.update(
            {
                "ac_rl_maximum_passes": ac.get("MaxPass"),
                "ac_rl_minimum_passes": ac.get("MinPass"),
                "ac_rl_minimum_converged_passes": ac.get("MinConvPass"),
                "ac_rl_convergence_percent": ac.get("PerError"),
                "ac_rl_percent_refinement": ac.get("PerRefine"),
            }
        )
    expected = {
        "adaptive_frequency": f"{spec.run_control.frequency_ghz:g}GHz",
        "capacitance_maximum_passes": spec.run_control.maximum_passes,
        "capacitance_minimum_passes": spec.run_control.minimum_passes,
        "capacitance_minimum_converged_passes": (
            spec.run_control.minimum_converged_passes
        ),
        "capacitance_convergence_percent": spec.run_control.convergence_percent,
        "capacitance_percent_refinement": spec.run_control.percent_refinement,
        "dc_enabled": False,
    }
    if spec.solve_ac_rl:
        expected.update(
            {
                "ac_rl_maximum_passes": spec.run_control.maximum_passes,
                "ac_rl_minimum_passes": spec.run_control.minimum_passes,
                "ac_rl_minimum_converged_passes": (
                    spec.run_control.minimum_converged_passes
                ),
                "ac_rl_convergence_percent": spec.run_control.convergence_percent,
                "ac_rl_percent_refinement": spec.run_control.percent_refinement,
            }
        )
    if native != expected:
        raise RuntimeError(f"Q3D saved setup readback mismatch: {native!r}")
    return {"name": spec.run_control.setup_name, "native": native}


def _export_q3d(
    app: Any, run_dir: Path, spec: Q3dSpec
) -> tuple[dict[str, str], dict[str, Any]]:
    output_dir = run_dir / "results" / "q3d"
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    summaries: dict[str, Any] = {}
    frequency_hz = spec.run_control.frequency_ghz * 1e9
    exports = (("C", "c"), ("AC RL", "ac_rl")) if spec.solve_ac_rl else (("C", "c"),)
    normalized: list[dict[str, Any]] = []
    for problem, stem in exports:
        path = output_dir / f"{stem}_matrix.csv"
        app.odesign.ExportMatrixData(
            str(path),
            problem,
            "",
            f"{spec.run_control.setup_name} : LastAdaptive",
            "Original",
            "ohm",
            "nH",
            "pF",
            "mho",
            frequency_hz,
            "Maxwell",
            0,
            False,
            15,
            20,
            1,
        )
        titles = {
            "C": {"Capacitance Matrix": "C", "Conductance Matrix": "G"},
            "AC RL": {"AC Inductance Matrix": "L", "AC Resistance Matrix": "R"},
        }[problem]
        matrix_options = (
            {
                "expected_label_set": {
                    net.name for net in spec.nets if net.net_type == "Signal"
                }
            }
            if problem == "C"
            else {}
        )
        rows, summary = parse_matrix_export(
            path,
            "Q3D",
            problem,
            spec.run_control.frequency_ghz,
            titles,
            **matrix_options,
        )
        if spec.solve_ac_rl:
            normalized.extend(rows)
        summaries[stem] = summary
        hashes[path.relative_to(run_dir).as_posix()] = file_sha256(path)
    if not spec.solve_ac_rl:
        path = output_dir / "c_matrix.csv"
        return hashes, {
            "matrices": {
                "path": path.relative_to(run_dir).as_posix(),
                "frequency_ghz": spec.run_control.frequency_ghz,
                "native": summaries["c"],
                "primary_rows": len(rows),
            }
        }
    normalized_path = output_dir / "matrices.csv"
    write_csv(
        normalized_path,
        normalized,
        fieldnames=["problem_type", "quantity", "row", "column", "value", "unit"],
    )
    hashes[normalized_path.relative_to(run_dir).as_posix()] = file_sha256(
        normalized_path
    )
    return hashes, {
        "matrices": {
            "frequency_ghz": spec.run_control.frequency_ghz,
            "native": summaries,
            "normalized_rows": len(normalized),
        }
    }
