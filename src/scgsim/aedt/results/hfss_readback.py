"""Strict offline validation for HFSS native model, setup, and solver readback."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from scgsim.aedt.results.convergence.hfss import read_hfss_convergence
from scgsim.aedt.results.readback_common import positive_unique_ids
from scgsim.aedt.specs.common import ModalPort
from scgsim.aedt.specs.hfss import (
    EigenmodeRunControl,
    HfssDrivenSpec,
    HfssEigenmodeSpec,
    HfssEprSpec,
    HfssRunControl,
    HfssSpec,
)


@dataclass
class HfssPecContext:
    """Source and assignment records shared by the ordered HFSS PEC checks."""

    spec: HfssDrivenSpec | HfssEigenmodeSpec
    materials: dict[str, Any]
    layers: dict[int, Any]
    effective_layers: dict[tuple[int, int], dict[str, Any]]
    pec_records: list[dict[str, Any]]
    sheet_records: list[dict[str, Any]]
    object_by_native_id: dict[int, str] = field(default_factory=dict)
    object_by_face_id: dict[int, str] = field(default_factory=dict)


def validate_hfss_pec_bindings(
    receipt: dict[str, Any], spec: HfssDrivenSpec | HfssEigenmodeSpec
) -> None:
    """Verify V3 PEC evidence against source material and object bindings."""
    _validate_hfss_pec_region(receipt, spec)
    records = receipt.get("materials")
    if not isinstance(records, list) or len(records) != len(spec.object_bindings):
        raise RuntimeError("HFSS V3 material binding evidence is invalid")
    context = HfssPecContext(
        spec,
        dict(spec.materials),
        {layer.layer: layer for layer in spec.layer_imports},
        (
            {}
            if spec.modeling is None
            else {
                (record["layer"], record["datatype"]): record
                for record in spec.effective_layer_imports
            }
        ),
        [],
        [],
    )
    for record, binding in zip(records, spec.object_bindings, strict=True):
        if _validate_hfss_pec_record(context, record, binding):
            context.pec_records.append(record)
    _validate_hfss_pec_identity_ids(context)
    if context.sheet_records:
        _validate_hfss_sheet_pec_boundary(context)


def _validate_hfss_pec_region(
    receipt: dict[str, Any], spec: HfssDrivenSpec | HfssEigenmodeSpec
) -> None:
    if spec.historical_physical or spec.historical_modeling:
        return
    envelope = cast(Any, spec.effective_region)
    region = receipt.get("region")
    keys = (
        "x_plus_um",
        "x_minus_um",
        "y_plus_um",
        "y_minus_um",
        "z_plus_um",
        "z_minus_um",
    )
    if (
        not isinstance(region, dict)
        or region.get("requested_padding_um") != list(spec.region_padding_um)
        or region.get("source_effective_envelope") != envelope
        or region.get("padding_um")
        != [envelope["effective_padding_um"][key] for key in keys]
        or not isinstance(region.get("native_bounding_box_um"), list)
        or len(region["native_bounding_box_um"]) != 6
    ):
        raise RuntimeError("HFSS source/effective Region envelope evidence differs")


def _validate_hfss_pec_record(
    context: HfssPecContext, record: Any, object_binding: Any
) -> bool:
    material = context.materials[object_binding.material_id]
    layer = context.layers[object_binding.layer]
    if context.effective_layers:
        layer = _validate_hfss_effective_material_geometry(
            context, record, object_binding
        )
    _validate_hfss_material_record(record, object_binding, material, layer)
    binding = record.get("hfss_pec_binding")
    if not material.is_superconducting:
        _validate_hfss_dielectric_record(context, record, material)
        return False
    _validate_hfss_pec_source_binding(record, binding, object_binding, material)
    if context.effective_layers and (
        binding.get("native_object_id") != record.get("native_object_id")
        or binding.get("native_object_type") != record.get("native_object_type")
        or binding.get("native_face_ids") != record.get("native_face_ids")
    ):
        raise RuntimeError("HFSS effective object and PEC identities differ")
    if binding.get("native_object_type") == "Solid":
        _validate_hfss_solid_pec(record, binding)
    elif binding.get("native_object_type") == "Sheet":
        _validate_hfss_sheet_pec(record, binding)
        context.sheet_records.append(record)
    else:
        raise RuntimeError("HFSS V3 PEC native object type is invalid")
    return True


def _validate_hfss_effective_material_geometry(
    context: HfssPecContext, record: Any, object_binding: Any
) -> Any:
    spec = context.spec
    candidates = [
        item
        for item in spec.layer_imports
        if object_binding.object_name.startswith(f"{item.layer_name}_")
    ]
    if len(candidates) != 1:
        raise RuntimeError("HFSS current source destination is ambiguous")
    layer = candidates[0]
    effective = context.effective_layers[(layer.layer, layer.datatype)]
    expected_fields = {
        "datatype": layer.datatype,
        "native_import_layer": effective["native_import_layer"],
        "physical_layer_id": effective["physical_layer_id"],
        "representation": effective["representation"],
        "source_z_min_um": effective["z_min_um"],
        "source_z_max_um": effective["z_max_um"],
        "physical_thickness_um": effective["physical_thickness_um"],
        "effective_z_min_um": effective["effective_z_min_um"],
        "effective_z_max_um": effective["effective_z_max_um"],
        "native_object_type": "Sheet"
        if effective["representation"] == "sheet"
        else "Solid",
    }
    if not isinstance(record, dict) or any(
        record.get(key) != value for key, value in expected_fields.items()
    ):
        raise RuntimeError(
            "HFSS effective material geometry differs from source mapping"
        )
    return layer


def _validate_hfss_material_record(
    record: Any, object_binding: Any, material: Any, layer: Any
) -> None:
    if (
        not isinstance(record, dict)
        or record.get("object_name") != object_binding.object_name
        or record.get("layer") != object_binding.layer
        or record.get("layer_name") != layer.layer_name
        or record.get("role") != object_binding.role
        or record.get("material_id") != material.material_id
        or record.get("kind") != material.kind
        or record.get("is_superconducting") is not material.is_superconducting
        or record.get("requested_library_name") != material.library_name
        or record.get("native_destination_layer_prefix") != layer.layer_name
    ):
        raise RuntimeError("HFSS V3 material binding does not match the spec")


def _validate_hfss_dielectric_record(
    context: HfssPecContext, record: dict[str, Any], material: Any
) -> None:
    if record.get("hfss_pec_binding") is not None:
        raise RuntimeError("non-PEC HFSS material has PEC binding evidence")
    if not context.effective_layers:
        return
    observed = record.get("observed")
    if (
        not isinstance(observed, dict)
        or not isinstance(observed.get("native_material_name"), str)
        or observed["native_material_name"].casefold()
        != material.library_name.casefold()
        or not positive_unique_ids([record.get("native_object_id")], 1)
    ):
        raise RuntimeError("HFSS dielectric material native binding differs")


def _validate_hfss_pec_source_binding(
    record: dict[str, Any], binding: Any, object_binding: Any, material: Any
) -> None:
    face_ids = binding.get("native_face_ids") if isinstance(binding, dict) else None
    if not isinstance(binding, dict) or (
        binding.get("source_object") != object_binding.object_name
        or binding.get("source_material_id") != material.material_id
        or binding.get("source_material_kind") != material.kind
        or binding.get("source_library_name") != material.library_name
        or not positive_unique_ids([binding.get("native_object_id")], 1)
        or not isinstance(face_ids, list)
        or not face_ids
        or not positive_unique_ids(face_ids, len(face_ids))
    ):
        raise RuntimeError("HFSS V3 PEC source binding evidence is invalid")


def _validate_hfss_solid_pec(record: dict[str, Any], binding: dict[str, Any]) -> None:
    evidence = binding.get("verified_evidence")
    if (
        binding.get("implementation") != "pec_material_solve_inside_false"
        or set(binding)
        != {
            "source_object",
            "source_material_id",
            "source_material_kind",
            "source_library_name",
            "native_object_id",
            "native_object_type",
            "native_face_ids",
            "implementation",
            "verified_evidence",
        }
        or not isinstance(evidence, dict)
        or set(evidence) != {"native_material_name", "native_solve_inside"}
        or not isinstance(evidence.get("native_material_name"), str)
        or evidence["native_material_name"].casefold() != "pec"
        or evidence.get("native_solve_inside") is not False
        or record.get("observed") != evidence
        or "requested_pec_boundary" in record
    ):
        raise RuntimeError("HFSS V3 solid PEC evidence is invalid")


def _validate_hfss_sheet_pec(record: dict[str, Any], binding: dict[str, Any]) -> None:
    if (
        binding.get("implementation") != "perfect_e_sheet"
        or set(binding)
        != {
            "source_object",
            "source_material_id",
            "source_material_kind",
            "source_library_name",
            "native_object_id",
            "native_object_type",
            "native_face_ids",
            "implementation",
            "verified_evidence",
        }
        or record.get("requested_pec_boundary") != "SCGSimPEC"
    ):
        raise RuntimeError("HFSS V3 sheet PEC evidence is invalid")


def _validate_hfss_pec_identity_ids(context: HfssPecContext) -> None:
    object_by_native_id: dict[int, str] = {}
    object_by_face_id: dict[int, str] = {}
    for record in context.pec_records:
        binding = record["hfss_pec_binding"]
        faces = set(binding["native_face_ids"])
        if set(object_by_face_id) & faces:
            raise RuntimeError("HFSS V3 PEC native faces are ambiguous")
        object_name = binding["source_object"]
        object_id = binding["native_object_id"]
        if object_id in object_by_native_id:
            raise RuntimeError("HFSS V3 PEC native object IDs are ambiguous")
        object_by_native_id[object_id] = object_name
        object_by_face_id.update({face_id: object_name for face_id in faces})
    if set(object_by_native_id) & set(object_by_face_id):
        raise RuntimeError("HFSS V3 PEC native assignment IDs are ambiguous")
    context.object_by_native_id = object_by_native_id
    context.object_by_face_id = object_by_face_id


def _validate_hfss_sheet_pec_boundary(context: HfssPecContext) -> None:
    target_faces_by_object: dict[str, set[int]] = {}
    target_object_id_by_object: dict[str, int] = {}
    all_target_faces: set[int] = set()
    for record in context.sheet_records:
        binding = record["hfss_pec_binding"]
        object_name = binding["source_object"]
        faces = set(binding["native_face_ids"])
        target_faces_by_object[object_name] = faces
        target_object_id_by_object[object_name] = binding["native_object_id"]
        all_target_faces.update(faces)
    reference = context.sheet_records[0]["hfss_pec_binding"].get("verified_evidence")
    raw_ids, typed = _validate_hfss_sheet_assignment_header(reference)
    targets = SheetPecTargets(
        target_faces_by_object, target_object_id_by_object, all_target_faces
    )
    _validate_hfss_sheet_assignments(context, raw_ids, typed, targets)
    _validate_hfss_sheet_observed(context, raw_ids, typed, target_faces_by_object)


def _validate_hfss_sheet_assignment_header(
    reference: Any,
) -> tuple[list[Any], list[Any]]:
    if not isinstance(reference, dict):
        raise RuntimeError("HFSS V3 sheet PEC evidence is invalid")
    raw_ids = reference.get("raw_assignment_ids")
    typed = reference.get("typed_assignment")
    if (
        reference.get("native_boundary_name") != "SCGSimPEC"
        or reference.get("native_boundary_type") != "Perfect E"
        or not isinstance(raw_ids, list)
        or not all(
            isinstance(value, int) and not isinstance(value, bool) for value in raw_ids
        )
        or not isinstance(typed, list)
        or len(typed) != len(raw_ids)
    ):
        raise RuntimeError("HFSS V3 sheet PEC boundary evidence is invalid")
    return raw_ids, typed


@dataclass(frozen=True)
class SheetPecTargets:
    """Native sheet objects and faces that the Perfect E boundary must cover."""

    faces_by_object: dict[str, set[int]]
    object_ids_by_name: dict[str, int]
    all_faces: set[int]


@dataclass(frozen=True)
class SheetPecAssignment:
    """One native object/face identity reported by the Perfect E boundary."""

    raw_id: int
    item: dict[str, Any]
    object_name: str
    faces: list[Any]
    object_owner: str | None
    face_owner: str | None


def _validate_hfss_sheet_assignments(
    context: HfssPecContext,
    raw_ids: list[Any],
    typed: list[Any],
    targets: SheetPecTargets,
) -> None:
    covered: set[int] = set()
    covered_objects: set[str] = set()
    for raw_id, item in zip(raw_ids, typed, strict=True):
        if (
            not isinstance(item, dict)
            or set(item) != {"raw_id", "native_kind", "object_name", "face_ids"}
            or item.get("raw_id") != raw_id
        ):
            raise RuntimeError("HFSS V3 PEC typed assignment is invalid")
        object_name = item.get("object_name")
        faces = item.get("face_ids")
        if object_name not in targets.faces_by_object or not isinstance(faces, list):
            raise RuntimeError("HFSS V3 PEC assignment targets a non-sheet object")
        object_owner = context.object_by_native_id.get(raw_id)
        face_owner = context.object_by_face_id.get(raw_id)
        if object_owner is not None and face_owner is not None:
            raise RuntimeError("HFSS V3 PEC assignment ID is ambiguous")
        if object_owner is None and face_owner is None:
            raise RuntimeError("HFSS V3 PEC assignment ID is unknown")
        _validate_hfss_sheet_assignment_kind(
            SheetPecAssignment(
                raw_id, item, object_name, faces, object_owner, face_owner
            ),
            targets,
        )
        covered.update(faces)
        covered_objects.add(object_name)
    if covered != targets.all_faces or covered_objects != set(targets.faces_by_object):
        raise RuntimeError("HFSS V3 PEC assignment coverage is incomplete")


def _validate_hfss_sheet_assignment_kind(
    assignment: SheetPecAssignment, targets: SheetPecTargets
) -> None:
    item = assignment.item
    if item.get("native_kind") == "object":
        if (
            assignment.object_owner != assignment.object_name
            or assignment.raw_id != targets.object_ids_by_name[assignment.object_name]
            or set(assignment.faces) != targets.faces_by_object[assignment.object_name]
        ):
            raise RuntimeError("HFSS V3 PEC object assignment is incomplete")
    elif item.get("native_kind") == "face":
        if assignment.face_owner != assignment.object_name or assignment.faces != [
            assignment.raw_id
        ]:
            raise RuntimeError("HFSS V3 PEC face assignment is invalid")
    else:
        raise RuntimeError("HFSS V3 PEC assignment kind is invalid")


def _validate_hfss_sheet_observed(
    context: HfssPecContext,
    raw_ids: list[Any],
    typed: list[Any],
    target_faces_by_object: dict[str, set[int]],
) -> None:
    for record in context.sheet_records:
        binding = record["hfss_pec_binding"]
        evidence = binding.get("verified_evidence")
        expected_covered = sorted(target_faces_by_object[binding["source_object"]])
        if (
            not isinstance(evidence, dict)
            or set(evidence)
            != {
                "native_boundary_name",
                "native_boundary_type",
                "raw_assignment_ids",
                "typed_assignment",
                "covered_face_ids",
            }
            or evidence.get("native_boundary_name") != "SCGSimPEC"
            or evidence.get("native_boundary_type") != "Perfect E"
            or evidence.get("raw_assignment_ids") != raw_ids
            or evidence.get("typed_assignment") != typed
            or evidence.get("covered_face_ids") != expected_covered
            or record.get("observed")
            != {
                "native_pec_boundary": "SCGSimPEC",
                "native_pec_boundary_type": "Perfect E",
                "native_pec_face_ids": expected_covered,
                "native_pec_objects": sorted(target_faces_by_object),
            }
        ):
            raise RuntimeError("HFSS V3 per-sheet PEC evidence is inconsistent")


def validate_hfss_setup(receipt: dict[str, Any], spec: Any) -> None:
    """Bind the native setup record to the requested HFSS controls."""
    if isinstance(spec, (HfssEigenmodeSpec, HfssEprSpec)):
        run_control = cast(EigenmodeRunControl, spec.run_control)
        native = {
            "minimum_frequency": f"{run_control.minimum_frequency_ghz:g}GHz",
            "num_modes": run_control.num_modes,
            "maximum_delta_frequency_percent": (
                run_control.maximum_delta_frequency_percent
            ),
            "maximum_passes": run_control.maximum_passes,
            "minimum_passes": run_control.minimum_passes,
            "minimum_converged_passes": run_control.minimum_converged_passes,
            "percent_refinement": run_control.percent_refinement,
        }
    else:
        run_control = cast(HfssRunControl, spec.run_control)
        native = {
            "solve_type": "Broadband",
            "low_frequency": f"{run_control.sweep.start_ghz:g}GHz",
            "high_frequency": f"{run_control.sweep.stop_ghz:g}GHz",
            "maximum_delta_s": run_control.maximum_delta_s,
            "maximum_passes": run_control.maximum_passes,
            "minimum_passes": run_control.minimum_passes,
            "minimum_converged_passes": run_control.minimum_converged_passes,
            "percent_refinement": run_control.percent_refinement,
        }
    setup = receipt.get("setup")
    expected = {"name": run_control.setup_name, "native": native}
    if (
        not isinstance(setup, dict)
        or (
            {key: setup.get(key) for key in expected}
            if isinstance(spec, HfssEprSpec)
            else setup
        )
        != expected
    ):
        raise RuntimeError("HFSS native setup evidence is invalid")


def validate_hfss_setup_and_convergence(
    root: Path, receipt: dict[str, Any], spec: HfssSpec
) -> None:
    """Verify HFSS setup and convergence artifacts against the completed receipt."""
    validate_hfss_setup(receipt, spec)
    historical_profile = (
        eigenmode_convergence_source(receipt) == "profile"
        if isinstance(spec, HfssEigenmodeSpec)
        else False
    )
    if receipt.get("convergence") != read_hfss_convergence(
        root, spec, historical_profile=historical_profile
    ):
        raise RuntimeError("HFSS native convergence evidence is invalid")


def eigenmode_convergence_source(receipt: dict[str, Any]) -> str:
    """Read the canonical source variant for Eigenmode convergence evidence."""
    convergence = receipt.get("convergence")
    sources = convergence.get("sources") if isinstance(convergence, dict) else None
    if not isinstance(sources, dict):
        raise RuntimeError("Eigenmode convergence sources are invalid")
    if set(sources) == {"export_convergence"}:
        return "export_convergence"
    if set(sources) == {"asol", "profile"}:
        return "profile"
    raise RuntimeError("Eigenmode convergence sources are not canonical")


def validate_eigenmode_readback(
    receipt: dict[str, Any], spec: HfssEigenmodeSpec | HfssEprSpec
) -> None:
    """Verify that the receipt contains the requested native Eigenmode outputs."""
    readback = receipt.get("result_readback")
    if isinstance(spec, HfssEprSpec):
        # Body-first loads belong to geometry; this producer has no ports field.
        if "ports" in receipt:
            raise RuntimeError("body-first Eigenmode receipt must not contain ports")
        readback = readback.get("final_modes") if isinstance(readback, dict) else None
    elif receipt.get("ports") != []:
        raise RuntimeError("HFSS Eigenmode receipt must not contain ports")
    values = readback.get("eigenmodes") if isinstance(readback, dict) else None
    if not isinstance(values, dict):
        raise TypeError("completed receipt has no Eigenmode result readback")
    run_control = cast(EigenmodeRunControl, spec.run_control)
    modes = run_control.num_modes
    frequencies = values.get("frequencies_ghz")
    q_factors = values.get("q_factors")
    if (
        values.get("modes") != modes
        or values.get("frequency_unit") != "GHz"
        or values.get("mode_indices") != list(range(1, modes + 1))
        or not isinstance(frequencies, list)
        or len(frequencies) != modes
        or not all(
            isinstance(value, (int, float)) and math.isfinite(value) and value > 0
            for value in frequencies
        )
        or not isinstance(q_factors, list)
        or len(q_factors) != modes
        or not all(
            isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
            for value in q_factors
        )
        or values.get("native_export") != "eigenmodes.eig"
        or not isinstance(values.get("native_export_bytes"), int)
        or values["native_export_bytes"] <= 0
    ):
        raise RuntimeError("completed receipt Eigenmode result readback is invalid")


def validate_modal_native_evidence(ports: Any, spec: HfssDrivenSpec) -> None:
    """Bind the native modal ports to the requested source ports."""
    if not isinstance(ports, list) or len(ports) != 2:
        raise RuntimeError("completed receipt has invalid modal ports")
    native_ports = spec.ports if spec.modeling is None else spec.effective_ports
    for record, port, native_port in zip(ports, spec.ports, native_ports, strict=True):
        if not isinstance(port, ModalPort) or not isinstance(record, dict):
            raise TypeError("completed receipt has invalid modal port identity")
        if (
            record.get("index") != port.index
            or record.get("boundary") != port.name
            or record.get("modal_excitation") != port.name
            or not isinstance(record.get("face_id"), int)
            or not isinstance(record.get("face_center_um"), list)
            or len(record["face_center_um"]) != 3
            or record.get("requested")
            != {
                "integration_line_um": [
                    list(point) for point in port.integration_line_um
                ],
                "modes": 1,
                "renormalize": False,
                "deembed_um": 0.0,
                "characteristic_impedance": "Zpi",
            }
        ):
            raise RuntimeError("completed receipt modal request does not match spec")
        modal_native_port = cast(ModalPort, native_port)
        if spec.modeling is not None and record.get("effective") != {
            "integration_line_um": [
                list(point) for point in modal_native_port.integration_line_um
            ]
        }:
            raise RuntimeError(
                "completed receipt modal effective endpoints differ from source map"
            )
        native = record.get("native")
        if not isinstance(native, dict):
            raise TypeError("completed receipt has no native modal evidence")
        expected_names = [item.name for item in spec.ports[: port.index]]
        expected_boundary = {
            "bound_type": "Wave Port",
            "wave_port_type": "Modal",
            "faces": [record["face_id"]],
            "num_modes": 1,
            "deembed": False,
            "mode_number": 1,
            "use_integration_line": True,
            "characteristic_impedance": "Zpi",
        }
        saved_boundary = native.get("saved_boundary")
        integration_line = (
            saved_boundary.get("integration_line_um")
            if isinstance(saved_boundary, dict)
            else None
        )
        if (
            native.get("excitation_names") != expected_names
            or native.get("boundary_properties")
            != {
                "Deembed": False,
                "Name": port.name,
                "Num Modes": "1",
                "Renorm All Modes": False,
                "Type": "Wave Port",
            }
            or not isinstance(saved_boundary, dict)
            or {
                key: value
                for key, value in saved_boundary.items()
                if key != "integration_line_um"
            }
            != expected_boundary
            or not isinstance(integration_line, list)
            or len(integration_line) != 2
            or any(
                not math.isclose(float(actual), wanted, abs_tol=1e-9)
                for actual_point, expected_point in zip(
                    integration_line, modal_native_port.integration_line_um, strict=True
                )
                for actual, wanted in zip(actual_point, expected_point, strict=True)
            )
        ):
            raise RuntimeError("completed receipt modal native evidence mismatch")
