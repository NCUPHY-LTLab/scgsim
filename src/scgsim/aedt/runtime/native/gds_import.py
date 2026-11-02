"""Import declared GDS objects and bind their source layers and materials."""

from __future__ import annotations

from typing import Any

from scgsim.aedt.runtime.native.common import (
    native_boundary_names,
    native_boundary_type,
    native_object_boolean_property,
    native_object_evidence,
    native_object_property,
    resolve_native_assignment,
)
from scgsim.aedt.specs.common import ObjectBinding
from scgsim.aedt.specs.hfss import HfssDrivenSpec, HfssEigenmodeSpec


def import_and_bind(
    hfss: Any, spec: HfssDrivenSpec | HfssEigenmodeSpec
) -> list[dict[str, Any]]:
    """Import HFSS geometry using the spec's source-to-effective layer records."""
    effective_layers = {
        (int(item["layer"]), int(item["datatype"])): item
        for item in spec.effective_layer_imports
    }
    mapping = {
        effective_layers[(item.layer, item.datatype)]["native_import_layer"]: [
            (
                effective_layers[(item.layer, item.datatype)]["effective_z_min_um"],
                effective_layers[(item.layer, item.datatype)]["effective_z_max_um"]
                - effective_layers[(item.layer, item.datatype)]["effective_z_min_um"],
            ),
            item.layer_name,
        ]
        for item in spec.layer_imports
    }
    if not hfss.import_gds_3d(str(spec.gds_path), mapping, units="um", import_method=1):
        raise RuntimeError("HFSS import_gds_3d failed")
    hfss.modeler.refresh_all_ids()
    actual = set(hfss.modeler.object_names)
    expected = {item.object_name for item in spec.object_bindings}
    if actual != expected:
        raise RuntimeError(
            f"import object readback mismatch: expected {sorted(expected)!r}, got {sorted(actual)!r}"
        )
    pec_sheets: list[str] = []
    pec_solids: list[str] = []
    observed: list[dict[str, Any]] = []
    for binding in spec.object_bindings:
        record = _bind_imported_object(hfss, spec, effective_layers, binding)
        implementation = record.get("hfss_pec_binding", {}).get("implementation")
        if implementation == "perfect_e_sheet":
            pec_sheets.append(binding.object_name)
        elif implementation == "pec_material_solve_inside_false":
            pec_solids.append(binding.object_name)
        observed.append(record)
    if pec_sheets:
        _verify_sheet_pec(hfss, pec_sheets, observed)
    elif pec_solids and "SCGSimPEC" in native_boundary_names(hfss):
        raise RuntimeError("solid PEC binding unexpectedly has SCGSimPEC boundary")
    return observed


def _bind_imported_object(
    hfss: Any,
    spec: HfssDrivenSpec | HfssEigenmodeSpec,
    effective_layers: dict[tuple[int, int], Any],
    binding: ObjectBinding,
) -> dict[str, Any]:
    """Bind one imported object to its authored layer and material implementation."""
    obj = hfss.modeler.get_object_from_name(binding.object_name)
    if obj is None:
        raise RuntimeError(f"missing declared object {binding.object_name!r}")
    # AEDT's Geometry3D ``Group`` remains ``Model`` after GDS import. PyAEDT
    # 1.3.0 exposes the explicit destination layer through the imported
    # object-name prefix, so bind every declared object to its exact source
    # layer/datatype record rather than the importer-only numeric layer.
    destination_name = obj.name
    matches = [
        (candidate, effective_layers[(candidate.layer, candidate.datatype)])
        for candidate in spec.layer_imports
        if destination_name.startswith(f"{candidate.layer_name}_")
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"import destination-pair mismatch for {binding.object_name!r}: "
            f"got {[candidate.layer_name for candidate, _ in matches]!r}"
        )
    layer, effective = matches[0]
    if binding.layer != layer.layer:
        raise RuntimeError(
            f"import source-layer mismatch for {binding.object_name!r}: "
            f"expected {binding.layer!r}, got {layer.layer!r}"
        )
    representation = effective["representation"]
    expected_native_type = {"solid": "Solid", "sheet": "Sheet"}.get(representation)
    if expected_native_type is None:
        raise ValueError(
            f"unsupported HFSS conductor representation: {representation!r}"
        )
    material = spec.materials[binding.material_id]
    native = native_object_evidence(hfss, binding.object_name)
    if native["native_object_type"] != expected_native_type:
        raise RuntimeError(
            f"HFSS imported representation mismatch for {binding.object_name!r}: "
            f"expected {expected_native_type!r} from the declared effective layer, "
            f"got {native['native_object_type']!r}"
        )
    record: dict[str, Any] = {
        "object_name": binding.object_name,
        "layer": binding.layer,
        "datatype": layer.datatype,
        "layer_name": layer.layer_name,
        "native_import_layer": effective["native_import_layer"],
        "role": binding.role,
        "material_id": material.material_id,
        "kind": material.kind,
        "is_superconducting": material.is_superconducting,
        "requested_library_name": material.library_name,
        "native_destination_layer_prefix": layer.layer_name,
        "physical_layer_id": effective["physical_layer_id"],
        "representation": representation,
        "source_z_min_um": effective["z_min_um"],
        "source_z_max_um": effective["z_max_um"],
        "physical_thickness_um": effective["physical_thickness_um"],
        "effective_z_min_um": effective["effective_z_min_um"],
        "effective_z_max_um": effective["effective_z_max_um"],
        **native,
    }
    if material.is_superconducting:
        if representation == "solid":
            obj.material_name = "pec"
            obj.solve_inside = False
            fresh = native_object_evidence(hfss, binding.object_name)
            if (
                fresh["native_object_id"] != native["native_object_id"]
                or fresh["native_object_type"] != native["native_object_type"]
                or set(fresh["native_face_ids"]) != set(native["native_face_ids"])
            ):
                raise RuntimeError(
                    f"HFSS PEC native object identity changed for {binding.object_name!r}"
                )
            observed_material = native_object_property(obj, "Material").strip('"')
            observed_solve_inside = native_object_boolean_property(obj, "Solve Inside")
            if observed_material.casefold() != "pec" or observed_solve_inside:
                raise RuntimeError(
                    f"HFSS solid PEC readback mismatch for {binding.object_name!r}"
                )
            record["hfss_pec_binding"] = {
                "source_object": binding.object_name,
                "source_material_id": material.material_id,
                "source_material_kind": material.kind,
                "source_library_name": material.library_name,
                **fresh,
                "implementation": "pec_material_solve_inside_false",
                "verified_evidence": {
                    "native_material_name": observed_material,
                    "native_solve_inside": False,
                },
            }
            record["observed"] = {
                "native_material_name": observed_material,
                "native_solve_inside": False,
            }
        elif representation == "sheet":
            record["requested_pec_boundary"] = "SCGSimPEC"
            record["hfss_pec_binding"] = {
                "source_object": binding.object_name,
                "source_material_id": material.material_id,
                "source_material_kind": material.kind,
                "source_library_name": material.library_name,
                **native,
                "implementation": "perfect_e_sheet",
            }
    else:
        record["observed"] = _assign_library_material(hfss, obj, binding, material)
    return record


def _verify_sheet_pec(
    hfss: Any, pec_sheets: list[str], observed: list[dict[str, Any]]
) -> None:
    """Assign the shared Perfect E sheet boundary and verify exact coverage."""
    boundary = hfss.assign_perfect_e(pec_sheets, name="SCGSimPEC")
    if boundary is None or "SCGSimPEC" not in native_boundary_names(hfss):
        raise RuntimeError("PEC assignment readback failed")
    boundary_type = native_boundary_type(hfss, "SCGSimPEC")
    if boundary_type != "Perfect E":
        raise RuntimeError("PEC native boundary type mismatch")
    raw_assignment_ids = [
        int(native_id)
        for native_id in hfss.oboundary.GetBoundaryAssignment("SCGSimPEC")
    ]
    resolved, covered_faces, covered_objects = resolve_native_assignment(
        hfss, raw_assignment_ids
    )
    target_faces = {
        int(face_id)
        for record in observed
        if record.get("hfss_pec_binding", {}).get("implementation") == "perfect_e_sheet"
        for face_id in record["hfss_pec_binding"]["native_face_ids"]
    }
    if covered_faces != target_faces or covered_objects != set(pec_sheets):
        raise RuntimeError("PEC native assignment mismatch")
    for record in observed:
        binding = record.get("hfss_pec_binding")
        if (
            isinstance(binding, dict)
            and binding.get("implementation") == "perfect_e_sheet"
        ):
            target = set(binding["native_face_ids"])
            binding["verified_evidence"] = {
                "native_boundary_name": "SCGSimPEC",
                "native_boundary_type": boundary_type,
                "raw_assignment_ids": raw_assignment_ids,
                "typed_assignment": resolved,
                "covered_face_ids": sorted(covered_faces & target),
            }
            record["observed"] = {
                "native_pec_boundary": "SCGSimPEC",
                "native_pec_boundary_type": boundary_type,
                "native_pec_face_ids": sorted(covered_faces & target),
                "native_pec_objects": sorted(covered_objects),
            }


def _assign_library_material(
    hfss: Any, obj: Any, binding: ObjectBinding, material: Any
) -> dict[str, Any]:
    """Apply the declared library material and verify its native name."""
    existing = hfss.materials.exists_material(material.library_name)
    if not existing:
        raise RuntimeError(
            f"AEDT library material is unavailable: {material.library_name!r}"
        )
    obj.material_name = material.library_name
    observed_material = native_object_property(obj, "Material").strip('"')
    if observed_material.casefold() != material.library_name.casefold():
        raise RuntimeError(f"material readback mismatch for {binding.object_name!r}")
    return {"native_material_name": observed_material}
