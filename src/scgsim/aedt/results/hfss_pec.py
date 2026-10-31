"""Check that the receipt's HFSS material and PEC evidence matches the spec.

PEC is perfect electric conductor. A superconducting layer is solved as PEC, either
as a solid with ``solve_inside`` off or as a sheet with a Perfect E boundary named
``SCGSimPEC``. Owns: the Region envelope, the per-object material records and the
PEC boundary assignment. It does not check ports, setup or convergence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

from scgsim.aedt.results.readback_common import positive_unique_ids
from scgsim.aedt.specs.hfss import HfssDrivenSpec, HfssEigenmodeSpec

_PEC_BOUNDARY_NAME = "SCGSimPEC"
_PEC_BINDING_KEYS = {
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
_SHEET_EVIDENCE_KEYS = {
    "native_boundary_name",
    "native_boundary_type",
    "raw_assignment_ids",
    "typed_assignment",
    "covered_face_ids",
}


def validate_hfss_pec_bindings(
    receipt: dict[str, Any], spec: HfssDrivenSpec | HfssEigenmodeSpec
) -> None:
    """Verify V3 PEC evidence against source material and object bindings."""
    _check_region_envelope(receipt, spec)
    records = receipt.get("materials")
    if not isinstance(records, list) or len(records) != len(spec.object_bindings):
        raise RuntimeError("HFSS V3 material binding evidence is invalid")
    pec_records, sheet_records = _check_material_records(records, spec)
    owners = _native_id_owners(pec_records)
    if not sheet_records:
        return
    _check_sheet_pec_assignment(sheet_records, owners)


def _check_region_envelope(
    receipt: dict[str, Any], spec: HfssDrivenSpec | HfssEigenmodeSpec
) -> None:
    envelope = spec.effective_region
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


@dataclass(frozen=True)
class _SourceMaps:
    """Spec lookups used for every material record."""

    materials: dict[str, Any]
    layers: dict[int, Any]
    effective_layers: dict[tuple[int, int], dict[str, Any]]

    @classmethod
    def of(cls, spec: HfssDrivenSpec | HfssEigenmodeSpec) -> _SourceMaps:
        return cls(
            materials=dict(spec.materials),
            layers={layer.layer: layer for layer in spec.layer_imports},
            effective_layers={
                (record["layer"], record["datatype"]): record
                for record in spec.effective_layer_imports
            },
        )


def _check_material_records(
    records: list[Any], spec: HfssDrivenSpec | HfssEigenmodeSpec
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Check each record in spec order; return (PEC records, sheet PEC records)."""
    maps = _SourceMaps.of(spec)
    pec_records: list[dict[str, Any]] = []
    sheet_records: list[dict[str, Any]] = []
    for record, object_binding in zip(records, spec.object_bindings, strict=True):
        material = maps.materials[object_binding.material_id]
        layer = _destination_layer(spec, maps, object_binding)
        _check_effective_geometry(record, layer, maps.effective_layers)
        _check_record_matches_binding(record, object_binding, material, layer)
        if not material.is_superconducting:
            _check_dielectric_record(record, material)
            continue
        native_object_type = _check_pec_binding(record, object_binding, material)
        if native_object_type == "Sheet":
            sheet_records.append(record)
        pec_records.append(record)
    return pec_records, sheet_records


def _destination_layer(
    spec: HfssDrivenSpec | HfssEigenmodeSpec, maps: _SourceMaps, object_binding: Any
) -> Any:
    """Find the imported layer an object landed in, by its native name prefix."""
    # An unknown layer raises KeyError here, as it always did; the layer used below
    # is the one chosen by name prefix.
    maps.layers[object_binding.layer]
    candidates = [
        item
        for item in spec.layer_imports
        if object_binding.object_name.startswith(f"{item.layer_name}_")
    ]
    if len(candidates) != 1:
        raise RuntimeError("HFSS current source destination is ambiguous")
    return candidates[0]


def _check_effective_geometry(
    record: Any, layer: Any, effective_layers: dict[tuple[int, int], Any]
) -> None:
    """The record must carry the effective (sheet or solid) geometry of its layer."""
    effective = effective_layers[(layer.layer, layer.datatype)]
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


def _check_record_matches_binding(
    record: dict[str, Any], object_binding: Any, material: Any, layer: Any
) -> None:
    if (
        record.get("object_name") != object_binding.object_name
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


def _check_dielectric_record(record: dict[str, Any], material: Any) -> None:
    if record.get("hfss_pec_binding") is not None:
        raise RuntimeError("non-PEC HFSS material has PEC binding evidence")
    observed = record.get("observed")
    if (
        not isinstance(observed, dict)
        or not isinstance(observed.get("native_material_name"), str)
        or observed["native_material_name"].casefold()
        != material.library_name.casefold()
        or not positive_unique_ids([record.get("native_object_id")], 1)
    ):
        raise RuntimeError("HFSS dielectric material native binding differs")


def _check_pec_binding(
    record: dict[str, Any], object_binding: Any, material: Any
) -> str:
    """Check one PEC record's binding; return its native object type."""
    binding = record.get("hfss_pec_binding")
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
    if (
        binding.get("native_object_id") != record.get("native_object_id")
        or binding.get("native_object_type") != record.get("native_object_type")
        or binding.get("native_face_ids") != record.get("native_face_ids")
    ):
        raise RuntimeError("HFSS effective object and PEC identities differ")
    native_object_type = binding.get("native_object_type")
    if native_object_type == "Solid":
        _check_solid_pec(record, binding)
    elif native_object_type == "Sheet":
        _check_sheet_pec_record(record, binding)
    else:
        raise RuntimeError("HFSS V3 PEC native object type is invalid")
    return native_object_type


def _check_solid_pec(record: dict[str, Any], binding: dict[str, Any]) -> None:
    evidence = binding.get("verified_evidence")
    if (
        binding.get("implementation") != "pec_material_solve_inside_false"
        or set(binding) != _PEC_BINDING_KEYS
        or not isinstance(evidence, dict)
        or set(evidence) != {"native_material_name", "native_solve_inside"}
        or not isinstance(evidence.get("native_material_name"), str)
        or evidence["native_material_name"].casefold() != "pec"
        or evidence.get("native_solve_inside") is not False
        or record.get("observed") != evidence
        or "requested_pec_boundary" in record
    ):
        raise RuntimeError("HFSS V3 solid PEC evidence is invalid")


def _check_sheet_pec_record(record: dict[str, Any], binding: dict[str, Any]) -> None:
    if (
        binding.get("implementation") != "perfect_e_sheet"
        or set(binding) != _PEC_BINDING_KEYS
        or record.get("requested_pec_boundary") != _PEC_BOUNDARY_NAME
    ):
        raise RuntimeError("HFSS V3 sheet PEC evidence is invalid")


class _NativeIdOwners(NamedTuple):
    """Source object name owning each native object ID and each native face ID."""

    by_object_id: dict[int, str]
    by_face_id: dict[int, str]


def _native_id_owners(pec_records: list[dict[str, Any]]) -> _NativeIdOwners:
    """Map native object IDs and face IDs to their source object, refusing overlap.

    HFSS reports a Perfect E assignment as a bare ID that may be an object or a
    face, so an ID that two things could own would make the assignment ambiguous.
    """
    object_by_native_id: dict[int, str] = {}
    object_by_face_id: dict[int, str] = {}
    for record in pec_records:
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
    return _NativeIdOwners(object_by_native_id, object_by_face_id)


def _check_sheet_pec_assignment(
    sheet_records: list[dict[str, Any]], owners: _NativeIdOwners
) -> None:
    """One shared Perfect E boundary must cover exactly the faces of every sheet."""
    all_target_faces: set[int] = set()
    target_faces_by_object: dict[str, set[int]] = {}
    target_object_id_by_object: dict[str, int] = {}
    for record in sheet_records:
        binding = record["hfss_pec_binding"]
        object_name = binding["source_object"]
        faces = set(binding["native_face_ids"])
        target_faces_by_object[object_name] = faces
        target_object_id_by_object[object_name] = binding["native_object_id"]
        all_target_faces.update(faces)

    reference = sheet_records[0]["hfss_pec_binding"].get("verified_evidence")
    if not isinstance(reference, dict):
        raise RuntimeError("HFSS V3 sheet PEC evidence is invalid")
    raw_ids = reference.get("raw_assignment_ids")
    typed = reference.get("typed_assignment")
    if (
        reference.get("native_boundary_name") != _PEC_BOUNDARY_NAME
        or reference.get("native_boundary_type") != "Perfect E"
        or not isinstance(raw_ids, list)
        or not all(
            isinstance(value, int) and not isinstance(value, bool) for value in raw_ids
        )
        or not isinstance(typed, list)
        or len(typed) != len(raw_ids)
    ):
        raise RuntimeError("HFSS V3 sheet PEC boundary evidence is invalid")

    covered: set[int] = set()
    covered_objects: set[str] = set()
    for raw_id, item in zip(raw_ids, typed, strict=True):
        _check_typed_assignment(
            raw_id,
            item,
            target_faces_by_object,
            target_object_id_by_object,
            owners,
        )
        covered.update(item["face_ids"])
        covered_objects.add(item["object_name"])
    if covered != all_target_faces or covered_objects != set(target_faces_by_object):
        raise RuntimeError("HFSS V3 PEC assignment coverage is incomplete")
    _check_per_sheet_evidence(sheet_records, target_faces_by_object, raw_ids, typed)


def _check_typed_assignment(
    raw_id: int,
    item: Any,
    target_faces_by_object: dict[str, set[int]],
    target_object_id_by_object: dict[str, int],
    owners: _NativeIdOwners,
) -> None:
    """One boundary assignment is either a whole sheet object or a single face."""
    if (
        not isinstance(item, dict)
        or set(item) != {"raw_id", "native_kind", "object_name", "face_ids"}
        or item.get("raw_id") != raw_id
    ):
        raise RuntimeError("HFSS V3 PEC typed assignment is invalid")
    object_name = item.get("object_name")
    faces = item.get("face_ids")
    if object_name not in target_faces_by_object or not isinstance(faces, list):
        raise RuntimeError("HFSS V3 PEC assignment targets a non-sheet object")
    object_owner = owners.by_object_id.get(raw_id)
    face_owner = owners.by_face_id.get(raw_id)
    if object_owner is not None and face_owner is not None:
        raise RuntimeError("HFSS V3 PEC assignment ID is ambiguous")
    if object_owner is None and face_owner is None:
        raise RuntimeError("HFSS V3 PEC assignment ID is unknown")
    if item.get("native_kind") == "object":
        if (
            object_owner != object_name
            or raw_id != target_object_id_by_object[object_name]
            or set(faces) != target_faces_by_object[object_name]
        ):
            raise RuntimeError("HFSS V3 PEC object assignment is incomplete")
    elif item.get("native_kind") == "face":
        if face_owner != object_name or faces != [raw_id]:
            raise RuntimeError("HFSS V3 PEC face assignment is invalid")
    else:
        raise RuntimeError("HFSS V3 PEC assignment kind is invalid")


def _check_per_sheet_evidence(
    sheet_records: list[dict[str, Any]],
    target_faces_by_object: dict[str, set[int]],
    raw_ids: list[int],
    typed: list[Any],
) -> None:
    """Every sheet repeats the shared boundary evidence plus its own covered faces."""
    for record in sheet_records:
        binding = record["hfss_pec_binding"]
        evidence = binding.get("verified_evidence")
        expected_covered = sorted(target_faces_by_object[binding["source_object"]])
        if (
            not isinstance(evidence, dict)
            or set(evidence) != _SHEET_EVIDENCE_KEYS
            or evidence.get("native_boundary_name") != _PEC_BOUNDARY_NAME
            or evidence.get("native_boundary_type") != "Perfect E"
            or evidence.get("raw_assignment_ids") != raw_ids
            or evidence.get("typed_assignment") != typed
            or evidence.get("covered_face_ids") != expected_covered
            or record.get("observed")
            != {
                "native_pec_boundary": _PEC_BOUNDARY_NAME,
                "native_pec_boundary_type": "Perfect E",
                "native_pec_face_ids": expected_covered,
                "native_pec_objects": sorted(target_faces_by_object),
            }
        ):
            raise RuntimeError("HFSS V3 per-sheet PEC evidence is inconsistent")
