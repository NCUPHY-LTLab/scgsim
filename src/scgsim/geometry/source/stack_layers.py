"""Read one stack layer record into a semantic Entity, with its material kind and its geometry, for the stack adapter and the plan."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from scgsim.geometry.models.input import SemanticEntitySpec
from scgsim.refusal import Declared


def _stack_layer(record: object) -> Declared:
    """A layer record of the stack, which the stack declares."""
    semantic_id = record.get("semantic_id") if isinstance(record, Mapping) else None
    return Declared(
        "stack layer" + ("" if semantic_id is None else f" {semantic_id!r}"),
        "the stack: the stack file's layers, or the component's conductor_regions "
        "and their PDK levels",
    )


def resolve_material_kind(
    record: Mapping[str, Any],
    *,
    material_id: Any,
    materials: Mapping[str, Any],
    owner: Declared,
) -> str:
    """Resolve one entity kind from explicit record and stack material facts."""
    if not isinstance(material_id, str) or not material_id:
        raise ValueError(
            owner.expected("material_id", "a non-empty string", material_id)
        )
    material = materials.get(material_id)
    if not isinstance(material, Mapping):
        raise TypeError(
            owner.expected(
                "material_id", "a material of the stack's materials", material_id
            )
        )
    stack_kind = material.get("kind")
    if stack_kind not in {"vacuum", "dielectric", "conductor"}:
        raise ValueError(
            owner.expected(
                f"material {material_id!r} kind",
                "vacuum, dielectric, or conductor",
                stack_kind,
            )
        )
    record_kind = record.get("material_kind")
    if record_kind is not None and record_kind not in {
        "vacuum",
        "dielectric",
        "conductor",
    }:
        raise ValueError(
            owner.expected(
                "material_kind", "vacuum, dielectric, or conductor", record_kind
            )
        )
    if record_kind is not None and record_kind != stack_kind:
        raise ValueError(
            owner.expected(
                "material_kind",
                f"materials[{material_id!r}].kind {stack_kind!r}",
                record_kind,
            )
        )
    return str(stack_kind)


def is_record_sequence(value: Any) -> bool:
    """Identify record sequences while excluding strings and bytes."""
    return isinstance(value, Sequence) and not isinstance(value, str | bytes)


def entity_from_layer_record(
    record: Any, *, materials: Mapping[str, Any]
) -> SemanticEntitySpec:
    """Build a semantic entity from one stack layer and its material record."""
    if not isinstance(record, Mapping):
        raise TypeError(_stack_layer(record).expected("record", "a mapping", record))

    layer = record.get("layer")
    datatype = record.get("datatype")
    if layer is None or datatype is None:
        raise ValueError(
            _stack_layer(record).refusal("it must define 'layer' and 'datatype'")
        )

    for required_field in ("semantic_id", "role", "material_id"):
        if required_field not in record:
            raise ValueError(
                _stack_layer(record).refusal(f"it must define {required_field!r}")
            )
    geometry = _layer_record_geometry(record)
    geometry.setdefault("gds_layer", layer)
    geometry.setdefault("gds_datatype", datatype)

    polygon_ids = record.get("polygon_ids", ())
    labels = record.get("labels", ())
    if isinstance(polygon_ids, str | bytes) or not isinstance(polygon_ids, Sequence):
        raise TypeError(
            _stack_layer(record).expected("polygon_ids", "a sequence", polygon_ids)
        )
    if isinstance(labels, str | bytes) or not isinstance(labels, Sequence):
        raise TypeError(_stack_layer(record).expected("labels", "a sequence", labels))

    return SemanticEntitySpec(
        semantic_id=record["semantic_id"],
        role=record["role"],
        material_id=record["material_id"],
        material_kind=resolve_material_kind(
            record,
            material_id=record["material_id"],
            materials=materials,
            owner=_stack_layer(record),
        ),
        priority=record.get("priority", 0),
        geometry_kind=record.get("geometry_kind", "layout_extrusion"),
        part_role=record.get("part_role"),
        attached_face_metal_semantic_id=record.get("attached_face_metal_semantic_id"),
        net_id=record.get("net_id"),
        polygon_ids=tuple(polygon_ids),
        labels=tuple(labels),
        host_void_semantic_id=record.get("host_void_semantic_id"),
        requires_construction_body=record.get("requires_construction_body", False),
        route_representations=record.get("route_representations", {}),
        geometry=geometry,
        metadata=record.get("metadata", {}),
    )


def _layer_record_geometry(record: Mapping[str, Any]) -> dict[str, Any]:
    """A stack layer's geometry: its ``geometry`` mapping, with ``z_um`` and ``thickness_um`` given beside it."""
    raw_geometry = record.get("geometry", {})
    if not isinstance(raw_geometry, Mapping):
        raise TypeError(
            _stack_layer(record).expected("geometry", "a mapping", raw_geometry)
        )
    geometry = dict(raw_geometry)
    if "z_um" in record or "thickness_um" in record:
        if "z_um" not in record or "thickness_um" not in record:
            raise ValueError(
                _stack_layer(record).refusal(
                    "it must define both 'z_um' and 'thickness_um', or use 'geometry'"
                )
            )
        geometry.setdefault("z_um", record["z_um"])
        geometry.setdefault("thickness_um", record["thickness_um"])
    if not geometry:
        raise ValueError(
            _stack_layer(record).refusal(
                "it must define 'geometry' or 'z_um' plus 'thickness_um'"
            )
        )
    return geometry
