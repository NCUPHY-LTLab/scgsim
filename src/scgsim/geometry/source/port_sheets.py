"""Mark the face-metal owners of authored lumped-port sheets on a stack, before the geometry is built.

A port sheet lies on the support plane of its host conductors, so each polygon of its target layer must be
its own Entity (``split_polygons_as_entities``) for the sheet to name exact hosts.  This is stack
preparation shared by every backend that builds SGB geometry.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from scgsim.refusal import Declared

_PORTS = Declared("authored port sheets", "the ports added to the simulation")
_STACK = Declared("the stack", "the stack the ports are added to")


def with_port_sheet_metadata(
    stack: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any]:
    """Attach authored ports after marking their exact face-metal owners splittable."""
    payload = copy.deepcopy(dict(stack))
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError(_STACK.expected("metadata", "a mapping", metadata))
    metadata = dict(metadata)
    existing = metadata.get("port_sheet_source_layers")
    if existing not in (None, [], ()) and records:
        raise ValueError(
            _STACK.refusal(
                "it already defines port_sheet_source_layers; add the ports in one place"
            )
        )
    effective_records = records if records else existing or ()
    if effective_records:
        layers = payload.get("layers")
        if not isinstance(layers, list):
            raise TypeError(_STACK.expected("layers", "a list", layers))
        rewritten_layers = list(layers)
        for record in effective_records:
            target_layer = _port_target_layer(record)
            for index in _target_layer_indexes(rewritten_layers, target_layer):
                rewritten_layers[index] = _splittable_face_metal(
                    rewritten_layers[index], target_layer
                )
        payload["layers"] = rewritten_layers
        metadata["port_sheet_source_layers"] = copy.deepcopy(list(effective_records))
    payload["metadata"] = metadata
    return payload


def _port_target_layer(record: Any) -> str:
    """The logical layer an authored port record names as its sheet's target."""
    if not isinstance(record, Mapping):
        raise TypeError(_PORTS.expected("record", "a mapping", record))
    target_layer = record.get("target_layer")
    if not isinstance(target_layer, str) or not target_layer:
        raise ValueError(
            _PORTS.expected(
                "target_layer", "a non-empty layer name", record.get("target_layer")
            )
        )
    return target_layer


def _target_layer_indexes(layers: Sequence[Any], target_layer: str) -> list[int]:
    """The stack layers whose semantic id or logical layer is the target; at least one must be."""
    matches = [
        index
        for index, layer in enumerate(layers)
        if isinstance(layer, Mapping)
        and (
            layer.get("semantic_id") == target_layer
            or (
                isinstance(layer.get("metadata"), Mapping)
                and layer["metadata"].get("logical_layer_id") == target_layer
            )
        )
    ]
    if not matches:
        raise ValueError(
            _PORTS.expected(
                "target_layer", "a logical layer of the stack", target_layer
            )
        )
    return matches


def _splittable_face_metal(layer: Any, target_layer: str) -> dict[str, Any]:
    """A face-metal stack layer whose polygons each become their own Entity."""
    if not isinstance(layer, Mapping):
        raise TypeError(_STACK.expected(f"layer {target_layer!r}", "a mapping", layer))
    if layer.get("role") != "metal" or layer.get("part_role") != "face_metal":
        raise ValueError(
            _PORTS.expected(
                f"target_layer {target_layer!r}",
                "a face-metal layer",
                (layer.get("role"), layer.get("part_role")),
            )
        )
    geometry = layer.get("geometry")
    if not isinstance(geometry, Mapping):
        raise TypeError(
            _STACK.expected(f"layer {target_layer!r} geometry", "a mapping", geometry)
        )
    rewritten = dict(layer)
    rewritten_geometry = dict(geometry)
    rewritten_geometry["split_polygons_as_entities"] = True
    rewritten["geometry"] = rewritten_geometry
    return rewritten
