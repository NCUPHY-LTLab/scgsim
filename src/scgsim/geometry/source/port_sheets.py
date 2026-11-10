"""Mark the face-metal owners of authored lumped-port sheets on a stack, before the geometry is built.

A port sheet lies on the support plane of its host conductors, so each polygon of its target layer must be
its own Entity (``split_polygons_as_entities``) for the sheet to name exact hosts.  This is stack
preparation shared by every backend that builds SGB geometry.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any


def with_port_sheet_metadata(
    stack: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any]:
    """Attach authored ports after marking their exact face-metal owners splittable."""
    payload = copy.deepcopy(dict(stack))
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError("stack metadata must be a mapping.")
    metadata = dict(metadata)
    existing = metadata.get("port_sheet_source_layers")
    if existing not in (None, [], ()) and records:
        raise ValueError("stack already defines port_sheet_source_layers.")
    effective_records = records if records else existing or ()
    if effective_records:
        layers = payload.get("layers")
        if not isinstance(layers, list):
            raise TypeError("stack layers must be a list for authored port sheets.")
        rewritten_layers = list(layers)
        for record in effective_records:
            if not isinstance(record, Mapping):
                raise TypeError("port-sheet source records must be mappings.")
            target_layer = record.get("target_layer")
            if not isinstance(target_layer, str) or not target_layer:
                raise ValueError("port-sheet source records require target_layer.")
            matches = [
                index
                for index, layer in enumerate(rewritten_layers)
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
                    f"authored port target_layer {target_layer!r} must match a logical stack layer."
                )
            for index in matches:
                layer = rewritten_layers[index]
                if not isinstance(layer, Mapping):
                    raise TypeError("matched logical stack layer must be a mapping.")
                if (
                    layer.get("role") != "metal"
                    or layer.get("part_role") != "face_metal"
                ):
                    raise ValueError(
                        f"authored port target_layer {target_layer!r} must have face-metal semantics."
                    )
                geometry = layer.get("geometry")
                if not isinstance(geometry, Mapping):
                    raise TypeError(
                        f"authored port target_layer {target_layer!r} must define geometry mapping."
                    )
                rewritten = dict(layer)
                rewritten_geometry = dict(geometry)
                rewritten_geometry["split_polygons_as_entities"] = True
                rewritten["geometry"] = rewritten_geometry
                rewritten_layers[index] = rewritten
        payload["layers"] = rewritten_layers
        metadata["port_sheet_source_layers"] = copy.deepcopy(list(effective_records))
    payload["metadata"] = metadata
    return payload
