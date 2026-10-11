"""Acquire source physical declarations and lower one effective geometry/stack/enclosure.

Request declarations and pure Z-map rules live in specs; this module owns
Entity/stack binding. Original source and serialized provenance stay immutable.
"""
from __future__ import annotations
from typing import Any
from scgsim.aedt.specs.modeling import (
    Modeling, PhysicalLayerSpec, EffectiveLayer, _ZMap, _require_source_film_interval,
)

def _source_physical_layers(build_input: Any, stack: Any, modeling: Modeling
                            ) -> tuple[tuple[PhysicalLayerSpec, ...], dict[str, str]]:
    """Read explicit source level declarations and bind qualified Entities."""
    from scgsim.semantics.route_a import geometry_z_range

    records = {item["semantic_id"]: item for item in stack["layers"]}
    records.update(stack["solution_regions"])
    layers = []
    bindings = {}
    for entity in build_input.entities:
        record = records.get(entity.metadata.get("source_semantic_id", entity.semantic_id), {})
        declaration = entity.metadata.get("aedt_modeling", record.get("metadata", {}).get("aedt_modeling"))
        if declaration is None:
            if modeling == "thin_film" and entity.material_kind == "conductor":
                raise ValueError(f"Thin Film conductor {entity.semantic_id!r} lacks declared physical layer modeling")
            continue
        low, high = geometry_z_range(record.get("geometry", entity.geometry), entity.semantic_id)
        layer = PhysicalLayerSpec(
            z_min_um=low, z_max_um=high,
            source_thickness_um=record.get("geometry", entity.geometry).get("thickness_um"),
            **declaration)
        layers.append(layer)
        bindings[entity.semantic_id] = layer.physical_layer_id
    mapping = _ZMap(modeling, layers)
    return tuple(mapping.layers.values()), bindings

def _mapped_geometry(geometry: Any, mapping: _ZMap, effective: EffectiveLayer | None = None) -> dict[str, Any]:
    from scgsim.aedt._data import detached
    from scgsim.semantics.route_a import geometry_z_range

    result = detached(geometry)
    low, high = geometry_z_range(geometry, "source geometry")
    if effective is not None and effective.representation == "sheet":
        _require_source_film_interval(low, high, effective.source)
    low, high = (effective.z_min_um, effective.z_max_um) if effective is not None else (mapping.map_z(low), mapping.map_z(high))
    result.update(z_min_um=low, z_max_um=high)
    if "z_um" in result:
        result["z_um"] = low
    if "thickness_um" in result:
        result["thickness_um"] = high - low
    return result

def _effective_geometry_input(build_input: Any, stack: Any, modeling: Modeling):
    """Derive one detached geometry/stack map without changing source facts."""
    from dataclasses import replace
    from scgsim.aedt._data import detached, canonical_sha256
    from scgsim.semantics.route_a import geometry_z_range
    from scgsim.geometry.models.representation import (
        EntityRepresentation, EffectiveEnclosure, RepresentationContext,
    )

    layers, bindings = _source_physical_layers(build_input, stack, modeling)
    mapping = _ZMap(modeling, layers)
    entities = []
    representations = {}
    enclosures = {}
    effective_stack = detached(stack)
    records = {record["semantic_id"]: record for record in effective_stack["layers"]}
    records.update(effective_stack["solution_regions"])
    for entity in build_input.entities:
        identity = bindings.get(entity.semantic_id)
        layer = mapping.layer(identity) if identity is not None else None
        geometry = _mapped_geometry(entity.geometry, mapping, layer)
        representation = "sheet" if layer is not None and layer.representation == "sheet" else "solid"
        primitive_representation = ("surface_sheet" if representation == "sheet" else "cutout_boundary_shell") if entity.material_kind == "conductor" else "solution_domain"
        trace = {"physical_layer_id": identity, "representation": representation,
                 "physical_thickness_um": (layer.source.physical_thickness_um if layer is not None and layer.source.kind in {"stack_film", "local_film"} else entity.geometry.get("thickness_um", entity.geometry.get("z_max_um", 0)-entity.geometry.get("z_min_um", 0))),
                 "source_geometry": detached(entity.geometry),
                 "effective_z_min_um": geometry["z_min_um"],
                 "effective_z_max_um": geometry["z_max_um"]}
        representations[entity.semantic_id] = EntityRepresentation(
            physical_layer_id=identity,
            representation=representation,
            source_z_interval_um=geometry_z_range(entity.geometry, entity.semantic_id),
            effective_z_interval_um=(geometry["z_min_um"], geometry["z_max_um"]),
        )
        metadata = {**entity.metadata, "aedt_effective_layer": trace}
        # Retain the recorded payload basis; planning consumes only the context.
        entities.append(replace(entity, geometry=geometry, metadata=metadata,
                                route_representations={**entity.route_representations, "_effective": primitive_representation}))
        record = records.get(entity.metadata.get("source_semantic_id", entity.semantic_id))
        if record is not None:
            record["geometry"] = geometry
            record["metadata"] = {**record.get("metadata", {}), "aedt_effective_layer": trace}
    # The source vacuum recipe already declares the full enclosure. Map its
    # endpoints once; a local film collapse must not recompute that enclosure
    # from shorter effective bodies plus the original source padding.
    from scgsim.geometry.planning.domain import _auto_vacuum_envelope_bounds

    effective_input = replace(build_input, entities=tuple(entities))
    context = RepresentationContext(entities=representations)
    for index, entity in enumerate(entities):
        if not entity.metadata.get("is_auto_vacuum_region"):
            continue
        body_bounds = _auto_vacuum_envelope_bounds(
            effective_input, route=None, representation_context=context
        )
        bounds = {**entity.geometry["domain_bounds_um"],
                  "z_min_um": entity.geometry["z_min_um"],
                  "z_max_um": entity.geometry["z_max_um"]}
        padding = {
            f"{axis}_minus_um": body_bounds[f"{axis}_min_um"] - bounds[f"{axis}_min_um"]
            for axis in ("x", "y", "z")
        }
        padding.update({
            f"{axis}_plus_um": bounds[f"{axis}_max_um"] - body_bounds[f"{axis}_max_um"]
            for axis in ("x", "y", "z")
        })
        envelope = {
            "source_geometry": detached(entity.metadata["aedt_effective_layer"]["source_geometry"]),
            "source_padding_um": detached(entity.metadata["vacuum_region_padding_um"]),
            "effective_bounds_um": bounds,
            "effective_body_bounds_um": body_bounds,
            "effective_padding_um": padding,
        }
        original_geometry = envelope["source_geometry"]
        original_low, original_high = geometry_z_range(original_geometry, entity.semantic_id)
        enclosures[entity.semantic_id] = EffectiveEnclosure(
            source_bounds_um={**original_geometry["domain_bounds_um"],
                              "z_min_um": original_low, "z_max_um": original_high},
            effective_bounds_um=bounds,
            effective_body_bounds_um=body_bounds,
            source_padding_um=envelope["source_padding_um"],
            effective_padding_um=padding,
        )
        entities[index] = replace(entity, metadata={**entity.metadata,
                                                    "aedt_effective_envelope": envelope})
        record = records.get(entity.metadata.get("source_semantic_id", entity.semantic_id))
        if record is not None:
            record["metadata"] = {**record.get("metadata", {}),
                                  "aedt_effective_envelope": envelope}

    trace = {**mapping.to_payload(), "source_entity_layers": bindings,
             "source_stack_sha256": canonical_sha256(detached(stack))}
    effective_stack["metadata"] = {**effective_stack.get("metadata", {}), "aedt_modeling": trace}
    return replace(build_input, entities=tuple(entities),
                   solution_regions=effective_stack["solution_regions"],
                   metadata={**build_input.metadata, "aedt_modeling": trace}), effective_stack, replace(context, enclosures=enclosures)
