"""Interpret a prepared input under the selected approximation profile.

Placement is already normalized by the source stage. The interpretation changes only
the declared representation and Z, and keeps the polygon sets, ownership, materials and
source contact evidence. It adds no contact graph that the source did not declare.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from scgsim.geometry.grid.grid import with_input_grid
from scgsim.geometry.interpretation.envelope import auto_vacuum_envelope_bounds
from scgsim.geometry.models.entities import requires_route_representation
from scgsim.geometry.models.physical import (
    PHYSICAL_LAYER_FIELD,
    ApproximationProfile,
    PhysicalLayerDeclaration,
    source_plain,
)
from scgsim.geometry.models.physical import (
    interpret_physical_layers as interpret_physical_layers,
)
from scgsim.geometry.source.intents import route_a_sheet_interfaces
from scgsim.semantics.route_a import geometry_z_range
from scgsim.semantics.snapshot import canonical_sha256


@with_input_grid
def interpret_physical_geometry(
    build_input: Any, stack: Mapping[str, Any], profile: ApproximationProfile
):
    """Return effective input/stack and the sole source-bound representation plan."""
    records = {item["semantic_id"]: item for item in stack["layers"]}
    records.update(stack["solution_regions"])
    declarations = []
    bindings = {}
    for entity in build_input.entities:
        record = records.get(
            entity.metadata.get("source_semantic_id", entity.semantic_id), {}
        )
        raw = entity.metadata.get(
            PHYSICAL_LAYER_FIELD, record.get("metadata", {}).get(PHYSICAL_LAYER_FIELD)
        )
        if raw is not None:
            declaration = PhysicalLayerDeclaration.from_payload(raw)
            declarations.append(declaration)
            bindings[entity.semantic_id] = declaration.physical_layer_id
        elif requires_route_representation(entity) and profile.name not in {
            "solid",
            "route_b",
        }:
            raise ValueError(
                f"source body {entity.semantic_id!r} lacks neutral physical declaration"
            )
    plan = interpret_physical_layers(
        declarations,
        profile,
        supporting_entities={
            item.semantic_id: {
                "geometry": item.geometry,
                "material_id": item.material_id,
                "material_kind": item.material_kind,
                "material_properties": stack["materials"][item.material_id],
            }
            for item in build_input.entities
            if item.semantic_id
            in {
                binding["supporting_entity_id"]
                for binding in profile.parameters.get("film_bindings", {}).values()
            }
        },
        source_sha256=canonical_sha256(
            {"input": source_plain(build_input), "stack": source_plain(stack)}
        ),
    )
    entities, facts = [], {}
    effective_stack = source_plain(stack)
    effective_records = {
        item["semantic_id"]: item for item in effective_stack["layers"]
    }
    effective_records.update(effective_stack["solution_regions"])
    for entity in build_input.entities:
        original_interval = geometry_z_range(entity.geometry, entity.semantic_id)
        mapped = plan.map_body(
            bindings.get(entity.semantic_id),
            original_interval,
            {**entity.geometry, "material_id": entity.material_id},
        )
        facts[entity.semantic_id] = mapped
        entities.append(
            replace(
                entity,
                geometry=mapped["effective_geometry"],
                metadata={**entity.metadata, "physical_representation": mapped},
            )
        )
        record = effective_records.get(
            entity.metadata.get("source_semantic_id", entity.semantic_id)
        )
        if record is not None:
            record["geometry"] = mapped["effective_geometry"]
            record["metadata"] = {
                **record.get("metadata", {}),
                "physical_representation": mapped,
            }
    plan = replace(plan, entities=facts)
    effective_input = replace(build_input, entities=tuple(entities))
    enclosures = {}
    for entity in entities:
        if not entity.metadata.get("is_auto_vacuum_region"):
            continue
        effective_body = auto_vacuum_envelope_bounds(
            effective_input, route=None, representation_context=plan.context
        )
        source_geometry = facts[entity.semantic_id]["source_geometry"]
        source_low, source_high = facts[entity.semantic_id]["source_interval_um"]
        source_bounds = {
            **source_geometry["domain_bounds_um"],
            "z_min_um": source_low,
            "z_max_um": source_high,
        }
        enclosures[entity.semantic_id] = plan.project_enclosure(
            effective_body_bounds_um=effective_body,
            source_padding_um=entity.metadata["vacuum_region_padding_um"],
            source_envelope_bounds_um=source_bounds,
        )
    plan = replace(plan, enclosures=enclosures)
    payload = plan.to_payload()
    effective_stack["metadata"] = {
        **effective_stack.get("metadata", {}),
        "physical_representation": payload,
    }
    effective_input = replace(
        effective_input,
        solution_regions=effective_stack["solution_regions"],
        metadata={**build_input.metadata, "physical_representation": payload},
    )
    # Junction lowering loads optional gdstk; declaration-only imports stay usable.
    from scgsim.geometry.junctions.derivation import derive_rectangular_junctions  # noqa: PLC0415 -- load optional gdstk only when interpreting executable geometry

    effective_input = derive_rectangular_junctions(effective_input)
    return (
        _with_context_sheet_interfaces(effective_input, plan.context),
        effective_stack,
        plan,
    )


def _with_context_sheet_interfaces(build_input: Any, context: Any) -> Any:
    """Bind each sheet interface to the profile's conductor representation.

    The source adapter runs before any approximation profile is selected, so
    its generated sheet intents cannot know which conductors this profile maps
    to sheets. The effective input regenerates them from the final polygons
    (including any junction-derived ends), whether or not a junction exists.
    """
    intents = dict(build_input.metadata.get("interface_intents_2d", {}))
    explicit = tuple(
        item
        for item in intents.get("interfaces", ())
        if item.get("intent_origin")
        not in {
            "generated_route_a_surface_sheet",
            "generated_representation_context_surface_sheet",
        }
    )
    generated = route_a_sheet_interfaces(
        build_input.entities,
        build_input.polygons,
        representation_context=context,
    )["interfaces"]
    intents["interfaces"] = (*explicit, *generated)
    return replace(
        build_input,
        metadata={**build_input.metadata, "interface_intents_2d": intents},
    )
