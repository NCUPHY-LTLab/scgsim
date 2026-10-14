"""One neutral selected-profile interpretation before backend construction.

Placement is already source-normalized. This interpreter changes only declared
representation/Z, preserving polygon sets, ownership, materials and source
contact evidence. Specific missing contact graphs are not fabricated here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass, replace
from typing import Any

from scgsim.geometry.models.physical import (
    PHYSICAL_LAYER_FIELD,
    ApproximationProfile,
    PhysicalLayerDeclaration,
    PhysicalRepresentationPlan,
)
from scgsim.geometry._primitives.entity_validation import _requires_route_representation
from scgsim.semantics.route_a import geometry_z_range, same_z
from scgsim.semantics.snapshot import _plain, canonical_sha256


def _source_plain(value: Any) -> Any:
    if is_dataclass(value):
        return {
            item.name: _source_plain(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {key: _source_plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_source_plain(item) for item in value]
    return _plain(value)


def interpret_physical_layers(
    declarations: Sequence[PhysicalLayerDeclaration],
    profile: ApproximationProfile,
    *,
    supporting_entities: Mapping[str, Any] | None = None,
    source_sha256: str | None = None,
) -> PhysicalRepresentationPlan:
    """Interpret declared levels without requiring a qualified contact graph."""
    index = {}
    for item in declarations:
        if not isinstance(item, PhysicalLayerDeclaration):
            raise TypeError("physical declarations must be typed source records")
        previous = index.setdefault(item.physical_layer_id, item)
        if previous != item:
            raise ValueError(f"conflicting declaration for {item.physical_layer_id!r}")
    removed = []
    anchors = {}
    supporting_geometry = {}
    support_planes = {}
    parameters = profile.parameters
    if profile.name in {"solid", "thin_film", "route_b"}:
        if parameters:
            raise ValueError(
                f"profile {profile.name!r} V1 has no additional parameters"
            )
        if profile.name == "thin_film":
            removed = [
                (item.physical_layer_id, *item.source_interval_um)
                for item in index.values()
                if item.feature_kind == "stack_film"
            ]
            anchors = {
                item.physical_layer_id: item.attachment_z_um
                for item in index.values()
                if item.feature_kind in {"stack_film", "local_film"}
            }
    else:
        if set(parameters) != {"film_bindings"} or not isinstance(
            parameters["film_bindings"], Mapping
        ):
            raise ValueError("face profiles require explicit film_bindings")
        bindings = parameters["film_bindings"]
        if not bindings:
            raise ValueError("face profile requires declared film bindings")
        if profile.name == "metal_gap_equivalent" and len(bindings) != 2:
            raise ValueError(
                "metal_gap_equivalent requires its two declared opposing films"
            )
        if profile.name == "substrate_face" and len(bindings) not in {1, 2}:
            raise NotImplementedError(
                "substrate_face V1 supports its declared one- or two-face profile"
            )
        if (
            profile.name == "metal_gap_equivalent"
            and len({binding["supporting_entity_id"] for binding in bindings.values()})
            != 2
        ):
            raise ValueError(
                "opposing films require distinct explicit supporting Entities"
            )
        for physical_id, binding in bindings.items():
            item = index[physical_id]
            if item.feature_kind != "stack_film":
                raise ValueError("face profile binding must identify a stack film")
            if set(binding) != {"supporting_entity_id", "supporting_side"}:
                raise ValueError("film supporting Entity/side must be explicit")
            if (
                supporting_entities is None
                or binding["supporting_entity_id"] not in supporting_entities
            ):
                raise ValueError("film profile supporting Entity is unavailable")
            host = supporting_entities[binding["supporting_entity_id"]]
            required = {
                "geometry",
                "material_id",
                "material_kind",
                "material_properties",
            }
            if not isinstance(host, Mapping) or not required <= set(host):
                raise ValueError(
                    "supporting Entity needs original geometry and material identity/properties"
                )
            if host["material_kind"] != "dielectric":
                raise ValueError("film supporting Entity must be a declared dielectric")
            if not host["material_id"] or not isinstance(
                host["material_properties"], Mapping
            ):
                raise ValueError("supporting Entity material evidence is unavailable")
            geometry = host["geometry"]
            supporting_geometry[binding["supporting_entity_id"]] = _source_plain(host)
            bounds = geometry_z_range(geometry, binding["supporting_entity_id"])
            side = binding["supporting_side"]
            if side not in {"lower", "upper"}:
                raise ValueError("supporting side must be lower or upper")
            plane = bounds[0 if side == "lower" else 1]
            if not same_z(plane, item.attachment_z_um):
                raise ValueError("film attachment and explicit supporting face differ")
            support_planes[physical_id] = (plane, side)
            anchors[physical_id] = item.attachment_z_um
            if profile.name == "metal_gap_equivalent":
                removed.append((physical_id, *item.source_interval_um))
    removed.sort(key=lambda item: item[1])
    for lower, upper in zip(removed, removed[1:]):
        if lower[2] > upper[1]:
            raise ValueError("distinct physical collapse intervals overlap")
    if profile.name == "metal_gap_equivalent":
        if (
            index[removed[0][0]].source_contact_facts["layer_attachment_face"]
            != "lower"
            or index[removed[1][0]].source_contact_facts["layer_attachment_face"]
            != "upper"
        ):
            raise ValueError(
                "metal_gap_equivalent requires lower/upper opposing attachments"
            )
    if len(anchors) == 2 and profile.name in {"substrate_face", "metal_gap_equivalent"}:
        lower, upper = sorted(anchors, key=anchors.get)
        lower_host = parameters["film_bindings"][lower]["supporting_entity_id"]
        upper_host = parameters["film_bindings"][upper]["supporting_entity_id"]
        if (
            lower_host == upper_host
            or support_planes[lower][1] != "upper"
            or support_planes[upper][1] != "lower"
        ):
            raise ValueError(
                "two films require distinct opposing dielectric substrate faces"
            )
        if index[upper].source_interval_um[0] <= index[lower].source_interval_um[1]:
            raise ValueError(
                "opposing films require a positive original inter-film cavity"
            )
    plan = PhysicalRepresentationPlan(
        source_sha256=source_sha256
        or canonical_sha256([item.to_payload() for item in index.values()]),
        profile=profile,
        declarations=tuple(index.values()),
        collapse_intervals=tuple(removed),
        sheet_anchors_um={},
        supporting_entities=supporting_geometry,
    )
    return replace(
        plan, sheet_anchors_um={key: plan.map_z(z) for key, z in anchors.items()}
    )


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
        elif _requires_route_representation(entity) and profile.name not in {
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
            {"input": _source_plain(build_input), "stack": _source_plain(stack)}
        ),
    )
    entities, facts = [], {}
    effective_stack = _source_plain(stack)
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
        from scgsim.geometry.planning.domain import _auto_vacuum_envelope_bounds

        effective_body = _auto_vacuum_envelope_bounds(
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
    return (
        replace(
            effective_input,
            solution_regions=effective_stack["solution_regions"],
            metadata={**build_input.metadata, "physical_representation": payload},
        ),
        effective_stack,
        plan,
    )
