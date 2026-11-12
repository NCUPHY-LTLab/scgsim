"""Resolve physical-group owners of planned surfaces and Entities, preserving grouped ownership."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from scgsim.geometry.models.common import (
    HIGH_COUNT_LOCAL_CONDUCTOR_PART_ROLES,
)
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.topology import SurfacePlanRecord
from scgsim.geometry.planning.topology import surface_owner_ids
from scgsim.semantics import ownership


def surface_physical_owner_ids(surface: SurfacePlanRecord) -> tuple[str, ...]:
    """Resolve a surface's physical owners from its semantic ownership."""
    owner_ids = surface.metadata.get("physical_owner_semantic_ids")
    return ownership.surface_physical_owner_ids(owner_ids, surface_owner_ids(surface))


def physical_group_owner_ids(
    build_input: GeometryBuildInput,
    owner_ids: Sequence[str],
) -> tuple[str, ...]:
    """Resolve semantic owners into physical group identities."""
    entities_by_id = {entity.semantic_id: entity for entity in build_input.entities}
    resolved = {
        owner_id: entity_physical_group_id(entities_by_id[owner_id])
        for owner_id in owner_ids
        if owner_id in entities_by_id
    }
    return ownership.physical_group_owner_ids(
        owner_ids,
        resolved,
    )


def entity_physical_group_id(entity: SemanticEntitySpec) -> str:
    """Resolve an entity's explicit or shared computational physical group."""
    auto_group_id = entity.metadata.get("auto_vacuum_group_id")
    if isinstance(auto_group_id, str) and auto_group_id:
        return auto_group_id
    if entity.part_role not in HIGH_COUNT_LOCAL_CONDUCTOR_PART_ROLES:
        return entity.semantic_id
    for key in ("physical_group_name", "physical_group_id", "semantic_group_id"):
        value = entity.metadata.get(key)
        if isinstance(value, str) and value:
            return value
    # Plan v2 keeps each occurrence as an Entity. Only explicitly net-bound,
    # same-family local conductors share a computational physical group.
    occurrence = entity.metadata.get("source_occurrence_path")
    local_id = entity.metadata.get("source_local_entity_id")
    if (
        isinstance(occurrence, str)
        and occurrence
        and isinstance(local_id, str)
        and local_id
        and isinstance(entity.net_id, str)
        and entity.net_id
    ):
        parent = occurrence.rpartition("/")[0]
        key = (parent, local_id, entity.part_role, entity.material_id, entity.net_id)
        digest = hashlib.sha256(
            json.dumps(key, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        return f"{local_id}__{digest}"
    return entity.semantic_id
