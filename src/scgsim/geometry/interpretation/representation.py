"""Read the prepared representation of an entity from the physical representation context."""

from __future__ import annotations

from scgsim.geometry.models.common import RouteLiteral
from scgsim.geometry.models.input import SemanticEntitySpec
from scgsim.geometry.models.refusals import internal
from scgsim.geometry.models.representation import (
    EntityRepresentation,
    RepresentationContext,
)


def entity_representation(
    entity: SemanticEntitySpec,
    *,
    route: RouteLiteral | None,
    context: RepresentationContext | None,
) -> EntityRepresentation | None:
    """Read an entity's prepared representation from the required context."""
    if context is None:
        raise ValueError(
            internal(
                "geometry planning was asked to run without a physical representation context"
            )
        )
    return context.entities.get(entity.semantic_id)


def planned_conductor_representation(
    entity: SemanticEntitySpec,
    *,
    route: RouteLiteral | None,
    context: RepresentationContext | None,
) -> str | None:
    """Return the shared planner's geometry kind without backend metadata reads."""
    if entity.role == "solution_region":
        return "solution_domain"
    record = entity_representation(entity, route=route, context=context)
    if record is None:
        return None
    return (
        "surface_sheet"
        if record.representation == "sheet"
        else "material_volume"
        if route == "C"
        else "cutout_boundary_shell"
    )


def is_sheet_context(context: RepresentationContext | None) -> bool:
    """Check whether the context contains any prepared sheet entity."""
    return context is not None and any(
        record.representation == "sheet" for record in context.entities.values()
    )
