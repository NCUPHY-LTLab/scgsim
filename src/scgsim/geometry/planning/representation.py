"""Read how each conductor is represented, from the prepared representation context.

A conductor is a ``sheet`` (zero thickness) or a ``solid`` (finite thickness) in the
context. This module maps that fact, plus the route, to the geometry kind the planner
uses: ``surface_sheet``, ``cutout_boundary_shell`` or ``material_volume``. It never reads
backend metadata.
"""

from __future__ import annotations

from dataclasses import replace

from scgsim.geometry.models.common import RouteLiteral
from scgsim.geometry.models.input import SemanticEntitySpec
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
    """Return the context record for an entity; a missing context is an error."""
    if context is None:
        raise ValueError(
            "current geometry planning requires an explicit physical representation context"
        )
    return context.entities.get(entity.semantic_id)


def is_sheet_entity(
    entity: SemanticEntitySpec,
    *,
    route: RouteLiteral | None,
    context: RepresentationContext | None,
) -> bool:
    """True when the context represents the entity as a zero-thickness sheet."""
    record = entity_representation(entity, route=route, context=context)
    return record is not None and record.representation == "sheet"


def is_solid_entity(
    entity: SemanticEntitySpec,
    *,
    route: RouteLiteral | None,
    context: RepresentationContext | None,
) -> bool:
    """True when the context represents the entity as a finite-thickness solid."""
    record = entity_representation(entity, route=route, context=context)
    return record is not None and record.representation == "solid"


def planned_conductor_representation(
    entity: SemanticEntitySpec,
    *,
    route: RouteLiteral | None,
    context: RepresentationContext | None,
) -> str | None:
    """Return the shared planner's geometry kind without backend metadata reads.

    A sheet is a ``surface_sheet``. A solid is a ``material_volume`` in Route C and a
    ``cutout_boundary_shell`` (a PEC shell cut out of its host) in Routes A and B.
    """
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
    """True when at least one entity in the context is represented as a sheet."""
    return context is not None and any(
        record.representation == "sheet" for record in context.entities.values()
    )


def project_native_route_representations(build_input, route, context):
    """Ephemeral route primitive choices projected only from the common plan."""
    if context is None:
        raise ValueError(
            "interpret_physical_geometry() and pass plan.context before native compilation"
        )
    entities = []
    for entity in build_input.entities:
        if entity.material_kind != "conductor":
            entities.append(entity)
            continue
        primitive = planned_conductor_representation(
            entity, route=route, context=context
        )
        if primitive is None:
            raise ValueError(
                f"common physical context lacks conductor {entity.semantic_id!r}"
            )
        entities.append(replace(entity, route_representations={route: primitive}))
    return replace(build_input, entities=tuple(entities))
