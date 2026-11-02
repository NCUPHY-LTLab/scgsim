"""Read explicit prepared geometry representation facts for shared planning."""

from __future__ import annotations

from dataclasses import replace

from scgsim.geometry.models.common import ROUTE_ALLOWED_REPRESENTATIONS, RouteLiteral
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.representation import (
    EntityRepresentation,
    RepresentationContext,
)
from scgsim.geometry.primitives.entity_validation import requires_route_representation


def entity_representation(
    entity: SemanticEntitySpec,
    *,
    route: RouteLiteral | None,
    context: RepresentationContext | None,
) -> EntityRepresentation | None:
    """Read an entity's prepared representation from the required context."""
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
    """Check whether the prepared entity is a sheet."""
    record = entity_representation(entity, route=route, context=context)
    return record is not None and record.representation == "sheet"


def is_solid_entity(
    entity: SemanticEntitySpec,
    *,
    route: RouteLiteral | None,
    context: RepresentationContext | None,
) -> bool:
    """Check whether the prepared entity is a solid."""
    record = entity_representation(entity, route=route, context=context)
    return record is not None and record.representation == "solid"


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


def validate_selected_route(
    build_input: GeometryBuildInput,
    route: RouteLiteral,
    *,
    representation_context=None,
) -> None:
    """Validate that every conductor has an explicit representation for route."""
    if representation_context is None:
        raise ValueError(
            "current compilation requires interpret_physical_geometry() and plan.context"
        )
    context: RepresentationContext = representation_context
    allowed = ROUTE_ALLOWED_REPRESENTATIONS.get(route)
    if allowed is None:
        raise ValueError(f"Unsupported route: {route}")

    errors: list[str] = []
    if route == "C" and build_input.port_sheet_regions:
        errors.append("Route C does not support Palace lumped-port sheets")
    for entity in build_input.entities:
        if not requires_route_representation(entity):
            continue
        representation = planned_conductor_representation(
            entity, route=route, context=context
        )
        if representation is None:
            errors.append(f"{entity.semantic_id} does not define Route {route}")
        elif representation not in allowed:
            errors.append(
                f"{entity.semantic_id} Route {route} representation "
                f"{representation!r} is not supported"
            )
        if route == "C" and entity.geometry.get("route_ab_fused_selector_mode"):
            errors.append(
                f"{entity.semantic_id} Route C cannot use route_ab_fused_selector_mode"
            )
    if errors:
        raise ValueError("Invalid selected route: " + "; ".join(errors))


def is_sheet_context(context: RepresentationContext | None) -> bool:
    """Check whether the context contains any prepared sheet entity."""
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
