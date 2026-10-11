"""Private accessors for explicit prepared geometry representation facts."""

from __future__ import annotations

from collections.abc import Mapping

from scgsim.geometry.models.input import SemanticEntitySpec
from scgsim.geometry.models.representation import (
    EntityRepresentation,
    RepresentationContext,
)
from scgsim.geometry.models.common import RouteLiteral


def entity_representation(
    entity: SemanticEntitySpec,
    *,
    route: RouteLiteral | None,
    context: RepresentationContext | None,
) -> EntityRepresentation | None:
    if context is not None:
        return context.entities.get(entity.semantic_id)
    if route is None:
        return None
    representation = entity.route_representations.get(route)
    if representation is None:
        return None
    if representation == "surface_sheet":
        return EntityRepresentation(
            physical_layer_id=None,
            representation="sheet",
            source_z_interval_um=_z_interval(entity.geometry),
            effective_z_interval_um=_z_interval(entity.geometry),
        )
    if representation in {"cutout_boundary_shell", "material_volume"}:
        return EntityRepresentation(
            physical_layer_id=None,
            representation="solid",
            source_z_interval_um=_z_interval(entity.geometry),
            effective_z_interval_um=_z_interval(entity.geometry),
        )
    return None


def is_sheet_entity(
    entity: SemanticEntitySpec,
    *,
    route: RouteLiteral | None,
    context: RepresentationContext | None,
) -> bool:
    record = entity_representation(entity, route=route, context=context)
    return record is not None and record.representation == "sheet"


def is_solid_entity(
    entity: SemanticEntitySpec,
    *,
    route: RouteLiteral | None,
    context: RepresentationContext | None,
) -> bool:
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
    if context is None:
        return entity.route_representations.get(route) if route is not None else None
    record = entity_representation(entity, route=route, context=context)
    if record is None:
        return None
    return (
        "surface_sheet" if record.representation == "sheet" else "cutout_boundary_shell"
    )


def is_sheet_context(context: RepresentationContext | None) -> bool:
    return context is not None and any(
        record.representation == "sheet" for record in context.entities.values()
    )


def _z_interval(geometry: Mapping[str, object]) -> tuple[float, float]:
    z_min = geometry.get("z_min_um")
    z_max = geometry.get("z_max_um")
    if z_min is None:
        z_min = geometry.get("z_um", 0.0)
    if z_max is None:
        z_max = float(z_min) + float(geometry.get("thickness_um", 0.0))
    return float(z_min), float(z_max)
