"""Reads of a semantic entity that more than one stage makes: its vertical range and whether it is a solution region or needs a route representation."""

from __future__ import annotations

from math import isfinite

from scgsim.geometry.models.input import SemanticEntitySpec


def entity_z_range_um(entity: SemanticEntitySpec) -> tuple[float, float]:
    """Resolve an entity's lower and upper Z coordinates in micrometres."""
    z_min_um = float(entity.geometry.get("z_min_um", entity.geometry.get("z_um", 0.0)))
    z_max_um = entity.geometry.get("z_max_um")
    if z_max_um is not None:
        z_max_um = float(z_max_um)
        if not isfinite(z_max_um):
            raise ValueError(f"{entity.semantic_id} has non-finite z_max_um")
        if z_max_um > z_min_um:
            return z_min_um, z_max_um

    thickness_um = float(entity.geometry.get("thickness_um", 0.0))
    return z_min_um, z_min_um + thickness_um


def is_solution_entity(entity: SemanticEntitySpec) -> bool:
    """Return whether an entity owns a vacuum or dielectric solution region."""
    return entity.role == "solution_region" and entity.material_kind in {
        "vacuum",
        "dielectric",
    }


def is_vacuum_solution_entity(entity: SemanticEntitySpec) -> bool:
    """Return whether an entity is a vacuum solution region."""
    return is_solution_entity(entity) and entity.material_kind == "vacuum"


def requires_route_representation(entity: SemanticEntitySpec) -> bool:
    """Return whether the entity's conductor requires route representation."""
    if is_solution_entity(entity):
        return False
    return entity.material_kind == "conductor"
