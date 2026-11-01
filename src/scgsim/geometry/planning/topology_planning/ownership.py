"""Project canonical surface ownership and ordered boundary-volume identities."""

from __future__ import annotations

from scgsim.geometry.models.topology import SurfacePlanRecord
from scgsim.geometry.primitives.entities import unique_ids
from scgsim.semantics.ownership import (
    project_legacy_interface_record_owners,
    surface_declared_owner_ids,
)


def surface_interface_record_owners(
    surface: SurfacePlanRecord,
    owners: tuple[str, ...],
) -> tuple[str, ...]:
    """Project structured sheet ownership onto the legacy two-owner record."""
    return project_legacy_interface_record_owners(
        owners,
        surface.metadata.get("boundary_volume_ids", ()),
        surface_id=surface.surface_id,
    )


def structured_surface_boundary_volume_ids(
    surface: SurfacePlanRecord,
    owners: tuple[str, ...],
) -> tuple[str, ...]:
    """Read and validate ordered boundary-volume owners for a structured surface."""
    raw = surface.metadata.get("boundary_volume_ids", ())
    boundary_ids = () if isinstance(raw, str) else tuple(str(value) for value in raw)
    expected_count = 1 if surface.metadata.get("sheet_contact_cap") else 2
    if len(boundary_ids) != expected_count or not set(boundary_ids).issubset(owners):
        raise ValueError(
            f"{surface.surface_id} structured interface requires {expected_count} ordered "
            "boundary_volume_ids from its owners"
        )
    return unique_ids(boundary_ids)


def surface_owner_ids(surface: SurfacePlanRecord) -> tuple[str, ...]:
    """Resolve the declared semantic owners of a surface."""
    return surface_declared_owner_ids(
        surface.metadata.get("owner_semantic_ids", (surface.owner_semantic_id,)),
        surface.owner_semantic_id,
    )
