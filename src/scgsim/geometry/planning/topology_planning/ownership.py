"""Project canonical surface ownership and ordered boundary-volume identities."""

from __future__ import annotations

from scgsim.geometry.models.topology import SurfacePlanRecord
from scgsim.semantics.ownership import surface_declared_owner_ids


def surface_owner_ids(surface: SurfacePlanRecord) -> tuple[str, ...]:
    """Resolve the declared semantic owners of a surface."""
    return surface_declared_owner_ids(
        surface.metadata.get("owner_semantic_ids", (surface.owner_semantic_id,)),
        surface.owner_semantic_id,
    )
