"""Expose surface planning at its established geometry-planning import path.

Implementations live by responsibility under surface_planning; these exports
remain the same function objects used by route and contribution consumers.
"""

from scgsim.geometry.planning.surface_planning.contacts import (
    contact_patches_by_entity_face,
    subtract_contact_patches_from_face,
)
from scgsim.geometry.planning.surface_planning.contracts import (
    with_surface_contract_metadata,
)
from scgsim.geometry.planning.surface_planning.contributions import (
    plan_representation_contribution_geometry,
)
from scgsim.geometry.planning.surface_planning.partitions import plan_surface_partitions
from scgsim.geometry.planning.surface_planning.planner import plan_route_surfaces
from scgsim.geometry.planning.surface_planning.sidewalls import (
    ConductorSidewallRequest,
    conductor_sidewall_geometry_refs,
)

__all__ = [
    "plan_route_surfaces",
    "plan_representation_contribution_geometry",
    "plan_surface_partitions",
    "with_surface_contract_metadata",
    "contact_patches_by_entity_face",
    "subtract_contact_patches_from_face",
    "conductor_sidewall_geometry_refs",
    "ConductorSidewallRequest",
]
