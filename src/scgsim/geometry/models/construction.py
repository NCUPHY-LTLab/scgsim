"""Records of a surface partition and of a temporary body that surface planning reads."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from scgsim.geometry.models.common import ConductorRepresentationLiteral, RouteLiteral


@dataclass(frozen=True)
class SurfacePartitionRecord:
    """Partition intent for a live child of a semantic or construction parent.

    ``parent_interface_id`` identifies a semantic interface or a geometric
    surface parent validated against the planner's pre-partition inventory.
    An exterior surface parent remains an exterior, not an invented interface.
    This record is not backend geometry or a physical group source. The child
    must be directly buildable; overlay masks are not supported.
    """

    partition_id: str
    parent_interface_id: str
    child_surface_id: str
    label: str
    valid_routes: tuple[RouteLiteral, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConstructionBodyPlanRecord:
    """Route A/B temporary body used to derive final cutout shell surfaces.

    This is not final solver geometry and must not receive a physical group.
    It records which semantic conductor blocks a host solution region and which
    final shell surfaces are expected to survive after the backend cut.
    """

    construction_body_id: str
    owner_semantic_id: str
    host_semantic_id: str
    representation: ConductorRepresentationLiteral
    geometry_ref: Mapping[str, Any]
    expected_surface_ids: tuple[str, ...] = ()
    valid_routes: tuple[RouteLiteral, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
