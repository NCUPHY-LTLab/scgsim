"""Carry the source and evidence shared by surface-planning stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scgsim.geometry.models.common import (
    RouteLiteral,
)
from scgsim.geometry.models.construction import (
    ConstructionBodyPlanRecord,
    SurfacePartitionRecord,
)
from scgsim.geometry.models.input import GeometryBuildInput
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.models.topology import (
    InterfacePlanRecord,
    MMContactRecord,
)
from scgsim.semantics import SemanticEvidenceFacade


@dataclass(frozen=True)
class SurfaceScope:
    """Share one request's source, evidence and representation through planning."""

    build_input: GeometryBuildInput
    route: RouteLiteral | None
    interfaces: tuple[InterfacePlanRecord, ...]
    surface_partitions: tuple[SurfacePartitionRecord, ...]
    construction_bodies: tuple[ConstructionBodyPlanRecord, ...]
    mm_contacts: tuple[MMContactRecord, ...]
    semantic_facts: SemanticEvidenceFacade
    representation_context: RepresentationContext | None
    conductor_component_ledger: tuple[dict[str, Any], ...] | None

    @property
    def context_planning(self) -> bool:
        """Identify planning driven by a route-neutral representation context."""
        return self.route is None and self.representation_context is not None
