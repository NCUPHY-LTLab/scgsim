"""The inputs that every surface-planning step of one route reads.

``plan_route_surfaces`` hands the same five facts to a dozen helpers. Bundling them
keeps each helper signature short. This module owns that bundle only; it holds no
planning logic.
"""

from __future__ import annotations

from dataclasses import dataclass

from scgsim.geometry.models.common import RouteLiteral
from scgsim.geometry.models.input import GeometryBuildInput
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.models.topology import InterfacePlanRecord
from scgsim.semantics import SemanticEvidenceFacade


@dataclass(frozen=True)
class SurfaceScope:
    """Read-only inputs shared by every step that plans surfaces for one route."""

    build_input: GeometryBuildInput
    # None means "plan from the prepared representation context instead of a route".
    route: RouteLiteral | None
    interfaces: tuple[InterfacePlanRecord, ...]
    semantic_facts: SemanticEvidenceFacade
    representation_context: RepresentationContext | None

    @property
    def context_planning(self) -> bool:
        """True when surfaces come from a representation context, not a named route."""
        return self.route is None and self.representation_context is not None
