"""Partition solution sidewalls at finite conductor edges and height changes."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from itertools import pairwise
from math import hypot
from typing import Any

from scgsim.geometry.models.common import RouteLiteral
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.planning.domain_planning.occupancy import (
    active_route_conductor_entities,
)
from scgsim.geometry.planning.representation import planned_conductor_representation
from scgsim.geometry.primitives.constants import TOPOLOGY_EPS_UM
from scgsim.geometry.primitives.entities import entity_z_range_um
from scgsim.geometry.primitives.entity_validation import (
    is_solution_entity,
    is_vacuum_solution_entity,
)
from scgsim.geometry.primitives.loops import edge_is_covered_by_loop_edge
from scgsim.geometry.primitives.planar_loops import clean_loop, ring_edges
from scgsim.geometry.primitives.spatial import (
    interpolate_2d,
    same_coordinate_z,
    segment_overlap_interval,
)


@dataclass(frozen=True)
class SolutionSidewallPlanner:
    """Carry solution bounds and route context through edge and height partitioning."""

    build_input: GeometryBuildInput
    _: KW_ONLY
    route: RouteLiteral | None
    solution: SemanticEntitySpec
    outer_loop: tuple[tuple[float, float], ...]
    z_min_um: float
    z_max_um: float
    representation_context: RepresentationContext | None = None

    def build(self) -> tuple[dict[str, Any], ...]:
        """Partition the complete solution sidewall into source-bound quads."""
        refs: list[dict[str, Any]] = []
        for ring_role, ring_loop in (
            ("outer", clean_loop(self.outer_loop)),
            *(
                (f"hole_{index:04d}", clean_loop(hole_loop))
                for index, hole_loop in enumerate(
                    self.solution.geometry.get("hole_loops", ())
                )
            ),
        ):
            for edge_index, (start, end) in enumerate(ring_edges(ring_loop)):
                parameters = self.edge_parameters(start=start, end=end)
                for first, second in pairwise(parameters):
                    segment_start = interpolate_2d(start, end, first)
                    segment_end = interpolate_2d(start, end, second)
                    if (
                        hypot(
                            segment_end[0] - segment_start[0],
                            segment_end[1] - segment_start[1],
                        )
                        <= TOPOLOGY_EPS_UM
                    ):
                        continue
                    edge_z_min_um = self.edge_z_min_um(
                        start=segment_start,
                        end=segment_end,
                        default_z_min_um=self.z_min_um,
                    )
                    edge_z_max_um = self.edge_z_max_um(
                        start=segment_start,
                        end=segment_end,
                        default_z_max_um=self.z_max_um,
                    )
                    if edge_z_max_um - edge_z_min_um <= TOPOLOGY_EPS_UM:
                        continue
                    refs.append(
                        {
                            "quad_points": (
                                (*segment_start, edge_z_min_um),
                                (*segment_end, edge_z_min_um),
                                (*segment_end, edge_z_max_um),
                                (*segment_start, edge_z_max_um),
                            ),
                            "sidewall_ring_role": ring_role,
                            "sidewall_edge_index": edge_index,
                        }
                    )
        return tuple(refs)

    def edge_parameters(
        self, *, start: tuple[float, float], end: tuple[float, float]
    ) -> tuple[float, ...]:
        """Split a vacuum exterior edge at exact finite-conductor endpoints."""
        if not (
            self.route == "B"
            or (self.route is None and self.representation_context is not None)
        ) or not is_vacuum_solution_entity(self.solution):
            return (0.0, 1.0)
        parameters = {0.0, 1.0}
        solution_z_min = float(self.solution.geometry["z_min_um"])
        solution_z_max = float(self.solution.geometry["z_max_um"])
        for entity in active_route_conductor_entities(
            self.build_input,
            self.route,
            representation_context=self.representation_context,
        ):
            if planned_conductor_representation(
                entity, route=self.route, context=self.representation_context
            ) not in {"cutout_boundary_shell", "material_volume"}:
                continue
            entity_z_min, entity_z_max = entity_z_range_um(entity)
            if not (
                same_coordinate_z(entity_z_min, solution_z_min)
                or same_coordinate_z(entity_z_max, solution_z_max)
            ):
                continue
            for edge_start, edge_end in ring_edges(
                clean_loop(entity.geometry["outer_loop"])
            ):
                interval = segment_overlap_interval(start, end, edge_start, edge_end)
                if interval is not None:
                    parameters.update(interval)
        return tuple(sorted(parameters))

    def edge_z_min_um(
        self,
        *,
        start: tuple[float, float],
        end: tuple[float, float],
        default_z_min_um: float,
    ) -> float:
        """Resolve the lower sidewall height at one canonical segment."""
        if not is_vacuum_solution_entity(self.solution):
            return default_z_min_um
        z_min_um = default_z_min_um
        for entity in active_route_conductor_entities(
            self.build_input,
            self.route,
            representation_context=self.representation_context,
        ):
            if (
                is_solution_entity(entity)
                or planned_conductor_representation(
                    entity, route=self.route, context=self.representation_context
                )
                not in {"cutout_boundary_shell", "material_volume"}
                or "outer_loop" not in entity.geometry
            ):
                continue
            entity_z_min_um, entity_z_max_um = entity_z_range_um(entity)
            if not same_coordinate_z(entity_z_min_um, default_z_min_um):
                continue
            if edge_is_covered_by_loop_edge(start, end, entity.geometry["outer_loop"]):
                z_min_um = max(z_min_um, entity_z_max_um)
        return z_min_um

    def edge_z_max_um(
        self,
        *,
        start: tuple[float, float],
        end: tuple[float, float],
        default_z_max_um: float,
    ) -> float:
        """Resolve the upper sidewall height at one canonical segment."""
        if not is_vacuum_solution_entity(self.solution):
            return default_z_max_um
        z_max_um = default_z_max_um
        for entity in active_route_conductor_entities(
            self.build_input,
            self.route,
            representation_context=self.representation_context,
        ):
            if (
                is_solution_entity(entity)
                or planned_conductor_representation(
                    entity, route=self.route, context=self.representation_context
                )
                not in {"cutout_boundary_shell", "material_volume"}
                or "outer_loop" not in entity.geometry
            ):
                continue
            entity_z_min_um, entity_z_max_um = entity_z_range_um(entity)
            if not same_coordinate_z(entity_z_max_um, default_z_max_um):
                continue
            if edge_is_covered_by_loop_edge(start, end, entity.geometry["outer_loop"]):
                z_max_um = min(z_max_um, entity_z_min_um)
        return z_max_um
