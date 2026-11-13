"""Sweep an automatic vacuum envelope into ordered complement components."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from itertools import pairwise
from math import isfinite
from types import ModuleType
from typing import Any

from scgsim.geometry.grid.booleans import boolean_polygon_region
from scgsim.geometry.grid.constants import TOPOLOGY_EPS_UM
from scgsim.geometry.grid.loops import (
    gdstk_surface_region,
    geometry_refs_from_gdstk_region,
    loop_signature,
)
from scgsim.geometry.grid.scope import UNGRIDDED
from scgsim.geometry.interpretation.envelope import (
    auto_vacuum_envelope_bounds,
    auto_vacuum_padding,
    auto_vacuum_solution_region,
    is_auto_vacuum_subtractor,
    loop_domain_bounds,
)
from scgsim.geometry.models.common import RouteLiteral
from scgsim.geometry.models.entities import entity_z_range_um
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.planning.domain_planning.occupancy import (
    auto_vacuum_component_contacts_subtractor,
    auto_vacuum_subtractor_xy_region,
    domain_bounds_loop,
    solution_entity_xy_region,
)
from scgsim.geometry.planning.spatial import same_coordinate_z


def prepare_auto_vacuum_solution_regions(
    build_input: GeometryBuildInput,
    route: RouteLiteral | None,
    *,
    representation_context: RepresentationContext | None = None,
) -> GeometryBuildInput:
    """Replace auto VACUUM_REGION with planner-side complement components."""
    auto_region = auto_vacuum_solution_region(build_input)
    if auto_region is None:
        return build_input
    if auto_region.material_kind != "vacuum":
        raise ValueError(
            "auto vacuum region must have vacuum material_kind before planning"
        )
    return _VacuumSweep(build_input, route, representation_context, auto_region).build()


@dataclass
class _VacuumSweep:
    """Carry one authored vacuum envelope and its ordered complement sweep."""

    build_input: GeometryBuildInput
    route: RouteLiteral | None
    representation_context: RepresentationContext | None
    auto_region: SemanticEntitySpec
    auto_metadata: dict[str, Any] = field(init=False)
    auto_padding: dict[str, float] = field(init=False)
    auto_bounds: dict[str, float] = field(init=False)
    envelope_loop: tuple[tuple[float, float], ...] = field(init=False)
    auto_z_min_um: float = field(init=False)
    auto_z_max_um: float = field(init=False)
    all_entities: tuple[SemanticEntitySpec, ...] = field(init=False)
    obstacle_entities: tuple[SemanticEntitySpec, ...] = field(init=False)
    z_slices: list[float] = field(init=False)
    gdstk: ModuleType = field(init=False)
    components: list[SemanticEntitySpec] = field(init=False)
    component_index: int = field(init=False)

    def build(self) -> GeometryBuildInput:
        """Prepare the envelope, sweep its complement and retain source order."""
        self.prepare_envelope()
        self.prepare_sweep()
        import gdstk  # noqa: PLC0415 -- defer the optional polygon engine until this operation reaches native geometry.

        self.gdstk = gdstk
        self.components: list[SemanticEntitySpec] = []
        self.component_index = 0
        for z_start, z_end in pairwise(self.z_slices):
            self.sweep_slice(z_start, z_end)
        if not self.components:
            raise ValueError(
                "auto VACUUM_REGION leaves no finite sweep vacuum geometry"
            )
        retained_entities = tuple(
            (
                entity
                for entity in self.all_entities
                if entity.semantic_id != self.auto_region.semantic_id
            )
        )
        return replace(
            self.build_input,
            entities=tuple(self.components) + tuple(retained_entities),
            solution_regions=self.build_input.solution_regions,
        )

    def prepare_envelope(self) -> None:
        """Resolve authored padding and bind the complete effective envelope."""
        self.auto_metadata = dict(self.auto_region.metadata)
        self.auto_padding = auto_vacuum_padding(
            self.auto_metadata, self.auto_region.semantic_id
        )
        enclosure = (
            self.representation_context.enclosures.get(self.auto_region.semantic_id)
            if self.representation_context is not None
            else None
        )
        # The prepared context owns complete effective bounds; source padding
        # remains provenance and is not mapped a second time.
        if self.route is None and enclosure is not None:
            self.auto_bounds = dict(enclosure.effective_bounds_um)
        else:
            self.auto_bounds = auto_vacuum_envelope_bounds(
                self.build_input,
                route=self.route,
                representation_context=self.representation_context,
            )
            self.auto_bounds = {
                "x_min_um": self.auto_bounds["x_min_um"]
                - self.auto_padding["x_minus_um"],
                "y_min_um": self.auto_bounds["y_min_um"]
                - self.auto_padding["y_minus_um"],
                "x_max_um": self.auto_bounds["x_max_um"]
                + self.auto_padding["x_plus_um"],
                "y_max_um": self.auto_bounds["y_max_um"]
                + self.auto_padding["y_plus_um"],
                "z_min_um": self.auto_bounds["z_min_um"]
                - self.auto_padding["z_minus_um"],
                "z_max_um": self.auto_bounds["z_max_um"]
                + self.auto_padding["z_plus_um"],
            }
        if not self.auto_bounds["x_min_um"] < self.auto_bounds["x_max_um"]:
            raise ValueError("auto VACUUM_REGION has non-positive padded x extent")
        if not self.auto_bounds["y_min_um"] < self.auto_bounds["y_max_um"]:
            raise ValueError("auto VACUUM_REGION has non-positive padded y extent")
        if not self.auto_bounds["z_min_um"] < self.auto_bounds["z_max_um"]:
            raise ValueError("auto VACUUM_REGION has non-positive padded z extent")
        self.auto_region = replace(
            self.auto_region,
            geometry={
                **dict(self.auto_region.geometry),
                "domain_bounds_um": {
                    "x_min_um": self.auto_bounds["x_min_um"],
                    "y_min_um": self.auto_bounds["y_min_um"],
                    "x_max_um": self.auto_bounds["x_max_um"],
                    "y_max_um": self.auto_bounds["y_max_um"],
                },
                "z_min_um": self.auto_bounds["z_min_um"],
                "z_max_um": self.auto_bounds["z_max_um"],
                "outer_loop": domain_bounds_loop(self.auto_bounds),
                "hole_loops": (),
            },
        )
        self.envelope_loop = domain_bounds_loop(self.auto_bounds)

    def prepare_sweep(self) -> None:
        """Collect ordered obstacles and validate their Z sweep events."""
        self.auto_z_min_um = float(self.auto_bounds["z_min_um"])
        self.auto_z_max_um = float(self.auto_bounds["z_max_um"])
        self.all_entities = tuple(self.build_input.entities)
        self.obstacle_entities = tuple(
            (
                entity
                for entity in self.all_entities
                if is_auto_vacuum_subtractor(
                    entity,
                    route=self.route,
                    representation_context=self.representation_context,
                )
                and entity.semantic_id != self.auto_region.semantic_id
                and (not bool(entity.metadata.get("is_auto_vacuum_region")))
            )
        )
        z_events = {self.auto_z_min_um, self.auto_z_max_um}
        for entity in self.obstacle_entities:
            entity_z_min_um, entity_z_max_um = entity_z_range_um(entity)
            if not (
                isfinite(entity_z_min_um)
                and isfinite(entity_z_max_um)
                and (entity_z_max_um > entity_z_min_um)
            ):
                raise ValueError(f"{entity.semantic_id} has non-positive z extent")
            z_events.update((entity_z_min_um, entity_z_max_um))
        self.z_slices = sorted(z_events)
        if len(self.z_slices) < 2:
            raise ValueError("auto VACUUM_REGION has no valid z sweep")

    def sweep_slice(self, z_start: float, z_end: float) -> None:
        """Subtract active obstacles from one in-envelope Z interval."""
        if z_end - z_start <= TOPOLOGY_EPS_UM:
            return None
        if (
            z_start < self.auto_z_min_um - TOPOLOGY_EPS_UM
            or z_end > self.auto_z_max_um + TOPOLOGY_EPS_UM
        ):
            return None
        base_region = solution_entity_xy_region(
            self.gdstk, self.auto_region, z_min_um=z_start, z_max_um=z_end
        )
        if not base_region:
            return None
        active_subtractor_region: tuple[Any, ...] = ()
        active_subtractors: list[SemanticEntitySpec] = []
        for entity in self.obstacle_entities:
            obstacle_z_min_um, obstacle_z_max_um = entity_z_range_um(entity)
            if (
                obstacle_z_min_um >= z_end - TOPOLOGY_EPS_UM
                or obstacle_z_max_um <= z_start + TOPOLOGY_EPS_UM
            ):
                continue
            subtractor_regions = auto_vacuum_subtractor_xy_region(
                self.gdstk, self.build_input, entity
            )
            if not subtractor_regions:
                continue
            active_subtractors.append(entity)
            if active_subtractor_region:
                active_subtractor_region = boolean_polygon_region(
                    UNGRIDDED,
                    self.gdstk,
                    active_subtractor_region,
                    subtractor_regions,
                    "or",
                )
            else:
                active_subtractor_region = subtractor_regions
        vacuum_region_refs = geometry_refs_from_gdstk_region(
            UNGRIDDED,
            {"outer_loop": self.envelope_loop},
            boolean_polygon_region(
                UNGRIDDED, self.gdstk, base_region, active_subtractor_region, "not"
            ),
        )
        if not vacuum_region_refs:
            return None
        for geometry_ref in sorted(
            vacuum_region_refs, key=lambda region: loop_signature(region["outer_loop"])
        ):
            self.append_component(geometry_ref, z_start, z_end, active_subtractors)

    def append_component(
        self,
        geometry_ref: Mapping[str, Any],
        z_start: float,
        z_end: float,
        active_subtractors: list[SemanticEntitySpec],
    ) -> None:
        """Append one complement component with exact source boundary ledgers."""
        component_region = gdstk_surface_region(UNGRIDDED, geometry_ref)
        component_subtractor_ids = tuple(
            sorted(
                (
                    entity.semantic_id
                    for entity in active_subtractors
                    if auto_vacuum_component_contacts_subtractor(
                        self.gdstk,
                        build_input=self.build_input,
                        component_geometry_ref=geometry_ref,
                        subtractor=entity,
                    )
                )
            )
        )
        boundary_entity_ids = {"bottom": [], "top": []}
        for entity in self.obstacle_entities:
            entity_z_min_um, entity_z_max_um = entity_z_range_um(entity)
            boundary_key: str | None = None
            if same_coordinate_z(entity_z_max_um, z_start):
                boundary_key = "bottom"
            elif same_coordinate_z(entity_z_min_um, z_end):
                boundary_key = "top"
            if boundary_key is None:
                continue
            entity_region = auto_vacuum_subtractor_xy_region(
                self.gdstk, self.build_input, entity
            )
            if boolean_polygon_region(
                UNGRIDDED, self.gdstk, component_region, entity_region, "and"
            ):
                boundary_entity_ids[boundary_key].append(entity.semantic_id)
        self.component_index += 1
        component_id = (
            self.auto_region.semantic_id
            if self.component_index == 1
            else f"{self.auto_region.semantic_id}__{self.component_index:04d}"
        )
        self.components.append(
            SemanticEntitySpec(
                semantic_id=component_id,
                role=self.auto_region.role,
                material_id=self.auto_region.material_id,
                material_kind=self.auto_region.material_kind,
                priority=self.auto_region.priority,
                geometry_kind=self.auto_region.geometry_kind,
                host_void_semantic_id=self.auto_region.host_void_semantic_id,
                route_representations=self.auto_region.route_representations,
                geometry={
                    "geometry_kind": self.auto_region.geometry.get(
                        "geometry_kind", self.auto_region.geometry_kind
                    ),
                    "outer_loop": geometry_ref["outer_loop"],
                    "hole_loops": geometry_ref.get("hole_loops", ()),
                    "z_min_um": float(z_start),
                    "z_max_um": float(z_end),
                    "domain_bounds_um": loop_domain_bounds(
                        geometry_ref["outer_loop"], geometry_ref.get("hole_loops", ())
                    ),
                    "from_soln_region": self.auto_region.semantic_id,
                },
                metadata={
                    **self.auto_metadata,
                    "is_auto_vacuum_region": True,
                    "auto_vacuum_group_id": self.auto_region.semantic_id,
                    "auto_vacuum_envelope_outer_loop": self.envelope_loop,
                    "auto_vacuum_envelope_bounds_um": dict(self.auto_bounds),
                    "auto_vacuum_component_index": self.component_index,
                    "auto_vacuum_z_range_um": (float(z_start), float(z_end)),
                    "auto_vacuum_subtracting_entity_ids": tuple(
                        component_subtractor_ids
                    ),
                    "auto_vacuum_boundary_entity_ids": {
                        boundary_key: tuple(sorted(set(entity_ids)))
                        for boundary_key, entity_ids in boundary_entity_ids.items()
                    },
                },
                polygon_ids=(),
                labels=(),
            )
        )
