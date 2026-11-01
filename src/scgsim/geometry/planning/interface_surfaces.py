"""Plan the surfaces owned by recognized interfaces.

``plan_interface_surfaces`` walks the interfaces in order and decides which rule applies
to each one. ``InterfaceSurfacePlanner`` holds what one interface needs from the route's
scope, so its steps do not pass the same arguments to each other.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from scgsim.geometry.models.construction import SurfacePartitionRecord
from scgsim.geometry.models.topology import (
    InterfacePlanRecord,
    MMContactRecord,
    SurfacePlanRecord,
)
from scgsim.geometry.planning.context.scope import SurfaceScope
from scgsim.geometry.planning.context.tags import resolve_physical_group_owner_ids
from scgsim.geometry.planning.domain.substrate_air import interface_surface_kinds
from scgsim.geometry.planning.sheets.surfaces import route_a_sheet_patch_surfaces
from scgsim.geometry.planning.solutions.adjacency import (
    route_a_sheet_boundary_volume_ids,
)
from scgsim.geometry.primitives.entities import (
    entity_by_id,
    geometry_ref_from_metadata,
    unique_ids,
)
from scgsim.geometry.primitives.surface_records import (
    RouteASheetPatch,
    is_route_a_sheet_interface,
)
from scgsim.semantics.ownership import interface_surface_owner_ids


def plan_interface_surfaces(
    scope: SurfaceScope,
    *,
    partitions_by_interface: Mapping[str, Sequence[SurfacePartitionRecord]],
    effective_sheet_surfaces: Mapping[str, Sequence[SurfacePlanRecord]],
    sheet_patches_by_interface: Mapping[str, Sequence[RouteASheetPatch]],
    sheet_contacts_by_face: Mapping[str, Sequence[MMContactRecord]],
) -> list[SurfacePlanRecord]:
    """Plan the surfaces owned by recognized interfaces, in interface order."""
    records: list[SurfacePlanRecord] = []
    for interface in scope.interfaces:
        planner = InterfaceSurfacePlanner(scope, interface)
        # Same-net conductor contacts are component provenance only.  Neither
        # Route A nor Route B may lower an internal MM face as solver geometry.
        if planner.is_hidden_contact:
            continue
        if interface.interface_id in effective_sheet_surfaces:
            records.extend(effective_sheet_surfaces[interface.interface_id])
            continue
        if planner.is_route_a_sheet:
            patches = tuple(sheet_patches_by_interface.get(interface.interface_id, ()))
            if not patches:
                raise ValueError(
                    f"{interface.interface_id} has no local Route A sheet patches"
                )
            records.extend(
                route_a_sheet_patch_surfaces(
                    scope,
                    interface=interface,
                    patches=patches,
                    sheet_contacts=sheet_contacts_by_face.get(
                        patches[0].sheet_entity_id, ()
                    ),
                )
            )
            continue
        records.extend(
            planner.plan(partitions_by_interface.get(interface.interface_id, ()))
        )
    return records


class InterfaceSurfacePlanner:
    """Plan one recognized interface as surfaces within one route's scope."""

    def __init__(self, scope: SurfaceScope, interface: InterfacePlanRecord) -> None:
        self.scope = scope
        self.interface = interface

    @property
    def is_route_a_sheet(self) -> bool:
        """True when this interface is the surface of a Route A sheet conductor."""
        return is_route_a_sheet_interface(
            self.scope.route,
            self.interface,
            representation_context=self.scope.representation_context,
        )

    @property
    def is_hidden_contact(self) -> bool:
        """True for a conductor contact that must not become solver geometry."""
        route = self.scope.route
        return (
            self.interface.recognition_rule == "coplanar_conductor_contact_patch"
            and (
                route in {"A", "B"}
                or (route is None and self.scope.representation_context is not None)
            )
            and bool(self.interface.metadata.get("hidden_solver_contact"))
        )

    def plan(
        self, partitions: Sequence[SurfacePartitionRecord]
    ) -> list[SurfacePlanRecord]:
        """Plan a single surface, or one child surface per partition."""
        surface_interface_id = self.surface_interface_id()
        kinds = interface_surface_kinds(
            self.scope.build_input,
            route=self.scope.route,
            interface=self.interface,
            representation_context=self.scope.representation_context,
        )
        owner_ids = self.owner_ids()
        boundary_volume_ids = self.boundary_volume_ids()
        metadata = {
            "interface_kinds": kinds,
            "owner_semantic_ids": owner_ids,
            "physical_owner_semantic_ids": resolve_physical_group_owner_ids(
                self.scope.build_input, owner_ids
            ),
            "boundary_volume_ids": boundary_volume_ids,
            "embedded_surface_sheet": self.is_route_a_sheet,
        }
        if partitions:
            return [
                self._partition_record(partition, surface_interface_id, metadata)
                for partition in partitions
            ]
        return [self._whole_record(surface_interface_id, metadata)]

    def surface_interface_id(self) -> str:
        """The interface id surfaces carry; a Route A sheet's id names its boundaries."""
        interface = self.interface
        if not self.is_route_a_sheet:
            return interface.interface_id
        build_input = self.scope.build_input
        entity = entity_by_id(build_input, interface.owner_semantic_ids[0])
        boundary_ids = route_a_sheet_boundary_volume_ids(build_input, entity)
        suffix = interface.interface_id.rsplit("__", 1)[-1]
        return (
            f"{interface.kind}__{entity.semantic_id}__"
            f"{'__'.join(boundary_ids)}__{suffix}"
        )

    def owner_ids(self) -> tuple[str, ...]:
        """Semantic owners of the surfaces, from metadata, the sheet, or the interface."""
        interface = self.interface
        sheet_owner_ids = None
        if self.is_route_a_sheet:
            build_input = self.scope.build_input
            entity = entity_by_id(build_input, interface.owner_semantic_ids[0])
            sheet_owner_ids = (
                entity.semantic_id,
                *route_a_sheet_boundary_volume_ids(build_input, entity),
            )
        return interface_surface_owner_ids(
            interface.metadata.get("surface_owner_semantic_ids"),
            route_a_sheet_owner_ids=sheet_owner_ids,
            fallback_owner_ids=interface.owner_semantic_ids,
        )

    def boundary_volume_ids(self) -> tuple[str, ...]:
        """Volumes the surfaces bound, from metadata, the sheet, or the interface."""
        interface = self.interface
        declared = interface.metadata.get("boundary_volume_ids")
        if declared is not None:
            if isinstance(declared, str):
                return (declared,)
            return unique_ids(declared)
        if self.is_route_a_sheet:
            build_input = self.scope.build_input
            return route_a_sheet_boundary_volume_ids(
                build_input,
                entity_by_id(build_input, interface.owner_semantic_ids[0]),
            )
        return unique_ids(interface.owner_semantic_ids)

    def _geometry_ref(self) -> dict[str, Any]:
        return {
            "from_interface_id": self.interface.interface_id,
            "source_polygon_ids": self.interface.source_polygon_ids,
            **geometry_ref_from_metadata(self.interface.metadata),
        }

    def _surface_role(self, suffix: str) -> str:
        if self.scope.context_planning:
            return f"representation_context_{suffix}"
        return f"{self.scope.route}_{suffix}"

    def _valid_routes(self) -> tuple[str, ...]:
        return () if self.scope.route is None else (self.scope.route,)

    def _partition_record(
        self,
        partition: SurfacePartitionRecord,
        surface_interface_id: str,
        metadata: Mapping[str, Any],
    ) -> SurfacePlanRecord:
        interface = self.interface
        return SurfacePlanRecord(
            surface_id=partition.child_surface_id,
            owner_semantic_id=interface.owner_semantic_ids[0],
            surface_role=self._surface_role("planned_interface_partition"),
            geometry_ref={
                **self._geometry_ref(),
                "partition_id": partition.partition_id,
                "parent_interface_id": partition.parent_interface_id,
                **geometry_ref_from_metadata(partition.metadata),
            },
            interface_id=surface_interface_id,
            parent_surface_id=f"SURF__{surface_interface_id}",
            partition_label=partition.label,
            solver_use=interface.solver_use or "solver_active",
            valid_routes=self._valid_routes(),
            # A fresh dict per record, so no two surfaces share mutable metadata.
            metadata=dict(metadata),
        )

    def _whole_record(
        self, surface_interface_id: str, metadata: Mapping[str, Any]
    ) -> SurfacePlanRecord:
        interface = self.interface
        return SurfacePlanRecord(
            surface_id=f"SURF__{surface_interface_id}",
            owner_semantic_id=interface.owner_semantic_ids[0],
            surface_role=self._surface_role("planned_interface"),
            geometry_ref=self._geometry_ref(),
            interface_id=surface_interface_id,
            solver_use=interface.solver_use or "solver_active",
            valid_routes=self._valid_routes(),
            metadata=dict(metadata),
        )
