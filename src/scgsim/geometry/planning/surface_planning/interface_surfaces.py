"""Resolve ordinary interface ownership and emit explicit partition children."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from scgsim.geometry.models.common import (
    RouteLiteral,
)
from scgsim.geometry.models.construction import (
    SurfacePartitionRecord,
)
from scgsim.geometry.models.input import GeometryBuildInput
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.models.topology import (
    InterfacePlanRecord,
    SurfacePlanRecord,
)
from scgsim.geometry.planning.domain import (
    interface_surface_kinds,
    route_a_sheet_boundary_volume_ids,
)
from scgsim.geometry.planning.surface_planning.scope import SurfaceScope
from scgsim.geometry.planning.tags import (
    physical_group_owner_ids,
)
from scgsim.geometry.primitives.entities import (
    entity_by_id,
    geometry_ref_from_metadata,
    unique_ids,
)
from scgsim.geometry.primitives.surface_records import (
    is_route_a_sheet_interface,
)
from scgsim.semantics import EvidenceResult
from scgsim.semantics.ownership import interface_surface_owner_ids


class InterfaceSurfacePlanner:
    """Retain one interface's owner, domain and partition decisions together."""

    def __init__(self, scope: SurfaceScope, interface: InterfacePlanRecord):
        self.scope = scope
        self.interface = interface

    def plan(
        self, partitions_by_interface: Mapping[str, Sequence[SurfacePartitionRecord]]
    ) -> tuple[SurfacePlanRecord, ...]:
        """Create the interface itself or its explicitly requested live children."""
        records: list[SurfacePlanRecord] = []
        geometry_ref = {
            "from_interface_id": self.interface.interface_id,
            "source_polygon_ids": self.interface.source_polygon_ids,
            **geometry_ref_from_metadata(self.interface.metadata),
        }
        route_a_sheet_evidence: tuple[EvidenceResult, ...] = ()
        primary_kind = next(
            (
                result.classification
                for result in route_a_sheet_evidence
                if self.interface.owner_semantic_ids[1]
                in (result.effective_domain_ids or ())
            ),
            None,
        )
        surface_interface_id = _surface_interface_id(
            self.scope.build_input,
            route=self.scope.route,
            interface=self.interface,
            primary_kind=primary_kind,
            representation_context=self.scope.representation_context,
        )
        interface_kinds = interface_surface_kinds(
            self.scope.build_input,
            route=self.scope.route,
            interface=self.interface,
            route_a_evidence=route_a_sheet_evidence,
            representation_context=self.scope.representation_context,
        )
        owner_semantic_ids = _interface_surface_owner_ids(
            self.scope.build_input,
            route=self.scope.route,
            interface=self.interface,
            route_a_evidence=route_a_sheet_evidence,
            representation_context=self.scope.representation_context,
        )
        boundary_volume_ids = _interface_boundary_volume_ids(
            self.scope.build_input,
            route=self.scope.route,
            interface=self.interface,
            route_a_evidence=route_a_sheet_evidence,
            representation_context=self.scope.representation_context,
        )
        physical_owner_semantic_ids = physical_group_owner_ids(
            self.scope.build_input,
            owner_semantic_ids,
        )
        embedded_sheet = is_route_a_sheet_interface(
            self.scope.route,
            self.interface,
            representation_context=self.scope.representation_context,
        )
        partitions = partitions_by_interface.get(self.interface.interface_id, ())
        if partitions:
            parent_surface_id = f"SURF__{surface_interface_id}"
            for partition in partitions:
                child_geometry_ref = {
                    **geometry_ref,
                    "partition_id": partition.partition_id,
                    "parent_interface_id": partition.parent_interface_id,
                    **geometry_ref_from_metadata(partition.metadata),
                }
                records.append(
                    SurfacePlanRecord(
                        surface_id=partition.child_surface_id,
                        owner_semantic_id=self.interface.owner_semantic_ids[0],
                        surface_role=(
                            "representation_context_planned_interface_partition"
                            if self.scope.context_planning
                            else f"{self.scope.route}_planned_interface_partition"
                        ),
                        geometry_ref=child_geometry_ref,
                        interface_id=surface_interface_id,
                        parent_surface_id=parent_surface_id,
                        partition_label=partition.label,
                        solver_use=self.interface.solver_use or "solver_active",
                        valid_routes=()
                        if self.scope.route is None
                        else (self.scope.route,),
                        metadata={
                            "interface_kinds": interface_kinds,
                            "owner_semantic_ids": owner_semantic_ids,
                            "physical_owner_semantic_ids": physical_owner_semantic_ids,
                            "boundary_volume_ids": boundary_volume_ids,
                            "embedded_surface_sheet": embedded_sheet,
                        },
                    )
                )
            return tuple(records)
        records.append(
            SurfacePlanRecord(
                surface_id=f"SURF__{surface_interface_id}",
                owner_semantic_id=self.interface.owner_semantic_ids[0],
                surface_role=(
                    "representation_context_planned_interface"
                    if self.scope.context_planning
                    else f"{self.scope.route}_planned_interface"
                ),
                geometry_ref=geometry_ref,
                interface_id=surface_interface_id,
                solver_use=self.interface.solver_use or "solver_active",
                valid_routes=() if self.scope.route is None else (self.scope.route,),
                metadata={
                    "interface_kinds": interface_kinds,
                    "owner_semantic_ids": owner_semantic_ids,
                    "physical_owner_semantic_ids": physical_owner_semantic_ids,
                    "boundary_volume_ids": boundary_volume_ids,
                    "embedded_surface_sheet": embedded_sheet,
                },
            )
        )
        return tuple(records)


def _interface_surface_owner_ids(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral | None,
    interface: InterfacePlanRecord,
    route_a_evidence: tuple[EvidenceResult, ...] = (),
    representation_context: RepresentationContext | None = None,
) -> tuple[str, ...]:
    raw_owner_ids = interface.metadata.get("surface_owner_semantic_ids")
    route_a_owner_ids = None
    if is_route_a_sheet_interface(
        route, interface, representation_context=representation_context
    ):
        route_a_owner_ids = unique_ids(
            owner_id
            for result in route_a_evidence
            for owner_id in (
                *result.source_owner_ids,
                *(result.effective_domain_ids or ()),
            )
        )
        if not route_a_owner_ids:
            entity = entity_by_id(build_input, interface.owner_semantic_ids[0])
            route_a_owner_ids = (
                entity.semantic_id,
                *route_a_sheet_boundary_volume_ids(build_input, entity),
            )
    return interface_surface_owner_ids(
        raw_owner_ids,
        route_a_sheet_owner_ids=route_a_owner_ids,
        fallback_owner_ids=interface.owner_semantic_ids,
    )


def _interface_boundary_volume_ids(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral | None,
    interface: InterfacePlanRecord,
    route_a_evidence: tuple[EvidenceResult, ...] = (),
    representation_context: RepresentationContext | None = None,
) -> tuple[str, ...]:
    raw_boundary_ids = interface.metadata.get("boundary_volume_ids")
    if raw_boundary_ids is not None:
        if isinstance(raw_boundary_ids, str):
            return (raw_boundary_ids,)
        return unique_ids(raw_boundary_ids)
    if is_route_a_sheet_interface(
        route, interface, representation_context=representation_context
    ):
        if route_a_evidence:
            return unique_ids(
                domain_id
                for result in route_a_evidence
                for domain_id in (result.effective_domain_ids or ())
            )
        return route_a_sheet_boundary_volume_ids(
            build_input,
            entity_by_id(build_input, interface.owner_semantic_ids[0]),
        )
    return unique_ids(interface.owner_semantic_ids)


def _surface_interface_id(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral | None,
    interface: InterfacePlanRecord,
    primary_kind: str | None = None,
    representation_context: RepresentationContext | None = None,
) -> str:
    if not is_route_a_sheet_interface(
        route, interface, representation_context=representation_context
    ):
        return interface.interface_id
    entity = entity_by_id(build_input, interface.owner_semantic_ids[0])
    boundary_ids = route_a_sheet_boundary_volume_ids(build_input, entity)
    suffix = interface.interface_id.rsplit("__", 1)[-1]
    return (
        f"{primary_kind or interface.kind}__{entity.semantic_id}__"
        f"{'__'.join(boundary_ids)}__{suffix}"
    )


def is_hidden_contact_interface(
    route: RouteLiteral | None,
    interface: InterfacePlanRecord,
    *,
    representation_context: RepresentationContext | None = None,
) -> bool:
    """Identify same-Net contact evidence that owns no solver-live face."""
    return (
        interface.recognition_rule == "coplanar_conductor_contact_patch"
        and (
            route in {"A", "B"}
            or (route is None and representation_context is not None)
        )
        and bool(interface.metadata.get("hidden_solver_contact"))
    )
