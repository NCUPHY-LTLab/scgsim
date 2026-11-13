"""Orchestrate ordered route surface planning from source and semantic evidence."""

from __future__ import annotations

from typing import Any, cast

from scgsim.geometry.interpretation.representation import is_sheet_context
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
    SurfacePlanRecord,
)
from scgsim.geometry.planning.domain_planning.boundaries import (
    plan_substrate_air_surfaces,
)
from scgsim.geometry.planning.evidence import (
    require_semantic_evidence_facade,
)
from scgsim.geometry.planning.surface_planning.conductors import ConductorSurfacePlanner
from scgsim.geometry.planning.surface_planning.contacts import (
    contact_patches_by_entity_face,
    route_a_sheet_contacts_by_face_metal,
    sheet_contact_loops_by_face_metal,
)
from scgsim.geometry.planning.surface_planning.contracts import (
    with_surface_contract_metadata,
)
from scgsim.geometry.planning.surface_planning.effective_sheets import (
    effective_contact_sheet_surfaces,
    effective_sheet_contact_sides,
)
from scgsim.geometry.planning.surface_planning.interface_surfaces import (
    InterfaceSurfacePlanner,
    is_hidden_contact_interface,
)
from scgsim.geometry.planning.surface_planning.scope import SurfaceScope
from scgsim.geometry.planning.surface_planning.sheets import (
    SheetSurfacePlanner,
    plan_route_a_sheet_patches,
)
from scgsim.geometry.planning.surface_records import (
    RouteASheetPatch,
    is_route_a_sheet_interface,
)
from scgsim.semantics import SemanticEvidenceFacade


def plan_route_surfaces(  # noqa: PLR0913 - preserve the documented route-planning request signature
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral | None,
    interfaces: tuple[InterfacePlanRecord, ...],
    surface_partitions: tuple[SurfacePartitionRecord, ...],
    construction_bodies: tuple[ConstructionBodyPlanRecord, ...],
    mm_contacts: tuple[MMContactRecord, ...] = (),
    semantic_facts: SemanticEvidenceFacade | None = None,
    representation_context: RepresentationContext | None = None,
    _conductor_component_ledger: tuple[dict[str, Any], ...] | None = None,
) -> tuple[SurfacePlanRecord, ...]:
    """Plan route-specific surfaces without building geometry.

    Route A should plan thin sheet interfaces and PEC shell/contact surfaces. Route B
    should plan host-owned cutout shell surfaces. Route C should plan retained
    material top/bottom/sidewall/contact surfaces.

    Partitioned interfaces must already be represented as child live surfaces
    in this stage. The backend should only receive surfaces it can build
    directly from point/curve/loop metadata.

    Route A `surface_sheet` conductors are not standalone surfaces here. They
    must appear as interface-owned `MS`, `MA`, `MM`, or `SA` surfaces, so metal
    coverage replaces the bare substrate-air interface instead of overlapping it.
    """
    semantic_facts = require_semantic_evidence_facade(
        build_input,
        route=route,
        facade=semantic_facts,
        representation_context=representation_context,
    )
    scope = SurfaceScope(
        build_input,
        route,
        interfaces,
        surface_partitions,
        construction_bodies,
        mm_contacts,
        semantic_facts,
        representation_context,
        _conductor_component_ledger,
    )
    return _RouteSurfacePlanner(scope).build()


class _RouteSurfacePlanner:
    """Keep the indexes and ordered records shared by one planning request."""

    def __init__(self, scope: SurfaceScope):
        self.scope = scope

    def build(self) -> tuple[SurfacePlanRecord, ...]:
        """Plan masks, domain surfaces, interfaces and finite conductors in order."""
        self.index_partitions()
        self.plan_effective_sheets()
        self.plan_sheet_patches()
        self.plan_substrate_air()
        self.index_contacts()
        self.plan_interfaces()
        self.plan_conductors()
        return self.finish()

    def index_partitions(self):
        """Retain explicit child partitions in caller order."""
        self.partitions_by_interface: dict[str, list[SurfacePartitionRecord]] = {}
        for partition in self.scope.surface_partitions:
            self.partitions_by_interface.setdefault(
                partition.parent_interface_id,
                [],
            ).append(partition)

    def plan_effective_sheets(self):
        """Resolve mixed contact exposure before the common sheet masks."""
        # Context planning establishes a non-None representation context.
        context = cast(RepresentationContext, self.scope.representation_context)
        self.effective_contacts = (
            effective_sheet_contact_sides(
                self.scope.build_input,
                self.scope.mm_contacts,
                representation_context=context,
            )
            if self.scope.context_planning
            else {}
        )
        self.effective_sheet_surfaces = {}
        self.effective_mask_patches = []
        for interface in self.scope.interfaces:
            if (
                is_route_a_sheet_interface(
                    self.scope.route,
                    interface,
                    representation_context=context,
                )
                and interface.owner_semantic_ids[0] in self.effective_contacts
            ):
                patches, surfaces = effective_contact_sheet_surfaces(
                    self.scope.build_input,
                    interface=interface,
                    contact_sides=self.effective_contacts[
                        interface.owner_semantic_ids[0]
                    ],
                    semantic_facts=self.scope.semantic_facts,
                    representation_context=context,
                )
                self.effective_mask_patches.extend(patches)
                self.effective_sheet_surfaces[interface.interface_id] = surfaces

    def plan_sheet_patches(self):
        """Partition remaining sheets into ordered local domain pairs."""
        self.route_a_sheet_patches = (
            plan_route_a_sheet_patches(
                self.scope.build_input,
                interfaces=tuple(
                    interface
                    for interface in self.scope.interfaces
                    if interface.interface_id not in self.effective_sheet_surfaces
                ),
                semantic_facts=self.scope.semantic_facts,
                route=self.scope.route,
                representation_context=self.scope.representation_context,
            )
            if self.scope.route == "A"
            or (
                self.scope.context_planning
                and is_sheet_context(self.scope.representation_context)
            )
            else ()
        )
        self.sheet_patches_by_interface: dict[str, list[RouteASheetPatch]] = {}
        for patch in self.route_a_sheet_patches:
            self.sheet_patches_by_interface.setdefault(
                patch.parent_interface_id, []
            ).append(patch)

    def plan_substrate_air(self):
        """Subtract the already planned sheets from substrate-air coverage."""
        self.records: list[SurfacePlanRecord] = []
        self.records.extend(
            plan_substrate_air_surfaces(
                self.scope.build_input,
                route=self.scope.route,
                semantic_facts=self.scope.semantic_facts,
                route_a_sheet_patches=(
                    *self.route_a_sheet_patches,
                    *self.effective_mask_patches,
                ),
                representation_context=self.scope.representation_context,
            )
        )

    def index_contacts(self):
        """Index sheet contacts before emitting any live interface records."""
        self.contact_faces = contact_patches_by_entity_face(self.scope.interfaces)
        self.sheet_contacts_by_face = (
            route_a_sheet_contacts_by_face_metal(
                self.scope.build_input, self.scope.mm_contacts
            )
            if self.scope.route == "A" or self.scope.context_planning
            else {}
        )
        self.normalized_sheet_loops = sheet_contact_loops_by_face_metal(
            self.scope.build_input,
            self.scope.mm_contacts,
            include_direct_bump_contacts=False,
        )

    def plan_interfaces(self):
        """Resolve hidden contacts, effective sheets and ordinary interfaces."""
        for interface in self.scope.interfaces:
            # Same-net conductor contacts are component provenance only.  Neither
            # Route A nor Route B may lower an internal MM face as solver geometry.
            if is_hidden_contact_interface(
                self.scope.route,
                interface,
                representation_context=self.scope.representation_context,
            ):
                continue
            if interface.interface_id in self.effective_sheet_surfaces:
                self.records.extend(
                    self.effective_sheet_surfaces[interface.interface_id]
                )
                continue
            if is_route_a_sheet_interface(
                self.scope.route,
                interface,
                representation_context=self.scope.representation_context,
            ):
                patches = tuple(
                    self.sheet_patches_by_interface.get(interface.interface_id, ())
                )
                if not patches:
                    raise ValueError(
                        f"{interface.interface_id} has no local Route A sheet patches"
                    )
                self.records.extend(
                    SheetSurfacePlanner(
                        self.scope,
                        interface,
                        patches,
                        self.sheet_contacts_by_face.get(patches[0].sheet_entity_id, ()),
                    ).plan()
                )
                continue

            self.records.extend(
                InterfaceSurfacePlanner(self.scope, interface).plan(
                    self.partitions_by_interface
                )
            )

    def plan_conductors(self):
        """Append finite conductor caps and walls after all interfaces."""
        self.records.extend(
            ConductorSurfacePlanner(
                self.scope, self.contact_faces, self.normalized_sheet_loops
            ).plan()
        )

    def finish(self):
        """Attach final ownership contracts after all route surfaces exist."""
        if self.scope.route == "C":
            return tuple(self.records)
        return with_surface_contract_metadata(
            self.scope.build_input,
            route=self.scope.route,
            interfaces=self.scope.interfaces,
            mm_contacts=self.scope.mm_contacts,
            surfaces=tuple(self.records),
            _conductor_component_ledger=self.scope.conductor_component_ledger,
        )
