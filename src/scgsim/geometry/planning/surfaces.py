"""Plan every surface of a route, in a fixed order, from interfaces and conductors.

``plan_route_surfaces`` is the entry point. It builds, in order: Route A sheet surfaces,
the substrate-air and domain boundary surfaces, one surface per recognized interface,
and the top, bottom and sidewall surfaces of shell or volume conductors, then attaches
the surface contract metadata. The steps live in ``sheets``, ``domain``,
``interface_surfaces``, ``conductors.shell_surfaces`` and ``surface_contract``.

This module also owns the non-gmsh seam for physical contribution planning
(``plan_surface_contribution_patches``). Mixed sheet and solid contacts exclude their
typed metal footprints before solution coverage is resolved. Finite walls bind authored
dielectric cavity boundaries or ledger-bound vacuum children; Palace route policies
remain caller-owned.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from scgsim.geometry.models.common import RouteLiteral
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
from scgsim.geometry.planning.conductors.contact_patches import (
    plan_conductor_contact_patches,
)
from scgsim.geometry.planning.conductors.face_contacts import (
    contact_patches_by_entity_face,
    route_a_sheet_contacts_by_face_metal,
    sheet_contact_loops_by_face_metal,
)
from scgsim.geometry.planning.conductors.mm_contacts import (
    plan_mm_contact_records,
    plan_mm_contact_records_with_components,
)
from scgsim.geometry.planning.conductors.shell_surfaces import (
    plan_conductor_shell_surfaces,
)
from scgsim.geometry.planning.context.evidence import (
    build_semantic_evidence_facade,
    require_semantic_evidence_facade,
)
from scgsim.geometry.planning.context.representation import (
    is_sheet_context,
    project_native_route_representations,
)
from scgsim.geometry.planning.context.scope import SurfaceScope
from scgsim.geometry.planning.domain.boundaries import (
    merge_solution_sidewall_interfaces,
    reconcile_solution_domain_boundaries,
)
from scgsim.geometry.planning.domain.substrate_air import (
    plan_substrate_air_surfaces,
)
from scgsim.geometry.planning.interface_surfaces import plan_interface_surfaces
from scgsim.geometry.planning.interfaces import recognize_route_interfaces
from scgsim.geometry.planning.sheets.effective_contacts import (
    effective_sheet_contact_sides,
    plan_effective_contact_sheets,
    refresh_generated_context_sheet_interfaces,
    refresh_generated_route_a_sheet_interfaces,
)
from scgsim.geometry.planning.sheets.patches import plan_route_a_sheet_patches
from scgsim.geometry.planning.solutions.auto_vacuum import (
    prepare_auto_vacuum_solution_regions,
)
from scgsim.geometry.planning.surface_contract import with_surface_contract_metadata
from scgsim.geometry.planning.validation import validate_selected_route
from scgsim.geometry.primitives.surface_records import (
    RouteASheetPatch,
)
from scgsim.semantics import SemanticEvidenceFacade


def plan_surface_contribution_patches(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    representation_context: RepresentationContext | None = None,
) -> tuple[GeometryBuildInput, tuple[SurfacePlanRecord, ...]]:
    """Derive local physical-side evidence without a backend construction plan.

    This non-Gmsh seam reuses the polygon Boolean and Semantic Core
    classifiers, but does not plan cutter bodies, canonical mesh topology,
    port lowering, volumes, tags, or backend identifiers.
    """

    if representation_context is None:
        raise ValueError(
            "interpret_physical_geometry() and pass plan.context for current contribution planning"
        )

    build_input = project_native_route_representations(
        build_input, route, representation_context
    )
    validate_selected_route(
        build_input, route, representation_context=representation_context
    )
    return _plan_surface_contribution_geometry(
        build_input, route=route, representation_context=representation_context
    )


def _plan_surface_contribution_geometry(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral | None,
    representation_context: RepresentationContext | None = None,
) -> tuple[GeometryBuildInput, tuple[SurfacePlanRecord, ...]]:
    """Shared geometric contribution primitive with an explicit geometry authority."""
    if route is None or representation_context is not None:
        if representation_context is None:
            raise ValueError(
                "route-neutral contribution planning requires representation context"
            )
        build_input = prepare_auto_vacuum_solution_regions(
            build_input, route=route, representation_context=representation_context
        )
        build_input = refresh_generated_context_sheet_interfaces(
            build_input, representation_context=representation_context
        )
    elif route == "A":
        build_input = refresh_generated_route_a_sheet_interfaces(build_input)
    semantic_facts = build_semantic_evidence_facade(
        build_input,
        route=route,
        representation_context=representation_context,
    )
    interfaces = recognize_route_interfaces(
        build_input, route=route, representation_context=representation_context
    )
    interfaces = plan_conductor_contact_patches(
        build_input,
        route=route,
        interfaces=interfaces,
        representation_context=representation_context,
    )
    component_ledger = None
    if route is None and representation_context is not None:
        interfaces, mm_contacts, component_ledger = (
            plan_mm_contact_records_with_components(
                build_input,
                route=route,
                interfaces=interfaces,
                representation_context=representation_context,
            )
        )
    elif route in {"A", "B"}:
        interfaces, mm_contacts = plan_mm_contact_records(
            build_input,
            route=route,
            interfaces=interfaces,
            representation_context=representation_context,
        )
    else:
        mm_contacts = ()
    surfaces = plan_route_surfaces(
        build_input,
        route=route,
        interfaces=interfaces,
        surface_partitions=plan_surface_partitions(
            build_input, route=route, interfaces=interfaces
        ),
        construction_bodies=(),
        mm_contacts=mm_contacts,
        semantic_facts=semantic_facts,
        representation_context=representation_context,
        _conductor_component_ledger=component_ledger,
    )
    surfaces = reconcile_solution_domain_boundaries(build_input, surfaces=surfaces)
    return build_input, merge_solution_sidewall_interfaces(
        build_input,
        surfaces=surfaces,
        semantic_facts=semantic_facts,
    )


def plan_representation_contribution_geometry(
    build_input: GeometryBuildInput,
    representation_context: RepresentationContext,
) -> tuple[GeometryBuildInput, tuple[SurfacePlanRecord, ...]]:
    """Plan shared contribution surfaces from an explicit neutral context."""
    return _plan_surface_contribution_geometry(
        build_input,
        route=None,
        representation_context=representation_context,
    )


def plan_surface_partitions(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral | None,
    interfaces: tuple[InterfacePlanRecord, ...],
) -> tuple[SurfacePartitionRecord, ...]:
    """Plan parent-interface partitions before live surfaces are created.

    This function only consumes explicit `build_input.metadata["surface_partitions"]`
    records.

    Child regions are partition intent, not backend geometry. Each returned
    `SurfacePartitionRecord` must point to a child `SurfacePlanRecord` that
    will be created directly by the backend. The parent interface itself is a
    semantic aggregate and must not be cut by OCC after creation.
    """
    interface_ids = {interface.interface_id for interface in interfaces}
    raw_partitions = build_input.metadata.get("surface_partitions", ())
    if raw_partitions in (None, ()):
        return ()
    if not isinstance(raw_partitions, tuple | list):
        raise TypeError("surface_partitions metadata must be a sequence")

    records: list[SurfacePartitionRecord] = []
    for index, intent in enumerate(raw_partitions):
        if not isinstance(intent, Mapping):
            raise TypeError("surface_partitions entries must be mappings")
        if route is None:
            if intent.get("valid_routes"):
                continue
            valid_routes: tuple[RouteLiteral, ...] = ()
        else:
            valid_routes = tuple(
                str(value) for value in intent.get("valid_routes", (route,))
            )
            if route not in valid_routes:
                continue
        parent_interface_id = str(intent.get("parent_interface_id", ""))
        if parent_interface_id not in interface_ids:
            raise ValueError(
                f"surface partition references unknown interface "
                f"{parent_interface_id!r}"
            )
        label = str(intent.get("label", "")).strip()
        if not label:
            raise ValueError("surface partition label must be non-empty")
        child_surface_id = str(
            intent.get("child_surface_id") or f"SURF__{parent_interface_id}__{label}"
        )
        records.append(
            SurfacePartitionRecord(
                partition_id=str(
                    intent.get("partition_id")
                    or f"PART__{parent_interface_id}__{index:04d}"
                ),
                parent_interface_id=parent_interface_id,
                child_surface_id=child_surface_id,
                label=label,
                valid_routes=valid_routes,
                metadata=dict(intent),
            )
        )
    return tuple(records)


def plan_route_surfaces(  # noqa: PLR0913 - published planner signature; arguments are the independent plan inputs
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

    The steps run in a fixed order because the first failing check decides which
    error the user sees: sheet patches, substrate-air surfaces, interface surfaces,
    conductor shell surfaces, then the surface contract metadata.
    """
    semantic_facts = require_semantic_evidence_facade(
        build_input,
        route=route,
        facade=semantic_facts,
        representation_context=representation_context,
    )
    scope = SurfaceScope(
        build_input=build_input,
        route=route,
        interfaces=interfaces,
        semantic_facts=semantic_facts,
        representation_context=representation_context,
    )
    partitions_by_interface = _partitions_by_parent_interface(surface_partitions)

    # Step 1: Route A sheets. Sheet conductors own no standalone surface; their
    # patches replace parts of the substrate-air interface they sit on.
    sheets = _plan_sheets(scope, mm_contacts)

    # Step 2: the substrate-air and domain boundary surfaces, with sheet footprints cut out.
    records: list[SurfacePlanRecord] = []
    records.extend(
        plan_substrate_air_surfaces(
            build_input,
            route=route,
            semantic_facts=semantic_facts,
            route_a_sheet_patches=(*sheets.route_a_patches, *sheets.mask_patches),
            representation_context=representation_context,
        )
    )

    # Step 3: one surface (or its partition children) per recognized interface.
    contact_faces = contact_patches_by_entity_face(interfaces)
    sheet_contacts_by_face = (
        route_a_sheet_contacts_by_face_metal(build_input, mm_contacts)
        if route == "A" or scope.context_planning
        else {}
    )
    normalized_sheet_loops = sheet_contact_loops_by_face_metal(
        build_input,
        mm_contacts,
        include_direct_bump_contacts=False,
    )
    records.extend(
        plan_interface_surfaces(
            scope,
            partitions_by_interface=partitions_by_interface,
            effective_sheet_surfaces=sheets.effective_surfaces,
            sheet_patches_by_interface=sheets.patches_by_interface(),
            sheet_contacts_by_face=sheet_contacts_by_face,
        )
    )

    # Step 4: top, bottom and sidewall surfaces of conductors that have a shell or volume.
    records.extend(
        plan_conductor_shell_surfaces(
            scope,
            mm_contacts=mm_contacts,
            construction_bodies=construction_bodies,
            contact_faces=contact_faces,
            normalized_sheet_loops=normalized_sheet_loops,
        )
    )
    if route == "C":
        return tuple(records)
    return with_surface_contract_metadata(
        build_input,
        route=route,
        interfaces=interfaces,
        mm_contacts=mm_contacts,
        surfaces=tuple(records),
        _conductor_component_ledger=_conductor_component_ledger,
    )


def _partitions_by_parent_interface(
    surface_partitions: tuple[SurfacePartitionRecord, ...],
) -> dict[str, list[SurfacePartitionRecord]]:
    """Group partition records by the interface they subdivide, keeping input order."""
    partitions_by_interface: dict[str, list[SurfacePartitionRecord]] = {}
    for partition in surface_partitions:
        partitions_by_interface.setdefault(
            partition.parent_interface_id,
            [],
        ).append(partition)
    return partitions_by_interface


@dataclass(frozen=True)
class _SheetPlan:
    """What the Route A sheet step hands to the later surface steps."""

    effective_surfaces: Mapping[str, Sequence[SurfacePlanRecord]]
    mask_patches: tuple[RouteASheetPatch, ...]
    route_a_patches: tuple[RouteASheetPatch, ...]

    def patches_by_interface(self) -> dict[str, list[RouteASheetPatch]]:
        """Group the Route A patches by the interface they subdivide."""
        grouped: dict[str, list[RouteASheetPatch]] = {}
        for patch in self.route_a_patches:
            grouped.setdefault(patch.parent_interface_id, []).append(patch)
        return grouped


def _plan_sheets(
    scope: SurfaceScope, mm_contacts: tuple[MMContactRecord, ...]
) -> _SheetPlan:
    """Plan sheets whose contacts hide one side, then patch the remaining sheets."""
    effective_contacts = (
        effective_sheet_contact_sides(
            scope.build_input,
            mm_contacts,
            representation_context=scope.representation_context,
        )
        if scope.context_planning
        else {}
    )
    effective_surfaces, mask_patches = plan_effective_contact_sheets(
        scope, effective_contacts
    )
    plans_patches = scope.route == "A" or (
        scope.context_planning and is_sheet_context(scope.representation_context)
    )
    route_a_patches = (
        plan_route_a_sheet_patches(
            scope.build_input,
            interfaces=tuple(
                interface
                for interface in scope.interfaces
                if interface.interface_id not in effective_surfaces
            ),
            semantic_facts=scope.semantic_facts,
            route=scope.route,
            representation_context=scope.representation_context,
        )
        if plans_patches
        else ()
    )
    return _SheetPlan(effective_surfaces, mask_patches, route_a_patches)
