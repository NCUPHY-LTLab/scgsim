"""Plan every surface of a route, in a fixed order, from interfaces and conductors.

``plan_route_surfaces`` is the entry point. It builds, in order: Route A sheet surfaces,
the substrate-air and domain boundary surfaces, one surface per recognized interface,
and the top, bottom and sidewall surfaces of shell or volume conductors, then attaches
the surface contract metadata. The steps live in ``sheet_patches``, ``sheet_surfaces``,
``surface_sheets``, ``domain``, ``surface_conductors`` and ``surface_contract``.

This module also owns the non-gmsh seam for physical contribution planning
(``plan_surface_contribution_patches``). Mixed sheet and solid contacts exclude their
typed metal footprints before solution coverage is resolved. Finite walls bind authored
dielectric cavity boundaries or ledger-bound vacuum children; Palace route policies
remain caller-owned.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from scgsim.geometry._primitives.entities import (
    _entity_by_id,
    _geometry_ref_from_metadata,
    _unique_ids,
)
from scgsim.geometry._primitives.surface_records import (
    _is_route_a_sheet_interface,
    _RouteASheetPatch,
)
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
from scgsim.geometry.planning.auto_vacuum import _prepare_auto_vacuum_solution_regions
from scgsim.geometry.planning.contact_patches import plan_conductor_contact_patches
from scgsim.geometry.planning.domain import (
    _interface_surface_kinds,
    _plan_substrate_air_surfaces,
)
from scgsim.geometry.planning.evidence import (
    build_semantic_evidence_facade,
    require_semantic_evidence_facade,
)
from scgsim.geometry.planning.interfaces import recognize_route_interfaces
from scgsim.geometry.planning.mm_contacts import (
    _plan_mm_contact_records_with_components,
    plan_mm_contact_records,
)
from scgsim.geometry.planning.representation import (
    is_sheet_context,
)
from scgsim.geometry.planning.sheet_patches import _plan_route_a_sheet_patches
from scgsim.geometry.planning.sheet_surfaces import _route_a_sheet_patch_surfaces
from scgsim.geometry.planning.solution_adjacency import (
    _route_a_sheet_boundary_volume_ids,
)
from scgsim.geometry.planning.solution_boundaries import (
    _merge_solution_sidewall_interfaces,
    _reconcile_solution_domain_boundaries,
)
from scgsim.geometry.planning.surface_conductors import plan_conductor_shell_surfaces
from scgsim.geometry.planning.surface_contacts import (
    _contact_patches_by_entity_face,
    _route_a_sheet_contacts_by_face_metal,
    _sheet_contact_loops_by_face_metal,
)
from scgsim.geometry.planning.surface_contract import _with_surface_contract_metadata
from scgsim.geometry.planning.surface_scope import SurfaceScope
from scgsim.geometry.planning.surface_sheets import (
    _effective_sheet_contact_sides,
    _refresh_generated_context_sheet_interfaces,
    _refresh_generated_route_a_sheet_interfaces,
    plan_effective_contact_sheets,
)
from scgsim.geometry.planning.tags import _physical_group_owner_ids
from scgsim.semantics import EvidenceResult, SemanticEvidenceFacade
from scgsim.semantics.ownership import interface_surface_owner_ids


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
    from scgsim.geometry.compiler.validation import validate_selected_route
    from scgsim.geometry.planning.representation import (
        project_native_route_representations,
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
        build_input = _prepare_auto_vacuum_solution_regions(
            build_input, route=route, representation_context=representation_context
        )
        build_input = _refresh_generated_context_sheet_interfaces(
            build_input, representation_context=representation_context
        )
    elif route == "A":
        build_input = _refresh_generated_route_a_sheet_interfaces(build_input)
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
            _plan_mm_contact_records_with_components(
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
    surfaces = _reconcile_solution_domain_boundaries(build_input, surfaces=surfaces)
    return build_input, _merge_solution_sidewall_interfaces(
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
    context_planning = scope.context_planning
    effective_contacts = (
        _effective_sheet_contact_sides(
            build_input, mm_contacts, representation_context=representation_context
        )
        if context_planning
        else {}
    )
    effective_sheet_surfaces, effective_mask_patches = plan_effective_contact_sheets(
        scope, effective_contacts
    )
    route_a_sheet_patches = (
        _plan_route_a_sheet_patches(
            build_input,
            interfaces=tuple(
                interface
                for interface in interfaces
                if interface.interface_id not in effective_sheet_surfaces
            ),
            semantic_facts=semantic_facts,
            route=route,
            representation_context=representation_context,
        )
        if route == "A"
        or (context_planning and is_sheet_context(representation_context))
        else ()
    )
    sheet_patches_by_interface: dict[str, list[_RouteASheetPatch]] = {}
    for patch in route_a_sheet_patches:
        sheet_patches_by_interface.setdefault(patch.parent_interface_id, []).append(
            patch
        )

    # Step 2: the substrate-air and domain boundary surfaces, with sheet footprints cut out.
    records: list[SurfacePlanRecord] = []
    records.extend(
        _plan_substrate_air_surfaces(
            build_input,
            route=route,
            semantic_facts=semantic_facts,
            route_a_sheet_patches=(*route_a_sheet_patches, *effective_mask_patches),
            representation_context=representation_context,
        )
    )

    # Step 3: one surface (or its partition children) per recognized interface.
    contact_faces = _contact_patches_by_entity_face(interfaces)
    sheet_contacts_by_face = (
        _route_a_sheet_contacts_by_face_metal(build_input, mm_contacts)
        if route == "A" or context_planning
        else {}
    )
    normalized_sheet_loops = _sheet_contact_loops_by_face_metal(
        build_input,
        mm_contacts,
        include_direct_bump_contacts=False,
    )
    records.extend(
        _plan_interface_surfaces(
            scope,
            partitions_by_interface=partitions_by_interface,
            effective_sheet_surfaces=effective_sheet_surfaces,
            sheet_patches_by_interface=sheet_patches_by_interface,
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
    return _with_surface_contract_metadata(
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


def _plan_interface_surfaces(
    scope: SurfaceScope,
    *,
    partitions_by_interface: Mapping[str, Sequence[SurfacePartitionRecord]],
    effective_sheet_surfaces: Mapping[str, Sequence[SurfacePlanRecord]],
    sheet_patches_by_interface: Mapping[str, Sequence[_RouteASheetPatch]],
    sheet_contacts_by_face: Mapping[str, Sequence[MMContactRecord]],
) -> list[SurfacePlanRecord]:
    """Plan the surfaces owned by recognized interfaces, in interface order."""
    route = scope.route
    representation_context = scope.representation_context
    records: list[SurfacePlanRecord] = []
    for interface in scope.interfaces:
        # Same-net conductor contacts are component provenance only.  Neither
        # Route A nor Route B may lower an internal MM face as solver geometry.
        if _is_hidden_contact_interface(
            route, interface, representation_context=representation_context
        ):
            continue
        if interface.interface_id in effective_sheet_surfaces:
            records.extend(effective_sheet_surfaces[interface.interface_id])
            continue
        if _is_route_a_sheet_interface(
            route, interface, representation_context=representation_context
        ):
            patches = tuple(sheet_patches_by_interface.get(interface.interface_id, ()))
            if not patches:
                raise ValueError(
                    f"{interface.interface_id} has no local Route A sheet patches"
                )
            records.extend(
                _route_a_sheet_patch_surfaces(
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
            _planned_interface_records(
                scope,
                interface,
                partitions_by_interface.get(interface.interface_id, ()),
            )
        )
    return records


def _planned_interface_records(
    scope: SurfaceScope,
    interface: InterfacePlanRecord,
    partitions: Sequence[SurfacePartitionRecord],
) -> list[SurfacePlanRecord]:
    """Plan one plain interface as a single surface, or as one child per partition."""
    build_input = scope.build_input
    route = scope.route
    representation_context = scope.representation_context
    geometry_ref = {
        "from_interface_id": interface.interface_id,
        "source_polygon_ids": interface.source_polygon_ids,
        **_geometry_ref_from_metadata(interface.metadata),
    }
    # TODO(human): this evidence is always empty, so `primary_kind` is always None
    # and the Route A evidence arguments below never carry data. Behavior kept as is.
    route_a_sheet_evidence: tuple[EvidenceResult, ...] = ()
    primary_kind = next(
        (
            result.classification
            for result in route_a_sheet_evidence
            if interface.owner_semantic_ids[1] in (result.effective_domain_ids or ())
        ),
        None,
    )
    surface_interface_id = _surface_interface_id(
        build_input,
        route=route,
        interface=interface,
        primary_kind=primary_kind,
        representation_context=representation_context,
    )
    interface_kinds = _interface_surface_kinds(
        build_input,
        route=route,
        interface=interface,
        route_a_evidence=route_a_sheet_evidence,
        representation_context=representation_context,
    )
    owner_semantic_ids = _interface_surface_owner_ids(
        build_input,
        route=route,
        interface=interface,
        route_a_evidence=route_a_sheet_evidence,
        representation_context=representation_context,
    )
    boundary_volume_ids = _interface_boundary_volume_ids(
        build_input,
        route=route,
        interface=interface,
        route_a_evidence=route_a_sheet_evidence,
        representation_context=representation_context,
    )
    physical_owner_semantic_ids = _physical_group_owner_ids(
        build_input,
        owner_semantic_ids,
    )
    embedded_sheet = _is_route_a_sheet_interface(
        route, interface, representation_context=representation_context
    )

    def metadata() -> dict[str, Any]:
        # A fresh dict per record, so no two surfaces share mutable metadata.
        return {
            "interface_kinds": interface_kinds,
            "owner_semantic_ids": owner_semantic_ids,
            "physical_owner_semantic_ids": physical_owner_semantic_ids,
            "boundary_volume_ids": boundary_volume_ids,
            "embedded_surface_sheet": embedded_sheet,
        }

    if partitions:
        parent_surface_id = f"SURF__{surface_interface_id}"
        return [
            SurfacePlanRecord(
                surface_id=partition.child_surface_id,
                owner_semantic_id=interface.owner_semantic_ids[0],
                surface_role=(
                    "representation_context_planned_interface_partition"
                    if scope.context_planning
                    else f"{route}_planned_interface_partition"
                ),
                geometry_ref={
                    **geometry_ref,
                    "partition_id": partition.partition_id,
                    "parent_interface_id": partition.parent_interface_id,
                    **_geometry_ref_from_metadata(partition.metadata),
                },
                interface_id=surface_interface_id,
                parent_surface_id=parent_surface_id,
                partition_label=partition.label,
                solver_use=interface.solver_use or "solver_active",
                valid_routes=() if route is None else (route,),
                metadata=metadata(),
            )
            for partition in partitions
        ]
    return [
        SurfacePlanRecord(
            surface_id=f"SURF__{surface_interface_id}",
            owner_semantic_id=interface.owner_semantic_ids[0],
            surface_role=(
                "representation_context_planned_interface"
                if scope.context_planning
                else f"{route}_planned_interface"
            ),
            geometry_ref=geometry_ref,
            interface_id=surface_interface_id,
            solver_use=interface.solver_use or "solver_active",
            valid_routes=() if route is None else (route,),
            metadata=metadata(),
        )
    ]


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
    if _is_route_a_sheet_interface(
        route, interface, representation_context=representation_context
    ):
        route_a_owner_ids = _unique_ids(
            owner_id
            for result in route_a_evidence
            for owner_id in (
                *result.source_owner_ids,
                *(result.effective_domain_ids or ()),
            )
        )
        if not route_a_owner_ids:
            entity = _entity_by_id(build_input, interface.owner_semantic_ids[0])
            route_a_owner_ids = (
                entity.semantic_id,
                *_route_a_sheet_boundary_volume_ids(build_input, entity),
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
        return _unique_ids(raw_boundary_ids)
    if _is_route_a_sheet_interface(
        route, interface, representation_context=representation_context
    ):
        if route_a_evidence:
            return _unique_ids(
                domain_id
                for result in route_a_evidence
                for domain_id in (result.effective_domain_ids or ())
            )
        return _route_a_sheet_boundary_volume_ids(
            build_input,
            _entity_by_id(build_input, interface.owner_semantic_ids[0]),
        )
    return _unique_ids(interface.owner_semantic_ids)


def _surface_interface_id(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral | None,
    interface: InterfacePlanRecord,
    primary_kind: str | None = None,
    representation_context: RepresentationContext | None = None,
) -> str:
    if not _is_route_a_sheet_interface(
        route, interface, representation_context=representation_context
    ):
        return interface.interface_id
    entity = _entity_by_id(build_input, interface.owner_semantic_ids[0])
    boundary_ids = _route_a_sheet_boundary_volume_ids(build_input, entity)
    suffix = interface.interface_id.rsplit("__", 1)[-1]
    return (
        f"{primary_kind or interface.kind}__{entity.semantic_id}__"
        f"{'__'.join(boundary_ids)}__{suffix}"
    )


def _is_hidden_contact_interface(
    route: RouteLiteral | None,
    interface: InterfacePlanRecord,
    *,
    representation_context: RepresentationContext | None = None,
) -> bool:
    return (
        interface.recognition_rule == "coplanar_conductor_contact_patch"
        and (
            route in {"A", "B"}
            or (route is None and representation_context is not None)
        )
        and bool(interface.metadata.get("hidden_solver_contact"))
    )
