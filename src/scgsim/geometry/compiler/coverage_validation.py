"""Own route, interface, partition and physical-tag coverage before/after lowering.

Checks retain their existing aggregation and failure order. Connectivity and
geometric overlap belong to topology_validation; native construction is downstream.
"""

from __future__ import annotations

from scgsim.geometry.models.common import ROUTE_ALLOWED_REPRESENTATIONS, RouteLiteral
from scgsim.geometry.models.construction import (
    ConstructionBodyPlanRecord,
    CutHostOperationRecord,
    SurfacePartitionRecord,
)
from scgsim.geometry.models.input import GeometryBuildInput
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.models.tags import BackendEntityTagRecord, TagPlanRecord
from scgsim.geometry.models.topology import (
    InterfacePlanRecord,
    SurfacePlanRecord,
    VolumePlanRecord,
)
from scgsim.geometry.planning.representation import planned_conductor_representation
from scgsim.geometry.primitives.entity_validation import (
    is_solution_entity,
    requires_route_representation,
)

_INTERFACE_KINDS = {"MM", "SS", "AA", "MS", "MA", "SA"}


def validate_selected_route(
    build_input: GeometryBuildInput,
    route: RouteLiteral,
    *,
    representation_context=None,
) -> None:
    """Validate that every conductor has an explicit representation for route."""
    if representation_context is None:
        raise ValueError(
            "current compilation requires interpret_physical_geometry() and plan.context"
        )
    context: RepresentationContext = representation_context
    allowed = ROUTE_ALLOWED_REPRESENTATIONS.get(route)
    if allowed is None:
        raise ValueError(f"Unsupported route: {route}")

    errors: list[str] = []
    if route == "C" and build_input.port_sheet_regions:
        errors.append("Route C does not support Palace lumped-port sheets")
    for entity in build_input.entities:
        if not requires_route_representation(entity):
            continue
        representation = planned_conductor_representation(
            entity, route=route, context=context
        )
        if representation is None:
            errors.append(f"{entity.semantic_id} does not define Route {route}")
        elif representation not in allowed:
            errors.append(
                f"{entity.semantic_id} Route {route} representation "
                f"{representation!r} is not supported"
            )
        if route == "C" and entity.geometry.get("route_ab_fused_selector_mode"):
            errors.append(
                f"{entity.semantic_id} Route C cannot use route_ab_fused_selector_mode"
            )
    if errors:
        raise ValueError("Invalid selected route: " + "; ".join(errors))


def validate_route_operation_coverage(
    *,
    construction_bodies: tuple[ConstructionBodyPlanRecord, ...],
    cut_operations: tuple[CutHostOperationRecord, ...],
    surfaces: tuple[SurfacePlanRecord, ...],
) -> None:
    """Validate Route A/B construction-body, cut, and shell-surface linkage."""
    surface_ids = {surface.surface_id for surface in surfaces}
    covered_surface_ids = {
        *surface_ids,
        *(
            surface.parent_surface_id
            for surface in surfaces
            if surface.parent_surface_id is not None
        ),
    }
    body_ids = {body.construction_body_id for body in construction_bodies}
    operation_body_ids = {
        construction_body_id
        for operation in cut_operations
        for construction_body_id in operation.construction_body_ids
    }
    operation_surface_ids = {
        surface_id
        for operation in cut_operations
        for surface_id in operation.exposed_surface_ids
    }

    errors: list[str] = []
    for body in construction_bodies:
        missing_surfaces = [
            surface_id
            for surface_id in body.expected_surface_ids
            if surface_id not in covered_surface_ids
        ]
        if missing_surfaces:
            errors.append(
                f"{body.construction_body_id} expects missing shell surfaces "
                f"{missing_surfaces!r}"
            )

    missing_operation_bodies = sorted(body_ids - operation_body_ids)
    if missing_operation_bodies:
        errors.append(
            f"construction bodies without cut operations {missing_operation_bodies!r}"
        )

    missing_operation_surfaces = sorted(operation_surface_ids - covered_surface_ids)
    if missing_operation_surfaces:
        errors.append(
            f"cut operations expose missing surfaces {missing_operation_surfaces!r}"
        )

    if errors:
        raise ValueError("Invalid route operation plan: " + "; ".join(errors))


def validate_surface_sheet_interface_coverage(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    surfaces: tuple[SurfacePlanRecord, ...],
) -> None:
    """Fail if Route A sheet conductors are not represented by interfaces."""
    sheet_ids = {
        entity.semantic_id
        for entity in build_input.entities
        if not is_solution_entity(entity)
        and entity.route_representations.get(route) == "surface_sheet"
    }
    if not sheet_ids:
        return

    covered_ids: set[str] = set()
    for surface in surfaces:
        if surface.interface_id is None:
            continue
        owner_ids = surface.metadata.get(
            "owner_semantic_ids",
            (surface.owner_semantic_id,),
        )
        if isinstance(owner_ids, str):
            owner_ids = (owner_ids,)
        covered_ids.update(str(owner_id) for owner_id in owner_ids)

    missing = sorted(sheet_ids - covered_ids)
    if missing:
        raise ValueError(
            "surface_sheet entities require planned interface surface coverage: "
            f"{missing!r}"
        )


def validate_route_volume_surface_refs(
    *,
    route: RouteLiteral,
    volumes: tuple[VolumePlanRecord, ...],
) -> None:
    """Fail fast when backend-live volumes have no planned boundaries."""
    missing = sorted(
        volume.volume_id
        for volume in volumes
        if not volume.construction_only and not volume.surface_refs
    )
    ambiguous_solution_shells = sorted(
        volume.volume_id
        for volume in volumes
        if route in {"A", "B"}
        and not volume.construction_only
        and volume.metadata.get("representation") == "solution_volume"
        and not getattr(volume, "exterior_surface_refs", ())
    )
    if missing:
        raise ValueError(
            f"Backend-live volumes require planned boundary surfaces: {missing!r}"
        )
    if ambiguous_solution_shells:
        raise ValueError(
            "solution volumes require explicit exterior shell refs: "
            f"{ambiguous_solution_shells!r}"
        )


def validate_interface_surface_source_of_truth(
    *,
    interfaces: tuple[InterfacePlanRecord, ...],
    surfaces: tuple[SurfacePlanRecord, ...],
) -> None:
    """Require every solver-relevant interface surface to come from InterfacePlan."""
    interface_ids = {interface.interface_id for interface in interfaces}
    errors: list[str] = []
    for surface in surfaces:
        if surface.construction_only:
            continue
        interface_kinds = _surface_interface_kinds(surface)
        if len(interface_kinds) > 2:
            errors.append(
                f"{surface.surface_id} has too many interface kinds {interface_kinds!r}"
            )
        if interface_kinds and surface.interface_id is None:
            errors.append(
                f"{surface.surface_id} has interface kinds "
                f"{interface_kinds!r} without InterfacePlan source"
            )
        if (
            surface.interface_id is not None
            and surface.interface_id not in interface_ids
        ):
            errors.append(
                f"{surface.surface_id} references unknown interface "
                f"{surface.interface_id}"
            )
    if errors:
        raise ValueError("Invalid interface source of truth: " + "; ".join(errors))


def validate_surface_partition_coverage(
    *,
    interfaces: tuple[InterfacePlanRecord, ...],
    surface_partitions: tuple[SurfacePartitionRecord, ...],
    surfaces: tuple[SurfacePlanRecord, ...],
) -> None:
    """Validate that partition records point to real parent/child plan ids."""
    surfaces_by_id = {surface.surface_id: surface for surface in surfaces}
    interface_ids = {interface.interface_id for interface in interfaces}
    interface_ids.update(
        surface.interface_id for surface in surfaces if surface.interface_id is not None
    )
    errors: list[str] = []
    for partition in surface_partitions:
        if partition.parent_interface_id not in interface_ids:
            errors.append(
                f"{partition.partition_id} references unknown parent interface "
                f"{partition.parent_interface_id}"
            )
        child_surface = surfaces_by_id.get(partition.child_surface_id)
        if child_surface is None:
            errors.append(
                f"{partition.partition_id} references unknown child surface "
                f"{partition.child_surface_id}"
            )
            continue
        geometry_ref = child_surface.geometry_ref
        if (
            "planar_loops_3d" not in geometry_ref
            and "outer_loop" not in geometry_ref
            and "loop_geometry_ref" not in geometry_ref
            and "quad_points" not in geometry_ref
        ):
            errors.append(f"{partition.partition_id} child surface lacks loop geometry")

    if errors:
        raise ValueError("Invalid surface partition plan: " + "; ".join(errors))


def validate_tag_plan_coverage(
    *,
    surfaces: tuple[SurfacePlanRecord, ...],
    volumes: tuple[VolumePlanRecord, ...],
    tags: tuple[TagPlanRecord, ...],
) -> None:
    """Validate that every exported geometry record has a tag plan.

    Interface surfaces, domain exterior boundary surfaces, and non-construction
    volumes must have `TagPlanRecord`s before OCC assigns dim-tags.
    """
    live_sources = {
        ("surface", surface.surface_id)
        for surface in surfaces
        if not surface.construction_only
        and not surface.metadata.get("hidden_solver_contact")
        and (
            surface.interface_id is not None
            or surface.surface_role == "domain_boundary"
            or surface.surface_role == "lumped_port"
        )
    }
    live_sources.update(
        ("volume", volume.volume_id)
        for volume in volumes
        if not volume.construction_only
    )
    tag_sources = {(tag.source_record_kind, tag.source_record_id) for tag in tags}

    missing = sorted(live_sources - tag_sources)
    extra = sorted(tag_sources - live_sources)
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing tags for {missing!r}")
        if extra:
            parts.append(f"tags reference non-live records {extra!r}")
        raise ValueError("Invalid tag plan coverage: " + "; ".join(parts))


def validate_backend_tag_ledger(
    *,
    backend_tags: tuple[BackendEntityTagRecord, ...],
) -> None:
    """Validate plan-id to backend-tag mapping after OCC lowering."""
    source_to_dimtags: dict[tuple[str, str], set[tuple[int, int]]] = {}
    dimtag_to_sources: dict[tuple[int, int], set[tuple[str, str]]] = {}
    for tag in backend_tags:
        source = (tag.source_record_kind, tag.source_record_id)
        source_to_dimtags.setdefault(source, set()).add(tag.dim_tag)
        dimtag_to_sources.setdefault(tag.dim_tag, set()).add(source)

    errors: list[str] = []
    for source, dimtags in source_to_dimtags.items():
        if len(dimtags) > 1:
            errors.append(f"{source!r} maps to multiple backend tags {dimtags!r}")
    for dimtag, sources in dimtag_to_sources.items():
        if len(sources) > 1:
            errors.append(f"backend tag {dimtag!r} is shared by sources {sources!r}")
    if errors:
        raise ValueError("Invalid backend tag ledger: " + "; ".join(errors))


def _surface_interface_kinds(surface: SurfacePlanRecord) -> tuple[str, ...]:
    kinds = surface.metadata.get("interface_kinds", ())
    if isinstance(kinds, str):
        kinds = (kinds,)
    return tuple(sorted({str(kind) for kind in kinds if str(kind) in _INTERFACE_KINDS}))
