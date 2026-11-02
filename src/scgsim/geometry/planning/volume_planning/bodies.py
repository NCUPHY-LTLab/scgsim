"""Plan cutter bodies, final shell-child bindings and host exclusions in source order."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from scgsim.geometry.models.common import (
    PolygonRing,
    RouteLiteral,
)
from scgsim.geometry.models.construction import (
    ConstructionBodyPlanRecord,
    CutHostOperationRecord,
)
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.models.topology import (
    InterfacePlanRecord,
    SurfacePlanRecord,
)
from scgsim.geometry.planning.domain import (
    conductor_sidewall_adjacent_solution_id,
    required_host_solution_id,
)
from scgsim.geometry.planning.interfaces import is_fully_normalized_face_metal
from scgsim.geometry.planning.surfaces import (
    ConductorSidewallRequest,
    conductor_sidewall_geometry_refs,
    contact_patches_by_entity_face,
    subtract_contact_patches_from_face,
)
from scgsim.geometry.primitives.entity_validation import (
    is_solution_entity,
)
from scgsim.geometry.primitives.geometry_refs import (
    route_entity_geometry_ref,
)
from scgsim.geometry.primitives.surface_records import (
    conductor_boundary_surface_id,
)


def plan_route_construction_bodies(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    interfaces: tuple[InterfacePlanRecord, ...] = (),
    representation_context: RepresentationContext | None = None,
) -> tuple[ConstructionBodyPlanRecord, ...]:
    """Plan Route A/B cutter bodies without making them final geometry.

    Route A only gives construction bodies to conductors represented as
    `cutout_boundary_shell` such as bumps/posts. Route A `surface_sheet`
    conductors are not cutters. Route B gives every `cutout_boundary_shell`
    conductor a construction body. Route C has no construction bodies because
    material volumes survive.
    """
    if route == "C":
        return ()

    contact_faces = contact_patches_by_entity_face(interfaces)
    scope = _BodyScope(
        build_input, route, interfaces, representation_context, contact_faces
    )
    records: list[ConstructionBodyPlanRecord] = []
    for entity in build_input.entities:
        if is_solution_entity(entity):
            continue
        representation = entity.route_representations.get(route)
        if representation != "cutout_boundary_shell":
            continue

        host_id = required_host_solution_id(build_input, entity)
        records.append(
            ConstructionBodyPlanRecord(
                construction_body_id=f"CBODY__{route}__{entity.semantic_id}",
                owner_semantic_id=entity.semantic_id,
                host_semantic_id=host_id,
                representation=representation,
                geometry_ref=route_entity_geometry_ref(
                    build_input,
                    route,
                    entity,
                    representation=representation,
                    interfaces=interfaces,
                ),
                expected_surface_ids=(
                    ()
                    if route == "B"
                    and is_fully_normalized_face_metal(build_input, entity)
                    else _cutout_shell_surface_ids(scope, entity)
                ),
                valid_routes=(route,),
            )
        )
    return tuple(records)


def plan_cut_host_operations(
    *,
    route: RouteLiteral,
    construction_bodies: tuple[ConstructionBodyPlanRecord, ...],
) -> tuple[CutHostOperationRecord, ...]:
    """Group Route A/B construction bodies into host-cut operation plans.

    This is semantic grouping and provenance in the current v1 backend. Exposed
    shell surfaces are already planned as `SurfacePlanRecord`s; these records
    explain which construction bodies belong to each host exclusion policy
    without asking the backend to discover new surfaces through boolean cuts.
    """
    if route == "C":
        return ()

    bodies_by_host: dict[str, list[ConstructionBodyPlanRecord]] = {}
    for body in construction_bodies:
        bodies_by_host.setdefault(body.host_semantic_id, []).append(body)

    return tuple(
        CutHostOperationRecord(
            operation_id=f"CUT__{route}__{host_id}",
            host_semantic_id=host_id,
            construction_body_ids=tuple(body.construction_body_id for body in bodies),
            exposed_surface_ids=tuple(
                surface_id
                for body in bodies
                for surface_id in body.expected_surface_ids
            ),
            valid_routes=(route,),
        )
        for host_id, bodies in bodies_by_host.items()
    )


def reconcile_construction_body_surface_ids(
    construction_bodies: tuple[ConstructionBodyPlanRecord, ...],
    *,
    surfaces: Sequence[SurfacePlanRecord],
) -> tuple[ConstructionBodyPlanRecord, ...]:
    """Bind preplanned cutter bodies to their final local surface children."""
    surface_ids_by_body: dict[str, list[str]] = {}
    for surface in surfaces:
        body_id = surface.geometry_ref.get("construction_body_id")
        if isinstance(body_id, str) and body_id:
            surface_ids_by_body.setdefault(body_id, []).append(surface.surface_id)
    return tuple(
        replace(
            body,
            expected_surface_ids=tuple(
                sorted(surface_ids_by_body.get(body.construction_body_id, ()))
            ),
        )
        if body.construction_body_id in surface_ids_by_body
        else body
        for body in construction_bodies
    )


@dataclass(frozen=True)
class _BodyScope:
    """Source interfaces and contact masks reused by every cutter-body footprint."""

    build_input: GeometryBuildInput
    route: RouteLiteral
    interfaces: tuple[InterfacePlanRecord, ...]
    representation_context: RepresentationContext | None
    contact_faces: Mapping[tuple[str, str], Sequence[PolygonRing]]


def _cutout_shell_surface_ids(
    scope: _BodyScope, entity: SemanticEntitySpec
) -> tuple[str, ...]:
    build_input = scope.build_input
    route = scope.route
    interfaces = scope.interfaces
    representation_context = scope.representation_context
    contact_faces = scope.contact_faces
    contact_faces = contact_faces or {}
    top_bottom_ids: list[str] = []
    for shell_part in ("top", "bottom"):
        base_surface_id = conductor_boundary_surface_id(
            route,
            entity,
            "cutout_boundary_shell",
            shell_part,
        )
        base_geometry_ref = {
            **route_entity_geometry_ref(
                build_input,
                route,
                entity,
                representation="cutout_boundary_shell",
                interfaces=interfaces,
            ),
            "shell_part": shell_part,
        }
        face_geometry_refs = subtract_contact_patches_from_face(
            base_geometry_ref,
            contact_faces.get((entity.semantic_id, shell_part), ()),
        )
        if len(face_geometry_refs) == 1:
            top_bottom_ids.append(base_surface_id)
        else:
            top_bottom_ids.extend(
                f"{base_surface_id}__P{index:04d}"
                for index, _ in enumerate(face_geometry_refs)
            )
    sidewall_adjacent_id = (
        conductor_sidewall_adjacent_solution_id(build_input, entity)
        if route == "C"
        else None
    )
    sidewall_refs = (
        ()
        if route == "C" and sidewall_adjacent_id is None
        else conductor_sidewall_geometry_refs(
            ConductorSidewallRequest(
                build_input,
                route=route,
                entity=entity,
                representation="cutout_boundary_shell",
                adjacent_solution_id=sidewall_adjacent_id,
                representation_context=representation_context,
                interfaces=interfaces,
            )
        )
    )
    sidewall_ids = tuple(
        conductor_boundary_surface_id(
            route,
            entity,
            "cutout_boundary_shell",
            f"sidewall_{edge_index:04d}",
        )
        for edge_index, _ in enumerate(sidewall_refs)
    )
    return (*top_bottom_ids, *sidewall_ids)
