"""Plan the top, bottom and sidewall surfaces of conductors that have a shell or volume.

Route B keeps a finite metal volume whose closed exterior shell is PEC (perfect electric
conductor) and excludes its interior from the solution domain; see
``docs/geometry-sgb.qmd``. The code also plans retained material volumes for Route C,
which that page does not list as a current direction. All of them need one surface per
face piece and per sidewall segment, each tagged with the solution entity it touches.
Route A sheet conductors are planned in ``sheet_surfaces`` instead.

This module also owns the interface and boundary-volume ids of those surfaces. The
sidewall segment geometry lives in ``conductor_sidewalls``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from scgsim.geometry._primitives.entity_validation import _is_solution_entity
from scgsim.geometry._primitives.geometry_refs import _route_entity_geometry_ref
from scgsim.geometry._primitives.surface_records import (
    _conductor_boundary_surface_id,
    _surface_contribution_provenance,
)
from scgsim.geometry.models.common import RouteLiteral
from scgsim.geometry.models.construction import ConstructionBodyPlanRecord
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.topology import MMContactRecord, SurfacePlanRecord
from scgsim.geometry.planning.conductor_faces import (
    _conductor_face_solution_pieces,
    _ConductorFace,
    _route_conductor_contact_footprints,
)
from scgsim.geometry.planning.conductor_sidewalls import (
    _conductor_sidewall_geometry_refs,
    _SidewallRequest,
)
from scgsim.geometry.planning.evidence import (
    conductor_solution_evidence,
    metal_metal_evidence,
)
from scgsim.geometry.planning.representation import planned_conductor_representation
from scgsim.geometry.planning.solution_adjacency import (
    _conductor_sidewall_adjacent_solution_id,
)
from scgsim.geometry.planning.surface_contacts import (
    _sheet_contact_loops_by_pad,
    _subtract_contact_patches_from_face,
)
from scgsim.geometry.planning.surface_scope import SurfaceScope
from scgsim.geometry.planning.tags import _physical_group_owner_ids
from scgsim.semantics import EvidenceResult

_Loop = tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class _ShellInputs:
    """Lookups computed once per route and read for every conductor entity."""

    scope: SurfaceScope
    body_by_surface_id: Mapping[str, ConstructionBodyPlanRecord]
    # Loops of coplanar conductor contacts, keyed by (entity id, "top" or "bottom").
    contact_faces: Mapping[tuple[str, str], tuple[_Loop, ...]]
    # Face-metal footprints in contact with a sheet or pad; they are not free faces.
    normalized_sheet_loops: Mapping[str, tuple[_Loop, ...]]
    normalized_pad_loops: Mapping[str, tuple[_Loop, ...]]
    # Representation-context planning replaces an entity by its contact footprints.
    footprints_by_entity: Mapping[str, Sequence[SemanticEntitySpec]]


@dataclass(frozen=True)
class _ConductorShell:
    """One conductor entity and the representation chosen for it."""

    entity: SemanticEntitySpec
    representation: str
    geometry_entities: tuple[SemanticEntitySpec, ...]

    @property
    def surface_role(self) -> str:
        """Role recorded on every surface of this conductor."""
        return (
            "cutout_boundary_shell"
            if self.representation == "cutout_boundary_shell"
            else "material_interface"
        )


def plan_conductor_shell_surfaces(
    scope: SurfaceScope,
    *,
    mm_contacts: Sequence[MMContactRecord],
    construction_bodies: Sequence[ConstructionBodyPlanRecord],
    contact_faces: Mapping[tuple[str, str], tuple[_Loop, ...]],
    normalized_sheet_loops: Mapping[str, tuple[_Loop, ...]],
) -> list[SurfacePlanRecord]:
    """Plan faces then sidewalls for every conductor kept as a shell or volume."""
    body_by_surface_id = {
        surface_id: body
        for body in construction_bodies
        for surface_id in body.expected_surface_ids
    }
    normalized_pad_loops = _sheet_contact_loops_by_pad(scope.build_input, mm_contacts)
    inputs = _ShellInputs(
        scope=scope,
        body_by_surface_id=body_by_surface_id,
        contact_faces=contact_faces,
        normalized_sheet_loops=normalized_sheet_loops,
        normalized_pad_loops=normalized_pad_loops,
        footprints_by_entity=_context_footprints_by_entity(scope),
    )
    records: list[SurfacePlanRecord] = []
    for entity in scope.build_input.entities:
        if _is_solution_entity(entity):
            continue
        representation = planned_conductor_representation(
            entity, route=scope.route, context=scope.representation_context
        )
        if representation not in {"cutout_boundary_shell", "material_volume"}:
            continue
        shell = _ConductorShell(
            entity=entity,
            representation=representation,
            geometry_entities=tuple(
                inputs.footprints_by_entity.get(entity.semantic_id, (entity,))
            ),
        )
        records.extend(_conductor_face_records(inputs, shell))
        records.extend(_conductor_sidewall_records(inputs, shell))
    return records


def _context_footprints_by_entity(
    scope: SurfaceScope,
) -> dict[str, list[SemanticEntitySpec]]:
    """Group contact footprints by entity when planning from a representation context."""
    footprints_by_entity: dict[str, list[SemanticEntitySpec]] = {}
    if scope.context_planning:
        for footprint in _route_conductor_contact_footprints(
            scope.build_input,
            scope.route,
            representation_context=scope.representation_context,
        ):
            footprints_by_entity.setdefault(footprint.semantic_id, []).append(footprint)
    return footprints_by_entity


def _conductor_face_records(
    inputs: _ShellInputs, shell: _ConductorShell
) -> list[SurfacePlanRecord]:
    """Plan the top and bottom faces, split by the solution entity each piece touches."""
    scope = inputs.scope
    entity = shell.entity
    records: list[SurfacePlanRecord] = []
    for shell_part in ("top", "bottom"):
        contact_loops = (
            *inputs.contact_faces.get((entity.semantic_id, shell_part), ()),
            *(
                inputs.normalized_sheet_loops.get(entity.semantic_id, ())
                if scope.route == "B" and entity.part_role == "face_metal"
                else inputs.normalized_pad_loops.get(entity.semantic_id, ())
                # The bottom of a Route A pad is where it lands on its face metal.
                if (scope.route == "A" or scope.context_planning)
                and shell_part == "bottom"
                else ()
            ),
        )
        face_pieces = tuple(
            piece
            for geometry_entity in shell.geometry_entities
            for piece in _conductor_face_solution_pieces(
                scope.build_input,
                _ConductorFace(
                    entity,
                    shell_part,
                    _subtract_contact_patches_from_face(
                        {
                            **_route_entity_geometry_ref(
                                scope.build_input,
                                scope.route,
                                geometry_entity,
                                representation=shell.representation,
                                interfaces=scope.interfaces,
                            ),
                            "shell_part": shell_part,
                        },
                        contact_loops,
                    ),
                ),
                route=scope.route,
                representation_context=scope.representation_context,
            )
        )
        for face_index in range(len(face_pieces)):
            records.append(
                _conductor_face_record(
                    inputs,
                    shell,
                    shell_part=shell_part,
                    face_pieces=face_pieces,
                    face_index=face_index,
                )
            )
    return records


def _conductor_face_record(
    inputs: _ShellInputs,
    shell: _ConductorShell,
    *,
    shell_part: str,
    face_pieces: Sequence[tuple[str, Mapping[str, Any]]],
    face_index: int,
) -> SurfacePlanRecord:
    """Build the surface for one face piece, classified by what it touches."""
    scope = inputs.scope
    entity = shell.entity
    adjacent_id, face_geometry_ref = face_pieces[face_index]
    contribution_id = (
        f"conductor-solution:{entity.semantic_id}:"
        f"{adjacent_id}:{shell_part}:{face_index:04d}"
    )
    interface_evidence = conductor_solution_evidence(
        scope.semantic_facts,
        contribution_id=contribution_id,
        patch_id=f"planned:{contribution_id}",
        conductor_id=entity.semantic_id,
        solution_id=adjacent_id,
        side=shell_part,
    )
    interface_kind = interface_evidence.classification
    base_surface_id = _conductor_boundary_surface_id(
        scope.route, entity, shell.representation, shell_part
    )
    # A face cut into several pieces gets one suffixed surface id per piece.
    surface_id = (
        base_surface_id
        if len(face_pieces) == 1
        else f"{base_surface_id}__P{face_index:04d}"
    )
    body = inputs.body_by_surface_id.get(surface_id)
    if body is None:
        body = inputs.body_by_surface_id.get(base_surface_id)
    face_owner_ids = interface_evidence.source_owner_ids
    return SurfacePlanRecord(
        surface_id=surface_id,
        owner_semantic_id=entity.semantic_id,
        surface_role=shell.surface_role,
        geometry_ref={
            **face_geometry_ref,
            "construction_body_id": (
                body.construction_body_id if body is not None else None
            ),
        },
        interface_id=_conductor_face_interface_id(
            interface_kind,
            entity.semantic_id,
            adjacent_id,
            shell_part,
            None if len(face_pieces) == 1 else face_index,
        ),
        valid_routes=() if scope.route is None else (scope.route,),
        solver_use="solver_active",
        metadata={
            "interface_kinds": (interface_kind,),
            "owner_semantic_ids": face_owner_ids,
            "physical_owner_semantic_ids": _physical_group_owner_ids(
                scope.build_input,
                face_owner_ids,
            ),
            "boundary_volume_ids": _route_conductor_boundary_volume_ids(
                scope.build_input,
                route=scope.route,
                entity=entity,
                face=shell_part,
                adjacent_solution_id=adjacent_id,
            ),
            "exposed_surface_role": shell_part,
            "source_provenance": _surface_contribution_provenance(
                parent_interface_id=(
                    f"conductor-face:{entity.semantic_id}:{shell_part}"
                ),
                patch_id=contribution_id,
                contributions=(interface_evidence,),
            ),
        },
    )


def _conductor_sidewall_records(
    inputs: _ShellInputs, shell: _ConductorShell
) -> list[SurfacePlanRecord]:
    """Plan one surface per exposed sidewall segment of the conductor outline."""
    scope = inputs.scope
    entity = shell.entity
    # Route C sidewalls face exactly one solution entity; the others are resolved per segment.
    sidewall_adjacent_id = (
        _conductor_sidewall_adjacent_solution_id(scope.build_input, entity)
        if scope.route == "C"
        else None
    )
    sidewall_geometry_refs = tuple(
        {
            **geometry_ref,
            **(
                {"source_polygon_ids": geometry_entity.polygon_ids}
                if scope.context_planning
                else {}
            ),
        }
        for geometry_entity in shell.geometry_entities
        for geometry_ref in (
            ()
            if scope.route == "C" and sidewall_adjacent_id is None
            else _conductor_sidewall_geometry_refs(
                scope.build_input,
                _SidewallRequest(
                    geometry_entity,
                    shell.representation,
                    sidewall_adjacent_id,
                    interfaces=scope.interfaces,
                    excluded_footprints=(
                        inputs.normalized_sheet_loops.get(entity.semantic_id, ())
                        if scope.route == "B" and entity.part_role == "face_metal"
                        else ()
                    ),
                ),
                route=scope.route,
                representation_context=scope.representation_context,
            )
        )
    )
    return [
        _conductor_sidewall_record(
            inputs,
            shell,
            geometry_ref=geometry_ref,
            edge_index=edge_index,
            default_adjacent_id=sidewall_adjacent_id,
        )
        for edge_index, geometry_ref in enumerate(sidewall_geometry_refs)
    ]


def _conductor_sidewall_record(
    inputs: _ShellInputs,
    shell: _ConductorShell,
    *,
    geometry_ref: Mapping[str, Any],
    edge_index: int,
    default_adjacent_id: str | None,
) -> SurfacePlanRecord:
    """Build the surface for one sidewall segment, classified by what it touches."""
    scope = inputs.scope
    route = scope.route
    entity = shell.entity
    shell_part = f"sidewall_{edge_index:04d}"
    surface_id = _conductor_boundary_surface_id(
        route, entity, shell.representation, shell_part
    )
    body = inputs.body_by_surface_id.get(surface_id)
    adjacent_owner_id, contribution_id, evidence = _sidewall_evidence(
        scope,
        entity,
        geometry_ref=geometry_ref,
        shell_part=shell_part,
        default_adjacent_id=default_adjacent_id,
    )
    interface_kind = evidence.classification
    # A segment on the domain's outer wall belongs to the conductor alone: it has no
    # neighbouring solution, so it gets no interface id and no interface kind.
    on_exterior = bool(geometry_ref.get("solution_exterior_boundary"))
    interface_id = (
        None
        if on_exterior
        else (
            f"{interface_kind}__{entity.semantic_id}__"
            f"{adjacent_owner_id}__"
            f"{shell_part.upper()}"
        )
    )
    boundary_volume_ids = (
        (entity.semantic_id,)
        if on_exterior
        else _conductor_boundary_volume_ids(route, entity, adjacent_owner_id)
    )
    owner_ids = (entity.semantic_id,) if on_exterior else evidence.source_owner_ids
    physical_owner_ids = _physical_group_owner_ids(scope.build_input, owner_ids)
    physical_name = (
        f"MA__{'__'.join(physical_owner_ids)}__SIDEWALL"
        if entity.part_role == "bump_body" and interface_kind == "MA"
        else None
    )
    return SurfacePlanRecord(
        surface_id=surface_id,
        owner_semantic_id=entity.semantic_id,
        surface_role=shell.surface_role,
        geometry_ref={
            **geometry_ref,
            "from_semantic_id": entity.semantic_id,
            "geometry_kind": entity.geometry_kind,
            "part_role": entity.part_role,
            "representation": shell.representation,
            "source_polygon_ids": geometry_ref.get(
                "source_polygon_ids", entity.polygon_ids
            ),
            "shell_part": shell_part,
            "construction_body_id": (
                body.construction_body_id if body is not None else None
            ),
        },
        interface_id=interface_id,
        valid_routes=() if route is None else (route,),
        solver_use="solver_active",
        metadata={
            "physical_name": physical_name,
            "interface_kinds": () if interface_id is None else (interface_kind,),
            "owner_semantic_ids": owner_ids,
            "physical_owner_semantic_ids": physical_owner_ids,
            "boundary_volume_ids": boundary_volume_ids,
            "exposed_surface_role": shell_part,
            **(
                {
                    "source_provenance": _surface_contribution_provenance(
                        parent_interface_id=f"conductor-sidewall:{entity.semantic_id}",
                        patch_id=contribution_id,
                        contributions=(evidence,),
                    )
                }
                if interface_id is not None
                else {}
            ),
        },
    )


def _sidewall_evidence(
    scope: SurfaceScope,
    entity: SemanticEntitySpec,
    *,
    geometry_ref: Mapping[str, Any],
    shell_part: str,
    default_adjacent_id: str | None,
) -> tuple[str, str, EvidenceResult]:
    """Classify a sidewall segment: metal-metal when a conductor touches it, else metal-solution.

    Returns the adjacent owner id, the contribution id and the evidence result.
    """
    adjacent_owner_id = str(
        geometry_ref.get(
            "adjacent_conductor_semantic_id",
            geometry_ref.get("adjacent_solution_id", default_adjacent_id),
        )
    )
    if not adjacent_owner_id or adjacent_owner_id == "None":
        raise ValueError(
            f"{entity.semantic_id} sidewall lacks exact adjacent solution provenance."
        )
    contribution_id = f"sidewall:{entity.semantic_id}:{adjacent_owner_id}:{shell_part}"
    if "adjacent_conductor_semantic_id" in geometry_ref:
        evidence = metal_metal_evidence(
            scope.semantic_facts,
            contribution_id=contribution_id,
            patch_id=f"planned:{contribution_id}",
            lower_id=entity.semantic_id,
            upper_id=adjacent_owner_id,
            side="sidewall",
        )
    else:
        evidence = conductor_solution_evidence(
            scope.semantic_facts,
            contribution_id=contribution_id,
            patch_id=f"planned:{contribution_id}",
            conductor_id=entity.semantic_id,
            solution_id=adjacent_owner_id,
            side="sidewall",
        )
    return adjacent_owner_id, contribution_id, evidence


def _conductor_face_interface_id(
    interface_kind: str,
    semantic_id: str,
    adjacent_id: str,
    shell_part: str,
    face_index: int | None,
) -> str:
    suffix = "" if face_index is None else f"__P{face_index:04d}"
    return (
        f"{interface_kind}__{semantic_id}__{adjacent_id}__{shell_part.upper()}{suffix}"
    )


def _conductor_boundary_volume_ids(
    route: RouteLiteral | None,
    entity: SemanticEntitySpec,
    adjacent_solution_id: str,
) -> tuple[str, ...]:
    if route == "C":
        return (entity.semantic_id, adjacent_solution_id)
    return (adjacent_solution_id,)


def _route_conductor_boundary_volume_ids(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral | None,
    entity: SemanticEntitySpec,
    face: str,
    adjacent_solution_id: str,
) -> tuple[str, ...]:
    """Keep each finite conductor face with its directly adjacent solution.

    Route-A boundary-attached components join an exterior shell through their
    exposed side/top faces; their retained bottom MS face belongs only to the
    substrate-side solution and is never duplicated into air.
    """
    del build_input, face
    return _conductor_boundary_volume_ids(route, entity, adjacent_solution_id)
