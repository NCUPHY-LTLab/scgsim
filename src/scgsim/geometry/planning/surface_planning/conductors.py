"""Plan finite conductor caps and sidewalls with their shared contact and body bindings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from scgsim.geometry.interpretation.representation import (
    planned_conductor_representation,
)
from scgsim.geometry.models.common import (
    PolygonRing,
    RouteLiteral,
)
from scgsim.geometry.models.entities import is_solution_entity
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.topology import (
    SurfacePlanRecord,
)
from scgsim.geometry.planning.domain_planning.faces import (
    ConductorFaceRequest,
    conductor_face_solution_pieces,
)
from scgsim.geometry.planning.domain_planning.hosts import (
    conductor_sidewall_adjacent_solution_id,
)
from scgsim.geometry.planning.domain_planning.occupancy import (
    route_conductor_contact_footprints,
)
from scgsim.geometry.planning.evidence import (
    conductor_solution_evidence,
    metal_metal_evidence,
)
from scgsim.geometry.planning.geometry_refs import route_entity_geometry_ref
from scgsim.geometry.planning.surface_planning.contacts import (
    sheet_contact_loops_by_pad,
    subtract_contact_patches_from_face,
)
from scgsim.geometry.planning.surface_planning.scope import SurfaceScope
from scgsim.geometry.planning.surface_planning.sidewalls import (
    ConductorSidewallRequest,
    conductor_sidewall_geometry_refs,
)
from scgsim.geometry.planning.surface_records import (
    conductor_boundary_surface_id,
    surface_contribution_provenance,
)
from scgsim.geometry.planning.tags import (
    physical_group_owner_ids,
)


class ConductorSurfacePlanner:
    """Share face contact masks and body bindings across each finite conductor."""

    def __init__(
        self,
        scope: SurfaceScope,
        contact_faces: Mapping[tuple[str, str], Sequence[PolygonRing]],
        normalized_sheet_loops: Mapping[str, Sequence[PolygonRing]],
    ):
        self.scope = scope
        self.contact_faces = contact_faces
        self.normalized_sheet_loops = normalized_sheet_loops
        self.records: list[SurfacePlanRecord] = []

    def plan(self) -> tuple[SurfacePlanRecord, ...]:
        """Index authored footprints, then plan caps and walls in source order."""
        self.index_footprints()
        for entity in self.scope.build_input.entities:
            if is_solution_entity(entity):
                continue
            self.entity = entity
            representation = planned_conductor_representation(
                entity,
                route=self.scope.route,
                context=self.scope.representation_context,
            )
            if representation in {"cutout_boundary_shell", "material_volume"}:
                self.representation = representation
                self.geometry_entities = tuple(
                    self.context_footprints_by_entity.get(entity.semantic_id, (entity,))
                )
                self.plan_faces()
                self.plan_sidewalls()
        return tuple(self.records)

    def index_footprints(self):
        """Index construction bodies and normalized contact footprints in source order."""
        self.construction_body_by_surface_id = {
            surface_id: body
            for body in self.scope.construction_bodies
            for surface_id in body.expected_surface_ids
        }
        self.normalized_pad_loops = sheet_contact_loops_by_pad(
            self.scope.build_input, self.scope.mm_contacts
        )
        self.context_footprints_by_entity: dict[str, list[SemanticEntitySpec]] = {}
        if self.scope.context_planning:
            for footprint in route_conductor_contact_footprints(
                self.scope.build_input,
                self.scope.route,
                representation_context=self.scope.representation_context,
            ):
                self.context_footprints_by_entity.setdefault(
                    footprint.semantic_id, []
                ).append(footprint)

    def plan_faces(self):
        """Plan top and bottom faces after subtracting their typed contact masks."""
        for shell_part in ("top", "bottom"):
            base_surface_id = conductor_boundary_surface_id(
                self.scope.route,
                self.entity,
                self.representation,
                shell_part,
            )
            contact_loops = (
                *self.contact_faces.get((self.entity.semantic_id, shell_part), ()),
                *(
                    self.normalized_sheet_loops.get(self.entity.semantic_id, ())
                    if self.scope.route == "B" and self.entity.part_role == "face_metal"
                    else self.normalized_pad_loops.get(self.entity.semantic_id, ())
                    if (self.scope.route == "A" or self.scope.context_planning)
                    and shell_part == "bottom"
                    else ()
                ),
            )
            face_pieces = tuple(
                piece
                for geometry_entity in self.geometry_entities
                for piece in conductor_face_solution_pieces(
                    ConductorFaceRequest(
                        self.scope.build_input,
                        route=self.scope.route,
                        entity=self.entity,
                        shell_part=shell_part,
                        geometry_refs=subtract_contact_patches_from_face(
                            {
                                **route_entity_geometry_ref(
                                    self.scope.route,
                                    geometry_entity,
                                    representation=self.representation,
                                    interfaces=self.scope.interfaces,
                                ),
                                "shell_part": shell_part,
                            },
                            contact_loops,
                        ),
                        representation_context=self.scope.representation_context,
                    )
                )
            )
            for face_index, (adjacent_id, face_geometry_ref) in enumerate(face_pieces):
                contribution_id = (
                    f"conductor-solution:{self.entity.semantic_id}:"
                    f"{adjacent_id}:{shell_part}:{face_index:04d}"
                )
                interface_evidence = conductor_solution_evidence(
                    self.scope.semantic_facts,
                    contribution_id=contribution_id,
                    patch_id=f"planned:{contribution_id}",
                    conductor_id=self.entity.semantic_id,
                    solution_id=adjacent_id,
                    side=shell_part,
                )
                interface_kind = interface_evidence.classification
                surface_id = (
                    base_surface_id
                    if len(face_pieces) == 1
                    else f"{base_surface_id}__P{face_index:04d}"
                )
                body = self.construction_body_by_surface_id.get(surface_id)
                if body is None:
                    body = self.construction_body_by_surface_id.get(base_surface_id)
                face_owner_ids = interface_evidence.source_owner_ids
                self.records.append(
                    SurfacePlanRecord(
                        surface_id=surface_id,
                        owner_semantic_id=self.entity.semantic_id,
                        surface_role=(
                            "cutout_boundary_shell"
                            if self.representation == "cutout_boundary_shell"
                            else "material_interface"
                        ),
                        geometry_ref={
                            **face_geometry_ref,
                            "construction_body_id": (
                                body.construction_body_id if body is not None else None
                            ),
                        },
                        interface_id=_conductor_face_interface_id(
                            interface_kind,
                            self.entity.semantic_id,
                            adjacent_id,
                            shell_part,
                            None if len(face_pieces) == 1 else face_index,
                        ),
                        valid_routes=()
                        if self.scope.route is None
                        else (self.scope.route,),
                        solver_use="solver_active",
                        metadata={
                            "interface_kinds": (interface_kind,),
                            "owner_semantic_ids": face_owner_ids,
                            "physical_owner_semantic_ids": (
                                physical_group_owner_ids(
                                    self.scope.build_input,
                                    face_owner_ids,
                                )
                            ),
                            "boundary_volume_ids": (
                                _route_conductor_boundary_volume_ids(
                                    self.scope.build_input,
                                    route=self.scope.route,
                                    entity=self.entity,
                                    face=shell_part,
                                    adjacent_solution_id=adjacent_id,
                                )
                            ),
                            "exposed_surface_role": shell_part,
                            "source_provenance": (
                                surface_contribution_provenance(
                                    parent_interface_id=(
                                        f"conductor-face:{self.entity.semantic_id}:"
                                        f"{shell_part}"
                                    ),
                                    patch_id=contribution_id,
                                    contributions=(interface_evidence,),
                                )
                            ),
                        },
                    )
                )

    def plan_sidewalls(self):
        """Bind exact sidewall segments to conductor or solution evidence."""
        sidewall_adjacent_id = (
            conductor_sidewall_adjacent_solution_id(self.scope.build_input, self.entity)
            if self.scope.route == "C"
            else None
        )
        sidewall_geometry_refs = tuple(
            {
                **geometry_ref,
                **(
                    {"source_polygon_ids": geometry_entity.polygon_ids}
                    if self.scope.context_planning
                    else {}
                ),
            }
            for geometry_entity in self.geometry_entities
            for geometry_ref in (
                ()
                if self.scope.route == "C" and sidewall_adjacent_id is None
                else conductor_sidewall_geometry_refs(
                    ConductorSidewallRequest(
                        self.scope.build_input,
                        route=self.scope.route,
                        entity=geometry_entity,
                        representation=self.representation,
                        adjacent_solution_id=sidewall_adjacent_id,
                        interfaces=self.scope.interfaces,
                        excluded_footprints=self.normalized_sheet_loops.get(
                            self.entity.semantic_id, ()
                        )
                        if self.scope.route == "B"
                        and self.entity.part_role == "face_metal"
                        else (),
                        representation_context=self.scope.representation_context,
                    )
                )
            )
        )
        self.append_sidewalls(sidewall_geometry_refs, sidewall_adjacent_id)

    def append_sidewalls(
        self,
        sidewall_geometry_refs: Sequence[Mapping[str, Any]],
        sidewall_adjacent_id: str | None,
    ):
        """Emit each wall's exact adjacency, evidence and body binding in order."""
        for edge_index, geometry_ref in enumerate(sidewall_geometry_refs):
            shell_part = f"sidewall_{edge_index:04d}"
            surface_id = conductor_boundary_surface_id(
                self.scope.route,
                self.entity,
                self.representation,
                shell_part,
            )
            body = self.construction_body_by_surface_id.get(surface_id)
            sidewall_adjacent_owner_id = str(
                geometry_ref.get(
                    "adjacent_conductor_semantic_id",
                    geometry_ref.get("adjacent_solution_id", sidewall_adjacent_id),
                )
            )
            if not sidewall_adjacent_owner_id or sidewall_adjacent_owner_id == "None":
                raise ValueError(
                    f"{self.entity.semantic_id} sidewall lacks exact adjacent solution provenance."
                )
            contribution_id = (
                f"sidewall:{self.entity.semantic_id}:"
                f"{sidewall_adjacent_owner_id}:{shell_part}"
            )
            if "adjacent_conductor_semantic_id" in geometry_ref:
                sidewall_evidence = metal_metal_evidence(
                    self.scope.semantic_facts,
                    contribution_id=contribution_id,
                    patch_id=f"planned:{contribution_id}",
                    lower_id=self.entity.semantic_id,
                    upper_id=sidewall_adjacent_owner_id,
                    side="sidewall",
                )
            else:
                sidewall_evidence = conductor_solution_evidence(
                    self.scope.semantic_facts,
                    contribution_id=contribution_id,
                    patch_id=f"planned:{contribution_id}",
                    conductor_id=self.entity.semantic_id,
                    solution_id=sidewall_adjacent_owner_id,
                    side="sidewall",
                )
            sidewall_interface_kind = sidewall_evidence.classification
            sidewall_interface_id = (
                None
                if geometry_ref.get("solution_exterior_boundary")
                else (
                    f"{sidewall_interface_kind}__{self.entity.semantic_id}__"
                    f"{sidewall_adjacent_owner_id}__"
                    f"{shell_part.upper()}"
                )
            )
            sidewall_boundary_volume_ids = (
                (self.entity.semantic_id,)
                if geometry_ref.get("solution_exterior_boundary")
                else _conductor_boundary_volume_ids(
                    self.scope.route,
                    self.entity,
                    sidewall_adjacent_owner_id,
                )
            )
            sidewall_owner_ids = (
                (self.entity.semantic_id,)
                if geometry_ref.get("solution_exterior_boundary")
                else sidewall_evidence.source_owner_ids
            )
            sidewall_physical_owner_ids = physical_group_owner_ids(
                self.scope.build_input,
                sidewall_owner_ids,
            )
            sidewall_physical_name = (
                f"MA__{'__'.join(sidewall_physical_owner_ids)}__SIDEWALL"
                if self.entity.part_role == "bump_body"
                and sidewall_interface_kind == "MA"
                else None
            )
            self.records.append(
                SurfacePlanRecord(
                    surface_id=surface_id,
                    owner_semantic_id=self.entity.semantic_id,
                    surface_role=(
                        "cutout_boundary_shell"
                        if self.representation == "cutout_boundary_shell"
                        else "material_interface"
                    ),
                    geometry_ref={
                        **geometry_ref,
                        "from_semantic_id": self.entity.semantic_id,
                        "geometry_kind": self.entity.geometry_kind,
                        "part_role": self.entity.part_role,
                        "representation": self.representation,
                        "source_polygon_ids": geometry_ref.get(
                            "source_polygon_ids", self.entity.polygon_ids
                        ),
                        "shell_part": shell_part,
                        "construction_body_id": (
                            body.construction_body_id if body is not None else None
                        ),
                    },
                    interface_id=sidewall_interface_id,
                    valid_routes=()
                    if self.scope.route is None
                    else (self.scope.route,),
                    solver_use="solver_active",
                    metadata={
                        "physical_name": sidewall_physical_name,
                        "interface_kinds": (
                            ()
                            if sidewall_interface_id is None
                            else (sidewall_interface_kind,)
                        ),
                        "owner_semantic_ids": sidewall_owner_ids,
                        "physical_owner_semantic_ids": (sidewall_physical_owner_ids),
                        "boundary_volume_ids": sidewall_boundary_volume_ids,
                        "exposed_surface_role": shell_part,
                        **(
                            {
                                "source_provenance": (
                                    surface_contribution_provenance(
                                        parent_interface_id=(
                                            f"conductor-sidewall:"
                                            f"{self.entity.semantic_id}"
                                        ),
                                        patch_id=contribution_id,
                                        contributions=(sidewall_evidence,),
                                    )
                                )
                            }
                            if sidewall_interface_id is not None
                            else {}
                        ),
                    },
                )
            )


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
