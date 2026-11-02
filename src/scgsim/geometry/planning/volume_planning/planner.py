"""Plan solution and retained material volumes from declared shared-surface ownership."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from scgsim.geometry.models.common import RouteLiteral
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.topology import (
    InnerPecVoidShellRecord,
    MMContactRecord,
    RouteABVolumePlanRecord,
    SurfacePlanRecord,
    SurfaceRefRecord,
    VolumePlanRecord,
)
from scgsim.geometry.planning.domain import (
    component_is_boundary_attached,
)
from scgsim.geometry.planning.tags import entity_physical_group_id
from scgsim.geometry.planning.topology import (
    structured_surface_boundary_volume_ids,
    surface_owner_ids,
)
from scgsim.geometry.planning.volume_planning.orientation import (
    inner_void_orientation,
    surface_orientation_for_volume,
)
from scgsim.geometry.primitives.entity_validation import (
    is_solution_entity,
    is_vacuum_solution_entity,
)
from scgsim.geometry.primitives.geometry_refs import (
    entity_geometry_ref,
)
from scgsim.geometry.primitives.surface_records import (
    surface_boundary_volume_ids,
)


def plan_route_volumes(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    surfaces: tuple[SurfacePlanRecord, ...],
    mm_contacts: tuple[MMContactRecord, ...] = (),
) -> tuple[VolumePlanRecord, ...]:
    """Plan volumes only after all boundary surfaces have stable ids.

    The same rule applies to Route A/B/C: solution domains and retained Route C
    conductors are closed by planned surfaces, then lowered through
    `addSurfaceLoop()` and `addVolume()`. This function must not create a
    volume from `domain_bounds_um`, `outer_loop`, or `thickness_um`; those values
    are only audit/planning metadata once surface ids exist.
    """
    surfaces_by_owner = _index_boundary_surfaces(build_input, surfaces)
    records: list[VolumePlanRecord] = []
    for entity in build_input.entities:
        if is_solution_entity(entity):
            records.append(
                _SolutionVolumePlanner(
                    build_input, route, entity, surfaces_by_owner, mm_contacts
                ).plan()
            )
            continue
        representation = entity.route_representations.get(route)
        if route == "C" and representation == "material_volume":
            records.append(
                _retained_material_volume(
                    route, entity, representation, surfaces_by_owner
                )
            )
    return tuple(records)


def _index_boundary_surfaces(
    build_input: GeometryBuildInput, surfaces: Sequence[SurfacePlanRecord]
) -> dict[str, list[SurfacePlanRecord]]:
    """Index each live source surface by its declared owning volume identities."""
    entity_ids = {entity.semantic_id for entity in build_input.entities}
    surfaces_by_owner: dict[str, list[SurfacePlanRecord]] = {}
    for surface in surfaces:
        owner_ids = (
            structured_surface_boundary_volume_ids(
                surface,
                surface_owner_ids(surface),
            )
            if surface.metadata.get("embedded_surface_sheet")
            else surface_boundary_volume_ids(
                surface,
                known_entity_ids=entity_ids,
            )
        )
        for owner_id in owner_ids:
            surfaces_by_owner.setdefault(str(owner_id), []).append(surface)
    return surfaces_by_owner


@dataclass
class _SolutionVolumePlanner:
    """Carry one solution's oriented boundaries and conductor components together."""

    build_input: GeometryBuildInput
    route: RouteLiteral
    entity: SemanticEntitySpec
    surfaces_by_owner: Mapping[str, Sequence[SurfacePlanRecord]]
    mm_contacts: Sequence[MMContactRecord]

    def plan(self) -> VolumePlanRecord:
        """Orient boundaries before grouping exterior faces and enclosed PEC voids."""
        self.plan_boundary_references()
        self.index_components()
        self.partition_boundaries()
        self.plan_void_shells()
        return self.assemble()

    def plan_boundary_references(self) -> None:
        """Retain physical ownership and orient every declared boundary in surface order."""
        self.physical_owner_semantic_id = (
            "VACUUM_REGION"
            if bool(self.entity.metadata.get("is_auto_vacuum_region"))
            and self.entity.material_kind == "vacuum"
            else None
        )
        self.all_refs = tuple(
            SurfaceRefRecord(
                surface_id=surface.surface_id,
                orientation=surface_orientation_for_volume(surface, self.entity),
                role="planned_boundary",
            )
            for surface in self.surfaces_by_owner.get(self.entity.semantic_id, ())
        )

    def index_components(self) -> None:
        """Index typed contact members and contact IDs without changing multiplicity."""
        self.void_surfaces: dict[str, list[SurfaceRefRecord]] = {}
        self.component_members: dict[str, set[str]] = {}
        self.contact_ids_by_component: dict[str, list[str]] = {}
        for contact in self.mm_contacts:
            self.component_members.setdefault(
                contact.conductor_component_id, set()
            ).update((contact.lower_entity_id, contact.upper_entity_id))
            self.contact_ids_by_component.setdefault(
                contact.conductor_component_id, []
            ).append(contact.contact_id)

    def partition_boundaries(self) -> None:
        """Separate boundary-attached conductor faces from enclosed component voids."""
        self.boundary_components = {
            str(surface.metadata["conductor_component_id"])
            for surface in self.surfaces_by_owner.get(self.entity.semantic_id, ())
            if surface.metadata.get("conductor_component_id") is not None
            and component_is_boundary_attached(
                self.build_input, solution=self.entity, surface=surface
            )
        }
        self.exterior_refs: list[SurfaceRefRecord] = []
        for surface, ref in zip(
            self.surfaces_by_owner.get(self.entity.semantic_id, ()),
            self.all_refs,
            strict=True,
        ):
            is_sheet_cap = (
                self.route == "A"
                and bool(surface.metadata.get("sheet_contact_cap"))
                and is_vacuum_solution_entity(self.entity)
            )
            component_id = str(
                surface.metadata.get(
                    "conductor_component_id",
                    f"COMP__{surface.owner_semantic_id}",
                )
            )
            is_component_boundary = component_id in self.boundary_components
            if (
                surface.surface_role == "cutout_boundary_shell" or is_sheet_cap
            ) and not is_component_boundary:
                self.component_members.setdefault(component_id, set()).add(
                    surface.owner_semantic_id
                )
                self.void_surfaces.setdefault(component_id, []).append(
                    replace(
                        ref,
                        orientation=inner_void_orientation(
                            self.build_input,
                            surface=surface,
                            component_members=self.component_members[component_id],
                        ),
                    )
                )
            else:
                self.exterior_refs.append(ref)

    def plan_void_shells(self) -> None:
        """Bind each void shell to its ordered references and sorted source membership."""
        self.void_shells = tuple(
            InnerPecVoidShellRecord(
                shell_id=f"VOID__{self.entity.semantic_id}__{component_id}",
                surface_refs=tuple(refs),
                conductor_component_id=component_id,
                owner_semantic_ids=tuple(sorted(self.component_members[component_id])),
            )
            for component_id, refs in self.void_surfaces.items()
        )

    def assemble(self) -> VolumePlanRecord:
        """Emit the established A/B shell record or retained Route C solution record."""
        if self.route in {"A", "B"}:
            return RouteABVolumePlanRecord(
                volume_id=f"VOL__{self.entity.semantic_id}",
                owner_semantic_id=self.entity.semantic_id,
                material_id=self.entity.material_id,
                surface_refs=self.all_refs,
                exterior_surface_refs=tuple(self.exterior_refs),
                inner_pec_void_shells=self.void_shells,
                valid_routes=(self.route,),
                metadata={
                    "representation": "solution_volume",
                    "material_kind": self.entity.material_kind,
                    "geometry_ref": dict(self.entity.geometry),
                    "physical_owner_semantic_id": self.physical_owner_semantic_id,
                    "inner_pec_void_shell_contact_ids": {
                        component_id: tuple(
                            self.contact_ids_by_component.get(component_id, ())
                        )
                        for component_id in self.void_surfaces
                    },
                },
            )
        return VolumePlanRecord(
            volume_id=f"VOL__{self.entity.semantic_id}",
            owner_semantic_id=self.entity.semantic_id,
            material_id=self.entity.material_id,
            surface_refs=self.all_refs,
            valid_routes=(self.route,),
            metadata={
                "representation": "solution_volume",
                "material_kind": self.entity.material_kind,
                "geometry_ref": dict(self.entity.geometry),
                "physical_owner_semantic_id": self.physical_owner_semantic_id,
            },
        )


def _retained_material_volume(
    route: RouteLiteral,
    entity: SemanticEntitySpec,
    representation: str,
    surfaces_by_owner: Mapping[str, Sequence[SurfacePlanRecord]],
) -> VolumePlanRecord:
    """Close a retained Route C conductor with its ordered planned material faces."""
    physical_owner_semantic_id = entity_physical_group_id(entity)
    return VolumePlanRecord(
        volume_id=f"VOL__{entity.semantic_id}",
        owner_semantic_id=entity.semantic_id,
        material_id=entity.material_id,
        surface_refs=tuple(
            SurfaceRefRecord(
                surface_id=surface.surface_id,
                orientation=surface_orientation_for_volume(
                    surface,
                    entity,
                ),
                role="planned_boundary",
            )
            for surface in surfaces_by_owner.get(
                entity.semantic_id,
                (),
            )
        ),
        valid_routes=(route,),
        metadata={
            "representation": representation,
            "material_kind": entity.material_kind,
            "geometry_ref": entity_geometry_ref(
                entity,
                representation=representation,
            ),
            "physical_owner_semantic_id": physical_owner_semantic_id,
        },
    )
