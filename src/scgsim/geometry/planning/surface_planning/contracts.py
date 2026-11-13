"""Attach component, net, layer and source ownership to final surface records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from scgsim.geometry.grid.entities import component_metadata, unique_ids
from scgsim.geometry.grid.entity_validation import is_solution_entity
from scgsim.geometry.models.common import (
    RouteLiteral,
)
from scgsim.geometry.models.input import GeometryBuildInput
from scgsim.geometry.models.topology import (
    InterfacePlanRecord,
    MMContactRecord,
    SurfacePlanRecord,
)
from scgsim.geometry.planning.surface_planning.contacts import is_face_pad_mm_contact
from scgsim.geometry.planning.tags import (
    surface_physical_owner_ids,
)
from scgsim.geometry.planning.topology_planning.ownership import surface_owner_ids


def with_surface_contract_metadata(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral | None,
    interfaces: tuple[InterfacePlanRecord, ...],
    mm_contacts: tuple[MMContactRecord, ...],
    surfaces: tuple[SurfacePlanRecord, ...],
    _conductor_component_ledger: tuple[dict[str, Any], ...] | None = None,
) -> tuple[SurfacePlanRecord, ...]:
    """Attach the explicit fields required by final surface export."""
    return _SurfaceContractBinder(
        build_input, route, interfaces, mm_contacts, _conductor_component_ledger
    ).bind(surfaces)


@dataclass
class _SurfaceContractBinder:
    """Share declared component indexes across the final ordered surface records."""

    build_input: GeometryBuildInput
    route: RouteLiteral | None
    interfaces: tuple[InterfacePlanRecord, ...]
    mm_contacts: tuple[MMContactRecord, ...]
    requested_component_ledger: tuple[dict[str, Any], ...] | None

    def bind(
        self, surfaces: tuple[SurfacePlanRecord, ...]
    ) -> tuple[SurfacePlanRecord, ...]:
        """Build component indexes once, then bind surfaces in caller order."""
        self.entities = {
            entity.semantic_id: entity for entity in self.build_input.entities
        }
        self.interface_by_id = {
            interface.interface_id: interface for interface in self.interfaces
        }
        # Context planning supplies the complete ledger from the same contact-graph
        # pass, including components without a planar MM face. Other routes retain
        # their MM-record projection. Never infer another graph from surface adjacency.
        self.component_ledger = (
            self.requested_component_ledger
            if self.requested_component_ledger is not None
            else component_metadata(self.mm_contacts)
        )
        self.component_by_entity = {
            entity_id: str(component["conductor_component_id"])
            for component in self.component_ledger
            for entity_id in component["members"]
        }
        self.component_net = {
            str(component["conductor_component_id"]): component["net_id"]
            for component in self.component_ledger
        }
        self.component_equipotential = {
            str(component["conductor_component_id"]): component["equipotential_id"]
            for component in self.component_ledger
        }
        return tuple(self.bind_surface(surface) for surface in surfaces)

    def bind_surface(self, surface: SurfacePlanRecord) -> SurfacePlanRecord:
        """Attach net, component, layer and source metadata in validation order."""
        owner_ids = surface_owner_ids(surface)
        structured_owner_ids = surface_physical_owner_ids(surface)
        conductors = tuple(
            self.entities[owner_id]
            for owner_id in owner_ids
            if owner_id in self.entities
            and not is_solution_entity(self.entities[owner_id])
        )
        net_ids = {entity.net_id for entity in conductors if entity.net_id}
        if len(net_ids) > 1:
            raise ValueError(f"{surface.surface_id} has ambiguous conductor net")
        component_ids = {
            self.component_by_entity[entity.semantic_id]
            for entity in conductors
            if entity.semantic_id in self.component_by_entity
        }
        if conductors and not component_ids:
            component_ids = {f"COMP__{conductors[0].semantic_id}"}
        if len(component_ids) > 1:
            raise ValueError(
                f"{surface.surface_id} spans independent conductor components"
            )
        equipotential_ids = {
            str(entity.metadata["equipotential_id"])
            for entity in conductors
            if entity.metadata.get("equipotential_id") is not None
        }
        if len(equipotential_ids) > 1:
            raise ValueError(f"{surface.surface_id} has ambiguous equipotential id")
        interface = self.interface_by_id.get(surface.interface_id or "")
        raw_interface_kinds = surface.metadata.get("interface_kinds", ())
        interface_kinds = (
            (str(raw_interface_kinds),)
            if isinstance(raw_interface_kinds, str)
            else tuple(str(kind) for kind in raw_interface_kinds)
        )
        face_kind = str(
            surface.metadata.get(
                "exposed_surface_role",
                surface.metadata.get("boundary_role", "interface"),
            )
        )
        if face_kind.startswith("sidewall_"):
            face_kind = "sidewall"
        source_layer_names = {
            str(entity.metadata["source_layer_name"])
            for entity in conductors
            if entity.metadata.get("source_layer_name") is not None
        }
        if len(source_layer_names) > 1:
            raise ValueError(
                f"{surface.surface_id} has ambiguous conductor source_layer_name"
            )
        existing_source_provenance = surface.metadata.get("source_provenance", {})
        if not isinstance(existing_source_provenance, Mapping):
            raise TypeError(f"{surface.surface_id} source_provenance must be a mapping")
        return replace(
            surface,
            metadata={
                **dict(surface.metadata),
                "route": self.route,
                "representation": str(
                    surface.geometry_ref.get(
                        "representation",
                        "surface_sheet" if conductors else "solution_surface",
                    )
                ),
                "interface_type": "_".join(interface_kinds)
                if interface_kinds
                else "boundary",
                "contact_kind": (
                    str(interface.metadata.get("contact_kind", "MM"))
                    if interface is not None
                    and interface.recognition_rule == "coplanar_conductor_contact_patch"
                    else None
                ),
                "face_kind": face_kind,
                # Physical groups can intentionally aggregate split
                # entities under a stable semantic group id.  Keep the
                # exact split owners in source provenance, while the
                # solver-live owner field names the actual group owner.
                "owner_semantic_ids": (
                    owner_ids
                    if any(
                        boundary_id not in structured_owner_ids
                        for boundary_id in surface.metadata.get(
                            "boundary_volume_ids", ()
                        )
                    )
                    else structured_owner_ids
                ),
                "net_id": (
                    self.component_net[next(iter(component_ids))]
                    if component_ids and next(iter(component_ids)) in self.component_net
                    else next(iter(net_ids), None)
                ),
                "conductor_component_id": next(iter(component_ids), None),
                "equipotential_id": (
                    self.component_equipotential[next(iter(component_ids))]
                    if component_ids
                    and next(iter(component_ids)) in self.component_equipotential
                    else next(iter(equipotential_ids), None)
                ),
                "source_provenance": {
                    **dict(existing_source_provenance),
                    "source_polygon_ids": _normalized_source_polygon_ids(
                        self.build_input,
                        owner_ids,
                        surface.geometry_ref,
                        mm_contacts=self.mm_contacts,
                    ),
                    "interface_id": surface.interface_id,
                    "route": self.route,
                    "conductor_source_layer_name": next(iter(source_layer_names), None),
                },
            },
        )


def _normalized_source_polygon_ids(
    build_input: GeometryBuildInput,
    owner_ids: Sequence[str],
    geometry_ref: Mapping[str, Any],
    *,
    mm_contacts: Sequence[MMContactRecord],
) -> tuple[str, ...]:
    """Keep only spatially participating M1 contact provenance."""
    ids: list[str] = list(geometry_ref.get("source_polygon_ids", ()))
    owner_set = set(owner_ids)
    component_id = geometry_ref.get("conductor_component_id")
    entities = {entity.semantic_id: entity for entity in build_input.entities}
    for record in mm_contacts:
        if not is_face_pad_mm_contact(record, entities):
            continue
        if component_id is not None and record.conductor_component_id != component_id:
            continue
        if not owner_set.intersection((record.lower_entity_id, record.upper_entity_id)):
            continue
        ids.extend(record.lower_source_fragment_ids)
        ids.extend(record.upper_source_fragment_ids)
    return unique_ids(ids)
