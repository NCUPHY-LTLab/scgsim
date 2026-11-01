"""Attach the surface contract metadata that downstream stages read from planned surfaces.

Runs after every surface of a route is planned, so it can describe contacts and
component provenance for the complete set. Does not create or remove surfaces.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, NamedTuple

from scgsim.geometry.models.common import RouteLiteral
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.topology import (
    InterfacePlanRecord,
    MMContactRecord,
    SurfacePlanRecord,
)
from scgsim.geometry.planning.conductors.face_contacts import is_face_pad_mm_contact
from scgsim.geometry.planning.context.tags import read_surface_physical_owner_ids
from scgsim.geometry.planning.context.topology import surface_owner_ids
from scgsim.geometry.primitives.entities import component_metadata, unique_ids
from scgsim.geometry.primitives.entity_validation import is_solution_entity


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
    # Context planning supplies the complete ledger from the same contact-graph
    # pass, including components without a planar MM face. Other routes retain
    # their MM-record projection. Never infer another graph from surface adjacency.
    component_ledger = (
        _conductor_component_ledger
        if _conductor_component_ledger is not None
        else component_metadata(mm_contacts)
    )
    lookups = _ContractLookups(
        build_input=build_input,
        route=route,
        mm_contacts=mm_contacts,
        entities={entity.semantic_id: entity for entity in build_input.entities},
        interface_by_id={interface.interface_id: interface for interface in interfaces},
        component_by_entity={
            entity_id: str(component["conductor_component_id"])
            for component in component_ledger
            for entity_id in component["members"]
        },
        component_net={
            str(component["conductor_component_id"]): component["net_id"]
            for component in component_ledger
        },
        component_equipotential={
            str(component["conductor_component_id"]): component["equipotential_id"]
            for component in component_ledger
        },
    )
    return tuple(_surface_with_contract(surface, lookups) for surface in surfaces)


@dataclass(frozen=True)
class _ContractLookups:
    """Lookups shared by every surface while its contract metadata is attached."""

    build_input: GeometryBuildInput
    route: RouteLiteral | None
    mm_contacts: Sequence[MMContactRecord]
    entities: Mapping[str, SemanticEntitySpec]
    interface_by_id: Mapping[str, InterfacePlanRecord]
    component_by_entity: Mapping[str, str]
    component_net: Mapping[str, Any]
    component_equipotential: Mapping[str, Any]


class _ConductorIdentity(NamedTuple):
    """The nets, components and equipotentials of the conductors a surface touches."""

    net_ids: set[str]
    component_ids: set[str]
    equipotential_ids: set[str]


def _surface_with_contract(
    surface: SurfacePlanRecord, lookups: _ContractLookups
) -> SurfacePlanRecord:
    """Return the surface with route, face kind, net, component and provenance filled in."""
    owner_ids = surface_owner_ids(surface)
    structured_owner_ids = read_surface_physical_owner_ids(surface)
    conductors = tuple(
        lookups.entities[owner_id]
        for owner_id in owner_ids
        if owner_id in lookups.entities
        and not is_solution_entity(lookups.entities[owner_id])
    )
    identity = _conductor_identity(surface, conductors, lookups)
    component_ids = identity.component_ids
    interface = lookups.interface_by_id.get(surface.interface_id or "")
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
            "route": lookups.route,
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
                    for boundary_id in surface.metadata.get("boundary_volume_ids", ())
                )
                else structured_owner_ids
            ),
            "net_id": (
                lookups.component_net[next(iter(component_ids))]
                if component_ids and next(iter(component_ids)) in lookups.component_net
                else next(iter(identity.net_ids), None)
            ),
            "conductor_component_id": next(iter(component_ids), None),
            "equipotential_id": (
                lookups.component_equipotential[next(iter(component_ids))]
                if component_ids
                and next(iter(component_ids)) in lookups.component_equipotential
                else next(iter(identity.equipotential_ids), None)
            ),
            "source_provenance": {
                **dict(existing_source_provenance),
                "source_polygon_ids": _normalized_source_polygon_ids(
                    lookups.build_input,
                    owner_ids,
                    surface.geometry_ref,
                    mm_contacts=lookups.mm_contacts,
                ),
                "interface_id": surface.interface_id,
                "route": lookups.route,
                "conductor_source_layer_name": next(iter(source_layer_names), None),
            },
        },
    )


def _conductor_identity(
    surface: SurfacePlanRecord,
    conductors: Sequence[SemanticEntitySpec],
    lookups: _ContractLookups,
) -> _ConductorIdentity:
    """Collect net, component and equipotential ids; a surface may have only one of each."""
    net_ids = {entity.net_id for entity in conductors if entity.net_id}
    if len(net_ids) > 1:
        raise ValueError(f"{surface.surface_id} has ambiguous conductor net")
    component_ids = {
        lookups.component_by_entity[entity.semantic_id]
        for entity in conductors
        if entity.semantic_id in lookups.component_by_entity
    }
    if conductors and not component_ids:
        component_ids = {f"COMP__{conductors[0].semantic_id}"}
    if len(component_ids) > 1:
        raise ValueError(f"{surface.surface_id} spans independent conductor components")
    equipotential_ids = {
        str(entity.metadata["equipotential_id"])
        for entity in conductors
        if entity.metadata.get("equipotential_id") is not None
    }
    if len(equipotential_ids) > 1:
        raise ValueError(f"{surface.surface_id} has ambiguous equipotential id")
    return _ConductorIdentity(net_ids, component_ids, equipotential_ids)


def _normalized_source_polygon_ids(
    build_input: GeometryBuildInput,
    owner_ids: Sequence[str],
    geometry_ref: Mapping[str, Any],
    *,
    mm_contacts: Sequence[MMContactRecord],
) -> tuple[str, ...]:
    """Keep only spatially participating M1 contact provenance."""
    ids = list(geometry_ref.get("source_polygon_ids", ()))
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
