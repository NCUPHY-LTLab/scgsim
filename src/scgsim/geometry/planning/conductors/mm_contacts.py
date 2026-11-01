"""Build metal-metal (MM) contact records and validate volumetric conductor overlaps.

Two conductors may touch only through an explicit contact; an unexplained volumetric
overlap is rejected here so later stages never see ambiguous ownership.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any, NamedTuple

from scgsim.geometry.models.common import RouteLiteral
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.models.topology import InterfacePlanRecord, MMContactRecord
from scgsim.geometry.planning.conductors.contact_patches import contact_patch_loops
from scgsim.geometry.planning.conductors.faces import route_conductor_contact_footprints
from scgsim.geometry.planning.solutions.occupancy import entity_occupied_region
from scgsim.geometry.primitives.constants import TOPOLOGY_EPS_UM
from scgsim.geometry.primitives.entities import entity_z_range_um
from scgsim.geometry.primitives.entity_validation import (
    resolve_contact_pad_attachment,
)
from scgsim.geometry.primitives.loops import (
    boolean_gdstk_region,
    normalized_loop,
    polygon_area,
)
from scgsim.geometry.primitives.spatial import is_same_z


def plan_mm_contact_records(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral | None,
    interfaces: tuple[InterfacePlanRecord, ...],
    representation_context: RepresentationContext | None = None,
) -> tuple[tuple[InterfacePlanRecord, ...], tuple[MMContactRecord, ...]]:
    """Turn finite-area same-net contact into hidden component provenance.

    Gmsh/OCC must never infer this graph from coincident bodies.  The contact
    record remains present for both Route A and Route B while its internal face
    is deliberately absent from solver physical groups.
    """
    enriched, records, _ = plan_mm_contact_records_with_components(
        build_input,
        route=route,
        interfaces=interfaces,
        representation_context=representation_context,
    )
    return enriched, records


def plan_mm_contact_records_with_components(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral | None,
    interfaces: tuple[InterfacePlanRecord, ...],
    representation_context: RepresentationContext | None = None,
) -> tuple[
    tuple[InterfacePlanRecord, ...],
    tuple[MMContactRecord, ...],
    tuple[dict[str, Any], ...],
]:
    """Return MM contacts and their single source-owned component ledger.

    Conductors that touch (by an explicit contact patch or by a normalized overlap)
    form one electrical component, which must carry a single net and equipotential.
    """
    live_ids = {
        entity.semantic_id
        for entity in route_conductor_contact_footprints(
            build_input, route, representation_context=representation_context
        )
    }
    # Component ownership retains the original Entity and every source polygon,
    # rather than whichever temporary contact footprint was processed last.
    entities = {
        entity.semantic_id: entity
        for entity in build_input.entities
        if entity.semantic_id in live_ids
    }
    contacts = _same_net_contact_pairs(interfaces, entities)
    normalizations = _validate_volumetric_conductor_contacts(
        build_input,
        route=route,
        entities=entities,
        representation_context=representation_context,
    )
    components = _group_conductors_into_components(entities, contacts, normalizations)
    records, enriched = _interface_contact_records(
        interfaces,
        entities,
        components,
        route=route,
        representation_context=representation_context,
    )
    records.extend(_normalized_contact_records(normalizations, components, route=route))
    records_tuple = tuple(records)
    return (
        tuple(enriched),
        records_tuple,
        _component_ledger(components, records_tuple),
    )


def _same_net_contact_pairs(
    interfaces: Sequence[InterfacePlanRecord],
    entities: Mapping[str, SemanticEntitySpec],
) -> list[tuple[SemanticEntitySpec, SemanticEntitySpec]]:
    """Return the lower and upper conductor of each contact patch, requiring one net.

    A contact between conductors on different nets would short them, so it is rejected.
    """
    contacts: list[tuple[SemanticEntitySpec, SemanticEntitySpec]] = []
    for interface in interfaces:
        if interface.recognition_rule != "coplanar_conductor_contact_patch":
            continue
        lower_id, upper_id = interface.owner_semantic_ids
        lower = entities.get(lower_id)
        upper = entities.get(upper_id)
        if lower is None or upper is None:
            raise ValueError(f"{interface.interface_id} references non-live conductors")
        if not lower.net_id or not upper.net_id:
            raise ValueError(f"{interface.interface_id} contact requires resolved nets")
        if lower.net_id != upper.net_id:
            raise ValueError(
                f"{interface.interface_id} shorts different nets "
                f"{lower.net_id!r} and {upper.net_id!r}"
            )
        contacts.append((lower, upper))
    return contacts


class _ConductorComponents(NamedTuple):
    """Electrical components: which component each conductor belongs to, and its net."""

    component_ids: dict[str, str]
    # component id -> (net id, equipotential id); None when the members declare none.
    values: dict[str, tuple[str | None, str | None]]


def _group_conductors_into_components(
    entities: Mapping[str, SemanticEntitySpec],
    contacts: Sequence[tuple[SemanticEntitySpec, SemanticEntitySpec]],
    normalizations: Sequence[tuple[SemanticEntitySpec, SemanticEntitySpec, Any]],
) -> _ConductorComponents:
    """Union touching conductors into ``COMP__<root id>`` components and read their nets."""
    parent = {entity_id: entity_id for entity_id in entities}

    def find(entity_id: str) -> str:
        while parent[entity_id] != entity_id:
            parent[entity_id] = parent[parent[entity_id]]
            entity_id = parent[entity_id]
        return entity_id

    def union(first: str, second: str) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            # The smaller id becomes the root, so component ids do not depend on order.
            parent[max(first_root, second_root)] = min(first_root, second_root)

    for lower, upper in contacts:
        union(lower.semantic_id, upper.semantic_id)
    for lower, upper, _ in normalizations:
        union(lower.semantic_id, upper.semantic_id)

    component_ids = {entity_id: f"COMP__{find(entity_id)}" for entity_id in entities}
    component_values: dict[str, tuple[str | None, str | None]] = {}
    for component_id in set(component_ids.values()):
        members = [
            entity
            for entity_id, entity in entities.items()
            if component_ids[entity_id] == component_id
        ]
        nets = {entity.net_id for entity in members if entity.net_id}
        equipotentials = {
            str(entity.metadata["equipotential_id"])
            for entity in members
            if entity.metadata.get("equipotential_id") is not None
        }
        if len(nets) > 1 or len(equipotentials) > 1:
            raise ValueError(f"{component_id} has conflicting net/equipotential ids")
        component_values[component_id] = (
            next(iter(nets), None),
            next(iter(equipotentials), None),
        )
    return _ConductorComponents(component_ids, component_values)


def _interface_contact_records(
    interfaces: Sequence[InterfacePlanRecord],
    entities: Mapping[str, SemanticEntitySpec],
    components: _ConductorComponents,
    *,
    route: RouteLiteral | None,
    representation_context: RepresentationContext | None,
) -> tuple[list[MMContactRecord], list[InterfacePlanRecord]]:
    """Make a contact record per contact-patch interface and mark the interface hidden.

    Returns the records and the interfaces in their input order, with the contact
    patches enriched by their component, net and ``hidden_solver_contact`` flag.
    """
    records: list[MMContactRecord] = []
    enriched: list[InterfacePlanRecord] = []
    for interface in interfaces:
        if interface.recognition_rule != "coplanar_conductor_contact_patch":
            enriched.append(interface)
            continue
        lower_id, upper_id = interface.owner_semantic_ids
        lower, upper = entities[lower_id], entities[upper_id]
        loop = normalized_loop(interface.metadata["outer_loop"])
        component_id = components.component_ids[lower_id]
        component_net, component_equipotential = components.values[component_id]
        contact_id = str(interface.metadata.get("contact_id") or interface.interface_id)
        lower_face = str(interface.metadata.get("lower_face", "top"))
        upper_face = str(interface.metadata.get("upper_face", "bottom"))
        records.append(
            MMContactRecord(
                contact_id=contact_id,
                lower_entity_id=lower_id,
                upper_entity_id=upper_id,
                lower_source_face_id=f"{lower_id}__{lower_face}",
                upper_source_face_id=f"{upper_id}__{upper_face}",
                lower_source_fragment_ids=lower.polygon_ids,
                upper_source_fragment_ids=upper.polygon_ids,
                outer_loop=loop,
                area_um2=abs(polygon_area(loop)),
                normal=(0.0, 0.0, 1.0),
                conductor_component_id=component_id,
                net_id=component_net,
                equipotential_id=component_equipotential,
                layer_provenance={
                    "lower": _source_layer_provenance(lower),
                    "upper": _source_layer_provenance(upper),
                },
                material_provenance={
                    "lower": lower.material_id,
                    "upper": upper.material_id,
                },
                source_provenance={
                    "interface_id": interface.interface_id,
                    "route": route,
                    "recognition_rule": interface.recognition_rule,
                    "source_polygon_ids": interface.source_polygon_ids,
                },
            )
        )
        enriched.append(
            replace(
                interface,
                metadata={
                    **dict(interface.metadata),
                    "conductor_component_id": component_id,
                    "net_id": component_net,
                    "equipotential_id": component_equipotential,
                    "hidden_solver_contact": route in {"A", "B"}
                    or (route is None and representation_context is not None),
                },
            )
        )
    return records, enriched


def _normalized_contact_records(
    normalizations: Sequence[
        tuple[
            SemanticEntitySpec,
            SemanticEntitySpec,
            tuple[tuple[tuple[float, float], ...], ...],
        ]
    ],
    components: _ConductorComponents,
    *,
    route: RouteLiteral | None,
) -> list[MMContactRecord]:
    """Make a contact record for each loop where a pad overlaps its face metal in z."""
    records: list[MMContactRecord] = []
    for lower, upper, loops in normalizations:
        component_id = components.component_ids[lower.semantic_id]
        for index, loop in enumerate(loops):
            records.append(
                MMContactRecord(
                    contact_id=(
                        f"NORMALIZED__{lower.semantic_id}__{upper.semantic_id}__"
                        f"{index:04d}"
                    ),
                    lower_entity_id=lower.semantic_id,
                    upper_entity_id=upper.semantic_id,
                    lower_source_face_id=f"{lower.semantic_id}__normalized_overlap",
                    upper_source_face_id=f"{upper.semantic_id}__normalized_overlap",
                    lower_source_fragment_ids=lower.polygon_ids,
                    upper_source_fragment_ids=upper.polygon_ids,
                    outer_loop=loop,
                    area_um2=abs(polygon_area(loop)),
                    normal=(0.0, 0.0, 1.0),
                    conductor_component_id=component_id,
                    net_id=components.values[component_id][0],
                    equipotential_id=components.values[component_id][1],
                    layer_provenance={
                        "lower": _source_layer_provenance(lower),
                        "upper": _source_layer_provenance(upper),
                    },
                    material_provenance={
                        "lower": lower.material_id,
                        "upper": upper.material_id,
                    },
                    source_provenance={
                        "route": route,
                        "source_polygon_ids": (*lower.polygon_ids, *upper.polygon_ids),
                    },
                )
            )
    return records


def _component_ledger(
    components: _ConductorComponents,
    records: Sequence[MMContactRecord],
) -> tuple[dict[str, Any], ...]:
    """List each component with its sorted members, contact ids, net and equipotential."""
    return tuple(
        {
            "conductor_component_id": component_id,
            "members": tuple(
                sorted(
                    entity_id
                    for entity_id, owner_component_id in components.component_ids.items()
                    if owner_component_id == component_id
                )
            ),
            "contact_ids": tuple(
                sorted(
                    record.contact_id
                    for record in records
                    if record.conductor_component_id == component_id
                )
            ),
            "net_id": components.values[component_id][0],
            "equipotential_id": components.values[component_id][1],
        }
        for component_id in sorted(components.values)
    )


def _source_layer_provenance(entity: SemanticEntitySpec) -> dict[str, Any]:
    """Exact authored source layer identity; no layer/datatype collapsing."""
    return {
        "source_layer_name": entity.metadata.get("source_layer_name"),
        "gds_layer": entity.geometry.get("gds_layer"),
        "gds_datatype": entity.geometry.get("gds_datatype"),
    }


def _validate_volumetric_conductor_contacts(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral | None,
    entities: Mapping[str, SemanticEntitySpec],
    representation_context: RepresentationContext | None = None,
) -> tuple[
    tuple[
        SemanticEntitySpec,
        SemanticEntitySpec,
        tuple[tuple[tuple[float, float], ...], ...],
    ],
    ...,
]:
    """Reject body overlap without explicit stacked-contact normalization."""
    import gdstk

    footprint_regions: dict[str, list[Any]] = {}
    for footprint in route_conductor_contact_footprints(
        build_input, route, representation_context=representation_context
    ):
        footprint_regions.setdefault(footprint.semantic_id, []).extend(
            entity_occupied_region(gdstk, footprint)
        )
    ordered = tuple(sorted(entities.values(), key=lambda entity: entity.semantic_id))
    normalized: list[
        tuple[
            SemanticEntitySpec,
            SemanticEntitySpec,
            tuple[tuple[tuple[float, float], ...], ...],
        ]
    ] = []
    # An authored contact_pad is not merely a label: it must normalize a real,
    # finite same-z overlap with the explicitly attached face-metal entity (or
    # semantic group).  A disconnected pad would otherwise be silently kept
    # as an unrelated PEC body under an attachment provenance claim.
    for pad in ordered:
        if pad.part_role != "contact_pad":
            continue
        resolve_contact_pad_attachment(pad, ordered)
    for index, lower in enumerate(ordered):
        lower_region = tuple(footprint_regions[lower.semantic_id])
        lower_min, lower_max = entity_z_range_um(lower)
        for upper in ordered[index + 1 :]:
            upper_min, upper_max = entity_z_range_um(upper)
            if min(lower_max, upper_max) - max(lower_min, upper_min) <= TOPOLOGY_EPS_UM:
                continue
            overlap = boolean_gdstk_region(
                gdstk, lower_region, tuple(footprint_regions[upper.semantic_id]), "and"
            )
            if not overlap:
                continue
            validate_volumetric_overlap_ownership(lower, upper, ordered)
            normalized.append(
                (
                    lower,
                    upper,
                    contact_patch_loops(overlap, lower.semantic_id, upper.semantic_id),
                )
            )
    return tuple(normalized)


def validate_volumetric_overlap_ownership(
    lower: SemanticEntitySpec,
    upper: SemanticEntitySpec,
    entities: Sequence[SemanticEntitySpec],
    *,
    finite_overlap: Callable[[SemanticEntitySpec, SemanticEntitySpec], bool]
    | None = None,
) -> None:
    """Require explicit stacked-contact ownership of actual body overlap."""
    if not lower.net_id or not upper.net_id:
        raise ValueError(
            f"{lower.semantic_id}/{upper.semantic_id} volumetric overlap "
            "requires resolved nets"
        )
    if lower.net_id != upper.net_id:
        raise ValueError(
            f"{lower.semantic_id}/{upper.semantic_id} volumetric overlap "
            "shorts different nets"
        )
    if not _has_explicit_volumetric_normalization(
        lower, upper, entities, finite_overlap=finite_overlap
    ):
        raise ValueError(
            f"{lower.semantic_id}/{upper.semantic_id} volumetric overlap "
            "requires explicit same-net UBM/M1/In normalization provenance"
        )


def _has_explicit_volumetric_normalization(
    lower: SemanticEntitySpec,
    upper: SemanticEntitySpec,
    entities: Sequence[SemanticEntitySpec],
    *,
    finite_overlap: Callable[[SemanticEntitySpec, SemanticEntitySpec], bool]
    | None = None,
) -> bool:
    lower_z_min, lower_z_max = entity_z_range_um(lower)
    upper_z_min, upper_z_max = entity_z_range_um(upper)
    if not (
        is_same_z(lower_z_min, upper_z_min) and is_same_z(lower_z_max, upper_z_max)
    ):
        return False
    return any(
        pad.part_role == "contact_pad"
        and face.part_role == "face_metal"
        and resolve_contact_pad_attachment(pad, entities, finite_overlap=finite_overlap)
        is face
        for pad, face in ((lower, upper), (upper, lower))
    )


def is_fully_normalized_face_metal(
    build_input: GeometryBuildInput, entity: SemanticEntitySpec
) -> bool:
    if entity.part_role != "face_metal":
        return False
    import gdstk

    face = entity_occupied_region(gdstk, entity)
    pads = tuple(
        entity_occupied_region(gdstk, pad)
        for pad in build_input.entities
        if pad.part_role == "contact_pad"
        and resolve_contact_pad_attachment(pad, build_input.entities) is entity
    )
    covered = boolean_gdstk_region(
        gdstk, face, tuple(polygon for region in pads for polygon in region), "not"
    )
    return not covered
