"""Look up the metal-to-metal contact loops that are cut out of conductor faces.

Owns the lookups from contact records to face loops. A contact patch is where two
conductors touch; the touching area is not a free face of either conductor, so
surface planning subtracts it. Does not plan the contact patches themselves; see
``contact_patches`` and ``mm_contacts``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.topology import InterfacePlanRecord, MMContactRecord
from scgsim.geometry.primitives.loops import (
    boolean_gdstk_region,
    canonical_loop_sort_key,
    contact_holes_are_simple,
    gdstk_surface_region,
    geometry_refs_from_gdstk_region,
    normalized_loop,
    same_loop_geometry,
)


def contact_patches_by_entity_face(
    interfaces: tuple[InterfacePlanRecord, ...],
) -> dict[tuple[str, str], tuple[tuple[tuple[float, float], ...], ...]]:
    records: dict[tuple[str, str], list[tuple[tuple[float, float], ...]]] = {}
    for interface in interfaces:
        if interface.recognition_rule != "coplanar_conductor_contact_patch":
            continue
        loop = normalized_loop(interface.metadata["outer_loop"])
        lower_id = str(interface.metadata["lower_entity_id"])
        upper_id = str(interface.metadata["upper_entity_id"])
        lower_face = str(interface.metadata.get("lower_face", "top"))
        upper_face = str(interface.metadata.get("upper_face", "bottom"))
        records.setdefault((lower_id, lower_face), []).append(loop)
        records.setdefault((upper_id, upper_face), []).append(loop)
    return {key: tuple(value) for key, value in records.items()}


def sheet_contact_loops_by_pad(
    build_input: GeometryBuildInput,
    records: Sequence[MMContactRecord],
) -> dict[str, tuple[tuple[tuple[float, float], ...], ...]]:
    """Return finite MM ledger loops for the contact-pad side of an M1 join."""
    entities = {entity.semantic_id: entity for entity in build_input.entities}
    result: dict[str, list[tuple[tuple[float, float], ...]]] = {}
    for record in records:
        if not is_face_pad_mm_contact(record, entities):
            continue
        lower = entities[record.lower_entity_id]
        upper = entities[record.upper_entity_id]
        pad = lower if lower.part_role == "contact_pad" else upper
        result.setdefault(pad.semantic_id, []).append(record.outer_loop)
    return {entity_id: tuple(loops) for entity_id, loops in result.items()}


def is_face_pad_mm_contact(
    record: MMContactRecord,
    entities: Mapping[str, SemanticEntitySpec],
) -> bool:
    return {
        entities[record.lower_entity_id].part_role,
        entities[record.upper_entity_id].part_role,
    } == {"contact_pad", "face_metal"}


def sheet_contact_loops_by_face_metal(
    build_input: GeometryBuildInput,
    records: Sequence[MMContactRecord],
    *,
    include_direct_bump_contacts: bool,
) -> dict[str, tuple[tuple[tuple[float, float], ...], ...]]:
    """Return finite M1 contact loops that require Route-A sheet cap topology."""
    entities = {entity.semantic_id: entity for entity in build_input.entities}
    result: dict[str, list[tuple[tuple[float, float], ...]]] = {}
    for record in records:
        lower = entities[record.lower_entity_id]
        upper = entities[record.upper_entity_id]
        roles = {lower.part_role, upper.part_role}
        if "face_metal" not in roles or (
            roles != {"face_metal", "contact_pad"}
            and not (
                include_direct_bump_contacts and roles == {"face_metal", "bump_body"}
            )
        ):
            continue
        face = lower if lower.part_role == "face_metal" else upper
        result.setdefault(face.semantic_id, []).append(record.outer_loop)
    return {entity_id: tuple(loops) for entity_id, loops in result.items()}


def route_a_sheet_contacts_by_face_metal(
    build_input: GeometryBuildInput,
    records: Sequence[MMContactRecord],
) -> dict[str, tuple[MMContactRecord, ...]]:
    """Project typed finite MM records into Route-A sheet cap authority."""
    entities = {entity.semantic_id: entity for entity in build_input.entities}
    result: dict[str, list[MMContactRecord]] = {}
    for record in records:
        lower = entities[record.lower_entity_id]
        upper = entities[record.upper_entity_id]
        roles = {lower.part_role, upper.part_role}
        if roles not in ({"face_metal", "contact_pad"}, {"face_metal", "bump_body"}):
            continue
        face = lower if lower.part_role == "face_metal" else upper
        result.setdefault(face.semantic_id, []).append(record)
    return {
        entity_id: tuple(
            sorted(
                contacts,
                key=lambda contact: (
                    canonical_loop_sort_key(normalized_loop(contact.outer_loop)),
                    contact.contact_id,
                ),
            )
        )
        for entity_id, contacts in result.items()
    }


def subtract_contact_patches_from_face(
    geometry_ref: Mapping[str, Any],
    contact_loops: Sequence[tuple[tuple[float, float], ...]],
) -> tuple[dict[str, Any], ...]:
    if not contact_loops:
        return (dict(geometry_ref),)
    outer_loop = normalized_loop(geometry_ref["outer_loop"])
    existing_holes = tuple(
        normalized_loop(hole_loop) for hole_loop in geometry_ref.get("hole_loops", ())
    )
    remaining_holes: list[tuple[tuple[float, float], ...]] = list(existing_holes)
    for contact_loop in contact_loops:
        if same_loop_geometry(contact_loop, outer_loop):
            return ()
        remaining_holes.append(normalized_loop(contact_loop))
    simple_ref = {
        **dict(geometry_ref),
        "hole_loops": tuple(remaining_holes),
        "contact_hole_loops": tuple(normalized_loop(loop) for loop in contact_loops),
    }
    if contact_holes_are_simple(outer_loop, remaining_holes):
        return (simple_ref,)

    import gdstk

    live_region = boolean_gdstk_region(
        gdstk,
        gdstk_surface_region(geometry_ref),
        tuple(gdstk.Polygon(loop) for loop in contact_loops),
        "not",
    )
    return geometry_refs_from_gdstk_region(geometry_ref, live_region)
