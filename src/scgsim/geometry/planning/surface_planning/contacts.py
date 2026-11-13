"""Project typed conductor contacts into planar face masks and sheet cap directions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any, cast

from scgsim.geometry.grid.constants import TOPOLOGY_EPS_UM
from scgsim.geometry.grid.grid import boolean_polygon_region
from scgsim.geometry.grid.loops import (
    canonical_loop_sort_key,
    contact_holes_are_simple,
    gdstk_surface_region,
    geometry_refs_from_gdstk_region,
    same_loop_geometry,
)
from scgsim.geometry.grid.planar_loops import clean_loop
from scgsim.geometry.models.common import PolygonRing, Vector3D
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.topology import (
    InterfacePlanRecord,
    MMContactRecord,
)


def contact_patches_by_entity_face(
    interfaces: tuple[InterfacePlanRecord, ...],
) -> dict[tuple[str, str], tuple[tuple[tuple[float, float], ...], ...]]:
    """Index recognized conductor contact loops by entity and face."""
    records: dict[tuple[str, str], list[tuple[tuple[float, float], ...]]] = {}
    for interface in interfaces:
        if interface.recognition_rule != "coplanar_conductor_contact_patch":
            continue
        loop = clean_loop(interface.metadata["outer_loop"])
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
        result.setdefault(pad.semantic_id, []).append(
            cast(PolygonRing, record.outer_loop)
        )
    return {entity_id: tuple(loops) for entity_id, loops in result.items()}


def is_face_pad_mm_contact(
    record: MMContactRecord,
    entities: Mapping[str, SemanticEntitySpec],
) -> bool:
    """Recognize the typed face-metal/contact-pad normalization pair."""
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
        result.setdefault(face.semantic_id, []).append(
            cast(PolygonRing, record.outer_loop)
        )
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
                    canonical_loop_sort_key(
                        clean_loop(cast(PolygonRing, contact.outer_loop))
                    ),
                    contact.contact_id,
                ),
            )
        )
        for entity_id, contacts in result.items()
    }


def route_a_sheet_contact_cap_face(
    build_input: GeometryBuildInput,
    *,
    sheet_entity: SemanticEntitySpec,
    contact: MMContactRecord,
) -> str:
    """Select the sheet side opposite its typed finite MM contact direction."""
    entities = {entity.semantic_id: entity for entity in build_input.entities}
    lower = entities.get(contact.lower_entity_id)
    upper = entities.get(contact.upper_entity_id)
    if (
        lower is None
        or upper is None
        or sheet_entity.semantic_id
        not in {
            contact.lower_entity_id,
            contact.upper_entity_id,
        }
    ):
        raise ValueError(f"{contact.contact_id} has ambiguous sheet contact members")
    normal = tuple(float(value) for value in cast(Vector3D, contact.normal))
    if len(normal) != 3 or not all(isfinite(value) for value in normal):
        raise ValueError(f"{contact.contact_id} has ambiguous sheet contact normal")
    roles = {lower.part_role, upper.part_role}
    if roles == {"face_metal", "contact_pad"}:
        # A typed contact_pad is the same-z under-sheet normalization authority.
        return "bottom"
    if roles != {"face_metal", "bump_body"}:
        raise ValueError(f"{contact.contact_id} has incompatible sheet contact roles")
    if (
        abs(normal[0]) > TOPOLOGY_EPS_UM
        or abs(normal[1]) > TOPOLOGY_EPS_UM
        or abs(normal[2]) <= TOPOLOGY_EPS_UM
    ):
        raise ValueError(f"{contact.contact_id} has ambiguous sheet contact normal")
    if sheet_entity.semantic_id == lower.semantic_id:
        contact_face = "top" if normal[2] > 0 else "bottom"
        source_face_id = contact.lower_source_face_id
        expected_source_face_id = f"{lower.semantic_id}__{contact_face}"
    else:
        contact_face = "bottom" if normal[2] > 0 else "top"
        source_face_id = contact.upper_source_face_id
        expected_source_face_id = f"{upper.semantic_id}__{contact_face}"
    expected_lower_face_id = (
        f"{lower.semantic_id}__{'top' if normal[2] > 0 else 'bottom'}"
    )
    expected_upper_face_id = (
        f"{upper.semantic_id}__{'bottom' if normal[2] > 0 else 'top'}"
    )
    if (
        source_face_id != expected_source_face_id
        or contact.lower_source_face_id != expected_lower_face_id
        or contact.upper_source_face_id != expected_upper_face_id
    ):
        raise ValueError(f"{contact.contact_id} has inconsistent sheet contact face")
    return "bottom" if contact_face == "top" else "top"


def subtract_contact_patches_from_face(
    geometry_ref: Mapping[str, Any],
    contact_loops: Sequence[tuple[tuple[float, float], ...]],
) -> tuple[dict[str, Any], ...]:
    """Remove conductor contact patches from a face geometry reference."""
    if not contact_loops:
        return (dict(geometry_ref),)
    outer_loop = clean_loop(geometry_ref["outer_loop"])
    existing_holes = tuple(
        clean_loop(hole_loop) for hole_loop in geometry_ref.get("hole_loops", ())
    )
    remaining_holes: list[tuple[tuple[float, float], ...]] = list(existing_holes)
    for contact_loop in contact_loops:
        if same_loop_geometry(contact_loop, outer_loop):
            return ()
        remaining_holes.append(clean_loop(contact_loop))
    simple_ref = {
        **dict(geometry_ref),
        "hole_loops": tuple(remaining_holes),
        "contact_hole_loops": tuple(clean_loop(loop) for loop in contact_loops),
    }
    if contact_holes_are_simple(outer_loop, remaining_holes):
        return (simple_ref,)

    import gdstk  # noqa: PLC0415 - optional native polygon dependency

    live_region = boolean_polygon_region(
        gdstk,
        gdstk_surface_region(geometry_ref),
        tuple(gdstk.Polygon(loop) for loop in contact_loops),
        "not",
    )
    return geometry_refs_from_gdstk_region(geometry_ref, live_region)
