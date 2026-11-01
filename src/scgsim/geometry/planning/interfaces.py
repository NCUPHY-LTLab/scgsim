"""Recognize interfaces from authored intents, and backfill interfaces for planned surfaces.

Owns turning ``interface_intents_2d`` metadata into ``InterfacePlanRecord`` objects
before any backend geometry exists, and completing the plan for surfaces that name an
interface nobody recognized. Planning contact patches and metal-metal (MM) records is in
``contact_patches`` and ``mm_contacts``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

from scgsim.geometry._primitives.constants import _INTERFACE_KIND_ORDER
from scgsim.geometry._primitives.entities import _intent_supports_route
from scgsim.geometry.models.common import RouteLiteral
from scgsim.geometry.models.input import GeometryBuildInput
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.models.topology import InterfacePlanRecord, SurfacePlanRecord
from scgsim.geometry.planning.topology import (
    _surface_interface_record_owners,
    _surface_owner_ids,
)


def complete_interface_plan_from_surfaces(
    *,
    interfaces: tuple[InterfacePlanRecord, ...],
    surfaces: tuple[SurfacePlanRecord, ...],
) -> tuple[InterfacePlanRecord, ...]:
    """Backfill InterfacePlan records for planned adjacency surfaces."""
    known = {interface.interface_id for interface in interfaces}
    completed = list(interfaces)
    for surface in surfaces:
        if surface.interface_id is None or surface.interface_id in known:
            continue
        kind = surface.interface_id.split("__", 1)[0]
        if kind not in _INTERFACE_KIND_ORDER:
            raise ValueError(f"invalid surface interface id: {surface.interface_id}")
        owners = _surface_owner_ids(surface)
        record_owners = _surface_interface_record_owners(surface, owners)
        if len(record_owners) != 2:
            raise ValueError(
                f"{surface.surface_id} interface needs two owners, got {owners!r}"
            )
        completed.append(
            InterfacePlanRecord(
                interface_id=surface.interface_id,
                kind=kind,  # type: ignore[arg-type]
                owner_semantic_ids=(record_owners[0], record_owners[1]),
                recognition_rule=str(
                    surface.metadata.get(
                        "recognition_rule",
                        "planned_surface_adjacency",
                    )
                ),
                solver_use=surface.solver_use,
                metadata={
                    "generated_from_surface_id": surface.surface_id,
                    **dict(surface.metadata),
                },
            )
        )
        known.add(surface.interface_id)
    return tuple(completed)


def recognize_route_interfaces(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral | None,
    representation_context: RepresentationContext | None = None,
) -> tuple[InterfacePlanRecord, ...]:
    """Recognize interfaces before creating OCC geometry.

    Route-specific recognizers should own each interface rule: draw/ground
    shared edges, XY footprint overlap contacts, stack domain boundaries, ports,
    and exposed EM boundaries. Current fixture metadata may seed those rules
    through `interface_intents_2d`, including generic `interfaces` entries with
    explicit `kind`.

    Interfaces come in three families, in this order: explicit ``interfaces``,
    ``metal_metal_contact_edges`` and ``metal_ground_contact_patches``. The two contact
    families are always metal-metal (MM).
    """
    intents = dict(build_input.metadata.get("interface_intents_2d", {}))
    records: list[InterfacePlanRecord] = []
    for index, intent in enumerate(intents.get("interfaces", ())):
        if not isinstance(intent, Mapping):
            raise TypeError("interfaces entries must be mappings")
        if not _intent_supports_context_or_route(
            intent, route=route, representation_context=representation_context
        ):
            continue
        kind = str(intent.get("kind", ""))
        if kind not in _INTERFACE_KIND_ORDER:
            raise ValueError(f"invalid interface kind: {kind!r}")
        owners = tuple(str(owner) for owner in intent.get("owner_semantic_ids", ()))
        if len(owners) != 2:
            raise ValueError(f"invalid interface owners: {intent!r}")
        records.append(
            InterfacePlanRecord(
                interface_id=str(
                    intent.get("interface_id") or f"{kind}__INTENT__{index:04d}"
                ),
                kind=kind,
                owner_semantic_ids=(owners[0], owners[1]),
                recognition_rule=str(intent.get("recognition_rule", "explicit")),
                source_polygon_ids=tuple(
                    str(value) for value in intent.get("source_polygon_ids", ())
                ),
                metadata=dict(intent),
            )
        )
    for family in _CONTACT_INTENT_FAMILIES:
        records.extend(
            _contact_intent_records(
                intents.get(family.key, ()),
                family,
                route=route,
                representation_context=representation_context,
            )
        )
    return tuple(records)


class _ContactIntentFamily(NamedTuple):
    """How one kind of contact intent becomes interface records."""

    key: str
    id_prefix: str
    owners_error: str
    recognition_rule: str


_CONTACT_INTENT_FAMILIES = (
    _ContactIntentFamily(
        key="metal_metal_contact_edges",
        id_prefix="MM__CONTACT_EDGE",
        owners_error="invalid MM edge intent owners",
        recognition_rule="draw_edge_overlaps_ground_mask_cutout_edge",
    ),
    _ContactIntentFamily(
        key="metal_ground_contact_patches",
        id_prefix="MM__CONTACT_PATCH",
        owners_error="invalid contact patch intent owners",
        recognition_rule="projected_xy_footprint_overlap",
    ),
)


def _contact_intent_records(
    contact_intents: Sequence[Any],
    family: _ContactIntentFamily,
    *,
    route: RouteLiteral | None,
    representation_context: RepresentationContext | None,
) -> list[InterfacePlanRecord]:
    """Turn one family of contact intents into metal-metal (MM) interface records.

    Each intent names its two owners and, optionally, its source polygons.
    """
    records: list[InterfacePlanRecord] = []
    for index, intent in enumerate(contact_intents):
        if not isinstance(intent, Mapping):
            raise TypeError(f"{family.key} entries must be mappings")
        if not _intent_supports_context_or_route(
            intent, route=route, representation_context=representation_context
        ):
            continue
        owners = tuple(str(owner) for owner in intent.get("owner_semantic_ids", ()))
        if len(owners) != 2:
            raise ValueError(f"{family.owners_error}: {intent!r}")
        records.append(
            InterfacePlanRecord(
                interface_id=f"{family.id_prefix}__{index:04d}",
                kind="MM",
                owner_semantic_ids=(owners[0], owners[1]),
                recognition_rule=family.recognition_rule,
                source_polygon_ids=tuple(
                    str(intent[key])
                    for key in ("source_polygon_id", "ground_polygon_id")
                    if key in intent
                ),
                metadata=dict(intent),
            )
        )
    return records


def _intent_supports_context_or_route(
    intent: Mapping[str, Any],
    *,
    route: RouteLiteral | None,
    representation_context: RepresentationContext | None,
) -> bool:
    if route is not None:
        return _intent_supports_route(intent, route)
    if representation_context is None:
        return False
    if intent.get("intent_origin") == "generated_representation_context_surface_sheet":
        owners = intent.get("owner_semantic_ids")
        if (
            not isinstance(owners, Sequence)
            or isinstance(owners, (str, bytes))
            or not owners
        ):
            return False
        owner = representation_context.entities.get(str(owners[0]))
        return owner is not None and owner.representation == "sheet"
    return "valid_routes" not in intent
