"""Recognize authored interface intents and complete surface-derived records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypedDict, cast

from scgsim.geometry.grid.constants import INTERFACE_KIND_ORDER
from scgsim.geometry.grid.entities import intent_supports_route
from scgsim.geometry.models.common import RouteLiteral
from scgsim.geometry.models.input import GeometryBuildInput
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.models.topology import InterfacePlanRecord


class InterfaceIntents(TypedDict, total=False):
    """Describe the three authored interface intent collections."""

    interfaces: Sequence[Mapping[str, Any]]
    metal_metal_contact_edges: Sequence[Mapping[str, Any]]
    metal_ground_contact_patches: Sequence[Mapping[str, Any]]


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
    """
    intents = cast(
        InterfaceIntents, dict(build_input.metadata.get("interface_intents_2d", {}))
    )
    records = _explicit_interface_intents(intents, route, representation_context)
    for index, intent in enumerate(intents.get("metal_metal_contact_edges", ())):
        if not isinstance(intent, Mapping):
            raise TypeError("metal_metal_contact_edges entries must be mappings")
        if not _intent_supports_context_or_route(
            intent, route=route, representation_context=representation_context
        ):
            continue
        owners = tuple((str(owner) for owner in intent.get("owner_semantic_ids", ())))
        if len(owners) != 2:
            raise ValueError(f"invalid MM edge intent owners: {intent!r}")
        records.append(
            InterfacePlanRecord(
                interface_id=f"MM__CONTACT_EDGE__{index:04d}",
                kind="MM",
                owner_semantic_ids=(owners[0], owners[1]),
                recognition_rule="draw_edge_overlaps_ground_mask_cutout_edge",
                source_polygon_ids=tuple(
                    (
                        str(intent[key])
                        for key in ("source_polygon_id", "ground_polygon_id")
                        if key in intent
                    )
                ),
                metadata=dict(intent),
            )
        )
    for index, intent in enumerate(intents.get("metal_ground_contact_patches", ())):
        if not isinstance(intent, Mapping):
            raise TypeError("metal_ground_contact_patches entries must be mappings")
        if not _intent_supports_context_or_route(
            intent, route=route, representation_context=representation_context
        ):
            continue
        owners = tuple((str(owner) for owner in intent.get("owner_semantic_ids", ())))
        if len(owners) != 2:
            raise ValueError(f"invalid contact patch intent owners: {intent!r}")
        records.append(
            InterfacePlanRecord(
                interface_id=f"MM__CONTACT_PATCH__{index:04d}",
                kind="MM",
                owner_semantic_ids=(owners[0], owners[1]),
                recognition_rule="projected_xy_footprint_overlap",
                source_polygon_ids=tuple(
                    (
                        str(intent[key])
                        for key in ("source_polygon_id", "ground_polygon_id")
                        if key in intent
                    )
                ),
                metadata=dict(intent),
            )
        )
    return tuple(records)


def _intent_supports_context_or_route(
    intent: Mapping[str, Any],
    *,
    route: RouteLiteral | None,
    representation_context: RepresentationContext | None,
) -> bool:
    if route is not None:
        return intent_supports_route(intent, route)
    if representation_context is None:
        return False
    if intent.get("intent_origin") == "generated_representation_context_surface_sheet":
        owners = intent.get("owner_semantic_ids")
        if (
            not isinstance(owners, Sequence)
            or isinstance(owners, (str, bytes))
            or (not owners)
        ):
            return False
        owner = representation_context.entities.get(str(owners[0]))
        return owner is not None and owner.representation == "sheet"
    return "valid_routes" not in intent


def _explicit_interface_intents(
    intents: InterfaceIntents,
    route: RouteLiteral | None,
    representation_context: RepresentationContext | None,
) -> list[InterfacePlanRecord]:
    """Read explicit interface records before edge and patch intents."""
    records: list[InterfacePlanRecord] = []
    for index, intent in enumerate(intents.get("interfaces", ())):
        if not isinstance(intent, Mapping):
            raise TypeError("interfaces entries must be mappings")
        if not _intent_supports_context_or_route(
            intent, route=route, representation_context=representation_context
        ):
            continue
        kind = str(intent.get("kind", ""))
        if kind not in INTERFACE_KIND_ORDER:
            raise ValueError(f"invalid interface kind: {kind!r}")
        owners = tuple((str(owner) for owner in intent.get("owner_semantic_ids", ())))
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
                    (str(value) for value in intent.get("source_polygon_ids", ()))
                ),
                metadata=dict(intent),
            )
        )
    return records
