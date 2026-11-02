"""Audit authored sites against structured ground, keepout and conductor records."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from scgsim.geometry.primitives.grid import polygon_boolean
from scgsim.geometry.source.ground_bump_geometry.layout import (
    gds_layer_polygons,
    gdstk_polygons,
    intersecting_blocker_ids,
    point2,
    polygon_center,
    polygon_points,
)


@dataclass(frozen=True)
class AuthoredSiteAuditContext:
    """Hold the exact source, GDS layers and PDK contract for one audit."""

    build_input: Any
    cell: Any
    bump_layer: tuple[int, int]
    under_bump_layer: tuple[int, int]
    spec: Mapping[str, Any]
    semantic_contract: Mapping[str, Any]


@dataclass(frozen=True)
class StructuredGroundContext:
    """Bind structured entities and polygons to one exact owner contract."""

    owner_ids: tuple[str, ...]
    expected_net: Any
    expected_equipotential: Any
    expected_host: Any
    entities: Mapping[str, Any]
    polygons_by_id: Mapping[str, Any]


@dataclass(frozen=True)
class AuthoredSitesContext:
    """Carry the shared facts used while validating the ordered authored sites."""

    audit: AuthoredSiteAuditContext
    pairs: Sequence[tuple[Any, Any, str, str]]
    ground: Sequence[Any]
    blockers: list[tuple[str, tuple[Any, ...]]]
    owner_ids: tuple[str, ...]
    owner_net: Any
    owner_host: Any


def audit_authored_indium_sites(context: AuthoredSiteAuditContext) -> dict[str, Any]:
    """Resolve structured ownership before checking authored bump footprints."""
    build_input = context.build_input
    polygons_by_id = {polygon.polygon_id: polygon for polygon in build_input.polygons}
    entities = {entity.semantic_id: entity for entity in build_input.entities}
    owner_ids = tuple(context.semantic_contract["owner_semantic_ids"])
    expected_net = context.semantic_contract["net_id"]
    expected_equipotential = context.semantic_contract["equipotential_id"]
    expected_host = context.semantic_contract["host_solution_volume_id"]
    owners, ground = _structured_ground_owners(
        StructuredGroundContext(
            owner_ids,
            expected_net,
            expected_equipotential,
            expected_host,
            entities,
            polygons_by_id,
        )
    )
    owner_net = next(iter({entity.net_id for entity, _ in owners}))
    owner_host = next(iter({entity.host_void_semantic_id for entity, _ in owners}))
    resolved_contract = _resolved_semantic_contract(
        owner_ids, owners, owner_net, expected_equipotential, owner_host
    )
    blockers = _blocking_footprints(context, polygons_by_id, owner_ids, owner_net)
    pairs = _authored_bump_pairs(context, build_input, polygons_by_id)
    authored_sites = _authored_site_records(
        AuthoredSitesContext(
            context, pairs, ground, blockers, owner_ids, owner_net, owner_host
        )
    )
    return {
        "ground_intersection": tuple(ground),
        "blockers": blockers,
        "authored_sites": authored_sites,
        "semantic_contract": resolved_contract,
    }


def _structured_ground_owners(
    context: StructuredGroundContext,
) -> tuple[list[tuple[Any, list[Any]]], Sequence[Any]]:
    owner_ids = context.owner_ids
    owners = []
    for semantic_id in owner_ids:
        entity = context.entities.get(semantic_id)
        if entity is None:
            raise ValueError(f"indium fill requires structured owner {semantic_id!r}.")
        if (
            entity.role != "metal"
            or entity.part_role != "face_metal"
            or entity.net_id != context.expected_net
            or entity.host_void_semantic_id != context.expected_host
            or entity.metadata.get("equipotential_id") != context.expected_equipotential
        ):
            raise ValueError(
                f"structured owner {semantic_id!r} disagrees with the authored bump contract."
            )
        selected = [
            context.polygons_by_id[key]
            for key in entity.polygon_ids
            if key in context.polygons_by_id
        ]
        if len(selected) != len(entity.polygon_ids) or not selected:
            raise ValueError(f"structured owner {semantic_id!r} lacks exact polygons.")
        owners.append((entity, gdstk_polygons(selected)))
    owner_nets = {entity.net_id for entity, _ in owners}
    owner_hosts = {entity.host_void_semantic_id for entity, _ in owners}
    owner_equipotentials = {
        entity.metadata.get("equipotential_id") for entity, _ in owners
    }
    if (
        owner_nets != {context.expected_net}
        or owner_equipotentials != {context.expected_equipotential}
        or owner_hosts != {context.expected_host}
    ):
        raise ValueError(
            "structured ground owners disagree on exact net/equipotential or host."
        )
    ground = polygon_boolean(owners[0][1], owners[1][1], "and", precision=1e-9)
    if not ground:
        raise ValueError("D0/D1 structured Ground intersection is empty.")
    return owners, ground


def _resolved_semantic_contract(
    owner_ids: tuple[str, ...],
    owners: Sequence[tuple[Any, Sequence[Any]]],
    owner_net: Any,
    expected_equipotential: Any,
    owner_host: Any,
) -> dict[str, Any]:
    return {
        "owner_semantic_ids": list(owner_ids),
        "net_id": owner_net,
        "equipotential_id": expected_equipotential,
        "host_solution_volume_id": owner_host,
        "owners": [
            {
                "semantic_id": entity.semantic_id,
                "role": entity.role,
                "part_role": entity.part_role,
                "net_id": entity.net_id,
                "equipotential_id": entity.metadata.get("equipotential_id"),
                "host_solution_volume_id": entity.host_void_semantic_id,
            }
            for entity, _ in owners
        ],
        "source_provenance": "pdk_ground_bump_fill",
    }


def _blocking_footprints(
    context: AuthoredSiteAuditContext,
    polygons_by_id: Mapping[str, Any],
    owner_ids: tuple[str, ...],
    owner_net: Any,
) -> list[tuple[str, tuple[Any, ...]]]:
    return [
        *_pdk_keepout_blockers(context),
        *_structured_conductor_blockers(context, polygons_by_id, owner_ids, owner_net),
    ]


def _pdk_keepout_blockers(
    context: AuthoredSiteAuditContext,
) -> list[tuple[str, tuple[Any, ...]]]:
    blockers: list[tuple[str, tuple[Any, ...]]] = []
    for record in context.spec.get("keepout_records", ()):
        if not isinstance(record, Mapping):
            raise TypeError("public indium keepout records must be mappings.")
        consumers = record.get("consumers", ())
        if isinstance(consumers, str | bytes) or not isinstance(consumers, Sequence):
            raise TypeError("public indium keepout consumers must be a sequence.")
        if "indium_ground_bump" not in consumers:
            continue
        name = record.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("public indium keepout requires an exact source name.")
        layer = point2(record.get("layer"), "keepout layer")
        polygons = tuple(
            gds_layer_polygons(context.cell, (int(layer[0]), int(layer[1])))
        )
        if polygons:
            blockers.append((f"pdk_keepout:{name}", polygons))
    return blockers


def _structured_conductor_blockers(
    context: AuthoredSiteAuditContext,
    polygons_by_id: Mapping[str, Any],
    owner_ids: tuple[str, ...],
    owner_net: Any,
) -> list[tuple[str, tuple[Any, ...]]]:
    blockers: list[tuple[str, tuple[Any, ...]]] = []
    for entity in context.build_input.entities:
        if entity.material_kind != "conductor":
            continue
        if not isinstance(entity.net_id, str) or not entity.net_id:
            raise ValueError(
                f"conductor {entity.semantic_id!r} has unknown net semantics for indium audit."
            )
        if not isinstance(entity.part_role, str) or not entity.part_role:
            raise ValueError(
                f"conductor {entity.semantic_id!r} has unknown part-role semantics for indium audit."
            )
        if (
            not isinstance(entity.host_void_semantic_id, str)
            or not entity.host_void_semantic_id
        ):
            raise ValueError(
                f"conductor {entity.semantic_id!r} has unknown host semantics for indium audit."
            )
        if entity.semantic_id in owner_ids or entity.net_id == owner_net:
            continue
        selected = [
            polygons_by_id[key] for key in entity.polygon_ids if key in polygons_by_id
        ]
        if len(selected) != len(entity.polygon_ids):
            raise ValueError(
                f"conductor {entity.semantic_id!r} polygon records are ambiguous for indium audit."
            )
        polygons = tuple(gdstk_polygons(selected))
        if polygons:
            blockers.append((f"structured_conductor:{entity.semantic_id}", polygons))
    return blockers


def _authored_bump_pairs(
    context: AuthoredSiteAuditContext,
    build_input: Any,
    polygons_by_id: Mapping[str, Any],
) -> list[tuple[Any, Any, str, str]]:
    bump_polygons = gds_layer_polygons(context.cell, context.bump_layer)
    under_polygons = gds_layer_polygons(context.cell, context.under_bump_layer)
    if not bump_polygons and not under_polygons:
        return []
    if len(bump_polygons) != len(under_polygons):
        raise ValueError("authored indium audit requires equal bump and UBM counts.")
    pairs = []
    used_under: set[int] = set()
    for bump in bump_polygons:
        matching = [
            index
            for index, under in enumerate(under_polygons)
            if not polygon_boolean((bump,), (under,), "not", precision=1e-9)
        ]
        if len(matching) != 1 or matching[0] in used_under:
            raise ValueError(
                "each authored bump requires one exact unique UBM footprint."
            )
        used_under.add(matching[0])
        source_entity_id, source_polygon_id = authored_bump_source_identity(
            build_input=build_input,
            bump_layer=context.bump_layer,
            bump=bump,
            polygons_by_id=polygons_by_id,
        )
        pairs.append(
            (bump, under_polygons[matching[0]], source_entity_id, source_polygon_id)
        )
    if len(used_under) != len(under_polygons):
        raise ValueError("each authored UBM must belong to one exact bump site.")
    return pairs


def _authored_site_records(
    context: AuthoredSitesContext,
) -> list[dict[str, Any]]:
    ordered_pairs = sorted(context.pairs, key=lambda item: polygon_center(item[0]))
    authored_sites = []
    for index, (bump, under, source_entity_id, source_polygon_id) in enumerate(
        ordered_pairs
    ):
        site_id = f"AUTHORED_INDIUM_SITE__{index:04d}"
        center = polygon_center(bump)
        footprint = (under,)
        if polygon_boolean(footprint, context.ground, "not", precision=1e-9):
            raise ValueError(f"authored site {site_id!r} is outside D0/D1 Ground.")
        sources = intersecting_blocker_ids(footprint, context.blockers)
        if sources:
            raise ValueError(
                f"authored site {site_id!r} intersects blocking sources {list(sources)!r}."
            )
        _require_nonoverlapping_ubm(index, footprint, ordered_pairs, site_id)
        digest = hashlib.sha256(
            json.dumps(
                {
                    "semantic_id": site_id,
                    "center_um": list(center),
                    "bump": polygon_points(bump),
                    "ubm": polygon_points(under),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        authored_sites.append(
            {
                "semantic_id": site_id,
                "center_um": list(center),
                "owner_semantic_ids": list(context.owner_ids),
                "net_id": context.owner_net,
                "host_solution_volume_id": context.owner_host,
                "bump_layer": list(context.audit.bump_layer),
                "under_bump_layer": list(context.audit.under_bump_layer),
                "geometry_placement_sha256": digest,
                "source_kind": "authored",
                "source_semantic_id": source_entity_id,
                "source_polygon_id": source_polygon_id,
            }
        )
        context.blockers.append((f"authored_site:{site_id}", footprint))
    return authored_sites


def _require_nonoverlapping_ubm(
    index: int,
    footprint: tuple[Any, ...],
    ordered_pairs: Sequence[tuple[Any, Any, str, str]],
    site_id: str,
) -> None:
    for other_index, (_, other_under, _, _) in enumerate(ordered_pairs):
        if index == other_index:
            continue
        if polygon_boolean(footprint, (other_under,), "and", precision=1e-9):
            raise ValueError(
                f"authored site {site_id!r} overlaps another authored UBM."
            )


def authored_bump_source_identity(
    *,
    build_input: Any,
    bump_layer: tuple[int, int],
    bump: Any,
    polygons_by_id: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    """Find the unique structured record owning one authored bump polygon."""
    layer_name = f"{bump_layer[0]}/{bump_layer[1]}"
    source_polygons = polygons_by_id or {
        polygon.polygon_id: polygon for polygon in build_input.polygons
    }
    matches: list[tuple[str, str]] = []
    for entity in build_input.entities:
        if entity.metadata.get("source_kind") != "authored":
            continue
        for polygon_id in entity.polygon_ids:
            polygon = source_polygons.get(polygon_id)
            if polygon is None or polygon.layer != layer_name:
                continue
            candidate = tuple(gdstk_polygons((polygon,)))
            if not polygon_boolean(
                (bump,), candidate, "not", precision=1e-9
            ) and not polygon_boolean(candidate, (bump,), "not", precision=1e-9):
                matches.append((entity.semantic_id, polygon_id))
    if len(matches) != 1:
        raise ValueError(
            "authored bump geometry must resolve to exactly one structured source "
            f"record; got {matches!r}."
        )
    return matches[0]
