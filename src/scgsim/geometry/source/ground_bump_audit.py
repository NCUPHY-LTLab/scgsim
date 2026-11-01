"""Audit authored indium ground bumps before any fill is generated.

Owns: confirming that the structured ground owners agree, finding the D0/D1 ground
intersection, collecting everything a bump must stay clear of, and pairing each
authored bump body with its under-bump metal (UBM) footprint.

Does not own: the fill lattice or the receipt (`ground_bumps.py`). The PDK defines
the bump shapes and keepout meaning; this module only checks mechanics.

Geometry is compared with gdstk booleans at a precision of 1e-9 um; that is the
tolerance used everywhere in this package for "exactly equal" polygons.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

# A blocker is a named group of polygons a bump footprint must not intersect.
Blocker = tuple[str, tuple[Any, ...]]

_BOOLEAN_PRECISION = 1e-9


def audit_authored_indium_sites(  # noqa: PLR0913 - keyword-only; each argument is one audited input
    *,
    build_input: Any,
    cell: Any,
    bump_layer: tuple[int, int],
    under_bump_layer: tuple[int, int],
    spec: Mapping[str, Any],
    semantic_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit authored sites and semantic ownership before either fill mode.

    Layout geometry only supplies footprint mechanics.  Ground ownership, net,
    role and host remain structured adapter records and ambiguous facts fail
    before a candidate can be emitted.
    """
    polygons_by_id = {polygon.polygon_id: polygon for polygon in build_input.polygons}
    owner_ids = tuple(semantic_contract["owner_semantic_ids"])
    owners = _resolve_ground_owners(build_input, polygons_by_id, semantic_contract)
    ground = _ground_intersection(owners)
    owner_net = next(iter({entity.net_id for entity, _ in owners}))
    owner_host = next(iter({entity.host_void_semantic_id for entity, _ in owners}))
    resolved_contract = {
        "owner_semantic_ids": list(owner_ids),
        "net_id": owner_net,
        "equipotential_id": semantic_contract["equipotential_id"],
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

    blockers = _pdk_keepout_blockers(spec, cell)
    blockers.extend(
        _conductor_blockers(build_input, polygons_by_id, owner_ids, owner_net)
    )
    pairs = _pair_bumps_with_ubm(build_input, cell, bump_layer, under_bump_layer)
    authored_sites = _authored_site_records(
        pairs,
        ground=ground,
        blockers=blockers,
        bump_layer=bump_layer,
        under_bump_layer=under_bump_layer,
        owner_ids=owner_ids,
        owner_net=owner_net,
        owner_host=owner_host,
    )
    return {
        "ground_intersection": tuple(ground),
        "blockers": blockers,
        "authored_sites": authored_sites,
        "semantic_contract": resolved_contract,
    }


def _resolve_ground_owners(
    build_input: Any,
    polygons_by_id: Mapping[str, Any],
    semantic_contract: Mapping[str, Any],
) -> list[tuple[Any, list[Any]]]:
    """Each named ground owner as (entity, gdstk polygons); all must agree."""
    entities = {entity.semantic_id: entity for entity in build_input.entities}
    owner_ids = tuple(semantic_contract["owner_semantic_ids"])
    expected_net = semantic_contract["net_id"]
    expected_equipotential = semantic_contract["equipotential_id"]
    expected_host = semantic_contract["host_solution_volume_id"]
    owners = []
    for semantic_id in owner_ids:
        entity = entities.get(semantic_id)
        if entity is None:
            raise ValueError(f"indium fill requires structured owner {semantic_id!r}.")
        if (
            entity.role != "metal"
            or entity.part_role != "face_metal"
            or entity.net_id != expected_net
            or entity.host_void_semantic_id != expected_host
            or entity.metadata.get("equipotential_id") != expected_equipotential
        ):
            raise ValueError(
                f"structured owner {semantic_id!r} disagrees with the authored bump contract."
            )
        selected = [
            polygons_by_id[key] for key in entity.polygon_ids if key in polygons_by_id
        ]
        if len(selected) != len(entity.polygon_ids) or not selected:
            raise ValueError(f"structured owner {semantic_id!r} lacks exact polygons.")
        owners.append((entity, _gdstk_polygons(selected)))
    owner_nets = {entity.net_id for entity, _ in owners}
    owner_hosts = {entity.host_void_semantic_id for entity, _ in owners}
    owner_equipotentials = {
        entity.metadata.get("equipotential_id") for entity, _ in owners
    }
    if (
        owner_nets != {expected_net}
        or owner_equipotentials != {expected_equipotential}
        or owner_hosts != {expected_host}
    ):
        raise ValueError(
            "structured ground owners disagree on exact net/equipotential or host."
        )
    return owners


def _ground_intersection(owners: Sequence[tuple[Any, list[Any]]]) -> list[Any]:
    """Region where both ground owners (D0 and D1) exist; bumps must sit inside it."""
    ground = _gdstk_boolean(owners[0][1], owners[1][1], "and")
    if not ground:
        raise ValueError("D0/D1 structured Ground intersection is empty.")
    return ground


def _gdstk_boolean(first: Sequence[Any], second: Sequence[Any], operation: str) -> Any:
    """gdstk boolean at the package-wide polygon-equality precision."""
    import gdstk

    return gdstk.boolean(first, second, operation, precision=_BOOLEAN_PRECISION)


def _pdk_keepout_blockers(spec: Mapping[str, Any], cell: Any) -> list[Blocker]:
    """Polygons of PDK keepout layers that list `indium_ground_bump` as a consumer."""
    blockers: list[Blocker] = []
    for record in spec.get("keepout_records", ()):
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
        polygons = tuple(gds_layer_polygons(cell, (int(layer[0]), int(layer[1]))))
        if polygons:
            blockers.append((f"pdk_keepout:{name}", polygons))
    return blockers


def _conductor_blockers(
    build_input: Any,
    polygons_by_id: Mapping[str, Any],
    owner_ids: tuple[str, ...],
    owner_net: str,
) -> list[Blocker]:
    """Every conductor on another net blocks a bump; unknown semantics fail early."""
    blockers: list[Blocker] = []
    for entity in build_input.entities:
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
        polygons = tuple(_gdstk_polygons(selected))
        if polygons:
            blockers.append((f"structured_conductor:{entity.semantic_id}", polygons))
    return blockers


def _pair_bumps_with_ubm(
    build_input: Any,
    cell: Any,
    bump_layer: tuple[int, int],
    under_bump_layer: tuple[int, int],
) -> list[tuple[Any, Any, str, str]]:
    """Pair each authored bump with its one exact UBM as (bump, ubm, entity, polygon)."""
    bump_polygons = gds_layer_polygons(cell, bump_layer)
    under_polygons = gds_layer_polygons(cell, under_bump_layer)
    if not bump_polygons and not under_polygons:
        return []
    if len(bump_polygons) != len(under_polygons):
        raise ValueError("authored indium audit requires equal bump and UBM counts.")
    pairs: list[tuple[Any, Any, str, str]] = []
    used_under: set[int] = set()
    for bump in bump_polygons:
        matching = [
            index
            for index, under in enumerate(under_polygons)
            if not _gdstk_boolean((bump,), (under,), "not")
        ]
        if len(matching) != 1 or matching[0] in used_under:
            raise ValueError(
                "each authored bump requires one exact unique UBM footprint."
            )
        used_under.add(matching[0])
        source_entity_id, source_polygon_id = _authored_bump_source_identity(
            build_input=build_input,
            bump_layer=bump_layer,
            bump=bump,
        )
        pairs.append(
            (bump, under_polygons[matching[0]], source_entity_id, source_polygon_id)
        )
    if len(used_under) != len(under_polygons):
        raise ValueError("each authored UBM must belong to one exact bump site.")
    return pairs


def _authored_site_records(  # noqa: PLR0913 - keyword-only; each argument is one audited fact
    pairs: Sequence[tuple[Any, Any, str, str]],
    *,
    ground: Sequence[Any],
    blockers: list[Blocker],
    bump_layer: tuple[int, int],
    under_bump_layer: tuple[int, int],
    owner_ids: tuple[str, ...],
    owner_net: str,
    owner_host: str,
) -> list[dict[str, Any]]:
    """One record per authored site, in center order.

    Appends each accepted footprint to `blockers`, so a generated fill site later
    avoids it.
    """
    ordered_pairs = sorted(pairs, key=lambda item: _polygon_center(item[0]))
    authored_sites: list[dict[str, Any]] = []
    for index, (bump, under, source_entity_id, source_polygon_id) in enumerate(
        ordered_pairs
    ):
        site_id = f"AUTHORED_INDIUM_SITE__{index:04d}"
        center = _polygon_center(bump)
        footprint = (under,)
        if _gdstk_boolean(footprint, ground, "not"):
            raise ValueError(f"authored site {site_id!r} is outside D0/D1 Ground.")
        sources = intersecting_blocker_ids(footprint, blockers)
        if sources:
            raise ValueError(
                f"authored site {site_id!r} intersects blocking sources {list(sources)!r}."
            )
        for other_index, (_, other_under, _, _) in enumerate(ordered_pairs):
            if index == other_index:
                continue
            if _gdstk_boolean(footprint, (other_under,), "and"):
                raise ValueError(
                    f"authored site {site_id!r} overlaps another authored UBM."
                )
        digest = hashlib.sha256(
            json.dumps(
                {
                    "semantic_id": site_id,
                    "center_um": list(center),
                    "bump": _polygon_points(bump),
                    "ubm": _polygon_points(under),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        authored_sites.append(
            {
                "semantic_id": site_id,
                "center_um": list(center),
                "owner_semantic_ids": list(owner_ids),
                "net_id": owner_net,
                "host_solution_volume_id": owner_host,
                "bump_layer": list(bump_layer),
                "under_bump_layer": list(under_bump_layer),
                "geometry_placement_sha256": digest,
                "source_kind": "authored",
                "source_semantic_id": source_entity_id,
                "source_polygon_id": source_polygon_id,
            }
        )
        blockers.append((f"authored_site:{site_id}", footprint))
    return authored_sites


def _polygon_center(polygon: Any) -> tuple[float, float]:
    """Center of the polygon's axis-aligned bounding box."""
    points = _polygon_points(polygon)
    return (
        (min(point[0] for point in points) + max(point[0] for point in points)) / 2,
        (min(point[1] for point in points) + max(point[1] for point in points)) / 2,
    )


def _polygon_points(polygon: Any) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in polygon.points]


def intersecting_blocker_ids(
    footprint: Sequence[Any], blockers: Sequence[tuple[str, Sequence[Any]]]
) -> tuple[str, ...]:
    """Names of the blockers whose polygons intersect `footprint`."""
    return tuple(
        source_id
        for source_id, polygons in blockers
        if polygons and _gdstk_boolean(footprint, polygons, "and")
    )


def point2(value: Any, name: str) -> tuple[float, float]:
    if (
        isinstance(value, str | bytes)
        or not isinstance(value, Sequence)
        or len(value) != 2
    ):
        raise TypeError(f"{name} must be a two-item numeric sequence.")
    return float(value[0]), float(value[1])


def gdstk_cell(library: Any, name: str) -> Any:
    matches = [cell for cell in library.cells if cell.name == name] if name else []
    if len(matches) == 1:
        return matches[0]
    tops = library.top_level()
    if len(tops) != 1:
        raise ValueError("generated component GDS needs one exact top cell.")
    return tops[0]


def gds_layer_polygons(cell: Any, layer: tuple[int, int]) -> list[Any]:
    return list(
        cell.get_polygons(layer=layer[0], datatype=layer[1], apply_repetitions=True)
    )


def _gdstk_polygons(polygons: Sequence[Any]) -> list[Any]:
    """Convert layout polygons to gdstk polygons, cutting holes out of the exterior."""
    import gdstk

    result = []
    for polygon in polygons:
        outer = [gdstk.Polygon(polygon.exterior)]
        holes = [gdstk.Polygon(hole) for hole in polygon.holes]
        result.extend(
            outer if not holes else gdstk.boolean(outer, holes, "not", precision=1e-9)
        )
    return result


def _authored_bump_source_identity(
    *,
    build_input: Any,
    bump_layer: tuple[int, int],
    bump: Any,
) -> tuple[str, str]:
    """Return the one structured source record that exactly owns a bump.

    This is a lossless source-record lookup: GDS geometry identifies only the
    already-authored structured polygon, never a net, owner, or host.
    """
    import gdstk

    layer_name = f"{bump_layer[0]}/{bump_layer[1]}"
    polygons_by_id = {polygon.polygon_id: polygon for polygon in build_input.polygons}
    matches: list[tuple[str, str]] = []
    for entity in build_input.entities:
        if entity.metadata.get("source_kind") != "authored":
            continue
        for polygon_id in entity.polygon_ids:
            polygon = polygons_by_id.get(polygon_id)
            if polygon is None or polygon.layer != layer_name:
                continue
            candidate = tuple(_gdstk_polygons((polygon,)))
            if not gdstk.boolean(
                (bump,), candidate, "not", precision=1e-9
            ) and not gdstk.boolean(candidate, (bump,), "not", precision=1e-9):
                matches.append((entity.semantic_id, polygon_id))
    if len(matches) != 1:
        raise ValueError(
            "authored bump geometry must resolve to exactly one structured source "
            f"record; got {matches!r}."
        )
    return matches[0]
