"""Lower declared port sheets (lumped ports drawn between two face-metal hosts) to surfaces.

Owns reading and validating a port sheet region and building its ``lumped_port``
surface. Cutting the port out of the host surfaces is Route-specific and lives in
``port_sheets_route_a`` and ``port_sheets_route_b``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Any, NamedTuple

from scgsim.geometry._primitives.entities import _component_metadata, _unique_ids
from scgsim.geometry._primitives.entity_validation import _is_solution_entity
from scgsim.geometry._primitives.loops import _loops_share_edge_overlap
from scgsim.geometry._primitives.spatial import _same_z
from scgsim.geometry.models.common import RouteLiteral
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.regions import PortSheetRegionRecord
from scgsim.geometry.models.topology import MMContactRecord, SurfacePlanRecord
from scgsim.geometry.planning.port_sheets_route_a import (
    _carve_route_a_port_sheet_from_host_plane,
    _port_sheet_remainder_geometry,
    _route_a_port_sheet_sheet_contract,
    _route_a_port_sheet_vacuum_volume_id,
)
from scgsim.geometry.planning.port_sheets_route_b import (
    _partition_route_b_port_sheet_sidewalls,
)


def _lower_port_sheet_regions(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    mm_contacts: tuple[MMContactRecord, ...],
    planned_surfaces: Sequence[SurfacePlanRecord] = (),
) -> tuple[SurfacePlanRecord, ...]:
    """Lower the one accepted Palace port-sheet shape before canonical topology.

    A port sheet is a lumped port drawn as a flat region between two face-metal hosts.
    Each region is validated, turned into one ``lumped_port`` surface, and then cut
    out of the surfaces planned before it. Route A carves the host plane; Route B
    partitions the host cutout sidewalls. Regions are lowered in declaration order
    because each one changes the surfaces the next one sees.
    """
    entities = {entity.semantic_id: entity for entity in build_input.entities}
    polygons = {polygon.polygon_id: polygon for polygon in build_input.polygons}
    component_by_entity = {
        entity_id: str(component["conductor_component_id"])
        for component in _component_metadata(mm_contacts)
        for entity_id in component["members"]
    }
    records: list[SurfacePlanRecord] = list(planned_surfaces)
    for region in build_input.port_sheet_regions:
        port = _read_port_sheet(
            region, route=route, entities=entities, polygons=polygons
        )
        plane = _port_sheet_host_plane(
            port, route=route, entities=entities, planned_surfaces=planned_surfaces
        )
        port_surface = _port_sheet_surface(
            port,
            route=route,
            plane=plane,
            component_by_entity=component_by_entity,
        )
        if route == "A":
            records = _carve_route_a_port_sheet_from_host_plane(
                records,
                port_surface=port_surface,
            )
        elif records:
            records = _partition_route_b_port_sheet_sidewalls(
                records,
                port_surface=port_surface,
                remainder=port.remainder,
                overlaps_by_host={
                    overlap.host_semantic_id: overlap for overlap in region.overlaps
                },
            )
        records.append(port_surface)
    return tuple(records)


@dataclass(frozen=True)
class _PortSheet:
    """A port sheet region whose metadata, hosts and remainder passed validation."""

    region: PortSheetRegionRecord
    source_name: str
    target_layer: str
    port_index: int
    direction: tuple[float, float, float]
    raw_direction: tuple[float, float, float]
    sign_convention: str
    host_ids: tuple[str, ...]
    host_entities: tuple[SemanticEntitySpec, ...]
    # The z of the host plane; both hosts must agree.
    z_values: tuple[float, ...]
    # Route B only: host metal thickness, empty for Route A.
    thickness_values: tuple[float, ...]
    host_void_ids: tuple[str | None, ...]
    # The port region left after the host overlaps are removed: outer and hole loops.
    remainder: Mapping[str, Any]


def _read_port_sheet(
    region: PortSheetRegionRecord,
    *,
    route: RouteLiteral,
    entities: Mapping[str, SemanticEntitySpec],
    polygons: Mapping[str, Any],
) -> _PortSheet:
    """Validate one port sheet region in a fixed order and gather its hosts.

    The order decides which error a user sees first, so do not reorder the checks.
    """
    metadata = region.metadata
    source_name = _port_sheet_string(metadata, "source_name", region)
    target_layer = _port_sheet_string(metadata, "target_layer", region)
    port_index = metadata.get("port_index")
    if (
        isinstance(port_index, bool)
        or not isinstance(port_index, int)
        or port_index < 1
    ):
        raise ValueError(f"{region.port_sheet_id} requires 1-based port_index")
    direction = _port_sheet_direction(region)
    raw_direction = _port_sheet_raw_direction(region)
    sign_convention = _port_sheet_string(metadata, "direction_sign_convention", region)
    host_ids, host_entities = _port_sheet_hosts(
        region, route=route, target_layer=target_layer, entities=entities
    )
    z_values = tuple(
        float(host.geometry.get("z_um", float("nan"))) for host in host_entities
    )
    if not all(isfinite(value) for value in z_values) or not _same_z(
        z_values[0], z_values[1]
    ):
        raise ValueError(f"{region.port_sheet_id} hosts require the same finite z_um")
    thickness_values: tuple[float, ...] = ()
    if route == "B":
        thickness_values = tuple(
            float(host.geometry.get("thickness_um", float("nan")))
            for host in host_entities
        )
        if (
            not all(isfinite(value) for value in thickness_values)
            or any(value <= 0.0 for value in thickness_values)
            or not _same_z(thickness_values[0], thickness_values[1])
        ):
            raise ValueError(
                f"{region.port_sheet_id} Route B hosts require the same "
                "finite positive thickness_um"
            )
    host_void_ids = tuple(host.host_void_semantic_id for host in host_entities)
    if route == "B" and (
        any(not host_void_id for host_void_id in host_void_ids)
        or len(set(host_void_ids)) != 1
        or host_void_ids[0] not in entities
        or not _is_solution_entity(entities[str(host_void_ids[0])])
    ):
        raise ValueError(
            f"{region.port_sheet_id} hosts require one common host_void solution"
        )
    if any(overlap.host_polygon_id not in polygons for overlap in region.overlaps):
        raise ValueError(f"{region.port_sheet_id} references an unknown host polygon")
    remainder = _port_sheet_remainder_geometry(region)
    if not all(
        _port_sheet_boundary_shares_segment(remainder, overlap.overlap_loop)
        for host_id in host_ids
        for overlap in region.overlaps
        if overlap.host_semantic_id == host_id
    ):
        raise ValueError(
            f"{region.port_sheet_id} remainder must share a finite segment "
            "with each host"
        )
    return _PortSheet(
        region=region,
        source_name=source_name,
        target_layer=target_layer,
        port_index=port_index,
        direction=direction,
        raw_direction=raw_direction,
        sign_convention=sign_convention,
        host_ids=host_ids,
        host_entities=host_entities,
        z_values=z_values,
        thickness_values=thickness_values,
        host_void_ids=host_void_ids,
        remainder=remainder,
    )


def _port_sheet_hosts(
    region: PortSheetRegionRecord,
    *,
    route: RouteLiteral,
    target_layer: str,
    entities: Mapping[str, SemanticEntitySpec],
) -> tuple[tuple[str, ...], tuple[SemanticEntitySpec, ...]]:
    """Return the two distinct face-metal hosts a port sheet joins, as ids and entities."""
    if len(region.overlaps) != 2:
        raise ValueError(
            f"{region.port_sheet_id} requires exactly two host overlap records"
        )
    host_ids = _unique_ids(overlap.host_semantic_id for overlap in region.overlaps)
    if len(host_ids) != 2:
        raise ValueError(
            f"{region.port_sheet_id} requires one overlap per distinct face_metal host"
        )
    hosts = tuple(entities.get(host_id) for host_id in host_ids)
    if any(host is None for host in hosts):
        raise ValueError(f"{region.port_sheet_id} references an unknown port host")
    host_entities = tuple(host for host in hosts if host is not None)
    if any(
        host.part_role != "face_metal"
        or not _port_sheet_target_layer_matches(host, target_layer)
        # The build input is projected to one route, so only that key exists.
        or host.route_representations.get(route)
        != {"A": "surface_sheet", "B": "cutout_boundary_shell"}[route]
        for host in host_entities
    ):
        raise ValueError(
            f"{region.port_sheet_id} hosts must be active A/B face_metal "
            f"on {target_layer}"
        )
    return host_ids, host_entities


class _PortPlane(NamedTuple):
    """Where a port sheet sits and which volumes it borders."""

    z_um: float
    boundary_volume_ids: tuple[str, ...]
    embedded_volume_id: str


def _port_sheet_host_plane(
    port: _PortSheet,
    *,
    route: RouteLiteral,
    entities: Mapping[str, SemanticEntitySpec],
    planned_surfaces: Sequence[SurfacePlanRecord],
) -> _PortPlane:
    """Return the port plane z, the volumes it borders and the volume that embeds it."""
    if route == "A":
        plane_z_um, boundary_volume_ids = _route_a_port_sheet_sheet_contract(
            port.host_ids,
            planned_surfaces,
            port.region.overlaps,
        )
        embedded_volume_id = _route_a_port_sheet_vacuum_volume_id(
            boundary_volume_ids,
            entities,
        )
        return _PortPlane(plane_z_um, boundary_volume_ids, embedded_volume_id)
    # Route B: the port sits at mid-thickness of the hosts, inside their host void.
    plane_z_um = port.z_values[0] + port.thickness_values[0] / 2.0
    return _PortPlane(plane_z_um, (port.host_void_ids[0],), str(port.host_void_ids[0]))


def _port_sheet_surface(
    port: _PortSheet,
    *,
    route: RouteLiteral,
    plane: _PortPlane,
    component_by_entity: Mapping[str, str],
) -> SurfacePlanRecord:
    """Build the ``lumped_port`` surface record, with provenance of its hosts and source."""
    region = port.region
    host_ids = port.host_ids
    plane_z_um, boundary_volume_ids, embedded_volume_id = plane
    host_polygon_ids = tuple(overlap.host_polygon_id for overlap in region.overlaps)
    owner_provenance = tuple(
        {
            "semantic_id": host.semantic_id,
            "net_id": host.net_id,
            "equipotential_id": host.metadata.get("equipotential_id"),
            "conductor_component_id": component_by_entity.get(
                host.semantic_id,
                f"COMP__{host.semantic_id}",
            ),
        }
        for host in port.host_entities
    )
    source_provenance = {
        "source_name": port.source_name,
        "port_index": port.port_index,
        "source_layer": region.source_layer,
        "source_polygon_id": region.source_polygon_id,
        "overlap_ids": tuple(overlap.overlap_id for overlap in region.overlaps),
        "overlaps": tuple(
            {
                "overlap_id": overlap.overlap_id,
                "host_semantic_id": overlap.host_semantic_id,
                "host_polygon_id": overlap.host_polygon_id,
                "overlap_loop": overlap.overlap_loop,
            }
            for overlap in region.overlaps
        ),
        "host_polygon_ids": host_polygon_ids,
        "route": route,
        "target_layer": port.target_layer,
        "direction": port.direction,
        "direction_raw": port.raw_direction,
        "direction_sign_convention": port.sign_convention,
    }
    return SurfacePlanRecord(
        surface_id=f"SURF__LUMPED_PORT__{_safe_port_sheet_name(port.source_name)}",
        owner_semantic_id=host_ids[0],
        surface_role="lumped_port",
        geometry_ref={
            "outer_loop": port.remainder["outer_loop"],
            "hole_loops": port.remainder["hole_loops"],
            "plane": {
                "axis": "z",
                "value_um": plane_z_um,
            },
            "representation": "lumped_port_sheet",
        },
        valid_routes=(route,),
        metadata={
            "route": route,
            "representation": "lumped_port_sheet",
            "interface_type": "lumped_port",
            "face_kind": "sheet",
            "owner_semantic_ids": host_ids,
            "physical_owner_semantic_ids": host_ids,
            "boundary_volume_ids": boundary_volume_ids,
            "route_a_boundary_port": route == "A",
            "embedded_surface": True,
            "embedded_volume_id": embedded_volume_id,
            "physical_name": f"LUMPED_PORT__{_safe_port_sheet_name(port.source_name)}",
            "source_provenance": source_provenance,
            "physical_attribute": {
                "port_index": port.port_index,
                "port_name": port.source_name,
                "source_layer": region.source_layer,
                "target_layer": port.target_layer,
                "embedded_volume_id": embedded_volume_id,
                "direction": port.direction,
                "owner_semantic_ids": host_ids,
                "owner_provenance": owner_provenance,
            },
        },
    )


def _port_sheet_boundary_shares_segment(
    remainder: Mapping[str, Any], overlap_loop: Any
) -> bool:
    return any(
        _loops_share_edge_overlap(boundary, overlap_loop)
        for boundary in (remainder["outer_loop"], *remainder["hole_loops"])
    )


def _port_sheet_string(
    metadata: Mapping[str, Any], key: str, region: PortSheetRegionRecord
) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{region.port_sheet_id} requires {key}")
    return value


def _port_sheet_direction(region: PortSheetRegionRecord) -> tuple[float, float, float]:
    raw = region.metadata.get("direction")
    if isinstance(raw, str | bytes) or not isinstance(raw, Sequence) or len(raw) != 3:
        raise ValueError(f"{region.port_sheet_id} requires a 3D direction")
    direction = tuple(float(value) for value in raw)
    if (
        not all(isfinite(value) for value in direction)
        or direction[2] != 0.0
        or direction[0] == direction[1] == 0.0
    ):
        raise ValueError(
            f"{region.port_sheet_id} direction must be finite, XY, and nonzero"
        )
    length = sqrt(direction[0] ** 2 + direction[1] ** 2)
    return (direction[0] / length, direction[1] / length, 0.0)


def _port_sheet_raw_direction(
    region: PortSheetRegionRecord,
) -> tuple[float, float, float]:
    raw = region.metadata.get("direction_raw")
    if isinstance(raw, str | bytes) or not isinstance(raw, Sequence) or len(raw) != 3:
        raise ValueError(f"{region.port_sheet_id} requires raw direction provenance")
    direction = tuple(float(value) for value in raw)
    if (
        not all(isfinite(value) for value in direction)
        or direction[2] != 0.0
        or direction[0] == direction[1] == 0.0
    ):
        raise ValueError(
            f"{region.port_sheet_id} raw direction must be finite, XY, and nonzero"
        )
    return direction


def _port_sheet_target_layer_matches(
    entity: SemanticEntitySpec, target_layer: str
) -> bool:
    logical_layer = entity.metadata.get("logical_layer_id")
    return target_layer in {
        str(entity.metadata.get("semantic_group_id", "")),
        logical_layer if isinstance(logical_layer, str) else "",
        f"{entity.geometry.get('gds_layer')}/{entity.geometry.get('gds_datatype')}",
    }


def _safe_port_sheet_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)
