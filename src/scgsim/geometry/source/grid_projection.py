"""Project assembled source polygons onto their bound XY grid.

Polygons are noded only against geometry they can touch: the operands whose Z
ranges meet. Two dies of a flip chip therefore never split each other's
outlines, while a bump is still noded with the metal on both faces it joins.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from itertools import combinations
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

from scgsim.geometry.grid.arrangement import build_grid_arrangement
from scgsim.geometry.grid.grid_regions import (
    FIX_CONTACT,
    GridArrangement,
    GridOperand,
    contact_changes,
    grid_change_message,
    regions,
)
from scgsim.geometry.grid.scope import GridScope
from scgsim.geometry.models.common import PolygonRing
from scgsim.geometry.models.grid import GeometryGridContext, geometry_grid_context
from scgsim.geometry.models.input import (
    GeometryBuildInput,
    LayoutPolygonSpec,
    SemanticEntitySpec,
)
from scgsim.geometry.models.regions import PortSheetOverlapRecord, PortSheetRegionRecord
from scgsim.geometry.source.ports import (
    PortSheetHost,
    is_port_sheet_host_entity,
    port_sheet_overlaps,
    port_sheet_target_layer_matches,
)
from scgsim.geometry.source.preimages import (
    PolygonPreimage,
    SourcePolygonInputs,
    source_polygon_inputs,
)
from scgsim.geometry.source.recipes import SourceGridAudit
from scgsim.semantics.route_a import geometry_z_range, same_z

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry


_ZRange = tuple[float, float]
_EVERY_Z: _ZRange = (-math.inf, math.inf)


@dataclass(frozen=True)
class _ProjectedPolygons:
    """Effective source polygons and the index used by later projections."""

    records: tuple[LayoutPolygonSpec, ...]
    by_id: Mapping[str, LayoutPolygonSpec]


@dataclass(frozen=True)
class _NodingGroups:
    """One grid arrangement per source Z range, and the range of each operand.

    The arrangement of a Z range holds every operand whose closed Z range meets
    it, which is the set the SGB nodes together on that range's levels and slab.
    Each operand is projected from the arrangement of its own range.
    """

    arrangements: Mapping[_ZRange, GridArrangement]
    z_ranges: Mapping[str, _ZRange]

    def arrangement_for(self, operand_id: str) -> GridArrangement:
        """Return the arrangement an operand is projected from."""
        return self.arrangements[self.z_ranges[operand_id]]

    @property
    def metadata(self) -> list[dict[str, object]]:
        """Describe each Z range's arrangement in ascending Z order.

        An operand without a Z range is recorded with ``z_range_um`` None.
        """
        return [
            {
                "z_range_um": None if z_range == _EVERY_Z else list(z_range),
                **arrangement.metadata,
            }
            for z_range, arrangement in self.arrangements.items()
        ]


def _signed_area_twice(ring: Sequence[Sequence[float]]) -> float:
    return sum(
        first[0] * second[1] - first[1] * second[0]
        for first, second in zip(ring, (*ring[1:], ring[0]), strict=False)
    )


def _polygon_ring(ring: Sequence[Sequence[float]]) -> PolygonRing:
    return tuple((point[0], point[1]) for point in ring)


def _oriented_like(ring: Sequence[Sequence[float]], source: PolygonRing) -> PolygonRing:
    source_ring = _polygon_ring(ring)
    if _signed_area_twice(source_ring) * _signed_area_twice(source) < 0:
        return tuple(reversed(source_ring))
    return source_ring


def _oriented_region(
    region: tuple[Sequence[Sequence[float]], Sequence[Sequence[Sequence[float]]]],
    source_exterior: PolygonRing,
) -> tuple[PolygonRing, tuple[PolygonRing, ...]]:
    """A gridded region turned like its source: the exterior as the source's, each hole against it."""
    exterior, holes = region
    oriented_exterior = _oriented_like(exterior, source_exterior)
    oriented_holes = tuple(
        _oriented_like(ring, tuple(reversed(oriented_exterior))) for ring in holes
    )
    return oriented_exterior, oriented_holes


def _port_operand_id(port: PortSheetRegionRecord) -> str:
    """The grid operand id of a port sheet, apart from every layout polygon id."""
    return "PORT::" + port.port_sheet_id


def apply_geometry_grid(
    build_input: GeometryBuildInput,
    context: GeometryGridContext,
    *,
    source_audits: tuple[SourceGridAudit, ...] = (),
) -> GeometryBuildInput:
    """Project one complete source input onto ``context``, retaining each raw input role.

    ``source_audits`` are the audits of the declared source recipes that read this
    input from its layout, in reading order; an input projected again on its own
    grid has none and keeps the audits it carries.
    """
    bound_context = _validate_bound_context(build_input, context)
    source = source_polygon_inputs(
        build_input,
        source_audits,
        use_source_preimage=bound_context is None or bool(source_audits),
    )
    groups = _noding_groups(build_input, source, context)
    projected_polygons = _project_layout_polygons(build_input, source, groups)

    from shapely import Polygon, union_all  # noqa: PLC0415 — native geometry is needed only after source projection.

    geometry_operations = _GeometryOperations(Polygon, union_all)

    contact_relations = _validate_conductor_contacts(
        build_input, source, projected_polygons, context
    )
    projected_ports = _project_port_regions(
        build_input,
        source,
        projected_polygons,
        groups,
        geometry_operations.polygon,
    )
    projected_entities = _project_entities(
        build_input,
        projected_polygons,
        groups,
        context,
        geometry_operations,
    )
    source_metadata = _project_metadata(
        build_input, source, context, groups, contact_relations
    )
    return replace(
        build_input,
        polygons=projected_polygons.records,
        entities=projected_entities,
        port_sheet_regions=projected_ports,
        metadata=source_metadata,
    )


def _validate_bound_context(
    build_input: GeometryBuildInput, context: GeometryGridContext
) -> GeometryGridContext | None:
    # A fresh input from the adapter is bound here; a bound one is projected again on its own grid.
    if "geometry_grid" not in build_input.metadata:
        return None
    bound_context = geometry_grid_context(build_input)
    if bound_context != context:
        raise ValueError("geometry grid context differs from the bound input policy")
    return bound_context


def _grid_operands(build_input: GeometryBuildInput) -> tuple[GridOperand, ...]:
    operands = [
        GridOperand(
            polygon.polygon_id,
            polygon.exterior,
            polygon.holes,
            contact=False,
            layer=polygon.layer,
        )
        for polygon in build_input.polygons
    ]
    operands.extend(
        GridOperand(
            _port_operand_id(port),
            port.exterior,
            port.holes,
            contact=False,
            layer=port.source_layer,
        )
        for port in build_input.port_sheet_regions
    )
    return tuple(operands)


def _noding_groups(
    build_input: GeometryBuildInput,
    source: SourcePolygonInputs,
    context: GeometryGridContext,
) -> _NodingGroups:
    operands = _grid_operands(build_input)
    z_ranges = _operand_z_ranges(build_input, source)
    arrangements: dict[_ZRange, GridArrangement] = {}
    by_members: dict[tuple[str, ...], GridArrangement] = {}
    for z_range in sorted(set(z_ranges.values())):
        members = tuple(
            operand
            for operand in operands
            if _z_ranges_meet(z_ranges[operand.operand_id], z_range)
        )
        key = tuple(operand.operand_id for operand in members)
        if key not in by_members:
            by_members[key] = build_grid_arrangement(members, context)
        arrangements[z_range] = by_members[key]
    return _NodingGroups(arrangements, z_ranges)


def _operand_z_ranges(
    build_input: GeometryBuildInput, source: SourcePolygonInputs
) -> dict[str, _ZRange]:
    z_ranges = {
        polygon.polygon_id: _owner_z_range(source.owners.get(polygon.polygon_id))
        for polygon in build_input.polygons
    }
    for port in build_input.port_sheet_regions:
        host_ranges = [
            _owner_z_range(host) for host in _port_sheet_hosts(port, source).values()
        ]
        # A port sheet lies on its hosts' layer; spanning every host keeps it
        # noded with each conductor it may overlap.
        z_ranges[_port_operand_id(port)] = (
            (min(low for low, _ in host_ranges), max(high for _, high in host_ranges))
            if host_ranges
            else _EVERY_Z
        )
    return z_ranges


def _owner_z_range(owner: SemanticEntitySpec | None) -> _ZRange:
    # Without a Z range nothing says which operands this one never touches, so
    # it is noded with all of them.
    if owner is None or not {"z_min_um", "z_max_um", "z_um"} & owner.geometry.keys():
        return _EVERY_Z
    return geometry_z_range(owner.geometry, owner.semantic_id)


def _z_ranges_meet(first: _ZRange, second: _ZRange) -> bool:
    # Closed ranges: metal that only touches another face still shares that
    # level, as a bump touches the metal on each die.
    low, high = max(first[0], second[0]), min(first[1], second[1])
    return low <= high or same_z(low, high)


def _port_sheet_hosts(
    port: PortSheetRegionRecord, source: SourcePolygonInputs
) -> dict[str, SemanticEntitySpec]:
    return {
        polygon_id: entity
        for polygon_id, entity in source.owners.items()
        if is_port_sheet_host_entity(entity)
        and port_sheet_target_layer_matches(
            entity, cast(str, port.metadata["target_layer"])
        )
    }


def _project_layout_polygons(
    build_input: GeometryBuildInput,
    source: SourcePolygonInputs,
    groups: _NodingGroups,
) -> _ProjectedPolygons:
    projected: list[LayoutPolygonSpec] = []
    for polygon in build_input.polygons:
        regions_for_polygon = groups.arrangement_for(
            polygon.polygon_id
        ).operand_regions(polygon.polygon_id)
        if len(regions_for_polygon) != 1:
            raise ValueError(
                "source polygon component identity changed at grid: "
                f"{polygon.polygon_id}"
            )
        oriented_exterior, oriented_holes = _oriented_region(
            regions_for_polygon[0], polygon.exterior
        )
        projected.append(
            replace(
                polygon,
                exterior=oriented_exterior,
                holes=oriented_holes,
                drawn_rings=_drawn_rings_on(
                    polygon, (oriented_exterior, *oriented_holes)
                ),
                metadata={
                    **polygon.metadata,
                    "unquantized_source_geometry": polygon.metadata.get(
                        "unquantized_source_geometry",
                        {
                            "exterior": source.preimages[polygon.polygon_id].exterior,
                            "holes": source.preimages[polygon.polygon_id].holes,
                        },
                    ),
                },
            )
        )
    records = tuple(projected)
    return _ProjectedPolygons(
        records=records,
        by_id={polygon.polygon_id: polygon for polygon in records},
    )


def _drawn_rings_on(
    polygon: LayoutPolygonSpec, rings: Sequence[PolygonRing]
) -> tuple[PolygonRing | None, ...]:
    # Each ring the grid emits keeps the drawn order of the input ring whose vertices it
    # holds: noding adds vertices to a ring and can reorder the holes, never its drawn ones.
    if not polygon.drawn_rings:
        return ()
    before = [
        {tuple(point) for point in ring} for ring in (polygon.exterior, *polygon.holes)
    ]
    carried: list[PolygonRing | None] = []
    for ring in rings:
        present = {tuple(point) for point in ring}
        found = [i for i, keys in enumerate(before) if keys <= present]
        carried.append(
            polygon.drawn_rings[found[0]]
            if len(found) == 1 and found[0] < len(polygon.drawn_rings)
            else None
        )
    return tuple(carried)


def _validate_conductor_contacts(
    build_input: GeometryBuildInput,
    source: SourcePolygonInputs,
    projected: _ProjectedPolygons,
    context: GeometryGridContext,
) -> list[dict[str, object]]:
    """The contact of each pair of conductors that share Z, refused where the grid changed it."""
    conductors = [
        entity for entity in build_input.entities if entity.material_kind == "conductor"
    ]
    raw_geometry, effective_geometry = _conductor_shapes(
        conductors, source, projected, context
    )
    contact_relations: list[dict[str, object]] = []
    for first, second in combinations(conductors, 2):
        first_z = geometry_z_range(first.geometry, first.semantic_id)
        second_z = geometry_z_range(second.geometry, second.semantic_id)
        if max(first_z[0], second_z[0]) > min(first_z[1], second_z[1]):
            continue
        pair = (first.semantic_id, second.semantic_id)
        requested = raw_geometry[pair[0]].relate(raw_geometry[pair[1]])
        realized = effective_geometry[pair[0]].relate(effective_geometry[pair[1]])
        if requested != realized:
            raise ValueError(
                _conductor_contact_refusal(
                    (first, second),
                    projected,
                    (raw_geometry[pair[0]], raw_geometry[pair[1]]),
                    (effective_geometry[pair[0]], effective_geometry[pair[1]]),
                    context.grid_size_um,
                )
            )
        contact_relations.append(
            {
                "entity_ids": pair,
                "requested_de9im": requested,
                "realized_de9im": realized,
            }
        )
    return contact_relations


def _conductor_shapes(
    conductors: Sequence[SemanticEntitySpec],
    source: SourcePolygonInputs,
    projected: _ProjectedPolygons,
    context: GeometryGridContext,
) -> tuple[dict[str, BaseGeometry], dict[str, BaseGeometry]]:
    """Each conductor's region as requested and as the grid realized it, by Entity id."""
    from shapely import Polygon, union_all  # noqa: PLC0415 — load native topology only after projection validation.

    raw_geometry = {
        entity.semantic_id: union_all(
            [
                Polygon(
                    source.preimages[polygon_id].exterior,
                    source.preimages[polygon_id].holes,
                )
                for polygon_id in entity.polygon_ids
            ]
        )
        for entity in conductors
    }
    effective_geometry = {
        entity.semantic_id: union_all(
            [
                Polygon(polygon.exterior, polygon.holes)
                for polygon in projected.records
                if polygon.polygon_id in entity.polygon_ids
            ],
            grid_size=context.grid_size_um,
        )
        for entity in conductors
    }
    return raw_geometry, effective_geometry


def _conductor_contact_refusal(
    pair: tuple[SemanticEntitySpec, SemanticEntitySpec],
    projected: _ProjectedPolygons,
    requested: tuple[BaseGeometry, BaseGeometry],
    realized: tuple[BaseGeometry, BaseGeometry],
    grid_size_um: float,
) -> str:
    """The refusal of two conductors whose contact the grid changed, on their layers and where it changed."""
    first, second = pair
    return grid_change_message(
        f"conductor contact changed at grid: {first.semantic_id}, {second.semantic_id}",
        [
            projected.by_id[polygon_id].layer
            for entity in pair
            for polygon_id in entity.polygon_ids
            if polygon_id in projected.by_id
        ],
        grid_size_um,
        contact_changes(requested, realized, grid_size_um),
        FIX_CONTACT,
    )


def _project_port_regions(
    build_input: GeometryBuildInput,
    source: SourcePolygonInputs,
    projected_polygons: _ProjectedPolygons,
    groups: _NodingGroups,
    geometry_polygon: Callable[..., BaseGeometry],
) -> tuple[PortSheetRegionRecord, ...]:
    return tuple(
        _project_port_region(port, source, projected_polygons, groups, geometry_polygon)
        for port in build_input.port_sheet_regions
    )


@dataclass(frozen=True)
class _HostOnGrid:
    """A conductor polygon a port sheet may touch: its Entity, as requested and as the grid realized it."""

    polygon_id: str
    entity: SemanticEntitySpec
    requested: PolygonPreimage
    realized: LayoutPolygonSpec


def _project_port_region(
    port: PortSheetRegionRecord,
    source: SourcePolygonInputs,
    projected_polygons: _ProjectedPolygons,
    groups: _NodingGroups,
    geometry_polygon: Callable[..., BaseGeometry],
) -> PortSheetRegionRecord:
    """One port sheet on the grid, with the contacts and terminal overlaps the grid kept."""
    operand_id = _port_operand_id(port)
    arrangement = groups.arrangement_for(operand_id)
    regions_for_port = arrangement.operand_regions(operand_id)
    if len(regions_for_port) != 1:
        raise ValueError(
            f"port sheet component identity changed at grid: {port.port_sheet_id}"
        )
    projected = _oriented_region(regions_for_port[0], port.exterior)
    hosts = [
        _HostOnGrid(
            polygon_id,
            entity,
            source.preimages[polygon_id],
            projected_polygons.by_id[polygon_id],
        )
        for polygon_id, entity in _port_sheet_hosts(port, source).items()
    ]
    contact_relations = _port_contact_relations(
        port, projected, hosts, arrangement.context.grid_size_um, geometry_polygon
    )
    overlaps = _port_terminal_overlaps(
        GridScope(arrangement.context), port, projected, hosts
    )
    return replace(
        port,
        exterior=projected[0],
        holes=projected[1],
        overlaps=overlaps,
        metadata={
            **port.metadata,
            "geometry_grid_contact_relations": contact_relations,
            "unquantized_source_geometry": {
                "exterior": port.exterior,
                "holes": port.holes,
            },
        },
    )


def _port_contact_relations(
    port: PortSheetRegionRecord,
    projected: tuple[PolygonRing, tuple[PolygonRing, ...]],
    hosts: Sequence[_HostOnGrid],
    grid_size_um: float,
    geometry_polygon: Callable[..., BaseGeometry],
) -> list[dict[str, str]]:
    """The contact of the port with each host it may touch, refused where the grid changed it."""
    raw_port = geometry_polygon(port.exterior, port.holes)
    effective_port = geometry_polygon(*projected)
    contact_relations: list[dict[str, str]] = []
    for host in hosts:
        raw_host = geometry_polygon(host.requested.exterior, host.requested.holes)
        effective_host = geometry_polygon(host.realized.exterior, host.realized.holes)
        requested = raw_port.relate(raw_host)
        realized = effective_port.relate(effective_host)
        if requested != realized:
            raise ValueError(
                grid_change_message(
                    "port contact changed at grid: "
                    f"{port.port_sheet_id}, {host.entity.semantic_id}, {host.polygon_id}",
                    [port.source_layer, host.realized.layer],
                    grid_size_um,
                    contact_changes(
                        (raw_port, raw_host),
                        (effective_port, effective_host),
                        grid_size_um,
                    ),
                    FIX_CONTACT,
                )
            )
        contact_relations.append(
            {
                "host_entity_id": host.entity.semantic_id,
                "host_polygon_id": host.polygon_id,
                "requested_de9im": requested,
                "realized_de9im": realized,
            }
        )
    return contact_relations


def _port_terminal_overlaps(
    scope: GridScope,
    port: PortSheetRegionRecord,
    projected: tuple[PolygonRing, tuple[PolygonRing, ...]],
    hosts: Sequence[_HostOnGrid],
) -> tuple[PortSheetOverlapRecord, ...]:
    """The port's overlaps with its authored terminals on the grid, which must be the authored ones."""
    authored_host_bindings = {
        (overlap.host_semantic_id, overlap.host_polygon_id) for overlap in port.overlaps
    }
    overlaps = port_sheet_overlaps(
        scope,
        port_sheet_id=port.port_sheet_id,
        port_polygon=LayoutPolygonSpec(
            port.source_polygon_id,
            port.source_layer,
            *projected,
        ),
        target_layer=cast(str, port.metadata["target_layer"]),
        hosts={
            host.polygon_id: PortSheetHost(host.entity, host.realized)
            for host in hosts
            if (host.entity.semantic_id, host.polygon_id) in authored_host_bindings
        },
    )
    if {
        (overlap.host_semantic_id, overlap.host_polygon_id) for overlap in overlaps
    } != authored_host_bindings:
        raise ValueError(f"port terminal overlap changed at grid: {port.port_sheet_id}")
    return overlaps


def _project_entities(
    build_input: GeometryBuildInput,
    projected_polygons: _ProjectedPolygons,
    groups: _NodingGroups,
    context: GeometryGridContext,
    geometry_operations: _GeometryOperations,
) -> tuple[SemanticEntitySpec, ...]:
    entities: list[SemanticEntitySpec] = []
    for entity in build_input.entities:
        owned_polygons = [
            projected_polygons.by_id[polygon_id] for polygon_id in entity.polygon_ids
        ]
        geometry = dict(entity.geometry)
        if owned_polygons and "outer_loop" in geometry:
            combined = geometry_operations.union_all(
                [
                    geometry_operations.polygon(polygon.exterior, polygon.holes)
                    for polygon in owned_polygons
                ],
                grid_size=context.grid_size_um,
            )
            if combined.geom_type != "Polygon":
                raise ValueError(
                    "source Entity support changed component identity at grid: "
                    f"{entity.semantic_id}"
                )
            # An Entity's polygons share its Z range, so they share one arrangement.
            arrangement = groups.arrangement_for(entity.polygon_ids[0])
            outer, holes = regions(
                combined, arrangement.vertices, context.grid_size_um
            )[0]
            geometry.update(outer_loop=outer, hole_loops=holes)
        entities.append(replace(entity, geometry=geometry))
    return tuple(entities)


def _source_geometry_payload(
    build_input: GeometryBuildInput,
    source: SourcePolygonInputs,
) -> dict[str, object]:
    return {
        "polygons": [
            {
                "polygon_id": polygon.polygon_id,
                "layer": polygon.layer,
                "exterior": source.preimages[polygon.polygon_id].exterior,
                "holes": source.preimages[polygon.polygon_id].holes,
            }
            for polygon in build_input.polygons
        ],
        "port_sheet_regions": [asdict(port) for port in build_input.port_sheet_regions],
    }


@dataclass(frozen=True)
class _GeometryOperations:
    """Native geometry constructors loaded after source projection validation."""

    polygon: Callable[..., BaseGeometry]
    union_all: Callable[..., BaseGeometry]


def _project_metadata(
    build_input: GeometryBuildInput,
    source: SourcePolygonInputs,
    context: GeometryGridContext,
    groups: _NodingGroups,
    contact_relations: list[dict[str, object]],
) -> dict[str, Any]:
    return {
        **build_input.metadata,
        "geometry_grid": context.to_payload(),
        "unquantized_source_sha256": hashlib.sha256(
            json.dumps(
                _source_geometry_payload(build_input, source),
                sort_keys=True,
                separators=(",", ":"),
                default=list,
            ).encode()
        ).hexdigest(),
        "geometry_grid_arrangement": groups.metadata,
        "geometry_grid_source_audits": source.recorded_audits,
        "geometry_grid_contact_relations": contact_relations,
    }
