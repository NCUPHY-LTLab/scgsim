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
from functools import wraps
from itertools import combinations
from typing import (
    TYPE_CHECKING,
    Any,
    Concatenate,
    ParamSpec,
    Protocol,
    TypedDict,
    TypeVar,
    cast,
)

from scgsim.geometry.models.common import PolygonRing
from scgsim.geometry.models.grid import GeometryGridContext, geometry_grid_context
from scgsim.geometry.models.input import (
    GeometryBuildInput,
    LayoutPolygonSpec,
    SemanticEntitySpec,
)
from scgsim.geometry.models.regions import PortSheetRegionRecord
from scgsim.geometry.primitives.grid import (
    GridArrangement,
    GridOperand,
    SourceGridAudit,
    SourcePolygonAudit,
    build_grid_arrangement,
    geometry_grid_scope,
    regions,
    source_grid_audits,
)
from scgsim.geometry.primitives.grid_regions import (
    FIX_CONTACT,
    contact_changes,
    grid_change_message,
)
from scgsim.geometry.source.ports import (
    is_port_sheet_host_entity,
    port_sheet_overlaps,
    port_sheet_target_layer_matches,
)
from scgsim.semantics.route_a import geometry_z_range, same_z

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry


class _SourcePolygonIdentity(TypedDict):
    source_entity_id: str
    source_polygon_id: str


_SourceRing = Sequence[Sequence[float]]
_ZRange = tuple[float, float]
_EVERY_Z: _ZRange = (-math.inf, math.inf)


@dataclass(frozen=True)
class _PolygonPreimage:
    """Requested source rings retained with their original container types."""

    exterior: _SourceRing
    holes: Sequence[_SourceRing]


@dataclass(frozen=True)
class _SourcePolygonInputs:
    """Entity ownership and physical source preimages for layout polygons."""

    owners: Mapping[str, SemanticEntitySpec]
    preimages: Mapping[str, _PolygonPreimage]


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


_Arguments = ParamSpec("_Arguments")
_Result = TypeVar("_Result")


class _PlanKCell(Protocol):
    @property
    def dbu(self) -> float: ...


class _PlanComponent(Protocol):
    @property
    def kcl(self) -> _PlanKCell: ...


class _SourceGridPlan(Protocol):
    @property
    def _component(self) -> _PlanComponent: ...


_PlanT = TypeVar("_PlanT", bound=_SourceGridPlan)


def source_grid_context(source_dbu_um: float) -> GeometryGridContext:
    """The model grid: the database unit of the emitted source GDS."""
    return GeometryGridContext(source_dbu_um)


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


def apply_geometry_grid(
    build_input: GeometryBuildInput, context: GeometryGridContext
) -> GeometryBuildInput:
    """Project one complete source input inside its grid audit scope."""
    with geometry_grid_scope(context):
        return _apply_geometry_grid(build_input, context)


def _validate_bound_context(
    build_input: GeometryBuildInput, context: GeometryGridContext
) -> GeometryGridContext | None:
    bound_context = geometry_grid_context(build_input)
    if bound_context is not None and bound_context != context:
        raise ValueError("geometry grid context differs from the bound input policy")
    return bound_context


def _source_polygon_inputs(
    build_input: GeometryBuildInput,
    fresh_audits: tuple[SourceGridAudit, ...],
    *,
    use_source_preimage: bool,
) -> _SourcePolygonInputs:
    stored_audits = cast(
        Sequence[SourceGridAudit],
        build_input.metadata.get("geometry_grid_source_audits", ()),
    )
    audits = (*fresh_audits, *stored_audits)
    earliest_preimages: dict[tuple[str, str], SourcePolygonAudit] = {}
    for audit in audits:
        for source_polygon in audit.get("requested_polygons", ()):
            earliest_preimages.setdefault(
                (audit["source_entity_id"], source_polygon["source_polygon_id"]),
                source_polygon,
            )

    owners = {
        polygon_id: entity
        for entity in build_input.entities
        for polygon_id in entity.polygon_ids
    }
    preimages: dict[str, _PolygonPreimage] = {}
    for polygon in build_input.polygons:
        # Geometry metadata is the serialized boundary for primitive source identity.
        identity = cast(
            _SourcePolygonIdentity | None,
            polygon.metadata.get("geometry_grid_source_identity"),
        )
        owner = owners.get(polygon.polygon_id)
        if identity is None and "source_polygon_id" in polygon.metadata:
            identity = {
                "source_entity_id": cast(str, polygon.metadata["source_semantic_id"]),
                "source_polygon_id": cast(str, polygon.metadata["source_polygon_id"]),
            }
        if identity is None and owner is not None:
            identity = {
                "source_entity_id": owner.semantic_id,
                "source_polygon_id": polygon.polygon_id,
            }
        saved_preimage = (
            None
            if not use_source_preimage or identity is None
            else earliest_preimages.get(
                (identity["source_entity_id"], identity["source_polygon_id"])
            )
        )
        if (
            use_source_preimage
            and "geometry_grid_source_identity" in polygon.metadata
            and saved_preimage is None
        ):
            raise ValueError(
                "declared physical source preimage is unavailable: "
                f"{polygon.polygon_id}"
            )
        if saved_preimage:
            # JSON audit payloads may contain lists; preserve those public metadata values.
            preimages[polygon.polygon_id] = _PolygonPreimage(
                exterior=cast(_SourceRing, saved_preimage["exterior"]),
                holes=cast(Sequence[_SourceRing], saved_preimage["holes"]),
            )
        else:
            preimages[polygon.polygon_id] = _PolygonPreimage(
                exterior=polygon.exterior,
                holes=polygon.holes,
            )
    return _SourcePolygonInputs(owners=owners, preimages=preimages)


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
            "PORT::" + port.port_sheet_id,
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
    source: _SourcePolygonInputs,
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
    build_input: GeometryBuildInput, source: _SourcePolygonInputs
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
        z_ranges["PORT::" + port.port_sheet_id] = (
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
    port: PortSheetRegionRecord, source: _SourcePolygonInputs
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
    source: _SourcePolygonInputs,
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
        exterior, holes = regions_for_polygon[0]
        oriented_exterior = _oriented_like(exterior, polygon.exterior)
        oriented_holes = tuple(
            _oriented_like(ring, tuple(reversed(oriented_exterior))) for ring in holes
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
    source: _SourcePolygonInputs,
    projected: _ProjectedPolygons,
    context: GeometryGridContext,
) -> list[dict[str, object]]:
    from shapely import Polygon, union_all  # noqa: PLC0415 — load native topology only after projection validation.

    conductors = [
        entity for entity in build_input.entities if entity.material_kind == "conductor"
    ]
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
    contact_relations: list[dict[str, object]] = []
    for first, second in combinations(conductors, 2):
        first_z = geometry_z_range(first.geometry, first.semantic_id)
        second_z = geometry_z_range(second.geometry, second.semantic_id)
        if max(first_z[0], second_z[0]) > min(first_z[1], second_z[1]):
            continue
        requested = raw_geometry[first.semantic_id].relate(
            raw_geometry[second.semantic_id]
        )
        realized = effective_geometry[first.semantic_id].relate(
            effective_geometry[second.semantic_id]
        )
        if requested != realized:
            raise ValueError(
                grid_change_message(
                    "conductor contact changed at grid: "
                    f"{first.semantic_id}, {second.semantic_id}",
                    [
                        projected.by_id[polygon_id].layer
                        for entity in (first, second)
                        for polygon_id in entity.polygon_ids
                        if polygon_id in projected.by_id
                    ],
                    context.grid_size_um,
                    contact_changes(
                        (
                            raw_geometry[first.semantic_id],
                            raw_geometry[second.semantic_id],
                        ),
                        (
                            effective_geometry[first.semantic_id],
                            effective_geometry[second.semantic_id],
                        ),
                        context.grid_size_um,
                    ),
                    FIX_CONTACT,
                )
            )
        contact_relations.append(
            {
                "entity_ids": (first.semantic_id, second.semantic_id),
                "requested_de9im": requested,
                "realized_de9im": realized,
            }
        )
    return contact_relations


def _project_port_regions(
    build_input: GeometryBuildInput,
    source: _SourcePolygonInputs,
    projected_polygons: _ProjectedPolygons,
    groups: _NodingGroups,
    geometry_polygon: Callable[..., BaseGeometry],
) -> tuple[PortSheetRegionRecord, ...]:
    projected_ports: list[PortSheetRegionRecord] = []
    for port in build_input.port_sheet_regions:
        operand_id = "PORT::" + port.port_sheet_id
        arrangement = groups.arrangement_for(operand_id)
        grid_size_um = arrangement.context.grid_size_um
        regions_for_port = arrangement.operand_regions(operand_id)
        if len(regions_for_port) != 1:
            raise ValueError(
                f"port sheet component identity changed at grid: {port.port_sheet_id}"
            )
        exterior, holes = regions_for_port[0]
        oriented_exterior = _oriented_like(exterior, port.exterior)
        oriented_holes = tuple(
            _oriented_like(ring, tuple(reversed(oriented_exterior))) for ring in holes
        )
        authored_host_bindings = {
            (overlap.host_semantic_id, overlap.host_polygon_id)
            for overlap in port.overlaps
        }
        eligible_hosts = _port_sheet_hosts(port, source)
        raw_port = geometry_polygon(port.exterior, port.holes)
        effective_port = geometry_polygon(oriented_exterior, oriented_holes)
        contact_relations: list[dict[str, str]] = []
        for polygon_id, host in eligible_hosts.items():
            raw_host = source.preimages[polygon_id]
            effective_host = projected_polygons.by_id[polygon_id]
            requested = raw_port.relate(
                geometry_polygon(raw_host.exterior, raw_host.holes)
            )
            realized = effective_port.relate(
                geometry_polygon(effective_host.exterior, effective_host.holes)
            )
            if requested != realized:
                raise ValueError(
                    grid_change_message(
                        "port contact changed at grid: "
                        f"{port.port_sheet_id}, {host.semantic_id}, {polygon_id}",
                        [port.source_layer, effective_host.layer],
                        grid_size_um,
                        contact_changes(
                            (
                                raw_port,
                                geometry_polygon(raw_host.exterior, raw_host.holes),
                            ),
                            (
                                effective_port,
                                geometry_polygon(
                                    effective_host.exterior, effective_host.holes
                                ),
                            ),
                            grid_size_um,
                        ),
                        FIX_CONTACT,
                    )
                )
            contact_relations.append(
                {
                    "host_entity_id": host.semantic_id,
                    "host_polygon_id": polygon_id,
                    "requested_de9im": requested,
                    "realized_de9im": realized,
                }
            )

        authored_hosts = {
            polygon_id: entity
            for polygon_id, entity in eligible_hosts.items()
            if (entity.semantic_id, polygon_id) in authored_host_bindings
        }
        overlaps = port_sheet_overlaps(
            port_sheet_id=port.port_sheet_id,
            port_polygon=LayoutPolygonSpec(
                port.source_polygon_id,
                port.source_layer,
                oriented_exterior,
                oriented_holes,
            ),
            target_layer=cast(str, port.metadata["target_layer"]),
            host_entity_by_polygon_id=authored_hosts,
            host_polygons_by_id={
                polygon_id: projected_polygons.by_id[polygon_id]
                for polygon_id in authored_hosts
            },
        )
        if {
            (overlap.host_semantic_id, overlap.host_polygon_id) for overlap in overlaps
        } != authored_host_bindings:
            raise ValueError(
                f"port terminal overlap changed at grid: {port.port_sheet_id}"
            )
        projected_ports.append(
            replace(
                port,
                exterior=oriented_exterior,
                holes=oriented_holes,
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
        )
    return tuple(projected_ports)


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
    source: _SourcePolygonInputs,
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
    source: _SourcePolygonInputs,
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
        "geometry_grid_source_audits": source_grid_audits()
        or build_input.metadata.get("geometry_grid_source_audits", ()),
        "geometry_grid_contact_relations": contact_relations,
    }


def _apply_geometry_grid(
    build_input: GeometryBuildInput, context: GeometryGridContext
) -> GeometryBuildInput:
    """Run named source projection steps while retaining each raw input role."""
    bound_context = _validate_bound_context(build_input, context)
    fresh_audits = source_grid_audits()
    source = _source_polygon_inputs(
        build_input,
        fresh_audits,
        use_source_preimage=bound_context is None or bool(fresh_audits),
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


def with_source_grid(
    function: Callable[Concatenate[_PlanT, _Arguments], _Result],
) -> Callable[Concatenate[_PlanT, _Arguments], _Result]:
    """Wrap one GeometryPlan operation in its actual source DBU scope."""

    @wraps(function)
    def prepare(
        plan: _PlanT, *args: _Arguments.args, **kwargs: _Arguments.kwargs
    ) -> _Result:
        source_dbu_um = float(
            plan._component.kcl.dbu  # noqa: SLF001 — GeometryPlan owns the native source DBU.
        )
        context = source_grid_context(source_dbu_um)
        with geometry_grid_scope(context):
            return function(plan, *args, **kwargs)

    return prepare
