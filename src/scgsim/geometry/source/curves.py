"""Normalize authored boundaries, bind them to source rings, and restore them after stack preparation.

This module owns source XY curve identity, including the drawn GDS ring a
reconstruction record's indices name. Prepared geometry owns its Z, materials,
and solution-region state.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, cast, overload

from scgsim.geometry.models.common import PolygonRing
from scgsim.geometry.models.grid import geometry_grid_context
from scgsim.geometry.models.input import (
    DRAWN_RING_METADATA,
    BoundaryCurveChainSpec,
    GdsBoundaryReconstructionSpec,
    GeometryBuildInput,
    LayoutPolygonSpec,
    SemanticEntitySpec,
    SourceCurveSpec,
    carried_drawn_ring,
)
from scgsim.geometry.models.junctions import is_junction_end_piece
from scgsim.geometry.models.regions import PortSheetRegionRecord
from scgsim.geometry.primitives.grid_regions import canonical_grid_point
from scgsim.geometry.primitives.planar_loops import cutline_boundary_loops
from scgsim.geometry.source.occurrence import read_polygons_by_layer
from scgsim.geometry.source.plan_assembly import transform_source_point

_BoundaryRecord = BoundaryCurveChainSpec | GdsBoundaryReconstructionSpec
_BoundaryInput = _BoundaryRecord | Mapping[str, Any]
_BoundaryPoint = Callable[[tuple[float, ...], Sequence[float]], Sequence[float]]
_Ring = Sequence[Sequence[float]]
GdsPoint = tuple[float, float]
# One GDS polygon's vertices in the order the polygon walks them.
GdsWalk = tuple[GdsPoint, ...]
# Keep tuple keys so binding errors retain the established source identifier text.
BoundaryKey = tuple[str, str, str, int | None]
SourceRings = dict[str, list[tuple[str, PolygonRing, tuple[PolygonRing, ...]]]]


@dataclass(frozen=True)
class _PreparedSourceGeometry:
    """Source-owned polygon and entity records in prepared input order."""

    polygons: tuple[LayoutPolygonSpec, ...]
    entities: tuple[SemanticEntitySpec, ...]


@dataclass(frozen=True)
class _SourceEntityIndex:
    """Authored and prepared entity views used during source restoration."""

    active_ids: set[str]
    source_by_id: Mapping[str, SemanticEntitySpec]
    prepared_by_id: Mapping[str, SemanticEntitySpec]
    split_hosts: Mapping[str, str]


def source_curve(value: SourceCurveSpec | Mapping[str, Any]) -> SourceCurveSpec:
    """Normalize one typed or serialized curve declaration at its input boundary."""
    if isinstance(value, SourceCurveSpec):
        value = {name: getattr(value, name) for name in value.__dataclass_fields__}
    if not isinstance(value, Mapping):
        raise TypeError("source curve must be a SourceCurveSpec or mapping")
    record = SourceCurveSpec(**dict(value))
    if record.kind not in {
        "line_segment",
        "circular_arc",
        "interpolation_spline",
        "bspline",
    }:
        raise ValueError(f"unsupported source curve kind {record.kind!r}")
    points = tuple(tuple(float(x) for x in point) for point in record.points_um)
    if any(len(point) != 2 for point in points):
        raise ValueError("source curve points must be XY coordinates in micrometres")
    if record.kind == "line_segment" and len(points) != 2:
        raise ValueError("source line needs two endpoints")
    if record.kind == "circular_arc" and len(points) != 3:
        raise ValueError("source circular arc needs start, on-arc and end points")
    return replace(
        record,
        points_um=points,
        weights=tuple(record.weights),
        knots=tuple(record.knots),
        multiplicities=tuple(record.multiplicities),
        parameter_interval=(
            None
            if record.parameter_interval is None
            else tuple(record.parameter_interval)
        ),
    )


@overload
def boundary_binding(
    value: BoundaryCurveChainSpec | Mapping[str, Any],
    *,
    reconstruction: Literal[False] = False,
) -> BoundaryCurveChainSpec: ...


@overload
def boundary_binding(
    value: GdsBoundaryReconstructionSpec | Mapping[str, Any],
    *,
    reconstruction: Literal[True],
) -> GdsBoundaryReconstructionSpec: ...


@overload
def boundary_binding(
    value: _BoundaryInput, *, reconstruction: bool
) -> _BoundaryRecord: ...


def boundary_binding(
    value: _BoundaryInput, *, reconstruction: bool = False
) -> _BoundaryRecord:
    """Normalize one authored boundary binding without changing source order."""
    cls = GdsBoundaryReconstructionSpec if reconstruction else BoundaryCurveChainSpec
    if isinstance(value, cls):
        value = {name: getattr(value, name) for name in value.__dataclass_fields__}
    if not isinstance(value, Mapping):
        raise TypeError("boundary binding must be a source record or mapping")
    # Caller mappings are JSON-shaped external records; model construction validates fields.
    data = cast(dict[str, Any], copy.deepcopy(dict(value)))
    if not reconstruction:
        data["curves"] = tuple(source_curve(curve) for curve in data["curves"])
    else:
        data["anchor_indices"] = tuple(data.get("anchor_indices", ()))
        data["segment_indices"] = tuple(data.get("segment_indices", ()))
    data["selector_point_um"] = tuple(data["selector_point_um"])
    record = cls(**data)
    if not record.boundary_id or not record.entity_id:
        raise ValueError("boundary and Entity identities must be nonempty")
    if record.role not in {"outer", "hole"}:
        raise ValueError("source boundary role must be outer or hole")
    if record.role == "hole":
        if (
            isinstance(record.hole_index, bool)
            or not isinstance(record.hole_index, int)
            or record.hole_index < 0
        ):
            raise ValueError("hole boundary needs a zero-based source hole index")
    elif record.hole_index is not None:
        raise ValueError("outer boundary cannot have a hole index")
    return record


@overload
def transform_boundary(
    value: _BoundaryInput,
    *,
    path: str,
    transform: tuple[float, ...],
    point: _BoundaryPoint,
    reconstruction: Literal[False] = False,
) -> BoundaryCurveChainSpec: ...


@overload
def transform_boundary(
    value: _BoundaryInput,
    *,
    path: str,
    transform: tuple[float, ...],
    point: _BoundaryPoint,
    reconstruction: Literal[True],
) -> GdsBoundaryReconstructionSpec: ...


@overload
def transform_boundary(
    value: _BoundaryInput,
    *,
    path: str,
    transform: tuple[float, ...],
    point: _BoundaryPoint,
    reconstruction: bool,
) -> _BoundaryRecord: ...


def transform_boundary(
    value: _BoundaryInput,
    *,
    path: str,
    transform: tuple[float, ...],
    point: _BoundaryPoint,
    reconstruction: bool = False,
) -> _BoundaryRecord:
    """Qualify local ownership and transform geometry without changing order."""
    record = boundary_binding(value, reconstruction=reconstruction)
    if "/" in record.entity_id or "/" in record.boundary_id:
        raise ValueError(
            "component-authored boundary and Entity IDs must be local names"
        )
    boundary_id = f"{path}/{record.boundary_id}"
    entity_id = f"{path}/{record.entity_id}"
    selector_point_um = tuple(point(transform, record.selector_point_um))
    metadata = {
        **record.metadata,
        "source_local_boundary_id": record.boundary_id,
        "source_transform": tuple(transform),
        "source_orientation": (
            1 if transform[0] * transform[3] - transform[1] * transform[2] > 0 else -1
        ),
    }
    if not reconstruction:
        # The same flag selected the chain-record variant above.
        curve_record = cast(BoundaryCurveChainSpec, record)
        curves = tuple(
            replace(
                curve,
                points_um=tuple(tuple(point(transform, xy)) for xy in curve.points_um),
            )
            for curve in curve_record.curves
        )
        return replace(
            curve_record,
            boundary_id=boundary_id,
            entity_id=entity_id,
            selector_point_um=selector_point_um,
            source_occurrence_path=path,
            metadata=metadata,
            curves=curves,
        )
    # The same flag selected the reconstruction-record variant above.
    reconstruction_record = cast(GdsBoundaryReconstructionSpec, record)
    return replace(
        reconstruction_record,
        boundary_id=boundary_id,
        entity_id=entity_id,
        selector_point_um=selector_point_um,
        source_occurrence_path=path,
        metadata=metadata,
    )


def select_boundary(
    record: _BoundaryRecord,
    rings: SourceRings,
    end_pieces: frozenset[str] = frozenset(),
) -> tuple[BoundaryKey, PolygonRing]:
    """Bind one boundary record to the ring its selector point and role name.

    The selector lies in exactly one Entity-owned logical polygon; ``role`` and
    ``hole_index`` then name that polygon's exterior or one of its holes.
    """
    candidates = _selected_polygons(record, rings, end_pieces)
    if len(candidates) != 1:
        raise ValueError(
            f"{record.boundary_id} selector must bind one Entity-owned logical polygon"
        )
    pid, outer, holes = candidates[0]
    ring = outer if record.role == "outer" else holes[cast(int, record.hole_index)]
    return (record.entity_id, pid, record.role, record.hole_index), cast(
        PolygonRing, tuple(tuple(float(x) for x in xy) for xy in ring)
    )


def _selected_polygons(
    record: _BoundaryRecord, rings: SourceRings, end_pieces: frozenset[str]
) -> list[tuple[str, PolygonRing, tuple[PolygonRing, ...]]]:
    # GDS polygon selection is optional until source boundaries are bound.
    import gdstk  # noqa: PLC0415 - optional native layout dependency

    candidates = []
    for pid, outer, holes in rings.get(record.entity_id, ()):
        # A junction end piece can overlap the terminal the selector names; it is never selected.
        if pid in end_pieces:
            continue
        region = [gdstk.Polygon(outer)]
        if holes:
            region = (
                gdstk.boolean(region, [gdstk.Polygon(h) for h in holes], "not") or []
            )
        if gdstk.inside([record.selector_point_um], region)[0]:
            candidates.append((pid, outer, holes))
    return candidates


def with_drawn_rings(
    records: Sequence[_BoundaryInput],
    emitted: GeometryBuildInput,
    gds_walks: Callable[[GdsBoundaryReconstructionSpec], Sequence[_Ring]],
) -> tuple[_BoundaryInput, ...]:
    """Record on each reconstruction record the drawn GDS ring its indices name.

    The source stage reads each polygon from the GDS walk its author drew, then fuses,
    rounds and nodes it: the ring it emits may start at another vertex, run the other way,
    and hold crossings the grid inserted.  A record's indices keep naming the drawn ring,
    so that ring is found here, among the walks ``gds_walks`` gives for the record's
    occurrence in assembly coordinates, and stored under ``DRAWN_RING_METADATA``.  Each
    binding stage carries the indices from it onto the ring it holds.

    The drawn ring of the ring a record selects in ``emitted`` is the one loop of a walk
    whose vertices all lie on that ring, in the order the walk first visits them.  A record
    whose ring has no such loop, or more than one, is returned as it is, and the binding
    stages refuse it; so is a record in serialized form or one that carries its ring.
    """
    # Without polygons, or without a grid, the source stage has emitted no ring to bind.
    context = geometry_grid_context(emitted) if emitted.polygons else None
    if not records or context is None:
        return tuple(records)
    rings = _polygon_rings(emitted)
    end_pieces = frozenset(
        polygon.polygon_id
        for polygon in emitted.polygons
        if is_junction_end_piece(polygon)
    )
    carried: list[_BoundaryInput] = []
    for record in records:
        if (
            isinstance(record, GdsBoundaryReconstructionSpec)
            and carried_drawn_ring(record) is None
        ):
            ring = _named_ring(record, rings, end_pieces)
            drawn = (
                None
                if ring is None
                else _drawn_ring(ring, gds_walks(record), context.grid_size_um)
            )
            if drawn is not None:
                record = replace(
                    record, metadata={**record.metadata, DRAWN_RING_METADATA: drawn}
                )
        carried.append(record)
    return tuple(carried)


def _polygon_rings(build_input: GeometryBuildInput) -> SourceRings:
    # Only polygons are read from GDS walks; an Entity without one has no drawn ring.
    polygons = {polygon.polygon_id: polygon for polygon in build_input.polygons}
    return {
        entity.semantic_id: [
            (pid, polygons[pid].exterior, polygons[pid].holes)
            for pid in entity.polygon_ids
            if pid in polygons
        ]
        for entity in build_input.entities
    }


def _named_ring(
    record: GdsBoundaryReconstructionSpec,
    rings: SourceRings,
    end_pieces: frozenset[str],
) -> PolygonRing | None:
    # The ring select_boundary binds; a selection it refuses is left for it to refuse.
    selected = _selected_polygons(record, rings, end_pieces)
    if len(selected) != 1:
        return None
    _, outer, holes = selected[0]
    if record.role == "outer":
        return outer
    if _is_index(record.hole_index, len(holes)):
        return holes[cast(int, record.hole_index)]
    return None


def _drawn_ring(
    ring: _Ring, walks: Sequence[_Ring], grid_size_um: float
) -> tuple[tuple[float, float], ...] | None:
    present = {canonical_grid_point(point, grid_size_um) for point in ring}
    matches = []
    for walk in walks:
        nodes = _walk_nodes(walk, grid_size_um)
        # Each vertex once, where the walk first reaches it: a walk around a hole comes
        # back to the cut-line vertex it entered the hole by.
        visits = tuple(dict.fromkeys(nodes))
        matches.extend(
            tuple(node for node in visits if node in loop)
            for loop in _walk_loops(nodes)
            if loop <= present
        )
    # The same walk drawn on two layers names one drawn ring; two different walks name none.
    found = set(matches)
    return found.pop() if len(found) == 1 else None


def _walk_nodes(walk: _Ring, grid_size_um: float) -> list[tuple[float, float]]:
    # The walk on the grid, each vertex placed as the source stage places it; vertices
    # the rounding joins in a row are one vertex.
    nodes = [
        cast(tuple[float, float], canonical_grid_point(point, grid_size_um))
        for point in walk
    ]
    return [node for index, node in enumerate(nodes) if node != nodes[index - 1]]


def _walk_loops(
    nodes: Sequence[tuple[float, float]],
) -> list[frozenset[tuple[float, float]]]:
    # A walk that visits each vertex once is one loop.  A GDS polygon with holes is one
    # walk joined by cut lines; its loops are those the source reader splits it into.
    if len(nodes) < 3:
        return []
    if len(set(nodes)) == len(nodes):
        return [frozenset(nodes)]
    try:
        outer, holes = cutline_boundary_loops(tuple(nodes))
    except ValueError:
        return []
    return [
        frozenset((float(x), float(y)) for x, y in loop) for loop in (outer, *holes)
    ]


def _is_index(value: object, count: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value < count


def gds_walks(
    polygons_by_layer: Mapping[Any, Sequence[Any]],
    transform: tuple[float, ...] = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
) -> tuple[GdsWalk, ...]:
    """Each GDS polygon's vertices in the order the polygon walks them, carried through ``transform``.

    Each vertex is moved in its place, so a mirroring transform keeps the drawn sequence
    and only reverses its direction.
    """
    return tuple(
        tuple(
            cast(GdsPoint, tuple(transform_source_point(transform, point)))
            for point in polygon.points
        )
        for layer_polygons in polygons_by_layer.values()
        for polygon in layer_polygons
    )


def occurrence_gds_walks(
    instances: Mapping[str, tuple[Any, Any, tuple[float, ...]]],
    source_cells: Mapping[str, Any],
    child_reference_indexes: Mapping[str, tuple[int, ...]],
    entities: Sequence[SemanticEntitySpec],
) -> Callable[[GdsBoundaryReconstructionSpec], tuple[GdsWalk, ...]]:
    """The GDS polygon walks of a record's occurrence as drawn, in assembled coordinates.

    The occurrence is the record's own, or else that of the Entity it names.  Its cell's
    polygons are read as occurrence extraction reads them, registered child occurrences
    left out, and carried through the occurrence transform.
    """
    entity_paths = {
        entity.semantic_id: entity.metadata.get("source_occurrence_path")
        for entity in entities
    }
    walks_by_path: dict[str, tuple[GdsWalk, ...]] = {}

    def walks(record: GdsBoundaryReconstructionSpec) -> tuple[GdsWalk, ...]:
        path = record.source_occurrence_path or entity_paths.get(record.entity_id)
        if path is None or path not in instances:
            return ()
        if path not in walks_by_path:
            cell, _, transform = instances[path]
            polygons = read_polygons_by_layer(
                source_cells[cell.name],
                excluded_reference_indexes=child_reference_indexes[path],
            )
            walks_by_path[path] = gds_walks(polygons, transform)
        return walks_by_path[path]

    return walks


def bind_source_curves(
    prepared_input: GeometryBuildInput, source_input: GeometryBuildInput
) -> GeometryBuildInput:
    """Carry detached authored XY bindings across a prepared-stack Z update.

    Entity/source polygon identities stay authored. Stack preparation owns Z and
    solution regions; this helper never recovers curves from prepared surfaces.
    """
    active = {
        record.entity_id
        for record in (
            *source_input.boundary_curves,
            *source_input.boundary_reconstruction,
        )
    }
    source_entities = {entity.semantic_id: entity for entity in source_input.entities}
    prepared_entities = {
        entity.semantic_id: entity for entity in prepared_input.entities
    }
    split_hosts = _restore_split_source_entities(
        prepared_input, source_entities, prepared_entities, active
    )
    source_index = _SourceEntityIndex(
        active, source_entities, prepared_entities, split_hosts
    )
    restored_geometry = _restore_authored_geometry(
        prepared_input,
        source_input,
        source_index,
    )
    restored_ports = _restore_prepared_ports(prepared_input, source_input, split_hosts)
    return replace(
        prepared_input,
        polygons=restored_geometry.polygons,
        entities=restored_geometry.entities,
        port_sheet_regions=restored_ports,
        boundary_curves=copy.deepcopy(source_input.boundary_curves),
        boundary_reconstruction=copy.deepcopy(source_input.boundary_reconstruction),
    )


def _restore_authored_geometry(
    prepared_input: GeometryBuildInput,
    source_input: GeometryBuildInput,
    source_index: _SourceEntityIndex,
) -> _PreparedSourceGeometry:
    """Reattach authored polygons and geometry while retaining prepared Z state."""
    polygons = {polygon.polygon_id: polygon for polygon in prepared_input.polygons}
    source_polygons = {polygon.polygon_id: polygon for polygon in source_input.polygons}
    entities = []
    emitted = set()
    for prepared_entity in prepared_input.entities:
        entity_id = source_index.split_hosts.get(
            prepared_entity.semantic_id, prepared_entity.semantic_id
        )
        if entity_id in emitted:
            continue
        emitted.add(entity_id)
        entity = source_index.prepared_by_id[entity_id]
        if entity.semantic_id in source_index.active_ids:
            source = source_index.source_by_id[entity.semantic_id]
            for polygon_id in source.polygon_ids:
                polygons[polygon_id] = source_polygons[polygon_id]
            geometry = dict(entity.geometry)
            for key in ("outer_loop", "hole_loops", "source_occurrence_polygons_um"):
                if key in source.geometry:
                    geometry[key] = copy.deepcopy(source.geometry[key])
            entity = replace(entity, polygon_ids=source.polygon_ids, geometry=geometry)
        entities.append(entity)
    return _PreparedSourceGeometry(tuple(polygons.values()), tuple(entities))


def _restore_prepared_ports(
    prepared_input: GeometryBuildInput,
    source_input: GeometryBuildInput,
    split_hosts: Mapping[str, str],
) -> tuple[PortSheetRegionRecord, ...]:
    """Restore authored overlap rows where prepared hosts were split."""
    source_ports = {
        region.port_sheet_id: region for region in source_input.port_sheet_regions
    }
    return _restore_source_port_overlaps(
        prepared_input.port_sheet_regions, source_ports, split_hosts
    )


def _restore_split_source_entities(
    prepared_input: GeometryBuildInput,
    source_entities: Mapping[str, SemanticEntitySpec],
    prepared_entities: dict[str, SemanticEntitySpec],
    active: set[str],
) -> dict[str, str]:
    """Rejoin only prepared children carrying explicit source lineage."""
    split_hosts: dict[str, str] = {}
    for entity_id in sorted(active - prepared_entities.keys()):
        children = tuple(
            entity
            for entity in prepared_input.entities
            if entity.metadata.get("source_semantic_id") == entity_id
            and "split_polygon_index" in entity.metadata
        )
        if not children:
            raise ValueError("prepared stack lost a curve-bound source Entity")
        first = children[0]
        source = source_entities[entity_id]
        fields = tuple(
            name
            for name in first.__dataclass_fields__
            if name not in {"semantic_id", "polygon_ids", "geometry", "metadata"}
        )
        z_fields = ("z_um", "z_min_um", "z_max_um", "thickness_um")
        incompatible_children = any(
            any(getattr(child, name) != getattr(first, name) for name in fields)
            or any(
                child.geometry.get(name) != first.geometry.get(name)
                for name in z_fields
            )
            or child.metadata.get("source_occurrence_path")
            != source.metadata.get("source_occurrence_path")
            for child in children
        )
        if incompatible_children or first.net_id != source.net_id:
            raise ValueError(
                "prepared curve-bound Entity children have incompatible ownership"
            )
        metadata = dict(first.metadata)
        metadata.pop("split_polygon_index", None)
        prepared_entities[entity_id] = replace(
            first, semantic_id=entity_id, metadata=metadata
        )
        split_hosts.update((child.semantic_id, entity_id) for child in children)
    return split_hosts


def _restore_source_port_overlaps(
    prepared_regions: Sequence[PortSheetRegionRecord],
    source_ports: Mapping[str, PortSheetRegionRecord],
    split_hosts: Mapping[str, str],
) -> tuple[PortSheetRegionRecord, ...]:
    """Restore authored host overlap records where preparation split a host."""
    ports = []
    for region in prepared_regions:
        overlaps = []
        emitted_hosts: set[str] = set()
        for overlap in region.overlaps:
            host_id = split_hosts.get(overlap.host_semantic_id)
            if host_id is None:
                overlaps.append(overlap)
                continue
            if host_id in emitted_hosts:
                continue
            original = source_ports.get(region.port_sheet_id)
            matches = (
                ()
                if original is None
                else tuple(
                    item
                    for item in original.overlaps
                    if item.host_semantic_id == host_id
                )
            )
            if not matches:
                raise ValueError(
                    "prepared curve-bound port lost its source host overlap"
                )
            overlaps.extend(copy.deepcopy(matches))
            emitted_hosts.add(host_id)
        ports.append(replace(region, overlaps=tuple(overlaps)))
    return tuple(ports)
