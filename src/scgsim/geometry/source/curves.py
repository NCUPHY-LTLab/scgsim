"""Normalize authored boundaries, bind them to source rings, and restore them after stack preparation.

This module owns source XY curve identity, including each source polygon's
drawn GDS rings (``LayoutPolygonSpec.drawn_rings``), whose vertices
reconstruction indices name. Prepared geometry owns its Z, materials, and
solution-region state.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, cast, overload

from scgsim.geometry.models.common import PolygonRing
from scgsim.geometry.models.grid import geometry_grid_context
from scgsim.geometry.models.input import (
    BoundaryCurveChainSpec,
    GdsBoundaryReconstructionSpec,
    GeometryBuildInput,
    LayoutPolygonSpec,
    SemanticEntitySpec,
    SourceCurveSpec,
)
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
    emitted: GeometryBuildInput,
    drawn_walks_of: Callable[[LayoutPolygonSpec], DrawnWalks | None],
) -> GeometryBuildInput:
    """The input with each polygon the source stage read carrying its ``drawn_rings``.

    The source stage reads each polygon from the GDS walk its author drew, then fuses,
    rounds and nodes it: the ring it emits may start at another vertex, run the other way,
    and hold crossings the grid inserted.  Each ring's drawn order is found here once, in
    the walks ``drawn_walks_of`` gives for the polygon's occurrence (see
    :meth:`DrawnWalks.drawn_ring`).  A polygon whose occurrence has no walks keeps the
    field empty.  Reconstruction records are left exactly as their authors declared them.
    """
    polygons = []
    for polygon in emitted.polygons:
        walks = drawn_walks_of(polygon)
        if walks is not None:
            polygon = replace(
                polygon,
                drawn_rings=tuple(
                    walks.drawn_ring(ring)
                    for ring in (polygon.exterior, *polygon.holes)
                ),
            )
        polygons.append(polygon)
    return replace(emitted, polygons=tuple(polygons))


@dataclass(frozen=True)
class DrawnWalks:
    """The loops of one occurrence's GDS polygon walks as drawn, found by their vertices.

    Each loop holds its vertices in the order the walk first visits them, placed on the
    model grid as the source stage places them.  A walk that visits each vertex once is
    one loop; a GDS polygon with holes is one walk joined by cut lines, and its loops are
    those the source reader splits it into.
    """

    loops: tuple[tuple[GdsPoint, ...], ...]
    by_vertex: Mapping[GdsPoint, tuple[int, ...]]

    @classmethod
    def read(cls, walks: Sequence[_Ring], grid_size_um: float) -> DrawnWalks:
        """Split each walk into its loops and index them by vertex."""
        loops: list[tuple[GdsPoint, ...]] = []
        for walk in walks:
            nodes = _walk_nodes(walk, grid_size_um)
            # A walk around a hole comes back to the cut-line vertex it entered it by.
            visits = tuple(dict.fromkeys(nodes))
            loops.extend(
                tuple(node for node in visits if node in loop)
                for loop in _walk_loops(nodes)
            )
        by_vertex: dict[GdsPoint, list[int]] = {}
        for index, loop in enumerate(loops):
            for node in loop:
                by_vertex.setdefault(node, []).append(index)
        return cls(
            tuple(loops), {node: tuple(found) for node, found in by_vertex.items()}
        )

    def drawn_ring(self, ring: _Ring) -> PolygonRing | None:
        """The drawn order of an emitted ring, or None when it is not one drawn loop.

        It is the one loop whose vertices all lie on the ring; the ring may hold more,
        the crossings the grid noded into it.  The same loop drawn on two layers is one
        drawn ring; two different loops, or none, leave the ring without one.
        """
        present = {cast(GdsPoint, tuple(map(float, point))) for point in ring}
        candidates = {i for node in present for i in self.by_vertex.get(node, ())}
        found = {self.loops[i] for i in candidates if set(self.loops[i]) <= present}
        return found.pop() if len(found) == 1 else None


def _walk_nodes(walk: _Ring, grid_size_um: float) -> list[GdsPoint]:
    # The walk on the grid, each vertex placed as the source stage places it; vertices
    # the rounding joins in a row are one vertex.
    nodes = [
        cast(GdsPoint, canonical_grid_point(point, grid_size_um)) for point in walk
    ]
    return [node for index, node in enumerate(nodes) if node != nodes[index - 1]]


def _walk_loops(nodes: Sequence[GdsPoint]) -> list[frozenset[GdsPoint]]:
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


def occurrence_drawn_walks(
    instances: Mapping[str, tuple[Any, Any, tuple[float, ...]]],
    source_cells: Mapping[str, Any],
    child_reference_indexes: Mapping[str, tuple[int, ...]],
    build_input: GeometryBuildInput,
) -> Callable[[LayoutPolygonSpec], DrawnWalks | None]:
    """The drawn walks of the occurrence each polygon of a GeometryPlan belongs to.

    A polygon's occurrence is that of its Entity.  The occurrence cell's polygons are read
    as occurrence extraction reads them, registered child occurrences left out, carried
    through the occurrence transform, and read once per occurrence.
    """
    paths = {
        polygon_id: entity.metadata.get("source_occurrence_path")
        for entity in build_input.entities
        for polygon_id in entity.polygon_ids
    }
    by_path: dict[str, DrawnWalks] = {}

    def drawn_walks(polygon: LayoutPolygonSpec) -> DrawnWalks | None:
        path = paths.get(polygon.polygon_id)
        if path is None or path not in instances:
            return None
        if path not in by_path:
            context = geometry_grid_context(build_input)
            if context is None:
                return None
            cell, _, transform = instances[path]
            polygons = read_polygons_by_layer(
                source_cells[cell.name],
                excluded_reference_indexes=child_reference_indexes[path],
            )
            by_path[path] = DrawnWalks.read(
                gds_walks(polygons, transform), context.grid_size_um
            )
        return by_path[path]

    return drawn_walks


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
