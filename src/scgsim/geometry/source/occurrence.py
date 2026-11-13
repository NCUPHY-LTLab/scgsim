"""Lower the polygon and port selections of an Entity or an occurrence, for the stack adapter and the plan. Uses nothing from planning.

Every reader takes the ``GridScope`` of the layout grid it reads on. An occurrence's
recipes read its cell in the cell's own coordinates and place the scope in the
occurrence's frame where rounding or a refusal needs layout coordinates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from functools import wraps
from math import isfinite
from threading import get_ident
from types import MappingProxyType
from typing import Any

from scgsim.geometry.grid.polygon_ops import (
    FIX_SELECTOR,
    FusedComponent,
    flattened_polygons,
    fused_gdstk_components,
    point_in_layout_polygon,
    point_in_ring,
    realized_on_lattice,
    ring_from_gdstk_polygon,
    selector_place,
    without_repeated_vertices,
)
from scgsim.geometry.grid.scope import GridScope
from scgsim.geometry.models.input import LayoutPolygonSpec, SemanticEntitySpec
from scgsim.geometry.models.refusals import Declared, internal
from scgsim.geometry.source.ground import DrawnPlane, derived_ground_polygons
from scgsim.geometry.source.normalization import gds_layer
from scgsim.geometry.source.recipes import Audited, audited_declared_source
from scgsim.geometry.source.stack_layers import is_record_sequence


def _entity(entity: SemanticEntitySpec) -> Declared:
    """An Entity's stack layer, which declares its selector points and occurrence polygons."""
    return Declared(f"Entity {entity.semantic_id!r}", "its stack layer's geometry")


@dataclass
class _OccurrenceExtractionCache:
    owner_thread: int = field(default_factory=get_ident)
    entries: dict = field(default_factory=dict)


# Ambient because it cannot change a result: it only spares reading one cell's
# polygons twice within one adapter call.
_EXTRACTION_CACHE: ContextVar[_OccurrenceExtractionCache | None] = ContextVar(
    "scgsim_occurrence_extraction_cache", default=None
)


@contextmanager
def _occurrence_extraction_scope():
    """Own read-only extracted polygons for one fresh source-library lifetime."""
    token = _EXTRACTION_CACHE.set(_OccurrenceExtractionCache())
    try:
        yield
    finally:
        _EXTRACTION_CACHE.reset(token)


def with_occurrence_extraction_cache(function):
    """Give each decorated adapter call its own extraction-cache scope."""

    @wraps(function)
    def scoped(*args, **kwargs):
        # Nested adapter reads own another library and therefore get a fresh cache.
        with _occurrence_extraction_scope():
            return function(*args, **kwargs)

    return scoped


def read_polygons_by_layer(
    cell: Any, *, excluded_reference_indexes: Sequence[int] = ()
) -> Mapping[tuple[int, int], tuple[Any, ...]]:
    """Read layer polygons, reusing extraction within the current thread scope."""
    cache = _EXTRACTION_CACHE.get()
    if cache is None or cache.owner_thread != get_ident():
        return _extract_polygons_by_layer(
            cell, excluded_reference_indexes=excluded_reference_indexes
        )
    exclusions = tuple(excluded_reference_indexes)
    key = (id(cell), exclusions)
    found = cache.entries.get(key)
    if found is not None:
        return found[1]
    result = MappingProxyType(
        _extract_polygons_by_layer(cell, excluded_reference_indexes=exclusions)
    )
    # Retain the exact cell strongly: an id cannot be reused within this scope.
    # Polygons remain private/read-only until callers convert them to source rings.
    cache.entries[key] = (cell, result)
    return result


def _extract_polygons_by_layer(
    cell: Any, *, excluded_reference_indexes: Sequence[int] = ()
) -> dict[tuple[int, int], tuple[Any, ...]]:
    result: dict[tuple[int, int], list[Any]] = {}
    # Canonical public PDK cells use hierarchy.  SGB's Level-0 contract is
    # flattened layout polygons, so preserve every referenced source polygon
    # rather than silently lowering only the top cell's direct shapes.
    polygons = flattened_polygons(
        cell, excluded_reference_indexes=excluded_reference_indexes
    )
    for polygon in without_repeated_vertices(polygons):
        result.setdefault(
            (int(polygon.layer), int(polygon.datatype)),
            [],
        ).append(polygon)
    return {key: tuple(value) for key, value in result.items()}


def gds_polygons_for_entity(
    scope: GridScope,
    entity: SemanticEntitySpec,
    *,
    polygons_by_layer: Mapping[tuple[int, int], tuple[Any, ...]],
) -> tuple[LayoutPolygonSpec, ...]:
    """Read an entity's layout polygons from occurrence data or GDS layers.

    A declared source recipe: its callers run it through ``audited_declared_source``.
    """
    occurrence_polygons = entity.geometry.get("source_occurrence_polygons_um")
    if occurrence_polygons is not None:
        return _source_occurrence_polygons(entity, occurrence_polygons)

    layer = int(entity.geometry["gds_layer"])
    datatype = int(entity.geometry["gds_datatype"])
    named = scope.naming(entity.semantic_id, f"{layer}/{datatype}")
    return _gds_layer_polygons(named, entity, polygons_by_layer, layer, datatype)


def _gds_layer_polygons(
    scope: GridScope,
    entity: SemanticEntitySpec,
    polygons_by_layer: Mapping[tuple[int, int], tuple[Any, ...]],
    layer: int,
    datatype: int,
) -> tuple[LayoutPolygonSpec, ...]:
    """The entity's polygons on its GDS layer, fused and selected as it declares."""
    candidates = realized_on_lattice(
        scope, polygons_by_layer.get((layer, datatype), ())
    )
    if not candidates:
        return ()
    selector = entity.geometry.get("selector_point_um")
    if not entity.geometry.get("route_ab_fused_selector_mode", False):
        return _source_layer_polygons(scope, entity, candidates, layer, datatype)
    excluded = _excluded_selector_points(entity)
    components = fused_gdstk_components(
        scope, _ordered_source_polygons(scope, candidates)
    )
    points = _fused_selector_points(entity, selector, excluded)
    matched = _matched_components(scope, entity, components, points)
    if selector is not None:
        selected = [components[matched[0]]]
    else:
        selected = [
            component
            for index, component in enumerate(components)
            if index not in matched
        ]
    return tuple(
        _fused_layout_polygon(entity, component, index, layer, datatype)
        for index, component in enumerate(selected)
    )


def _excluded_selector_points(entity: SemanticEntitySpec) -> Sequence[Any]:
    """The points whose fused regions the entity leaves out, as declared."""
    excluded = entity.geometry.get("exclude_selector_points_um", ())
    if excluded is None:
        excluded = ()
    if isinstance(excluded, str | bytes) or not isinstance(excluded, Sequence):
        raise TypeError(
            _entity(entity).expected(
                "exclude_selector_points_um", "a point sequence", excluded
            )
        )
    return excluded


def _fused_selector_points(
    entity: SemanticEntitySpec, selector: Any, excluded: Sequence[Any]
) -> tuple[tuple[float, float], ...]:
    """The selector point the entity keeps, then each point it excludes, in declared order."""
    return (
        *(
            (_selector_point(entity, "selector_point_um", selector),)
            if selector is not None
            else ()
        ),
        *(
            _selector_point(entity, "exclude_selector_points_um", point)
            for point in excluded
        ),
    )


def _matched_components(
    scope: GridScope,
    entity: SemanticEntitySpec,
    components: Sequence[FusedComponent],
    points: Sequence[tuple[float, float]],
) -> list[int]:
    """The component each selector point lies in; each point must lie in exactly one, all distinct."""
    matched: list[int] = []
    for point in points:
        indexes = [
            index
            for index, component in enumerate(components)
            if point_in_layout_polygon(point, component.polygon)
        ]
        if len(indexes) != 1:
            raise ValueError(
                scope.refusal(
                    f"{entity.semantic_id} selector point {point!r} matched "
                    f"{len(indexes)} fused polygons",
                    [
                        selector_place(
                            scope,
                            point,
                            [components[i].polygon.exterior for i in indexes],
                            [component.polygon.exterior for component in components],
                        )
                    ],
                    FIX_SELECTOR,
                )
            )
        if indexes[0] in matched:
            earlier = points[matched.index(indexes[0])]
            region = scope.place(components[indexes[0]].polygon.exterior)
            raise ValueError(
                scope.refusal(
                    f"{entity.semantic_id} selector points must match distinct polygons",
                    [
                        f"the selector points at {scope.place([earlier])} and "
                        f"{scope.place([point])} both lie in the region at {region}"
                    ],
                    "Fix the declaration or the layout: give each selector point a region "
                    "of its own, or draw the regions apart.",
                )
            )
        matched.append(indexes[0])
    return matched


def _fused_layout_polygon(
    entity: SemanticEntitySpec,
    component: FusedComponent,
    index: int,
    layer: int,
    datatype: int,
) -> LayoutPolygonSpec:
    """One selected fused component as the entity's layout polygon, naming the drawn polygons it fuses."""
    return LayoutPolygonSpec(
        polygon_id=f"{entity.semantic_id}__P{index:04d}",
        layer=f"{layer}/{datatype}",
        exterior=component.polygon.exterior,
        holes=component.polygon.holes,
        object_name=entity.semantic_id,
        net_name=entity.net_id,
        metadata={
            "gds_layer": layer,
            "gds_datatype": datatype,
            "source": "gds_polygon",
            "source_polygon_indexes": component.source_indexes,
            "source_polygon_ids": tuple(
                f"POLY__{entity.semantic_id}__SOURCE__{source_index:04d}"
                for source_index in component.source_indexes
            ),
            "source_area_um2": component.source_area_um2,
        },
    )


def _source_occurrence_polygons(
    entity: SemanticEntitySpec, records: Any
) -> tuple[LayoutPolygonSpec, ...]:
    if not is_record_sequence(records):
        raise TypeError(
            _entity(entity).expected(
                "source_occurrence_polygons_um", "a sequence", records
            )
        )
    return tuple(
        LayoutPolygonSpec(
            polygon_id=f"{entity.semantic_id}__P{index:04d}",
            layer=f"{entity.geometry['gds_layer']}/{entity.geometry['gds_datatype']}",
            exterior=record["exterior"],
            holes=record["holes"],
            object_name=entity.semantic_id,
            net_name=entity.net_id,
            metadata={
                "source": "gds_source_occurrence",
                "source_polygon_index": index,
            },
        )
        for index, record in enumerate(records)
    )


def _source_layer_polygons(
    scope: GridScope,
    entity: SemanticEntitySpec,
    candidates: Sequence[Any],
    layer: int,
    datatype: int,
) -> tuple[LayoutPolygonSpec, ...]:
    """The drawn polygons the entity's selector point lies in, or all of them without one."""
    selector = entity.geometry.get("selector_point_um")
    if selector is not None:
        point = (float(selector[0]), float(selector[1]))
        selected = [
            polygon
            for polygon in candidates
            if point_in_ring(point, ring_from_gdstk_polygon(scope, polygon))
        ]
        if len(selected) != 1:
            raise ValueError(
                scope.refusal(
                    f"{entity.semantic_id} selector_point_um matched {len(selected)} polygons",
                    [
                        selector_place(
                            scope,
                            point,
                            [
                                ring_from_gdstk_polygon(scope, polygon)
                                for polygon in selected
                            ],
                            [
                                ring_from_gdstk_polygon(scope, polygon)
                                for polygon in candidates
                            ],
                        )
                    ],
                    FIX_SELECTOR,
                )
            )
    else:
        selected = list(candidates)
    return tuple(
        LayoutPolygonSpec(
            polygon_id=f"POLY__{entity.semantic_id}__{index:04d}",
            layer=f"{layer}/{datatype}",
            exterior=ring_from_gdstk_polygon(scope, polygon),
            object_name=entity.semantic_id,
            net_name=entity.net_id,
        )
        for index, polygon in enumerate(selected)
    )


def _selector_point(
    entity: SemanticEntitySpec, name: str, value: Any
) -> tuple[float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str | bytes)
        or len(value) != 2
    ):
        raise ValueError(_entity(entity).expected(name, "a 2D point", value))
    return (float(value[0]), float(value[1]))


def _ordered_source_polygons(
    scope: GridScope, polygons: Sequence[Any]
) -> tuple[Any, ...]:
    return tuple(
        sorted(
            polygons,
            key=lambda polygon: (
                min(point[0] for point in ring_from_gdstk_polygon(scope, polygon)),
                min(point[1] for point in ring_from_gdstk_polygon(scope, polygon)),
                polygon.area(),
            ),
        )
    )


def _occurrence_point(
    transform: Sequence[float], point: Sequence[float]
) -> list[float]:
    a, b, c, d, x, y = (float(value) for value in transform)
    px, py = float(point[0]), float(point[1])
    return [a * px + b * py + x, c * px + d * py + y]


def _occurrence_inverse_point(
    transform: Sequence[float], point: Sequence[float]
) -> list[float]:
    a, b, c, d, x, y = (float(value) for value in transform)
    determinant = a * d - b * c
    if not isfinite(determinant) or abs(determinant) < 1e-12:
        raise ValueError(
            internal(f"an occurrence transform {tuple(transform)} is singular")
        )
    px, py = float(point[0]) - x, float(point[1]) - y
    return [(d * px - b * py) / determinant, (-c * px + a * py) / determinant]


def _occurrence_polygon_record(
    polygon: LayoutPolygonSpec, transform: Sequence[float]
) -> dict[str, Any]:
    return {
        "exterior": [_occurrence_point(transform, point) for point in polygon.exterior],
        "holes": [
            [_occurrence_point(transform, point) for point in hole]
            for hole in polygon.holes
        ],
    }


def _local_selector_geometry(
    entity: SemanticEntitySpec, transform: Sequence[float]
) -> dict[str, Any]:
    """The entity's geometry with its selector points mapped into the occurrence's cell."""
    geometry = dict(entity.geometry)
    geometry.pop("source_occurrence_polygons_um", None)
    for key in ("selector_point_um",):
        if key in geometry:
            geometry[key] = _occurrence_inverse_point(transform, geometry[key])
    for key in ("include_selector_points_um", "exclude_selector_points_um"):
        if key in geometry:
            geometry[key] = [
                _occurrence_inverse_point(transform, point) for point in geometry[key]
            ]
    return geometry


def occurrence_polygons_for_entity(
    scope: GridScope,
    entity: SemanticEntitySpec,
    *,
    transform: Sequence[float],
    polygons_by_layer: Mapping[tuple[int, int], tuple[Any, ...]],
) -> Audited[list[dict[str, Any]]]:
    """Transform the selected entity polygons into occurrence coordinates.

    ``polygons_by_layer`` are the polygons the occurrence's cell draws itself, in its
    own coordinates; ``transform`` places them in the layout.
    """
    local_entity = replace(entity, geometry=_local_selector_geometry(entity, transform))
    selected = audited_declared_source(
        gds_polygons_for_entity,
        scope.placed(transform),
        local_entity,
        polygons_by_layer=polygons_by_layer,
    )
    return Audited(
        [_occurrence_polygon_record(polygon, transform) for polygon in selected.value],
        selected.audits,
    )


def occurrence_ground_polygons_for_entity(
    scope: GridScope,
    entity: SemanticEntitySpec,
    *,
    transform: Sequence[float],
    plane_layer: Sequence[int],
    polygons_by_layer: Mapping[tuple[int, int], tuple[Any, ...]],
) -> Audited[list[dict[str, Any]]]:
    """Derive an occurrence's own positive Ground before its one affine map.

    The explicit PDK plane layer supplies the actual local outline; registered
    descendants supply neither that outline nor the local mask/include recipe.
    """
    placed = scope.placed(transform)
    geometry = _local_selector_geometry(entity, transform)
    plane_key = gds_layer(
        plane_layer,
        Declared(f"plane of Ground {entity.semantic_id!r}", "the PDK layer stack"),
        "source_layer_datatype",
    )
    plane = polygons_by_layer.get(plane_key, ())
    if not plane:
        named = placed.naming(entity.semantic_id, f"{plane_key[0]}/{plane_key[1]}")
        raise ValueError(
            named.refusal(
                f"{entity.semantic_id} occurrence has no declared plane-bound geometry",
                [_drawn_without(named, polygons_by_layer)],
                "Fix the layout or the declaration: draw the Ground's plane on this "
                "layer in the component, or name the layer its plane is drawn on.",
            )
        )
    derived = audited_declared_source(
        derived_ground_polygons,
        placed,
        replace(entity, geometry=geometry),
        polygons_by_layer=polygons_by_layer,
        plane=DrawnPlane(plane, plane_layer),
    )
    return Audited(
        [_occurrence_polygon_record(polygon, transform) for polygon in derived.value],
        derived.audits,
    )


def occurrence_port_polygon(
    scope: GridScope,
    record: Mapping[str, Any],
    *,
    transform: Sequence[float],
    polygons_by_layer: Mapping[tuple[int, int], tuple[Any, ...]],
) -> Audited[dict[str, Any]]:
    """Read and transform the polygon selected by one occurrence port record.

    The sheet is selected in the cell's own coordinates; a refusal locates it in the layout.
    """
    layer = (record["layer"], record["datatype"])
    point = _occurrence_inverse_point(transform, record["selector_point_um"])
    selected = [
        polygon
        for polygon in polygons_by_layer.get(layer, ())
        if point_in_ring(point, ring_from_gdstk_polygon(scope, polygon))
    ]
    if len(selected) != 1:
        drawn = [
            ring_from_gdstk_polygon(scope, p) for p in polygons_by_layer.get(layer, ())
        ]
        named = scope.placed(transform).naming(
            str(record["name"]), f"{layer[0]}/{layer[1]}"
        )
        raise ValueError(
            named.refusal(
                f"GeometryPlan port {record['name']!r} matched {len(selected)} local "
                "sheet polygons",
                [
                    selector_place(
                        named,
                        point,
                        [ring_from_gdstk_polygon(named, p) for p in selected],
                        drawn,
                        what="the port's centre",
                    )
                ],
                "Fix the layout: draw the port's sheet as one polygon on its layer, "
                "under the port's centre.",
            )
        )
    local = LayoutPolygonSpec(
        polygon_id="source_port_sheet",
        layer=f"{layer[0]}/{layer[1]}",
        exterior=ring_from_gdstk_polygon(scope, selected[0]),
    )
    return Audited(_occurrence_polygon_record(local, transform))


def occurrence_include_polygon(
    scope: GridScope,
    record: Mapping[str, Any],
    *,
    transform: Sequence[float],
    polygons_by_layer: Mapping[tuple[int, int], tuple[Any, ...]],
) -> Audited[dict[str, Any]]:
    """Select one connected local metal region, including its authored pieces.

    The region is selected in the cell's own coordinates; a refusal locates it in the layout.
    """
    layer = gds_layer(
        record.get("layer"),
        Declared(
            f"ground contribution {record.get('source_local_id')!r}",
            "the component's component_semantics ground_plane_contributions",
        ),
        "layer",
    )
    point = _occurrence_inverse_point(transform, record["selector_point_um"])
    components = fused_gdstk_components(scope, polygons_by_layer.get(layer, ()))
    selected = [
        component.polygon
        for component in components
        if point_in_layout_polygon((point[0], point[1]), component.polygon)
    ]
    if len(selected) != 1:
        name = str(record.get("source_local_id", "ground contribution"))
        named = scope.placed(transform).naming(name, f"{layer[0]}/{layer[1]}")
        raise ValueError(
            named.refusal(
                f"ground contribution {record.get('source_local_id')!r} matched "
                f"{len(selected)} local source polygons",
                [
                    selector_place(
                        named,
                        point,
                        [polygon.exterior for polygon in selected],
                        [component.polygon.exterior for component in components],
                    )
                ],
                FIX_SELECTOR,
            )
        )
    return Audited(_occurrence_polygon_record(selected[0], transform))


def _drawn_without(
    scope: GridScope, polygons_by_layer: Mapping[tuple[int, int], Sequence[Any]]
) -> str:
    """Where an occurrence draws, when it draws nothing on the layer a refusal names."""
    corners = [
        corner
        for polygons in polygons_by_layer.values()
        for polygon in polygons
        for corner in polygon.bounding_box()
    ]
    if not corners:
        return "the occurrence draws nothing"
    return (
        "the occurrence draws nothing on this layer; what it draws lies within "
        f"{scope.place(corners)}"
    )
