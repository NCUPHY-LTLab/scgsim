"""Lower source Entity and occurrence polygon/port selections for adapter and plan. Never borrow compiler planning helpers."""

from __future__ import annotations

from scgsim.geometry._primitives.grid import (
    in_occurrence_frame,
    preserve_declared_source,
)

from collections.abc import Mapping, Sequence
from copy import deepcopy
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from functools import wraps
from threading import get_ident
from types import MappingProxyType
from math import isfinite
from typing import Any

from scgsim.geometry.models.input import LayoutPolygonSpec, SemanticEntitySpec
from scgsim.geometry.source._normalization import _gds_layer
from scgsim.geometry.source._polygon_ops import (
    _fused_gdstk_components,
    _point_in_layout_polygon,
    _point_in_ring,
    _ring_from_gdstk_polygon,
)


@dataclass(frozen=True)
class _SourceRecipeJob:
    function: Any = None
    args: tuple = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    expected: Any = None
    comparison_error: str | None = None
    error: Exception | None = None
    traceback: Any = None


def _source_recipe_job(
    function,
    *args,
    source_cell=None,
    source_polygons=None,
    excluded_reference_indexes=(),
    expected=None,
    comparison_error=None,
    **kwargs,
):
    """Prepare worker-owned inputs on the source thread, before pool dispatch."""
    if source_cell is not None:
        source_polygons = _polygons_by_layer(
            source_cell, excluded_reference_indexes=excluded_reference_indexes
        )
        kwargs["cell"] = None
    copied = deepcopy(kwargs)
    if source_polygons is not None:
        copied["polygons_by_layer"] = {
            layer: tuple(polygon.copy() for polygon in polygons)
            for layer, polygons in source_polygons.items()
        }
    return _SourceRecipeJob(
        function, deepcopy(args), copied, deepcopy(expected), comparison_error
    )


def _failed_source_recipe(error):
    # Preparation errors join the same source-order failure stream as recipes.
    return _SourceRecipeJob(error=error, traceback=error.__traceback__)


def _run_source_recipe(job):
    if job.error is not None:
        raise job.error.with_traceback(job.traceback)
    result = job.function(*job.args, **job.kwargs)
    if job.comparison_error is not None and result != job.expected:
        raise ValueError(job.comparison_error)
    return result


def _resolve_material_kind(
    record: Mapping[str, Any],
    *,
    material_id: Any,
    materials: Mapping[str, Any],
    context: str,
) -> str:
    """Resolve one entity kind from explicit record and stack material facts."""
    if not isinstance(material_id, str) or not material_id:
        raise ValueError(f"{context} needs a non-empty material_id")
    material = materials.get(material_id)
    if not isinstance(material, Mapping):
        raise TypeError(f"{context} material_id {material_id!r} is not in materials")
    stack_kind = material.get("kind")
    if stack_kind not in {"vacuum", "dielectric", "conductor"}:
        raise ValueError(
            f"{context} material {material_id!r} needs kind vacuum, dielectric, or conductor"
        )
    record_kind = record.get("material_kind")
    if record_kind is not None and record_kind not in {
        "vacuum",
        "dielectric",
        "conductor",
    }:
        raise ValueError(
            f"{context} material_kind must be vacuum, dielectric, or conductor"
        )
    if record_kind is not None and record_kind != stack_kind:
        raise ValueError(
            f"{context} material_kind disagrees with materials[{material_id!r}].kind"
        )
    return str(stack_kind)


def _is_record_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes)


@dataclass
class _OccurrenceExtractionCache:
    owner_thread: int = field(default_factory=get_ident)
    entries: dict = field(default_factory=dict)


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


def _with_occurrence_extraction_cache(function):
    @wraps(function)
    def scoped(*args, **kwargs):
        # Nested adapter reads own another library and therefore get a fresh cache.
        with _occurrence_extraction_scope():
            return function(*args, **kwargs)

    return scoped


def _polygons_by_layer(
    cell: Any, *, excluded_reference_indexes: Sequence[int] = ()
) -> Mapping[tuple[int, int], tuple[Any, ...]]:
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
    if excluded_reference_indexes:
        polygons = list(cell.get_polygons(apply_repetitions=True, depth=0))
        excluded = set(excluded_reference_indexes)
        for index, reference in enumerate(cell.references):
            if index not in excluded:
                polygons.extend(
                    reference.get_polygons(apply_repetitions=True, depth=None)
                )
    else:
        polygons = cell.get_polygons(apply_repetitions=True, depth=None)
    for polygon in polygons:
        result.setdefault(
            (int(polygon.layer), int(polygon.datatype)),
            [],
        ).append(polygon)
    return {key: tuple(value) for key, value in result.items()}


def _entity_from_layer_record(
    record: Any, *, materials: Mapping[str, Any]
) -> SemanticEntitySpec:
    if not isinstance(record, Mapping):
        raise TypeError("stack_file 'layers' items must be mappings")

    layer = record.get("layer")
    datatype = record.get("datatype")
    if layer is None or datatype is None:
        raise ValueError("stack_file layer records must define 'layer' and 'datatype'")

    for required_field in ("semantic_id", "role", "material_id"):
        if required_field not in record:
            raise ValueError(f"stack_file layer records must define {required_field!r}")

    raw_geometry = record.get("geometry", {})
    if not isinstance(raw_geometry, Mapping):
        raise TypeError("stack_file layer record 'geometry' must be a mapping")
    geometry = dict(raw_geometry)
    if "z_um" in record or "thickness_um" in record:
        if "z_um" not in record or "thickness_um" not in record:
            raise ValueError(
                "stack_file layer records must define both 'z_um' and "
                "'thickness_um', or use 'geometry'"
            )
        geometry.setdefault("z_um", record["z_um"])
        geometry.setdefault("thickness_um", record["thickness_um"])
    if not geometry:
        raise ValueError(
            "stack_file layer records must define 'geometry' or "
            "'z_um' plus 'thickness_um'"
        )
    geometry.setdefault("gds_layer", layer)
    geometry.setdefault("gds_datatype", datatype)

    polygon_ids = record.get("polygon_ids", ())
    labels = record.get("labels", ())
    if isinstance(polygon_ids, str | bytes) or not isinstance(polygon_ids, Sequence):
        raise TypeError("stack_file layer record 'polygon_ids' must be a sequence")
    if isinstance(labels, str | bytes) or not isinstance(labels, Sequence):
        raise TypeError("stack_file layer record 'labels' must be a sequence")

    return SemanticEntitySpec(
        semantic_id=record["semantic_id"],
        role=record["role"],
        material_id=record["material_id"],
        material_kind=_resolve_material_kind(
            record,
            material_id=record["material_id"],
            materials=materials,
            context=f"layer {record['semantic_id']!r}",
        ),
        priority=record.get("priority", 0),
        geometry_kind=record.get("geometry_kind", "layout_extrusion"),
        part_role=record.get("part_role"),
        attached_face_metal_semantic_id=record.get("attached_face_metal_semantic_id"),
        net_id=record.get("net_id"),
        polygon_ids=tuple(polygon_ids),
        labels=tuple(labels),
        host_void_semantic_id=record.get("host_void_semantic_id"),
        requires_construction_body=record.get("requires_construction_body", False),
        route_representations=record.get("route_representations", {}),
        geometry=geometry,
        metadata=record.get("metadata", {}),
    )


@preserve_declared_source
def _gds_polygons_for_entity(
    entity: SemanticEntitySpec,
    *,
    polygons_by_layer: Mapping[tuple[int, int], tuple[Any, ...]],
) -> tuple[LayoutPolygonSpec, ...]:
    occurrence_polygons = entity.geometry.get("source_occurrence_polygons_um")
    if occurrence_polygons is not None:
        if not _is_record_sequence(occurrence_polygons):
            raise TypeError(
                f"{entity.semantic_id} occurrence polygons must be a sequence"
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
            for index, record in enumerate(occurrence_polygons)
        )

    def selector_point(name: str, value: Any) -> tuple[float, float]:
        if (
            not isinstance(value, Sequence)
            or isinstance(value, str | bytes)
            or len(value) != 2
        ):
            raise ValueError(f"{entity.semantic_id} {name} must be a 2D point")
        return (float(value[0]), float(value[1]))

    def ordered(polygons: Sequence[Any]) -> tuple[Any, ...]:
        return tuple(
            sorted(
                polygons,
                key=lambda polygon: (
                    min(point[0] for point in _ring_from_gdstk_polygon(polygon)),
                    min(point[1] for point in _ring_from_gdstk_polygon(polygon)),
                    polygon.area(),
                ),
            )
        )

    layer = int(entity.geometry["gds_layer"])
    datatype = int(entity.geometry["gds_datatype"])
    candidates = polygons_by_layer.get((layer, datatype), ())
    if not candidates:
        return ()

    selector = entity.geometry.get("selector_point_um")
    if not entity.geometry.get("route_ab_fused_selector_mode", False):
        if selector is not None:
            selected = [
                polygon
                for polygon in candidates
                if _point_in_ring(
                    (float(selector[0]), float(selector[1])),
                    _ring_from_gdstk_polygon(polygon),
                )
            ]
            if len(selected) != 1:
                raise ValueError(
                    f"{entity.semantic_id} selector_point_um matched "
                    f"{len(selected)} polygons"
                )
        else:
            selected = list(candidates)
        return tuple(
            LayoutPolygonSpec(
                polygon_id=f"POLY__{entity.semantic_id}__{index:04d}",
                layer=f"{layer}/{datatype}",
                exterior=_ring_from_gdstk_polygon(polygon),
                object_name=entity.semantic_id,
                net_name=entity.net_id,
            )
            for index, polygon in enumerate(selected)
        )

    excluded = entity.geometry.get("exclude_selector_points_um", ())
    if excluded is None:
        excluded = ()
    if isinstance(excluded, str | bytes) or not isinstance(excluded, Sequence):
        raise TypeError(
            f"{entity.semantic_id} exclude_selector_points_um must be a point sequence"
        )
    components = _fused_gdstk_components(ordered(candidates))
    points = (
        *(
            (selector_point("selector_point_um", selector),)
            if selector is not None
            else ()
        ),
        *(selector_point("exclude_selector_points_um", point) for point in excluded),
    )
    matched: list[int] = []
    for point in points:
        indexes = [
            index
            for index, (polygon, _, _) in enumerate(components)
            if _point_in_layout_polygon(point, polygon)
        ]
        if len(indexes) != 1:
            raise ValueError(
                f"{entity.semantic_id} selector point {point!r} matched "
                f"{len(indexes)} fused polygons"
            )
        if indexes[0] in matched:
            raise ValueError(
                f"{entity.semantic_id} selector points must match distinct polygons"
            )
        matched.append(indexes[0])
    if selector is not None:
        selected = [components[matched[0]]]
    else:
        selected = [
            component
            for index, component in enumerate(components)
            if index not in matched
        ]

    return tuple(
        LayoutPolygonSpec(
            polygon_id=f"{entity.semantic_id}__P{index:04d}",
            layer=f"{layer}/{datatype}",
            exterior=polygon.exterior,
            holes=polygon.holes,
            object_name=entity.semantic_id,
            net_name=entity.net_id,
            metadata={
                "gds_layer": layer,
                "gds_datatype": datatype,
                "source": "gds_polygon",
                "source_polygon_indexes": source_indexes,
                "source_polygon_ids": tuple(
                    f"POLY__{entity.semantic_id}__SOURCE__{source_index:04d}"
                    for source_index in source_indexes
                ),
                "source_area_um2": source_area,
            },
        )
        for index, (polygon, source_indexes, source_area) in enumerate(selected)
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
        raise ValueError("GeometryPlan occurrence transform is singular")
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


@in_occurrence_frame
def _occurrence_polygons_for_entity(
    entity: SemanticEntitySpec,
    *,
    cell: Any,
    transform: Sequence[float],
    excluded_reference_indexes: Sequence[int] = (),
    polygons_by_layer: Mapping[tuple[int, int], tuple[Any, ...]] | None = None,
) -> list[dict[str, Any]]:
    if polygons_by_layer is None:
        polygons_by_layer = _polygons_by_layer(
            cell, excluded_reference_indexes=excluded_reference_indexes
        )
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
    local_entity = replace(entity, geometry=geometry)
    selected = _gds_polygons_for_entity(
        local_entity,
        polygons_by_layer=polygons_by_layer,
    )
    return [_occurrence_polygon_record(polygon, transform) for polygon in selected]


@in_occurrence_frame
def _occurrence_ground_polygons_for_entity(
    entity: SemanticEntitySpec,
    *,
    cell: Any,
    transform: Sequence[float],
    plane_layer: Sequence[int],
    excluded_reference_indexes: Sequence[int] = (),
    polygons_by_layer: Mapping[tuple[int, int], tuple[Any, ...]] | None = None,
) -> list[dict[str, Any]]:
    """Derive an occurrence's own positive Ground before its one affine map.

    The explicit PDK plane layer supplies the actual local outline; registered
    descendants supply neither that outline nor the local mask/include recipe.
    """
    if polygons_by_layer is None:
        polygons_by_layer = _polygons_by_layer(
            cell, excluded_reference_indexes=excluded_reference_indexes
        )
    from scgsim.geometry.source.ground import _derived_ground_polygons

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
    polygons = polygons_by_layer
    plane = polygons.get(_gds_layer(plane_layer, entity.semantic_id), ())
    if not plane:
        raise ValueError(
            f"{entity.semantic_id} occurrence has no declared plane-bound geometry"
        )
    derived = _derived_ground_polygons(
        replace(entity, geometry=geometry),
        polygons_by_layer=polygons,
        cell_bounds_um={},
        domain_bounds_by_semantic_id={},
        plane_polygons=plane,
    )
    return [_occurrence_polygon_record(polygon, transform) for polygon in derived]


def _occurrence_port_polygon(
    record: Mapping[str, Any],
    *,
    cell: Any,
    transform: Sequence[float],
    excluded_reference_indexes: Sequence[int] = (),
    polygons_by_layer: Mapping[tuple[int, int], tuple[Any, ...]] | None = None,
) -> dict[str, Any]:
    if polygons_by_layer is None:
        polygons_by_layer = _polygons_by_layer(
            cell, excluded_reference_indexes=excluded_reference_indexes
        )
    layer = (record["layer"], record["datatype"])
    point = _occurrence_inverse_point(transform, record["selector_point_um"])
    selected = [
        polygon
        for polygon in polygons_by_layer.get(layer, ())
        if _point_in_ring(point, _ring_from_gdstk_polygon(polygon))
    ]
    if len(selected) != 1:
        raise ValueError(
            f"GeometryPlan port {record['name']!r} matched {len(selected)} local sheet polygons"
        )
    local = LayoutPolygonSpec(
        polygon_id="source_port_sheet",
        layer=f"{layer[0]}/{layer[1]}",
        exterior=_ring_from_gdstk_polygon(selected[0]),
    )
    return _occurrence_polygon_record(local, transform)


def _lumped_ring_identity(ring):
    """Exact source ring identity, independent only of start and traversal."""
    points = tuple(tuple(point) for point in ring)
    if points[0] == points[-1]:
        points = points[:-1]
    reverse = tuple(reversed(points))
    return min(
        order[index:] + order[:index]
        for order in (points, reverse)
        for index in range(len(order))
    )


def _occurrence_lumped_polygon(support, *, cell, excluded_reference_indexes=()):
    """Bind the complete authored local ring to actual original GDS geometry."""
    from scgsim.geometry._primitives.loops import _split_gdstk_cutline_loop

    declared = (
        _lumped_ring_identity(support.exterior),
        sorted(_lumped_ring_identity(ring) for ring in support.holes),
    )
    selected = []
    for polygon in _polygons_by_layer(
        cell, excluded_reference_indexes=excluded_reference_indexes
    ).get(support.source_layer, ()):
        exterior, holes = _split_gdstk_cutline_loop(_ring_from_gdstk_polygon(polygon))
        local = LayoutPolygonSpec(support.source_polygon_id, "", exterior, holes)
        actual = (
            _lumped_ring_identity(exterior),
            sorted(_lumped_ring_identity(ring) for ring in holes),
        )
        if actual == declared:
            selected.append(local)
    if len(selected) != 1:
        raise ValueError(
            f"lumped support {support.support_id!r} matched {len(selected)} source polygons"
        )
    return selected[0]


def _occurrence_include_polygon(
    record: Mapping[str, Any],
    *,
    cell: Any,
    transform: Sequence[float],
    excluded_reference_indexes: Sequence[int] = (),
    polygons_by_layer: Mapping[tuple[int, int], tuple[Any, ...]] | None = None,
) -> dict[str, Any]:
    """Select one connected local metal region, including its authored pieces."""
    if polygons_by_layer is None:
        polygons_by_layer = _polygons_by_layer(
            cell, excluded_reference_indexes=excluded_reference_indexes
        )
    layer = _gds_layer(
        record.get("layer"), str(record.get("source_local_id", "ground contribution"))
    )
    point = _occurrence_inverse_point(transform, record["selector_point_um"])
    components = _fused_gdstk_components(polygons_by_layer.get(layer, ()))
    selected = [
        polygon
        for polygon, _, _ in components
        if _point_in_layout_polygon((point[0], point[1]), polygon)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"ground contribution {record.get('source_local_id')!r} matched "
            f"{len(selected)} local source polygons"
        )
    return _occurrence_polygon_record(selected[0], transform)
