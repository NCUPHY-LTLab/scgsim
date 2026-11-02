"""Carry loaded adapter facts and lower stack/entity records without source-file policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, TypeGuard

from scgsim.geometry.models.grid import GeometryGridContext
from scgsim.geometry.models.input import LayoutPolygonSpec, SemanticEntitySpec
from scgsim.geometry.source.ground import derived_ground_polygons
from scgsim.geometry.source.occurrence import (
    entity_from_layer_record,
    gds_polygons_for_entity,
    is_record_sequence,
    resolve_material_kind,
)

if TYPE_CHECKING:
    from gdstk import Cell, Polygon

# Stack geometry and metadata are extensible JSON supplied to their existing
# source validators. Any preserves that payload boundary and its error order.


@dataclass(frozen=True)
class StackRecords:
    """Validated stack containers with their original layout and stack paths."""

    gds_path: Path
    stack_path: Path
    layers: Sequence[Any]
    solution_regions: Mapping[str, Any]
    materials: Mapping[str, Any]
    metadata: Mapping[str, Any]


class LoadedLibrary(Protocol):
    """Native read_gds result, whose cells are parsed Cell objects, not RawCells."""

    @property
    def precision(self) -> float:
        """Return the native database precision in metres."""
        ...

    @property
    def cells(self) -> Sequence[Cell]:
        """Return the parsed cells in native library order."""
        ...

    def top_level(self) -> Sequence[Cell]:
        """Return native top-level cells in their original order."""
        ...


@dataclass(frozen=True)
class AdapterSource:
    """One fresh native library and selected source cell under its declared grid."""

    stack: StackRecords
    library: LoadedLibrary
    cell: Cell
    cell_bounds_um: Mapping[str, float]
    polygons_by_layer: Mapping[tuple[int, int], tuple[Polygon, ...]]
    source_dbu_um: float
    grid_context: GeometryGridContext


class LoweredEntity(NamedTuple):
    """A canonical entity paired with its source-owned polygon records."""

    entity: SemanticEntitySpec
    polygons: tuple[LayoutPolygonSpec, ...]


def record_sequence(value: object) -> TypeGuard[Sequence[Any]]:
    """Narrow the existing record-sequence predicate without adding validation."""
    return is_record_sequence(value)


def solution_region_entity(
    semantic_id: object,
    record: object,
    *,
    materials: Mapping[str, Any],
    cell_bounds_um: Mapping[str, float],
) -> SemanticEntitySpec:
    """Lower one solution record with the original material and bounds validation."""
    if not isinstance(semantic_id, str):
        raise TypeError("solution region ids must be strings")
    if not isinstance(record, Mapping):
        raise TypeError("stack_file 'solution_regions' values must be mappings")

    geometry = record.get("geometry", record)
    if not isinstance(geometry, Mapping):
        raise TypeError("solution region 'geometry' must be a mapping")

    geometry = dict(geometry)
    padding_um = float(geometry.get("padding_um", 0.0))
    geometry.setdefault(
        "domain_bounds_um",
        {
            "x_min_um": cell_bounds_um["x_min_um"] - padding_um,
            "y_min_um": cell_bounds_um["y_min_um"] - padding_um,
            "x_max_um": cell_bounds_um["x_max_um"] + padding_um,
            "y_max_um": cell_bounds_um["y_max_um"] + padding_um,
        },
    )

    return SemanticEntitySpec(
        semantic_id=semantic_id,
        role=record.get("role", "solution_region"),
        material_id=record.get("material_id", semantic_id),
        material_kind=resolve_material_kind(
            record,
            material_id=record.get("material_id", semantic_id),
            materials=materials,
            context=f"solution region {semantic_id!r}",
        ),
        priority=record.get("priority", 0),
        geometry_kind=record.get("geometry_kind", "domain"),
        geometry=geometry,
        metadata=record.get("metadata", {}),
    )


def lower_layer_record(
    record: Any,
    *,
    materials: Mapping[str, Any],
    polygons_by_layer: Mapping[tuple[int, int], tuple[Any, ...]],
    cell_bounds_um: Mapping[str, float],
    domain_bounds_by_semantic_id: Mapping[str, Mapping[str, Any]],
) -> tuple[LoweredEntity, ...]:
    """Lower a worker-private layer record through the existing source recipes."""
    entity = entity_from_layer_record(record, materials=materials)
    geometry_source = str(entity.geometry.get("geometry_source", "gds_polygon"))
    if geometry_source == "die_face_minus_ground_mask":
        occurrence_polygons = entity.geometry.get("source_occurrence_polygons_um")
        if occurrence_polygons is not None:
            entity_polygons = tuple(
                LayoutPolygonSpec(
                    polygon_id=f"{entity.semantic_id}__P{index:04d}",
                    layer=f"{record['layer']}/{record['datatype']}",
                    exterior=region["exterior"],
                    holes=region["holes"],
                    object_name=entity.semantic_id,
                    net_name=entity.net_id,
                    metadata={"source": "die_face_minus_ground_mask"},
                )
                for index, region in enumerate(occurrence_polygons)
            )
        else:
            entity_polygons = derived_ground_polygons(
                entity,
                polygons_by_layer=polygons_by_layer,
                cell_bounds_um=cell_bounds_um,
                domain_bounds_by_semantic_id=domain_bounds_by_semantic_id,
            )
        if len(entity_polygons) > 1:
            return tuple(
                _split_polygon_entity(entity, polygon, index)
                for index, polygon in enumerate(entity_polygons)
            )
        return (
            LoweredEntity(
                _entity_with_polygon_geometry(entity, entity_polygons), entity_polygons
            ),
        )
    elif geometry_source == "gds_polygon":
        entity_polygons = gds_polygons_for_entity(
            entity,
            polygons_by_layer=polygons_by_layer,
        )
    else:
        entity_polygons = ()

    if entity.geometry.get("split_polygons_as_entities") or (
        entity.geometry.get("route_ab_fused_selector_mode", False)
        and entity.geometry.get("selector_point_um") is None
    ):
        return tuple(
            _split_polygon_entity(entity, polygon, index)
            for index, polygon in enumerate(entity_polygons)
        )
    entity = _entity_with_polygon_geometry(entity, entity_polygons)
    return (LoweredEntity(entity, entity_polygons),)


def _split_polygon_entity(
    entity: SemanticEntitySpec,
    polygon: LayoutPolygonSpec,
    index: int,
) -> LoweredEntity:
    semantic_id = f"{entity.semantic_id}_{index:04d}"
    split_polygon = LayoutPolygonSpec(
        polygon_id=f"{semantic_id}__P0000",
        layer=polygon.layer,
        exterior=polygon.exterior,
        holes=polygon.holes,
        object_name=semantic_id,
        net_name=polygon.net_name,
        port_name=polygon.port_name,
        metadata={
            **dict(polygon.metadata),
            "source_semantic_id": entity.semantic_id,
            "source_polygon_id": polygon.polygon_id,
            "split_polygon_index": index,
        },
    )
    geometry = {
        **dict(entity.geometry),
        "outer_loop": split_polygon.exterior,
        "hole_loops": split_polygon.holes,
    }
    # Selection lists are consumed by this adapter; duplicating them onto each
    # split entity would turn generated-fill metadata quadratic.
    geometry.pop("include_selector_points_um", None)
    geometry.pop("exclude_selector_points_um", None)
    metadata = {
        **dict(entity.metadata),
        **(
            {"source_semantic_id": entity.semantic_id}
            if "source_occurrence_path" in entity.metadata
            else {"semantic_group_id": entity.semantic_id}
        ),
        "split_polygon_index": index,
    }
    return LoweredEntity(
        SemanticEntitySpec(
            semantic_id=semantic_id,
            role=entity.role,
            material_id=entity.material_id,
            material_kind=entity.material_kind,
            priority=entity.priority,
            geometry_kind=entity.geometry_kind,
            part_role=entity.part_role,
            attached_face_metal_semantic_id=entity.attached_face_metal_semantic_id,
            net_id=entity.net_id,
            polygon_ids=(split_polygon.polygon_id,),
            labels=entity.labels,
            host_void_semantic_id=entity.host_void_semantic_id,
            requires_construction_body=entity.requires_construction_body,
            route_representations=entity.route_representations,
            geometry=geometry,
            metadata=metadata,
        ),
        (split_polygon,),
    )


def _entity_with_polygon_geometry(
    entity: SemanticEntitySpec,
    polygons: tuple[LayoutPolygonSpec, ...],
) -> SemanticEntitySpec:
    if not polygons:
        return entity
    geometry = dict(entity.geometry)
    geometry.pop("include_selector_points_um", None)
    geometry.pop("exclude_selector_points_um", None)
    if len(polygons) == 1:
        geometry.setdefault("outer_loop", polygons[0].exterior)
        geometry.setdefault("hole_loops", polygons[0].holes)
    polygon_ids = tuple(polygon.polygon_id for polygon in polygons)
    return SemanticEntitySpec(
        semantic_id=entity.semantic_id,
        role=entity.role,
        material_id=entity.material_id,
        material_kind=entity.material_kind,
        priority=entity.priority,
        geometry_kind=entity.geometry_kind,
        part_role=entity.part_role,
        attached_face_metal_semantic_id=entity.attached_face_metal_semantic_id,
        net_id=entity.net_id,
        polygon_ids=polygon_ids,
        labels=entity.labels,
        host_void_semantic_id=entity.host_void_semantic_id,
        requires_construction_body=entity.requires_construction_body,
        route_representations=entity.route_representations,
        geometry=geometry,
        metadata=entity.metadata,
    )
