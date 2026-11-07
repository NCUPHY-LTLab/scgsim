"""Build source-owned port sheet regions and their conductor overlaps."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from math import isfinite, sqrt
from typing import Any, Protocol, TypeGuard, cast

from scgsim.geometry.models.input import (
    GeometryBuildInput,
    LayoutPolygonSpec,
    SemanticEntitySpec,
)
from scgsim.geometry.models.regions import PortSheetOverlapRecord, PortSheetRegionRecord
from scgsim.geometry.primitives.grid import (
    current_grid_size_um,
    polygon_boolean,
)
from scgsim.geometry.source.polygon_ops import (
    layout_polygon_region,
    ring_from_gdstk_polygon,
)

SOURCE_OCCURRENCE_POLYGON_METADATA_KEY = "source_occurrence_polygon_um"


def port_sheet_regions_from_stack_metadata(
    stack_metadata: Mapping[str, Any],
    *,
    polygons_by_layer: Mapping[tuple[int, int], tuple[Any, ...]],
    host_entities: Sequence[SemanticEntitySpec],
    host_polygons: Sequence[LayoutPolygonSpec],
) -> tuple[PortSheetRegionRecord, ...]:
    """Build source port regions using stack metadata and exact host ownership."""
    raw_records = stack_metadata.get("port_sheet_source_layers", ())
    if raw_records is None:
        return ()
    if not _is_source_record_sequence(raw_records):
        raise TypeError("metadata.port_sheet_source_layers must be a sequence")

    host_entity_by_polygon_id = {
        polygon_id: entity
        for entity in host_entities
        for polygon_id in entity.polygon_ids
    }
    host_polygons_by_id = {polygon.polygon_id: polygon for polygon in host_polygons}
    regions: list[PortSheetRegionRecord] = []
    for source_index, record in enumerate(raw_records):
        (
            source_name,
            port_index,
            target_layer,
            direction,
            raw_direction,
            sign_convention,
            layer,
            datatype,
        ) = _validated_port_metadata(record)
        source_polygons = _selected_port_polygons(
            record,
            layer=layer,
            datatype=datatype,
            source_name=source_name,
            polygons_by_layer=polygons_by_layer,
        )
        if not source_polygons:
            raise ValueError(
                f"port_sheet_source_layers entry {layer}/{datatype} has no polygons"
            )
        for polygon_index, source_polygon in source_polygons:
            port_sheet_id = f"PORT_SHEET__{source_name}__{polygon_index:04d}"
            scoped_hosts = _hosts_for_source_occurrence(
                host_entity_by_polygon_id, record, target_layer
            )
            overlaps = port_sheet_overlaps(
                port_sheet_id=port_sheet_id,
                port_polygon=source_polygon,
                target_layer=target_layer,
                host_entity_by_polygon_id=scoped_hosts,
                host_polygons_by_id=host_polygons_by_id,
            )
            regions.append(
                PortSheetRegionRecord(
                    port_sheet_id=port_sheet_id,
                    source_layer=f"{layer}/{datatype}",
                    source_polygon_id=source_polygon.polygon_id,
                    exterior=source_polygon.exterior,
                    holes=source_polygon.holes,
                    overlaps=overlaps,
                    metadata={
                        "source_index": source_index,
                        "source_name": source_name,
                        "port_index": port_index,
                        "target_layer": target_layer,
                        "direction": direction,
                        "direction_raw": raw_direction,
                        "direction_sign_convention": sign_convention,
                        "source": "palace_lumped_port_sheet",
                        **(
                            {"source_occurrence_path": record["source_occurrence_path"]}
                            if "source_occurrence_path" in record
                            else {}
                        ),
                    },
                )
            )
    return tuple(regions)


def _normalized_port_direction(
    raw_direction: Any,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if (
        isinstance(raw_direction, str | bytes)
        or not isinstance(raw_direction, Sequence)
        or len(raw_direction) != 3
    ):
        raise ValueError("port_sheet_source_layers entries must define a 3D direction")
    converted = tuple(float(value) for value in raw_direction)
    direction = (converted[0], converted[1], converted[2])
    if (
        not all(isfinite(value) for value in direction)
        or direction[2] != 0.0
        or direction[0] == direction[1] == 0.0
    ):
        raise ValueError(
            "port_sheet_source_layers direction must be finite, XY, and nonzero"
        )
    length = sqrt(direction[0] ** 2 + direction[1] ** 2)
    return direction, (direction[0] / length, direction[1] / length, 0.0)


def _is_source_record_sequence(value: Any) -> TypeGuard[Sequence[Any]]:
    """Narrow the untrusted metadata sequence while preserving item validation."""
    return isinstance(value, Sequence) and not isinstance(value, str | bytes)


def _validated_port_metadata(
    record: Any,
) -> tuple[
    str,
    int,
    str,
    tuple[float, float, float],
    tuple[float, float, float],
    str,
    int,
    int,
]:
    if not isinstance(record, Mapping):
        raise TypeError("metadata.port_sheet_source_layers items must be mappings")
    if record.get("source") != "palace_lumped_port_sheet":
        raise ValueError(
            "port_sheet_source_layers currently supports only "
            "source='palace_lumped_port_sheet'"
        )
    if "layer" not in record or "datatype" not in record:
        raise ValueError(
            "metadata.port_sheet_source_layers entries must define "
            "'layer' and 'datatype'"
        )
    source_name = record.get("name")
    if not isinstance(source_name, str) or not source_name:
        raise ValueError("port_sheet_source_layers entries must define name")
    port_index = record.get("port_index")
    if (
        isinstance(port_index, bool)
        or not isinstance(port_index, int)
        or port_index < 1
    ):
        raise ValueError(
            "port_sheet_source_layers entries must define 1-based port_index"
        )
    target_layer = record.get("target_layer")
    if not isinstance(target_layer, str) or not target_layer:
        raise ValueError("port_sheet_source_layers entries must define target_layer")
    raw_direction, direction = _normalized_port_direction(record.get("direction"))
    sign_convention = record.get("direction_sign_convention", "authored")
    if not isinstance(sign_convention, str) or not sign_convention:
        raise ValueError(
            "port_sheet_source_layers direction_sign_convention must be non-empty"
        )
    return (
        source_name,
        port_index,
        target_layer,
        direction,
        raw_direction,
        sign_convention,
        int(record["layer"]),
        int(record["datatype"]),
    )


def _selected_port_polygons(
    record: Mapping[str, Any],
    *,
    layer: int,
    datatype: int,
    source_name: str,
    polygons_by_layer: Mapping[tuple[int, int], tuple[Any, ...]],
) -> tuple[tuple[int, LayoutPolygonSpec], ...]:
    occurrence_polygon = record.get(SOURCE_OCCURRENCE_POLYGON_METADATA_KEY)
    if occurrence_polygon is not None:
        polygon = LayoutPolygonSpec(
            polygon_id=f"PORT_SHEET__{source_name}__SOURCE__P0000",
            layer=f"{layer}/{datatype}",
            exterior=occurrence_polygon["exterior"],
            holes=occurrence_polygon["holes"],
            metadata={
                "gds_layer": layer,
                "gds_datatype": datatype,
                "source": "palace_lumped_port_sheet",
                "source_name": source_name,
            },
        )
        return ((0, polygon),)

    source_polygons = polygons_by_layer.get((layer, datatype), ())
    selector = record.get("selector_point_um")
    if selector is None:
        selected = tuple(enumerate(source_polygons))
    else:
        if (
            isinstance(selector, (str, bytes))
            or not isinstance(selector, Sequence)
            or len(selector) != 2
        ):
            raise ValueError("port sheet selector_point_um must be a 2D point")
        point = (float(selector[0]), float(selector[1]))
        selected = tuple(
            (index, polygon)
            for index, polygon in enumerate(source_polygons)
            if _point_in_ring(point, ring_from_gdstk_polygon(polygon))
        )
        if len(selected) != 1:
            raise ValueError(
                f"port sheet {source_name!r} selector matched {len(selected)} polygons"
            )
    return tuple(
        (
            index,
            LayoutPolygonSpec(
                polygon_id=f"PORT_SHEET_{layer}_{datatype}__P{index:04d}",
                layer=f"{layer}/{datatype}",
                exterior=ring_from_gdstk_polygon(polygon),
                metadata={
                    "gds_layer": layer,
                    "gds_datatype": datatype,
                    "source": "palace_lumped_port_sheet",
                    "source_name": source_name,
                },
            ),
        )
        for index, polygon in selected
    )


def _hosts_for_source_occurrence(
    hosts: Mapping[str, SemanticEntitySpec],
    record: Mapping[str, Any],
    target_layer: str,
) -> Mapping[str, SemanticEntitySpec]:
    if record.get(SOURCE_OCCURRENCE_POLYGON_METADATA_KEY) is None:
        return hosts
    occurrence_path = record["source_occurrence_path"]
    return {
        polygon_id: entity
        for polygon_id, entity in hosts.items()
        if entity.metadata.get("source_occurrence_path") == occurrence_path
        or str(entity.metadata.get("source_occurrence_path", "")).startswith(
            f"{occurrence_path}/"
        )
        or (
            entity.geometry.get("geometry_source") == "die_face_minus_ground_mask"
            and entity.metadata.get("logical_layer_id") == target_layer
            and any(
                source.get("source_occurrence_path") == occurrence_path
                for source in entity.metadata.get(
                    "ground_plane_contribution_sources", ()
                )
            )
        )
    }


def port_sheet_overlaps(
    *,
    port_sheet_id: str,
    port_polygon: LayoutPolygonSpec,
    target_layer: str,
    host_entity_by_polygon_id: Mapping[str, SemanticEntitySpec],
    host_polygons_by_id: Mapping[str, LayoutPolygonSpec],
) -> tuple[PortSheetOverlapRecord, ...]:
    """Intersect a port polygon with eligible source host polygons."""
    import gdstk  # noqa: PLC0415 — keep the optional geometry dependency lazy.

    port_region = layout_polygon_region(gdstk, port_polygon)
    overlaps: list[PortSheetOverlapRecord] = []
    for host_polygon_id, host_polygon in host_polygons_by_id.items():
        host_entity = host_entity_by_polygon_id.get(host_polygon_id)
        if (
            host_entity is None
            or not is_port_sheet_host_entity(host_entity)
            or not port_sheet_target_layer_matches(host_entity, target_layer)
        ):
            continue
        overlap_region = polygon_boolean(
            port_region,
            layout_polygon_region(gdstk, host_polygon),
            "and",
            precision=current_grid_size_um(),
        )
        for index, overlap_polygon in enumerate(overlap_region or ()):
            area_polygon = cast(_AreaPolygon, overlap_polygon)
            if abs(float(area_polygon.area())) <= 1e-8:
                continue
            overlaps.append(
                PortSheetOverlapRecord(
                    overlap_id=(
                        f"PORT_SHEET_OVERLAP__{port_sheet_id}__"
                        f"{host_entity.semantic_id}__{len(overlaps):04d}"
                    ),
                    port_sheet_id=port_sheet_id,
                    port_polygon_id=port_polygon.polygon_id,
                    host_semantic_id=host_entity.semantic_id,
                    host_polygon_id=host_polygon_id,
                    overlap_loop=ring_from_gdstk_polygon(overlap_polygon),
                    metadata={"overlap_polygon_index": index},
                )
            )
    return tuple(overlaps)


def port_sheet_target_layer_matches(
    entity: SemanticEntitySpec, target_layer: str
) -> bool:
    """Return whether source metadata binds the entity to the requested layer."""
    semantic_group = entity.metadata.get("semantic_group_id")
    logical_layer = entity.metadata.get("logical_layer_id")
    return target_layer in {
        f"{entity.geometry.get('gds_layer')}/{entity.geometry.get('gds_datatype')}",
        semantic_group if isinstance(semantic_group, str) else "",
        logical_layer if isinstance(logical_layer, str) else "",
    }


def is_port_sheet_host_entity(entity: SemanticEntitySpec) -> bool:
    """Return whether an entity can own a source port sheet overlap."""
    if entity.material_kind != "conductor":
        return False
    if not entity.polygon_ids:
        return False
    geometry_source = str(entity.geometry.get("geometry_source", "gds_polygon"))
    return geometry_source in {"gds_polygon", "die_face_minus_ground_mask"}


def _point_in_ring(
    point: tuple[float, float], ring: Sequence[tuple[float, float]]
) -> bool:
    x, y = point
    inside = False
    previous_index = len(ring) - 1
    for index, (current_x, current_y) in enumerate(ring):
        previous_x, previous_y = ring[previous_index]
        crosses = (current_y > y) != (previous_y > y)
        if crosses:
            intersection_x = (previous_x - current_x) * (y - current_y) / (
                previous_y - current_y
            ) + current_x
            if x < intersection_x:
                inside = not inside
        previous_index = index
    return inside


class _AreaPolygon(Protocol):
    """The GDS polygon behavior omitted from the Boolean return annotation."""

    def area(self) -> float:
        """Return the polygon's unsigned area."""
        ...


def select_authored_ports(
    build_input: GeometryBuildInput,
    port_sheet_source_layers: Sequence[Mapping[str, Any]] | None,
) -> GeometryBuildInput:
    """Keep only the requested port sheets, with exact source identities and request-order port indices.

    ``None`` keeps every authored sheet. A name with no authored sheet raises ``ValueError``.
    """
    if port_sheet_source_layers is not None:
        selected = []
        for source_index, record in enumerate(port_sheet_source_layers):
            matches = tuple(
                region
                for region in build_input.port_sheet_regions
                if region.metadata.get("source_name") == record["name"]
            )
            if not matches:
                raise ValueError(
                    f"normalized source has no authored port {record['name']!r}"
                )
            selected.extend(
                replace(
                    region,
                    metadata={
                        **region.metadata,
                        "source_index": source_index,
                        "port_index": record["port_index"],
                    },
                )
                for region in matches
            )
        build_input = replace(
            build_input,
            port_sheet_regions=tuple(selected),
            metadata={
                **build_input.metadata,
                "port_sheet_source_layers": list(port_sheet_source_layers),
            },
        )
    return build_input
