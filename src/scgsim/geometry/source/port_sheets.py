"""Lower `metadata.port_sheet_source_layers` into `PortSheetRegionRecord`s.

Owns: reading and validating the port-sheet source records of a stack file,
finding the GDS polygons that draw each sheet, and recording which conductor
polygons each sheet overlaps.

Does not own: Palace port semantics. A port sheet is source geometry only; the
Palace stage decides how a sheet becomes a lumped port, and nothing here creates
a backend-live surface.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Any

from scgsim.geometry.models.input import LayoutPolygonSpec, SemanticEntitySpec
from scgsim.geometry.models.regions import PortSheetOverlapRecord, PortSheetRegionRecord
from scgsim.geometry.source._occurrence import _is_record_sequence
from scgsim.geometry.source._polygon_ops import (
    _point_in_ring,
    _ring_from_gdstk_polygon,
)
from scgsim.geometry.source.ground import _layout_polygon_region

_PORT_SHEET_SOURCE = "palace_lumped_port_sheet"

# A source polygon with its index among the polygons of its GDS layer. The index
# is part of the generated port-sheet id, so it must be the layer-wide index even
# when a selector keeps only some polygons.
_IndexedPolygon = tuple[int, LayoutPolygonSpec]


@dataclass(frozen=True)
class _PortSheetSource:
    """Validated fields of one `port_sheet_source_layers` record."""

    name: str
    port_index: int
    target_layer: str
    direction_raw: tuple[float, ...]
    direction: tuple[float, float, float]
    sign_convention: str
    layer: int
    datatype: int


def _port_sheet_regions_from_stack_metadata(
    stack_metadata: Mapping[str, Any],
    *,
    polygons_by_layer: Mapping[tuple[int, int], tuple[Any, ...]],
    host_entities: Sequence[SemanticEntitySpec],
    host_polygons: Sequence[LayoutPolygonSpec],
) -> tuple[PortSheetRegionRecord, ...]:
    """Build one region per source polygon of every port-sheet source layer."""
    records = stack_metadata.get("port_sheet_source_layers", ())
    if records is None:
        return ()
    if not _is_record_sequence(records):
        raise TypeError("metadata.port_sheet_source_layers must be a sequence")

    host_entity_by_polygon_id = {
        polygon_id: entity
        for entity in host_entities
        for polygon_id in entity.polygon_ids
    }
    host_polygons_by_id = {polygon.polygon_id: polygon for polygon in host_polygons}
    port_regions: list[PortSheetRegionRecord] = []
    for source_index, record in enumerate(records):
        source = _read_port_sheet_source(record)
        occurrence_polygon = record.get("source_occurrence_polygon_um")
        if occurrence_polygon is not None:
            source_polygons = _occurrence_sheet_polygons(source, occurrence_polygon)
        else:
            source_polygons = _gds_sheet_polygons(source, record, polygons_by_layer)
        if not source_polygons:
            raise ValueError(
                "port_sheet_source_layers entry "
                f"{source.layer}/{source.datatype} has no polygons"
            )
        for polygon_index, source_polygon in source_polygons:
            port_sheet_id = f"PORT_SHEET__{source.name}__{polygon_index:04d}"
            scoped_hosts = host_entity_by_polygon_id
            if occurrence_polygon is not None:
                scoped_hosts = _hosts_in_occurrence(
                    host_entity_by_polygon_id,
                    occurrence_path=record["source_occurrence_path"],
                    target_layer=source.target_layer,
                )
            overlaps = _port_sheet_overlaps(
                port_sheet_id=port_sheet_id,
                port_polygon=source_polygon,
                target_layer=source.target_layer,
                host_entity_by_polygon_id=scoped_hosts,
                host_polygons_by_id=host_polygons_by_id,
            )
            port_regions.append(
                PortSheetRegionRecord(
                    port_sheet_id=port_sheet_id,
                    source_layer=f"{source.layer}/{source.datatype}",
                    source_polygon_id=source_polygon.polygon_id,
                    exterior=source_polygon.exterior,
                    holes=source_polygon.holes,
                    overlaps=overlaps,
                    metadata=_port_sheet_metadata(source_index, source, record),
                )
            )
    return tuple(port_regions)


def _read_port_sheet_source(record: Any) -> _PortSheetSource:
    """Validate one source record.

    The checks run in a fixed order because it decides which error a user sees
    first.
    """
    if not isinstance(record, Mapping):
        raise TypeError("metadata.port_sheet_source_layers items must be mappings")
    if record.get("source") != _PORT_SHEET_SOURCE:
        raise ValueError(
            "port_sheet_source_layers currently supports only "
            "source='palace_lumped_port_sheet'"
        )
    if "layer" not in record or "datatype" not in record:
        raise ValueError(
            "metadata.port_sheet_source_layers entries must define "
            "'layer' and 'datatype'"
        )
    name = record.get("name")
    if not isinstance(name, str) or not name:
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
    direction_raw, direction = _read_port_sheet_direction(record.get("direction"))
    sign_convention = record.get("direction_sign_convention", "authored")
    if not isinstance(sign_convention, str) or not sign_convention:
        raise ValueError(
            "port_sheet_source_layers direction_sign_convention must be non-empty"
        )
    return _PortSheetSource(
        name=name,
        port_index=port_index,
        target_layer=target_layer,
        direction_raw=direction_raw,
        direction=direction,
        sign_convention=sign_convention,
        layer=int(record["layer"]),
        datatype=int(record["datatype"]),
    )


def _read_port_sheet_direction(
    direction_raw: Any,
) -> tuple[tuple[float, ...], tuple[float, float, float]]:
    """Return the authored direction and its unit vector.

    The sheet direction lies in the XY plane; a vertical or zero direction has no
    meaning for a planar lumped-port sheet.
    """
    if (
        isinstance(direction_raw, str | bytes)
        or not isinstance(direction_raw, Sequence)
        or len(direction_raw) != 3
    ):
        raise ValueError("port_sheet_source_layers entries must define a 3D direction")
    direction = tuple(float(value) for value in direction_raw)
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


def _port_sheet_polygon_metadata(source: _PortSheetSource) -> dict[str, Any]:
    return {
        "gds_layer": source.layer,
        "gds_datatype": source.datatype,
        "source": _PORT_SHEET_SOURCE,
        "source_name": source.name,
    }


def _occurrence_sheet_polygons(
    source: _PortSheetSource, occurrence_polygon: Mapping[str, Any]
) -> tuple[_IndexedPolygon, ...]:
    """The sheet polygon recorded from one GDS source occurrence (always index 0)."""
    return (
        (
            0,
            LayoutPolygonSpec(
                polygon_id=f"PORT_SHEET__{source.name}__SOURCE__P0000",
                layer=f"{source.layer}/{source.datatype}",
                exterior=occurrence_polygon["exterior"],
                holes=occurrence_polygon["holes"],
                metadata=_port_sheet_polygon_metadata(source),
            ),
        ),
    )


def _gds_sheet_polygons(
    source: _PortSheetSource,
    record: Mapping[str, Any],
    polygons_by_layer: Mapping[tuple[int, int], tuple[Any, ...]],
) -> tuple[_IndexedPolygon, ...]:
    """Sheet polygons on the source GDS layer, optionally picked by a point."""
    layer, datatype = source.layer, source.datatype
    all_source_polygons = polygons_by_layer.get((layer, datatype), ())
    selector = record.get("selector_point_um")
    if selector is None:
        selected_polygons = tuple(enumerate(all_source_polygons))
    else:
        if (
            isinstance(selector, (str, bytes))
            or not isinstance(selector, Sequence)
            or len(selector) != 2
        ):
            raise ValueError("port sheet selector_point_um must be a 2D point")
        point = (float(selector[0]), float(selector[1]))
        selected_polygons = tuple(
            (index, polygon)
            for index, polygon in enumerate(all_source_polygons)
            if _point_in_ring(point, _ring_from_gdstk_polygon(polygon))
        )
        if len(selected_polygons) != 1:
            raise ValueError(
                f"port sheet {source.name!r} selector matched "
                f"{len(selected_polygons)} polygons"
            )
    return tuple(
        (
            index,
            LayoutPolygonSpec(
                polygon_id=f"PORT_SHEET_{layer}_{datatype}__P{index:04d}",
                layer=f"{layer}/{datatype}",
                exterior=_ring_from_gdstk_polygon(polygon),
                metadata=_port_sheet_polygon_metadata(source),
            ),
        )
        for index, polygon in selected_polygons
    )


def _hosts_in_occurrence(
    host_entity_by_polygon_id: Mapping[str, SemanticEntitySpec],
    *,
    occurrence_path: str,
    target_layer: str,
) -> dict[str, SemanticEntitySpec]:
    """Host polygons that belong to the sheet's own cell occurrence.

    A sheet drawn in a child cell may only land on conductors of that cell or of
    cells below it. The derived ground plane has no occurrence of its own, so it
    counts when one of its recorded ground contributions comes from this
    occurrence on the target layer.
    """
    return {
        polygon_id: entity
        for polygon_id, entity in host_entity_by_polygon_id.items()
        if entity.metadata.get("source_occurrence_path") == occurrence_path
        or str(entity.metadata.get("source_occurrence_path", "")).startswith(
            f"{occurrence_path}/"
        )
        or (
            entity.geometry.get("geometry_source") == "die_face_minus_ground_mask"
            and entity.metadata.get("logical_layer_id") == target_layer
            and any(
                contribution.get("source_occurrence_path") == occurrence_path
                for contribution in entity.metadata.get(
                    "ground_plane_contribution_sources", ()
                )
            )
        )
    }


def _port_sheet_metadata(
    source_index: int, source: _PortSheetSource, record: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "source_index": source_index,
        "source_name": source.name,
        "port_index": source.port_index,
        "target_layer": source.target_layer,
        "direction": source.direction,
        "direction_raw": source.direction_raw,
        "direction_sign_convention": source.sign_convention,
        "source": _PORT_SHEET_SOURCE,
        **(
            {"source_occurrence_path": record["source_occurrence_path"]}
            if "source_occurrence_path" in record
            else {}
        ),
    }


def _port_sheet_overlaps(
    *,
    port_sheet_id: str,
    port_polygon: LayoutPolygonSpec,
    target_layer: str,
    host_entity_by_polygon_id: Mapping[str, SemanticEntitySpec],
    host_polygons_by_id: Mapping[str, LayoutPolygonSpec],
) -> tuple[PortSheetOverlapRecord, ...]:
    """Boolean-AND the sheet with each eligible host polygon, in host order."""
    import gdstk

    port_region = _layout_polygon_region(gdstk, port_polygon)
    overlaps: list[PortSheetOverlapRecord] = []
    for host_polygon_id, host_polygon in host_polygons_by_id.items():
        host_entity = host_entity_by_polygon_id.get(host_polygon_id)
        if (
            host_entity is None
            or not _is_port_sheet_host_entity(host_entity)
            or not _port_sheet_target_layer_matches(host_entity, target_layer)
        ):
            continue
        overlap_region = gdstk.boolean(
            port_region,
            _layout_polygon_region(gdstk, host_polygon),
            "and",
            precision=1e-9,
        )
        for index, overlap_polygon in enumerate(overlap_region or ()):
            # Slivers at or below 1e-8 um^2 are boolean round-off, not overlap.
            if abs(float(overlap_polygon.area())) <= 1e-8:
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
                    overlap_loop=_ring_from_gdstk_polygon(overlap_polygon),
                    metadata={"overlap_polygon_index": index},
                )
            )
    return tuple(overlaps)


def _port_sheet_target_layer_matches(
    entity: SemanticEntitySpec, target_layer: str
) -> bool:
    """True when `target_layer` names the entity by GDS pair, group or logical layer."""
    semantic_group = entity.metadata.get("semantic_group_id")
    logical_layer = entity.metadata.get("logical_layer_id")
    return target_layer in {
        f"{entity.geometry.get('gds_layer')}/{entity.geometry.get('gds_datatype')}",
        semantic_group if isinstance(semantic_group, str) else "",
        logical_layer if isinstance(logical_layer, str) else "",
    }


def _is_port_sheet_host_entity(entity: SemanticEntitySpec) -> bool:
    """Only drawn conductors can host a port sheet."""
    if entity.material_kind != "conductor":
        return False
    if not entity.polygon_ids:
        return False
    geometry_source = str(entity.geometry.get("geometry_source", "gds_polygon"))
    return geometry_source in {"gds_polygon", "die_face_minus_ground_mask"}
