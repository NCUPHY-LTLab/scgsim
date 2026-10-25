"""Bind exact source terminals and derive rectangular junctions before planning."""

from scgsim.geometry._primitives.grid import with_input_grid

from dataclasses import replace

from scgsim.geometry.models.input import LayoutPolygonSpec
from scgsim.geometry.models.junctions import RectangularJunctionSpec
from scgsim.geometry.source.adapter import _port_sheet_overlaps


@with_input_grid
def with_rectangular_junctions(build_input, specs):
    """Bind caller-selected source owners without searching other occurrences."""
    requests = tuple(specs)
    if any(not isinstance(item, RectangularJunctionSpec) for item in requests):
        raise TypeError("rectangular junctions require RectangularJunctionSpec")
    for field in ("junction_id", "port_sheet_id"):
        identities = [getattr(item, field) for item in requests]
        if len(set(identities)) != len(identities):
            raise ValueError(f"duplicate rectangular junction {field}")
    entities = {item.semantic_id: item for item in build_input.entities}
    polygons = {item.polygon_id: item for item in build_input.polygons}
    replacements = {}
    for request in requests:
        matches = [
            item
            for item in build_input.port_sheet_regions
            if item.port_sheet_id == request.port_sheet_id
        ]
        if len(matches) != 1:
            raise ValueError("rectangular junction requires one exact source sheet")
        region = matches[0]
        if (
            region.source_polygon_id != request.source_polygon_id
            or region.metadata.get("source_occurrence_path")
            != request.source_occurrence_path
        ):
            raise ValueError("rectangular junction source sheet ownership differs")
        selected = {}
        for terminal in (request.terminal_a, request.terminal_b):
            entity = entities.get(terminal.source_entity_id)
            if entity is None or entity.material_kind != "conductor":
                raise ValueError(
                    "rectangular junction terminal source Entity is missing or not conductor"
                )
            if (
                entity.net_id != terminal.net_id
                or entity.metadata.get("source_occurrence_path")
                != terminal.source_occurrence_path
            ):
                raise ValueError(
                    "rectangular junction terminal Net/occurrence ownership differs"
                )
            selected.update({pid: entity for pid in entity.polygon_ids})
        overlaps = _port_sheet_overlaps(
            port_sheet_id=region.port_sheet_id,
            port_polygon=LayoutPolygonSpec(
                region.source_polygon_id,
                region.source_layer,
                region.exterior,
                region.holes,
            ),
            target_layer=region.metadata["target_layer"],
            host_entity_by_polygon_id=selected,
            host_polygons_by_id={pid: polygons[pid] for pid in selected},
        )
        if len(overlaps) != 2 or {item.host_semantic_id for item in overlaps} != {
            request.terminal_a.source_entity_id,
            request.terminal_b.source_entity_id,
        }:
            raise ValueError(
                "rectangular junction requires one actual overlap per explicit terminal"
            )
        replacements[region.port_sheet_id] = replace(region, overlaps=overlaps)
    return replace(
        build_input,
        port_sheet_regions=tuple(
            replacements.get(item.port_sheet_id, item)
            for item in build_input.port_sheet_regions
        ),
        metadata={
            **build_input.metadata,
            "rectangular_junctions": tuple(item.to_payload() for item in requests),
        },
    )


