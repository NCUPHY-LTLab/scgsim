"""Bind explicit source terminals to port sheets before geometry planning."""

from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import cast

from scgsim.geometry.models.input import (
    GeometryBuildInput,
    LayoutPolygonSpec,
    SemanticEntitySpec,
)
from scgsim.geometry.models.junctions import RectangularJunctionSpec
from scgsim.geometry.models.regions import PortSheetRegionRecord
from scgsim.geometry.primitives.grid import with_input_grid
from scgsim.geometry.source.ports import port_sheet_overlaps


@with_input_grid
def with_rectangular_junctions(
    build_input: GeometryBuildInput, specs: Iterable[object]
) -> GeometryBuildInput:
    """Bind caller-selected source owners without searching other occurrences."""
    requests = _validated_requests(specs)
    _validate_unique_request_ids(requests)
    entities = {item.semantic_id: item for item in build_input.entities}
    polygons = {item.polygon_id: item for item in build_input.polygons}
    replacements = {
        request.port_sheet_id: _bind_request(build_input, request, entities, polygons)
        for request in requests
    }
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


def _validated_requests(specs: Iterable[object]) -> tuple[RectangularJunctionSpec, ...]:
    """Materialize requests once, then retain the original type error contract."""
    requests = tuple(specs)
    if any(not isinstance(item, RectangularJunctionSpec) for item in requests):
        raise TypeError("rectangular junctions require RectangularJunctionSpec")
    # The runtime check establishes this tuple's record type without coercion.
    return cast(tuple[RectangularJunctionSpec, ...], requests)


def _validate_unique_request_ids(
    requests: tuple[RectangularJunctionSpec, ...],
) -> None:
    """Reject duplicate request identities in their established field order."""
    for field in ("junction_id", "port_sheet_id"):
        identities = [getattr(item, field) for item in requests]
        if len(set(identities)) != len(identities):
            raise ValueError(f"duplicate rectangular junction {field}")


def _bind_request(
    build_input: GeometryBuildInput,
    request: RectangularJunctionSpec,
    entities: Mapping[str, SemanticEntitySpec],
    polygons: Mapping[str, LayoutPolygonSpec],
) -> PortSheetRegionRecord:
    """Validate one sheet, its explicit terminals, and their actual overlaps."""
    matches = [
        item
        for item in build_input.port_sheet_regions
        if item.port_sheet_id == request.port_sheet_id
    ]
    if len(matches) != 1:
        raise ValueError("rectangular junction requires one exact source sheet")
    region = matches[0]
    _validate_sheet_ownership(region, request)
    selected = _selected_terminal_polygons(request, entities)
    overlaps = port_sheet_overlaps(
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
    expected_hosts = {
        request.terminal_a.source_entity_id,
        request.terminal_b.source_entity_id,
    }
    if (
        len(overlaps) != 2
        or {item.host_semantic_id for item in overlaps} != expected_hosts
    ):
        raise ValueError(
            "rectangular junction requires one actual overlap per explicit terminal"
        )
    return replace(region, overlaps=overlaps)


def _validate_sheet_ownership(
    region: PortSheetRegionRecord, request: RectangularJunctionSpec
) -> None:
    if (
        region.source_polygon_id != request.source_polygon_id
        or region.metadata.get("source_occurrence_path")
        != request.source_occurrence_path
    ):
        raise ValueError("rectangular junction source sheet ownership differs")


def _selected_terminal_polygons(
    request: RectangularJunctionSpec,
    entities: Mapping[str, SemanticEntitySpec],
) -> dict[str, SemanticEntitySpec]:
    selected: dict[str, SemanticEntitySpec] = {}
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
        selected.update({polygon_id: entity for polygon_id in entity.polygon_ids})
    return selected
