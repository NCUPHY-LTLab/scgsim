"""Bind explicit source terminals to port sheets before geometry planning."""

from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import cast

from scgsim.geometry.grid.grid_regions import grid_change_message
from scgsim.geometry.grid.scope import GridScope
from scgsim.geometry.models.grid import format_box, geometry_grid_context
from scgsim.geometry.models.input import (
    GeometryBuildInput,
    LayoutPolygonSpec,
    SemanticEntitySpec,
)
from scgsim.geometry.models.junctions import (
    RectangularJunctionSpec,
    junction_net_id,
    junction_occurrence_path,
)
from scgsim.geometry.models.refusals import Declared
from scgsim.geometry.models.regions import (
    PortSheetOverlapRecord,
    PortSheetRegionRecord,
)
from scgsim.geometry.source.ports import port_sheet_hosts, port_sheet_overlaps

_REQUEST = Declared(
    "rectangular junction request",
    "the RectangularJunctionSpec passed to the plan or the simulation",
)


def with_rectangular_junctions(
    build_input: GeometryBuildInput, specs: Iterable[object]
) -> GeometryBuildInput:
    """Bind caller-selected source owners without searching other occurrences.

    The sheet's overlaps with its terminals are measured on the input's declared grid.
    """
    scope = GridScope(geometry_grid_context(build_input))
    requests = _validated_requests(specs)
    _validate_unique_request_ids(requests)
    entities = {item.semantic_id: item for item in build_input.entities}
    polygons = {item.polygon_id: item for item in build_input.polygons}
    replacements = {
        request.port_sheet_id: _bind_request(
            scope, build_input, request, entities, polygons
        )
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
        raise TypeError(
            _REQUEST.expected("every request", "a RectangularJunctionSpec", requests)
        )
    # The runtime check establishes this tuple's record type without coercion.
    return cast(tuple[RectangularJunctionSpec, ...], requests)


def _validate_unique_request_ids(
    requests: tuple[RectangularJunctionSpec, ...],
) -> None:
    """Reject duplicate request identities in their established field order."""
    for field in ("junction_id", "port_sheet_id"):
        identities = [getattr(item, field) for item in requests]
        if len(set(identities)) != len(identities):
            raise ValueError(
                _REQUEST.refusal(f"duplicate rectangular junction {field}")
            )


def _bind_request(
    scope: GridScope,
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
        raise ValueError(
            _REQUEST.part(repr(request.junction_id)).expected(
                "port_sheet_id",
                "exactly one port sheet of the input",
                request.port_sheet_id,
            )
        )
    region = matches[0]
    _validate_sheet_ownership(region, request)
    selected = _selected_terminal_polygons(request, entities)
    overlaps = port_sheet_overlaps(
        scope,
        port_sheet_id=region.port_sheet_id,
        port_polygon=LayoutPolygonSpec(
            region.source_polygon_id,
            region.source_layer,
            region.exterior,
            region.holes,
        ),
        target_layer=region.metadata["target_layer"],
        hosts=port_sheet_hosts(selected, {pid: polygons[pid] for pid in selected}),
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
            _overlap_count_refusal(build_input, region, overlaps, polygons, request)
        )
    return replace(region, overlaps=overlaps)


def _overlap_count_refusal(
    build_input: GeometryBuildInput,
    region: PortSheetRegionRecord,
    overlaps: tuple[PortSheetOverlapRecord, ...],
    polygons: Mapping[str, LayoutPolygonSpec],
    request: RectangularJunctionSpec,
) -> str:
    """Where the sheet overlaps each terminal, when it does not overlap each one exactly once."""
    grid_um = geometry_grid_context(build_input).grid_size_um
    findings = []
    for terminal in (request.terminal_a, request.terminal_b):
        host = terminal.source_entity_id
        mine = [item for item in overlaps if item.host_semantic_id == host]
        if not mine:
            findings.append(f"it does not overlap {host}")
            continue
        layers = sorted({polygons[item.host_polygon_id].layer for item in mine})
        places = ", ".join(_loop_place(item.overlap_loop, grid_um) for item in mine)
        times = "once" if len(mine) == 1 else f"{len(mine)} times"
        findings.append(
            f"it overlaps {host} on layer {', '.join(layers)} {times}, at {places}"
        )
    name = region.metadata.get("source_name", region.port_sheet_id)
    return grid_change_message(
        "rectangular junction requires one actual overlap per explicit terminal: "
        f"lumped port {name!r}",
        [region.source_layer],
        grid_um,
        findings,
        "Fix the layout: draw the sheet so that it overlaps each of its two conductors "
        "once, at its own end.",
    )


def _loop_place(loop, grid_um: float | None) -> str:
    xs, ys = [x for x, _ in loop], [y for _, y in loop]
    return format_box((min(xs), min(ys), max(xs), max(ys)), grid_um)


def _validate_sheet_ownership(
    region: PortSheetRegionRecord, request: RectangularJunctionSpec
) -> None:
    if (
        region.source_polygon_id != request.source_polygon_id
        or junction_occurrence_path(region.metadata) != request.source_occurrence_path
    ):
        raise ValueError(
            _REQUEST.part(repr(request.junction_id)).refusal(
                "its source sheet ownership (polygon and occurrence) differs from the sheet's"
            )
        )


def _selected_terminal_polygons(
    request: RectangularJunctionSpec,
    entities: Mapping[str, SemanticEntitySpec],
) -> dict[str, SemanticEntitySpec]:
    selected: dict[str, SemanticEntitySpec] = {}
    for terminal in (request.terminal_a, request.terminal_b):
        entity = entities.get(terminal.source_entity_id)
        if entity is None or entity.material_kind != "conductor":
            raise ValueError(
                _REQUEST.part(repr(request.junction_id)).expected(
                    "terminal source Entity",
                    "a conductor Entity of the input",
                    terminal.source_entity_id,
                )
            )
        if (
            junction_net_id(entity) != terminal.net_id
            or junction_occurrence_path(entity.metadata)
            != terminal.source_occurrence_path
        ):
            raise ValueError(
                _REQUEST.part(repr(request.junction_id)).refusal(
                    f"terminal {terminal.source_entity_id!r} Net/occurrence ownership differs from the Entity's"
                )
            )
        selected.update({polygon_id: entity for polygon_id in entity.polygon_ids})
    return selected
