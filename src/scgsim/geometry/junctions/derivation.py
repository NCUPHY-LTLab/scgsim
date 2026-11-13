"""Derive rectangular source junctions after physical interpretation."""

from __future__ import annotations

import json
import math
from dataclasses import replace

from scgsim.geometry.junctions.lattice import drawn_exterior, lattice_rectangle
from scgsim.geometry.junctions.partition import partition_junctions
from scgsim.geometry.junctions.terminal import junction_terminal_line
from scgsim.geometry.models.grid import geometry_grid_context
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.junctions import JunctionGeometry, RectangularJunctionSpec
from scgsim.geometry.models.regions import PortSheetRegionRecord


def derive_rectangular_junctions(build_input: GeometryBuildInput) -> GeometryBuildInput:
    """Derive once after the single physical interpretation, before any surfaces.

    The steps: read each request's sheet as a rectangle on the grid, partition the
    input around those rectangles, then bind each sheet to its central rectangle and
    mark the terminal Entities that gained an end piece.
    """
    # An input without a declared grid is refused before anything else.
    geometry_grid_context(build_input)
    requests = tuple(
        RectangularJunctionSpec.from_payload(item)
        for item in build_input.metadata.get("rectangular_junctions", ())
    )
    if not requests:
        return build_input
    regions = {item.port_sheet_id: item for item in build_input.port_sheet_regions}
    geometries = [
        _junction_geometry(build_input, request, regions[request.port_sheet_id])
        for request in requests
    ]
    result = partition_junctions(
        build_input, geometries, build_input.metadata["source_dbu_um"]
    )
    return replace(
        result,
        entities=_entities_with_junction_support(result, requests),
        port_sheet_regions=_sheets_on_central_rectangles(result, requests),
    )


def _junction_geometry(
    build_input: GeometryBuildInput,
    request: RectangularJunctionSpec,
    region: PortSheetRegionRecord,
) -> JunctionGeometry:
    """The request's sheet as a rectangle on the grid along the port direction, and its width.

    A refusal carries a note naming the junction, its grid and its direction.
    """
    dx, dy = region.metadata["direction"][:2]
    norm = math.hypot(dx, dy)
    dx, dy = (dx / norm, dy / norm)
    try:
        # The sheet as its four corners on the grid; its width is theirs.
        corners = lattice_rectangle(
            region.exterior,
            build_input.metadata["source_dbu_um"],
            (dx, dy),
            drawn=drawn_exterior(region),
            port=f"lumped port {region.metadata.get('source_name', region.port_sheet_id)!r}",
            layer=region.source_layer,
        ).corners_um
        across = [-dy * x + dx * y for x, y in corners]
        geometry = JunctionGeometry(
            request.junction_id,
            request.source_polygon_id,
            request.terminal_a.net_id,
            request.terminal_b.net_id,
            (dx, dy),
            max(across) - min(across),
        )
        junction_terminal_line(
            {"exterior": corners, "holes": region.holes}, geometry, 0.0
        )
    except ValueError as error:
        error.add_note(
            json.dumps(
                {
                    "junction_id": request.junction_id,
                    "geometry_grid": geometry_grid_context(build_input).to_payload(),
                    "requested_direction_xy": [dx, dy],
                    "realization_status": "incompatible_with_existing_shape_contract",
                },
                sort_keys=True,
            )
        )
        raise
    return geometry


def _sheets_on_central_rectangles(
    result: GeometryBuildInput, requests: tuple[RectangularJunctionSpec, ...]
) -> tuple[PortSheetRegionRecord, ...]:
    """Each requested sheet as its central rectangle, recording its partition, binding and lattice."""
    records = result.metadata["junction_partitions"]
    polygons = {item.polygon_id: item for item in result.polygons}
    by_port = {item.port_sheet_id: item for item in requests}
    regions = []
    for region in result.port_sheet_regions:
        request = by_port.get(region.port_sheet_id)
        if request is None:
            regions.append(region)
            continue
        record = records[request.junction_id]
        central = polygons[record["central_polygon_id"]]
        regions.append(
            replace(
                region,
                exterior=central.exterior,
                holes=central.holes,
                metadata={
                    **region.metadata,
                    "rectangular_junction_partition": request.junction_id,
                    "rectangular_junction_binding": request.to_payload(),
                    "rectangular_junction_source_geometry": record[
                        "authored_source_geometry"
                    ],
                    "rectangular_junction_lattice": record["lattice"],
                },
            )
        )
    return tuple(regions)


def _entities_with_junction_support(
    result: GeometryBuildInput, requests: tuple[RectangularJunctionSpec, ...]
) -> tuple[SemanticEntitySpec, ...]:
    """Each terminal Entity marked with the junctions whose end pieces it now owns."""
    owner_ids = {
        terminal.source_entity_id
        for item in requests
        for terminal in (item.terminal_a, item.terminal_b)
    }
    return tuple(
        replace(
            entity,
            metadata={
                **entity.metadata,
                "junction_derived_support": tuple(
                    item.junction_id
                    for item in requests
                    if entity.semantic_id
                    in {
                        item.terminal_a.source_entity_id,
                        item.terminal_b.source_entity_id,
                    }
                ),
            },
        )
        if entity.semantic_id in owner_ids
        else entity
        for entity in result.entities
    )
