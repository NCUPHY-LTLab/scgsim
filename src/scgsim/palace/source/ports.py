"""Bind authored layout ports to Palace lumped-port records, without Sim state.

Owns the requested port, the port it resolves to (``LayoutPortBinding``, which ``mesh`` cuts and
``config`` writes) and one step of a lumped port's preparation: asking the shared geometry for the
rectangle Palace's lumped element needs (``with_rectangular_ports``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from scgsim.geometry import (
    GeometryBuildInput,
    JunctionTerminalBinding,
    PortSheetRegionRecord,
    RectangularJunctionSpec,
    SemanticEntitySpec,
    with_rectangular_junctions,
)
from scgsim.geometry.models.grid import format_box, geometry_grid_context
from scgsim.geometry.models.junctions import (
    junction_net_id,
    junction_occurrence_path,
)


@dataclass(frozen=True)
class RequestedPort:
    """One caller-selected authored port with its target layer and its circuit, as ``LayoutPortBinding.circuit``."""

    name: str
    layer: str
    circuit: Mapping[str, float]


@dataclass(frozen=True)
class LayoutPortBinding:
    """One requested port bound to an authored sheet of the plan; the n-th port requested is Palace ``P<n>``.

    ``circuit`` holds the port's Palace ``LumpedPort`` circuit keys with their SI values, such as
    ``{"L": 1e-8}`` (henry) for an Eigenmode port; the port's ``LumpedPort`` row carries them unchanged.
    """

    index: int
    name: str
    target_layer: str
    source_layer: str
    direction: tuple[float, float, float]
    circuit: Mapping[str, float]


def with_rectangular_ports(build_input: GeometryBuildInput) -> GeometryBuildInput:
    """Request the shared rectangular partition for every lumped-port sheet of ``build_input``.

    Palace's uniform lumped element (Lumped Port or Lumped RLC) must be a rectangle: it
    compares the port's bounding-box length with its length projected on the port direction.
    A sheet that lands on a round conductor ends on that conductor's arc, so SCGSim keeps only
    the central rectangle of the drawn sheet as the port and gives the two end pieces to the
    conductors the sheet lands on.  The partition itself, its cut rule and its refusals are
    the shared geometry's (``scgsim.geometry.planning.junction_partition``); this step only
    names the two terminals of each sheet.  Requests the plan already holds are kept for the
    sheets this simulation selected; a request for a sheet it did not select is dropped with
    that sheet, since only the selected sheets reach the build.
    """
    selected = {region.port_sheet_id for region in build_input.port_sheet_regions}
    existing = tuple(
        RectangularJunctionSpec.from_payload(item)
        for item in build_input.metadata.get("rectangular_junctions", ())
    )
    kept = tuple(item for item in existing if item.port_sheet_id in selected)
    requested = {item.port_sheet_id for item in kept}
    entities = {entity.semantic_id: entity for entity in build_input.entities}
    grid_um = geometry_grid_context(build_input).grid_size_um
    added = tuple(
        _port_request(region, entities, grid_um)
        for region in build_input.port_sheet_regions
        if region.port_sheet_id not in requested
    )
    if not added and len(kept) == len(existing):
        return build_input
    return with_rectangular_junctions(build_input, (*kept, *added))


def _port_request(
    region: PortSheetRegionRecord,
    entities: Mapping[str, SemanticEntitySpec],
    grid_um: float | None,
) -> RectangularJunctionSpec:
    """The partition request of one port sheet: its own identity and its two host conductors.

    Terminal A is the host at the start of the port direction and Terminal B the host it
    points to, the order the partition cuts its two ends in.
    """
    name = region.metadata.get("source_name", region.port_sheet_id)
    hosts = tuple(dict.fromkeys(item.host_semantic_id for item in region.overlaps))
    if len(hosts) != 2:
        xs = [x for x, _ in region.exterior]
        ys = [y for _, y in region.exterior]
        box = format_box((min(xs), min(ys), max(xs), max(ys)), grid_um)
        raise ValueError(
            f"lumped port {name!r} must land on two conductors to be made a rectangle; "
            f"it lands on {len(hosts)}: {list(hosts)}; the sheet is drawn on layer "
            f"{region.source_layer} at {box}. Fix the layout or the port declaration: "
            "draw the sheet so that it lands on exactly two conductors of its target "
            f"layer {region.metadata.get('target_layer')!r}, one at each end."
        )
    terminals = [_terminal(entities[host]) for host in hosts]
    terminals.sort(key=lambda item: _along_direction(region, item.source_entity_id))
    return RectangularJunctionSpec(
        junction_id=region.port_sheet_id,
        port_sheet_id=region.port_sheet_id,
        source_polygon_id=region.source_polygon_id,
        source_occurrence_path=junction_occurrence_path(region.metadata),
        terminal_a=terminals[0],
        terminal_b=terminals[1],
    )


def _terminal(entity: SemanticEntitySpec) -> JunctionTerminalBinding:
    """The terminal binding of one host conductor; one without a Net is a Net of its own."""
    return JunctionTerminalBinding(
        source_entity_id=entity.semantic_id,
        source_occurrence_path=junction_occurrence_path(entity.metadata),
        net_id=junction_net_id(entity),
    )


def _along_direction(region: PortSheetRegionRecord, host: str) -> float:
    """The mean position of the sheet's landing on ``host``, measured along the port direction."""
    dx, dy = region.metadata["direction"][:2]
    points = [
        point
        for overlap in region.overlaps
        if overlap.host_semantic_id == host
        for point in overlap.overlap_loop
    ]
    return sum(x * dx + y * dy for x, y in points) / len(points)
