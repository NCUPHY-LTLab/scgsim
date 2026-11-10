"""Bind authored layout ports to Palace lumped-port records, without Sim state.

Owns two steps of a lumped port's preparation: resolving the authored port of a Component
(``resolve_layout_ports``), and asking the shared geometry for the rectangle Palace's lumped
element needs (``with_rectangular_ports``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any, cast

from scgsim.geometry import (
    GeometryBuildInput,
    JunctionTerminalBinding,
    PortSheetRegionRecord,
    RectangularJunctionSpec,
    SemanticEntitySpec,
    with_rectangular_junctions,
)
from scgsim.geometry.models.junctions import junction_net_id

from .config import LayoutPortBinding


@dataclass(frozen=True)
class RequestedPort:
    """One caller-selected authored port with its target layer and inductance."""

    name: str
    layer: str
    inductance: float


def resolve_layout_ports(
    component: Any, ports: list[RequestedPort]
) -> tuple[list[LayoutPortBinding], list[dict[str, Any]]]:
    """Resolve exact authored layers and outward orientations in request order."""
    resolved: list[LayoutPortBinding] = []
    records: list[dict[str, Any]] = []
    used_layers: set[tuple[int, int]] = set()
    component_ports = getattr(component, "ports", None)
    if component_ports is None:
        raise TypeError("component must expose authored ports.")
    for index, requested in enumerate(ports, start=1):
        try:
            port = component_ports[requested.name]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"component has no authored port {requested.name!r}."
            ) from exc
        source = _port_layer(port)
        if source in used_layers:
            raise ValueError(
                "layout_sheet ports must use distinct authored GDS layers."
            )
        used_layers.add(source)
        orientation = getattr(port, "orientation", None)
        if not isinstance(orientation, Real) or isinstance(orientation, bool):
            raise TypeError(
                f"authored port {requested.name!r} needs numeric orientation."
            )
        angle = math.radians(float(orientation))
        direction = (round(math.cos(angle), 15), round(math.sin(angle), 15), 0.0)
        length = math.hypot(direction[0], direction[1])
        if not math.isfinite(length) or length == 0.0:
            raise ValueError(f"authored port {requested.name!r} has invalid direction.")
        direction = (direction[0] / length, direction[1] / length, 0.0)
        source_layer = f"{source[0]}/{source[1]}"
        resolved.append(
            LayoutPortBinding(
                index=index,
                name=requested.name,
                target_layer=requested.layer,
                source_layer=source_layer,
                direction=direction,
                inductance=requested.inductance,
            )
        )
        records.append(
            {
                "layer": source[0],
                "datatype": source[1],
                "name": requested.name,
                "source": "palace_lumped_port_sheet",
                "port_index": index,
                "target_layer": requested.layer,
                "direction": list(direction),
                "direction_sign_convention": "gdsfactory_port_orientation_outward",
            }
        )
    return resolved, records


def _port_layer(port: Any) -> tuple[int, int]:
    raw = getattr(port, "layer", None)
    if isinstance(raw, (tuple, list)) and len(raw) == 2:
        return (int(raw[0]), int(raw[1]))
    if hasattr(raw, "layer") and hasattr(raw, "datatype"):
        layer_record = cast(Any, raw)
        return (int(layer_record.layer), int(layer_record.datatype))
    try:
        info = port.kcl.layout.get_info(int(str(raw)))
        return (int(info.layer), int(info.datatype))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("layout_sheet port requires an exact GDS layer.") from exc


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
    added = tuple(
        _port_request(region, entities)
        for region in build_input.port_sheet_regions
        if region.port_sheet_id not in requested
    )
    if not added and len(kept) == len(existing):
        return build_input
    return with_rectangular_junctions(build_input, (*kept, *added))


def _port_request(
    region: PortSheetRegionRecord, entities: Mapping[str, SemanticEntitySpec]
) -> RectangularJunctionSpec:
    """The partition request of one port sheet: its own identity and its two host conductors.

    Terminal A is the host at the start of the port direction and Terminal B the host it
    points to, the order the partition cuts its two ends in.
    """
    name = region.metadata.get("source_name", region.port_sheet_id)
    hosts = tuple(dict.fromkeys(item.host_semantic_id for item in region.overlaps))
    if len(hosts) != 2:
        raise ValueError(
            f"lumped port {name!r} must land on two conductors to be made a rectangle; "
            f"it lands on {len(hosts)}: {list(hosts)}."
        )
    terminals = [_terminal(entities[host]) for host in hosts]
    terminals.sort(key=lambda item: _along_direction(region, item.source_entity_id))
    return RectangularJunctionSpec(
        junction_id=region.port_sheet_id,
        port_sheet_id=region.port_sheet_id,
        source_polygon_id=region.source_polygon_id,
        source_occurrence_path=str(region.metadata.get("source_occurrence_path")),
        terminal_a=terminals[0],
        terminal_b=terminals[1],
    )


def _terminal(entity: SemanticEntitySpec) -> JunctionTerminalBinding:
    """The terminal binding of one host conductor; one without a Net is a Net of its own."""
    return JunctionTerminalBinding(
        source_entity_id=entity.semantic_id,
        source_occurrence_path=str(entity.metadata.get("source_occurrence_path")),
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
