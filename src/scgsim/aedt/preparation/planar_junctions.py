"""Bind explicit circuit junctions to authored or partitioned source sheets."""

from __future__ import annotations

import math
from collections.abc import Mapping
from fractions import Fraction
from typing import Any

from scgsim.aedt.preparation.junction_partition import (
    closed_contact,
    junction_contact_inventory,
    junction_geometry,
    junction_source_binding,
)
from scgsim.aedt.specs.epr import (
    PlanarJunction,
)
from scgsim.geometry import GeometryBuildInput
from scgsim.geometry.planning.junction_partition import junction_terminal_line
from scgsim.geometry.source.validation import validate_geometry_input


def junction_polygon(
    source: Mapping[str, Any], junction: PlanarJunction, polygons: Mapping[str, Any]
):
    """Select the authored or partitioned polygon recorded for one junction."""
    record = source.get("junction_partitions", {}).get(junction.junction_id)
    if record is None:
        # Historical prepared geometry retains its recorded authored sheet.
        return polygons[junction.source_polygon_id], None
    if record.get("method") != "scgsim.aedt.junction-partition.v1":
        raise ValueError("unknown junction partition method")
    if record["authored_polygon_id"] != junction.source_polygon_id:
        raise ValueError("junction partition authored source identity differs")
    return polygons[record["central_polygon_id"]], record


def planar_junction_from_port(  # noqa: PLR0913 -- preserve the supported explicit preparation keyword API.
    build_input: GeometryBuildInput,
    *,
    port_name: str,
    junction_id: str,
    terminal_a_net: str,
    terminal_b_net: str,
    inductance_h: float,
    capacitance_f: float,
) -> PlanarJunction:
    """Bind an explicit circuit junction to one normalized authored port sheet."""

    if not isinstance(build_input, GeometryBuildInput):
        raise TypeError("build_input must be GeometryBuildInput")
    if not isinstance(port_name, str) or not port_name:
        raise ValueError("port_name must be non-empty text")
    validate_geometry_input(build_input)
    matches = tuple(
        region
        for region in build_input.port_sheet_regions
        if region.metadata.get("source_name") == port_name
    )
    if len(matches) != 1:
        raise ValueError(
            f"port {port_name!r} requires exactly one normalized source sheet; "
            f"found {len(matches)}"
        )
    region = matches[0]

    # Adapter overlap inventories contain positive area only. Binding also owns
    # complete terminal-edge contact, and therefore checks the closed XY source.
    def source_rings(exterior, holes):
        return tuple(
            tuple(tuple(Fraction(str(float(v))) for v in point) for point in ring)
            for ring in (exterior, *holes)
        )

    inventory = junction_contact_inventory(
        build_input, region.source_polygon_id, terminal_a_net, terminal_b_net
    )
    sheet = source_rings(region.exterior, region.holes)
    polygons = {polygon.polygon_id: polygon for polygon in build_input.polygons}
    overlap_nets = set()
    for entity in build_input.entities:
        if entity.semantic_id not in inventory["eligible_entity_ids"]:
            continue
        for pid in entity.polygon_ids:
            if pid == region.source_polygon_id:
                continue
            polygon = polygons[pid]
            if closed_contact([sheet], [source_rings(polygon.exterior, polygon.holes)]):
                overlap_nets.add(entity.net_id)
    if overlap_nets != {terminal_a_net, terminal_b_net}:
        raise ValueError(
            f"port {port_name!r} overlaps must identify exactly the explicit "
            "terminal A and B Nets"
        )
    direction = region.metadata["direction"]
    dx, dy = float(direction[0]), float(direction[1])
    magnitude = math.hypot(dx, dy)
    dx, dy = dx / magnitude, dy / magnitude
    across = tuple(-dy * x + dx * y for x, y in region.exterior)
    width_um = max(across) - min(across)
    junction = PlanarJunction(
        junction_id=junction_id,
        source_polygon_id=region.source_polygon_id,
        terminal_a_net=terminal_a_net,
        terminal_b_net=terminal_b_net,
        direction_xy=(dx, dy),
        width_um=width_um,
        inductance_h=inductance_h,
        capacitance_f=capacitance_f,
        metadata={
            "source_port_name": port_name,
            "junction_support_plane": junction_source_binding(inventory),
        },
    )
    junction_terminal_line(
        {"exterior": region.exterior, "holes": region.holes},
        junction_geometry(junction),
        0.0,
    )
    return junction
