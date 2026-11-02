"""Read authored port directions and finite remainder geometry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite, sqrt
from typing import Any, cast

from scgsim.geometry.models.input import SemanticEntitySpec
from scgsim.geometry.models.regions import PortSheetRegionRecord
from scgsim.geometry.primitives.constants import TOPOLOGY_EPS_UM
from scgsim.geometry.primitives.grid import boolean_polygon_region
from scgsim.geometry.primitives.loops import (
    gdstk_surface_region,
    geometry_refs_from_gdstk_region,
    loops_share_edge_overlap,
)
from scgsim.geometry.primitives.planar_loops import clean_loop, polygon_area


def port_sheet_remainder_geometry(region: PortSheetRegionRecord) -> dict[str, Any]:
    """Subtract the exact host overlaps and require one finite active remainder."""
    import gdstk  # noqa: PLC0415 -- load the optional polygon engine only at this native geometry boundary.

    authored = gdstk_surface_region(
        {"outer_loop": region.exterior, "hole_loops": region.holes}
    )
    overlaps = tuple(
        gdstk.Polygon(clean_loop(overlap.overlap_loop)) for overlap in region.overlaps
    )
    remainder = boolean_polygon_region(gdstk, authored, overlaps, "not")
    refs = geometry_refs_from_gdstk_region(
        {"outer_loop": region.exterior, "hole_loops": region.holes}, remainder
    )
    if len(refs) != 1:
        raise ValueError(
            f"{region.port_sheet_id} requires exactly one finite active remainder"
        )
    result = refs[0]
    area = abs(polygon_area(clean_loop(result["outer_loop"]))) - sum(
        abs(polygon_area(clean_loop(hole))) for hole in result["hole_loops"]
    )
    if not isfinite(area) or area <= TOPOLOGY_EPS_UM:
        raise ValueError(f"{region.port_sheet_id} active remainder must be finite")
    return result


def port_sheet_boundary_shares_segment(
    remainder: Mapping[str, Any], overlap_loop: Any
) -> bool:
    """Test whether an active sheet boundary shares a finite host segment."""
    return any(
        loops_share_edge_overlap(boundary, overlap_loop)
        for boundary in (remainder["outer_loop"], *remainder["hole_loops"])
    )


def port_sheet_string(
    metadata: Mapping[str, Any], key: str, region: PortSheetRegionRecord
) -> str:
    """Read one required nonempty authored port metadata string."""
    value = metadata.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{region.port_sheet_id} requires {key}")
    return value


def port_sheet_direction(region: PortSheetRegionRecord) -> tuple[float, float, float]:
    """Normalize the authored finite nonzero XY direction vector."""
    raw = region.metadata.get("direction")
    if isinstance(raw, str | bytes) or not isinstance(raw, Sequence) or len(raw) != 3:
        raise ValueError(f"{region.port_sheet_id} requires a 3D direction")
    direction = tuple(float(value) for value in raw)
    if (
        not all(isfinite(value) for value in direction)
        or direction[2] != 0.0
        or direction[0] == direction[1] == 0.0
    ):
        raise ValueError(
            f"{region.port_sheet_id} direction must be finite, XY, and nonzero"
        )
    length = sqrt(direction[0] ** 2 + direction[1] ** 2)
    return (direction[0] / length, direction[1] / length, 0.0)


def port_sheet_raw_direction(
    region: PortSheetRegionRecord,
) -> tuple[float, float, float]:
    """Read the original finite nonzero XY direction for provenance."""
    raw = region.metadata.get("direction_raw")
    if isinstance(raw, str | bytes) or not isinstance(raw, Sequence) or len(raw) != 3:
        raise ValueError(f"{region.port_sheet_id} requires raw direction provenance")
    direction = tuple(float(value) for value in raw)
    if (
        not all(isfinite(value) for value in direction)
        or direction[2] != 0.0
        or direction[0] == direction[1] == 0.0
    ):
        raise ValueError(
            f"{region.port_sheet_id} raw direction must be finite, XY, and nonzero"
        )
    return cast(tuple[float, float, float], direction)


def port_sheet_target_layer_matches(
    entity: SemanticEntitySpec, target_layer: str
) -> bool:
    """Test authored semantic, logical or GDS layer identity for a port host."""
    logical_layer = entity.metadata.get("logical_layer_id")
    return target_layer in {
        str(entity.metadata.get("semantic_group_id", "")),
        logical_layer if isinstance(logical_layer, str) else "",
        f"{entity.geometry.get('gds_layer')}/{entity.geometry.get('gds_datatype')}",
    }


def safe_port_sheet_name(value: str) -> str:
    """Convert a source name into the existing deterministic surface-name spelling."""
    return "".join(character if character.isalnum() else "_" for character in value)
