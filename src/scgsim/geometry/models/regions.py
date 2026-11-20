"""Records of the port-sheet regions of a layout and the overlaps they declare, and the order of a sheet's hosts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from scgsim.geometry.models.common import Coordinate, PolygonRing


@dataclass(frozen=True)
class PortSheetOverlapRecord:
    """One overlap cell between a Palace lumped-port sheet and a host polygon.

    Palace lumped-port sheets are the only intentional 2D overlay currently
    supported. Each overlap is explicit so later lowering can perform a local
    fragment operation without turning global overlap into accepted geometry.
    """

    overlap_id: str
    port_sheet_id: str
    port_polygon_id: str
    host_semantic_id: str
    host_polygon_id: str
    overlap_loop: PolygonRing
    operation: str = "local_fragment_required"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PortSheetRegionRecord:
    """Adapter-owned Palace lumped-port source polygon and its host overlaps.

    This is not a backend-live surface and must not be exported as a physical
    surface group by itself. It is the 2D region-layer contract that says which
    source polygons are allowed to overlap and where local fragmentation will be
    needed when port-sheet lowering is implemented.
    """

    port_sheet_id: str
    source_layer: str
    source_polygon_id: str
    exterior: PolygonRing
    holes: tuple[PolygonRing, ...] = ()
    overlaps: tuple[PortSheetOverlapRecord, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


def hosts_along_direction(region: PortSheetRegionRecord) -> tuple[str, ...]:
    """The sheet's host conductors in the order its authored direction meets them: the start host, then the end host.

    The direction (``metadata["direction"]``) points from the conductor at the sheet's start to the one at its
    end.  A host stands at the mean position of the sheet's landing on it, the vertices of its overlap loops,
    measured along that direction; hosts at one position keep the order their overlaps were found in.  This is
    the one rule for a port sheet's host order: SGB records the hosts in it, and Palace's partition names its
    terminal A and B by it.
    """
    dx, dy = region.metadata["direction"][:2]
    landings: dict[str, list[Coordinate]] = {}
    for overlap in region.overlaps:
        landings.setdefault(overlap.host_semantic_id, []).extend(overlap.overlap_loop)

    def along(host: str) -> float:
        points = landings[host]
        return sum(x * dx + y * dy for x, y in points) / len(points)

    return tuple(sorted(landings, key=along))
