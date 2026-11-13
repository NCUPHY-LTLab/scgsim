"""Loop comparisons, contact-hole checks and the geometry references of gdstk regions, built on the planar-loop and grid operations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from scgsim.geometry.grid.booleans import (
    boolean_polygon_region,
    loop_inside_loop,
    split_gdstk_cutline_loop,
)
from scgsim.geometry.grid.constants import TOPOLOGY_EPS_UM
from scgsim.geometry.grid.coordinates import coordinate_2d_key, segment_overlap_interval
from scgsim.geometry.grid.planar_loops import (
    LoopCoordinates,
    PlanarLoop,
    PolygonCoordinates,
    clean_loop,
    filter_gdstk_polygons,
    point_on_planar_segment,
    ring_edges,
)
from scgsim.geometry.grid.scope import GridScope
from scgsim.geometry.models.input import SemanticEntitySpec


def loops_touch_without_area(left: LoopCoordinates, right: LoopCoordinates) -> bool:
    """Reject a coplanar contact that has no finite-area footprint."""
    left_loop, right_loop = clean_loop(left), clean_loop(right)
    return any(
        point_on_planar_segment(point, start, end)
        for point in left_loop
        for start, end in ring_edges(right_loop)
    ) or any(
        point_on_planar_segment(point, start, end)
        for point in right_loop
        for start, end in ring_edges(left_loop)
    )


def canonical_loop_sort_key(loop: PlanarLoop) -> PlanarLoop:
    """Return the least rotation across both orientations of a clean loop."""
    clean = clean_loop(loop)
    candidates: list[tuple[tuple[float, float], ...]] = []
    reversed_loop = tuple(reversed(clean))
    for candidate in (clean, reversed_loop):
        for offset in range(len(candidate)):
            rotated = candidate[offset:] + candidate[:offset]
            candidates.append(rotated)
    return min(candidates)


def loop_signature(loop: LoopCoordinates) -> tuple[tuple[float, float], ...]:
    """Return unordered coordinate keys, or no signature for malformed optional input."""
    try:
        points = clean_loop(loop)
    except Exception:  # noqa: BLE001 - malformed optional loop has no signature
        return ()
    return tuple(sorted(coordinate_2d_key(point) for point in points))


def same_loop_geometry(left: LoopCoordinates, right: LoopCoordinates) -> bool:
    """Compare unordered clean-loop coordinate signatures."""
    return loop_signature(left) == loop_signature(right)


def contact_holes_are_simple(
    scope: GridScope,
    outer_loop: tuple[tuple[float, float], ...],
    hole_loops: Sequence[tuple[tuple[float, float], ...]],
) -> bool:
    """Check interior holes and reject shared retained boundary support."""
    for hole_loop in hole_loops:
        if not _loop_strictly_inside_loop(scope, hole_loop, outer_loop):
            return False
    hole_records = tuple(
        (_loop_bounds_tuple(hole_loop), hole_loop) for hole_loop in hole_loops
    )
    for index, (left_bounds, left) in enumerate(hole_records):
        for right_bounds, right in hole_records[index + 1 :]:
            if not _bounds_tuple_may_overlap(left_bounds, right_bounds):
                continue
            if loops_share_edge_overlap(left, right):
                return False
    return True


def _loop_strictly_inside_loop(
    scope: GridScope, loop: LoopCoordinates, container: LoopCoordinates
) -> bool:
    """Require a real interior hole, excluding any boundary touch."""
    if not loop_inside_loop(scope, loop, container):
        return False
    loop_points = clean_loop(loop)
    container_points = clean_loop(container)
    if any(
        point_on_planar_segment(point, start, end)
        for point in loop_points
        for start, end in ring_edges(container_points)
    ):
        return False
    return not loops_share_edge_overlap(loop_points, container_points)


def _loop_bounds_tuple(
    loop: tuple[tuple[float, float], ...],
) -> tuple[float, float, float, float]:
    return (
        min(point[0] for point in loop),
        min(point[1] for point in loop),
        max(point[0] for point in loop),
        max(point[1] for point in loop),
    )


def _bounds_tuple_may_overlap(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    """Cheap candidate filter; contact/hole validity is checked on loops."""
    return not (
        left[2] <= right[0] + TOPOLOGY_EPS_UM
        or right[2] <= left[0] + TOPOLOGY_EPS_UM
        or left[3] <= right[1] + TOPOLOGY_EPS_UM
        or right[3] <= left[1] + TOPOLOGY_EPS_UM
    )


def simple_interior_hole_loops(
    scope: GridScope,
    parent_geometry_ref: Mapping[str, Any],
    plane_conductors: Sequence[SemanticEntitySpec],
    *,
    conductor_geometry_refs: Sequence[Mapping[str, Any]] = (),
) -> tuple[tuple[tuple[float, float], ...], ...] | None:
    # This shortcut constructs a new hole list from conductor footprints.  An
    # already-holed solution interface must take the Boolean path so authored
    # substrate holes remain absent from the live SA surface.
    """Use separate conductor holes only when their interior and overlap checks pass."""
    if parent_geometry_ref.get("hole_loops"):
        return None
    if any(entity.geometry.get("hole_loops") for entity in plane_conductors):
        return None
    if any(geometry_ref.get("hole_loops") for geometry_ref in conductor_geometry_refs):
        return None
    base_loop = clean_loop(parent_geometry_ref["outer_loop"])
    hole_loops = tuple(
        clean_loop(entity.geometry["outer_loop"])
        for entity in plane_conductors
        if "outer_loop" in entity.geometry
    ) + tuple(
        clean_loop(geometry_ref["outer_loop"])
        for geometry_ref in conductor_geometry_refs
        if "outer_loop" in geometry_ref
    )
    if len(hole_loops) != len(plane_conductors) + len(conductor_geometry_refs):
        return None
    if any(
        not loop_inside_loop(scope, hole_loop, base_loop) for hole_loop in hole_loops
    ):
        return None
    # Overlapping conductor footprints must be unioned before they become
    # solution-interface holes; separate overlapping holes create a third
    # coincident rim when a normalized contact-pad cap is added.
    import gdstk  # noqa: PLC0415 - defer native geometry loading until this operation requires it

    for index, left in enumerate(hole_loops):
        for right in hole_loops[index + 1 :]:
            if boolean_polygon_region(
                scope,
                gdstk,
                (gdstk.Polygon(left),),
                (gdstk.Polygon(right),),
                "and",
            ):
                return None
    all_loops = (base_loop, *hole_loops)
    for index, left in enumerate(all_loops):
        for right in all_loops[index + 1 :]:
            if loops_share_edge_overlap(left, right):
                return None
    return hole_loops


def loops_share_edge_overlap(
    left: tuple[tuple[float, float], ...],
    right: tuple[tuple[float, float], ...],
) -> bool:
    """Find retained positive-length overlap between the two loop boundaries."""
    return any(
        segment_overlap_interval(left_start, left_end, right_start, right_end)
        is not None
        for left_start, left_end in ring_edges(left)
        for right_start, right_end in ring_edges(right)
    )


def edge_is_covered_by_loop_edge(
    start: tuple[float, float],
    end: tuple[float, float],
    loop: LoopCoordinates,
) -> bool:
    """Test full segment coverage under the existing overlap interval tolerance."""
    return any(
        interval is not None
        and interval[0] <= TOPOLOGY_EPS_UM
        and interval[1] >= 1.0 - TOPOLOGY_EPS_UM
        for edge_start, edge_end in ring_edges(clean_loop(loop))
        for interval in (segment_overlap_interval(start, end, edge_start, edge_end),)
    )


def loop_centroid(loop: LoopCoordinates) -> tuple[float, float]:
    """Return the arithmetic mean of supplied vertex coordinates."""
    points = tuple((float(point[0]), float(point[1])) for point in loop)
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def gdstk_surface_region(
    scope: GridScope,
    geometry_ref: Mapping[str, Any],
) -> tuple[PolygonCoordinates, ...]:
    """Lower exterior and holes to native polygons, subtracting the holes in ``scope``."""
    import gdstk  # noqa: PLC0415 - defer native geometry loading until this operation requires it

    outer = gdstk.Polygon(clean_loop(geometry_ref["outer_loop"]))
    holes = tuple(
        gdstk.Polygon(clean_loop(hole_loop))
        for hole_loop in geometry_ref.get("hole_loops", ())
    )
    if not holes:
        return (outer,)
    return boolean_polygon_region(scope, gdstk, (outer,), holes, "not")


def geometry_ref_from_gdstk_polygon(
    scope: GridScope,
    parent_geometry_ref: Mapping[str, Any],
    polygon: PolygonCoordinates,
) -> dict[str, Any]:
    """Copy metadata and replace boundaries using the cleaned native polygon walk."""
    geometry_ref = dict(parent_geometry_ref)
    outer_loop, hole_loops = split_gdstk_cutline_loop(scope, clean_loop(polygon.points))
    geometry_ref["outer_loop"] = outer_loop
    geometry_ref["hole_loops"] = hole_loops
    return geometry_ref


def geometry_refs_from_gdstk_region(
    scope: GridScope,
    parent_geometry_ref: Mapping[str, Any],
    region: tuple[PolygonCoordinates, ...],
) -> tuple[dict[str, Any], ...]:
    """Lower each retained polygon while preserving the parent metadata."""
    return tuple(
        geometry_ref_from_gdstk_polygon(scope, parent_geometry_ref, polygon)
        for polygon in filter_gdstk_polygons(region)
    )
