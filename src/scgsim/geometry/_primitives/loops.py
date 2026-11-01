"""Compose contact and geometry-reference operations from planar and grid primitives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from scgsim.geometry._primitives.constants import _TOPOLOGY_EPS_UM
from scgsim.geometry._primitives.grid import (
    boolean_polygon_region,
    loop_inside_loop,
    split_gdstk_cutline_loop,
)
from scgsim.geometry._primitives.planar_loops import (
    cancel_reversed_planar_edges,
    clean_loop,
    filter_gdstk_polygons,
    polygon_area,
    ring_edges,
    simple_planar_loops_from_edges,
)
from scgsim.geometry._primitives.spatial import (
    _coordinate_2d_key,
    _segment_overlap_interval,
)
from scgsim.geometry.models.input import SemanticEntitySpec
from scgsim.geometry.models.topology import CurveRefRecord


def _loops_touch_without_area(left: Any, right: Any) -> bool:
    """Reject a coplanar contact that has no finite-area footprint."""
    left_loop, right_loop = clean_loop(left), clean_loop(right)
    return any(
        _point_on_segment(point, start, end)
        for point in left_loop
        for start, end in ring_edges(right_loop)
    ) or any(
        _point_on_segment(point, start, end)
        for point in right_loop
        for start, end in ring_edges(left_loop)
    )


def _canonical_loop_sort_key(loop: tuple[tuple[float, float], ...]) -> tuple:
    clean = clean_loop(loop)
    candidates: list[tuple[tuple[float, float], ...]] = []
    reversed_loop = tuple(reversed(clean))
    for candidate in (clean, reversed_loop):
        for offset in range(len(candidate)):
            rotated = candidate[offset:] + candidate[:offset]
            candidates.append(rotated)
    return min(candidates)


def _loop_signature(loop: Any) -> tuple[tuple[float, float], ...]:
    try:
        points = clean_loop(loop)
    except Exception:  # noqa: BLE001 - malformed optional loop has no signature
        return ()
    return tuple(sorted(_coordinate_2d_key(point) for point in points))


def _same_loop_geometry(left: Any, right: Any) -> bool:
    return _loop_signature(left) == _loop_signature(right)


def _contact_holes_are_simple(
    outer_loop: tuple[tuple[float, float], ...],
    hole_loops: Sequence[tuple[tuple[float, float], ...]],
) -> bool:
    for hole_loop in hole_loops:
        if not _loop_strictly_inside_loop(hole_loop, outer_loop):
            return False
    hole_records = tuple(
        (_loop_bounds_tuple(hole_loop), hole_loop) for hole_loop in hole_loops
    )
    for index, (left_bounds, left) in enumerate(hole_records):
        for right_bounds, right in hole_records[index + 1 :]:
            if not _bounds_tuple_may_overlap(left_bounds, right_bounds):
                continue
            if _loops_share_edge_overlap(left, right):
                return False
    return True


def _loop_strictly_inside_loop(loop: Any, container: Any) -> bool:
    """Require a real interior hole, excluding any boundary touch."""
    if not loop_inside_loop(loop, container):
        return False
    loop_points = clean_loop(loop)
    container_points = clean_loop(container)
    if any(
        _point_on_segment(point, start, end)
        for point in loop_points
        for start, end in ring_edges(container_points)
    ):
        return False
    return not _loops_share_edge_overlap(loop_points, container_points)


def _point_on_segment(
    point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]
) -> bool:
    dx, dy = end[0] - start[0], end[1] - start[1]
    cross = (point[0] - start[0]) * dy - (point[1] - start[1]) * dx
    if abs(cross) > _TOPOLOGY_EPS_UM:
        return False
    return (
        min(start[0], end[0]) - _TOPOLOGY_EPS_UM
        <= point[0]
        <= max(start[0], end[0]) + _TOPOLOGY_EPS_UM
        and min(start[1], end[1]) - _TOPOLOGY_EPS_UM
        <= point[1]
        <= max(start[1], end[1]) + _TOPOLOGY_EPS_UM
    )


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
        left[2] <= right[0] + _TOPOLOGY_EPS_UM
        or right[2] <= left[0] + _TOPOLOGY_EPS_UM
        or left[3] <= right[1] + _TOPOLOGY_EPS_UM
        or right[3] <= left[1] + _TOPOLOGY_EPS_UM
    )


def _simple_interior_hole_loops(
    parent_geometry_ref: Mapping[str, Any],
    plane_conductors: Sequence[SemanticEntitySpec],
    *,
    conductor_geometry_refs: Sequence[Mapping[str, Any]] = (),
) -> tuple[tuple[tuple[float, float], ...], ...] | None:
    # This shortcut constructs a new hole list from conductor footprints.  An
    # already-holed solution interface must take the Boolean path so authored
    # substrate holes remain absent from the live SA surface.
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
    if any(not loop_inside_loop(hole_loop, base_loop) for hole_loop in hole_loops):
        return None
    # Overlapping conductor footprints must be unioned before they become
    # solution-interface holes; separate overlapping holes create a third
    # coincident rim when a normalized contact-pad cap is added.
    import gdstk  # optional native geometry dependency

    for index, left in enumerate(hole_loops):
        for right in hole_loops[index + 1 :]:
            if boolean_polygon_region(
                gdstk,
                (gdstk.Polygon(left),),
                (gdstk.Polygon(right),),
                "and",
            ):
                return None
    all_loops = (base_loop, *hole_loops)
    for index, left in enumerate(all_loops):
        for right in all_loops[index + 1 :]:
            if _loops_share_edge_overlap(left, right):
                return None
    return hole_loops


def _loops_share_edge_overlap(
    left: tuple[tuple[float, float], ...],
    right: tuple[tuple[float, float], ...],
) -> bool:
    return any(
        _segment_overlap_interval(left_start, left_end, right_start, right_end)
        is not None
        for left_start, left_end in ring_edges(left)
        for right_start, right_end in ring_edges(right)
    )


def _edge_is_covered_by_loop_edge(
    start: tuple[float, float],
    end: tuple[float, float],
    loop: Any,
) -> bool:
    return any(
        interval is not None
        and interval[0] <= _TOPOLOGY_EPS_UM
        and interval[1] >= 1.0 - _TOPOLOGY_EPS_UM
        for edge_start, edge_end in ring_edges(clean_loop(loop))
        for interval in (_segment_overlap_interval(start, end, edge_start, edge_end),)
    )


def _loop_centroid(loop: Any) -> tuple[float, float]:
    points = tuple((float(point[0]), float(point[1])) for point in loop)
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def _ordered_loop_signature(
    curve_refs: tuple[CurveRefRecord, ...],
) -> tuple[tuple[str, int], ...]:
    """Return a rotation-stable loop signature without discarding orientation."""
    signature = tuple((ref.curve_id, ref.orientation) for ref in curve_refs)
    if not signature:
        return ()
    rotations = (
        signature[index:] + signature[:index] for index in range(len(signature))
    )
    return min(rotations)


def _gdstk_surface_region(geometry_ref: Mapping[str, Any]) -> tuple[Any, ...]:
    import gdstk  # optional native geometry dependency

    outer = gdstk.Polygon(clean_loop(geometry_ref["outer_loop"]))
    holes = tuple(
        gdstk.Polygon(clean_loop(hole_loop))
        for hole_loop in geometry_ref.get("hole_loops", ())
    )
    if not holes:
        return (outer,)
    return boolean_polygon_region(gdstk, (outer,), holes, "not")


def _geometry_ref_from_gdstk_polygon(
    parent_geometry_ref: Mapping[str, Any],
    polygon: Any,
) -> dict[str, Any]:
    geometry_ref = dict(parent_geometry_ref)
    outer_loop, hole_loops = split_gdstk_cutline_loop(clean_loop(polygon.points))
    geometry_ref["outer_loop"] = outer_loop
    geometry_ref["hole_loops"] = hole_loops
    return geometry_ref


def _geometry_refs_from_gdstk_region(
    parent_geometry_ref: Mapping[str, Any],
    region: tuple[Any, ...],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        _geometry_ref_from_gdstk_polygon(parent_geometry_ref, polygon)
        for polygon in filter_gdstk_polygons(region)
    )


# Transitional aliases preserve callers during the coordinated internal migration.
_clean_loop = clean_loop
_ring_edges = ring_edges
_filter_gdstk_polygons = filter_gdstk_polygons
_cancel_reversed_planar_edges = cancel_reversed_planar_edges
_simple_planar_loops_from_edges = simple_planar_loops_from_edges
_polygon_area = polygon_area
_loop_inside_loop = loop_inside_loop
_boolean_gdstk_region = boolean_polygon_region
_split_gdstk_cutline_loop = split_gdstk_cutline_loop
