"""Reject partially overlapping horizontal surfaces before backend lowering.

Owns: the same-plane overlap check on solver-live surfaces and the 2D polygon
helpers it needs.

Does not own: sidewall or arbitrary-plane overlap, which is a separate contract
and must fail fast before those cases are advertised as mesh-safe.

Overlap is judged with gdstk booleans at a precision of 1e-9 um; an overlap
smaller than `BOOLEAN_AREA_EPS_UM2` (or the relative limit) is boolean
round-off, not a real overlap.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from scgsim.geometry.models.topology import SurfacePlanRecord
from scgsim.geometry.primitives.entity_validation import (
    BOOLEAN_AREA_EPS_UM2,
    BOOLEAN_RELATIVE_AREA_EPS,
)


def validate_no_surface_overlap(
    *,
    surfaces: tuple[SurfacePlanRecord, ...],
) -> None:
    """Reject horizontal same-plane partial overlap for all solver-live surfaces.

    This guard is intentionally not allowed to skip high-count conductors by
    semantic owner. Physical groups may collapse many instances into one solver
    label, but topology validation must still see every live surface. General
    sidewall/arbitrary-plane overlap validation is a separate contract and must
    fail fast before those cases are advertised as mesh-safe.
    """
    horizontal_surfaces = tuple(
        (surface, z_um, region, _region_bounds(region))
        for surface in surfaces
        if not surface.construction_only
        for z_um, region in (_horizontal_surface_region(surface),)
        if region
    )
    outer_loop_signatures = {
        surface.surface_id: _loop2d_signature(surface.geometry_ref.get("outer_loop"))
        for surface, _, _, _ in horizontal_surfaces
    }
    known_hole_signatures = {
        surface.surface_id: _known_hole_signatures(surface)
        for surface, _, _, _ in horizontal_surfaces
    }
    surfaces_by_z = {}
    for surface, z_um, region, bounds in horizontal_surfaces:
        surfaces_by_z.setdefault(round(z_um, 9), []).append((surface, region, bounds))

    errors: list[str] = []
    for z_um, z_surfaces in surfaces_by_z.items():
        ordered = sorted(z_surfaces, key=lambda item: item[2][0])
        for index, (left, left_region, left_bounds) in enumerate(ordered):
            for right, right_region, right_bounds in ordered[index + 1 :]:
                if right_bounds[0] > left_bounds[2] + 1e-9:
                    break
                if not _bounds_overlap(left_bounds, right_bounds):
                    continue
                left_outer = outer_loop_signatures[left.surface_id]
                right_outer = outer_loop_signatures[right.surface_id]
                if (
                    left_outer is not None
                    and left_outer in known_hole_signatures[right.surface_id]
                ) or (
                    right_outer is not None
                    and right_outer in known_hole_signatures[left.surface_id]
                ):
                    continue
                if not _regions_overlap(left_region, right_region):
                    continue
                errors.append(
                    f"{left.surface_id} overlaps {right.surface_id} on z={z_um}"
                )
                break
            if errors:
                break
        if errors:
            break
    if errors:
        raise ValueError("Invalid surface overlap: " + "; ".join(errors))


def _horizontal_surface_region(
    surface: SurfacePlanRecord,
) -> tuple[float, tuple[Any, ...]]:
    geometry_ref = surface.geometry_ref
    if "outer_loop" not in geometry_ref or "quad_points" in geometry_ref:
        return 0.0, ()
    import gdstk

    outer = gdstk.Polygon(_clean_loop2d(geometry_ref["outer_loop"]))
    holes = tuple(
        gdstk.Polygon(_clean_loop2d(hole_loop))
        for hole_loop in geometry_ref.get("hole_loops", ())
    )
    if holes:
        region = tuple(gdstk.boolean((outer,), holes, "not", precision=1e-9) or ())
    else:
        region = (outer,)
    return _geometry_ref_z_um(geometry_ref), tuple(
        polygon
        for polygon in region
        if abs(float(polygon.area())) > BOOLEAN_AREA_EPS_UM2
    )


def _regions_overlap(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    import gdstk

    overlap = gdstk.boolean(left, right, "and", precision=1e-9) or ()
    overlap_area = _region_area(overlap)
    area_limit = max(
        BOOLEAN_AREA_EPS_UM2,
        BOOLEAN_RELATIVE_AREA_EPS * min(_region_area(left), _region_area(right)),
    )
    return overlap_area > area_limit


def _region_area(region: tuple[Any, ...]) -> float:
    return sum(abs(float(polygon.area())) for polygon in region)


def _known_hole_signatures(
    surface: SurfacePlanRecord,
) -> set[tuple[tuple[float, float], ...]]:
    geometry_ref = surface.geometry_ref
    return {
        signature
        for key in ("hole_loops", "contact_hole_loops")
        for loop in geometry_ref.get(key, ())
        for signature in (_loop2d_signature(loop),)
        if signature is not None
    }


def _loop2d_signature(loop: Any) -> tuple[tuple[float, float], ...] | None:
    if loop is None:
        return None
    return tuple(sorted(_coordinate_2d_key(point) for point in _clean_loop2d(loop)))


def _coordinate_2d_key(point: tuple[float, float]) -> tuple[float, float]:
    return (round(float(point[0]), 9), round(float(point[1]), 9))


def _region_bounds(region: tuple[Any, ...]) -> tuple[float, float, float, float]:
    boxes = [polygon.bounding_box() for polygon in region]
    return (
        min(float(box[0][0]) for box in boxes),
        min(float(box[0][1]) for box in boxes),
        max(float(box[1][0]) for box in boxes),
        max(float(box[1][1]) for box in boxes),
    )


def _bounds_overlap(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    return not (
        left[2] <= right[0] + 1e-9
        or right[2] <= left[0] + 1e-9
        or left[3] <= right[1] + 1e-9
        or right[3] <= left[1] + 1e-9
    )


def _clean_loop2d(loop: Any) -> tuple[tuple[float, float], ...]:
    points = tuple((float(point[0]), float(point[1])) for point in loop)
    if len(points) > 1 and points[0] == points[-1]:
        points = points[:-1]
    if len(points) < 3:
        raise ValueError("surface loop requires at least 3 points")
    return points


def _geometry_ref_z_um(geometry_ref: Mapping[str, Any]) -> float:
    z_min_um = float(geometry_ref.get("z_min_um", geometry_ref.get("z_um", 0.0)))
    if geometry_ref.get("shell_part") == "top":
        return z_min_um + float(geometry_ref.get("thickness_um", 0.0))
    if geometry_ref.get("shell_part") == "bottom":
        return z_min_um
    plane = geometry_ref.get("plane") or geometry_ref.get("contact_plane")
    if isinstance(plane, Mapping) and plane.get("axis") == "z":
        return float(plane["value_um"])
    return z_min_um
