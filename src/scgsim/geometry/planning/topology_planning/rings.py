"""Read and orient surface boundary rings in three dimensions."""

from __future__ import annotations

from typing import Any

from scgsim.geometry.models.topology import SurfacePlanRecord
from scgsim.geometry.primitives.planar_loops import clean_loop, polygon_area
from scgsim.geometry.primitives.spatial import geometry_ref_surface_z_um


def surface_ring3d_specs(
    surface: SurfacePlanRecord,
) -> tuple[tuple[str, int, tuple[tuple[float, float, float], ...]], ...]:
    """Read the outer and hole rings of a surface in three dimensions."""
    geometry_ref = surface.geometry_ref
    if "planar_loops_3d" in geometry_ref:
        return tuple(
            ("outer" if index == 0 else "hole", max(0, index - 1), _clean_ring3d(ring))
            for index, ring in enumerate(geometry_ref["planar_loops_3d"])
        )
    if "quad_points" in geometry_ref:
        return (("outer", 0, _clean_ring3d(geometry_ref["quad_points"])),)
    if "outer_loop" not in geometry_ref:
        raise ValueError(f"{surface.surface_id} requires outer_loop or quad_points")
    z_um = geometry_ref_surface_z_um(geometry_ref)
    outer_loop = _canonical_planar_loop_orientation(
        clean_loop(geometry_ref["outer_loop"])
    )
    specs = [
        (
            "outer",
            0,
            tuple((x, y, z_um) for x, y in outer_loop),
        )
    ]
    specs.extend(
        (
            "hole",
            index,
            tuple(
                (x, y, z_um)
                for x, y in _canonical_planar_loop_orientation(clean_loop(hole_loop))
            ),
        )
        for index, hole_loop in enumerate(geometry_ref.get("hole_loops", ()))
    )
    return tuple(specs)


def _canonical_planar_loop_orientation(
    loop: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    """Use one XY loop direction for OCC plane surfaces and their holes."""
    if polygon_area(loop) < 0:
        return tuple(reversed(loop))
    return loop


def _clean_ring3d(ring: Any) -> tuple[tuple[float, float, float], ...]:
    points = tuple(
        (float(point[0]), float(point[1]), float(point[2])) for point in ring
    )
    if len(points) > 1 and points[0] == points[-1]:
        points = points[:-1]
    if len(points) < 3:
        raise ValueError("3D loop requires at least 3 unique points")
    return points
