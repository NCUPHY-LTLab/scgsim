"""Orient source surface rings for owning material volumes and solution void shells."""

from __future__ import annotations

from math import hypot
from typing import cast

from scgsim.geometry.models.common import SurfaceOrientationLiteral, Vector3D
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.topology import (
    SurfacePlanRecord,
)
from scgsim.geometry.planning.domain import (
    entity_occupied_region,
    solution_bounds,
)
from scgsim.geometry.planning.topology import (
    surface_ring3d_specs,
)
from scgsim.geometry.primitives.constants import TOPOLOGY_EPS_UM
from scgsim.geometry.primitives.entities import entity_by_id, entity_z_range_um
from scgsim.geometry.primitives.entity_validation import (
    is_solution_entity,
)
from scgsim.geometry.primitives.planar_loops import clean_loop
from scgsim.geometry.primitives.spatial import vector_cross, vector_subtract


def surface_orientation_for_volume(
    surface: SurfacePlanRecord,
    entity: SemanticEntitySpec,
) -> SurfaceOrientationLiteral:
    """Orient a planned surface as an outward face of one owning volume."""
    normal = _surface_normal_vector(surface)
    surface_centroid = _surface_centroid(surface)
    volume_center = _entity_volume_center_um(entity)
    outward = tuple(
        surface_centroid[index] - volume_center[index] for index in range(3)
    )
    dot = sum(normal[index] * outward[index] for index in range(3))
    if abs(dot) <= 1e-9:
        raise ValueError(
            f"{surface.surface_id} has ambiguous orientation for {entity.semantic_id}"
        )
    return "forward" if dot > 0 else "reversed"


def inner_void_orientation(
    build_input: GeometryBuildInput,
    *,
    surface: SurfacePlanRecord,
    component_members: set[str],
) -> SurfaceOrientationLiteral:
    """Orient from conductor center, then reverse for a solution inner shell."""
    centers = tuple(
        _entity_volume_center_um(entity_by_id(build_input, member))
        for member in component_members
    )
    component_center = tuple(
        sum(center[index] for center in centers) / len(centers) for index in range(3)
    )
    normal = _surface_normal_vector(surface)
    centroid = _surface_centroid(surface)
    dot = sum(
        normal[index] * (centroid[index] - component_center[index])
        for index in range(3)
    )
    if abs(dot) <= 1e-9:
        # A symmetric perforated conductor can put a sidewall centroid exactly
        # on the component center.  Classify the normal locally against the
        # component's occupied planar region instead of inventing a global
        # orientation convention for that degenerate center-vector case.
        normal_xy_length = hypot(normal[0], normal[1])
        if normal_xy_length > TOPOLOGY_EPS_UM:
            import gdstk  # noqa: PLC0415 - optional native polygon dependency for this degenerate-center branch

            regions = tuple(
                region
                for member in component_members
                for region in entity_occupied_region(
                    gdstk,
                    entity_by_id(build_input, member),
                )
            )
            if regions:
                epsilon = max(1e-6, normal_xy_length * 1e-6)
                plus = (
                    centroid[0] + epsilon * normal[0] / normal_xy_length,
                    centroid[1] + epsilon * normal[1] / normal_xy_length,
                )
                minus = (
                    centroid[0] - epsilon * normal[0] / normal_xy_length,
                    centroid[1] - epsilon * normal[1] / normal_xy_length,
                )
                plus_inside, minus_inside = gdstk.inside((plus, minus), regions)
                if plus_inside != minus_inside:
                    # A forward normal points out of the conductor precisely
                    # when its positive local probe is outside the component.
                    return "reversed" if not plus_inside else "forward"
        raise ValueError(
            f"{surface.surface_id} has ambiguous component void orientation"
        )
    return "reversed" if dot > 0 else "forward"


def _surface_normal_vector(surface: SurfacePlanRecord) -> tuple[float, float, float]:
    if surface.normal_hint is not None:
        return cast(Vector3D, tuple(float(value) for value in surface.normal_hint))
    ring = surface_ring3d_specs(surface)[0][2]
    origin = ring[0]
    for first_index in range(1, len(ring) - 1):
        first = vector_subtract(ring[first_index], origin)
        for second_index in range(first_index + 1, len(ring)):
            second = vector_subtract(ring[second_index], origin)
            normal = vector_cross(first, second)
            length_sq = sum(value * value for value in normal)
            if length_sq > 1e-18:
                return normal
    raise ValueError(f"{surface.surface_id} has no nondegenerate normal")


def _surface_centroid(surface: SurfacePlanRecord) -> tuple[float, float, float]:
    points = tuple(
        coordinate
        for _, _, ring in surface_ring3d_specs(surface)
        for coordinate in ring
    )
    if not points:
        raise ValueError(f"{surface.surface_id} has no coordinates")
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
        sum(point[2] for point in points) / len(points),
    )


def _entity_volume_center_um(
    entity: SemanticEntitySpec,
) -> tuple[float, float, float]:
    if is_solution_entity(entity):
        bounds = solution_bounds(entity)
        return (
            (float(bounds["x_min_um"]) + float(bounds["x_max_um"])) / 2.0,
            (float(bounds["y_min_um"]) + float(bounds["y_max_um"])) / 2.0,
            (float(entity.geometry["z_min_um"]) + float(entity.geometry["z_max_um"]))
            / 2.0,
        )
    if "outer_loop" not in entity.geometry:
        raise ValueError(f"{entity.semantic_id} requires outer_loop for volume center")
    loop = clean_loop(entity.geometry["outer_loop"])
    z_min_um, z_max_um = entity_z_range_um(entity)
    return (
        (min(point[0] for point in loop) + max(point[0] for point in loop)) / 2.0,
        (min(point[1] for point in loop) + max(point[1] for point in loop)) / 2.0,
        (z_min_um + z_max_um) / 2.0,
    )
