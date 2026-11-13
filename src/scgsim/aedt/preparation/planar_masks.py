"""Bind physical support masks on coplanar prepared EPR surfaces."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from typing import Any, cast

from scgsim.aedt.data import canonical_sha256
from scgsim.aedt.preparation.planar_source import plain
from scgsim.geometry.grid.spatial import geometry_ref_surface_z_um

INSET_CIRCLE_POINTS = 128
INSET_METHOD = "klayout_complement_minkowski.v1"

_KLAYOUT_COORD_MIN = -(2**31)
_KLAYOUT_COORD_MAX = 2**31 - 1


def bind_physical_support_masks(
    bindings: list[dict[str, Any]],
    support_bindings: list[dict[str, Any]],
    *,
    source_dbu_um: float,
) -> list[dict[str, Any]]:
    """Bind requested attribution regions to their complete physical support."""
    if not bindings:
        return []
    import klayout.db as k  # noqa: PLC0415 -- optional mask geometry dependency is loaded only when masks are bound.

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for binding in support_bindings:
        groups.setdefault(_plane_group_key(binding), []).append(binding)
    support_by_group: dict[
        tuple[Any, ...],
        tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]],
    ] = {}
    for key, group in groups.items():
        plane = group[0]["mask_plane"]
        projected = [_project_plane_region(item["mask_plane"], plane) for item in group]
        union = klayout_region(projected, source_dbu_um, k)
        if union.is_empty():
            raise RuntimeError("physical EPR support union is empty")
        support_by_group[key] = (
            plane,
            regions_from_klayout(union, source_dbu_um),
            group,
        )
    result: list[dict[str, Any]] = []
    for binding in bindings:
        support = support_by_group.get(_plane_group_key(binding))
        if support is None:
            raise RuntimeError("selected EPR surface lacks complete physical support")
        plane, support_regions, group = support
        result.append(
            {
                **binding,
                "mask_plane": {**plane, "exterior": [], "holes": []},
                "mask_support": {
                    "group_identity_sha256": canonical_sha256(
                        {
                            "plane": plane,
                            "classification": binding["contribution"]["classification"],
                            "side": binding["contribution"]["side"],
                            "material_id": binding["effective_material_id"],
                            "members": sorted(item["binding_id"] for item in group),
                        }
                    ),
                    "attribution_regions": [
                        _project_plane_region(binding["mask_plane"], plane)
                    ],
                    # Offset the complete union before intersecting owner
                    # attribution; internal owner seams are not boundaries.
                    "support_regions": support_regions,
                    "source_dbu_um": source_dbu_um,
                    "inset_method": INSET_METHOD,
                    "circle_points": INSET_CIRCLE_POINTS,
                    "klayout_version": version("klayout"),
                },
            }
        )
    return result


def surface_mask_plane(geometry_ref: Mapping[str, Any]) -> dict[str, Any]:
    """Represent a horizontal source surface in its physical mask plane."""
    outer = geometry_ref.get("outer_loop")
    plane = cast(Mapping[str, Any], geometry_ref.get("plane"))
    if isinstance(outer, Sequence) and not isinstance(outer, (str, bytes)):
        if (
            not isinstance(plane, Mapping) or plane.get("axis") != "z"
        ) and geometry_ref.get("shell_part") not in {"top", "bottom"}:
            raise ValueError("planar EPR surface loop requires an explicit Z plane")
        if geometry_ref.get("shell_part") not in {"top", "bottom"} and (
            isinstance(plane.get("value_um"), bool)
            or not isinstance(plane.get("value_um"), (int, float))
        ):
            raise ValueError("planar EPR surface requires finite plane Z")
        z_um = geometry_ref_surface_z_um(geometry_ref)
        if not math.isfinite(z_um):
            raise ValueError("planar EPR surface requires finite plane Z")
        return {
            "origin_um": [0.0, 0.0, z_um],
            "u": [1.0, 0.0, 0.0],
            "v": [0.0, 1.0, 0.0],
            "exterior": plain(outer),
            "holes": plain(geometry_ref.get("hole_loops", ())),
        }
    raise ValueError("horizontal EPR surface requires a planar loop")


def _plane_group_key(binding: Mapping[str, Any]) -> tuple[Any, ...]:
    plane = binding["mask_plane"]
    u = tuple(float(value) for value in plane["u"])
    v = tuple(float(value) for value in plane["v"])
    normal = (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )
    length = math.sqrt(sum(value * value for value in normal))
    normal = tuple(value / length for value in normal)
    first = next((value for value in normal if abs(value) > 1e-12), 1.0)
    if first < 0.0:
        normal = tuple(-value for value in normal)
    origin = tuple(float(value) for value in plane["origin_um"])
    offset = sum(normal[index] * origin[index] for index in range(3))
    evidence = binding["contribution"]
    return (
        *(round(value, 12) for value in normal),
        round(offset, 12),
        evidence["classification"],
        evidence["side"],
        binding["effective_material_id"],
    )


def _project_plane_region(
    source: Mapping[str, Any], target: Mapping[str, Any]
) -> dict[str, Any]:
    source_origin = tuple(float(value) for value in source["origin_um"])
    source_u = tuple(float(value) for value in source["u"])
    source_v = tuple(float(value) for value in source["v"])
    target_origin = tuple(float(value) for value in target["origin_um"])
    target_u = tuple(float(value) for value in target["u"])
    target_v = tuple(float(value) for value in target["v"])

    def project(point: Sequence[float]) -> list[float]:
        global_point = tuple(
            source_origin[index]
            + float(point[0]) * source_u[index]
            + float(point[1]) * source_v[index]
            for index in range(3)
        )
        delta = tuple(global_point[index] - target_origin[index] for index in range(3))
        return [
            sum(delta[index] * target_u[index] for index in range(3)),
            sum(delta[index] * target_v[index] for index in range(3)),
        ]

    return {
        "exterior": [project(point) for point in source["exterior"]],
        "holes": [
            [project(point) for point in ring] for ring in source.get("holes", ())
        ],
    }


def _klayout_point(point: Sequence[float], dbu_um: float, k: Any) -> Any:
    if len(point) != 2:
        raise ValueError("EPR mask point must have two local coordinates")
    values = [float(value) for value in point]
    if any(
        not math.isfinite(value)
        or not _KLAYOUT_COORD_MIN < value / dbu_um < _KLAYOUT_COORD_MAX
        for value in values
    ):
        raise OverflowError("EPR mask coordinate exceeds KLayout integer range")
    return k.DPoint(*values).to_itype(dbu_um)


def klayout_region(regions: Sequence[Mapping[str, Any]], dbu_um: float, k: Any) -> Any:
    """Build a KLayout region from source exterior and hole rings."""
    result = k.Region()
    for region in regions:
        exterior = [_klayout_point(point, dbu_um, k) for point in region["exterior"]]
        polygon = k.Polygon(exterior)
        for hole in region.get("holes", ()):
            polygon.insert_hole([_klayout_point(point, dbu_um, k) for point in hole])
        result.insert(polygon)
    return result.merged()


def regions_from_klayout(region: Any, dbu_um: float) -> list[dict[str, Any]]:
    # Split point contacts without changing the integer polygon set. Iterate
    # stored pieces so implicit merging cannot recreate a self-touching contour.
    """Detach KLayout polygons into sorted physical-coordinate rings."""
    simple = region.merged(True, 1)

    def points(iterator: Any) -> list[list[float]]:
        return [[point.x * dbu_um, point.y * dbu_um] for point in iterator]

    result = [
        {
            "exterior": points(polygon.each_point_hull()),
            "holes": [
                points(polygon.each_point_hole(index))
                for index in range(polygon.holes())
            ],
        }
        for polygon in simple.each()
    ]
    return sorted(result, key=canonical_sha256)
