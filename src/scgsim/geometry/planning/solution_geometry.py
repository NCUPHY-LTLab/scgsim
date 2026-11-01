"""XY occupancy regions and bounds of solution entities (dielectric, vacuum, auto-vacuum pieces).

Owns the gdstk region helpers every domain-planning module shares. Does not decide
adjacency, surfaces or boundaries; those build on these regions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from scgsim.geometry._primitives.entity_validation import _is_solution_entity
from scgsim.geometry._primitives.loops import (
    _boolean_gdstk_region,
    _clean_loop,
    _filter_gdstk_polygons,
    _split_gdstk_cutline_loop,
)
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec


def _solution_entities(
    build_input: GeometryBuildInput,
) -> tuple[SemanticEntitySpec, ...]:
    return tuple(
        entity for entity in build_input.entities if _is_solution_entity(entity)
    )


def _solution_bounds(entity: SemanticEntitySpec) -> Mapping[str, float]:
    bounds = entity.geometry.get("domain_bounds_um")
    if not isinstance(bounds, Mapping):
        raise TypeError(f"{entity.semantic_id} requires domain_bounds_um")
    return bounds


def _solution_entity_xy_region(
    gdstk: Any,
    entity: SemanticEntitySpec,
    *,
    z_min_um: float | None = None,
    z_max_um: float | None = None,
) -> tuple[Any, ...]:
    del z_min_um, z_max_um
    geometry = entity.geometry
    if "outer_loop" in geometry:
        return _xy_region_from_loops(
            gdstk,
            geometry["outer_loop"],
            tuple(geometry.get("hole_loops", ())),
        )
    bounds = _solution_bounds(entity)
    return (gdstk.Polygon(_domain_bounds_loop(bounds)),)


def _xy_region_from_loops(
    gdstk: Any,
    outer_loop: Any,
    hole_loops: Sequence[Any],
) -> tuple[Any, ...]:
    outer = _clean_loop(outer_loop)
    split_outer, *split_holes = _split_gdstk_cutline_loop(outer)
    holes = tuple(
        _clean_loop(loop)
        for loop in tuple(hole_loops) + tuple(split_holes)
        if len(loop) >= 3
    )
    if not holes:
        return (gdstk.Polygon(split_outer),)
    return _filter_gdstk_polygons(
        gdstk.boolean(
            (gdstk.Polygon(split_outer),),
            tuple(gdstk.Polygon(hole_loop) for hole_loop in holes),
            "not",
            precision=1e-9,
        )
    )


def _auto_vacuum_subtractor_xy_region(
    gdstk: Any,
    build_input: GeometryBuildInput,
    entity: SemanticEntitySpec,
) -> tuple[Any, ...]:
    """Use the normalized polygons for a source entity lacking a planned loop."""
    if _is_solution_entity(entity) or "outer_loop" in entity.geometry:
        return _solution_entity_xy_region(gdstk, entity)

    polygons_by_id = {polygon.polygon_id: polygon for polygon in build_input.polygons}
    region: tuple[Any, ...] = ()
    for polygon_id in entity.polygon_ids:
        polygon = polygons_by_id[polygon_id]
        polygon_region = _xy_region_from_loops(
            gdstk,
            polygon.exterior,
            polygon.holes,
        )
        region = (
            polygon_region
            if not region
            else _boolean_gdstk_region(gdstk, region, polygon_region, "or")
        )
    if not entity.polygon_ids:
        return _solution_entity_xy_region(gdstk, entity)
    return region


def _entity_occupied_region(
    gdstk: Any,
    entity: SemanticEntitySpec,
    *,
    build_input: GeometryBuildInput | None = None,
) -> tuple[Any, ...]:
    """Use planned loops or the complete explicitly bound source polygon set."""
    if "outer_loop" not in entity.geometry:
        if build_input is None or not entity.polygon_ids:
            return ()
        return _auto_vacuum_subtractor_xy_region(gdstk, build_input, entity)
    outer = gdstk.Polygon(_clean_loop(entity.geometry["outer_loop"]))
    holes = tuple(
        gdstk.Polygon(_clean_loop(hole_loop))
        for hole_loop in entity.geometry.get("hole_loops", ())
    )
    if not holes:
        return _filter_gdstk_polygons((outer,))
    return _boolean_gdstk_region(gdstk, (outer,), holes, "not")


def _domain_bounds_loop(bounds: Mapping[str, Any]) -> tuple[tuple[float, float], ...]:
    return (
        (float(bounds["x_min_um"]), float(bounds["y_min_um"])),
        (float(bounds["x_max_um"]), float(bounds["y_min_um"])),
        (float(bounds["x_max_um"]), float(bounds["y_max_um"])),
        (float(bounds["x_min_um"]), float(bounds["y_max_um"])),
    )


def _loop_domain_bounds(
    outer_loop: tuple[tuple[float, float], ...],
    hole_loops: tuple[tuple[tuple[float, float], ...], ...] = (),
) -> dict[str, float]:
    points = tuple(
        point for loop in (outer_loop, *hole_loops) for point in _clean_loop(loop)
    )
    return {
        "x_min_um": min(point[0] for point in points),
        "y_min_um": min(point[1] for point in points),
        "x_max_um": max(point[0] for point in points),
        "y_max_um": max(point[1] for point in points),
    }


def _entity_loop_bounds(entity: SemanticEntitySpec) -> Mapping[str, float]:
    points = tuple(
        point
        for loop in (
            entity.geometry["outer_loop"],
            *entity.geometry.get("hole_loops", ()),
        )
        for point in _clean_loop(loop)
    )
    return {
        "x_min_um": min(point[0] for point in points),
        "y_min_um": min(point[1] for point in points),
        "x_max_um": max(point[0] for point in points),
        "y_max_um": max(point[1] for point in points),
    }
