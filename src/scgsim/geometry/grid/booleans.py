"""Polygon Booleans in a grid scope, and the split of a drawn polygon's cutline walk into its rings.

On a model grid a Boolean is one labelled overlay noded on that grid; on no grid it is
gdstk's at the caller's precision. Every operation takes the ``GridScope`` it runs in.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, cast

from scgsim.geometry.grid.arrangement import build_grid_arrangement
from scgsim.geometry.grid.constants import TOPOLOGY_EPS_UM
from scgsim.geometry.grid.grid_regions import GridOperand, PolygonRegion
from scgsim.geometry.grid.planar_loops import (
    LoopCoordinates,
    PlanarLoop,
    PolygonCoordinates,
    boundary_meeting_points,
    clean_loop,
    cutline_boundary_edges,
    cutline_boundary_loops,
    filter_gdstk_polygons,
    polygon_area,
    repeated_boundary_edges,
)
from scgsim.geometry.grid.scope import GridScope

if TYPE_CHECKING:
    from types import ModuleType

    from gdstk import Polygon as GdstkPolygon

    from scgsim.geometry.grid.grid_regions import GridRegion


def polygon_boolean(
    scope: GridScope,
    left: Sequence[PolygonCoordinates],
    right: Sequence[PolygonCoordinates],
    operation: str,
    *,
    precision: float = 1e-9,
) -> Sequence[PolygonCoordinates]:
    """One labelled overlay on the scope's grid; gdstk's Boolean at ``precision`` without one.

    Without a grid run AEDT's native planning and a junction partition's off-axis local frame.
    """
    if scope.grid is None:
        import gdstk  # noqa: PLC0415 - defer native geometry loading until this operation requires it

        # Operands on no grid follow the native gdstk contract. These annotations
        # preserve direct native dispatch and its own malformed-input errors.
        return gdstk.boolean(
            cast("Sequence[GdstkPolygon]", left),
            cast("Sequence[GdstkPolygon]", right),
            cast("Literal['or', 'and', 'xor', 'not']", operation),
            precision=precision,
        )
    left_ids: set[str] = set()
    right_ids: set[str] = set()
    values = []
    for label, polygons, side in (("L", left, left_ids), ("R", right, right_ids)):
        for index, polygon in enumerate(polygons):
            outer, holes = polygon_rings(scope, polygon)
            operand_id = f"{label}{index}"
            side.add(operand_id)
            values.append(
                GridOperand(
                    operand_id,
                    tuple(scope.to_layout(p) for p in outer),
                    tuple(tuple(scope.to_layout(p) for p in ring) for ring in holes),
                    physical=False,
                    contact=False,
                )
            )
    arrangement = build_grid_arrangement(
        values, scope.grid, apply_grid=not scope.reference
    )
    selected = []
    for face in arrangement.faces:
        first = any(name in left_ids for name in face.operand_ids)
        second = any(name in right_ids for name in face.operand_ids)
        retain = {
            "and": first and second,
            "or": first or second,
            "not": first and not second,
            "xor": first != second,
        }[operation]
        if retain:
            selected.append(face.face_id)
    return [
        PolygonRegion(
            tuple(scope.from_layout(p) for p in outer),
            tuple(tuple(scope.from_layout(p) for p in ring) for ring in holes),
        )
        for outer, holes in arrangement.face_regions(selected)
    ]


def boolean_polygon_region(
    scope: GridScope,
    gdstk: ModuleType,
    left: Sequence[PolygonCoordinates],
    right: Sequence[PolygonCoordinates],
    operation: str,
) -> tuple[PolygonCoordinates, ...]:
    """Apply the polygon Boolean in ``scope`` and retain positive-area output polygons."""
    if not left:
        return ()
    if operation == "not" and not right:
        return filter_gdstk_polygons(left)
    result = polygon_boolean(
        scope,
        left,
        right,
        operation,
        precision=1e-9,
    )
    return filter_gdstk_polygons(result or ())


def polygon_rings(scope: GridScope, polygon: PolygonCoordinates) -> GridRegion:
    """Read stored regions or split the unchanged float-converted native walk."""
    if isinstance(polygon, PolygonRegion):
        return polygon.exterior, polygon.holes

    # Gdstk supplies XY pairs; this annotation records that native boundary
    # while leaving conversion, ordering and errors exactly as before.
    points = cast("PlanarLoop", tuple(tuple(map(float, p)) for p in polygon.points))
    return split_gdstk_cutline_loop(scope, points)


def loop_inside_loop(
    scope: GridScope, loop: LoopCoordinates, container: LoopCoordinates
) -> bool:
    """Test containment through the intersection area of the two loops in ``scope``."""
    import gdstk  # noqa: PLC0415 - defer native geometry loading until this operation requires it

    loop_points = clean_loop(loop)
    clean_container = clean_loop(container)
    loop_area = abs(polygon_area(loop_points))
    intersection = boolean_polygon_region(
        scope,
        gdstk,
        (gdstk.Polygon(loop_points),),
        (gdstk.Polygon(clean_container),),
        "and",
    )
    intersection_area = sum(
        abs(polygon_area(polygon.points)) for polygon in intersection
    )
    return loop_area - intersection_area <= max(TOPOLOGY_EPS_UM, loop_area * 1e-9)


def split_gdstk_cutline_loop(
    scope: GridScope,
    loop: tuple[tuple[float, float], ...],
) -> tuple[
    tuple[tuple[float, float], ...], tuple[tuple[tuple[float, float], ...], ...]
]:
    """Rebuild cutline boundaries and verify that each hole lies inside the outer loop."""
    try:
        outer, holes = cutline_boundary_loops(loop)
    except ValueError as error:
        # Every refusal of the walk is about the drawn boundary: keep its lead phrase
        # and locate it on the same walk, in the assembled layout frame.
        raise ValueError(_boundary_walk_refusal(scope, str(error), loop)) from error
    stray = next(
        (hole for hole in holes if not loop_inside_loop(scope, hole, outer)), None
    )
    if stray is not None:
        raise ValueError(
            scope.refusal(
                "gdstk cutline polygon has a non-interior boundary loop",
                [
                    f"its boundary loop at {scope.place(stray)} lies outside its outer loop"
                ],
                _FIX_DISJOINT,
            )
        )
    return outer, holes


FIX_EMPTY = "Fix the layout: draw the polygon with a positive area, or remove it."
_FIX_MEETING = (
    "Fix the layout: where two parts of one region meet only at a point, "
    "give them a shared edge or separate them."
)
_FIX_RETRACED = "Fix the layout: draw the outline so that it runs along each edge once."
_FIX_DISJOINT = (
    "Fix the layout: draw separate parts as separate polygons, "
    "or join them by a shared edge."
)


def _boundary_walk_refusal(scope: GridScope, lead: str, loop: PlanarLoop) -> str:
    """The refusal of a drawn boundary that is not simple loops, at the edges or points that break it."""
    edges = cutline_boundary_edges(loop)
    if retraced := repeated_boundary_edges(edges):
        findings = [f"its boundary runs twice along {scope.place(e)}" for e in retraced]
        return scope.refusal(lead, findings, _FIX_RETRACED)
    if meeting := boundary_meeting_points(edges):
        findings = [
            f"its boundary meets itself at {scope.place((p,))}" for p in meeting
        ]
        return scope.refusal(lead, findings, _FIX_MEETING)
    return scope.refusal(
        lead, [f"its boundary at {scope.place(loop)} encloses no area"], FIX_EMPTY
    )
