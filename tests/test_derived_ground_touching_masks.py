"""A derived ground keeps the exterior and holes its Boolean returns (Issue #87).

The ground is the plane minus its mask. Under a grid, that Boolean already
returns each region as an exterior and its holes. Two valid results used to be
refused because the recipe rebuilt the loops from the region's cutline walk:

* a hole that touches the exterior, or another hole, at one vertex: the walk
  meets that vertex four times and is no longer a set of simple loops;
* masks placed through references rotated by angles that are not quarter
  turns: the unrounded source reference the grid audit compares against holds
  vertex pairs one float step apart, joined by edges the walk repeats.

Each case is drawn with gdstk and derived under the 1 nm source grid, so both
the unrounded reference and the gridded ground are built.
"""

from __future__ import annotations

import math

import pytest
from geometry_grid_fixtures import GRID, rectangle, ring_area

from scgsim.geometry import SemanticEntitySpec
from scgsim.geometry.primitives.grid import geometry_grid_scope
from scgsim.geometry.source.ground import derived_ground_polygons
from scgsim.geometry.source.occurrence import read_polygons_by_layer

gdstk = pytest.importorskip("gdstk")

MASK_LAYER = (110, 0)
PLANE = rectangle(0.0, 0.0, 100.0, 100.0)
# A 20 um diamond standing on the plane's lower edge at (50, 0).
DIAMOND_ON_EDGE = ((50.0, 0.0), (60.0, 10.0), (50.0, 20.0), (40.0, 10.0))


def derived_ground(die, plane=PLANE):
    """The die's ground, the plane minus its mask layer, derived under the 1 nm source grid."""
    entity = SemanticEntitySpec(
        "chip/ground",
        "metal",
        "PEC",
        "conductor",
        1,
        "layout_extrusion",
        net_id="Ground",
        geometry={
            "geometry_source": "die_face_minus_ground_mask",
            "gds_layer": MASK_LAYER[0],
            "gds_datatype": MASK_LAYER[1],
        },
    )
    with geometry_grid_scope(GRID):
        return derived_ground_polygons(
            entity,
            polygons_by_layer=read_polygons_by_layer(die),
            cell_bounds_um={},
            domain_bounds_by_semantic_id={},
            plane_polygons=(gdstk.Polygon(plane),),
        )


def die_with_masks(*masks):
    """A die cell holding each mask polygon drawn directly on the mask layer."""
    die = gdstk.Cell("die")
    for mask in masks:
        die.add(gdstk.Polygon(mask, *MASK_LAYER))
    return die


def ground_area(polygon):
    """The area of one derived ground polygon: its exterior minus its holes."""
    return ring_area(polygon.exterior) - sum(ring_area(hole) for hole in polygon.holes)


def test_two_masks_touching_at_a_corner_leave_two_holes():
    """Two 20 um squares meeting at (40, 40) cut two holes that share only that vertex."""
    lower = rectangle(20.0, 20.0, 40.0, 40.0)
    upper = rectangle(40.0, 40.0, 60.0, 60.0)
    (ground,) = derived_ground(die_with_masks(lower, upper))
    assert set(ground.exterior) == set(PLANE)
    assert sorted(set(hole) for hole in ground.holes) == sorted(
        (set(lower), set(upper))
    )
    first, second = ground.holes
    assert set(first) & set(second) == {(40.0, 40.0)}
    assert ground_area(ground) == 100.0 * 100.0 - 2 * 20.0 * 20.0


def test_a_hole_touching_the_exterior_at_one_vertex_is_kept():
    """A diamond standing on the plane's edge is a hole that shares the vertex (50, 0) with it."""
    (ground,) = derived_ground(die_with_masks(DIAMOND_ON_EDGE))
    (hole,) = ground.holes
    assert set(hole) == set(DIAMOND_ON_EDGE)
    assert set(ground.exterior) == {*PLANE, (50.0, 0.0)}
    assert ground_area(ground) == 100.0 * 100.0 - 200.0


def test_holes_touching_the_exterior_and_each_other_are_kept():
    """A second diamond stacked on the first touches it at (50, 20); both stay holes."""
    stacked = tuple((x, y + 20.0) for x, y in DIAMOND_ON_EDGE)
    (ground,) = derived_ground(die_with_masks(DIAMOND_ON_EDGE, stacked))
    assert sorted(set(hole) for hole in ground.holes) == sorted(
        (set(DIAMOND_ON_EDGE), set(stacked))
    )
    assert ground_area(ground) == 100.0 * 100.0 - 2 * 200.0


def strips_rotated_two_ways():
    """Two 50 x 10 um strips sharing a long edge, both turned 21 degrees.

    One strip is turned by a single 21 degree reference, the other by three
    nested 7 degree references, so the corners of their shared edge come out
    one float step apart when the cell is flattened.
    """
    lower = gdstk.Cell("lower_strip")
    lower.add(gdstk.Polygon(rectangle(0.0, 0.0, 50.0, 10.0), *MASK_LAYER))
    upper = gdstk.Cell("upper_strip")
    upper.add(gdstk.Polygon(rectangle(0.0, 10.0, 50.0, 20.0), *MASK_LAYER))
    turned = upper
    for depth in range(3):
        outer = gdstk.Cell(f"upper_turn_{depth}")
        outer.add(gdstk.Reference(turned, rotation=math.radians(7.0)))
        turned = outer
    die = gdstk.Cell("die")
    die.add(gdstk.Reference(turned))
    die.add(gdstk.Reference(lower, rotation=math.radians(21.0)))
    return die


def test_the_flattened_strips_hold_vertex_pairs_one_float_step_apart():
    """The case is real: the shared edge's corners differ by less than 1e-12 um after flattening."""
    with geometry_grid_scope(GRID):
        polygons = read_polygons_by_layer(strips_rotated_two_ways())[MASK_LAYER]
    points = sorted(
        {tuple(point) for polygon in polygons for point in polygon.points.tolist()}
    )
    near = [
        (a, b)
        for a, b in zip(points, points[1:], strict=False)
        if math.dist(a, b) < 1e-12
    ]
    assert len(near) == 2


def test_strips_rotated_two_ways_derive_one_ground_with_one_hole():
    """The strips cut one 50 x 20 um hole, rounded to the grid.

    Its four corners are hole vertices within one grid step; the ends of the
    shared edge stay as vertices, rounded off the corners' straight line.
    """
    (ground,) = derived_ground(
        strips_rotated_two_ways(), plane=rectangle(-200.0, -200.0, 200.0, 200.0)
    )
    (hole,) = ground.holes
    assert len(hole) == 6
    # Rounding each vertex by at most half a step moves the area by under perimeter x step.
    assert abs(ring_area(hole) - 50.0 * 20.0) < 140.0 * 0.001
    turn = math.radians(21.0)
    corners = [
        (
            x * math.cos(turn) - y * math.sin(turn),
            x * math.sin(turn) + y * math.cos(turn),
        )
        for x, y in rectangle(0.0, 0.0, 50.0, 20.0)
    ]
    assert all(
        min(math.dist(corner, vertex) for vertex in hole) < 0.001 for corner in corners
    )
