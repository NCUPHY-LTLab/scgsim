"""A ground mask drawn twice, once through a rotated instance and once directly, derives one clean ground.

This is the case of Issue #104: the PDK keeps a 45 degree line's ground mask on the line instance,
rotated 225 degrees, and also unions it into the die's ground mask, where KLayout has already rounded
it to the database grid.  Deriving the ground must round the rotated instance the same way, so the two
copies coincide and the plane minus the mask has the single mask's outline, with no nanometre steps.
Reading the cell keeps the instance as authored: that is the geometry the source audits compare against.
"""

from __future__ import annotations

import math

import pytest
from geometry_grid_fixtures import GRID

from scgsim.geometry import SemanticEntitySpec
from scgsim.geometry.grid.grid import geometry_grid_scope
from scgsim.geometry.source.ground import derived_ground_polygons
from scgsim.geometry.source.occurrence import read_polygons_by_layer

gdstk = pytest.importorskip("gdstk")

MASK_LAYER = (110, 0)
ROTATION_DEG = 225.0
# The line's ground mask in the line's own frame: a 16 um wide strip, 100 um long, on the lattice.
LOCAL_MASK = ((0.0, -8.0), (100.0, -8.0), (100.0, 8.0), (0.0, 8.0))
PLANE = ((-200.0, -200.0), (200.0, -200.0), (200.0, 200.0), (-200.0, 200.0))


def lattice(value):
    """The nearest 1 nm lattice coordinate, as KLayout rounds a flattened point."""
    return round(value * 1000) / 1000


def direct_mask():
    """The rotated strip as the die's ground mask stores it: rotated, then rounded to the lattice."""
    c, s = math.cos(math.radians(ROTATION_DEG)), math.sin(math.radians(ROTATION_DEG))
    return [(lattice(c * x - s * y), lattice(s * x + c * y)) for x, y in LOCAL_MASK]


def layout(*, with_instance):
    """The die cell: the direct mask copy, and optionally the line instance carrying its own copy."""
    line = gdstk.Cell("line")
    line.add(gdstk.Polygon(LOCAL_MASK, *MASK_LAYER))
    die = gdstk.Cell("die")
    die.add(gdstk.Polygon(direct_mask(), *MASK_LAYER))
    if with_instance:
        die.add(gdstk.Reference(line, rotation=math.radians(ROTATION_DEG)))
    return die


def derived_ground(die):
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
        polygons = read_polygons_by_layer(die)
        return derived_ground_polygons(
            entity,
            polygons_by_layer=polygons,
            cell_bounds_um={},
            domain_bounds_by_semantic_id={},
            plane_polygons=(gdstk.Polygon(PLANE),),
        )


def edges(ring):
    """Each edge length of a closed ring."""
    return [math.dist(a, b) for a, b in zip(ring, (*ring[1:], ring[0]), strict=False)]


def test_reading_the_cell_keeps_the_rotated_instance_as_authored():
    """Flattening keeps the floating-point rotation; only the ground's realization rounds it."""
    with geometry_grid_scope(GRID):
        polygons = read_polygons_by_layer(layout(with_instance=True))[MASK_LAYER]
    rings = sorted(sorted(map(tuple, polygon.points.tolist())) for polygon in polygons)
    assert sorted(direct_mask()) in rings
    (instance,) = [ring for ring in rings if ring != sorted(direct_mask())]
    assert all(lattice(x) != x or lattice(y) != y for x, y in instance)


def test_a_mask_drawn_twice_derives_the_single_mask_ground():
    """The ground equals the plane minus one mask copy, and no edge is shorter than 10 nm."""
    (twice,) = derived_ground(layout(with_instance=True))
    (once,) = derived_ground(layout(with_instance=False))
    assert (twice.exterior, twice.holes) == (once.exterior, once.holes)
    (hole,) = twice.holes
    assert len(hole) == 4
    assert min(edges(hole)) > 0.010
