"""Source recipes round flattened points on the assembly lattice, not in the occurrence's own frame.

An occurrence rotated -45 degrees holds an unregistered child rotated +45 degrees, so the two
rotations cancel and the assembly sees the child's rectangle exactly as drawn. Rounding the cell's
local points first would move the corner at (2, 3) nm: locally it is (-0.7071, 3.5355) nm, which
rounds to (-1, 4) nm and maps back to (2.1213, 3.5355) nm, nearest the assembly node (2, 4) nm.
"""

from __future__ import annotations

import math

import pytest
from geometry_grid_fixtures import GRID

from scgsim.geometry import SemanticEntitySpec
from scgsim.geometry.grid.grid import geometry_grid_scope
from scgsim.geometry.grid.grid_regions import canonical_grid_point
from scgsim.geometry.source.occurrence import (
    occurrence_ground_polygons_for_entity,
    occurrence_polygons_for_entity,
)

gdstk = pytest.importorskip("gdstk")

METAL = (1, 0)
PLANE = (2, 0)
MASK = (3, 0)
# A 10 um square whose lower-left corner sits at (2, 3) nm, on the 1 nm lattice.
RECTANGLE = ((0.002, 0.003), (10.002, 10.003))
IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def rotation(degrees):
    """The occurrence transform (a, b, c, d, x, y) of a rotation about the origin."""
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return (c, -s, s, c, 0.0, 0.0)


def drawn(*shapes):
    """A cell holding each (layer, rectangle) shape as drawn."""
    cell = gdstk.Cell("drawn")
    for layer, (low, high) in shapes:
        cell.add(gdstk.rectangle(low, high, *layer))
    return cell


def nested(*shapes):
    """The occurrence cell: an unregistered +45 degree child holding the same shapes."""
    child = drawn(*shapes)
    occurrence = gdstk.Cell("occurrence")
    occurrence.add(gdstk.Reference(child, rotation=math.radians(45.0)))
    return occurrence


def entity(**geometry):
    """A conductor entity reading the given source geometry."""
    return SemanticEntitySpec(
        "chip/metal",
        "metal",
        "PEC",
        "conductor",
        1,
        "layout_extrusion",
        net_id="A",
        geometry=geometry,
    )


def nodes(records):
    """The assembly lattice nodes of every exterior and hole, ring by ring.

    Each ring is compared by its cyclic node sequence, whatever its start point
    and direction: the recipe orders rings in its own local frame.
    """

    def ring_nodes(ring):
        cycle = tuple(canonical_grid_point(point, GRID.grid_size_um) for point in ring)
        return min(
            order[start:] + order[:start]
            for order in (cycle, cycle[::-1])
            for start in range(len(order))
        )

    return sorted(
        (
            ring_nodes(record["exterior"]),
            tuple(sorted(map(ring_nodes, record["holes"]))),
        )
        for record in records
    )


def furthest_from_lattice(records):
    """How far any realized point lies from its own assembly lattice node, in um."""
    return max(
        math.dist(point, canonical_grid_point(point, GRID.grid_size_um))
        for record in records
        for ring in (record["exterior"], *record["holes"])
        for point in ring
    )


def test_cancelling_rotations_realize_the_drawn_polygon():
    """The nested occurrence's polygon lands on the same assembly nodes as the drawn one."""
    metal = entity(gds_layer=METAL[0], gds_datatype=METAL[1])
    with geometry_grid_scope(GRID):
        direct = occurrence_polygons_for_entity(
            metal, cell=drawn((METAL, RECTANGLE)), transform=IDENTITY
        )
        framed = occurrence_polygons_for_entity(
            metal, cell=nested((METAL, RECTANGLE)), transform=rotation(-45.0)
        )
    assert nodes(framed) == nodes(direct)
    assert furthest_from_lattice(framed) < 1e-9


def test_cancelling_rotations_derive_the_drawn_ground():
    """The ground Boolean sees the framed plane and mask on the same nodes as the drawn ones."""
    ground = entity(
        geometry_source="die_face_minus_ground_mask",
        gds_layer=MASK[0],
        gds_datatype=MASK[1],
    )
    shapes = ((PLANE, ((-20.0, -20.0), (30.0, 30.0))), (MASK, RECTANGLE))
    with geometry_grid_scope(GRID):
        direct = occurrence_ground_polygons_for_entity(
            ground, cell=drawn(*shapes), transform=IDENTITY, plane_layer=PLANE
        )
        framed = occurrence_ground_polygons_for_entity(
            ground, cell=nested(*shapes), transform=rotation(-45.0), plane_layer=PLANE
        )
    assert nodes(framed) == nodes(direct)
    assert furthest_from_lattice(framed) < 1e-9
