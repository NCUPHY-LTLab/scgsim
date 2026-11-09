"""A cell placed at a quarter turn is read on the lattice it is drawn on (Issue #112).

A claw drawn as two rectangles, an arm and a bar sharing one end, is placed through a reference
rotated 270 degrees. gdstk places it with cos(3 pi / 2) = -1.8e-16 instead of 0, so the arm's long
bottom edge tilts by about 2e-14 um and ends 1e-14 um off the bar's corner. The unrounded source
reference then kept a 1e-14 um step where the bar meets the arm, and splitting that union's
cutline walk refused it: the step's two vertices each lie on the other's edge within the topology
tolerance, so the split repeated an edge (the fused claw) or branched at a vertex (the ground's
hole). A quarter turn maps the lattice onto itself, so SCGSim now places such a reference exactly.
"""

from __future__ import annotations

import json
import math
from datetime import datetime

import pytest
from geometry_grid_fixtures import GRID, ring_area

from scgsim.geometry import SemanticEntitySpec, build_gds_stack_geometry_input
from scgsim.geometry.primitives.grid import geometry_grid_scope
from scgsim.geometry.source.ground import derived_ground_polygons
from scgsim.geometry.source.occurrence import read_polygons_by_layer
from scgsim.geometry.source.polygon_ops import flattened_polygons

gdstk = pytest.importorskip("gdstk")

LAYER = (1, 0)
# The claw in its own frame, each rectangle walked as KLayout stores it.
ARM = ((32.325, 94.75), (32.325, 190.75), (48.325, 190.75), (48.325, 94.75))
BAR = ((-48.325, 174.75), (-48.325, 190.75), (48.325, 190.75), (48.325, 174.75))
# The arm (16 x 96 um) and the bar (96.65 x 16 um) overlap in one 16 x 16 um square.
CLAW_AREA_UM2 = 16 * 96 + 96.65 * 16 - 16 * 16
PLANE = ((-300.0, -300.0), (300.0, -300.0), (300.0, 300.0), (-300.0, 300.0))
QUARTER_TURNS_DEG = (90.0, 180.0, 270.0, -90.0)


def turned(ring, degrees, *, x_reflection=False):
    """The ring reflected and turned as a reference places it, computed exactly on the lattice."""
    c, s = (
        round(math.cos(math.radians(degrees))),
        round(math.sin(math.radians(degrees))),
    )
    return tuple(
        (c * x - s * y + 0.0, s * x + c * y + 0.0)
        for x, y in ((x, -y if x_reflection else y) for x, y in ring)
    )


def claw_cell(degrees, *, through_reference, x_reflection=False):
    """A top cell holding the claw, through a turned reference or drawn in place."""
    top = gdstk.Cell("TOP")
    if through_reference:
        claw = gdstk.Cell("CLAW")
        claw.add(gdstk.Polygon(ARM, *LAYER), gdstk.Polygon(BAR, *LAYER))
        top.add(
            gdstk.Reference(
                claw, rotation=math.radians(degrees), x_reflection=x_reflection
            )
        )
        return top, claw
    for ring in (ARM, BAR):
        top.add(gdstk.Polygon(turned(ring, degrees, x_reflection=x_reflection), *LAYER))
    return top, None


def rings(polygons):
    """Each polygon's walk as coordinate tuples."""
    return [tuple(map(tuple, polygon.points.tolist())) for polygon in polygons]


@pytest.mark.parametrize("x_reflection", [False, True])
@pytest.mark.parametrize("degrees", QUARTER_TURNS_DEG)
def test_a_quarter_turn_reference_is_read_on_the_lattice(degrees, x_reflection):
    """The referenced claw reads as exactly the claw drawn in place; gdstk alone is off the lattice."""
    top, _ = claw_cell(degrees, through_reference=True, x_reflection=x_reflection)
    expected = [turned(ring, degrees, x_reflection=x_reflection) for ring in (ARM, BAR)]
    assert rings(read_polygons_by_layer(top)[LAYER]) == expected
    native = rings(top.get_polygons(apply_repetitions=True, depth=None))
    assert native != expected
    assert all(
        math.dist(a, b) < 1e-12
        for ring, exact in zip(native, expected, strict=True)
        for a, b in zip(ring, exact, strict=True)
    )


def test_flattening_keeps_gdstk_order_and_nesting():
    """Own shapes, then references; repetitions per shape; nested references compose as gdstk's do."""
    leaf = gdstk.Cell("LEAF")
    leaf.add(gdstk.rectangle((0, 0), (1, 1), layer=4))
    slanted = gdstk.Cell("SLANTED")
    slanted.add(gdstk.rectangle((0, 0), (2, 1), layer=6))
    middle = gdstk.Cell("MIDDLE")
    middle.add(
        gdstk.rectangle((1, 2), (3, 5), layer=1),
        gdstk.FlexPath([(0, 0), (5, 0)], 1, layer=2),
        gdstk.Reference(leaf, (10, 0), rotation=math.pi / 2, magnification=2),
        gdstk.Reference(slanted, (3, 3), rotation=math.radians(45)),
    )
    top = gdstk.Cell("TOP")
    top.add(gdstk.rectangle((-1, -1), (0, 0), layer=5))
    top.add(
        gdstk.Reference(
            middle,
            (100, 7),
            rotation=math.radians(270),
            x_reflection=True,
            columns=2,
            rows=1,
            spacing=(50, 0),
        )
    )
    native = top.get_polygons(apply_repetitions=True, depth=None)
    exact = flattened_polygons(top)
    assert [(p.layer, p.datatype) for p in exact] == [
        (p.layer, p.datatype) for p in native
    ]
    for ours, theirs in zip(rings(exact), rings(native), strict=True):
        assert len(ours) == len(theirs)
        assert all(math.dist(a, b) < 1e-12 for a, b in zip(ours, theirs, strict=True))


def test_any_other_rotation_keeps_gdstk_placement():
    """A 45 degree reference is off the lattice as authored, and is read exactly as gdstk places it."""
    top, _ = claw_cell(45.0, through_reference=True)
    assert rings(flattened_polygons(top)) == rings(
        top.get_polygons(apply_repetitions=True, depth=None)
    )


def test_excluded_references_are_left_out():
    """A top-cell reference listed as excluded contributes no polygons."""
    top, _ = claw_cell(270.0, through_reference=True)
    top.add(gdstk.rectangle((0, 0), (1, 1), *LAYER))
    assert rings(flattened_polygons(top, excluded_reference_indexes=(0,))) == [
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    ]


def write_source(tmp_path, *, through_reference):
    """A GDS of the claw at 270 degrees and a stack reading layer 1 as one fused selector component."""
    library = gdstk.Library(unit=1e-6, precision=1e-9)
    top, claw = claw_cell(270.0, through_reference=through_reference)
    library.add(top, *((claw,) if claw is not None else ()))
    name = "reference" if through_reference else "in_place"
    gds = tmp_path / f"{name}.gds"
    library.write_gds(gds, timestamp=datetime(2000, 1, 1))
    stack = {
        "layers": [
            {
                "semantic_id": "claw",
                "role": "conductor",
                "material_id": "PEC",
                "layer": LAYER[0],
                "datatype": LAYER[1],
                "geometry": {
                    "z_um": 0,
                    "thickness_um": 1,
                    "route_ab_fused_selector_mode": True,
                },
            }
        ],
        "solution_regions": {},
        "materials": {"PEC": {"kind": "conductor"}},
        "metadata": {"top_cell_name": top.name},
    }
    stack_path = tmp_path / f"{name}.json"
    stack_path.write_text(json.dumps(stack), encoding="utf-8")
    return gds, stack_path


def test_a_turned_two_rectangle_claw_fuses_like_the_claw_drawn_in_place(tmp_path):
    """The fused selector reads the referenced claw from GDS as the one L-shaped polygon drawn in place."""
    referenced, in_place = (
        build_gds_stack_geometry_input(gds_file=gds, stack_file=stack)
        for gds, stack in (
            write_source(tmp_path, through_reference=True),
            write_source(tmp_path, through_reference=False),
        )
    )
    assert referenced.polygons == in_place.polygons
    (claw,) = referenced.polygons
    assert ring_area(claw.exterior) == pytest.approx(CLAW_AREA_UM2, abs=1e-9)


def derived_ground(top):
    """The plane minus the claw on the ground mask layer, derived under the 1 nm source grid."""
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
            "gds_layer": LAYER[0],
            "gds_datatype": LAYER[1],
        },
    )
    with geometry_grid_scope(GRID):
        return derived_ground_polygons(
            entity,
            polygons_by_layer=read_polygons_by_layer(top),
            cell_bounds_um={},
            domain_bounds_by_semantic_id={},
            plane_polygons=(gdstk.Polygon(PLANE),),
        )


def test_a_turned_two_rectangle_mask_derives_the_ground_of_the_mask_drawn_in_place():
    """The ground's hole is the claw's L shape whether the mask comes through the reference or not."""
    (referenced,) = derived_ground(claw_cell(270.0, through_reference=True)[0])
    (in_place,) = derived_ground(claw_cell(270.0, through_reference=False)[0])
    assert (referenced.exterior, referenced.holes) == (
        in_place.exterior,
        in_place.holes,
    )
    (hole,) = referenced.holes
    assert ring_area(hole) == pytest.approx(CLAW_AREA_UM2, abs=1e-9)
