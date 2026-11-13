"""Grid, small-arc and port-sheet refusals name the object, the layer and the place in um.

Each test draws a small layout with the defect at a known place and reads the
coordinates back out of the message: they must fall on the defect, within its
size.  A second defect shows which one is reported, or that both are.
"""

from __future__ import annotations

import re
from dataclasses import replace

import gdstk
import pytest
from sgb_inputs import gds_pad_in_ground, thin_film_pair
from test_rectangular_junction_geometry import fixture as junction_fixture

from scgsim.geometry.models.grid import GeometryGridContext
from scgsim.geometry.models.input import (
    BoundaryCurveChainSpec,
    LayoutPolygonSpec,
    SourceCurveSpec,
)
from scgsim.geometry.primitives.grid import (
    GridOperand,
    build_grid_arrangement,
    geometry_grid_scope,
)
from scgsim.geometry.sgb.arrangement.derivation.labelled import derive_labelled_arrangements
from scgsim.geometry.sgb.errors import InputError
from scgsim.geometry.sgb.prisms.catalogue import build_prism_catalogue
from scgsim.geometry.source.grid import apply_geometry_grid
from scgsim.geometry.source.ground import derived_ground_polygons

GRID = GeometryGridContext(0.001)
NUMBER = r"-?\d+\.\d+"
POINT = rf"\(({NUMBER}), ({NUMBER})\)"


def places(message: str) -> list[tuple[float, float, float, float]]:
    """Every point or box ``(x0, y0)..(x1, y1) um`` in a message, as a box."""
    found = []
    for match in re.finditer(rf"{POINT}(?:\.\.{POINT})? um", message):
        x0, y0, x1, y1 = match.groups()
        found.append(
            (float(x0), float(y0), float(x1 or x0), float(y1 or y0)),
        )
    return found


def on(place, defect, slack=0.001):
    """Whether a reported box lies on the defect box ``(x0, y0, x1, y1)``, within ``slack`` um."""
    x0, y0, x1, y1 = defect
    return (
        place[0] >= x0 - slack
        and place[1] >= y0 - slack
        and place[2] <= x1 + slack
        and place[3] <= y1 + slack
    )


def rectangle(x0, y0, x1, y1):
    """A counter-clockwise rectangle ring."""
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


# ------------------------------------------------------------- topology changed at grid


def ground_with_slits(*slits):
    """A 4 x 4 um ground declared with mask slits 0.2 nm wide, each closing on the grid."""
    build, _ = junction_fixture()
    ground = replace(
        build.entities[0],
        semantic_id="chip/ground",
        net_id="Ground",
        geometry={"mask_layer": (1, 0)},
    )
    arguments = dict(
        polygons_by_layer={
            (1, 0): tuple(
                gdstk.rectangle((x, y), (x + 0.0002, y + 1)) for x, y in slits
            )
        },
        cell_bounds_um={"x_min_um": 0, "x_max_um": 4, "y_min_um": 0, "y_max_um": 4},
        domain_bounds_by_semantic_id={},
        plane_polygons=(gdstk.rectangle((0, 0), (4, 4)),),
    )
    with geometry_grid_scope(GRID), pytest.raises(ValueError) as error:
        derived_ground_polygons(ground, **arguments)
    return str(error.value)


def test_a_declared_hole_that_closes_on_the_grid_is_located():
    """The slit at x = 1 um, y = 1..2 um is the place the message names."""
    message = ground_with_slits((1, 1))
    assert message.startswith(
        "declared physical source component/hole topology changed at grid: chip/ground"
    )
    assert "a drawn hole disappears" in message and "0.2 nm = 0.2 grid steps" in message
    assert "grid g = 0.001 um" in message and "Fix the layout" in message
    (place,) = places(message)
    assert on(place, (1, 1, 1.0002, 2))


def test_two_declared_holes_that_close_are_both_located():
    """Each slit is reported at its own place."""
    message = ground_with_slits((1, 1), (3, 2.5))
    first, second = places(message)
    assert on(first, (1, 1, 1.0002, 2))
    assert on(second, (3, 2.5, 3.0002, 3.5))


@pytest.mark.parametrize("left_x", (2, 2.0002))
def test_a_physical_operand_whose_neck_snaps_apart_is_located_on_its_layer(left_x):
    """A neck 0.4 nm wide at x = 2..3 um, y = 1 um splits the operand in two.

    With ``left_x`` = 2.0002 the left block's edge also moves when it snaps; that
    moved edge is not the neck and must not widen the reported place.
    """
    neck = (
        (0, 0),
        (left_x, 0),
        (left_x, 1),
        (3, 1),
        (3, 0),
        (5, 0),
        (5, 2),
        (3, 2),
        (3, 1.0004),
        (left_x, 1.0004),
        (left_x, 2),
        (0, 2),
    )
    with pytest.raises(ValueError) as error:
        build_grid_arrangement(
            (GridOperand("bridge", neck, contact=False, layer="metal"),),
            GRID,
        )
    message = str(error.value)
    assert message.startswith(
        "physical geometry collapsed or changed component/hole topology at grid: bridge on layer metal"
    )
    assert "splits apart" in message and "narrower than the grid" in message
    (place,) = places(message)
    assert on(place, (2, 1, 3, 1.0004))


@pytest.mark.parametrize("outer_x", (1, 0.9998))
def test_two_holes_whose_wall_snaps_away_merge_at_the_wall(outer_x):
    """Holes at x = 1..2 um and 2.0002..3 um, y = 1..2 um, lose the 0.2 nm wall between them.

    With ``outer_x`` = 0.9998 the first hole's outer edge also moves when it
    snaps; the merge is still reported at the wall only.
    """
    plate = GridOperand(
        "plate",
        rectangle(0, 0, 4, 3),
        (rectangle(outer_x, 1, 2, 2), rectangle(2.0002, 1, 3, 2)),
        contact=False,
        layer="metal",
    )
    with pytest.raises(ValueError) as error:
        build_grid_arrangement((plate,), GRID)
    message = str(error.value)
    assert "two drawn holes merge" in message
    assert "narrowest drawn gap between the merged holes is 0.2 nm" in message
    (place,) = places(message)
    assert on(place, (2, 1, 2.0002, 2))


def test_two_necks_that_snap_apart_are_both_located():
    """Three blocks joined by necks 0.4 nm wide at x = 2..3 um and x = 5..6 um."""
    necks = (
        (0, 0),
        (2, 0),
        (2, 1),
        (3, 1),
        (3, 0),
        (5, 0),
        (5, 1),
        (6, 1),
        (6, 0),
        (8, 0),
        (8, 2),
        (6, 2),
        (6, 1.0004),
        (5, 1.0004),
        (5, 2),
        (3, 2),
        (3, 1.0004),
        (2, 1.0004),
        (2, 2),
        (0, 2),
    )
    with pytest.raises(ValueError) as error:
        build_grid_arrangement(
            (GridOperand("bridge", necks, contact=False, layer="metal"),), GRID
        )
    first, second = places(str(error.value))
    assert on(first, (2, 1, 3, 1.0004))
    assert on(second, (5, 1, 6, 1.0004))


# ------------------------------------------------- split and merge shapes, round 2


def drawn_operand(boxes, name="part", **policy):
    """One operand drawn as the union of boxes ``(x0, y0, x1, y1)``, optionally with collinear vertices."""
    from shapely import box, union_all  # noqa: PLC0415 -- geometry is needed only to draw the fixture

    shape = union_all([box(*b) for b in boxes])
    return GridOperand(
        name,
        tuple(shape.exterior.coords[:-1]),
        tuple(tuple(r.coords[:-1]) for r in shape.interiors),
        layer="metal",
        **{"contact": False, **policy},
    )


def refusal(*operands):
    """The message of the grid refusal the operands raise."""
    with pytest.raises(ValueError) as error:
        build_grid_arrangement(operands, GRID)
    return str(error.value)


@pytest.mark.parametrize("extra_x", ((), (2.5,), (2.2, 2.5, 2.7)))
def test_a_neck_drawn_with_collinear_vertices_is_one_place(extra_x):
    """GEOS returns the collapsed neck as one line per drawn vertex; they are joined into one place."""
    neck = (
        (0, 0),
        (2, 0),
        (2, 1),
        *((x, 1) for x in extra_x),
        (3, 1),
        (3, 0),
        (5, 0),
        (5, 2),
        (3, 2),
        (3, 1.0004),
        *((x, 1.0004) for x in reversed(extra_x)),
        (2, 1.0004),
        (2, 2),
        (0, 2),
    )
    message = refusal(GridOperand("bridge", neck, contact=False, layer="metal"))
    assert "could not be isolated" not in message
    assert "narrower than the grid along 1000 nm" in message
    (place,) = places(message)
    assert on(place, (2, 1, 3, 1.0004))


def test_a_bent_neck_is_one_place_with_its_whole_length():
    """A neck running 2 um right, then 3 um up, collapses to one bent line 5 um long."""
    message = refusal(
        drawn_operand(
            [(0, 0, 2, 2), (2, 1, 4, 1.0004), (3.9996, 1, 4, 4), (3, 4, 5, 6)]
        )
    )
    assert "narrower than the grid along 5000 nm" in message
    (place,) = places(message)
    assert on(place, (2, 1, 4, 4))


def test_a_neck_that_collapses_to_a_point_is_honestly_not_isolated():
    """Two blocks joined at a 0.4 nm corner square meet at one point; no box is guessed for it."""
    message = refusal(
        drawn_operand(
            [
                (0, 0, 2.0001, 2.0001),
                (2.0, 2.0, 2.0004, 2.0004),
                (2.0003, 2.0003, 4, 4),
            ]
        )
    )
    assert "could not be isolated" in message
    blocks = places(message)
    assert on(blocks[0], (0, 0, 2.0001, 2.0001))
    assert on(blocks[1], (2.0, 2.0, 4, 4))


def test_a_part_with_a_surviving_middle_reports_each_neck():
    """Blocks at x = 0..2, 3..4 and 5..7 joined by two necks: three parts, two places."""
    message = refusal(
        drawn_operand(
            [
                (0, 0, 2, 2),
                (2, 1, 3, 1.0004),
                (3, 0, 4, 2),
                (4, 1, 5, 1.0004),
                (5, 0, 7, 2),
            ]
        )
    )
    first, second = places(message)
    assert on(first, (2, 1, 3, 1.0004))
    assert on(second, (4, 1, 5, 1.0004))


def test_two_necks_between_the_same_blocks_are_both_reported():
    """Necks at y = 0.5 and y = 1.5 um; the drawn hole between them closes as well."""
    message = refusal(
        drawn_operand(
            [(0, 0, 2, 2), (2, 0.5, 3, 0.5004), (2, 1.5, 3, 1.5004), (3, 0, 5, 2)]
        )
    )
    assert message.count("splits apart") == 2 and "a drawn hole disappears" in message
    found = places(message)
    assert on(found[0], (2, 0.5, 3, 0.5004))
    assert on(found[1], (2, 1.5, 3, 1.5004))
    assert on(found[2], (2, 0.5, 3, 1.5004))


def test_a_part_that_splits_into_three_at_a_t_shaped_neck_is_one_place():
    """Three blocks joined at one T-shaped neck: one connected residue reaches all three."""
    message = refusal(
        drawn_operand(
            [
                (0, 0, 2, 2),
                (2, 1, 3, 1.0004),
                (3, 0, 5, 2),
                (2.5, 1, 2.5004, 4),
                (1.5, 4, 3.5, 6),
            ]
        )
    )
    (place,) = places(message)
    assert on(place, (2, 1, 3, 4))


@pytest.mark.parametrize(
    ("low", "high", "line_y"),
    ((1, 1.0004, 1.0), (1.0001, 1.0004, 1.0), (1.0006, 1.0009, 1.001)),
)
def test_a_neck_on_or_off_a_grid_line_is_located_where_it_collapses(low, high, line_y):
    """A neck drawn on a grid line, or between two, is reported on the grid line it snaps to."""
    message = refusal(drawn_operand([(0, 0, 2, 2), (2, low, 3, high), (3, 0, 5, 2)]))
    (place,) = places(message)
    assert place == (2.0, line_y, 3.0, line_y)


def test_two_rectangles_where_one_collapses_end_in_the_contact_refusal():
    """The reviewer's case: (0, 0)..(0.0002, 1) collapses to nothing beside (0, 0)..(1, 1).

    The collapsed operand is an empty collection with no boundary; the refusal is
    still the contact ValueError, located on the drawn strip.
    """
    message = refusal(
        GridOperand("thin", rectangle(0, 0, 0.0002, 1), physical=False),
        GridOperand("full", rectangle(0, 0, 1, 1), physical=False),
    )
    assert message.startswith("physical contact changed at grid: thin, full")
    found = places(message)
    assert found and all(on(p, (0, 0, 0.0002, 1)) for p in found)


@pytest.mark.parametrize(
    ("physical", "contact"),
    ((True, True), (True, False), (False, True), (False, False)),
)
def test_an_operand_that_collapses_completely_is_refused_by_its_policy(
    physical, contact
):
    """A 0.2 nm strip beside a full square: each policy that audits it refuses with a place."""
    operands = (
        GridOperand(
            "thin", rectangle(0, 0, 0.0002, 1), physical=physical, contact=contact
        ),
        GridOperand("full", rectangle(0, 0, 1, 1), physical=physical, contact=contact),
    )
    if not (physical or contact):
        build_grid_arrangement(operands, GRID)
        return
    message = refusal(*operands)
    found = places(message)
    assert found and all(on(p, (0, 0, 0.0002, 1)) for p in found)


def test_three_holes_merge_at_each_wall():
    """Holes at x = 1..2, 2.0002..3 and 3.0002..4 um lose both 0.2 nm walls."""
    plate = GridOperand(
        "plate",
        rectangle(-1, -1, 6, 4),
        (rectangle(1, 1, 2, 2), rectangle(2.0002, 1, 3, 2), rectangle(3.0002, 1, 4, 2)),
        contact=False,
    )
    message = refusal(plate)
    assert message.count("two drawn holes merge") == 2
    first, second = places(message)
    assert on(first, (2, 1, 2.0002, 2))
    assert on(second, (3, 1, 3.0002, 2))


# -------------------------------------------------------------- contact changed at grid


def test_two_physical_operands_that_snap_into_contact_are_located_at_each_place():
    """A U-shaped operand 0.2 nm from a bar touches it along two edges after snapping."""
    bar = GridOperand("bar", rectangle(0, 0, 1, 3), physical=False, layer="m1")
    u = GridOperand(
        "u",
        (
            (1.0002, 0),
            (2, 0),
            (2, 3),
            (1.0002, 3),
            (1.0002, 2),
            (1.5, 2),
            (1.5, 1),
            (1.0002, 1),
        ),
        physical=False,
        layer="m2",
    )
    with pytest.raises(ValueError) as error:
        build_grid_arrangement((bar, u), GRID)
    message = str(error.value)
    assert message.startswith(
        "physical contact changed at grid: bar, u on layers m1, m2"
    )
    assert "narrowest drawn gap between them is 0.2 nm = 0.2 grid steps" in message
    lower, upper = places(message)
    assert on(lower, (1, 0, 1.0002, 1))
    assert on(upper, (1, 2, 1.0002, 3))


def test_a_contained_conductor_that_snaps_onto_the_boundary_is_located():
    """The inner rectangle 0.2 nm inside the outer one comes to lie on its edge y = 0.

    Both intersections equal the inner rectangle; only the boundaries' relation
    changes (212FF1FF2 to 212F11FF2), and that is the place reported.
    """
    build, _ = junction_fixture()
    outer = replace(build.polygons[0], exterior=rectangle(-1, -0.0002, 2, 2))
    inner = replace(build.polygons[1], exterior=rectangle(0, 0, 1, 1))
    build = replace(build, polygons=(outer, inner), port_sheet_regions=())
    with pytest.raises(ValueError) as error:
        apply_geometry_grid(build, GRID)
    message = str(error.value)
    assert message.startswith(
        "conductor contact changed at grid: chip/left, chip/right"
    )
    assert "they touch along an edge after snapping" in message
    assert "could not be located" not in message
    (place,) = places(message)
    assert on(place, (0, -0.0002, 1, 0))


def test_a_conductor_that_snaps_back_inside_another_is_located():
    """The inner rectangle reaches 0.2 nm below the outer one; snapping pulls it back inside.

    The part outside is lost and the boundaries come to touch along y = 0: every
    place reported lies on that 0.2 nm strip.
    """
    build, _ = junction_fixture()
    outer = replace(build.polygons[0], exterior=rectangle(-1, 0, 2, 2))
    inner = replace(build.polygons[1], exterior=rectangle(0, -0.0002, 1, 1))
    build = replace(build, polygons=(outer, inner), port_sheet_regions=())
    with pytest.raises(ValueError) as error:
        apply_geometry_grid(build, GRID)
    message = str(error.value)
    assert "the drawn part of the second outside the first" in message
    found = places(message)
    assert found and all(on(p, (0, -0.0002, 1, 0)) for p in found)


def conductors_near_the_left_terminal(x0):
    """The junction fixture with a third conductor 0.2 nm below the left terminal, from ``x0``."""
    build, _ = junction_fixture()
    other = LayoutPolygonSpec("other", "metal", rectangle(x0, -1, 5, -0.5002))
    entity = replace(
        build.entities[0], semantic_id="chip/other", net_id="C", polygon_ids=("other",)
    )
    left = replace(
        build.polygons[0],
        exterior=((-2, -0.5), (3, -0.5), (3, 1), (2, 1), (2, 2), (-2, 2)),
    )
    right = replace(
        build.polygons[1],
        exterior=((7, -0.5), (12, -0.5), (12, 2), (8, 2), (8, 1), (7, 1)),
    )
    return replace(
        build,
        polygons=(left, right, other),
        entities=(*build.entities, entity),
    )


def test_a_conductor_gap_that_closes_on_the_grid_is_located():
    """The bar 0.2 nm below the left terminal touches it along y = -0.5 um, x = -2..3 um."""
    build = conductors_near_the_left_terminal(-1.5)
    with pytest.raises(ValueError) as error:
        apply_geometry_grid(build, GRID)
    message = str(error.value)
    assert message.startswith(
        "conductor contact changed at grid: chip/left, chip/other on layer metal"
    )
    (place, *_) = places(message)
    assert on(place, (-1.5, -0.5002, 3, -0.5))


def test_of_two_closing_conductor_gaps_the_first_pair_is_reported():
    """The bar closes on the left and the right terminal; the pair met first is the one named."""
    build = conductors_near_the_left_terminal(-1.5)
    build = replace(
        build,
        polygons=(
            *build.polygons[:2],
            replace(build.polygons[2], exterior=rectangle(-1.5, -1, 9, -0.5002)),
        ),
    )
    with pytest.raises(ValueError) as error:
        apply_geometry_grid(build, GRID)
    message = str(error.value)
    assert "chip/left, chip/other" in message and "chip/right" not in message
    assert all(on(p, (-1.5, -0.5002, 3, -0.5)) for p in places(message))


def test_a_port_contact_that_closes_on_the_grid_is_located():
    """A conductor 0.2 nm above the port sheet touches it along y = 2 um, x = 4..5 um."""
    build, _ = junction_fixture()
    polygon = LayoutPolygonSpec(
        "other", "metal", ((4, 2.0002), (5, 2.0002), (5, 3), (4, 3))
    )
    entity = replace(
        build.entities[0], semantic_id="chip/other", polygon_ids=("other",)
    )
    build = replace(
        build, polygons=(*build.polygons, polygon), entities=(*build.entities, entity)
    )
    with pytest.raises(ValueError) as error:
        apply_geometry_grid(build, GRID)
    message = str(error.value)
    assert message.startswith(
        "port contact changed at grid: sheet, chip/other, other on layers junction, metal"
    )
    (place,) = places(message)
    assert on(place, (4, 2, 5, 2.0002))


# ---------------------------------------------------- the arc is too small for the grid


def pad_with_tiny_arcs(tmp_path, *heights):
    """The pad (40..60 um square) with its west edge interrupted by 0.5 nm arcs at ``heights``."""
    pad = gds_pad_in_ground(tmp_path, with_port=False)
    curves = [
        SourceCurveSpec("line_segment", ((40.0, 40.0), (60.0, 40.0))),
        SourceCurveSpec("line_segment", ((60.0, 40.0), (60.0, 60.0))),
        SourceCurveSpec("line_segment", ((60.0, 60.0), (40.0, 60.0))),
    ]
    top = 60.0
    for y in sorted(heights, reverse=True):
        curves.append(SourceCurveSpec("line_segment", ((40.0, top), (40.0, y + 0.001))))
        curves.append(
            SourceCurveSpec(
                "circular_arc", ((40.0, y + 0.001), (39.9995, y + 0.0005), (40.0, y))
            )
        )
        top = y
    curves.append(SourceCurveSpec("line_segment", ((40.0, top), (40.0, 40.0))))
    chain = BoundaryCurveChainSpec("pad_outline", "pad", (50.0, 50.0), tuple(curves))
    build = replace(pad.build_input, boundary_curves=(chain,))
    with pytest.raises(InputError) as error:
        build_prism_catalogue(build, pad.representation)
    layer = next(p.layer for p in build.polygons if p.polygon_id in _pad_ids(build))
    return str(error.value), layer


def _pad_ids(build):
    return next(e.polygon_ids for e in build.entities if e.semantic_id == "pad")


def test_an_arc_too_small_for_the_grid_is_located_on_its_entity_and_layer(tmp_path):
    """The 0.5 nm arc on the west edge at y = 50 um is the place the message names."""
    message, layer = pad_with_tiny_arcs(tmp_path, 50.0)
    assert message.startswith(f"pad_outline (Entity 'pad', layer {layer}) curve 4")
    assert (
        "the arc is too small for the grid (radius 0.5 nm = 0.5 grid steps" in message
    )
    centre, collapsed = places(message)
    assert on(centre, (39.9995, 50.0, 40.0, 50.001))
    assert on(collapsed, (39.9995, 50.0, 40.0, 50.001))


def test_of_two_arcs_too_small_the_first_in_the_chain_is_reported(tmp_path):
    """The chain runs down the west edge, so the arc at y = 50 um comes before the one at 45 um."""
    message, _ = pad_with_tiny_arcs(tmp_path, 50.0, 45.0)
    assert all(on(p, (39.9995, 50.0, 40.0, 50.001)) for p in places(message))


# ---------------------------------------------------------------------- port sheets


def test_a_port_sheet_with_no_host_is_located_on_its_layer():
    """The thin-film port spans x = 3.5..6.5 um, y = 1..3 um on layer port."""
    effective = thin_film_pair(0.5)
    port = replace(effective.build_input.port_sheet_regions[0], overlaps=())
    bare = replace(effective.build_input, port_sheet_regions=(port,))
    with pytest.raises(InputError) as error:
        build_prism_catalogue(bare, effective.representation)
    message = str(error.value)
    assert message.startswith(
        "port sheet 'port_sheet' has no host overlap, so its plane is undefined"
    )
    assert (
        "on layer port" in message
        and "Fix the layout or the port declaration" in message
    )
    (place,) = places(message)
    assert place == (3.5, 1.0, 6.5, 3.0)


def test_of_two_port_sheets_with_no_host_the_first_is_reported():
    """Ports are read in source order: the first sheet without a host is the one named."""
    effective = thin_film_pair(0.5)
    region = replace(effective.build_input.port_sheet_regions[0], overlaps=())
    second = replace(
        region, port_sheet_id="second", exterior=rectangle(3.5, 5.0, 6.5, 6.0)
    )
    bare = replace(effective.build_input, port_sheet_regions=(region, second))
    with pytest.raises(InputError) as error:
        build_prism_catalogue(bare, effective.representation)
    message = str(error.value)
    assert "'port_sheet'" in message and "'second'" not in message
    assert places(message) == [(3.5, 1.0, 6.5, 3.0)]


def test_a_port_sheet_with_hosts_on_two_planes_is_located(tmp_path):
    """The pad-in-ground port spans x = 29..41 um, y = 48..52 um."""
    effective = gds_pad_in_ground(tmp_path, attachment=("lower", "upper"))
    with pytest.raises(InputError) as error:
        build_prism_catalogue(effective.build_input, effective.representation)
    message = str(error.value)
    assert "do not share one authored support plane" in message
    assert "Fix the declarations or the layout" in message
    (place,) = places(message)
    assert place == (29.0, 48.0, 41.0, 52.0)


def covered_ports(*boxes):
    """The thin-film pair with one port sheet per box, each drawn inside the left host."""
    effective = thin_film_pair(0.5)
    region = effective.build_input.port_sheet_regions[0]
    ports = tuple(
        replace(region, port_sheet_id=f"port_{k}", exterior=rectangle(*box), holes=())
        for k, box in enumerate(boxes)
    )
    build = replace(effective.build_input, port_sheet_regions=ports)
    with pytest.raises(InputError) as error:
        derive_labelled_arrangements(build, effective.representation)
    return str(error.value)


def test_a_port_sheet_covered_by_its_hosts_is_located():
    """The sheet drawn at x = 1..2 um, y = 1..2 um lies wholly on the left host."""
    message = covered_ports((1.0, 1.0, 2.0, 2.0))
    assert message.startswith("port sheet 'port_0' has no active remainder")
    assert "on layer port" in message and "its hosts (chip/left, chip/right)" in message
    (place,) = places(message)
    assert place == (1.0, 1.0, 2.0, 2.0)


def test_of_two_covered_port_sheets_the_first_is_reported():
    """Ports are checked in source order: the first covered sheet is the one named."""
    message = covered_ports((1.0, 1.0, 2.0, 2.0), (2.5, 2.0, 3.5, 3.0))
    assert "'port_0'" in message and "'port_1'" not in message
    (place,) = places(message)
    assert place == (1.0, 1.0, 2.0, 2.0)
