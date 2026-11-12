"""The boundary binding's refusals and the grid audit's hole finding name the object and the place in um.

The fourth slice of the located layout refusals (``test_located_refusals.py``,
``test_located_layout_refusals.py``, ``test_located_junction_refusals.py`` and
``test_located_source_and_sgb_refusals.py`` hold the first three).  The junction partition binds
a lumped port's terminal conductors with ``construction.curved_geometry.boundaries`` before SGB
runs, so a boundary record on a terminal is refused there.  Each test declares a record on the
pad of the junction device of ``test_located_junction_refusals``, prepares the port as
``mesh()`` and ``simulation_drc`` do, and reads every coordinate back out of the refusal: each
must lie on the ring, segment or vertex at fault.  The last tests locate a hole that has no
drawn ring of its own: one drawn through a cut line in a port sheet's outline, and one a
mouth narrower than the grid closes off.
"""

from __future__ import annotations

from dataclasses import replace

import gdstk
import pytest
from test_located_junction_refusals import ELECTRODE, PAD, device, refusal
from test_located_refusals import GRID, on, places
from test_rectangular_junction_geometry import fixture as junction_fixture
from test_simulation_drc import check

from scgsim.geometry import with_rectangular_junctions
from scgsim.geometry.models.input import (
    GdsBoundaryReconstructionSpec,
    LayoutPolygonSpec,
)
from scgsim.geometry.planning.junction_partition import derive_rectangular_junctions
from scgsim.geometry.source.grid import apply_geometry_grid
from scgsim.palace import EigenmodeSim

# The pad as its GDS cell stores it: KLayout walks it clockwise from its lower-left corner,
# and the reconstruction indices name this walk.
DRAWN = ((-20.0, -10.0), (-20.0, 10.0), (0.0, 10.0), (0.0, -10.0))
PAD_RING = (-20.0, -10.0, 0.0, 10.0)
RECORD = "chip/device/PAD_ARC (Entity 'chip/device/pad', layer 1/0)"
KEY = "('chip/device/pad', 'chip/device/pad__P0000', 'outer', None)"
FIX_INDICES = "Fix the declaration: "


def vertex(index):
    """Drawn vertex ``index`` of the pad, as a box."""
    x, y = DRAWN[index]
    return (x, y, x, y)


def pad_arc(**fields):
    """An open circular reconstruction of the pad's drawn segments 0 and 1, anchored at 0, 1, 2."""
    return {
        "boundary_id": "PAD_ARC",
        "entity_id": "pad",
        "selector_point_um": (-10.0, 0.0),
        "role": "outer",
        "kind": "circular_arc",
        "segment_indices": [0, 1],
        "anchor_indices": [0, 1, 2],
        **fields,
    }


def pad_square(boundary_id, selector_point_um=(-10.0, 0.0)):
    """An authored boundary of the pad's outer ring: its four sides as lines."""
    sides = zip(DRAWN, (*DRAWN[1:], DRAWN[0]), strict=True)
    return {
        "boundary_id": boundary_id,
        "entity_id": "pad",
        "selector_point_um": selector_point_um,
        "curves": [{"kind": "line_segment", "points_um": side} for side in sides],
    }


def device_with(monkeypatch, *, reconstruction=(), curves=()):
    """The junction device with these boundary records declared on its pad."""
    component = device(monkeypatch, [PAD, ELECTRODE])
    (polygon,) = component.get_polygons(by="tuple")[(1, 0)]
    dbu = component.kcl.dbu
    assert tuple((p.x * dbu, p.y * dbu) for p in polygon.each_point_hull()) == DRAWN
    semantics = component.info["component_semantics"]
    semantics["boundary_reconstruction"] = list(reconstruction)
    semantics["boundary_curves"] = list(curves)
    component.info["component_semantics"] = semantics
    return component


def binding_refusal(monkeypatch, **records):
    """The refusal of the port's preparation, the one ``mesh()`` and ``simulation_drc`` run."""
    return refusal(device_with(monkeypatch, **records))


# ------------------------------------------------------------- the reconstruction's indices


@pytest.mark.parametrize(
    ("segments", "given"),
    [([], "it lists none"), ([0, 7], "segment index 7 names no drawn segment")],
    ids=["none", "out-of-range"],
)
def test_segments_that_name_no_drawn_segment_are_refused_at_the_ring(
    monkeypatch, segments, given
):
    """The refusal names the ring the record selects, the pad at (-20, -10)..(0, 10) um."""
    message = binding_refusal(
        monkeypatch, reconstruction=[pad_arc(segment_indices=segments)]
    )
    assert message.startswith(
        f"{RECORD} requires explicit original segment indices: {given}; the ring it "
        "selects, at "
    )
    assert f"has 4 drawn segments, numbered 0 to 3. {FIX_INDICES}" in message
    (place,) = places(message)
    assert on(place, PAD_RING) and on(PAD_RING, place)


def test_segments_out_of_drawn_order_are_refused_where_the_chain_breaks(monkeypatch):
    """Segments 0 and 2 skip the top edge: 0 ends at (-20, 10) um, 2 starts at (0, 10) um."""
    message = binding_refusal(
        monkeypatch, reconstruction=[pad_arc(segment_indices=[0, 2])]
    )
    assert message.startswith(
        f"{RECORD} segments must form one ordered source chain: segment 0 ends at "
    )
    assert message.endswith("each starting where the one before it ends.")
    end, start = places(message)
    assert on(end, vertex(1)) and on(start, vertex(2))


def test_a_segment_listed_twice_is_refused_on_it(monkeypatch):
    """Segment 0, the pad's left side from (-20, -10) to (-20, 10) um, is listed twice."""
    message = binding_refusal(
        monkeypatch, reconstruction=[pad_arc(segment_indices=[0, 0])]
    )
    assert message.startswith(
        f"{RECORD} segments must form one ordered source chain: segment 0, from "
    )
    assert "is listed 2 times" in message
    start, end = places(message)
    assert on(start, vertex(0)) and on(end, vertex(1))


@pytest.mark.parametrize(
    ("closed", "segments", "declared", "expected"),
    [
        (True, [0, 1], "declared closed", [vertex(0), vertex(2), PAD_RING]),
        (False, [0, 1, 2, 3], "declared open", [PAD_RING, vertex(0)]),
    ],
    ids=["closed-over-two-sides", "open-around-the-ring"],
)
def test_a_closure_that_differs_from_the_chain_is_refused_at_the_chain(
    monkeypatch, closed, segments, declared, expected
):
    """A closed record over two sides, or an open one around all four, names where its chain runs."""
    message = binding_refusal(
        monkeypatch,
        reconstruction=[pad_arc(segment_indices=segments, closed=closed)],
    )
    assert message.startswith(
        f"{RECORD} closure differs from the selected source chain: it is {declared}"
    )
    assert message.endswith("when its segments run once around the whole ring.")
    found = places(message)
    assert len(found) == len(expected)
    assert all(on(p, defect) for p, defect in zip(found, expected, strict=True))


def test_an_anchor_off_the_chain_is_refused_at_its_drawn_vertex(monkeypatch):
    """Anchor 3, at (0, -10) um, is not on the chain from (-20, -10) to (0, 10) um."""
    message = binding_refusal(
        monkeypatch, reconstruction=[pad_arc(anchor_indices=[0, 3, 2])]
    )
    assert message.startswith(
        f"{RECORD} anchors must belong to the selected source chain: anchor 3, the "
        "drawn vertex at "
    )
    anchor, start, end = places(message)
    assert on(anchor, vertex(3)) and on(start, vertex(0)) and on(end, vertex(2))


def test_anchors_that_leave_the_chain_ends_are_refused_at_both(monkeypatch):
    """Anchors 1, 0, 2 start at (-20, 10) um, not at the chain's start (-20, -10) um."""
    message = binding_refusal(
        monkeypatch, reconstruction=[pad_arc(anchor_indices=[1, 0, 2])]
    )
    assert message.startswith(
        f"{RECORD} reconstruction must retain the original chain endpoints: its chain "
        "runs from drawn vertex 0 at "
    )
    found = places(message)
    expected = [vertex(0), vertex(2), vertex(1), vertex(2)]
    assert all(on(p, defect) for p, defect in zip(found, expected, strict=True))
    assert len(found) == len(expected)


@pytest.mark.parametrize(
    ("record", "given", "anchors"),
    [
        (pad_arc(anchor_indices=[0, 2]), "it gives 2", [0, 2]),
        (
            pad_arc(
                segment_indices=[0, 1, 2, 3], closed=True, anchor_indices=[0, 1, 2, 3]
            ),
            "it gives 4",
            [0, 1, 2, 3],
        ),
        (
            pad_arc(segment_indices=[0, 1, 2, 3], closed=True, anchor_indices=[]),
            "it lists none, so its anchors are every vertex of its chain, 5 in all",
            [0, 1, 2, 3, 0],
        ),
    ],
    ids=["open-two", "closed-four", "closed-none"],
)
def test_a_circular_reconstruction_without_three_anchors_is_refused_at_them(
    monkeypatch, record, given, anchors
):
    """Each anchor it gives, or takes for want of any, is named at its drawn vertex."""
    message = binding_refusal(monkeypatch, reconstruction=[record])
    assert message.startswith(
        f"{RECORD} circular reconstruction requires exactly three anchors: {given}: "
    )
    found = places(message)
    assert len(found) == len(anchors)
    assert all(on(p, vertex(a)) for p, a in zip(found, anchors, strict=True))


@pytest.mark.parametrize(
    ("record", "anchors"),
    [
        (pad_arc(anchor_indices=[0, 0, 2]), [0, 0, 2]),
        (
            pad_arc(
                segment_indices=[0, 1, 2, 3], closed=True, anchor_indices=[0, 0, 1]
            ),
            [0, 0, 1],
        ),
    ],
    ids=["open", "closed"],
)
def test_anchors_that_define_no_circle_are_refused_at_them(
    monkeypatch, record, anchors
):
    """A drawn vertex given twice leaves three points on one line, so no circle runs through them."""
    message = binding_refusal(monkeypatch, reconstruction=[record])
    assert message.startswith(
        f"{RECORD}: the three arc points are collinear, so they define no circle: its "
        "anchors are drawn vertex "
    )
    assert "two of them are one vertex. Fix the declaration" in message
    found = places(message)
    assert all(on(p, vertex(a)) for p, a in zip(found, anchors, strict=True))
    assert len(found) == len(anchors)


def test_two_reconstructions_of_one_drawn_segment_are_refused_on_it(monkeypatch):
    """Both records replace drawn segment 1, the pad's top side from (-20, 10) to (0, 10) um."""
    second = pad_arc(
        boundary_id="PAD_ARC_2", segment_indices=[1, 2], anchor_indices=[1, 2, 3]
    )
    message = binding_refusal(monkeypatch, reconstruction=[pad_arc(), second])
    assert message.startswith(
        f"overlapping reconstruction segment claims on {KEY}: chip/device/PAD_ARC and "
        "chip/device/PAD_ARC_2 (Entity 'chip/device/pad' on layer 1/0) both replace the "
        "ring between "
    )
    found = places(message)
    assert len(found) == 2
    assert {next(i for i in (1, 2) if on(p, vertex(i))) for p in found} == {1, 2}


# ------------------------------------------------------------- records that claim one ring


def test_two_authored_boundaries_of_one_ring_are_refused_at_the_ring(monkeypatch):
    """Two authored records select the pad's outer ring, at (-20, -10)..(0, 10) um."""
    message = binding_refusal(
        monkeypatch,
        curves=[pad_square("PAD_OUTER"), pad_square("PAD_OUTER_2", (-5.0, 0.0))],
    )
    assert message == (
        f"multiple complete authored boundaries claim {KEY}: chip/device/PAD_OUTER and "
        "chip/device/PAD_OUTER_2 (Entity 'chip/device/pad' on layer 1/0) select the ring "
        "at (-20.000, -10.000)..(0.000, 10.000) um. Fix the declaration: give each ring "
        "one complete boundary record."
    )


def test_an_authored_and_a_reconstructed_boundary_of_one_ring_are_refused_at_it(
    monkeypatch,
):
    """An authored record and a reconstruction both select the pad's outer ring."""
    message = binding_refusal(
        monkeypatch, curves=[pad_square("PAD_OUTER")], reconstruction=[pad_arc()]
    )
    assert message.startswith(
        f"authored and reconstructed boundaries both claim {KEY}: chip/device/PAD_OUTER "
        "and chip/device/PAD_ARC (Entity 'chip/device/pad' on layer 1/0) select the ring at "
    )
    assert message.endswith("by reconstruction records, not both.")
    (place,) = places(message)
    assert on(place, PAD_RING) and on(PAD_RING, place)


# --------------------------------------------------------------- the ring a record selects


def test_a_selector_point_off_the_entity_is_refused_at_the_point(monkeypatch):
    """The selector point (50, 50) um lies outside the pad at (-20, -10)..(0, 10) um."""
    message = binding_refusal(
        monkeypatch, reconstruction=[pad_arc(selector_point_um=(50.0, 50.0))]
    )
    assert message.startswith(
        "chip/device/PAD_ARC selector must bind one Entity-owned logical polygon of "
        "'chip/device/pad'; it binds 0: the selector point lies at "
    )
    assert (
        "outside every polygon of the Entity on layer 1/0, which lie within" in message
    )
    point, extent = places(message)
    assert on(point, (50, 50, 50, 50)) and on(extent, PAD_RING) and on(PAD_RING, extent)


def test_a_boundary_of_a_hole_the_polygon_lacks_is_refused_at_the_polygon(monkeypatch):
    """A hole record on the pad, which has no hole, names the pad at (-20, -10)..(0, 10) um.

    Before, the binding indexed the missing hole and raised a bare ``IndexError``.
    """
    message = binding_refusal(
        monkeypatch, reconstruction=[pad_arc(role="hole", hole_index=0)]
    )
    assert message.startswith(
        "chip/device/PAD_ARC: 'chip/device/pad' polygon 'chip/device/pad__P0000' has no "
        "hole 0: the polygon on layer 1/0 at "
    )
    assert message.endswith("give the hole_index of one of its holes, counted from 0.")
    (place,) = places(message)
    assert on(place, PAD_RING) and on(PAD_RING, place)


def test_simulation_drc_reports_the_located_binding_refusal(monkeypatch):
    """The layout check refuses the same record, named in the layout's own frame, at the break."""
    component = device_with(
        monkeypatch, reconstruction=[pad_arc(segment_indices=[0, 2])]
    )
    with pytest.raises(ValueError) as error:
        check(EigenmodeSim, component)
    message = str(error.value)
    assert message.startswith(
        "chip/PAD_ARC (Entity 'chip/pad', layer 1/0) segments must form one ordered "
        "source chain: segment 0 ends at "
    )
    end, start = places(message)
    assert on(end, vertex(1)) and on(start, vertex(2))


# ------------------------------------------------------------ an Entity drawn on two layers

# The left terminal of the junction fixture, (-2, 0)..(3, 2) um on layer metal, and a cap
# (-2, 0)..(-1, 2) um over its left end on layer metal_b, both owned by chip/left.  A layout
# read through GeometryPlan never gives one Entity two overlapping polygons (each Entity is
# one conductor region, fused on its one layer), so the input is built by hand and goes
# through the junction partition, which binds the terminal's boundary records.
LEFT_BOX = (-2.0, 0.0, 3.0, 2.0)
CAP_BOX = (-2.0, 0.0, -1.0, 2.0)


def two_layer_terminal_refusal(selector_point_um, **fields):
    """The partition's refusal of a reconstruction on chip/left, drawn on two layers."""
    build, spec = junction_fixture()
    cap = LayoutPolygonSpec("left_cap", "metal_b", ((-2, 0), (-1, 0), (-1, 2), (-2, 2)))
    entities = tuple(
        replace(e, polygon_ids=(*e.polygon_ids, "left_cap"))
        if e.semantic_id == "chip/left"
        else e
        for e in build.entities
    )
    record = GdsBoundaryReconstructionSpec(
        "chip/LEFT_ARC",
        "chip/left",
        selector_point_um,
        "circular_arc",
        segment_indices=(0, 1),
        anchor_indices=(0, 1, 2),
        **fields,
    )
    build = replace(
        build,
        polygons=(*build.polygons, cap),
        entities=entities,
        boundary_reconstruction=(record,),
    )
    with pytest.raises(ValueError) as error:
        derive_rectangular_junctions(with_rectangular_junctions(build, (spec,)))
    return str(error.value)


def test_a_selector_in_two_polygons_names_the_layer_of_each():
    """The point (-1.5, 1) um lies in the terminal on metal and in its cap on metal_b."""
    message = two_layer_terminal_refusal((-1.5, 1.0))
    assert message.startswith(
        "chip/LEFT_ARC selector must bind one Entity-owned logical polygon of "
        "'chip/left'; it binds 2: the selector point lies at "
    )
    point, left, cap = places(message)
    assert on(point, (-1.5, 1.0, -1.5, 1.0))
    assert on(left, LEFT_BOX) and on(LEFT_BOX, left)
    assert on(cap, CAP_BOX) and on(CAP_BOX, cap)
    assert (
        "inside each of its polygons: (-2.0, 0.0)..(3.0, 2.0) um on layer metal, "
        "(-2.0, 0.0)..(-1.0, 2.0) um on layer metal_b. Fix the declaration"
    ) in message


def test_a_selector_off_an_entity_on_two_layers_names_both_layers():
    """The point (5, 5) um lies outside both; the refusal names both layers and their extent."""
    message = two_layer_terminal_refusal((5.0, 5.0))
    assert (
        "it binds 0: the selector point lies at (5.0, 5.0) um, outside every polygon of "
        "the Entity on layers metal, metal_b, which lie within "
    ) in message
    point, extent = places(message)
    assert on(point, (5.0, 5.0, 5.0, 5.0)) and on(extent, LEFT_BOX)
    assert on(LEFT_BOX, extent)


def test_a_missing_hole_names_the_layer_of_the_selected_polygon():
    """A hole record whose selector (1, 0.5) um picks the terminal on metal names metal alone."""
    message = two_layer_terminal_refusal((1.0, 0.5), role="hole", hole_index=0)
    assert message.startswith(
        "chip/LEFT_ARC: 'chip/left' polygon 'left' has no hole 0: the polygon on layer "
        "metal at "
    )
    (place,) = places(message)
    assert on(place, LEFT_BOX) and on(LEFT_BOX, place)


# ------------------------------------------------- a hole with no drawn ring of its own


def test_a_port_sheet_hole_drawn_through_a_cut_line_is_not_called_undrawn(monkeypatch):
    """``simulation_drc`` refuses a sheet drawn with a hole at x = 0.8..1.2 um, y = 0.1..0.3 um at the hole.

    gdstk draws a polygon with a hole as one walk joined to it by a cut line, and a port
    sheet keeps that walk as its outline ring, so the hole has no ring of its own until
    the grid closes the cut line.  The finding names the hole without calling it undrawn
    (before: "a hole that is not drawn appears at", the same place).
    """
    (sheet,) = gdstk.boolean(
        gdstk.rectangle((-1, -0.5), (3, 0.5)),
        gdstk.rectangle((0.8, 0.1), (1.2, 0.3)),
        "not",
    )
    component = device(monkeypatch, [PAD, ELECTRODE], sheet=sheet.points.tolist())
    with pytest.raises(ValueError) as error:
        check(EigenmodeSim, component)
    message = str(error.value)
    assert message.startswith(
        "physical geometry collapsed or changed component/hole topology at grid: "
        "PORT::PORT_SHEET__chip/P1__0000 on layer 9/0 (grid g = 0.001 um): a hole with "
        "no drawn ring of its own appears at "
    )
    assert "not drawn" not in message
    (place,) = places(message)
    assert on(place, (0.8, 0.1, 1.2, 0.3)) and on((0.8, 0.1, 1.2, 0.3), place)


# A sheet with a slot at x = 4..6 um, y = 0.5..1.5 um, open to its top side through a mouth
# 0.4 nm wide at x = 5 um.
SLOTTED = (
    (0, 0),
    (10, 0),
    (10, 2),
    (5.0002, 2),
    (5.0002, 1.5),
    (6, 1.5),
    (6, 0.5),
    (4, 0.5),
    (4, 1.5),
    (4.9998, 1.5),
    (4.9998, 2),
    (0, 2),
)


def test_a_hole_a_mouth_narrower_than_the_grid_closes_off_is_located():
    """The 0.4 nm mouth closes on the 1 nm grid; the slot becomes a hole, named at the slot."""
    build, _ = junction_fixture()
    sheet = replace(build.port_sheet_regions[0], exterior=SLOTTED)
    with pytest.raises(ValueError) as error:
        apply_geometry_grid(replace(build, port_sheet_regions=(sheet,)), GRID)
    message = str(error.value)
    assert message.startswith(
        "physical geometry collapsed or changed component/hole topology at grid: "
        "PORT::sheet on layer junction"
    )
    assert "a hole with no drawn ring of its own appears at " in message
    (place,) = places(message)
    assert on(place, (4, 0.5, 6, 1.5)) and on((4, 0.5, 6, 1.5), place)
