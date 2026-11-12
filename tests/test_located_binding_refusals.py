"""A boundary record on a lumped port's terminal is refused by SGB's binding; the grid audit's hole finding names the place in um.

The fourth slice of the located layout refusals (``test_located_refusals.py``,
``test_located_layout_refusals.py``, ``test_located_junction_refusals.py`` and
``test_located_source_and_sgb_refusals.py`` hold the first three).  The junction partition reads
a lumped port's terminal conductors through SGB's own binding (``sgb.prisms.bound_rings``), so a
record on a terminal is refused there, with the ``InputError`` and the message SGB gives the same
record off a port; ``test_located_sgb_binding_refusals.py`` checks each refusal on and off a
port.  The tests here declare a record on the pad of the junction device of
``test_located_junction_refusals``, or on a terminal built by hand, and read every coordinate
back out of the refusal: each must lie on the ring, segment or vertex at fault.  The last tests
locate a hole that has no drawn ring of its own: one drawn through a cut line in a port sheet's
outline, and one a mouth narrower than the grid closes off.
"""

from __future__ import annotations

from dataclasses import replace

import gdstk
import pytest
from test_located_junction_refusals import (
    ELECTRODE,
    PAD,
    device,
    device_plan,
)
from test_located_refusals import GRID, on, places
from test_rectangular_junction_geometry import fixture as junction_fixture
from test_simulation_drc import check

from scgsim.geometry import with_rectangular_junctions
from scgsim.geometry.models.input import (
    GdsBoundaryReconstructionSpec,
    LayoutPolygonSpec,
)
from scgsim.geometry.planning.junction_partition import derive_rectangular_junctions
from scgsim.geometry.sgb.errors import InputError
from scgsim.geometry.source.grid import apply_geometry_grid
from scgsim.palace import EigenmodeSim

# The pad as its GDS cell stores it: KLayout walks it clockwise from its lower-left corner,
# and the reconstruction indices name this walk.
DRAWN = ((-20.0, -10.0), (-20.0, 10.0), (0.0, 10.0), (0.0, -10.0))
RECORD = "chip/device/PAD_ARC (Entity 'chip/device/pad', layer 1/0)"


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


def declaration_refusal(monkeypatch, **records):
    """The refusal of the plan's own preparation, where the source stage reads the records."""
    with pytest.raises(ValueError) as error:
        device_plan(device_with(monkeypatch, **records)).prepare()
    return str(error.value)


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
    """A drawn vertex given twice leaves three points on one line, so no circle runs through them.

    The source stage decides it when the plan is prepared, on the vertices as the pad's cell
    draws them, before any stage binds the record.
    """
    message = declaration_refusal(monkeypatch, reconstruction=[record])
    assert message.startswith(
        f"{RECORD}: the three arc points are collinear, so they define no circle: its "
        "anchors are drawn vertex "
    )
    assert "two of them are one vertex. Fix the declaration" in message
    found = places(message)
    assert all(on(p, vertex(a)) for p, a in zip(found, anchors, strict=True))
    assert len(found) == len(anchors)


def test_simulation_drc_reports_the_located_binding_refusal(monkeypatch):
    """The layout check refuses a record on a port's terminal with SGB's ``InputError``, named in the layout's own frame, at the break.

    Before, the junction partition bound the terminal itself and raised a ``ValueError``.
    """
    component = device_with(
        monkeypatch, reconstruction=[pad_arc(segment_indices=[0, 2])]
    )
    with pytest.raises(InputError) as error:
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
# one conductor region, fused on its one layer), so the input is built by hand, with the
# model grid declared, and goes through the junction partition, which reads the terminal's
# boundary records through SGB's binding.
LEFT_BOX = (-2.0, 0.0, 3.0, 2.0)
CAP_BOX = (-2.0, 0.0, -1.0, 2.0)


def two_layer_terminal(selector_point_um, **fields):
    """The hand-built junction input with a reconstruction on chip/left, drawn on two layers."""
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
        **{"segment_indices": (0, 1), "anchor_indices": (0, 1, 2), **fields},
    )
    build = replace(
        build,
        polygons=(*build.polygons, cap),
        entities=entities,
        boundary_reconstruction=(record,),
    )
    return with_rectangular_junctions(build, (spec,))


def two_layer_terminal_refusal(selector_point_um, **fields):
    """SGB's refusal, in the partition, of a reconstruction on chip/left, drawn on two layers."""
    build = two_layer_terminal(selector_point_um, **fields)
    gridded = replace(
        build, metadata={**build.metadata, "geometry_grid": GRID.to_payload()}
    )
    with pytest.raises(InputError) as error:
        derive_rectangular_junctions(gridded)
    return str(error.value)


def test_a_selector_in_two_polygons_names_each_polygon():
    """The point (-1.5, 1) um lies in the terminal on metal and in its cap on metal_b.

    SGB names each polygon by where it lies (before: the partition's own binding also named
    each polygon's layer).
    """
    message = two_layer_terminal_refusal((-1.5, 1.0))
    assert message.startswith(
        "chip/LEFT_ARC selector must bind one Entity-owned logical polygon of "
        "'chip/left'; it binds 2: the selector point lies at "
    )
    point, left, cap = places(message)
    assert on(point, (-1.5, 1.0, -1.5, 1.0))
    assert on(left, LEFT_BOX) and on(LEFT_BOX, left)
    assert on(cap, CAP_BOX) and on(CAP_BOX, cap)


def test_a_selector_off_an_entity_on_two_layers_names_both_layers():
    """The point (5, 5) um lies outside both; the refusal names both layers and their extent."""
    message = two_layer_terminal_refusal((5.0, 5.0))
    assert (
        "it binds 0: the selector point lies at (5.000, 5.000) um, outside every polygon "
        "of the Entity on layer metal, metal_b, which lie within "
    ) in message
    point, extent = places(message)
    assert on(point, (5.0, 5.0, 5.0, 5.0)) and on(extent, LEFT_BOX)
    assert on(LEFT_BOX, extent)


def test_a_missing_hole_is_refused_at_the_selected_polygon():
    """A hole record whose selector (1, 0.5) um picks the terminal names it at (-2, 0)..(3, 2) um.

    SGB names the Entity's layers, metal and metal_b (before: the partition's own binding
    named the selected polygon's layer, metal, alone).
    """
    message = two_layer_terminal_refusal((1.0, 0.5), role="hole", hole_index=0)
    assert message.startswith(
        "chip/LEFT_ARC: 'chip/left' polygon 0 has no hole 0: the polygon on layer "
        "metal, metal_b at "
    )
    (place,) = places(message)
    assert on(place, LEFT_BOX) and on(LEFT_BOX, place)


def test_anchors_that_define_no_circle_on_an_input_built_by_hand_are_refused_by_the_binding():
    """Anchors 0, 0, 2 of the terminal drawn from (-2, 0) through (3, 0) to (3, 1) um.

    An input built by hand never passes the source stage, which decides the anchors of a
    layout it reads, so the binding decides here, on the ring it holds.
    """
    message = two_layer_terminal_refusal((1.0, 0.5), anchor_indices=(0, 0, 2))
    assert message.startswith(
        "chip/LEFT_ARC (Entity 'chip/left', layer metal, metal_b): the three arc points "
        "are collinear, so they define no circle: its anchors are drawn vertex "
    )
    assert "two of them are one vertex. Fix the declaration" in message
    first, again, last = places(message)
    assert on(first, (-2, 0, -2, 0)) and on(again, (-2, 0, -2, 0))
    assert on(last, (3, 1, 3, 1))


def test_a_terminal_record_on_an_input_without_a_grid_is_refused():
    """A record on a port's terminal needs the model grid SGB binds it on; this input declares none.

    ``GeometryPlan`` always declares the grid, so only an input built by hand reaches this
    (before: the partition bound the record in floats without a grid).
    """
    with pytest.raises(InputError) as error:
        derive_rectangular_junctions(two_layer_terminal((1.0, 0.5)))
    assert str(error.value).startswith(
        "the input declares no geometry grid (metadata 'geometry_grid')"
    )


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
