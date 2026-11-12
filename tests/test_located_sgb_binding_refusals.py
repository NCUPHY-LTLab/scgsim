"""A boundary record that names no chain or no circle is refused with the record, its Entity and layer, the place in um and who acts.

The fifth slice of the located layout refusals (``test_located_binding_refusals.py`` holds the
fourth).  Three stages refuse a boundary record:

* The source stage (``source.curves``) decides, when a plan is prepared, whether a declared arc
  defines a circle: a component's arc on the points as the component writes them, a
  reconstruction's anchors on the vertices as their cell draws them.  An occurrence transform
  turns both in floats, where three points on one line seldom stay on one line.
* SGB (``sgb.prisms.curves``) binds the records of a conductor that is not a lumped port's
  terminal.
* The construction binder binds a terminal's records first, for the junction partition, and
  gives SGB's refusal for the same fault, word for word.

Each test declares a record on a pad, runs ``simulation_drc``, the plan's preparation or
``mesh()``'s, and reads every coordinate back out of the refusal: each must lie on the ring,
segment, vertex or arc point at fault.
"""

from __future__ import annotations

import math
import re

import pytest
from sgb_inputs import gds_pad_in_ground
from test_located_refusals import on, places
from test_located_source_and_sgb_refusals import pads_with_selectors
from test_lumped_ports_at_any_angle import device_with_port, placed, turned
from test_palace_rectangular_ports import prepared_port
from test_sgb_prisms import with_boundaries
from test_simulation_drc import (
    LAYER_STACK,
    MATERIALS,
    NO_VACUUM_STACK,
    MeshingReached,
    check,
    configured,
)

import scgsim.palace.mesh.build as build_module
from scgsim.geometry import GeometryPlan
from scgsim.geometry.construction.curved_geometry.boundaries import (
    source_boundaries,
    source_rings,
)
from scgsim.geometry.models.input import (
    BoundaryCurveChainSpec,
    GdsBoundaryReconstructionSpec,
    SourceCurveSpec,
)
from scgsim.geometry.sgb.errors import InputError
from scgsim.geometry.sgb.prisms.catalogue import build_prism_catalogue
from scgsim.palace import EigenmodeSim

# The pad as its GDS cell stores it: KLayout walks it clockwise from its lower-left corner,
# and the reconstruction indices name this walk.
DRAWN = ((0.0, 0.0), (0.0, 20.0), (20.0, 20.0), (20.0, 0.0))
PAD_RING = (0.0, 0.0, 20.0, 20.0)
RECORD = "chip/PAD_ARC (Entity 'chip/pad', layer 1/0)"


def vertex(index):
    """Drawn vertex ``index`` of the pad, as a box."""
    x, y = DRAWN[index]
    return (x, y, x, y)


def pad_arc(**fields):
    """An open circular reconstruction of the pad's drawn segments 0 and 1, anchored at 0, 1, 2."""
    return {
        "boundary_id": "PAD_ARC",
        "entity_id": "pad",
        "selector_point_um": (10.0, 10.0),
        "role": "outer",
        "kind": "circular_arc",
        "segment_indices": [0, 1],
        "anchor_indices": [0, 1, 2],
        **fields,
    }


def pads_with(monkeypatch, *, port=False, reconstruction=(), curves=()):
    """``two_pads`` with these boundary records on the pad; with ``port`` the pad is a lumped port's terminal."""
    component = pads_with_selectors(monkeypatch, port=port)
    (polygon,) = component.get_polygons(by="tuple")[(1, 0)]
    dbu = component.kcl.dbu
    assert tuple((p.x * dbu, p.y * dbu) for p in polygon.each_point_hull()) == DRAWN
    semantics = component.info["component_semantics"]
    semantics["boundary_reconstruction"] = list(reconstruction)
    semantics["boundary_curves"] = list(curves)
    component.info["component_semantics"] = semantics
    return component


def sgb_refusal(monkeypatch, **records):
    """SGB's refusal of the records on the pad, as ``simulation_drc`` reports it."""
    with pytest.raises(InputError) as error:
        check(EigenmodeSim, pads_with(monkeypatch, **records))
    return str(error.value)


def terminal_refusal(monkeypatch, **records):
    """The construction binder's refusal of the records, with the pad a lumped port's terminal."""
    with pytest.raises(ValueError) as error:
        check(EigenmodeSim, pads_with(monkeypatch, port=True, **records))
    return str(error.value)


def one_refusal(monkeypatch, **records):
    """SGB's refusal, checked to be the construction binder's for the same records."""
    message = sgb_refusal(monkeypatch, **records)
    assert terminal_refusal(monkeypatch, **records) == message
    return message


def declared_refusal(monkeypatch, **records):
    """The source stage's refusal of the records, the same with the pad off and on a lumped port.

    ``simulation_drc`` raises it while it prepares the plan, before a binding stage sees the
    record.  Off a port only SGB would bind it, and SGB's refusals are no ``ValueError``.
    """
    messages = []
    for port in (False, True):
        with pytest.raises(ValueError) as error:
            check(EigenmodeSim, pads_with(monkeypatch, port=port, **records))
        messages.append(str(error.value))
    assert messages[0] == messages[1]
    return messages[0]


# ------------------------------------------------------------- the reconstruction's indices


@pytest.mark.parametrize(
    ("segments", "given"),
    [([], "it lists none"), ([0, 7], "segment index 7 names no drawn segment")],
    ids=["none", "out-of-range"],
)
def test_segments_that_name_no_drawn_segment_are_refused_at_the_ring(
    monkeypatch, segments, given
):
    """The refusal names the ring the record selects, the pad at (0, 0)..(20, 20) um."""
    message = one_refusal(
        monkeypatch, reconstruction=[pad_arc(segment_indices=segments)]
    )
    assert message.startswith(
        f"{RECORD} requires explicit original segment indices: {given}; the ring it "
        "selects, at "
    )
    assert "has 4 drawn segments, numbered 0 to 3. Fix the declaration: " in message
    (place,) = places(message)
    assert on(place, PAD_RING) and on(PAD_RING, place)


def test_segments_out_of_drawn_order_are_refused_where_the_chain_breaks(monkeypatch):
    """Segments 0 and 2 skip the top side: 0 ends at (0, 20) um, 2 starts at (20, 20) um."""
    message = one_refusal(monkeypatch, reconstruction=[pad_arc(segment_indices=[0, 2])])
    assert message.startswith(
        f"{RECORD} segments must form one ordered source chain: segment 0 ends at "
    )
    assert message.endswith("each starting where the one before it ends.")
    end, start = places(message)
    assert on(end, vertex(1)) and on(start, vertex(2))


def test_a_segment_listed_twice_is_refused_on_it(monkeypatch):
    """Segment 0, the pad's left side from (0, 0) to (0, 20) um, is listed twice."""
    message = one_refusal(monkeypatch, reconstruction=[pad_arc(segment_indices=[0, 0])])
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
    message = one_refusal(
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
    """Anchor 3, at (20, 0) um, is not on the chain from (0, 0) to (20, 20) um."""
    message = one_refusal(
        monkeypatch, reconstruction=[pad_arc(anchor_indices=[0, 3, 2])]
    )
    assert message.startswith(
        f"{RECORD} anchors must belong to the selected source chain: anchor 3, the "
        "drawn vertex at "
    )
    anchor, start, end = places(message)
    assert on(anchor, vertex(3)) and on(start, vertex(0)) and on(end, vertex(2))


def test_anchors_that_leave_the_chain_ends_are_refused_at_both(monkeypatch):
    """Anchors 1, 0, 2 start at (0, 20) um, not at the chain's start (0, 0) um."""
    message = one_refusal(
        monkeypatch, reconstruction=[pad_arc(anchor_indices=[1, 0, 2])]
    )
    assert message.startswith(
        f"{RECORD} reconstruction must retain the original chain endpoints: its chain "
        "runs from drawn vertex 0 at "
    )
    found = places(message)
    expected = [vertex(0), vertex(2), vertex(1), vertex(2)]
    assert len(found) == len(expected)
    assert all(on(p, defect) for p, defect in zip(found, expected, strict=True))


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
    message = one_refusal(monkeypatch, reconstruction=[record])
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
    """A drawn vertex given twice leaves three points on one line, so no circle runs through them.

    The source stage decides it, on the vertices as the pad's cell draws them.
    """
    message = declared_refusal(monkeypatch, reconstruction=[record])
    assert message.startswith(
        f"{RECORD}: the three arc points are collinear, so they define no circle: its "
        "anchors are drawn vertex "
    )
    assert "two of them are one vertex. Fix the declaration" in message
    found = places(message)
    assert len(found) == len(anchors)
    assert all(on(p, vertex(a)) for p, a in zip(found, anchors, strict=True))


@pytest.mark.parametrize("kind", ["bspline", "interpolation_spline"])
def test_a_reconstruction_of_a_kind_sgb_does_not_carry_is_refused_at_the_ring(
    monkeypatch, kind
):
    """A spline reconstruction of the pad names the ring it selects, at (0, 0)..(20, 20) um.

    On a lumped port's terminal the construction binder accepts the record, because the
    compiler builds spline reconstructions; the junction partition then refuses the spline
    on the terminal's outline, located (``test_located_junction_refusals.py``).
    """
    message = sgb_refusal(monkeypatch, reconstruction=[pad_arc(kind=kind)])
    assert message.startswith(
        f"{RECORD}: reconstruction kind {kind!r} is not supported yet (this version "
        "carries lines and circular arcs); the ring it selects lies at "
    )
    assert message.endswith(
        "Fix the declaration: give the record kind 'circular_arc', or remove it to keep "
        "the ring as drawn."
    )
    (place,) = places(message)
    assert on(place, PAD_RING) and on(PAD_RING, place)


def test_two_reconstructions_of_one_drawn_segment_are_refused_on_it(monkeypatch):
    """Both records replace drawn segment 1, the pad's top side from (0, 20) to (20, 20) um.

    Each stage names the ring by its own key; the words after it are the same.
    """
    second = pad_arc(
        boundary_id="PAD_ARC_2", segment_indices=[1, 2], anchor_indices=[1, 2, 3]
    )
    records = {"reconstruction": [pad_arc(), second]}
    message = sgb_refusal(monkeypatch, **records)
    assert message.startswith(
        "overlapping reconstruction segment claims on ring ('chip/pad', 0, 0): "
        "chip/PAD_ARC and chip/PAD_ARC_2 (Entity 'chip/pad' on layer 1/0) both replace "
        "the ring between "
    )
    assert message.endswith(
        "Fix the declaration: give each drawn segment of the ring to one reconstruction "
        "record."
    )
    found = places(message)
    assert len(found) == 2
    assert {next(i for i in (1, 2) if on(p, vertex(i))) for p in found} == {1, 2}
    terminal = terminal_refusal(monkeypatch, **records)
    assert terminal.split(": ", 1)[1] == message.split(": ", 1)[1]


# --------------------------------------------------------------- an authored arc on one line

# The pad's outline with its bottom side drawn as an arc whose middle point lies on that side.
FLAT_ARC = [
    {"kind": "circular_arc", "points_um": [[0, 0], [10, 0], [20, 0]]},
    {"kind": "line_segment", "points_um": [[20, 0], [20, 20]]},
    {"kind": "line_segment", "points_um": [[20, 20], [0, 20]]},
    {"kind": "line_segment", "points_um": [[0, 20], [0, 0]]},
]


def test_an_authored_arc_on_one_line_is_refused_at_its_points(monkeypatch):
    """The arc from (0, 0) through (10, 0) to (20, 0) um lies on the pad's bottom side.

    The source stage decides it, on the points as the component writes them, whether or not
    the pad is on a lumped port (before the fifth slice the junction partition reached the
    arc of a terminal and divided by zero).
    """
    outline = {
        "boundary_id": "PAD_OUTER",
        "entity_id": "pad",
        "selector_point_um": (10.0, 10.0),
        "curves": FLAT_ARC,
    }
    message = declared_refusal(monkeypatch, curves=[outline])
    assert message.startswith(
        "chip/PAD_OUTER (Entity 'chip/pad', layer 1/0) curve 0: the three arc points "
        "are collinear, so they define no circle: its start "
    )
    assert message.endswith(
        "lie on one line. Fix the layout: draw the arc through three points that do not "
        "lie on one line, or draw it as a line segment."
    )
    start, middle, end = places(message)
    assert on(start, (0, 0, 0, 0)) and on(middle, (10, 0, 10, 0))
    assert on(end, (20, 0, 20, 0))


# ---------------------------------- a component's arc on one line, in a turned occurrence

# On one line exactly, but in floats their area is 2.2e-16, and 0 in some turned frames.
SLANT_ARC = ((0.0, 0.0), (0.2, 0.6), (2.0, 6.0))
# A pad with the side from (0, 0) to (2, 6) um, and an electrode clear of it.
SLANT_PAD = ((0.0, 0.0), (2.0, 6.0), (8.0, 4.0), (6.0, -2.0))
FAR_ELECTRODE = ((20.0, 0.0), (30.0, 0.0), (30.0, 10.0), (20.0, 10.0))
# A turn by atan(3/4) keeps these layouts on the 1 nm grid; turns by 6 and 20 degrees do not.
ON_GRID_TURN = math.degrees(math.atan2(3, 4))
DEVICE_NETS = {"pad": ("chip/device/pad",), "electrode": ("chip/device/electrode",)}
FIX_ARC = (
    "lie on one line. Fix the layout: draw the arc through three points that do not lie "
    "on one line, or draw it as a line segment."
)


def outline_with_arc(pad, arc=SLANT_ARC):
    """An authored boundary of ``pad`` whose first side is ``arc`` and whose other sides are lines."""
    sides = zip(pad[1:], (*pad[2:], pad[0]), strict=True)
    return {
        "boundary_id": "PAD_OUTER",
        "entity_id": "pad",
        "selector_point_um": tuple(sum(c) / len(pad) for c in zip(*pad, strict=True)),
        "curves": [
            {"kind": "circular_arc", "points_um": arc},
            *({"kind": "line_segment", "points_um": side} for side in sides),
        ],
    }


def pad_device(monkeypatch, pad, *, curves=(), reconstruction=()):
    """A device of ``pad`` on layer 1/0 and an electrode on 2/0, with these boundary records on the pad."""
    import gdsfactory as gf  # noqa: PLC0415 -- optional layout stack, imported only by the tests that need it

    monkeypatch.setattr(
        gf.pdk,
        "_ACTIVE_PDK",
        gf.pdk._ACTIVE_PDK,  # noqa: SLF001 -- preserve the active PDK during test setup
    )
    gf.gpdk.PDK.activate()
    device = gf.Component()
    device.add_polygon(pad, layer=(1, 0))
    device.add_polygon(FAR_ELECTRODE, layer=(2, 0))
    regions = (("pad", 1, pad), ("electrode", 2, FAR_ELECTRODE))
    device.info["component_semantics"] = {
        "schema_version": 2,
        "conductor_regions": [
            {
                "semantic_id": name,
                "level": "metal",
                "gds_layer": [layer, 0],
                "geometry": {
                    "geometry_source": "gds_polygon",
                    "selector_point_um": tuple(
                        sum(c) / len(ring) for c in zip(*ring, strict=True)
                    ),
                },
            }
            for name, layer, ring in regions
        ],
        "boundary_curves": list(curves),
        "boundary_reconstruction": list(reconstruction),
    }
    return device


def turned_plan(device, turn):
    """The plan of a chip that places ``device`` as ``chip/device`` turned by ``turn`` degrees, and the chip and reference."""
    chip, reference = placed(device, turn)
    plan = GeometryPlan(
        chip,
        root_id="chip",
        layer_stack=LAYER_STACK,
        material_records=MATERIALS,
        coupon_padding_um=0.0,
    )
    plan.register_instance("chip/device", reference)
    return plan, chip, reference


def occurrence_refusal(device, turn, nets=None):
    """The refusal of a chip that places ``device`` turned by ``turn`` degrees, from its plan and from ``simulation_drc``."""
    plan, chip, reference = turned_plan(device, turn)
    with pytest.raises(ValueError) as planned:
        plan.prepare()
    with pytest.raises(ValueError) as checked:
        check(EigenmodeSim, chip, instances={"chip/device": reference}, nets=nets)
    assert str(checked.value) == str(planned.value)
    return str(planned.value)


def assert_arc_placed(message, turn):
    """The refusal names the arc of ``chip/device/pad`` where a turn by ``turn`` degrees puts SLANT_ARC."""
    assert message.startswith(
        "chip/device/PAD_OUTER (Entity 'chip/device/pad', layer 1/0) curve 0: the three "
        "arc points are collinear, so they define no circle: its start "
    )
    assert message.endswith(FIX_ARC)
    found = places(message)
    assert len(found) == len(SLANT_ARC)
    for place, point in zip(found, SLANT_ARC, strict=True):
        x, y = turned(point, turn)
        assert on(place, (x, y, x, y))


@pytest.mark.parametrize("turn", [6.0, ON_GRID_TURN], ids=["6-degrees", "37-degrees"])
def test_a_components_arc_on_one_line_is_refused_in_a_turned_occurrence(
    monkeypatch, turn
):
    """The pad's side from (0, 0) through (0.2, 0.6) to (2, 6) um, its device turned by 6 or 36.9 degrees.

    The turn moves each point in floats and the three leave the line, so SGB, which binds
    this pad, no longer found the arc flat (before: at 6 degrees it refused the arc's end
    as off the grid; at 36.9 degrees, where the ends stay on the grid, it built a circle
    some 1e16 um across and failed its own check, ``arc_edges_do_not_cross``).  The source
    stage decides on the points as the component writes them, and places the arc where the
    occurrence puts it.
    """
    device = pad_device(monkeypatch, SLANT_PAD, curves=[outline_with_arc(SLANT_PAD)])
    assert_arc_placed(occurrence_refusal(device, turn), turn)


def test_a_terminals_arc_on_one_line_is_refused_in_a_turned_occurrence(monkeypatch):
    """The same arc on a lumped port's terminal, its device turned by 36.9 degrees.

    The port runs along (4, 3) in the device, so the turn keeps its sheet on the grid and
    the construction binder is reached (before: it read the turned points, found them off
    the line and passed the arc on; SGB then failed ``arc_edges_do_not_cross``).  At 6
    degrees the sheet leaves the grid and the port is refused before any binding.
    """
    device = slanted_terminal(monkeypatch, (4, 3), 1360, declared=True)
    message = occurrence_refusal(device, ON_GRID_TURN, DEVICE_NETS)
    assert_arc_placed(message, ON_GRID_TURN)


def test_an_arc_nearly_on_one_line_is_built_as_before(monkeypatch):
    """A middle point at (0.2, 0.601) um lies 0.3 nm off the chord: a valid arc 5.7 mm in radius.

    The three points are not on one line as the component writes them, so no stage refuses
    the arc; SGB builds it, in a turned occurrence as in a straight one.
    """
    nearly_flat = ((0.0, 0.0), (0.2, 0.601), (2.0, 6.0))
    outline = outline_with_arc(SLANT_PAD, nearly_flat)
    for turn in (0.0, ON_GRID_TURN):
        device = pad_device(monkeypatch, SLANT_PAD, curves=[outline])
        chip, reference = placed(device, turn)
        check(EigenmodeSim, chip, instances={"chip/device": reference})


# ------------------------------ anchors on one line in their cell, in a turned occurrence

# A pad whose top side zigzags: its vertices at x = 0, 3 and 7 um lie on the line y = 0, with a
# peak between each two.  KLayout walks it clockwise from its lower-left corner.
ZIGZAG = (
    (0.0, 0.0),
    (1.0, 1.0),
    (3.0, 0.0),
    (5.0, 1.0),
    (7.0, 0.0),
    (7.0, -3.0),
    (0.0, -3.0),
)
ZIGZAG_DRAWN = (ZIGZAG[-1], *ZIGZAG[:-1])
ON_THE_LINE = (1, 3, 5)


@pytest.mark.parametrize("turn", [6.0, 20.0], ids=["6-degrees", "20-degrees"])
def test_anchors_on_one_line_in_their_cell_are_refused_in_a_turned_occurrence(
    monkeypatch, turn
):
    """A reconstruction of the zigzag anchored at its vertices (0, 0), (3, 0) and (7, 0) um.

    Turned by 6 or 20 degrees and rounded to the 1 nm grid, the three vertices leave the
    line by less than a grid step, so SGB solved a circle through them and replaced the
    zigzag by an arc as flat as a line (before: ``simulation_drc`` passed).  The source
    stage decides on the vertices as their cell draws them, and names each anchor where
    the occurrence puts it.
    """
    record = {
        "boundary_id": "PAD_ARC",
        "entity_id": "pad",
        "selector_point_um": (3.5, -1.5),
        "kind": "circular_arc",
        "segment_indices": [1, 2, 3, 4],
        "anchor_indices": list(ON_THE_LINE),
    }
    device = pad_device(monkeypatch, ZIGZAG, reconstruction=[record])
    (polygon,) = device.get_polygons(by="tuple")[(1, 0)]
    dbu = device.kcl.dbu
    walk = tuple((p.x * dbu, p.y * dbu) for p in polygon.each_point_hull())
    assert walk == ZIGZAG_DRAWN
    message = occurrence_refusal(device, turn)
    assert message.startswith(
        "chip/device/PAD_ARC (Entity 'chip/device/pad', layer 1/0): the three arc points "
        "are collinear, so they define no circle: its anchors are drawn vertex 1 at "
    )
    assert "they lie on one line. Fix the declaration: choose three anchors" in message
    found = places(message)
    assert len(found) == len(ON_THE_LINE)
    for place, index in zip(found, ON_THE_LINE, strict=True):
        x, y = turned(ZIGZAG_DRAWN[index], turn)
        assert on(place, (x, y, x, y))


def zigzag_plan(monkeypatch, anchors, turn):
    """A chip that turns the zigzag's device by ``turn`` degrees, its arc over drawn segments 1 to 4 declared on the plan."""
    plan, _, _ = turned_plan(pad_device(monkeypatch, ZIGZAG), turn)
    plan.set_boundary_reconstruction(
        [
            {
                "boundary_id": "chip/device/PAD_ARC",
                "entity_id": "chip/device/pad",
                "selector_point_um": turned((3.5, -1.5), turn),
                "kind": "circular_arc",
                "segment_indices": [1, 2, 3, 4],
                "anchor_indices": list(anchors),
            }
        ]
    )
    return plan


@pytest.mark.parametrize("turn", [6.0, 20.0], ids=["6-degrees", "20-degrees"])
def test_anchors_on_one_line_in_their_cell_are_refused_on_the_plan_in_a_turned_occurrence(
    monkeypatch, turn
):
    """The zigzag's anchors (0, 0), (3, 0) and (7, 0) um on the plan: refused as unturned (before: SGB built an arc)."""
    messages = []
    for angle in (0.0, turn):
        with pytest.raises(ValueError) as error:
            zigzag_plan(monkeypatch, ON_THE_LINE, angle).prepare()
        messages.append(str(error.value))
    assert messages[1].startswith(
        "chip/device/PAD_ARC (Entity 'chip/device/pad', layer 1/0): the three arc points "
        "are collinear, so they define no circle: its anchors are drawn vertex 1 at "
    )
    unplaced = [re.sub(r"\([^()]*\) um", "", message) for message in messages]
    assert unplaced[0] == unplaced[1]
    for place, index in zip(places(messages[1]), ON_THE_LINE, strict=True):
        x, y = turned(ZIGZAG_DRAWN[index], turn)
        assert on(place, (x, y, x, y))


def test_anchors_that_define_a_circle_on_the_plan_build_in_a_turned_occurrence(
    monkeypatch, tmp_path
):
    """The zigzag's anchors (0, 0), (1, 1) and (7, 0) um on the plan, turned by 20 degrees: SGB builds the arc."""
    snapshot = zigzag_plan(monkeypatch, (1, 2, 5), 20.0).prepare()
    sim = configured(EigenmodeSim, snapshot, None)
    sim.set_output_dir(tmp_path / "run")

    def meshing_reached(*_args, **_kwargs):
        raise MeshingReached

    monkeypatch.setattr(build_module, "mesh_sgb", meshing_reached)
    with pytest.raises(MeshingReached):
        sim.mesh()


# --------------------------------- an arc declared on the plan, in assembly coordinates


def along(direction, s, t):
    """The grid point s v + t w in micrometres, for v = ``direction`` and w = v turned a quarter."""
    p, q = direction
    return ((s * p - t * q) / 1000, (s * q + t * p) / 1000)


def slanted_terminal(monkeypatch, direction, sheet_start, *, declared):
    """A pad with the side from (0, 0) to (2, 6) um on a lumped port along ``direction``.

    In grid steps of v = ``direction`` along the port and of w across it, the pad ends at
    s = 1500 with a side square to the port, the electrode runs from s = 1600 to 2600, both
    across t = -150..150, and the sheet runs from s = ``sheet_start`` to 1700, t = -20..20.
    With ``declared`` the component declares that side as SLANT_ARC.
    """

    def box(s0, s1, t0, t1):
        corners = ((s0, t0), (s1, t0), (s1, t1), (s0, t1))
        return [along(direction, s, t) for s, t in corners]

    component = device_with_port(
        monkeypatch,
        pad=slanted_pad(direction),
        electrode=box(1600, 2600, -150, 150),
        sheet=box(sheet_start, 1700, -20, 20),
        port_centre=along(direction, (sheet_start + 1700) / 2, 0),
        port_width_um=40 * math.hypot(*direction) / 1000,
        direction=direction,
    )
    if declared:
        semantics = component.info["component_semantics"]
        semantics["boundary_curves"] = [outline_with_arc(slanted_pad(direction))]
        component.info["component_semantics"] = semantics
    return component


def slanted_pad(direction):
    """The pad of ``slanted_terminal``: its side from (0, 0) to (2, 6) um, and the side that meets the port."""
    return [
        SLANT_ARC[0],
        SLANT_ARC[2],
        along(direction, 1500, 150),
        along(direction, 1500, -150),
    ]


@pytest.mark.parametrize(
    ("direction", "sheet_start"),
    [((19, 2), 1312), ((4, 3), 1360)],
    ids=["6-degrees", "37-degrees"],
)
def test_an_arc_declared_on_the_plan_is_refused_by_the_binding_on_a_slanted_terminal(
    monkeypatch, direction, sheet_start
):
    """The arc from (0, 0) through (0.2, 0.6) to (2, 6) um, declared on the plan, on a port along (19, 2) or (4, 3).

    No occurrence transform touches a plan's own declaration, so the binding stages decide
    it; a terminal's is the construction binder, in ``mesh()``'s preparation.  Each sheet
    starts where the junction's local frame, 6.0 or 36.9 degrees, turns the three points
    onto one line in floats, where the partition once divided by zero.  The binder finds
    them on one line exactly, as SGB does, and refuses the arc at its points.
    """
    component = slanted_terminal(monkeypatch, direction, sheet_start, declared=False)
    plan = GeometryPlan(
        component,
        root_id="chip",
        layer_stack=NO_VACUUM_STACK,
        material_records=MATERIALS,
        coupon_padding_um=0.0,
    )
    plan.set_nets({"pad": ("chip/pad",), "electrode": ("chip/electrode",)})
    outline = outline_with_arc(slanted_pad(direction))
    plan.set_boundary_curves([{**outline, "entity_id": "chip/pad"}])
    sim = EigenmodeSim()
    sim.set_plan(plan.prepare())
    sim.add_port("chip/P1", layer="metal", layout_sheet=True, inductance=1e-9)
    with pytest.raises(ValueError) as error:
        prepared_port(sim)
    message = str(error.value)
    assert message.startswith(
        "PAD_OUTER (Entity 'chip/pad', layer 1/0) curve 0: the three arc points are "
        "collinear, so they define no circle: its start "
    )
    assert message.endswith(FIX_ARC)
    start, middle, end = places(message)
    assert on(start, (0, 0, 0, 0)) and on(middle, (0.2, 0.6, 0.2, 0.6))
    assert on(end, (2, 6, 2, 6))


# ----------------------------------- an input built by hand: the binding stages decide


def stage_refusals(tmp_path, **records):
    """SGB's refusal of records put by hand on the pad of ``gds_pad_in_ground``, checked to be the construction binder's.

    No source stage reads records attached to an effective input, so each binding stage
    decides them, on the points it solves the circle on.  The pad is drawn (40, 40),
    (60, 40), (60, 60), (40, 60) um.
    """
    pad = with_boundaries(gds_pad_in_ground(tmp_path, with_port=False), **records)
    build = pad.build_input
    with pytest.raises(InputError) as sgb:
        build_prism_catalogue(build, pad.representation)
    with pytest.raises(ValueError) as binder:
        source_boundaries(build, source_rings(build))
    assert str(binder.value) == str(sgb.value)
    owned = next(e.polygon_ids for e in build.entities if e.semantic_id == "pad")
    layer = next(p.layer for p in build.polygons if p.polygon_id in owned)
    return str(sgb.value), layer


def test_an_arc_on_one_line_put_on_an_input_by_hand_is_refused_by_each_binding_stage(
    tmp_path,
):
    """The pad's bottom side as an arc from (40, 40) through (50, 40) to (60, 40) um."""
    corners = ((40.0, 40.0), (60.0, 40.0), (60.0, 60.0), (40.0, 60.0))
    sides = zip(corners[1:], (*corners[2:], corners[0]), strict=True)
    outline = BoundaryCurveChainSpec(
        "pad_outline",
        "pad",
        (50.0, 50.0),
        (
            SourceCurveSpec("circular_arc", (corners[0], (50.0, 40.0), corners[1])),
            *(SourceCurveSpec("line_segment", side) for side in sides),
        ),
    )
    message, layer = stage_refusals(tmp_path, boundary_curves=(outline,))
    assert message.startswith(
        f"pad_outline (Entity 'pad', layer {layer}) curve 0: the three arc points are "
        "collinear, so they define no circle: its start "
    )
    assert message.endswith(FIX_ARC)
    start, middle, end = places(message)
    assert on(start, (40, 40, 40, 40)) and on(middle, (50, 40, 50, 40))
    assert on(end, (60, 40, 60, 40))


def test_anchors_with_no_circle_put_on_an_input_by_hand_are_refused_by_each_binding_stage(
    tmp_path,
):
    """Anchors 0, 0, 2 of the pad's first two drawn sides, from (40, 40) to (60, 60) um."""
    record = GdsBoundaryReconstructionSpec(
        "pad_corner",
        "pad",
        (50.0, 50.0),
        "circular_arc",
        segment_indices=(0, 1),
        anchor_indices=(0, 0, 2),
    )
    message, layer = stage_refusals(tmp_path, boundary_reconstruction=(record,))
    assert message.startswith(
        f"pad_corner (Entity 'pad', layer {layer}): the three arc points are collinear, "
        "so they define no circle: its anchors are drawn vertex 0 at "
    )
    assert "two of them are one vertex. Fix the declaration" in message
    first, again, last = places(message)
    assert on(first, (40, 40, 40, 40)) and on(again, (40, 40, 40, 40))
    assert on(last, (60, 60, 60, 60))
