"""SGB's boundary binding refusals name the record, its Entity and layer, the place in um and who acts.

The fifth slice of the located layout refusals (``test_located_binding_refusals.py`` holds the
fourth).  SGB (``sgb.prisms.curves``) binds the boundary records of a conductor that is not a
lumped port's terminal.  On a terminal the junction partition binds them first, with the
construction binder, and gives SGB's refusal for the same fault, word for word.  Each test
declares a record on the pad of ``two_pads``, runs ``simulation_drc`` with the pad off a lumped
port and, where both stages refuse it, on one, and reads every coordinate back out of the
refusal: each must lie on the ring, segment, vertex or arc point at fault.
"""

from __future__ import annotations

import math

import pytest
from test_located_refusals import on, places
from test_located_source_and_sgb_refusals import pads_with_selectors
from test_lumped_ports_at_any_angle import device_with_port
from test_simulation_drc import check

from scgsim.geometry.sgb.errors import InputError
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
    """A drawn vertex given twice leaves three points on one line, so no circle runs through them."""
    message = one_refusal(monkeypatch, reconstruction=[record])
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

    With the pad on a lumped port the junction partition reached the arc before SGB and
    divided by zero (before: ``ZeroDivisionError: float division by zero``); it now gives
    SGB's refusal.
    """
    outline = {
        "boundary_id": "PAD_OUTER",
        "entity_id": "pad",
        "selector_point_um": (10.0, 10.0),
        "curves": FLAT_ARC,
    }
    message = one_refusal(monkeypatch, curves=[outline])
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


# ------------------------------------------ an arc on one line exactly, on a slanted terminal

# On one line exactly, but in floats their area is 2.2e-16, and 0 in some turned frames.
SLANT_ARC = ((0.0, 0.0), (0.2, 0.6), (2.0, 6.0))


def along(direction, s, t):
    """The grid point s v + t w in micrometres, for v = ``direction`` and w = v turned a quarter."""
    p, q = direction
    return ((s * p - t * q) / 1000, (s * q + t * p) / 1000)


def slanted_terminal(monkeypatch, direction, sheet_start):
    """A pad whose side from (0, 0) to (2, 6) um is authored as SLANT_ARC, on a lumped port along ``direction``.

    In grid steps of v = ``direction`` along the port and of w across it, the pad ends at
    s = 1500 with a side square to the port, the electrode runs from s = 1600 to 2600, both
    across t = -150..150, and the sheet runs from s = ``sheet_start`` to 1700, t = -20..20.
    """

    def box(s0, s1, t0, t1):
        corners = ((s0, t0), (s1, t0), (s1, t1), (s0, t1))
        return [along(direction, s, t) for s, t in corners]

    pad = [
        SLANT_ARC[0],
        SLANT_ARC[2],
        along(direction, 1500, 150),
        along(direction, 1500, -150),
    ]
    component = device_with_port(
        monkeypatch,
        pad=pad,
        electrode=box(1600, 2600, -150, 150),
        sheet=box(sheet_start, 1700, -20, 20),
        port_centre=along(direction, (sheet_start + 1700) / 2, 0),
        port_width_um=40 * math.hypot(*direction) / 1000,
        direction=direction,
    )
    sides = zip(pad[1:], (*pad[2:], pad[0]), strict=True)
    outline = {
        "boundary_id": "PAD_OUTER",
        "entity_id": "pad",
        "selector_point_um": tuple(sum(c) / len(pad) for c in zip(*pad, strict=True)),
        "curves": [
            {"kind": "circular_arc", "points_um": SLANT_ARC},
            *({"kind": "line_segment", "points_um": side} for side in sides),
        ],
    }
    semantics = component.info["component_semantics"]
    semantics["boundary_curves"] = [outline]
    component.info["component_semantics"] = semantics
    return component


@pytest.mark.parametrize(
    ("direction", "sheet_start"),
    [((19, 2), 1312), ((4, 3), 1360)],
    ids=["6-degrees", "37-degrees"],
)
def test_an_arc_on_one_line_only_exactly_is_refused_on_a_slanted_terminal(
    monkeypatch, direction, sheet_start
):
    """The arc from (0, 0) through (0.2, 0.6) to (2, 6) um, on a port along (19, 2) or (4, 3).

    Each sheet starts where the junction's local frame, 6.0 or 36.9 degrees, turns the three
    points onto one line in floats, so the partition divided by zero (before:
    ``ZeroDivisionError: float division by zero`` from ``simulation_drc``).  The binding now
    finds them on one line exactly, as SGB does, and refuses the arc at its points.
    """
    with pytest.raises(ValueError) as error:
        check(EigenmodeSim, slanted_terminal(monkeypatch, direction, sheet_start))
    message = str(error.value)
    assert message.startswith(
        "chip/PAD_OUTER (Entity 'chip/pad', layer 1/0) curve 0: the three arc points "
        "are collinear, so they define no circle: its start "
    )
    start, middle, end = places(message)
    assert on(start, (0, 0, 0, 0)) and on(middle, (0.2, 0.6, 0.2, 0.6))
    assert on(end, (2, 6, 2, 6))
