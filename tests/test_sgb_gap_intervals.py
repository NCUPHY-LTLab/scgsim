"""Inset bands are measured from the part of a source edge that faces a gap, not from the whole source edge.

A finite conductor's top face is measured from the gap edges of its attachment plane.  A source edge that
faces a gap along part of its length (a pad that hangs over the substrate edge or over a trench, a trace
abutting the side of a pad) is an origin along that part only: the rest of it has no gap before it.  A round pad
whose gap arc ends on the die edge is refused: SCGSim continues a conductor cut by the die edge only along a
straight edge.
"""

from __future__ import annotations

import math
from fractions import Fraction

import pytest
from sgb_scenes import GRID, LINEAGE, conductor, footprint, rect_um

from scgsim.geometry.sgb.cutfamilies.gap_intervals import (
    ArcSpan,
    LineSpan,
    Piece,
    split_at_gaps,
)
from scgsim.geometry.sgb.cutfamilies.requests import ConductorSurface, InsetRequest
from scgsim.geometry.sgb.emission.measures import planned_face_area_um2
from scgsim.geometry.sgb.errors import InputError
from scgsim.geometry.sgb.model import ConformalModel, build_model_from_catalogue
from scgsim.geometry.sgb.plane import point_in_ring
from scgsim.geometry.sgb.records.carriers import (
    LineEdge,
    RingPrimitive,
    circle_arcs,
)
from scgsim.geometry.sgb.records.grid import exact_z
from scgsim.geometry.sgb.records.prisms import (
    Footprint,
    PrismCatalogue,
    PrismRecord,
)
from scgsim.geometry.sgb.records.support import SupportRegion

TOP = ConductorSurface("pad", "top")
ROUTES = ("B", "C")


def chip(substrate_x1: float) -> list[PrismRecord]:
    """A substrate whose right edge is at ``substrate_x1`` um, under vacuum that reaches further right."""
    return [
        PrismRecord(
            "substrate",
            footprint(rect_um(-12, -12, substrate_x1, 12)),
            exact_z(-2),
            exact_z(0),
            "Si",
            "dielectric",
        ),
        PrismRecord(
            "vacuum",
            footprint(rect_um(-12, -12, 14, 12)),
            exact_z(-2),
            exact_z(4),
            "air",
            "vacuum",
        ),
    ]


def ends(prim: RingPrimitive) -> tuple[tuple[int, int], tuple[int, int]]:
    """The lattice end points of a ring primitive."""
    if isinstance(prim, LineEdge):
        return prim.p, prim.q
    (sx, sy), (ex, ey) = prim.start, prim.end
    return (int(sx), int(sy)), (int(ex), int(ey))


def build(route: str, prisms: list[PrismRecord]) -> ConformalModel:
    """A model with one inset request on the top of the pad, validated."""
    request = InsetRequest("pad_top", TOP, (0.5,))
    model = build_model_from_catalogue(
        PrismCatalogue(GRID, tuple(prisms)), route, insets=(request,)
    )
    assert not model.report.failures
    return model


def support_of(model: ConformalModel) -> SupportRegion:
    """The support region of the model's only inset family."""
    support = model.labelled.catalogue.families[0].support
    assert support is not None
    return support


def band_at(
    model: ConformalModel, x_um: float, y_um: float
) -> tuple[tuple[str, int], ...]:
    """The band labels of the face of the metal level that covers a point (um)."""
    layer = next(
        layer for layer in model.labelled.level_layers if layer.z == exact_z("0.2")
    )
    point = (round(x_um * 1000), round(y_um * 1000), 1)
    (face,) = [
        fi
        for fi in range(1, len(layer.arr.faces))
        if sum(
            point_in_ring(layer.arr.cycle_points(c), point)
            for c in layer.arr.face_cycles(fi)
        )
        % 2
    ]
    return layer.labels[face].bands


def band_areas_um2(model: ConformalModel) -> dict[int, float]:
    """Emitted area of every band group of the pad's top face, by band index (read from the faces, not the names)."""
    areas: dict[int, float] = {}
    for group in model.routed.groups.values():
        faces = (
            [model.topology.faces[fid] for fid in group.ids] if group.dim == 2 else []
        )
        tops = [
            f
            for f in faces
            if f.bands and f.neg_domain is not None and f.neg_domain.name == "pad"
        ]
        if tops and len(tops) == len(faces):
            (index,) = {f.bands[0][1] for f in tops}
            areas[index] = sum(planned_face_area_um2(model.topology, f) for f in tops)
    return areas


# --------------------------------------------------------------------------- a pad hanging over the substrate edge


@pytest.mark.parametrize("route", ROUTES)
def test_a_source_edge_that_faces_a_gap_along_part_of_its_length_is_an_origin_there_only(
    route,
):
    """The 8 um pad crosses the substrate edge x = 0: only the x <= 0 halves of y = +-4 face a gap."""
    model = build(
        route,
        [*chip(0), conductor("pad", rect_um(-4, -4, 4, 4), 0, "0.2", "N")],
    )
    (ring,) = support_of(model).rings
    (mask,) = support_of(model).origin_mask
    assert [(*ends(p), origin) for p, origin in zip(ring, mask, strict=True)] == [
        ((-4000, -4000), (0, -4000), True),
        ((0, -4000), (4000, -4000), False),
        ((4000, -4000), (4000, 4000), False),
        ((4000, 4000), (0, 4000), False),
        ((0, 4000), (-4000, 4000), True),
        ((-4000, 4000), (-4000, -4000), True),
    ]
    assert band_at(model, 3.9, -3.9) == (("pad_top", 1),)  # 3.9 um from every gap edge
    assert band_at(model, -3.9, -3.9) == (("pad_top", 0),)
    # Left strip, two half-edge strips and two quarter discs at the ends x = 0.
    near_gap = 0.5 * 8 + 2 * 3.5 * 0.5 + 2 * math.pi * 0.5**2 / 4
    areas = band_areas_um2(model)
    assert areas[0] == pytest.approx(near_gap, rel=1e-6)
    assert areas[1] == pytest.approx(64 - near_gap, rel=1e-6)


def chip_with_a_trench(rim_x: float) -> list[PrismRecord]:
    """The substrate reaching x = 12 um with a vacuum trench from its rim x = ``rim_x`` to x = 8 um, under vacuum.

    The die encloses the trench, so the trench's rim is no die edge: a gap edge that ends there keeps the half
    circle of its bands.
    """
    substrate = footprint(rect_um(-12, -12, 12, 12), [rect_um(rim_x, -8, 8, 8)])
    return [
        PrismRecord(
            "substrate", substrate, exact_z(-2), exact_z(0), "Si", "dielectric"
        ),
        *chip(12)[1:],
    ]


def round_pad() -> PrismRecord:
    """A pad of radius 4 um about the origin, drawn as one exact circle (four quarter arcs)."""
    outline, carried = Footprint.from_primitives(
        circle_arcs("pad", (0, 0), 4000, True, LINEAGE)
    )
    return PrismRecord(
        "pad",
        outline,
        exact_z(0),
        exact_z("0.2"),
        "Al",
        "conductor",
        net_id="N",
        arcs=carried,
    )


@pytest.mark.parametrize("route", ROUTES)
def test_a_round_pad_hanging_over_a_trench_is_an_origin_along_the_gap_arc_only(
    route,
):
    """The trench's rim x = -1.5 crosses the circle mid-arc; the arc is cut at the lattice point of the crossing."""
    model = build(route, [*chip_with_a_trench(-1.5), round_pad()])
    (ring,) = support_of(model).rings
    (mask,) = support_of(model).origin_mask
    gap = [prim for prim, origin in zip(ring, mask, strict=True) if origin]
    free = [prim for prim, origin in zip(ring, mask, strict=True) if not origin]
    assert all(
        max(x for x, _ in ends(p)) <= -1500 for p in gap
    )  # stops at the substrate edge
    assert (
        len(free) == 4
    )  # the two left quarter arcs cut at the crossings, the two right-hand ones whole
    assert band_at(model, -0.5, 3.8) == (
        ("pad_top", 1),
    )  # 1.004 um from the end of the gap arc
    assert band_at(model, -3.7, 0.0) == (("pad_top", 0),)


@pytest.mark.parametrize("route", ROUTES)
def test_a_round_pad_hanging_over_the_die_edge_is_refused(route):
    """The die edge x = -1.5 cuts the same circle mid-arc: the gap arc ends on the die edge, which SCGSim does not model.

    A conductor cut by the die edge is continued straight past it, which an arc cannot be (Issue #138).  The
    refusal names the pad and the first cut on its ring, where the arc crosses the die edge below the centre.
    """
    request = InsetRequest("pad_top", TOP, (0.5,))
    with pytest.raises(InputError) as raised:
        build_model_from_catalogue(
            PrismCatalogue(GRID, (*chip(-1.5), round_pad())), route, insets=(request,)
        )
    assert (
        "the die edge cuts 'pad' at (-1.5000, -3.7080) um, where its gap edge is an exact arc"
        in str(raised.value)
    )


# --------------------------------------------------------------------------- a trace abutting a pad


@pytest.mark.parametrize("route", ROUTES)
def test_the_contact_interval_of_a_pad_edge_is_not_an_origin_but_the_edge_either_side_is(
    route,
):
    """A 2 um trace of the same net abuts the right edge of the pad; the gap lies above and below the contact."""
    model = build(
        route,
        [
            *chip(12),
            conductor("pad", rect_um(-4, -4, 4, 4), 0, "0.2", "N"),
            conductor("trace", rect_um(4, -1, 12, 1), 0, "0.2", "N"),
        ],
    )
    (ring,) = support_of(model).rings
    (mask,) = support_of(model).origin_mask
    right_edge = [
        (*ends(p), origin)
        for p, origin in zip(ring, mask, strict=True)
        if all(x == 4000 for x, _ in ends(p))
    ]
    assert right_edge == [
        ((4000, -4000), (4000, -1000), True),
        ((4000, -1000), (4000, 1000), False),
        ((4000, 1000), (4000, 4000), True),
    ]
    # 0.1 um from the contact interval, but 1.005 um from the nearest gap edge.
    assert band_at(model, 3.9, 0) == (("pad_top", 1),)
    assert band_at(model, 3.9, 2.5) == (("pad_top", 0),)


# --------------------------------------------------------------------------- cutting an edge at gaps


def test_an_edge_with_no_gap_along_it_is_one_piece_that_faces_none():
    """A gap elsewhere on the line, or none at all, cuts nothing."""
    span = LineSpan((0, 0), (10, 0))
    assert split_at_gaps(span, []) == [Piece((0, 0), (10, 0), False)]
    assert split_at_gaps(span, [LineSpan((20, 0), (30, 0))]) == [
        Piece((0, 0), (10, 0), False)
    ]


def test_gaps_that_touch_are_one_run_and_gaps_in_the_other_sense_are_ignored():
    """Overlapping or touching gap edges join; a gap edge travelled the other way is no gap of this face."""
    span = LineSpan((0, 0), (10, 0))
    gaps = [
        LineSpan((6, 0), (9, 0)),
        LineSpan((-3, 0), (2, 0)),
        LineSpan((2, 0), (4, 0)),
        LineSpan((10, 0), (0, 0)),
    ]
    assert split_at_gaps(span, gaps) == [
        Piece((0, 0), (4, 0), True),
        Piece((4, 0), (6, 0), False),
        Piece((6, 0), (9, 0), True),
        Piece((9, 0), (10, 0), False),
    ]


def test_a_gap_edge_that_covers_the_whole_edge_and_more_makes_it_one_piece_facing_a_gap():
    """The gap edge reaching past both ends cuts nothing."""
    span = LineSpan((0, 0), (10, 0))
    assert split_at_gaps(span, [LineSpan((-5, 0), (15, 0))]) == [
        Piece((0, 0), (10, 0), True)
    ]


def test_an_arc_is_cut_at_a_gap_end_that_lies_on_its_angular_span():
    """Arcs are ordered by angle about the centre; a gap arc travelled the other way is ignored."""
    centre = (Fraction(0), Fraction(0))
    span = ArcSpan(centre, True, (10, 0), (0, 10))
    beyond = ArcSpan(centre, True, (0, 10), (-10, 0))
    across = ArcSpan(centre, True, (7, 7), (-10, 0))
    opposite = ArcSpan(centre, False, (7, 7), (10, 0))
    assert split_at_gaps(span, [beyond]) == [Piece((10, 0), (0, 10), False)]
    assert split_at_gaps(span, [across]) == [
        Piece((10, 0), (7, 7), False),
        Piece((7, 7), (0, 10), True),
    ]
    assert split_at_gaps(span, [opposite]) == [Piece((10, 0), (0, 10), False)]
