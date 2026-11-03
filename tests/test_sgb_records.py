"""Grid conversion, rings, footprints, carriers and circles on the lattice."""

from __future__ import annotations

import math
from fractions import Fraction

import pytest

from scgsim.geometry.sgb.errors import InputError
from scgsim.geometry.sgb.prisms.arcs import arc_through_points, full_circle
from scgsim.geometry.sgb.records.carriers import Lineage, base_carrier_id
from scgsim.geometry.sgb.records.grid import Grid, exact_z
from scgsim.geometry.sgb.records.prisms import (
    Footprint,
    PortIdentity,
    PrismRecord,
    oriented_ring,
)

LINEAGE = Lineage("polygon_edge", "e", 0, 0, 3)


@pytest.mark.parametrize(
    ("value", "units"), [(40.001, 40001), (0.1 + 0.2, 300), (-12.345, -12345), (0.0, 0)]
)
def test_declared_lengths_land_exactly_on_the_lattice(value, units):
    """The decimal text of a micrometre value is converted, never its binary expansion."""
    assert Grid(0.001).lattice_units(value) == units


def test_a_length_off_the_lattice_is_reported_not_rounded():
    """Fail closed: only float representation noise is tolerated."""
    assert Grid(0.001).lattice_units(0.0005) is None
    assert Grid(0.01).lattice_units(0.004) is None


def test_the_grid_must_be_a_positive_length():
    """A zero or negative grid is a defect."""
    with pytest.raises(ValueError, match="positive length"):
        Grid(0.0)


@pytest.mark.parametrize(
    ("value", "exact"),
    [
        (0.1, Fraction(1, 10)),
        (0.30000000000000004, Fraction(3, 10)),
        ("1/5", Fraction(1, 5)),
        (3, Fraction(3)),
    ],
)
def test_z_values_are_exact_decimals_and_float_noise_never_makes_a_level(value, exact):
    """Binary noise below a micro-nanometre is representation hygiene, not a Z threshold."""
    assert exact_z(value) == exact


def test_distinct_declared_z_values_stay_distinct_even_below_the_grid():
    """0.4 nm and 0 are two levels."""
    assert exact_z(0.0004) != exact_z(0.0)
    assert exact_z(0.0004) == Fraction(1, 2500)


def test_oriented_ring_reverses_edge_numbers_with_the_ring():
    """Lineage keeps pointing at the source edge after the ring is reversed to CCW."""
    clockwise = [(0, 10), (10, 10), (10, 0), (0, 0)]
    ring, numbers = oriented_ring(clockwise, ccw=True)
    assert ring == ((0, 0), (10, 0), (10, 10), (0, 10))
    # (0,0)->(10,0) is source edge 2, (10,0)->(10,10) is 1, (10,10)->(0,10) is 0, (0,10)->(0,0) is 3
    assert numbers == (2, 1, 0, 3)


def test_oriented_ring_drops_the_closing_point_and_repeated_points():
    """The edge that follows a dropped duplicate keeps its source number."""
    ring, numbers = oriented_ring(
        [(0, 0), (10, 0), (10, 0), (10, 10), (0, 10), (0, 0)], ccw=True
    )
    assert ring == ((0, 0), (10, 0), (10, 10), (0, 10))
    assert numbers == (0, 2, 3, 4)


def test_a_ring_needs_three_distinct_points():
    """Degenerate rings fail closed."""
    with pytest.raises(ValueError, match="three distinct points"):
        oriented_ring([(0, 0), (1, 1), (0, 0)], ccw=True)


def test_footprint_area_subtracts_holes():
    """Outer minus holes, in twice-area units."""
    footprint = Footprint.from_rings(
        [(0, 0), (10, 0), (10, 10), (0, 10)], [[(2, 2), (4, 2), (4, 4), (2, 4)]]
    )
    assert footprint.twice_area() == 2 * (100 - 4)


def test_prism_invariants_fail_closed():
    """A port prism is a sheet carrying an identity; a conductor needs a Net."""
    footprint = Footprint.from_rings([(0, 0), (10, 0), (10, 10)])
    with pytest.raises(ValueError, match="z1 < z0"):
        PrismRecord("x", footprint, Fraction(1), Fraction(0), "v", "vacuum")
    with pytest.raises(ValueError, match="port identity"):
        PrismRecord("x", footprint, Fraction(0), Fraction(0), "p", "port")
    identity = PortIdentity("x", "X", 1, (1.0, 0.0, 0.0), (), "m", "9/0")
    with pytest.raises(ValueError, match="a port prism is a sheet"):
        PrismRecord(
            "x", footprint, Fraction(0), Fraction(1), "p", "port", port=identity
        )


def test_witness_piece_ids_name_their_arc():
    """A witness chord id shortens back to the arc carrier id."""
    assert base_carrier_id("pad/hole/q0#12") == "pad/hole/q0"
    assert base_carrier_id("pad/polygon0/ring0/edge3") == "pad/polygon0/ring0/edge3"


def test_a_three_point_arc_is_cut_into_pieces_of_at_most_a_quarter_turn():
    """A half circle gives two quarter pieces with exact end points and one shared lattice cut."""
    pieces = arc_through_points(
        "a", (1000, 0), (Fraction(0), Fraction(1000)), (-1000, 0), LINEAGE, "test arc"
    )
    assert len(pieces) == 2
    assert pieces[0].start == (1000, 0) and pieces[1].end == (-1000, 0)
    assert pieces[0].end == pieces[1].start == (Fraction(0), Fraction(1000))
    assert all(piece.ccw and piece.root_id == "a" for piece in pieces)
    assert all(abs(piece.sweep()[1] - math.pi / 2) < 1e-9 for piece in pieces)


def test_a_full_circle_through_three_anchors_is_four_pieces_from_the_first_anchor():
    """The pieces chain end to start and close on the first anchor."""
    anchors = ((3000, 0), (0, 3000), (-3000, 0))
    pieces = full_circle("c", anchors, True, LINEAGE, "test circle")
    assert len(pieces) == 4
    assert pieces[0].start == pieces[-1].end == (3000, 0)
    assert all(a.end == b.start for a, b in zip(pieces, pieces[1:], strict=False))
    clockwise = full_circle("c", anchors, False, LINEAGE, "test circle")
    assert all(not piece.ccw for piece in clockwise)


def test_the_smallest_circle_on_the_lattice_still_has_four_distinct_pieces():
    """A one-unit circle (four lattice points) is the smallest arc the grid can carry."""
    pieces = full_circle(
        "tiny", ((1, 0), (0, 1), (-1, 0)), True, LINEAGE, "tiny circle"
    )
    assert [piece.end for piece in pieces] == [(0, 1), (-1, 0), (0, -1), (1, 0)]


def test_three_collinear_points_define_no_circle():
    """Fail closed with the arc's name."""
    with pytest.raises(
        InputError, match="test arc: the three arc points are collinear"
    ):
        arc_through_points(
            "a", (0, 0), (Fraction(1), Fraction(0)), (2, 0), LINEAGE, "test arc"
        )
