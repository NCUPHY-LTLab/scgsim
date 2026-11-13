"""Grid conversion, rings, footprints, carriers and circles on the lattice."""

from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal
from fractions import Fraction

import pytest

from scgsim.geometry.sgb.errors import InputError
from scgsim.geometry.sgb.prisms.arcs import arc_through_points, full_circle
from scgsim.geometry.sgb.exact import circle_math
from scgsim.geometry.sgb.exact.carriers import CarrierRole, Lineage, circle_arcs
from scgsim.geometry.sgb.exact.grid import Grid, exact_z
from scgsim.geometry.sgb.exact.prisms import (
    Footprint,
    PortIdentity,
    PrismRecord,
    oriented_ring,
)

LINEAGE = Lineage("polygon_edge", "e", 0, 0, 3)
GRID = Grid(0.001)


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


def test_witness_pieces_name_their_arc_in_fields():
    """A witness chord states its arc and number as fields, and its whole carrier is the arc's."""
    (arc, *_) = circle_arcs(
        "pad/hole", (0, 0), 1000, True, Lineage("polygon_edge", "pad")
    )
    pieces = arc.pieces()
    assert [p.id for p in pieces[:2]] == ["pad/hole/q0#0", "pad/hole/q0#1"]
    assert {(p.curve, p.role) for p in pieces} == {("pad/hole/q0", CarrierRole.CURVE)}
    assert [p.piece for p in pieces] == list(range(len(pieces)))
    assert pieces[3].whole().id == "pad/hole/q0" and pieces[3].whole().piece is None
    slab = arc.pieces(CarrierRole.SLAB_EDGE)[0]
    assert (
        slab.role is CarrierRole.SLAB_EDGE
        and slab.whole().role is CarrierRole.SLAB_EDGE
    )


def test_an_entity_id_with_a_hash_keeps_its_whole_id():
    """A ``#`` in an entity id is part of the id: a straight carrier is a whole curve whatever its id holds."""
    footprint = Footprint.from_rings(((0, 0), (10, 0), (10, 10), (0, 10)))
    prism = PrismRecord(
        "pad#1", footprint, Fraction(0), Fraction(0), "PEC", "conductor", net_id="n"
    )
    first = prism.carriers()[0]
    assert first.edge_carrier().curve_id == first.id == "pad#1/polygon0/ring0/edge0"


def test_a_three_point_arc_is_cut_into_pieces_of_at_most_a_quarter_turn():
    """A half circle gives two quarter pieces with exact end points and one shared lattice cut."""
    pieces = arc_through_points(
        "a",
        (1000, 0),
        (Fraction(0), Fraction(1000)),
        (-1000, 0),
        LINEAGE,
        "test arc",
        grid=GRID,
    )
    assert len(pieces) == 2
    assert pieces[0].start == (1000, 0) and pieces[1].end == (-1000, 0)
    assert pieces[0].end == pieces[1].start == (Fraction(0), Fraction(1000))
    assert all(piece.ccw and piece.root_id == "a" for piece in pieces)
    assert all(abs(float(piece.sweep()[1]) - math.pi / 2) < 1e-9 for piece in pieces)


def test_a_full_circle_through_three_anchors_is_four_pieces_from_the_first_anchor():
    """The pieces chain end to start and close on the first anchor."""
    anchors = ((3000, 0), (0, 3000), (-3000, 0))
    pieces = full_circle("c", anchors, True, LINEAGE, "test circle", grid=GRID)
    assert len(pieces) == 4
    assert pieces[0].start == pieces[-1].end == (3000, 0)
    assert all(a.end == b.start for a, b in zip(pieces, pieces[1:], strict=False))
    clockwise = full_circle("c", anchors, False, LINEAGE, "test circle", grid=GRID)
    assert all(not piece.ccw for piece in clockwise)


def test_the_smallest_circle_on_the_lattice_still_has_four_distinct_pieces():
    """A one-unit circle (four lattice points) is the smallest arc the grid can carry."""
    pieces = full_circle(
        "tiny", ((1, 0), (0, 1), (-1, 0)), True, LINEAGE, "tiny circle", grid=GRID
    )
    assert [piece.end for piece in pieces] == [(0, 1), (-1, 0), (0, -1), (1, 0)]


def test_three_collinear_points_define_no_circle():
    """Fail closed with the arc's name."""
    with pytest.raises(
        InputError, match="test arc: the three arc points are collinear"
    ):
        arc_through_points(
            "a",
            (0, 0),
            (Fraction(1), Fraction(0)),
            (2, 0),
            LINEAGE,
            "test arc",
            grid=GRID,
        )


def test_witness_points_are_pinned_so_ids_cannot_drift_with_the_platform_math_library():
    """Golden lattice points of a few arcs; the witness is computed in fixed-precision decimals."""
    small = circle_arcs("c", (3, -7), 50, True, LINEAGE)[0]
    assert small.witness_points() == [
        (53, -7), (52, 3), (49, 12), (45, 21), (38, 28), (31, 35), (22, 39), (13, 42), (3, 43),
    ]  # fmt: skip
    clockwise = circle_arcs("c", (3, -7), 800, False, LINEAGE)[1].witness_points()
    assert (len(clockwise), clockwise[:3], clockwise[-2:]) == (
        33,
        [(3, -807), (-36, -806), (-75, -803)],
        [(-796, -46), (-797, -7)],
    )
    assert _digest(clockwise) == "5e0bf40185e05648"
    large = circle_arcs("c", (3, -7), 8000, False, LINEAGE)[1].witness_points()
    assert (len(large), _digest(large)) == (101, "b5cc7ea9177cd68c")
    pieces = arc_through_points(
        "a",
        (1000, 0),
        (Fraction(1), Fraction(1000)),
        (-1000, 1),
        LINEAGE,
        "x",
        grid=GRID,
    )
    assert [_digest(p.witness_points()) for p in pieces] == [
        "2b1da6546698bedd",
        "43614f7d84e0513e",
    ]
    circle = full_circle(
        "f", ((7001, 3), (0, 7003), (-7001, 5)), True, LINEAGE, "x", grid=GRID
    )
    assert [_digest(p.witness_points()) for p in circle] == [
        "eda2d83bf95164e9", "2d775ce19f759553", "1c3162d833ad6590", "ab34267c07cf84ac",
    ]  # fmt: skip
    assert [p.end for p in circle[:3]] == [(-1, 7003), (-7001, 1), (1, -6999)]


def _digest(points):
    return hashlib.sha256(json.dumps(points).encode()).hexdigest()[:16]


def test_decimal_trigonometry_matches_the_known_values_to_many_digits():
    """The series are accurate far beyond the lattice (checked here with the default 28-digit context)."""
    sine, cosine = circle_math.sin_cos(circle_math.PI / 6)
    assert abs(sine - Decimal("0.5")) < Decimal("1e-25")
    assert abs(cosine - circle_math.sqrt(Decimal(3)) / 2) < Decimal("1e-25")
    assert abs(
        circle_math.atan2(Decimal(1), Decimal(-1)) - 3 * circle_math.PI / 4
    ) < Decimal("1e-25")
    assert abs(circle_math.acos(Decimal("0.5")) - circle_math.PI / 3) < Decimal("1e-25")
