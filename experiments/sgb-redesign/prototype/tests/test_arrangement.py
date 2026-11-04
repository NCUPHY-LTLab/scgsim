"""Faces, holes and nesting in the half-edge structure."""

from __future__ import annotations

from sgbproto.arrangement import build_arrangement, interior_point
from sgbproto.geometry import point_in_ring
from sgbproto.snap_rounding import Segment, snap_round


def ring(points, name):
    n = len(points)
    return [Segment(points[i], points[(i + 1) % n], frozenset([f"{name}{i}"])) for i in range(n)]


def build(*rings):
    segments = [s for k, r in enumerate(rings) for s in ring(r, f"r{k}_")]
    return build_arrangement(snap_round(segments).edges)


def test_island_inside_hole_inside_square_nests_correctly():
    outer = [(0, 0), (100, 0), (100, 100), (0, 100)]
    hole = [(10, 10), (90, 10), (90, 90), (10, 90)]
    island = [(40, 40), (60, 40), (60, 60), (40, 60)]
    arr = build(outer, hole, island)
    # faces: unbounded, ring face (outer minus hole), hole face (minus island), island face
    assert len(arr.faces) == 4
    areas = sorted(arr.face_twice_area(i) for i in range(1, 4))
    assert areas == sorted([2 * (100 * 100 - 80 * 80), 2 * (80 * 80 - 20 * 20), 2 * 20 * 20])
    assert arr.total_bounded_twice_area() == arr.top_level_twice_area() == 2 * 100 * 100
    assert len(arr.faces[0].holes) == 1


def test_disjoint_squares_are_two_holes_of_the_unbounded_face():
    arr = build([(0, 0), (10, 0), (10, 10), (0, 10)], [(20, 0), (30, 0), (30, 10), (20, 10)])
    assert len(arr.faces) == 3
    assert len(arr.faces[0].holes) == 2


def test_interior_point_is_strictly_inside_a_face_with_a_hole():
    arr = build([(0, 0), (100, 0), (100, 100), (0, 100)], [(1, 1), (99, 1), (99, 99), (1, 99)][::-1])
    for fi in range(1, len(arr.faces)):
        p = interior_point(arr, fi)
        outer = arr.cycle_points(arr.faces[fi].outer)
        assert point_in_ring(outer, p)
        for hole in arr.faces[fi].holes:
            assert not point_in_ring(arr.cycle_points(hole), p)


def test_reflex_polygon_interior_point_is_inside():
    l_shape = [(0, 0), (100, 0), (100, 30), (30, 30), (30, 100), (0, 100)]
    arr = build(l_shape)
    p = interior_point(arr, 1)
    assert point_in_ring(arr.cycle_points(arr.faces[1].outer), p)
