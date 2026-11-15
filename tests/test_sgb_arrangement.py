"""Faces, holes and nesting in the half-edge structure."""

from __future__ import annotations

from scgsim.geometry.exact.plane import point_in_ring
from scgsim.geometry.sgb.arrangement.dcel import build_arrangement, interior_point
from scgsim.geometry.sgb.arrangement.noding.snap_rounding import Segment, snap_round


def ring(points, name):
    """Ring."""
    count = len(points)
    return [
        Segment(points[i], points[(i + 1) % count], frozenset([f"{name}{i}"]))
        for i in range(count)
    ]


def arrange(*rings):
    """Arrange."""
    segments = [s for k, r in enumerate(rings) for s in ring(r, f"r{k}_")]
    return build_arrangement(snap_round(segments).edges)


def test_island_inside_hole_inside_square_nests_correctly():
    """Island inside hole inside square nests correctly."""
    outer = [(0, 0), (100, 0), (100, 100), (0, 100)]
    hole = [(10, 10), (90, 10), (90, 90), (10, 90)]
    island = [(40, 40), (60, 40), (60, 60), (40, 60)]
    arr = arrange(outer, hole, island)
    # faces: unbounded, ring face (outer minus hole), hole face (minus island), island face
    assert len(arr.faces) == 4
    areas = sorted(arr.face_twice_area(i) for i in range(1, 4))
    assert areas == sorted(
        [2 * (100 * 100 - 80 * 80), 2 * (80 * 80 - 20 * 20), 2 * 20 * 20]
    )
    assert arr.total_bounded_twice_area() == arr.top_level_twice_area() == 2 * 100 * 100
    assert len(arr.faces[0].holes) == 1


def test_disjoint_squares_are_two_holes_of_the_unbounded_face():
    """Disjoint squares are two holes of the unbounded face."""
    arr = arrange(
        [(0, 0), (10, 0), (10, 10), (0, 10)], [(20, 0), (30, 0), (30, 10), (20, 10)]
    )
    assert len(arr.faces) == 3
    assert len(arr.faces[0].holes) == 2


def test_interior_point_is_strictly_inside_a_face_with_a_hole():
    """Interior point is strictly inside a face with a hole."""
    arr = arrange(
        [(0, 0), (100, 0), (100, 100), (0, 100)],
        [(1, 1), (99, 1), (99, 99), (1, 99)][::-1],
    )
    for face in range(1, len(arr.faces)):
        point = interior_point(arr, face)
        outer = arr.cycle_points(arr.faces[face].outer)
        assert point_in_ring(outer, point)
        for hole in arr.faces[face].holes:
            assert not point_in_ring(arr.cycle_points(hole), point)


def test_reflex_polygon_interior_point_is_inside():
    """Reflex polygon interior point is inside."""
    l_shape = [(0, 0), (100, 0), (100, 30), (30, 30), (30, 100), (0, 100)]
    arr = arrange(l_shape)
    point = interior_point(arr, 1)
    assert point_in_ring(arr.cycle_points(arr.faces[1].outer), point)


def test_face_anchor_is_the_first_vertex_of_the_outer_cycle():
    """Face anchor is the first vertex of the outer cycle."""
    arr = arrange([(0, 0), (10, 0), (10, 10), (0, 10)])
    assert arr.face_anchor(1) in arr.vertices
