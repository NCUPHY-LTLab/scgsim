"""Snap rounding guarantees (design brief section 3) with a shapely oracle (used only here)."""

from __future__ import annotations

import pytest

shapely = pytest.importorskip("shapely", reason="the planar oracle needs shapely")

from sgb_scenes import (  # noqa: E402
    crossing_pairs,
    max_deviation,
    pass_through_vertices,
    random_layout,
    random_segments,
)
from shapely.geometry import LineString  # noqa: E402
from shapely.ops import polygonize, unary_union  # noqa: E402

from scgsim.geometry.sgb.arrangement.dcel import build_arrangement  # noqa: E402
from scgsim.geometry.sgb.arrangement.noding.snap_rounding import Segment, snap_round  # noqa: E402
from scgsim.geometry.sgb.arrangement.noding.spatial_hash import HotGrid  # noqa: E402

CASES = [
    (seed, count, extent)
    for seed in range(12)
    for count, extent in ((20, 200), (40, 5000), (60, 60))
]
SNAP_BOUND = 2**0.5 / 2


@pytest.mark.parametrize(("seed", "count", "extent"), CASES)
def test_output_is_planar_on_grid_with_no_short_edges(seed, count, extent):
    """Every vertex is a lattice point, no edge is shorter than one unit, nothing crosses."""
    segments = random_segments(seed, count, extent)
    edges = snap_round(segments).edges
    assert all(isinstance(c, int) for e in edges for c in (*e.a, *e.b))
    assert all((e.a[0] - e.b[0]) ** 2 + (e.a[1] - e.b[1]) ** 2 >= 1 for e in edges)
    assert crossing_pairs(edges) == 0
    assert pass_through_vertices(edges) == 0


@pytest.mark.parametrize("seed", range(12))
def test_edges_stay_within_snap_bound_of_a_carrier_when_hot_pixels_are_sparse(seed):
    """One rounding keeps each vertex within sqrt(2)/2 of its carriers."""
    segments = random_segments(seed, 40, 100_000)
    edges = snap_round(segments).edges
    assert max_deviation(edges, segments) <= SNAP_BOUND + 1e-9


@pytest.mark.parametrize("seed", range(6))
def test_dense_inputs_report_deviation_but_stay_planar(seed):
    """Iterated rounding can drift past sqrt(2)/2 when hot pixels touch; planarity still holds."""
    segments = random_segments(seed, 60, 40)
    edges = snap_round(segments).edges
    assert max_deviation(edges, segments) < 3.0
    assert crossing_pairs(edges) == 0


def test_every_edge_keeps_all_carrier_ids_of_overlapping_input():
    """Every edge keeps all carrier ids of overlapping input."""
    a = Segment((0, 0), (100, 0), frozenset(["a"]))
    b = Segment((40, 0), (160, 0), frozenset(["b"]))
    edges = snap_round([a, b]).edges
    by_span = {(e.a[0], e.b[0]): e.carriers for e in edges}
    assert by_span == {(0, 40): {"a"}, (40, 100): {"a", "b"}, (100, 160): {"b"}}


def test_crossing_produces_a_shared_hot_pixel_vertex():
    """Crossing produces a shared hot pixel vertex."""
    x = Segment((0, 0), (10, 10), frozenset(["x"]))
    y = Segment((0, 10), (10, 0), frozenset(["y"]))
    edges = snap_round([x, y]).edges
    assert len(edges) == 4
    assert {p for e in edges for p in (e.a, e.b)} >= {(5, 5)}


@pytest.mark.parametrize(
    ("segment", "makes_hot"),
    [
        (
            ((3, 6), (6, 3)),
            True,
        ),  # x + y = 9 touches only the lower-left corner (4.5, 4.5)
        (((3, 8), (8, 3)), False),  # x + y = 11 touches only the upper-right corner
        (((3, 2), (8, 7)), False),  # x - y = 1 touches only the lower-right corner
        (((2, 3), (7, 8)), False),  # x - y = -1 touches only the upper-left corner
        (((3, 3), (8, 8)), True),  # the diagonal passes through the pixel
    ],
)
def test_a_segment_touching_only_a_pixel_corner_makes_it_hot_only_at_the_lower_left(
    segment, makes_hot
):
    """Half-open pixels ``[c - 1/2, c + 1/2)``: only the lower-left corner belongs to the pixel."""
    touching = HotGrid([(5, 5)]).touching(*segment)
    assert (touching == [(5, 5)]) is makes_hot


@pytest.mark.parametrize(
    ("seed", "count", "extent"),
    [(s, 30, e) for s in range(10) for e in (80, 3000)],
)
def test_oracle_face_count_and_area_match_shapely(seed, count, extent):
    """Oracle face count and area match shapely."""
    edges = snap_round(random_segments(seed, count, extent)).edges
    arr = build_arrangement(edges)
    lines = [LineString([e.a, e.b]) for e in edges]
    polygons = list(polygonize(shapely.node(shapely.MultiLineString(lines))))
    assert len(arr.faces) - 1 == len(polygons)
    assert arr.total_bounded_twice_area() / 2 == pytest.approx(
        sum(p.area for p in polygons), abs=1e-6
    )
    assert arr.total_bounded_twice_area() == arr.top_level_twice_area()
    assert unary_union(polygons).area == pytest.approx(
        arr.top_level_twice_area() / 2, abs=1e-6
    )


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("route", ["A", "B"])
def test_layout_carriers_snap_within_bound_and_planar(seed, route):
    """Layout sources are on the grid, so only oblique crossings move vertices."""
    catalogue = random_layout(40, seed=seed, route=route)
    segments = [
        Segment(c.p, c.q, frozenset([c.id]))
        for prism in catalogue.prisms
        for c in prism.carriers()
    ]
    edges = snap_round(segments).edges
    assert max_deviation(edges, segments) <= SNAP_BOUND + 1e-9
    assert crossing_pairs(edges) == 0
    assert pass_through_vertices(edges) == 0
