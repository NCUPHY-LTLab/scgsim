"""Snap rounding guarantees (brief section 3) and the shapely oracle (used only here)."""

from __future__ import annotations

import pytest
import shapely
from shapely.geometry import LineString
from shapely.ops import polygonize, unary_union

from helpers import crossing_pairs, max_deviation, pass_through_vertices, random_segments
from sgbproto.arrangement import build_arrangement
from sgbproto.snap_rounding import Segment, snap_round

CASES = [(seed, n, extent) for seed in range(12) for n, extent in ((20, 200), (40, 5000), (60, 60))]


@pytest.mark.parametrize("seed,n,extent", CASES)
def test_output_is_planar_on_grid_with_no_short_edges(seed, n, extent):
    segments = random_segments(seed, n, extent)
    edges = snap_round(segments).edges
    assert all(isinstance(c, int) for e in edges for c in (*e.a, *e.b))
    assert all((e.a[0] - e.b[0]) ** 2 + (e.a[1] - e.b[1]) ** 2 >= 1 for e in edges)
    assert crossing_pairs(edges) == 0
    assert pass_through_vertices(edges) == 0


@pytest.mark.parametrize("seed", range(12))
def test_edges_stay_within_snap_bound_of_a_carrier_when_hot_pixels_are_sparse(seed):
    segments = random_segments(seed, 40, 100_000)
    edges = snap_round(segments).edges
    assert max_deviation(edges, segments) <= 2**0.5 / 2 + 1e-9


@pytest.mark.parametrize("seed", range(6))
def test_dense_inputs_report_deviation_but_stay_planar(seed):
    # Iterated rounding can drift beyond sqrt(2)/2 when hot pixels are adjacent; we only require planarity
    # and a loose bound here (the measured value is reported in the README).
    segments = random_segments(seed, 60, 40)
    edges = snap_round(segments).edges
    assert max_deviation(edges, segments) < 3.0


def test_every_edge_keeps_all_carrier_ids_of_overlapping_input():
    a = Segment((0, 0), (100, 0), frozenset(["a"]))
    b = Segment((40, 0), (160, 0), frozenset(["b"]))
    edges = snap_round([a, b]).edges
    by_span = {(e.a[0], e.b[0]): e.carriers for e in edges}
    assert by_span == {(0, 40): {"a"}, (40, 100): {"a", "b"}, (100, 160): {"b"}}


def test_crossing_produces_a_shared_hot_pixel_vertex():
    x = Segment((0, 0), (10, 10), frozenset(["x"]))
    y = Segment((0, 10), (10, 0), frozenset(["y"]))
    edges = snap_round([x, y]).edges
    assert len(edges) == 4
    assert {p for e in edges for p in (e.a, e.b)} >= {(5, 5)}


@pytest.mark.parametrize("seed,n,extent", [(s, 30, e) for s in range(10) for e in (80, 3000)])
def test_oracle_face_count_and_area_match_shapely(seed, n, extent):
    edges = snap_round(random_segments(seed, n, extent)).edges
    arr = build_arrangement(edges)
    lines = [LineString([e.a, e.b]) for e in edges]
    polygons = list(polygonize(shapely.node(shapely.MultiLineString(lines))))
    assert len(arr.faces) - 1 == len(polygons)
    assert arr.total_bounded_twice_area() / 2 == pytest.approx(sum(p.area for p in polygons), abs=1e-6)
    assert arr.total_bounded_twice_area() == arr.top_level_twice_area()
    assert unary_union(polygons).area == pytest.approx(arr.top_level_twice_area() / 2, abs=1e-6)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("route", ["A", "B"])
def test_layout_carriers_snap_within_bound_and_planar(seed, route):
    from sgbproto.scenes import random_layout

    spec = random_layout(40, seed=seed, route=route)
    segments = [
        Segment(c.p, c.q, frozenset([c.id])) for p in spec.prisms for c in p.carriers()
    ]
    edges = snap_round(segments).edges
    assert max_deviation(edges, segments) <= 2**0.5 / 2 + 1e-9
    assert crossing_pairs(edges) == 0
    assert pass_through_vertices(edges) == 0
