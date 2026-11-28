"""Real cutline boundary reconstruction and scoped Boolean containment regressions."""

from types import SimpleNamespace

import gdstk
import pytest

from scgsim.geometry.grid.booleans import polygon_rings, split_gdstk_cutline_loop
from scgsim.geometry.grid.grid_regions import PolygonRegion
from scgsim.geometry.grid.loops import geometry_ref_from_gdstk_polygon
from scgsim.geometry.grid.planar_loops import (
    clean_loop,
    cutline_boundary_loops,
    polygon_area,
)
from scgsim.geometry.grid.scope import UNGRIDDED, GridScope
from scgsim.geometry.models.grid import GeometryGridContext

OUTER = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))
HOLE = ((2.0, 2.0), (2.0, 4.0), (4.0, 4.0), (4.0, 2.0))
SCOPES = (UNGRIDDED, GridScope(GeometryGridContext(0.001)))


def _boundary_edges(loop):
    return {
        tuple(sorted((start, end)))
        for start, end in zip(loop, (*loop[1:], loop[0]), strict=False)
    }


@pytest.mark.parametrize("scope", SCOPES)
def test_real_gdstk_hole_cutline_walk(scope):
    """Recover physical hole boundaries from real native cutline serialization."""
    polygon = gdstk.boolean(
        gdstk.Polygon(OUTER), gdstk.Polygon(HOLE), "not", precision=1e-9
    )[0]
    walk = tuple(tuple(map(float, point)) for point in polygon.points)
    outer, holes = split_gdstk_cutline_loop(scope, walk)
    assert polygon_rings(scope, polygon) == (outer, holes)
    ref = geometry_ref_from_gdstk_polygon(scope, {"kind": "surface"}, polygon)
    assert abs(polygon_area(outer)) == 100.0
    assert len(holes) == 1
    assert _boundary_edges(holes[0]) == _boundary_edges(HOLE)
    assert ref["kind"] == "surface"
    assert abs(polygon_area(ref["outer_loop"])) == 100.0
    assert _boundary_edges(ref["hole_loops"][0]) == _boundary_edges(HOLE)


@pytest.mark.parametrize("scope", SCOPES)
@pytest.mark.parametrize("reverse", (False, True))
def test_subdivided_reversed_connector_cancels(scope, reverse):
    # One return connector has an extra collinear vertex; atomic splitting must
    # cancel both copies before reconstructing the physical boundary cycles.
    """Cancel unequal connector subdivisions in either walk orientation."""
    walk = (OUTER[0], *HOLE, HOLE[0], (1.0, 1.0), OUTER[0], *OUTER[1:])
    if reverse:
        walk = tuple(reversed(walk))
    outer, holes = split_gdstk_cutline_loop(scope, walk)
    assert _boundary_edges(outer) == _boundary_edges(OUTER)
    assert len(holes) == 1
    assert _boundary_edges(holes[0]) == _boundary_edges(HOLE)


@pytest.mark.parametrize("scope", SCOPES)
def test_noninterior_boundary_still_fails_scoped_boolean_validation(scope):
    """Keep noninterior boundary failure under both Boolean execution scopes."""
    outside = ((12.0, 2.0), (12.0, 4.0), (14.0, 4.0), (14.0, 2.0))
    walk = tuple(PolygonRegion(OUTER, (outside,)).points)
    outer, holes = cutline_boundary_loops(walk)
    assert _boundary_edges(outer) == _boundary_edges(OUTER)
    assert len(holes) == 1
    with pytest.raises(
        ValueError, match="gdstk cutline polygon has a non-interior boundary loop"
    ):
        split_gdstk_cutline_loop(scope, walk)


@pytest.mark.parametrize("scope", SCOPES)
def test_repeated_retained_edges_still_fail(scope):
    """Reject repeated retained boundaries before containment validation."""
    with pytest.raises(
        ValueError, match="gdstk cutline repeats a retained boundary edge"
    ):
        split_gdstk_cutline_loop(scope, (*OUTER, *OUTER))


@pytest.mark.parametrize("scope", SCOPES)
def test_polygon_rings_retains_raw_coordinates_and_collinear_vertices(scope):
    """Preserve the two entry points distinct coordinate-cleaning contracts."""
    walk = ((0.1234567894, 0.0), (1.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0))
    polygon = SimpleNamespace(points=walk)
    outer, holes = polygon_rings(scope, polygon)
    ref = geometry_ref_from_gdstk_polygon(scope, {}, polygon)
    assert set(outer) == set(walk)
    assert holes == ()
    assert set(ref["outer_loop"]) == set(clean_loop(walk))
    assert ref["hole_loops"] == ()
    assert (0.1234567894, 0.0) in outer
    assert (0.123456789, 0.0) in ref["outer_loop"]
    assert (1.0, 0.0) not in ref["outer_loop"]
