"""Real cutline boundary reconstruction and scoped Boolean containment regressions."""

from inspect import signature
from types import SimpleNamespace

import gdstk
import pytest

import scgsim.geometry.source._occurrence as occurrence_source
import scgsim.geometry.source.ground as ground_source
from scgsim.geometry._primitives.grid import (
    PolygonRegion,
    geometry_grid_scope,
    in_occurrence_frame,
    polygon_rings,
    preserve_declared_source,
    split_gdstk_cutline_loop,
)
from scgsim.geometry._primitives.loops import _geometry_ref_from_gdstk_polygon
from scgsim.geometry._primitives.planar_loops import (
    clean_loop,
    cutline_boundary_loops,
    polygon_area,
)
from scgsim.geometry.models.grid import GeometryGridContext

OUTER = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))
HOLE = ((2.0, 2.0), (2.0, 4.0), (4.0, 4.0), (4.0, 2.0))
SCOPES = (None, GeometryGridContext(0.001))


def _boundary_edges(loop):
    return {
        tuple(sorted((start, end)))
        for start, end in zip(loop, (*loop[1:], loop[0]), strict=False)
    }


@pytest.mark.parametrize("context", SCOPES)
def test_real_gdstk_hole_cutline_walk(context):
    """Recover physical hole boundaries from real native cutline serialization."""
    polygon = gdstk.boolean(
        gdstk.Polygon(OUTER), gdstk.Polygon(HOLE), "not", precision=1e-9
    )[0]
    walk = tuple(tuple(map(float, point)) for point in polygon.points)
    with geometry_grid_scope(context):
        outer, holes = split_gdstk_cutline_loop(walk)
        assert polygon_rings(polygon) == (outer, holes)
        ref = _geometry_ref_from_gdstk_polygon({"kind": "surface"}, polygon)
    assert abs(polygon_area(outer)) == 100.0
    assert len(holes) == 1
    assert _boundary_edges(holes[0]) == _boundary_edges(HOLE)
    assert ref["kind"] == "surface"
    assert abs(polygon_area(ref["outer_loop"])) == 100.0
    assert _boundary_edges(ref["hole_loops"][0]) == _boundary_edges(HOLE)


@pytest.mark.parametrize("context", SCOPES)
@pytest.mark.parametrize("reverse", (False, True))
def test_subdivided_reversed_connector_cancels(context, reverse):
    # One return connector has an extra collinear vertex; atomic splitting must
    # cancel both copies before reconstructing the physical boundary cycles.
    """Cancel unequal connector subdivisions in either walk orientation."""
    walk = (OUTER[0], *HOLE, HOLE[0], (1.0, 1.0), OUTER[0], *OUTER[1:])
    if reverse:
        walk = tuple(reversed(walk))
    with geometry_grid_scope(context):
        outer, holes = split_gdstk_cutline_loop(walk)
    assert _boundary_edges(outer) == _boundary_edges(OUTER)
    assert len(holes) == 1
    assert _boundary_edges(holes[0]) == _boundary_edges(HOLE)


@pytest.mark.parametrize("context", SCOPES)
def test_noninterior_boundary_still_fails_scoped_boolean_validation(context):
    """Keep noninterior boundary failure under both Boolean execution scopes."""
    outside = ((12.0, 2.0), (12.0, 4.0), (14.0, 4.0), (14.0, 2.0))
    walk = tuple(PolygonRegion(OUTER, (outside,)).points)
    outer, holes = cutline_boundary_loops(walk)
    assert _boundary_edges(outer) == _boundary_edges(OUTER)
    assert len(holes) == 1
    with (
        geometry_grid_scope(context),
        pytest.raises(
            ValueError, match="gdstk cutline polygon has a non-interior boundary loop"
        ),
    ):
        split_gdstk_cutline_loop(walk)


@pytest.mark.parametrize("context", SCOPES)
def test_repeated_retained_edges_still_fail(context):
    """Reject repeated retained boundaries before containment validation."""
    with (
        geometry_grid_scope(context),
        pytest.raises(
            ValueError, match="gdstk cutline repeats a retained boundary edge"
        ),
    ):
        split_gdstk_cutline_loop((*OUTER, *OUTER))


@pytest.mark.parametrize("context", SCOPES)
def test_polygon_rings_retains_raw_coordinates_and_collinear_vertices(context):
    """Preserve the two entry points distinct coordinate-cleaning contracts."""
    walk = ((0.1234567894, 0.0), (1.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0))
    polygon = SimpleNamespace(points=walk)
    with geometry_grid_scope(context):
        outer, holes = polygon_rings(polygon)
        ref = _geometry_ref_from_gdstk_polygon({}, polygon)
    assert set(outer) == set(walk)
    assert holes == ()
    assert set(ref["outer_loop"]) == set(clean_loop(walk))
    assert ref["hole_loops"] == ()
    assert (0.1234567894, 0.0) in outer
    assert (0.123456789, 0.0) in ref["outer_loop"]
    assert (1.0, 0.0) not in ref["outer_loop"]


@pytest.mark.parametrize("decorate", (in_occurrence_frame, preserve_declared_source))
def test_grid_decorators_preserve_function_metadata(decorate):
    """Preserve introspection and execute the original function through its scope."""

    def original(entity: str, *, transform: tuple) -> tuple:
        """Return the supplied entity and occurrence transform."""
        return entity, transform

    wrapped = decorate(original)
    assert wrapped.__name__ == original.__name__
    assert wrapped.__qualname__ == original.__qualname__
    assert wrapped.__doc__ == original.__doc__
    assert wrapped.__module__ == original.__module__
    assert wrapped.__annotations__ == original.__annotations__
    assert wrapped.__wrapped__ is original
    assert signature(wrapped) == signature(original)
    transform = (1, 0, 0, 1, 3, 4)
    with geometry_grid_scope(None):
        assert wrapped("entity", transform=transform) == ("entity", transform)


@pytest.mark.parametrize(
    ("module", "name"),
    (
        (occurrence_source, "_gds_polygons_for_entity"),
        (occurrence_source, "_occurrence_polygons_for_entity"),
        (occurrence_source, "_occurrence_ground_polygons_for_entity"),
        (ground_source, "_derived_ground_polygons"),
    ),
)
def test_source_entry_points_preserve_decorated_metadata(module, name):
    """Keep actual source entry point names, docs and signatures available."""
    entry = getattr(module, name)
    original = entry.__wrapped__
    assert entry.__name__ == original.__name__ == name
    assert entry.__doc__ == original.__doc__
    assert entry.__annotations__ == original.__annotations__
    assert signature(entry) == signature(original)
