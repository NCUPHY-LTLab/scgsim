"""Outline, Net-contact, lumped-port and curve refusals name the object, the layer and the place in um.

The second slice of the located layout refusals; ``test_located_refusals.py`` holds the first.
Each test draws a layout with the defect at a known place, runs the public flow
(``simulation_drc``) where it reaches the refusal, and reads the coordinates back out of the
message: they must fall on the defect.  The last tests show why the identity refusals that
follow the grid audits are internal invariants: the located audit refuses first.
"""

from __future__ import annotations

import re
from dataclasses import replace

import gdstk
import pytest
from sgb_inputs import gds_pad_in_ground
from test_located_refusals import GRID, on, places, rectangle
from test_rectangular_junction_geometry import fixture as junction_fixture
from test_sgb_prisms import with_boundaries
from test_simulation_drc import check, two_pads

from scgsim.documentation import page_url
from scgsim.geometry.grid.booleans import split_gdstk_cutline_loop
from scgsim.geometry.grid.grid_regions import PolygonRegion
from scgsim.geometry.grid.scope import GridScope
from scgsim.geometry.models.input import BoundaryCurveChainSpec, SourceCurveSpec
from scgsim.geometry.sgb.errors import (
    ContactError,
    NonManifoldError,
    OverlapError,
    UnsupportedCurveError,
)
from scgsim.geometry.sgb.prisms.catalogue import build_prism_catalogue
from scgsim.geometry.source.grid_projection import apply_geometry_grid
from scgsim.geometry.source.ground import DrawnPlane, derived_ground_polygons
from scgsim.geometry.source.recipes import audited_declared_source
from scgsim.palace import EigenmodeSim

# Pad outlines inside the 20 um square of ``two_pads``.
HOLE_TOUCHING_THE_OUTLINE = (
    (0, 0),
    (10, 0),
    (8, 5),
    (12, 5),
    (10, 0),
    (20, 0),
    (20, 20),
    (0, 20),
)
TWO_HOLES_TOUCHING_THE_OUTLINE = (
    (0, 0),
    (5, 0),
    (4, 3),
    (6, 3),
    (5, 0),
    (15, 0),
    (14, 3),
    (16, 3),
    (15, 0),
    (20, 0),
    (20, 20),
    (0, 20),
)
SQUARES_MEETING_AT_A_CORNER = (
    (0, 0),
    (10, 0),
    (10, 10),
    (20, 10),
    (20, 20),
    (10, 20),
    (10, 10),
    (0, 10),
)


def point(x, y):
    """A defect that is one point, as a box."""
    return (x, y, x, y)


# ------------------------------------------------------- an outline that meets itself


@pytest.mark.parametrize(
    ("pad", "defects"),
    [
        (HOLE_TOUCHING_THE_OUTLINE, [point(10, 0)]),
        (SQUARES_MEETING_AT_A_CORNER, [point(10, 10)]),
        (TWO_HOLES_TOUCHING_THE_OUTLINE, [point(5, 0), point(15, 0)]),
    ],
    ids=["hole", "corner", "two-holes"],
)
def test_an_outline_that_meets_itself_is_refused_at_each_point(
    monkeypatch, pad, defects
):
    """``simulation_drc`` names the pad, its layer and every point where its outline meets itself."""
    with pytest.raises(ValueError) as error:
        check(EigenmodeSim, two_pads(monkeypatch, gap_um=10.0, pad=pad))
    message = str(error.value)
    assert message.startswith(
        "gdstk cutline boundary is not a simple loop: chip/pad on layer 1/0"
    )
    assert "Fix the layout" in message
    found = places(message)
    assert len(found) == len(defects)
    assert all(on(place, defect) for place, defect in zip(found, defects, strict=True))


def test_a_ground_include_refusal_names_the_include_layer():
    """An include polygon on layer 3/0 that meets itself is refused on 3/0, not on the mask layer 1/0."""
    build, _ = junction_fixture()
    ground = replace(
        build.entities[0],
        semantic_id="chip/ground",
        net_id="Ground",
        geometry={
            "mask_layer": (1, 0),
            "include_layer": (3, 0),
            "include_selector_points_um": ((5.0, 15.0),),
        },
    )
    with pytest.raises(ValueError) as error:
        audited_declared_source(
            derived_ground_polygons,
            GridScope(GRID),
            ground,
            polygons_by_layer={(3, 0): (gdstk.Polygon(HOLE_TOUCHING_THE_OUTLINE),)},
            plane=DrawnPlane((gdstk.rectangle((0, 0), (20, 20)),)),
        )
    message = str(error.value)
    assert message.startswith(
        "gdstk cutline boundary is not a simple loop: chip/ground on layer 3/0"
    )
    (place,) = places(message)
    assert on(place, point(10, 0))


def test_a_ground_plane_refusal_names_the_plane_layer():
    """A plane outline on layer 2/0 that meets itself is refused on 2/0, not on the mask layer 1/0."""
    build, _ = junction_fixture()
    ground = replace(
        build.entities[0],
        semantic_id="chip/ground",
        net_id="Ground",
        geometry={"mask_layer": (1, 0)},
    )
    with pytest.raises(ValueError) as error:
        audited_declared_source(
            derived_ground_polygons,
            GridScope(GRID),
            ground,
            polygons_by_layer={(1, 0): (gdstk.rectangle((2, 15), (4, 17)),)},
            plane=DrawnPlane((gdstk.Polygon(HOLE_TOUCHING_THE_OUTLINE),), (2, 0)),
        )
    message = str(error.value)
    assert message.startswith(
        "gdstk cutline boundary is not a simple loop: chip/ground on layer 2/0"
    )
    (place,) = places(message)
    assert on(place, point(10, 0))


# The occurrence transform of a component placed at (100, 50) um.
PLACED = (1, 0, 0, 1, 100, 50)


def walk_refusal(walk):
    """The refusal of one drawn boundary walk of ``chip/pad`` on 1/0, read in the placed frame."""
    scope = GridScope(GRID).placed(PLACED).naming("chip/pad", "1/0")
    with pytest.raises(ValueError) as error:
        split_gdstk_cutline_loop(scope, walk)
    return str(error.value)


def test_a_meeting_point_is_reported_where_the_occurrence_places_it():
    """The point (10, 0) um of the cell is (110, 50) um in the assembled layout."""
    message = walk_refusal(HOLE_TOUCHING_THE_OUTLINE)
    (place,) = places(message)
    assert on(place, point(110, 50))


def test_an_outline_that_runs_its_edges_twice_is_refused_along_each_edge():
    """A square walked twice repeats each of its four sides; each side is reported."""
    square = rectangle(0, 0, 10, 10)
    message = walk_refusal((*square, *square))
    assert message.startswith(
        "gdstk cutline repeats a retained boundary edge: chip/pad on layer 1/0"
    )
    assert "Fix the layout" in message
    found = places(message)
    sides = [(100, 50, 110, 50), (110, 50, 110, 60), (100, 60, 110, 60)]
    sides.append((100, 50, 100, 60))
    assert len(found) == 4
    assert all(any(on(place, side) for side in sides) for place in found)


def test_a_boundary_loop_outside_its_outer_loop_is_located():
    """A loop drawn at x = 12..14 um beside a 10 um square is the place reported."""
    outside = ((12.0, 2.0), (12.0, 4.0), (14.0, 4.0), (14.0, 2.0))
    walk = tuple(PolygonRegion(rectangle(0, 0, 10, 10), (outside,)).points)
    message = walk_refusal(walk)
    assert message.startswith(
        "gdstk cutline polygon has a non-interior boundary loop: chip/pad on layer 1/0"
    )
    (place,) = places(message)
    assert on(place, (112, 52, 114, 54))


def test_an_outline_that_encloses_nothing_is_located():
    """A walk out and back along one line cancels; the line is the place reported."""
    message = walk_refusal(((0, 0), (10, 0)))
    assert message.startswith(
        "gdstk cutline polygon has no simple boundary loop: chip/pad on layer 1/0"
    )
    (place,) = places(message)
    assert on(place, (100, 50, 110, 50))


# --------------------------------------------------------- conductors of two Nets


def test_a_contact_between_two_nets_names_each_entity_on_its_layer(monkeypatch):
    """Abutting pads of two Nets touch along x = 20 um, y = 0..20 um."""
    with pytest.raises(ContactError) as error:
        check(EigenmodeSim, two_pads(monkeypatch, gap_um=0.0))
    message = str(error.value)
    assert "chip/pad_0000 on layer metal" in message
    assert "chip/electrode_0000 on layer metal" in message
    assert "Fix the layout or the Net declaration" in message
    (place,) = places(message)
    assert on(place, (20, 0, 20, 20))


def test_conductors_of_two_nets_that_overlap_are_located_in_the_overlap(monkeypatch):
    """Pads of two Nets drawn 5 um over each other overlap at x = 15..20 um."""
    with pytest.raises(OverlapError) as error:
        check(EigenmodeSim, two_pads(monkeypatch, gap_um=-5.0))
    message = str(error.value)
    assert message.startswith("different-Net conductors overlap near")
    assert "chip/pad_0000 on layer metal" in message
    assert "Fix the layout or the Net declaration" in message
    (place,) = places(message)
    assert on(place, (15, 0, 20, 20))


def test_conductors_of_one_net_meeting_at_a_corner_are_a_located_pinch(monkeypatch):
    """A pad below the electrode meets it only at the corner (20, 0) um, pinching the vacuum."""
    below = ((0, -20), (20, -20), (20, 0), (0, 0))
    with pytest.raises(NonManifoldError) as error:
        check(
            EigenmodeSim,
            two_pads(monkeypatch, gap_um=0.0, pad=below),
            nets={"Ground": ("chip/pad", "chip/electrode")},
        )
    message = str(error.value)
    assert message.startswith("non-manifold pinch:")
    assert "chip/pad_0000 on layer metal" in message
    # The corner is inside the planar extent: a pinch between two conductors, not one at the vacuum's edge.
    assert "pinched between conductors" in message
    assert message.endswith(
        "Fix the layout: give them a shared edge or separate them. See "
        + page_url("docs/known-geometry-problems", "pinch-inside-the-layout")
    )
    assert re.search(
        r" See https://ncuphy-ltlab\.github\.io/scgsim/v\d+\.\d+\.\d+(-dev)?/"
        r"docs/known-geometry-problems\.html#pinch-inside-the-layout$",
        message,
    )
    (place,) = places(message)
    assert on(place, point(20, 0))


# ------------------------------------------------------- a lumped port on one conductor


def port_on_the_pad_only(monkeypatch):
    """``two_pads`` 10 um apart, with the port sheet at x = 15..25 um reaching only the pad."""
    import gdsfactory as gf  # noqa: PLC0415 -- optional layout stack, imported only by the tests that need it

    semantics = two_pads(monkeypatch, gap_um=10.0, port=True).info[
        "component_semantics"
    ]
    component = gf.Component()
    component.add_polygon(rectangle(0, 0, 20, 20), layer=(1, 0))
    component.add_polygon(rectangle(30, 0, 60, 20), layer=(2, 0))
    component.add_polygon(rectangle(15, 8, 25, 12), layer=(9, 0))
    component.add_port(name="P1", center=(22, 10), width=4, orientation=0, layer=(9, 0))
    component.info["component_semantics"] = semantics
    return component


def test_a_lumped_port_on_one_conductor_is_located_on_its_layer(monkeypatch):
    """The refusal names the port, its one host, the sheet's layer and the sheet's box."""
    with pytest.raises(ValueError) as error:
        check(EigenmodeSim, port_on_the_pad_only(monkeypatch))
    message = str(error.value)
    assert message.startswith(
        "lumped port 'chip/P1' must land on two conductors to be made a rectangle; "
        "it lands on 1: ['chip/pad_0000']; the sheet is drawn on layer 9/0"
    )
    assert "Fix the layout or the port declaration" in message
    (place,) = places(message)
    assert on(place, (15, 8, 25, 12)) and on((15, 8, 25, 12), place)


# --------------------------------------------------------------- an unsupported curve


def test_an_unsupported_curve_is_located_by_its_ends(tmp_path):
    """The B-spline of a pad outline runs from (60, 40) um to (40, 60) um; both ends are named."""
    pad = gds_pad_in_ground(tmp_path, with_port=False)
    chain = BoundaryCurveChainSpec(
        "pad_outline",
        "pad",
        (50.0, 50.0),
        (
            SourceCurveSpec("line_segment", ((40.0, 40.0), (60.0, 40.0))),
            SourceCurveSpec(
                "bspline", ((60.0, 40.0), (60.0, 60.0), (40.0, 60.0)), degree=2
            ),
            SourceCurveSpec("line_segment", ((40.0, 60.0), (40.0, 40.0))),
        ),
    )
    curved = with_boundaries(pad, boundary_curves=(chain,))
    with pytest.raises(UnsupportedCurveError) as error:
        build_prism_catalogue(curved.build_input, curved.representation)
    message = str(error.value)
    assert message.startswith("pad_outline (Entity 'pad'") and "curve 1:" in message
    assert "Fix the layout" in message
    start, end = places(message)
    assert on(start, point(60, 40)) and on(end, point(40, 60))


# ------------------------------- the identity refusals after the grid audits are invariants


NECK = (
    (0, 0),
    (4, 0),
    (4, 1),
    (6, 1),
    (6, 0),
    (10, 0),
    (10, 2),
    (6, 2),
    (6, 1.0004),
    (4, 1.0004),
    (4, 2),
    (0, 2),
)


def test_a_port_sheet_that_splits_on_the_grid_is_refused_located_before_its_identity():
    """A sheet whose 0.4 nm neck snaps apart never reaches ``port sheet component identity``.

    Its ``PORT::`` operand is physical, so the topology audit of the same arrangement refuses
    it first, at the neck.
    """
    build, _ = junction_fixture()
    sheet = replace(build.port_sheet_regions[0], exterior=NECK)
    with pytest.raises(ValueError) as error:
        apply_geometry_grid(replace(build, port_sheet_regions=(sheet,)), GRID)
    message = str(error.value)
    assert message.startswith(
        "physical geometry collapsed or changed component/hole topology at grid: "
        "PORT::sheet on layer junction"
    )
    (place,) = places(message)
    assert on(place, (4, 1, 6, 1.0004))


def test_a_polygon_that_splits_on_the_grid_is_refused_located_before_its_identity():
    """A conductor polygon with the same neck never reaches ``source polygon component identity``."""
    build, _ = junction_fixture()
    left = replace(build.polygons[0], exterior=tuple((x - 12, y) for x, y in NECK))
    with pytest.raises(ValueError) as error:
        apply_geometry_grid(replace(build, polygons=(left, *build.polygons[1:])), GRID)
    message = str(error.value)
    assert message.startswith(
        "physical geometry collapsed or changed component/hole topology at grid: left"
    )
    (place,) = places(message)
    assert on(place, (-8, 1, -6, 1.0004))


# ------------------------------------------- a polygon narrower than the database unit


def test_a_sliver_narrower_than_the_database_unit_is_located_on_its_layer(monkeypatch):
    """A 0.2 nm sliver at x = 5 um, y = 25..26 um collapses to a line when the GDS stores it."""
    component = two_pads(monkeypatch, gap_um=10.0)
    component.add_polygon(rectangle(5, 25, 5.0002, 26), layer=(1, 0))
    with pytest.raises(ValueError) as error:
        check(EigenmodeSim, component)
    message = str(error.value)
    assert message.startswith(
        "GDS polygon requires at least 3 unique points: chip/pad on layer 1/0"
    )
    assert "Fix the layout" in message
    (place,) = places(message)
    assert on(place, (5, 25, 5.0002, 26))
