"""The minimal extreme-case fixtures: each passes every structural check or is refused early with the expected error.

Every fixture is a few prisms (under 100 faces), so the validators run in milliseconds.  The first test
is the catalogue itself; the others pin what each fixture is *for* (so a fixture cannot silently stop
testing its risk) and the regressions the catalogue found.
"""

from __future__ import annotations

import math

import pytest
from sgb_extreme_cases import (
    FIXTURES,
    cases,
    catalogue_of,
    chip,
    metal,
    refinement,
)
from sgb_scenes import rect_um

from scgsim.geometry.sgb.errors import ContactError, NonManifoldError
from scgsim.geometry.sgb.model import build_model_from_catalogue
from scgsim.geometry.sgb.records.grid import exact_z


def build(name: str, route: str):
    """Build one fixture under a route."""
    return build_model_from_catalogue(FIXTURES[name].build(route), route)


@pytest.mark.parametrize(
    ("fixture", "route"), cases(), ids=[f"{f.name}-{r}" for f, r in cases()]
)
def test_every_fixture_passes_every_check_or_is_refused_early_with_the_expected_error(
    fixture, route
):
    """Validation covers closure, orientation, partition and cylinders for each (fixture, route)."""
    expected = fixture.expect_error.get(route)
    if expected is not None:
        with pytest.raises(expected):
            build_model_from_catalogue(fixture.build(route), route)
        return
    model = build_model_from_catalogue(fixture.build(route), route)
    assert not model.report.failures
    assert model.counts["faces"] < 100  # a handful of prisms, a few dozen faces
    unnamed = [
        f
        for f in model.topology.faces.values()
        if model.routed.is_emitted_face(f.neg_volume, f.pos_volume)
        and model.routed.face_group[f.id] is None
    ]
    assert unnamed == []


def test_the_catalogue_covers_the_listed_risks():
    """The risks the design brief lists are all present."""
    names = set(FIXTURES)
    for wanted in (
        "near_tangent_arc_line", "arcs_crossing_shallow", "edges_1_to_3_g", "kissing_corners_same_net",
        "collinear_overlap", "hole_touches_outer", "arc_with_foreign_vertex", "floating_conductor",
        "nested_holes", "stacked_corner_touch", "port_edge_on_metal_sidewall", "near_coincident_layers_2g",
    ):  # fmt: skip
        assert wanted in names
    assert len(cases()) >= 40


# --------------------------------------------------------------------------- what the fixtures are for


def horizontal_edge_lengths(topology) -> list[float]:
    """Lengths of the edges that stay in their plane."""
    lengths = []
    for edge in topology.edges.values():
        a, b = topology.vertices[edge.v0], topology.vertices[edge.v1]
        if (a.x, a.y) != (b.x, b.y):
            lengths.append(math.hypot(a.x - b.x, a.y - b.y))
    return lengths


@pytest.mark.parametrize("route", ["A", "B", "C"])
def test_edges_one_to_three_grid_units_long_are_kept_exactly(route):
    """No edge shorter than g exists, and 1, 2, 3 g and the 1.4 g diagonal all survive."""
    lengths = horizontal_edge_lengths(build("edges_1_to_3_g", route).topology)
    for wanted in (1.0, 2.0, 3.0, math.sqrt(2)):
        assert any(abs(x - wanted) < 1e-9 for x in lengths), wanted


def test_a_gap_below_a_grid_unit_merges_two_layers_and_a_gap_of_two_makes_a_real_sliver():
    """0.4 g snaps to the same pixel (no sliver); 2 g is a genuine 2 nm face."""

    def smallest_horizontal_face(name):
        """Smallest horizontal face."""
        topology = build(name, "B").topology
        return min(
            f.twice_xy_area for f in topology.faces.values() if f.kind == "horizontal"
        )

    assert smallest_horizontal_face("near_coincident_layers_sub_grid") > 1_000_000
    assert (
        smallest_horizontal_face("near_coincident_layers_2g") <= 2 * 2 * 8000
    )  # 2 g wide, 8 um long


def test_the_near_tangent_circle_and_line_cross_but_the_exactly_tangent_pair_only_touches():
    """A 2 nm secant is a crossing; an exact tangent is a pinch of finite metal (but legal for sheets)."""
    near = build("near_tangent_arc_line", "B").topology
    assert (
        near.counts()["volumes"] == 4
    )  # the disk, the bar, the substrate and the vacuum
    with pytest.raises(NonManifoldError, match=r"bar.*disk|disk.*bar"):
        build("exactly_tangent_arc_line", "B")
    assert build("exactly_tangent_arc_line", "A").report is not None


def test_refused_fixtures_name_both_entities_the_net_and_the_location():
    """Error messages carry what a person needs to find the defect."""
    with pytest.raises(NonManifoldError) as error:
        build("kissing_corners_same_net", "B")
    text = str(error.value)
    for needle in ("padL", "padR", "Net N", "(0.0000, 0.0000)", "vertical edge"):
        assert needle in text
    with pytest.raises(ContactError, match=r"N1.*N2|N2.*N1"):
        build("kissing_corners_different_nets", "A")


def test_a_floating_conductor_is_an_inner_shell_in_b_a_volume_in_c_and_an_embedded_sheet_in_a():
    """One fixture, three routes, three representations of a conductor surrounded by vacuum."""
    b = build("floating_conductor", "B")
    shells_b = sorted(
        len(v.shells)
        for vid, v in b.topology.volumes.items()
        if vid in b.routed.emitted_volumes
    )
    assert shells_b == [1, 2]  # the substrate; the vacuum with an inner void shell
    c = build("floating_conductor", "C")
    assert sorted(len(v.shells) for v in c.topology.volumes.values()) == [1, 1, 2]
    a = build("floating_conductor", "A")
    assert sum(len(v.embedded_faces) for v in a.topology.volumes.values()) == 1


def test_nested_holes_give_faces_with_several_loops():
    """An island in a hole in an island in a hole."""
    topology = build("nested_holes", "A").topology
    assert sum(1 for f in topology.faces.values() if len(f.loops) >= 2) >= 3


def test_a_pinched_hole_is_one_cycle_of_eight_edges_for_sheets():
    """The hole of two squares touching at a corner is one boundary cycle through the pinch vertex twice."""
    topology = build("kissing_corners_same_net", "A").topology
    pinched = [
        f for f in topology.faces.values() if len(f.loops) == 2 and len(f.loops[1]) == 8
    ]
    assert len(pinched) == 1


@pytest.mark.parametrize("route", ["A", "B"])
def test_a_port_edge_on_a_metal_sidewall_keeps_its_group_in_both_routes(route):
    """The port stays one named surface even where it touches a wall."""
    assert "PORT::p1" in build("port_edge_on_metal_sidewall", route).routed.groups


def test_a_refinement_curve_ending_on_a_boundary_keeps_that_vertex_on_the_boundary_edge():
    """Regression: the curve's end on a pad boundary is a vertex of the boundary edge, not a loose point."""
    family = refinement("ref", (-10_000, 500), (0, 500), exact_z(0))
    catalogue = catalogue_of(
        "ref_touch",
        "A",
        [*chip(), metal("pad", "N", rect_um(-10, -5, 10, 5), "A")],
        (family,),
    )
    topology = build_model_from_catalogue(catalogue, "A").topology
    host = next(f for f in topology.faces.values() if f.refinement_edges)
    boundary = {
        v
        for loop in host.loops
        for eid, _ in loop
        for v in (topology.edges[eid].v0, topology.edges[eid].v1)
    }
    assert topology.edges[host.refinement_edges[0]].v0 in boundary


def test_stacked_slabs_touching_at_a_vertex_keep_their_levels_conformal():
    """Same-Net slabs meeting at one corner of a level and a vertex on a wall edge both validate."""
    for name in ("stacked_corner_touch", "stacked_vertex_on_wall_edge"):
        for route in ("B", "C"):
            assert not build(name, route).report.failures
