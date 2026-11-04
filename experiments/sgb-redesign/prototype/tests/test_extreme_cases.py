"""The minimal extreme-case fixtures: every one runs through every check and each exercises its own risk (phase 4).

Each fixture is a few prisms (under 100 faces), so the official checkers run in milliseconds here. The first
test is the catalogue itself; the others pin what each fixture is *for* (so a fixture cannot silently stop testing
its risk) and the regressions the catalogue found.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from sgbproto.brep_writer import plan_brep
from sgbproto.errors import ContactError, NonManifoldError
from sgbproto.extreme_cases import FIXTURES, band_klayout_rows, cases, run_fixture
from sgbproto.occt_check import OCP_AVAILABLE
from sgbproto.pipeline import build
from sgbproto.records import ModelSpec


def ids(case):
    return f"{case[0].name}-{case[1]}"


@pytest.mark.parametrize("fixture,route", cases(), ids=[ids(c) for c in cases()])
def test_every_fixture_passes_every_check_or_is_refused_early_with_the_expected_error(tmp_path, fixture, route):
    row = run_fixture(fixture, route, tmp_path, occt=OCP_AVAILABLE)
    assert row["ok"], {k: row.get(k) for k in ("error", "writer_refused", "gmsh", "occt")}
    if fixture.expect_error.get(route):
        assert row["error"]["type"] == fixture.expect_error[route].__name__
        assert row["times_s"]["build"] < 1.0
    elif not row.get("writer_refused"):
        assert row["plan_counts"]["faces"] < 100  # a handful of prisms, a few dozen faces
        assert all(g["conformal"] for g in row["gmsh"].values())
        for report in row.get("occt", {}).values():
            assert report["brepcheck_faulty"] == 0 and not report["bop_faulty"] and report["plan_mismatches"] == 0


def test_the_catalogue_covers_the_listed_risks():
    names = set(FIXTURES)
    for wanted in ("near_tangent_arc_line", "arcs_crossing_shallow", "edges_1_to_3_g", "kissing_corners_same_net",
                   "collinear_overlap", "hole_touches_outer", "neck_split_by_inset", "band_vanishes", "reflex_small_angle",
                   "arc_with_foreign_vertex", "floating_conductor", "nested_holes", "stacked_corner_touch",
                   "port_edge_on_metal_sidewall", "refinement_crossing_band_edge", "near_coincident_layers_2g"):
        assert wanted in names
    assert len(cases()) >= 35


# --------------------------------------------------------------------------- what the fixtures are for


def horizontal_edge_lengths(topo) -> list[float]:
    out = []
    for e in topo.edges.values():
        a, b = topo.vertices[e.v0], topo.vertices[e.v1]
        if (a.x, a.y) != (b.x, b.y):
            out.append(math.hypot(a.x - b.x, a.y - b.y))
    return out


@pytest.mark.parametrize("route", ["A", "B"])
def test_edges_one_to_three_grid_units_long_are_kept_exactly(route):
    lengths = horizontal_edge_lengths(build(FIXTURES["edges_1_to_3_g"].build(route)).topology)
    for wanted in (1.0, 2.0, 3.0, math.sqrt(2)):
        assert any(abs(x - wanted) < 1e-9 for x in lengths), wanted


def test_a_gap_below_a_grid_unit_merges_two_layers_and_a_gap_of_two_makes_a_real_sliver():
    def smallest_horizontal_face(name):
        topo = build(FIXTURES[name].build("B")).topology
        return min(f.twice_xy_area for f in topo.faces.values() if f.kind == "horizontal")

    assert smallest_horizontal_face("near_coincident_layers_sub_grid") > 1_000_000
    assert smallest_horizontal_face("near_coincident_layers_2g") <= 2 * 2 * 8000  # 2 g wide, 8 um long


def test_the_near_tangent_circle_and_line_cross_but_the_exactly_tangent_pair_only_touches():
    near = build(FIXTURES["near_tangent_arc_line"].build("B")).topology
    assert near.counts()["volumes"] == 3  # one conductor volume (bar and disk are one body), substrate, vacuum
    with pytest.raises(NonManifoldError, match=r"bar.*disk|disk.*bar"):
        build(FIXTURES["exactly_tangent_arc_line"].build("B"))
    assert build(FIXTURES["exactly_tangent_arc_line"].build("A")).report is not None  # two sheets meeting at a point are legal


def test_refused_fixtures_name_both_entities_the_net_and_the_location():
    with pytest.raises(NonManifoldError) as err:
        build(FIXTURES["kissing_corners_same_net"].build("B"))
    text = str(err.value)
    assert "padL" in text and "padR" in text and "Net N" in text and "(0.0000, 0.0000)" in text and "vertical edge" in text
    with pytest.raises(ContactError, match=r"N1.*N2|N2.*N1"):
        build(FIXTURES["kissing_corners_different_nets"].build("A"))


def test_a_band_that_vanishes_has_no_innermost_group_and_a_split_neck_gives_two_components():
    result = build(FIXTURES["band_vanishes"].build("A"))
    bands = {name.rsplit("band=", 1)[1] for name in result.topology.groups if "::band=" in name}
    assert bands == {"0", "1"}  # R_0.2 is empty: no band 2
    rows = band_klayout_rows(FIXTURES["neck_split_by_inset"], "A")
    components = {r["margin_units"]: r["components_klayout"] for r in rows if r["region"] == "band_top"}
    assert components[100] == 1 and components[400] == 2
    assert not build(FIXTURES["neck_split_by_inset"].build("A")).report.failures


def test_the_inset_arcs_of_a_split_neck_end_at_their_crossing_without_stubs():
    topo = build(FIXTURES["neck_split_by_inset"].build("A")).topology
    points = {(v.x, v.y) for v in topo.vertices.values()}
    assert (-4312, 0) in points and (4312, 0) in points  # the crossing of the two corner arcs at the neck mouth
    assert not points & {(-4310, 2), (-4310, -2), (4310, 2), (4310, -2)}  # the ends the untrimmed chords used to have


def test_a_foreign_vertex_splits_the_arc_and_the_direct_writer_refuses_the_five_edge_wall():
    spec = FIXTURES["arc_with_foreign_vertex"].build("A")
    sheet = build(spec).topology
    plain = build(ModelSpec(spec.grid, spec.prisms, name="no_family", meta={"route": "A"})).topology
    arcs = lambda topo: sum(1 for e in topo.edges.values() if getattr(e, "curve", None) is not None)
    assert arcs(sheet) == arcs(plain) + 1  # the foreign vertex splits one quarter arc into two arc edges
    topo = build(FIXTURES["arc_with_foreign_vertex"].build("B")).topology
    walls = [f for f in topo.faces.values() if getattr(f, "curved", False)]
    assert sorted(len(f.loops[0]) for f in walls) == [4, 4, 4, 5]
    with pytest.raises(Exception, match="lie on different circles"):
        plan_brep(topo)


def test_a_floating_conductor_is_an_inner_shell_in_route_b_and_an_embedded_sheet_in_route_a():
    topo_b = build(FIXTURES["floating_conductor"].build("B")).topology
    assert sorted(len(v.shells) for v in topo_b.volumes.values() if v.emitted) == [1, 2]
    topo_a = build(FIXTURES["floating_conductor"].build("A")).topology
    assert sum(len(v.embedded_faces) for v in topo_a.volumes.values()) == 1


def test_nested_holes_give_faces_with_several_loops():
    topo = build(FIXTURES["nested_holes"].build("A")).topology
    assert sum(1 for f in topo.faces.values() if len(f.loops) >= 2) >= 3


def test_a_pinched_hole_is_written_as_two_wires_that_share_the_pinch_vertex(tmp_path):
    topo = build(FIXTURES["kissing_corners_same_net"].build("A")).topology
    pinched = [f for f in topo.faces.values() if len(f.loops) == 2 and len(f.loops[1]) == 8]
    assert len(pinched) == 1  # the hole boundary of two squares touching at a corner: one cycle, eight edges
    plan = plan_brep(topo)
    assert max(plan.face_nwires) == 3  # outer wire + one wire per lobe
    if OCP_AVAILABLE:
        assert run_fixture(FIXTURES["kissing_corners_same_net"], "A", tmp_path)["ok"]


def test_a_port_edge_on_a_metal_sidewall_keeps_its_group_in_both_routes():
    for route in ("A", "B"):
        assert "PORT::p1" in build(FIXTURES["port_edge_on_metal_sidewall"].build(route)).topology.groups


def test_a_refinement_curve_crossing_band_edges_is_split_at_every_crossing():
    topo = build(FIXTURES["refinement_crossing_band_edge"].build("A")).topology
    pieces = sum(len(f.refinement_edges) for f in topo.faces.values())
    assert pieces == 5  # 2 outer bands + 2 mid bands + the core, crossed at x = -9.5, -8, 8, 9.5


# --------------------------------------------------------------------------- regressions found by the catalogue


def test_a_refinement_curve_ending_on_a_boundary_keeps_that_vertex_on_the_boundary_edge():
    from sgbproto.extreme_cases import GRID, chip, metal, spec_of
    from sgbproto.records import CurveFamilyRecord, Lineage, LineCarrier, z_value
    from sgbproto.scenes import rect_um

    carrier = LineCarrier("ref0", (-10_000, 500), (0, 500), Lineage("family_member", family_id="ref", member="ref0"))
    ref = CurveFamilyRecord("ref", "refinement", (carrier,), (z_value(0),), "refinement_only")
    topo = build(spec_of("ref_touch", "A", chip() + [metal("pad", "N", rect_um(-10, -5, 10, 5), "A")], (ref,))).topology
    host = next(f for f in topo.faces.values() if f.refinement_edges)
    boundary = {v for lp in host.loops for eid, _ in lp for v in (topo.edges[eid].v0, topo.edges[eid].v1)}
    edge = topo.edges[host.refinement_edges[0]]
    assert edge.v0 in boundary  # the curve's end on the pad boundary is a vertex of the boundary edge, not a loose point
    assert GRID.g == 0.001


def test_the_load_check_of_physical_groups_does_not_scan_every_entity_per_group(tmp_path):
    """Regression: the group comparison was groups x entities; 300 prisms (hundreds of groups) must verify at once."""
    import time

    import gmsh

    from sgbproto.brep_writer import verify_loaded, write_xao
    from sgbproto.scenes import random_layout

    topo = build(random_layout(300, seed=2, route="B")).topology
    written = write_xao(topo, tmp_path / "m.xao", sidecar=False)
    gmsh.initialize(["-noenv"])
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(written.path))
        gmsh.model.occ.synchronize()
        t0 = time.perf_counter()
        report = verify_loaded(topo, written.plan, mesh_size=None, groups=True)
    finally:
        gmsh.finalize()
    assert report["ok"] and report["groups"]["loaded"] >= 300
    assert report["timings_s"]["physical_groups"] < 0.5  # about 0.01 s linear; the quadratic scan took seconds at 1000
    assert time.perf_counter() - t0 < 10
