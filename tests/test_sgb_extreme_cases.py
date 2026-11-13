"""The minimal extreme-case fixtures: each passes every structural check or is refused early with the expected error.

Every fixture is a few prisms (under 100 faces), so the validators run in milliseconds.  The first test
is the catalogue itself; the others pin what each fixture is *for* (so a fixture cannot silently stop
testing its risk) and the regressions the catalogue found.
"""

from __future__ import annotations

import math

import jsonschema
import pytest
from sgb_catalogue_report import measure
from sgb_extreme_cases import (
    FIXTURES,
    build_model,
    cases,
    catalogue_of,
    chip,
    metal,
    refinement,
)
from sgb_scenes import rect_um

from scgsim.contracts.sgb_sidecar import read_sidecar, sidecar_schema
from scgsim.geometry.sgb.build import emit_model
from scgsim.geometry.sgb.errors import ContactError, NonManifoldError
from scgsim.geometry.sgb.model import build_model_from_catalogue
from scgsim.geometry.sgb.exact.grid import exact_z
from scgsim.geometry.sgb.checks.slivers.report import sliver_report
from scgsim.geometry.sgb.checks.verification.loaded import verify_xao


def build(name: str, route: str):
    """Build one fixture under a route, with its band and refinement requests."""
    return build_model(FIXTURES[name], route)


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
            build_model(fixture, route)
        return
    model = build_model(fixture, route)
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


# --------------------------------------------------------------------------- emission, load check, OCCT, quality and drift


def accepted_cases():
    """The (fixture, route) pairs that build (the refused ones never reach emission)."""
    return [(f, r) for f, r in cases() if r not in f.expect_error]


@pytest.mark.parametrize(
    ("fixture", "route"),
    accepted_cases(),
    ids=[f"{f.name}-{r}" for f, r in accepted_cases()],
)
def test_every_fixture_emits_files_gmsh_loads_and_a_sidecar_that_validates(
    fixture, route, tmp_path
):
    """The whole path for each fixture: write, validate the sidecar against its schema, load in Gmsh and check."""
    model = build_model(fixture, route)
    artifacts = emit_model(model, tmp_path)
    sidecar = read_sidecar(artifacts.sidecar)
    assert not list(
        jsonschema.Draft202012Validator(sidecar_schema()).iter_errors(sidecar)
    )
    report = verify_xao(artifacts.xao, sidecar)
    assert report.ok, report.problems
    assert sidecar["summary"]["counts"]["faces"] == model.counts[
        "faces"
    ] - _unemitted_faces(model)


def _unemitted_faces(model) -> int:
    """Faces that bound no emitted volume (Route A and B conductor interiors): planned, never written."""
    return sum(
        1
        for f in model.topology.faces.values()
        if not model.routed.is_emitted_face(f.neg_volume, f.pos_volume)
    )


@pytest.mark.occt
@pytest.mark.parametrize(
    ("fixture", "route"),
    accepted_cases(),
    ids=[f"{f.name}-{r}" for f, r in accepted_cases()],
)
def test_every_fixture_passes_the_official_occt_checks(fixture, route, tmp_path):
    """BRepCheck, the Boolean argument analyzer (with self-interference) and ShapeAnalysis, against the plan."""
    pytest.importorskip("OCP")
    from scgsim.geometry.sgb.output.emission.plan import plan_brep  # noqa: PLC0415 -- used by this test only
    from scgsim.geometry.sgb.checks.verification.occt import check_file  # noqa: PLC0415 -- needs OCP

    model = build_model(fixture, route)
    artifacts = emit_model(model, tmp_path)
    plan = plan_brep(model.topology, model.routed)
    for path in (artifacts.brep, artifacts.xao):
        report = check_file(path, model.topology, plan, self_interference=True)
        assert report.ok, f"{fixture.name} {route} {path.suffix}: {report.problems}"


@pytest.mark.parametrize(
    ("fixture", "route"),
    accepted_cases(),
    ids=[f"{f.name}-{r}" for f, r in accepted_cases()],
)
def test_snap_drift_stays_within_one_grid_unit_on_every_fixture(fixture, route):
    """Reopen trigger of decision D7: drift above 1 g on the fixture catalogue would revisit bounded-drift snap rounding."""
    summary = sliver_report(build_model(fixture, route))["summary"]
    assert summary["max_snap_drift_grid"] <= 1.0, (
        f"{fixture.name} {route}: {summary['max_snap_drift_grid']} g"
    )
    assert summary["max_arc_deviation_grid"] <= 1.0


def test_the_catalogue_records_mesh_quality_and_drift_per_fixture(tmp_path):
    """The record that D2 and D7 are judged on: a coarse mesh per fixture, with ``minSICN`` and ``gamma`` percentiles."""
    row = measure(
        build_model(FIXTURES["near_tangent_arc_line"], "B"), tmp_path, with_occt=False
    )
    assert row["load_check"] == "ok" and row["mesh"] == "ok"
    for key in ("minSICN_min", "minSICN_p1", "gamma_min", "gamma_p1", "max_snap_drift_grid", "max_arc_deviation_grid",
                "narrowest_face_um", "tets"):  # fmt: skip
        assert row[key] is not None, key
    assert (
        0 < row["minSICN_min"] <= row["minSICN_p1"] <= 1
        and 0 < row["gamma_min"] <= row["gamma_p1"] <= 1
    )
    baseline = measure(
        build_model(FIXTURES["baseline"], "A"), tmp_path / "baseline", with_occt=False
    )
    assert (
        baseline["minSICN_min"] > 0.05
    )  # a clean box of two prisms and a sheet meshes well
