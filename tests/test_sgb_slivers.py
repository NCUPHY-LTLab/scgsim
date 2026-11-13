"""Sliver report: measurements with location and lineage; no geometry change, no failure."""

from __future__ import annotations

import math

import pytest
import sgb_supported_cells as disk_qubit
from sgb_inputs import gds_pad_in_ground
from sgb_scenes import pad_in_ground, random_layout
from sgb_supported_kit import catalogue, type_bands

from scgsim.geometry.sgb.checks.slivers.feature_size import Prim, prim_distance
from scgsim.geometry.sgb.checks.slivers.report import sliver_report
from scgsim.geometry.sgb.model import build_conformal_model, build_model_from_catalogue


def test_report_has_sorted_measurements_with_location_and_lineage():
    """Rows are worst first; each has a location and the carriers or entities it came from."""
    model = build_model_from_catalogue(pad_in_ground("B"), "B")
    before = model.identity_digest()
    report = sliver_report(model.labelled, model.topology, model.routed, top=5)
    assert model.identity_digest() == before  # report only
    for key, field in (
        ("shortest_edges", "value_um"),
        ("narrowest_faces", "value_um"),
        ("smallest_angles", "value_deg"),
    ):
        rows = report[key]
        assert 0 < len(rows) <= 5
        values = [r[field] for r in rows]
        assert values == sorted(values)
        assert all(len(r["location_um"]) in (2, 3) and r["sources"] for r in rows)
    # the shortest edge of the Route B example is a 0.1 um vertical wall segment (the port plane cuts the metal)
    shortest = report["shortest_edges"][0]
    assert shortest["kind"] == "vertical" and math.isclose(
        shortest["value_um"], 0.1, rel_tol=1e-6
    )
    assert set(shortest["sources"]) <= {
        "pad",
        "ground",
        "p1",
    }  # a wall column inherits its walls' entities


def test_narrowest_face_width_is_the_local_feature_size_on_exact_carriers():
    """A 30 x 10 um port: the feature size is the distance between its long sides, not 2A/P."""
    model = build_model_from_catalogue(pad_in_ground("A"), "A")
    narrow = sliver_report(model.labelled, model.topology, model.routed, top=50)[
        "narrowest_faces"
    ]
    port = next(r for r in narrow if r["group"] == "PORT::p1")
    assert math.isclose(port["value_um"], 10.0, rel_tol=1e-9)
    assert math.isclose(
        port["width_2a_over_p_um"], 2 * (30 * 10) / (2 * (30 + 10)), rel_tol=1e-9
    )


@pytest.mark.parametrize("name", ["cell07_cpw45_ro", "cell07_cpw45_xy"])
def test_a_triangle_face_reports_its_smallest_altitude_not_zero(name):
    """The 45-degree CPW leaves straight triangles in the die corners; each reports 2 A / longest side.

    A loop runs some of a triangle's edges backwards; the width must still pair each end point
    with its own vertex (issue #129: a reversed edge once measured a vertex against itself, 0.0).
    """
    model = build_model_from_catalogue(
        catalogue(disk_qubit.SCENES[name]()), "B", insets=type_bands()
    )
    faces = sliver_report(
        model.labelled,
        model.topology,
        model.routed,
        top=len(model.topology.faces),
        drift=False,
    )
    widths = {row["id"]: row["value_um"] for row in faces["narrowest_faces"]}
    assert all(width > 0 for width in widths.values())
    triangles = _straight_triangles(model)
    assert len(triangles) >= 4  # two ground corners, each at z = 0 and at the metal top
    for fid, corners in triangles.items():
        twice_area = abs(
            (corners[1][0] - corners[0][0]) * (corners[2][1] - corners[0][1])
            - (corners[2][0] - corners[0][0]) * (corners[1][1] - corners[0][1])
        )
        longest = max(math.dist(corners[i], corners[i - 1]) for i in range(3))
        assert widths[fid] == pytest.approx(twice_area / longest, abs=1e-6)


def _straight_triangles(model):
    """Horizontal faces bounded by three straight edges, as their corners in micrometres."""
    topo, g = model.topology, model.labelled.catalogue.grid.size_um
    triangles = {}
    for fid, face in topo.faces.items():
        loop = face.loops[0]
        if face.kind != "horizontal" or len(face.loops) != 1 or len(loop) != 3:
            continue
        edges = [(topo.edges[eid], sign) for eid, sign in loop]
        if all(edge.curve is None for edge, _ in edges):
            starts = [topo.vertices[e.v0 if s > 0 else e.v1] for e, s in edges]
            triangles[fid] = [(v.x * g, v.y * g) for v in starts]
    return triangles


def test_feature_size_primitives_arc_to_arc_and_arc_to_line():
    """Distances between concentric arcs and between an arc and a line beside it."""

    def quarter(r):
        """Quarter."""
        return Prim((r, 0.0), (0.0, r), (0.0, 0.0), r, 0.0, math.pi / 2)

    assert prim_distance(quarter(100.0), quarter(130.0)) == pytest.approx(30.0)
    line = Prim((140.0, -10.0), (140.0, 50.0))
    assert prim_distance(quarter(100.0), line) == pytest.approx(40.0)


def test_smallest_angle_reflects_oblique_geometry_and_drift_is_reported():
    """Rotated features give acute wedges; the snap drift stays within the measured bound."""
    model = build_model_from_catalogue(
        random_layout(30, seed=3, route="A", overlap_fraction=1.0), "A"
    )
    report = sliver_report(model.labelled, model.topology, model.routed)
    assert report["smallest_angles"][0]["value_deg"] < 90.0
    drift = report["snap_drift"]
    assert drift["edges_measured"] > 0 and drift["max_grid_units"] <= 1.0
    assert (
        report["summary"]["smallest_angle_deg"]
        == report["smallest_angles"][0]["value_deg"]
    )


def test_arc_edges_are_reported_with_their_deviation_from_the_exact_carrier():
    """The worst arc deviations come first."""
    model = build_model_from_catalogue(pad_in_ground("A"), "A")
    report = sliver_report(model.labelled, model.topology, model.routed, top=3)
    arcs = report["arc_edges"]
    assert arcs["count"] >= 5 and arcs["max_deviation_grid"] <= 1.0
    worst = [w["value_grid_units"] for w in arcs["worst"]]
    assert worst == sorted(worst, reverse=True)
    assert report["summary"]["max_arc_deviation_grid"] == arcs["max_deviation_grid"]


def test_slabs_thinner_than_the_grid_are_reported_never_a_failure(tmp_path):
    """No Z threshold: a 0.4 nm metal builds under every route; the report lists the slab and validation warns."""
    effective = gds_pad_in_ground(tmp_path, thickness_um=0.0004, with_port=False)
    for route in ("B", "C"):
        model = build_conformal_model(
            effective.build_input, effective.representation, route
        )
        assert not model.report.failures
        warning = next(
            c for c in model.report.checks if c.name == "slabs_thinner_than_g_are_kept"
        )
        assert warning.warning and not warning.passed and "0.4" in warning.detail
        vertical = next(
            c
            for c in model.report.checks
            if c.name == "vertical_edges_shorter_than_g_are_kept"
        )
        assert vertical.warning and not vertical.passed
        report = sliver_report(model.labelled, model.topology, model.routed)
        assert [s["thickness_um"] for s in report["thin_slabs"]] == [
            pytest.approx(0.0004)
        ]
        assert report["summary"]["thin_slab_count"] == 1
        assert report["shortest_edges"][0]["value_um"] == pytest.approx(0.0004)
