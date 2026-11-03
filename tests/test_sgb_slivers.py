"""Sliver report: measurements with location and lineage; no geometry change, no failure."""

from __future__ import annotations

import math

import pytest
from sgb_inputs import gds_pad_in_ground
from sgb_scenes import pad_in_ground, random_layout

from scgsim.geometry.sgb.model import build_conformal_model, build_model_from_catalogue
from scgsim.geometry.sgb.slivers.feature_size import Prim, prim_distance
from scgsim.geometry.sgb.slivers.report import sliver_report


def test_report_has_sorted_measurements_with_location_and_lineage():
    """Rows are worst first; each has a location and the carriers or entities it came from."""
    model = build_model_from_catalogue(pad_in_ground("B"), "B")
    before = model.identity_digest()
    report = sliver_report(model, top=5)
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
    narrow = sliver_report(model, top=50)["narrowest_faces"]
    port = next(r for r in narrow if r["group"] == "PORT::p1")
    assert math.isclose(port["value_um"], 10.0, rel_tol=1e-9)
    assert math.isclose(
        port["width_2a_over_p_um"], 2 * (30 * 10) / (2 * (30 + 10)), rel_tol=1e-9
    )


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
    report = sliver_report(model)
    assert report["smallest_angles"][0]["value_deg"] < 90.0
    drift = report["snap_drift"]
    assert drift["edges_measured"] > 0 and drift["max_grid_units"] <= 1.0
    assert (
        report["summary"]["smallest_angle_deg"]
        == report["smallest_angles"][0]["value_deg"]
    )


def test_arc_edges_are_reported_with_their_deviation_from_the_exact_carrier():
    """The worst arc deviations come first."""
    report = sliver_report(build_model_from_catalogue(pad_in_ground("A"), "A"), top=3)
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
        report = sliver_report(model)
        assert [s["thickness_um"] for s in report["thin_slabs"]] == [
            pytest.approx(0.0004)
        ]
        assert report["summary"]["thin_slab_count"] == 1
        assert report["shortest_edges"][0]["value_um"] == pytest.approx(0.0004)
