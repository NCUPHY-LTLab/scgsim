"""Sliver report: measurements with location and lineage, no geometry change."""

from __future__ import annotations

import math

import pytest

from sgbproto.pipeline import build, ids_digest
from sgbproto.scenes import pad_in_ground, random_layout
from sgbproto.sliver import sliver_report


def test_report_has_sorted_measurements_with_location_and_lineage():
    result = build(pad_in_ground("B"))
    before = ids_digest(result)
    report = sliver_report(result, top=5)
    assert ids_digest(result) == before  # report only
    for key, field in (("shortest_edges", "value_um"), ("narrowest_faces", "value_um"), ("smallest_angles", "value_deg")):
        rows = report[key]
        assert 0 < len(rows) <= 5
        values = [r[field] for r in rows]
        assert values == sorted(values)
        assert all(len(r["location_um"]) in (2, 3) and r["sources"] for r in rows)
    # the shortest edge of the Route B example is a 0.1 um vertical wall segment (the port plane cuts the metal)
    shortest = report["shortest_edges"][0]
    assert shortest["kind"] == "vertical" and math.isclose(shortest["value_um"], 0.1, rel_tol=1e-6)
    assert set(shortest["sources"]) <= {"pad", "ground", "p1"}  # a wall column inherits its walls' entities


def test_narrowest_face_width_is_the_local_feature_size_on_exact_carriers():
    result = build(pad_in_ground("A"))
    narrow = sliver_report(result, top=50)["narrowest_faces"]
    port = next(r for r in narrow if r["group"] == "PORT::p1")
    assert math.isclose(port["value_um"], 10.0, rel_tol=1e-9)  # 30 x 10 um rectangle: distance between its long sides
    assert math.isclose(port["width_2a_over_p_um"], 2 * (30 * 10) / (2 * (30 + 10)), rel_tol=1e-9)  # kept as extra field


def test_feature_size_of_a_ring_between_concentric_arcs_is_the_radial_gap_not_a_chord_quantity():
    """A band of 0.1 um around the circular hole: 2A/P on the chord polygon misstates it, the feature size does not."""
    result = build(pad_in_ground("A", bands_um=(0.1, 0.5, 2.0)))
    rows = sliver_report(result, top=500)["narrowest_faces"]
    ring = [r for r in rows if r["group"] == "SHEET::pad::band=0" and r["bounded"] is False]
    assert min(r["value_um"] for r in ring) == pytest.approx(0.1, abs=2e-3)


def test_feature_size_primitives_arc_to_arc_and_arc_to_line():
    from sgbproto.feature_size import Prim, prim_distance

    quarter = lambda r: Prim((r, 0.0), (0.0, r), (0.0, 0.0), r, 0.0, math.pi / 2)
    assert prim_distance(quarter(100.0), quarter(130.0)) == pytest.approx(30.0)  # concentric
    line = Prim((140.0, -10.0), (140.0, 50.0))
    assert prim_distance(quarter(100.0), line) == pytest.approx(40.0)  # line beside the arc: nearest point is interior


def test_smallest_angle_reflects_oblique_geometry_and_drift_is_reported():
    result = build(random_layout(30, seed=3, route="A", overlap_fraction=1.0))
    report = sliver_report(result)
    assert report["smallest_angles"][0]["value_deg"] < 90.0
    drift = report["snap_drift"]
    assert drift["edges_measured"] > 0 and drift["max_grid_units"] <= 1.0
    assert report["summary"]["smallest_angle_deg"] == report["smallest_angles"][0]["value_deg"]


def test_arc_edges_are_reported_with_their_deviation_from_the_exact_carrier():
    report = sliver_report(build(pad_in_ground("A", bands_um=(0.1, 0.5, 2.0))), top=3)
    arcs = report["arc_edges"]
    assert arcs["count"] > 40 and arcs["max_deviation_grid"] <= 1.0
    assert [w["value_grid_units"] for w in arcs["worst"]] == sorted((w["value_grid_units"] for w in arcs["worst"]), reverse=True)
    assert report["summary"]["max_arc_deviation_grid"] == arcs["max_deviation_grid"]
