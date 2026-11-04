"""Arc carriers, the polyline witness, merging back into arc edges, and realisation (brief section 3, D3)."""

from __future__ import annotations

import math
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest
import shapely
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import box

from sgbproto.arcs import face_exact_area, realise, sampled_crossings
from sgbproto.pipeline import build, id_sets, ids_digest
from sgbproto.records import (
    ArcCarrier, Footprint, Grid, LineEdge, Lineage, ModelSpec, PrismRecord, circle_arcs, z_value,
)
from sgbproto.scenes import _chip_prisms, pad_in_ground

ROOT = Path(__file__).resolve().parents[1]
LINEAGE = Lineage("polygon_edge", "disc")


def disc_model(radius: int, extra_rect=None, centre=(0, 0)) -> ModelSpec:
    """A circular sheet conductor (four quarter arcs), optionally overlapped by a rectangle of the same Net."""
    grid = Grid()
    prisms = _chip_prisms(grid, 10.0)
    cx, cy = centre
    footprint, arcs = Footprint.from_primitives(circle_arcs("disc", centre, radius, True, LINEAGE))
    prisms.append(PrismRecord("disc", footprint, z_value(0), z_value(0), "Al", "conductor", net_id="N", arcs=arcs))
    if extra_rect:
        x0, y0, x1, y1 = extra_rect
        prisms.append(PrismRecord("bar", Footprint.from_rings([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]),
                                  z_value(0), z_value(0), "Al", "conductor", net_id="N"))
    return ModelSpec(grid, tuple(prisms), name="disc", meta={"route": "A"})


def sheet_faces(result):
    layer = result.level_layers[1]
    return layer, [fi for fi in range(1, len(layer.arr.faces))
                   if layer.labels[fi].sheet is not None and layer.labels[fi].sheet.name == "N"]


def test_witness_chords_have_sagitta_at_most_a_quarter_grid_unit_and_end_exactly_on_the_arc_ends():
    for radius in (50, 800, 8000):
        arc = circle_arcs("c", (3, -7), radius, True, LINEAGE)[0]
        pts = arc.witness_points()
        assert pts[0] == (3 + radius, -7) and pts[-1] == (3, -7 + radius)
        centre = (3.0, -7.0)
        sagitta = max(radius - math.hypot((p[0] + q[0]) / 2 - centre[0], (p[1] + q[1]) / 2 - centre[1])
                      for p, q in zip(pts, pts[1:]))
        assert sagitta <= 0.25 + 0.75  # ideal chord <= g/4, plus the half-unit rounding of both witness end points


def test_a_full_circle_is_four_quarter_arcs_so_no_edge_is_a_loop():
    arcs = circle_arcs("c", (0, 0), 1000, False, LINEAGE)
    assert [a.start for a in arcs] == [a.end for a in arcs[-1:] + arcs[:-1]]
    assert all(not a.ccw and abs(a.sweep()[1] + math.pi / 2) < 1e-12 for a in arcs)


def test_witness_pieces_merge_back_into_exactly_one_edge_per_quarter_arc():
    result = build(disc_model(3000))
    layer, faces = sheet_faces(result)
    arc_edges = [e for e, c in zip(layer.arr.edges, layer.edge_curve) if c is not None]
    assert len(arc_edges) == 4
    assert all("#" not in c for e in layer.arr.edges for c in e.carriers)  # no witness piece survives
    assert len(faces) == 1 and len(layer.arr.cycles[layer.arr.faces[faces[0]].outer].half_edges) == 4


def test_realised_arc_passes_through_the_snapped_end_points_with_the_carrier_radius_and_side():
    root = circle_arcs("c", (0, 0), 5000, True, LINEAGE)[0]
    a, b = (5000, 0), (3536, 3535)
    r = realise(root, root, a, b)
    for p in (a, b):
        assert math.hypot(p[0] - r.centre[0], p[1] - r.centre[1]) == pytest.approx(r.radius, abs=1e-9)
    assert r.radius == pytest.approx(5000.0)
    assert r.ccw and not r.bulge_left  # the carrier centre is on the left of a -> b, so the arc bulges right
    assert r.deviation < 0.75  # snapped end points move the circle by well under the snap bound


def test_circle_face_area_is_pi_r_squared_for_exact_end_points():
    for radius in (3000, 4321):
        result = build(disc_model(radius))
        layer, faces = sheet_faces(result)
        area = face_exact_area(layer.arr, layer.edge_curve, faces[0])
        assert area == pytest.approx(math.pi * radius**2, rel=1e-9)


def test_circle_overlapped_by_a_bar_matches_the_shapely_union_within_the_snap_bound():
    spec = disc_model(3000, extra_rect=(-1000, -5000, 1000, 5000))  # crosses the circle at snapped points
    result = build(spec)
    layer, faces = sheet_faces(result)
    assert len(faces) == 1
    union = ShapelyPoint(0, 0).buffer(3000, 4096).union(box(-1000, -5000, 1000, 5000))
    measured = face_exact_area(layer.arr, layer.edge_curve, faces[0])
    perimeter = union.length
    assert abs(measured - union.area) <= perimeter * 1.0  # each boundary point moves by < 1 grid unit
    assert abs(measured - union.area) / union.area < 1e-3
    deviations = [c.deviation for c in layer.edge_curve if c is not None]
    assert max(deviations) <= 1.0


@pytest.mark.parametrize("route", ["A", "B"])
def test_realised_arcs_never_cross_other_edges_by_dense_sampling(route):
    result = build(pad_in_ground(route, bands_um=(0.1, 0.5, 2.0)))
    checked = 0
    for layer in result.slabs + result.level_layers:
        checked += sum(c is not None for c in layer.edge_curve)
        assert sampled_crossings(layer.arr, layer.edge_curve, step=1.0) == []  # one-unit sampling
    assert checked > 40


def test_arc_example_ids_are_identical_for_every_tile_and_worker_count_and_hash_seed():
    spec = pad_in_ground("B", bands_um=(0.1, 0.5, 2.0))
    reference = id_sets(build(spec))
    for tiles, workers in ((2, 1), (3, 1), (4, 2)):
        assert id_sets(build(spec, tiles=tiles, workers=workers)) == reference
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from sgbproto.pipeline import build, ids_digest\nfrom sgbproto.scenes import pad_in_ground\n"
        "print(ids_digest(build(pad_in_ground('B', bands_um=(0.1, 0.5, 2.0)))))\n"
    ) % str(ROOT)
    outputs = {
        subprocess.run([sys.executable, "-c", code], env={"PYTHONHASHSEED": seed, "PATH": ""}, capture_output=True,
                       text=True, check=True).stdout.strip()
        for seed in ("0", "7")
    }
    assert outputs == {ids_digest(build(spec))}


def test_pad_area_and_volume_match_the_analytic_values_with_a_true_circle_and_rounded_corner():
    pad_area_um2 = 100 * 60 - 20 * 15 - 12 * 12 / 2 - 25 * (1 - math.pi / 4) - math.pi * 64
    result = build(pad_in_ground("A"))
    layer = result.level_layers[1]
    pad_faces = [fi for fi in range(1, len(layer.arr.faces))
                 if layer.labels[fi].sheet is not None and layer.labels[fi].sheet.name == "pad"]
    assert face_exact_area(layer.arr, layer.edge_curve, pad_faces[0]) * 1e-6 == pytest.approx(pad_area_um2, rel=1e-9)
    b = build(pad_in_ground("B"))
    pad_volume = next(float(s.signed_volume) for v in b.topology.volumes.values() if v.domain.name == "pad" for s in v.shells)
    assert pad_volume * 1e-6 == pytest.approx(pad_area_um2 * 0.2, rel=1e-9)  # grid^2 * um -> um^3


def test_polygon_hole_variant_still_works_and_has_no_arc_edges_for_the_hole():
    result = build(pad_in_ground("A", hole_sides=16, rounded_corner=False))
    assert all(c is None for layer in result.level_layers for c in layer.edge_curve)
