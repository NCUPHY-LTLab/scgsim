"""Arc carriers, the polyline witness, merging back into arc edges and realisation (design decision D3)."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest

shapely = pytest.importorskip("shapely", reason="the area oracle needs shapely")

from sgb_scenes import LINEAGE, disc_model, pad_in_ground  # noqa: E402
from shapely.geometry import Point as ShapelyPoint  # noqa: E402
from shapely.geometry import box  # noqa: E402

from scgsim.geometry.exact.carriers import Lineage, circle_arcs  # noqa: E402
from scgsim.geometry.sgb.arrangement.arcs import (  # noqa: E402
    face_exact_area,
    realise,
    sampled_crossings,
)
from scgsim.geometry.sgb.arrangement.derivation.labelled import derive_from_catalogue  # noqa: E402

TESTS = Path(__file__).resolve().parent


def sheet_faces(result):
    """The level layer at z = 0 and the faces of the sheet of Net N."""
    layer = result.level_layers[1]
    faces = [
        fi
        for fi in range(1, len(layer.arr.faces))
        if layer.labels[fi].sheet is not None and layer.labels[fi].sheet.net == "N"
    ]
    return layer, faces


def test_witness_chords_have_sagitta_at_most_a_quarter_grid_unit_and_end_exactly_on_the_arc_ends():
    """Witness chords have sagitta at most a quarter grid unit and end exactly on the arc ends."""
    for radius in (50, 800, 8000):
        arc = circle_arcs("c", (3, -7), radius, True, LINEAGE)[0]
        points = arc.witness_points()
        assert points[0] == (3 + radius, -7) and points[-1] == (3, -7 + radius)
        sagitta = max(
            radius - math.hypot((p[0] + q[0]) / 2 - 3.0, (p[1] + q[1]) / 2 + 7.0)
            for p, q in zip(points, points[1:], strict=False)
        )
        # the ideal chord is at most g/4; rounding both witness end points adds up to g/2 per side
        assert sagitta <= 0.25 + 0.75


def test_a_full_circle_is_four_quarter_arcs_so_no_edge_is_a_loop():
    """A full circle is four quarter arcs so no edge is a loop."""
    arcs = circle_arcs("c", (0, 0), 1000, False, LINEAGE)
    assert [a.start for a in arcs] == [a.end for a in arcs[-1:] + arcs[:-1]]
    assert all(
        not a.ccw and abs(float(a.sweep()[1]) + math.pi / 2) < 1e-12 for a in arcs
    )


def test_witness_pieces_merge_back_into_exactly_one_edge_per_quarter_arc():
    """Witness pieces merge back into exactly one edge per quarter arc."""
    result = derive_from_catalogue(disc_model(3000))
    layer, faces = sheet_faces(result)
    arc_edges = [
        e
        for e, c in zip(layer.arr.edges, layer.edge_curve, strict=True)
        if c is not None
    ]
    assert len(arc_edges) == 4
    assert all(
        c.piece is None for e in layer.arr.edges for c in e.carriers
    )  # no witness piece survives
    assert len(faces) == 1
    assert len(layer.arr.cycles[layer.arr.faces[faces[0]].outer].half_edges) == 4


def test_realised_arc_passes_through_the_snapped_end_points_with_the_carrier_radius_and_side():
    """Realised arc passes through the snapped end points with the carrier radius and side."""
    root = circle_arcs("c", (0, 0), 5000, True, LINEAGE)[0]
    a, b = (5000, 0), (3536, 3535)
    realised = realise(root, root, a, b)
    for p in (a, b):
        distance = math.hypot(p[0] - realised.centre[0], p[1] - realised.centre[1])
        assert distance == pytest.approx(realised.radius, abs=1e-9)
    assert realised.radius == pytest.approx(5000.0)
    # the carrier centre is on the left of a -> b, so the arc bulges right
    assert realised.ccw and not realised.bulge_left
    assert (
        realised.deviation < 0.75
    )  # snapped end points move the circle by well under the snap bound


@pytest.mark.parametrize("radius", [3000, 4321])
def test_circle_face_area_is_pi_r_squared_for_exact_end_points(radius):
    """Circle face area is pi r squared for exact end points."""
    result = derive_from_catalogue(disc_model(radius))
    layer, faces = sheet_faces(result)
    area = face_exact_area(layer.arr, layer.edge_curve, faces[0])
    assert area == pytest.approx(math.pi * radius**2, rel=1e-9)


def test_circle_overlapped_by_a_bar_matches_the_shapely_union_within_the_snap_bound():
    """Two entities of one Net keep their seam (D5); together the two faces cover the union."""
    result = derive_from_catalogue(
        disc_model(3000, extra_rect=(-1000, -5000, 1000, 5000))
    )
    layer, faces = sheet_faces(result)
    # the bar (higher priority) owns the overlap and splits the disc into its two lobes
    assert len(faces) == 3
    union = ShapelyPoint(0, 0).buffer(3000, 4096).union(box(-1000, -5000, 1000, 5000))
    measured = sum(face_exact_area(layer.arr, layer.edge_curve, f) for f in faces)
    assert (
        abs(measured - union.area) <= union.length * 1.0
    )  # each boundary point moves by < 1 grid unit
    assert abs(measured - union.area) / union.area < 1e-3
    assert max(c.deviation for c in layer.edge_curve if c is not None) <= 1.0


@pytest.mark.parametrize("route", ["A", "B"])
def test_realised_arcs_never_cross_other_edges_by_dense_sampling(route):
    """Realised arcs never cross other edges by dense sampling."""
    result = derive_from_catalogue(pad_in_ground(route))
    checked = 0
    for layer in (*result.slabs, *result.level_layers):
        checked += sum(c is not None for c in layer.edge_curve)
        assert sampled_crossings(layer.arr, layer.edge_curve, step=1.0) == []
    assert (
        checked >= 5
    )  # the hole (four quarters) and the rounded corner, at least once


def test_pad_area_matches_the_analytic_value_with_a_true_circle_and_rounded_corner():
    """Pad area matches the analytic value with a true circle and rounded corner."""
    pad_area_um2 = (
        100 * 60 - 20 * 15 - 12 * 12 / 2 - 25 * (1 - math.pi / 4) - math.pi * 64
    )
    result = derive_from_catalogue(pad_in_ground("A"))
    layer = result.level_layers[1]
    faces = [
        fi
        for fi in range(1, len(layer.arr.faces))
        if layer.labels[fi].sheet is not None and layer.labels[fi].sheet.name == "pad"
    ]
    assert len(faces) == 1
    area = (
        face_exact_area(layer.arr, layer.edge_curve, faces[0]) * 1e-6
    )  # grid^2 -> um^2
    assert area == pytest.approx(pad_area_um2, rel=1e-9)


def test_arc_edges_keep_the_lineage_of_the_source_arc():
    """Arc edges keep the lineage of the source arc."""
    result = derive_from_catalogue(pad_in_ground("A"))
    layer = result.level_layers[1]
    arc_sources = {
        source
        for edge_sources, curve in zip(
            layer.edge_sources, layer.edge_curve, strict=True
        )
        if curve is not None
        for source in edge_sources
    }
    assert {s.split("/q")[0] for s in arc_sources} == {"pad/hole", "pad/corner"}
    lineage = result.arcs["pad/hole/q0"].lineage
    assert lineage == Lineage("polygon_edge", "pad", 0, 1, 0)


def test_arc_example_ids_are_identical_for_every_hash_seed():
    """Arc example ids are identical for every hash seed."""
    catalogue = pad_in_ground("B")
    code = (
        f"import sys; sys.path.insert(0, {str(TESTS)!r})\n"
        "from scgsim.geometry.sgb.arrangement.derivation.labelled import derive_from_catalogue\n"
        "from sgb_scenes import pad_in_ground\n"
        "print(derive_from_catalogue(pad_in_ground('B')).identity_digest())\n"
    )
    outputs = {
        subprocess.run(
            [sys.executable, "-c", code],
            env={"PYTHONHASHSEED": seed, "PATH": ""},
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        for seed in ("0", "7")
    }
    assert outputs == {derive_from_catalogue(catalogue).identity_digest()}


def test_without_the_rounded_corner_only_the_circular_hole_remains_an_arc():
    """Without the rounded corner only the circular hole remains an arc."""
    result = derive_from_catalogue(pad_in_ground("A", rounded_corner=False))
    arcs = [c for c in result.level_layers[1].edge_curve if c is not None]
    assert arcs
    assert {c.root.split("/q")[0] for c in arcs} == {"pad/hole"}
