"""Surface-EPR inset bands (brief sections 5 and 6): exact corner rule, distance origins, KLayout cross-check."""

from __future__ import annotations

import math
from fractions import Fraction

import pytest

from sgbproto.emit import emit
from sgbproto.klayout_check import comparison_rows, exact_band_areas
from sgbproto.pipeline import build
from sgbproto.scenes import pad_in_ground

MARGINS = (0.1, 0.5, 2.0)
PAD_AREA_UM2 = 100 * 60 - 20 * 15 - 12 * 12 / 2 - 25 * (1 - math.pi / 4) - math.pi * 64
GAP_AREA_UM2 = 160 * 120 - (PAD_AREA_UM2 + math.pi * 64)


@pytest.fixture(scope="module")
def banded():
    spec = pad_in_ground("A", bands_um=MARGINS)
    return spec, build(spec)


def family(spec, name):
    return next(f for f in spec.families if f.family_id == name)


def arcs_at(fam, centre, ccw=None):
    return [a for a in fam.arcs if a.centre == (Fraction(centre[0]), Fraction(centre[1])) and (ccw is None or a.ccw == ccw)]


def test_reflex_vertices_become_arcs_of_radius_m_centred_at_the_vertex(banded):
    spec, _ = banded
    pad = family(spec, "band_pad_top")
    for vertex in ((-10000, -15000), (10000, -15000)):  # the two reflex corners of the notch
        found = {a.radius for a in arcs_at(pad, vertex, ccw=False)}
        assert found == {Fraction(round(m * 1000)) for m in MARGINS}
    gap = family(spec, "band_gap")  # the pad's convex corners are reflex corners of the gap region
    for vertex in ((50000, 18000), (38000, 30000), (50000, -30000)):
        assert {a.radius for a in arcs_at(gap, vertex, ccw=False)} == {Fraction(round(m * 1000)) for m in MARGINS}


def test_convex_corners_stay_sharp_no_arcs_at_convex_vertices_of_the_pad(banded):
    spec, _ = banded
    pad = family(spec, "band_pad_top")
    for vertex in ((50000, -30000), (50000, 18000), (38000, 30000), (-50000, 30000), (-10000, -30000)):
        assert arcs_at(pad, vertex) == []


def sweep_total(arcs):
    return sum(abs(a.sweep()[1]) for a in arcs)


def test_hole_becomes_a_concentric_circle_of_radius_r_plus_m(banded):
    spec, _ = banded
    pad = family(spec, "band_pad_top")
    centre = (-25000, 5000)
    for m in MARGINS:
        radius = Fraction(8000 + round(m * 1000))
        ring = [a for a in arcs_at(pad, centre, ccw=False) if a.radius == radius]
        assert sweep_total(ring) == pytest.approx(2 * math.pi, rel=1e-6)  # a full concentric circle, not a polygon


def test_rounded_corner_of_radius_rho_becomes_rho_minus_m_and_vanishes_when_m_reaches_rho():
    spec = pad_in_ground("A", bands_um=(0.5, 5.0, 6.0), band_regions=("pad_top",))
    pad = family(spec, "band_pad_top")
    centre = (-45000, -25000)
    radii = {a.radius for a in arcs_at(pad, centre, ccw=True)}
    assert radii == {Fraction(4500)}  # rho - 0.5; for m >= rho = 5 um the arc is gone (sharp corner)
    quarter = [a for a in arcs_at(pad, centre, ccw=True)]
    assert sweep_total(quarter) == pytest.approx(math.pi / 2, rel=1e-3)


def test_bands_partition_each_support_region_exactly(banded):
    spec, result = banded
    for name, area in (("band_pad_top", PAD_AREA_UM2), ("band_gap", GAP_AREA_UM2)):
        areas = exact_band_areas(result, name, 1)
        assert sorted(areas) == [0, 1, 2, 3]
        assert sum(areas.values()) * 1e-6 == pytest.approx(area, rel=1e-9)


def test_semantic_names_carry_the_band_index_and_unions_list_primary_groups(banded):
    _, result = banded
    groups = result.topology.groups
    for k in range(4):
        assert f"SHEET::pad::band={k}" in groups and f"SA::substrate|vacuum::band={k}" in groups
    assert not any(n.startswith(("MA::pad", "MS::pad")) for n in groups)  # one sheet face is both MA and MS
    assert "PEC::pad" not in groups  # the whole pad sheet is banded: its primary groups are the SHEET bands
    assert result.topology.unions["PEC::pad"] == [f"SHEET::pad::band={k}" for k in range(4)]
    rows = result.topology.surface_epr_rows  # MA and MS rows both reference the sheet group
    assert all(rows[f"{kind}::pad::band={k}"] == f"SHEET::pad::band={k}" for k in range(4) for kind in ("MA", "MS"))
    assert all(rows[f"SA::substrate|vacuum::band={k}"] == f"SA::substrate|vacuum::band={k}" for k in range(4))
    assert "PORT::p1" in groups  # the port is one group although the bands cut it into several faces
    assert len(groups["PORT::p1"]["ids"]) > 1


def test_adding_the_port_does_not_change_any_band_boundary():
    with_port = pad_in_ground("A", bands_um=MARGINS)
    without = pad_in_ground("A", bands_um=MARGINS, with_port=False)
    assert with_port.families == without.families  # the generator never sees the port: same carriers, same arcs
    a, b = build(with_port), build(without)
    for name in ("band_pad_top", "band_gap"):
        area_a, area_b = exact_band_areas(a, name, 1), exact_band_areas(b, name, 1)
        assert area_a.keys() == area_b.keys()
        for k in area_a:
            assert area_a[k] == pytest.approx(area_b[k], rel=1e-9)


def test_band_areas_agree_with_the_klayout_erosion_within_the_oracle_rounding_noise(banded):
    spec, result = banded
    for fam in spec.families:
        for row in comparison_rows(result, fam.support, fam.family_id, 1):
            assert abs(row["difference_um2"]) <= row["oracle_rounding_bound_um2"]
            assert abs(row["relative"]) < 1e-4
            assert row["components_klayout"] == 1


def test_topology_change_a_neck_splits_and_a_band_vanishes_without_failing_validation():
    spec = pad_in_ground("A", bands_um=(4.0, 9.0, 30.0), band_regions=("pad_top",))  # 30 um > the inscribed radius
    result = build(spec)
    areas = exact_band_areas(result, "band_pad_top", 1)
    assert sorted(areas) == [0, 1, 2]  # B_3 = R_30 is empty: the band vanished
    assert sum(areas.values()) * 1e-6 == pytest.approx(PAD_AREA_UM2, rel=1e-9)
    layer = result.level_layers[1]
    count = {}
    for fi in range(1, len(layer.arr.faces)):
        for _, k in layer.labels[fi].bands:
            count[k] = count.get(k, 0) + 1
    assert count[0] == 2 and count[2] > 1  # outer ring and hole ring; the interior splits in several pieces
    assert not result.report.failures


def test_palace_check_every_band_is_its_own_group_shared_by_tets_on_both_sides(tmp_path):
    spec = pad_in_ground("A", bands_um=MARGINS)
    result = build(spec)
    supports = {f.family_id: f.support for f in spec.families}
    out = emit(result.topology, tmp_path, "m", mesh_size=40.0, supports=supports,
               refine_families=("band_pad_top", "band_gap"), refine_size=1.0, refine_distance=(1.0, 20.0))
    assert out.mesh["conformal"], out.mesh["problems"]
    bands = out.mesh["bands"]
    assert len(bands) == 8
    for name, row in bands.items():
        assert row["triangles"] > 0 and row["all_shared_by_tets"] and row["centroids_in_band"], name
        assert len(row["tets_per_side"]) == 2 and len(set(row["tets_per_side"].values())) == 1  # both volumes, equally


def test_route_b_bands_the_pad_top_as_ma_and_the_pad_bottom_as_ms_by_default():
    spec = pad_in_ground("B", bands_um=MARGINS)
    assert [f.family_id for f in spec.families] == ["band_pad_top", "band_pad_bottom", "band_gap"]
    topo = build(spec).topology
    for k in range(4):
        assert f"MA::pad::band={k}" in topo.groups and f"MS::pad::band={k}" in topo.groups
    assert topo.surface_epr_rows["MS::pad::band=2"] == "MS::pad::band=2"
    assert set(topo.unions["PEC::pad"]) >= {f"{kind}::pad::band={k}" for kind in ("MA", "MS") for k in range(4)}
    assert spec.name == "pad_in_ground_B_bands" and pad_in_ground("A").name == "pad_in_ground_A"


def test_straight_inset_members_keep_their_generating_primitive_in_the_lineage(banded):
    spec, result = banded
    pad = family(spec, "band_pad_top")
    for carrier in pad.carriers:
        ring_edge = carrier.id.split("@")[0]  # <support>/ring<r>/edge<k>
        assert ring_edge.startswith("band_pad_top/ring") and "/edge" in ring_edge
        assert carrier.lineage.member == ring_edge and carrier.lineage.family_id == "band_pad_top"
    arc_members = {a.lineage.member for a in pad.arcs}
    assert any("/corner" in m for m in arc_members) and any("/edge" in m for m in arc_members)  # reflex arcs, offset arcs
    sources = {s for srcs in result.topology.lineage.values() for s in srcs}
    assert any(s.startswith("band_pad_top/ring0/edge") for s in sources)


def test_band_area_monotonicity_is_validated_and_a_broken_partition_fails_closed(banded):
    from sgbproto.errors import ValidationError
    from sgbproto.records import LevelLabel
    from sgbproto.validate import validate

    spec, result = banded
    assert any(c.name == "band_area_monotone" and c.passed for c in result.report.checks)
    layer = result.level_layers[1]
    victim = next(fi for fi in range(1, len(layer.arr.faces)) if layer.labels[fi].bands and layer.labels[fi].bands[0][1] == 3)
    saved = layer.labels[victim]
    layer.dissolved.labels[victim] = LevelLabel(saved.below, saved.above, saved.sheet, ())  # the deepest band loses a cell
    try:
        report = validate(spec, result.levels, result.slabs, result.level_layers, result.topology, check_planarity=False)
        assert any(c.name == "band_area_monotone" for c in report.failures)
        with pytest.raises(ValidationError):
            report.raise_if_failed()
    finally:
        layer.dissolved.labels[victim] = saved


def test_face_membership_is_area_weighted_and_non_exclusive(banded):
    _, result = banded
    faces = result.topology.faces
    pad_sheet = next(f for f in faces.values() if f.group == "SHEET::pad::band=3")
    assert pad_sheet.membership["pad"] == pytest.approx(1.0)  # the deepest band lies wholly inside the pad
    assert pad_sheet.membership["substrate"] == pytest.approx(1.0)  # overlaps count for every covering entity
    assert sum(pad_sheet.membership.values()) > 1.0  # shares may sum to more than 1
    port_faces = [f for f in faces.values() if f.group == "PORT::p1"]
    assert all(f.membership["p1"] == pytest.approx(1.0) for f in port_faces)
