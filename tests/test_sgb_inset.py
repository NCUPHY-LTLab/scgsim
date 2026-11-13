"""Surface-EPR inset bands: the exact corner rule, distance origins, request API and the KLayout cross-check."""

from __future__ import annotations

import math
from fractions import Fraction

import pytest
from sgb_extreme_cases import FIXTURES, build_model
from sgb_scenes import (
    GRID,
    chip_prisms,
    conductor,
    exact_band_areas,
    pad_in_ground,
    rect_um,
)

from scgsim.geometry.sgb.bands.requests import (
    ConductorSurface,
    InsetRequest,
    SolutionInterface,
)
from scgsim.geometry.sgb.checks.validation.validate import validate_model
from scgsim.geometry.sgb.errors import InputError, ValidationError
from scgsim.geometry.sgb.exact.labels import LevelLabel
from scgsim.geometry.sgb.exact.prisms import PrismCatalogue
from scgsim.geometry.sgb.exact.stopwatch import Stopwatch
from scgsim.geometry.sgb.model import build_model_from_catalogue
from scgsim.geometry.sgb.output.sidecar.lineage import CarrierLineage

MARGINS = (0.1, 0.5, 2.0)
PAD_AREA_UM2 = 100 * 60 - 20 * 15 - 12 * 12 / 2 - 25 * (1 - math.pi / 4) - math.pi * 64
HOLE_AREA_UM2 = math.pi * 64
GAP_AREA_UM2 = (
    160 * 120 - PAD_AREA_UM2
)  # the ring around the pad and the cell inside its hole


def pad_requests(route: str, margins=MARGINS) -> tuple[InsetRequest, ...]:
    """Bands on the pad's top (and, finite, bottom) face and on the substrate-vacuum interface."""
    pad = (
        (InsetRequest("band_pad_top", ConductorSurface("pad", "sheet"), margins),)
        if route == "A"
        else (
            InsetRequest("band_pad_top", ConductorSurface("pad", "top"), margins),
            InsetRequest("band_pad_bottom", ConductorSurface("pad", "bottom"), margins),
        )
    )
    return (
        *pad,
        InsetRequest("band_gap", SolutionInterface("substrate", "vacuum"), margins),
    )


@pytest.fixture(scope="module")
def oracle():
    """The KLayout erosion oracle; its tests are skipped where the optional KLayout is missing."""
    pytest.importorskip("klayout.db", reason="the erosion cross-check needs KLayout")
    import sgb_klayout  # noqa: PLC0415 -- imports KLayout, so only after the importorskip above

    return sgb_klayout


@pytest.fixture(scope="module")
def banded():
    """The pad in ground under Route A with three margins on the pad sheet and on the gap."""
    return build_model_from_catalogue(pad_in_ground("A"), "A", insets=pad_requests("A"))


def family(model, name):
    """A cut family of a model by id."""
    return next(f for f in model.labelled.catalogue.families if f.family_id == name)


def arcs_at(fam, centre, ccw=None):
    """Arc members of a family centred at a lattice point."""
    return [
        a
        for a in fam.arcs
        if a.centre == (Fraction(centre[0]), Fraction(centre[1]))
        and (ccw is None or a.ccw == ccw)
    ]


def sweep_total(arcs):
    """Total angle swept by arcs, in radians."""
    return sum(abs(float(a.sweep()[1])) for a in arcs)


def test_reflex_vertices_become_arcs_of_radius_m_centred_at_the_vertex(banded):
    """The two reflex corners of the notch, and the pad's convex corners seen from the gap."""
    pad = family(banded, "band_pad_top")
    for vertex in ((-10000, -15000), (10000, -15000)):
        assert {a.radius for a in arcs_at(pad, vertex, ccw=False)} == {
            Fraction(round(m * 1000)) for m in MARGINS
        }
    gap = family(banded, "band_gap")
    for vertex in ((50000, 18000), (38000, 30000), (50000, -30000)):
        assert {a.radius for a in arcs_at(gap, vertex, ccw=False)} == {
            Fraction(round(m * 1000)) for m in MARGINS
        }


def test_convex_corners_stay_sharp_no_arcs_at_convex_vertices_of_the_pad(banded):
    """Offset lines cross at a convex corner; there is no rounding."""
    pad = family(banded, "band_pad_top")
    for vertex in (
        (50000, -30000),
        (50000, 18000),
        (38000, 30000),
        (-50000, 30000),
        (-10000, -30000),
    ):
        assert arcs_at(pad, vertex) == []


def test_hole_becomes_a_concentric_circle_of_radius_r_plus_m(banded):
    """A circular hole offsets to a full concentric circle, not a polygon."""
    pad = family(banded, "band_pad_top")
    for m in MARGINS:
        radius = Fraction(8000 + round(m * 1000))
        ring = [
            a for a in arcs_at(pad, (-25000, 5000), ccw=False) if a.radius == radius
        ]
        assert sweep_total(ring) == pytest.approx(2 * math.pi, rel=1e-6)


def test_rounded_corner_of_radius_rho_becomes_rho_minus_m_and_vanishes_when_m_reaches_rho():
    """A convex arc shrinks to rho - m and is gone (a sharp corner) for m >= rho."""
    request = InsetRequest(
        "band_pad_top", ConductorSurface("pad", "sheet"), (0.5, 5.0, 6.0)
    )
    model = build_model_from_catalogue(pad_in_ground("A"), "A", insets=(request,))
    arcs = arcs_at(family(model, "band_pad_top"), (-45000, -25000), ccw=True)
    assert {a.radius for a in arcs} == {Fraction(4500)}
    assert sweep_total(arcs) == pytest.approx(math.pi / 2, rel=1e-3)


def test_bands_partition_each_support_region_exactly(banded):
    """The band areas add up to the support area (an exact comparison, 1e-9)."""
    for name, area in (("band_pad_top", PAD_AREA_UM2), ("band_gap", GAP_AREA_UM2)):
        areas = exact_band_areas(banded, name)
        assert sorted(areas) == [0, 1, 2, 3]
        assert sum(areas.values()) * 1e-6 == pytest.approx(area, rel=1e-9)


def test_semantic_names_carry_the_band_index_and_unions_list_primary_groups(banded):
    """Route A: one sheet group per band carries both the MA and MS rows; the gap bands are SA groups."""
    groups = banded.routed.groups
    for k in range(4):
        assert (
            f"pad::band={k}" in groups
            and f"SA::substrate|vacuum__interface::band={k}" in groups
        )
    assert not any(n.startswith(("MA::pad", "MS::pad", "pad::band=4")) for n in groups)
    assert "pad" not in groups  # the whole pad sheet is banded
    assert banded.routed.unions["NET::pad"] == [f"pad::band={k}" for k in range(4)]
    rows = banded.routed.surface_epr_rows
    assert all(
        rows[f"{kind}::pad__sheet::band={k}"] == f"pad::band={k}"
        for k in range(4)
        for kind in ("MA", "MS")
    )
    assert all(
        rows[f"SA::substrate|vacuum__interface::band={k}"]
        == f"SA::substrate|vacuum__interface::band={k}"
        for k in range(4)
    )
    assert (
        len(groups["PORT::p1"].ids) == 1
    )  # the band seams stop at the port boundary: one face


def test_adding_the_port_changes_no_band_but_keeps_the_seams_out_of_it():
    """The pre-pass sees physical prisms only: same supports; the port takes its footprint away from the bands."""
    with_port = build_model_from_catalogue(
        pad_in_ground("A"), "A", insets=pad_requests("A")
    )
    without = build_model_from_catalogue(
        pad_in_ground("A", with_port=False), "A", insets=pad_requests("A")
    )
    assert [f.support for f in with_port.labelled.catalogue.families] == [
        f.support for f in without.labelled.catalogue.families
    ]
    port_um2 = 30 * 10  # the port outline 50..80 x -5..5 um
    for name in ("band_pad_top", "band_gap"):
        a = exact_band_areas(with_port, name, include_ports=False)
        b = exact_band_areas(without, name)
        assert a.keys() == b.keys()
        assert all(
            a[k] <= b[k] * (1 + 1e-9) for k in a
        )  # the port only takes area away
    removed = sum(exact_band_areas(without, "band_gap").values()) - sum(
        exact_band_areas(with_port, "band_gap", include_ports=False).values()
    )
    assert removed * 1e-6 == pytest.approx(port_um2, rel=1e-9)
    assert exact_band_areas(with_port, "band_pad_top") == exact_band_areas(
        without, "band_pad_top"
    )  # the pad sheet is nowhere near the port


@pytest.mark.slow
@pytest.mark.parametrize("name", ["band_pad_top", "band_gap"])
def test_band_areas_agree_with_the_klayout_erosion_within_the_oracle_rounding_noise(
    banded, oracle, name
):
    """Independent oracle: KLayout's polygonal erosion of the same support region."""
    for row in oracle.comparison_rows(banded, name):
        assert abs(row["difference_um2"]) <= row["oracle_rounding_bound_um2"]
        assert abs(row["relative"]) < 1e-4


def test_topology_change_a_band_vanishes_without_failing_validation():
    """30 um is beyond the inscribed radius of the pad: R_30 is empty, so there is no innermost band."""
    request = InsetRequest(
        "band_pad_top", ConductorSurface("pad", "sheet"), (4.0, 9.0, 30.0)
    )
    model = build_model_from_catalogue(pad_in_ground("A"), "A", insets=(request,))
    areas = exact_band_areas(model, "band_pad_top")
    assert sorted(areas) == [0, 1, 2]
    assert sum(areas.values()) * 1e-6 == pytest.approx(PAD_AREA_UM2, rel=1e-9)
    layer = model.labelled.level_layers[1]
    count: dict[int, int] = {}
    for fi in range(1, len(layer.arr.faces)):
        for _, k in layer.labels[fi].bands:
            count[k] = count.get(k, 0) + 1
    assert (
        count[0] == 2 and count[2] > 1
    )  # outer ring and hole ring; the interior splits in several pieces
    assert not model.report.failures


def test_route_b_and_c_band_the_pad_top_as_ma_and_the_pad_bottom_as_ms():
    """Finite pad: top face MA, bottom face MS, each band its own group; the sidewalls stay one group."""
    for route in ("B", "C"):
        model = build_model_from_catalogue(
            pad_in_ground("B"), route, insets=pad_requests("B")
        )
        groups = model.routed.groups
        for k in range(4):
            assert (
                f"MA::pad__top::band={k}" in groups
                and f"MS::pad__bottom::band={k}" in groups
            )
        assert (
            "MA::pad__sidewall" in groups
        )  # the sidewalls are not banded (out of version 1)
        assert (
            model.routed.surface_epr_rows["MS::pad__bottom::band=2"]
            == "MS::pad__bottom::band=2"
        )
        assert set(model.routed.unions["NET::pad"]) >= {
            f"MA::pad__top::band={k}" for k in range(4)
        } | {f"MS::pad__bottom::band={k}" for k in range(4)}


def test_straight_inset_members_keep_their_generating_primitive_in_the_lineage(banded):
    """Straight members offset a primitive of the support (role ``edge``); arc members offset one or turn round
    a reflex corner; the emitted edges carry members of the family's outer ring."""
    pad = family(banded, "band_pad_top")
    rings = pad.support.rings
    for carrier in pad.carriers:
        member = carrier.lineage.band_member
        assert carrier.lineage.family_id == "band_pad_top" and member is not None
        assert member.role == "edge" and member.index < len(rings[member.ring])
    roles = {a.lineage.band_member.role for a in pad.arcs if a.lineage.band_member}
    assert {"corner", "edge"} <= roles
    lineage = CarrierLineage.of(banded)
    members = [
        found.band_member
        for sources in banded.topology.lineage.values()
        for s in sources
        if (found := lineage.lineage(s)) is not None
        and found.family_id == "band_pad_top"
        and found.band_member is not None
    ]
    assert any(m.ring == 0 and m.role == "edge" for m in members)


def test_band_area_monotonicity_is_validated_and_a_broken_partition_fails_closed(
    banded,
):
    """The deepest band loses a cell: the bands no longer cover the support and validation fails."""
    assert any(
        c.name == "band_area_monotone" and c.passed for c in banded.report.checks
    )
    layer = banded.labelled.level_layers[1]
    victim = next(
        fi
        for fi in range(1, len(layer.arr.faces))
        if layer.labels[fi].bands and layer.labels[fi].bands[0][1] == 3
    )
    saved = layer.labels[victim]
    layer.dissolved.labels[victim] = LevelLabel(
        saved.below, saved.above, saved.sheet, ()
    )
    try:
        report = validate_model(
            banded.labelled, banded.topology, banded.routed, Stopwatch()
        )
        assert any(c.name == "band_area_monotone" for c in report.failures)
        with pytest.raises(ValidationError):
            report.raise_if_failed()
    finally:
        layer.dissolved.labels[victim] = saved


def test_face_membership_is_area_weighted_and_non_exclusive(banded):
    """The deepest band lies wholly inside the pad; overlaps count for every covering entity."""
    faces = banded.topology.faces
    deepest = next(
        f for f in faces.values() if banded.routed.face_group[f.id] == "pad::band=3"
    )
    assert deepest.membership["pad"] == pytest.approx(1.0)
    assert deepest.membership["substrate"] == pytest.approx(1.0)
    assert sum(deepest.membership.values()) > 1.0
    assert all(
        f.membership["p1"] == pytest.approx(1.0)
        for f in faces.values()
        if banded.routed.face_group[f.id] == "PORT::p1"
    )


# --------------------------------------------------------------------------- requests fail closed


def request_error(request, route="A", catalogue=None):
    """The message of the InputError an invalid request raises."""
    with pytest.raises(InputError) as error:
        build_model_from_catalogue(
            catalogue or pad_in_ground(route), route, insets=(request,)
        )
    return str(error.value)


def test_requests_that_name_no_face_or_a_wrong_face_fail_closed():
    """Unknown entity, wrong side for the representation, a face with no cell."""
    assert "not a conductor entity" in request_error(
        InsetRequest("r", ConductorSurface("nope", "sheet"), (0.1,))
    )
    assert "is a sheet; ask for side='sheet'" in request_error(
        InsetRequest("r", ConductorSurface("pad", "top"), (0.1,))
    )
    assert "is finite" in request_error(
        InsetRequest("r", ConductorSurface("pad", "sheet"), (0.1,)), "B"
    )
    assert "not a solution region" in request_error(
        InsetRequest("r", SolutionInterface("pad", "vacuum"), (0.1,))
    )
    assert "share no interface" in request_error(
        InsetRequest("r", SolutionInterface("vacuum", "vacuum"), (0.1,))
    )


def test_margins_must_be_positive_increasing_multiples_of_the_grid():
    """Margins are exact lattice lengths."""
    pad = ConductorSurface("pad", "sheet")
    assert "needs at least one margin" in request_error(InsetRequest("r", pad, ()))
    assert "not a positive multiple" in request_error(InsetRequest("r", pad, (0.0005,)))
    assert "not a positive multiple" in request_error(InsetRequest("r", pad, (-0.1,)))
    assert "strictly increasing" in request_error(InsetRequest("r", pad, (0.5, 0.1)))


def test_the_expected_interface_type_is_checked_against_the_labels():
    """``interface='MA'`` on a Route A sheet fails: a sheet face is SHEET (both MA and MS rows)."""
    model = build_model_from_catalogue(
        pad_in_ground("B"),
        "B",
        insets=(
            InsetRequest("r", ConductorSurface("pad", "top"), (0.5,), interface="MA"),
        ),
    )
    assert "MA::pad__top::band=1" in model.routed.groups
    message = request_error(
        InsetRequest("r", ConductorSurface("pad", "top"), (0.5,), interface="MS"), "B"
    )
    assert "expected a MS face" in message and "MA" in message


def test_request_ids_must_be_unique_and_the_interface_z_must_be_unambiguous():
    """Two requests with one id, and an interface that lies on two levels."""
    pad = ConductorSurface("pad", "sheet")
    first = InsetRequest("same", pad, (0.5,))
    with pytest.raises(InputError, match="repeated: same"):
        build_model_from_catalogue(pad_in_ground("A"), "A", insets=(first, first))
    z0 = chip_prisms(GRID, 40.0)
    stacked = PrismCatalogue(
        GRID, (*z0, conductor("m", rect_um(-5, -5, 5, 5), 0, "0.2", "N"))
    )
    # substrate|vacuum meet only at z = 0 here, so the interface is unambiguous; a z that is not a level is empty
    request = InsetRequest(
        "gap", SolutionInterface("substrate", "vacuum", z_um=0.2), (0.5,)
    )
    assert "share no interface" in request_error(request, "B", stacked)


# --------------------------------------------------------------------------- the band extreme fixtures


@pytest.mark.parametrize("route", ["A", "B", "C"])
def test_a_band_that_vanishes_has_no_innermost_group(route):
    """R_0.2 of a 0.3 um strip is empty: only bands 0 and 1 exist, and the model validates."""
    model = build_model(FIXTURES["band_vanishes"], route)
    bands = {
        name.rsplit("band=", 1)[1] for name in model.routed.groups if "::band=" in name
    }
    assert bands == {"0", "1"}


@pytest.mark.parametrize("route", ["A", "B", "C"])
def test_a_split_neck_gives_two_components_after_the_neck_closes(route, oracle):
    """KLayout counts one eroded component at 0.1 um and two at 0.4 um; the pipeline's bands match its areas."""
    model = build_model(FIXTURES["neck_split_by_inset"], route)
    rows = oracle.comparison_rows(model, "band_dumbbell_top")
    components = {r["margin_units"]: r["components_klayout"] for r in rows}
    assert components[100] == 1 and components[400] == 2
    for row in rows:
        assert abs(row["difference_um2"]) <= row["oracle_rounding_bound_um2"]


def test_the_inset_arcs_of_a_split_neck_end_at_their_crossing_without_stubs():
    """The two corner arcs at the neck mouth cross at (-4312, 0); the untrimmed chords' ends are gone."""
    model = build_model(FIXTURES["neck_split_by_inset"], "A")
    points = {(v.x, v.y) for v in model.topology.vertices.values()}
    assert (-4312, 0) in points and (4312, 0) in points
    assert not points & {(-4310, 2), (-4310, -2), (4310, 2), (4310, -2)}


@pytest.mark.parametrize("route", ["A", "B", "C"])
def test_a_reflex_corner_with_a_tiny_angle_keeps_the_bands_consistent(route, oracle):
    """A 0.6 degree slit apex: bands still partition the face and match the oracle within its noise."""
    model = build_model(FIXTURES["reflex_small_angle"], route)
    for row in oracle.comparison_rows(model, "band_slit_top"):
        assert abs(row["difference_um2"]) <= row["oracle_rounding_bound_um2"]


@pytest.mark.parametrize("route", ["A", "B", "C"])
def test_cut_family_curves_crossing_band_edges_are_split_at_every_crossing(route):
    """The rectangle's long sides at y = 0.5 and 1.5 um cross the 0.5 and 2 um band boundaries at x = -9.5, -8, 8 and 9.5."""
    model = build_model(FIXTURES["family_crossing_band_edge"], route)
    assert not model.report.failures
    known = model.labelled.catalogue.known_carriers()
    pieces = [
        e
        for e in model.topology.edges.values()
        if e.kind == "horizontal"
        and any(c in known and known[c].lineage.family_id == "guide" for c in e.sources)
    ]
    assert (
        len(pieces) == 2 * 5 + 2
    )  # each long side in five pieces, each short side whole


# --------------------------------------------------------------------------- real inputs and new levels


def test_bands_on_a_real_thin_film_sheet_use_the_qualified_entity_id():
    """Source input through S0/S1: ``chip/left::band=k``, and the lineage prefix is the request id."""
    from sgb_inputs import thin_film_pair  # noqa: PLC0415 -- a fixture used by the real-input tests only

    from scgsim.geometry.sgb.model import build_conformal_model  # noqa: PLC0415 -- used by these tests only

    effective = thin_film_pair(0.5)
    request = InsetRequest(
        "bands/left", ConductorSurface("chip/left", "sheet"), (0.25, 1.0)
    )
    model = build_conformal_model(
        effective.build_input, effective.representation, "A", insets=(request,)
    )
    groups = model.routed.groups
    assert {f"chip/left::band={k}" for k in range(3)} <= set(groups)
    assert "chip/right" in groups  # the other Entity is untouched
    assert (
        model.routed.surface_epr_rows["MA::chip/left__sheet::band=2"]
        == "chip/left::band=2"
    )
    fam = family(model, "bands/left")
    assert all(
        c.lineage.family_id == "bands/left" and c.lineage.band_member is not None
        for c in fam.carriers
    )
    assert not model.report.failures


def test_bands_on_the_top_of_a_real_finite_conductor(tmp_path):
    """GDS input, Route B: the pad's top face bands are ``MA::pad::band=k`` and the port does not move them."""
    from sgb_inputs import gds_pad_in_ground  # noqa: PLC0415 -- a fixture used by the real-input tests only

    from scgsim.geometry.sgb.model import build_conformal_model  # noqa: PLC0415 -- used by these tests only

    effective = gds_pad_in_ground(tmp_path)
    request = InsetRequest(
        "pad_top", ConductorSurface("pad", "top"), (1.0, 3.0), interface="MA"
    )
    model = build_conformal_model(
        effective.build_input, effective.representation, "B", insets=(request,)
    )
    assert {f"MA::pad__top::band={k}" for k in range(3)} <= set(model.routed.groups)
    assert exact_band_areas(model, "pad_top").keys() == {0, 1, 2}
    assert sum(exact_band_areas(model, "pad_top").values()) * 1e-6 == pytest.approx(
        400.0, rel=1e-9
    )


def test_a_family_at_a_z_no_prism_defines_adds_an_ordinary_level():
    """A required family at z = 0.1 um in a Route A model: a new level, no active-set change, a valid model."""
    from scgsim.geometry.sgb.exact.carriers import Lineage, LineCarrier  # noqa: PLC0415 -- used by this test only
    from scgsim.geometry.sgb.exact.families import CurveFamilyRecord  # noqa: PLC0415 -- used by this test only

    base = pad_in_ground("A")
    carrier = LineCarrier(
        "cap0",
        (60_000, 40_000),
        (70_000, 40_000),
        Lineage("family_member", family_id="cap", member="cap0"),
    )
    cap = CurveFamilyRecord("cap", "cap", (carrier,), (Fraction(1, 10),))
    plain = build_model_from_catalogue(base, "A")
    model = build_model_from_catalogue(
        PrismCatalogue(base.grid, base.prisms, (cap,)), "A"
    )
    assert Fraction(1, 10) in model.labelled.levels.levels
    assert len(model.labelled.levels.levels) == len(plain.labelled.levels.levels) + 1
    assert model.labelled.levels.slab_count == plain.labelled.levels.slab_count + 1
    assert sum(
        v.shells[0].signed_volume for v in model.topology.volumes.values()
    ) == sum(v.shells[0].signed_volume for v in plain.topology.volumes.values())
    assert not model.report.failures
