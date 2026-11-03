"""End to end on the running examples: arrangements, assembly, routes and validation (stages S2-S9)."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from fractions import Fraction

import pytest
import shapely
from sgb_scenes import (
    GRID,
    airbridge,
    footprint,
    lumped_port,
    pad_in_ground,
    random_layout,
    rect_um,
)
from shapely.geometry import Polygon
from shapely.ops import unary_union

from scgsim.geometry.sgb.assembly.lineage import CarrierEntities
from scgsim.geometry.sgb.errors import (
    OverlapError,
    RouteError,
    SgbError,
    ValidationError,
)
from scgsim.geometry.sgb.model import build_model_from_catalogue
from scgsim.geometry.sgb.parallel import Runtime
from scgsim.geometry.sgb.records.families import CurveFamilyRecord
from scgsim.geometry.sgb.records.grid import exact_z
from scgsim.geometry.sgb.records.prisms import PrismCatalogue, PrismRecord
from scgsim.geometry.sgb.validation.validate import validate_model


@pytest.fixture(scope="module")
def pad_a():
    """The pad-in-ground example under Route A."""
    return build_model_from_catalogue(pad_in_ground("A"), "A")


@pytest.fixture(scope="module")
def pad_b():
    """The pad-in-ground example under Route B."""
    return build_model_from_catalogue(pad_in_ground("B"), "B")


@pytest.fixture(scope="module")
def bridge():
    """The airbridge example under Route B."""
    return build_model_from_catalogue(airbridge(), "B")


def volume_by_name(model):
    """Volumes by their owning entity (one volume per entity in these examples)."""
    return {v.domain.name: v for v in model.topology.volumes.values()}


def total_volume(model):
    """Sum of the signed volume of every shell of every volume."""
    return sum(
        (s.signed_volume for v in model.topology.volumes.values() for s in v.shells),
        Fraction(0),
    )


def test_all_invariants_pass(pad_a, pad_b, bridge):
    """Validation ran and nothing failed."""
    for model in (pad_a, pad_b, bridge):
        assert not model.report.failures
        names = {c.name for c in model.report.checks}
        assert {
            "shells_closed_and_oriented",
            "groups_partition_each_dimension",
        } <= names
        assert {"no_different_net_contact", "no_non_manifold_pinch"} <= names


def test_route_a_has_two_volumes_and_per_entity_sheet_faces(pad_a):
    """Route A: sheets are SHEET::<entity>::sheet faces, the port is PORT::<id>, no conductor volume exists."""
    groups = pad_a.routed.groups
    assert set(volume_by_name(pad_a)) == {"substrate", "vacuum"}
    assert len(groups["SHEET::ground::sheet"].ids) == 1
    assert len(groups["SHEET::pad::sheet"].ids) == 1
    assert len(groups["PORT::p1"].ids) == 1
    # the gap region and the hole in the pad are the two plain substrate/vacuum interface cells
    assert len(groups["SA::substrate|vacuum::interface"].ids) == 2


def test_route_b_excludes_conductor_volumes_but_keeps_their_boundary(pad_b):
    """Route B: the conductor volumes are not emitted; their boundary faces are MA/MS groups of their Entity."""
    volumes = volume_by_name(pad_b)
    emitted = {
        pad_b.topology.volumes[v].domain.name for v in pad_b.routed.emitted_volumes
    }
    assert emitted == {"substrate", "vacuum"}
    assert set(volumes) - emitted == {"ground", "pad"}
    groups = pad_b.routed.groups
    assert not any(
        name.startswith(("SHEET::", "NET::")) for name in groups
    )  # unions are not Gmsh groups
    assert {
        "MA::pad::top",
        "MS::pad::bottom",
        "MA::ground::top",
        "MS::ground::bottom",
    } <= set(groups)
    assert pad_b.routed.unions["NET::pad"] == [
        "MA::pad::sidewall",
        "MA::pad::top",
        "MS::pad::bottom",
    ]
    assert pad_b.routed.unions["NET::Ground"] == [
        "MA::ground::sidewall",
        "MA::ground::top",
        "MS::ground::bottom",
    ]
    # the lumped port is a sheet inside the vacuum volume: an embedded face
    assert len(volumes["vacuum"].embedded_faces) == 1


def test_volumes_are_exactly_conserved(pad_a, pad_b):
    """Integer arithmetic: the volumes add up to the chip box, curved walls included."""
    side, depth = 300_000, 200  # grid units, micrometres (z is not scaled)
    assert total_volume(pad_a) == side * side * depth
    assert total_volume(pad_b) == side * side * depth
    ground_area = side * side - 160_000 * 120_000
    assert volume_by_name(pad_b)["ground"].shells[0].signed_volume == Fraction(
        ground_area
    ) * Fraction(1, 5)


def test_airbridge_has_one_volume_per_entity_across_three_slabs(bridge):
    """Per-Entity attribution: the five same-Net conductor entities are five volumes, not one."""
    conductors = [
        v for v in bridge.topology.volumes.values() if v.domain.kind == "conductor"
    ]
    assert sorted(v.domain.name for v in conductors) == [
        "deck", "ground_left", "ground_right", "post_left", "post_right",
    ]  # fmt: skip
    assert total_volume(bridge) == 300_000 * 300_000 * 200
    assert len(bridge.labelled.slabs) == 5
    assert all(v.shells[0].role == "outer" for v in conductors)


def test_route_a_rejects_a_film_that_is_still_finite():
    """A film conductor (stack or local film) that the profile left finite fails closed and names it."""
    films = tuple(
        replace(p, feature_kind="stack_film") if p.material_kind == "conductor" else p
        for p in pad_in_ground("B").prisms
    )
    with pytest.raises(
        RouteError, match=r"films are still finite.*ground, pad"
    ) as error:
        build_model_from_catalogue(PrismCatalogue(GRID, films), "A")
    assert "thin_film" in str(error.value)


def test_route_a_accepts_finite_conductors_that_are_not_films():
    """Finite bodies (``finite_volume``, or undeclared as today's cutout shells) keep their volume, excluded."""
    for kind in ("finite_volume", None):
        bodies = tuple(
            replace(p, feature_kind=kind) if p.material_kind == "conductor" else p
            for p in pad_in_ground("B").prisms
        )
        model = build_model_from_catalogue(PrismCatalogue(GRID, bodies), "A")
        emitted = {
            model.topology.volumes[v].domain.name for v in model.routed.emitted_volumes
        }
        assert emitted == {"substrate", "vacuum"}
        assert {"MA::pad::top", "MS::pad::bottom"} <= set(model.routed.groups)


def test_unknown_route_is_rejected():
    """Only A, B and C exist."""
    with pytest.raises(RouteError, match="unknown route"):
        build_model_from_catalogue(pad_in_ground("A"), "D")


def test_sheet_conductor_overlap_of_different_nets_fails_closed():
    """A sheet of another Net overlapping the pad sheet is an overlap error."""
    base = pad_in_ground("A")
    extra = PrismRecord(
        "short",
        footprint(rect_um(-1, -1, 1, 1)),
        exact_z(0),
        exact_z(0),
        "Al",
        "conductor",
        net_id="other",
    )
    bad = PrismCatalogue(GRID, (*base.prisms, extra))
    with pytest.raises(OverlapError, match="overlap"):
        build_model_from_catalogue(bad, "A")


def test_conductor_areas_match_shapely_union_of_the_source_polygons():
    """Per Net, the conductor cells of the slab add up to the union of the source footprints."""
    catalogue = random_layout(40, seed=11, route="B")
    model = build_model_from_catalogue(catalogue, "B")
    slab = next(s for s in model.labelled.slabs if s.plane == "slab[0,1/5]")
    by_net: dict[str, int] = {}
    for fi in range(1, len(slab.arr.faces)):
        domain = slab.labels[fi]
        if domain is not None and domain.kind == "conductor":
            by_net[domain.net] = by_net.get(domain.net, 0) + slab.arr.face_twice_area(
                fi
            )
    for net, area2 in by_net.items():
        polygons = [p for p in catalogue.prisms if p.net_id == net]
        union = unary_union(
            [Polygon(p.footprint.outer, p.footprint.holes) for p in polygons]
        )
        perimeter = sum(
            abs(r[(i + 1) % len(r)][0] - r[i][0])
            + abs(r[(i + 1) % len(r)][1] - r[i][1])
            for p in polygons
            for r in p.footprint.rings()
            for i in range(len(r))
        )  # grid units; snapping moves a boundary by < 1 unit
        assert area2 / 2 == pytest.approx(union.area, abs=perimeter), net
    assert shapely.is_valid(
        unary_union([Polygon(p.footprint.outer) for p in catalogue.prisms[:2]])
    )


def test_a_tampered_shell_is_rejected_by_validation(pad_b):
    """Dropping a face from a shell makes the closure check fail, and raise_if_failed stops."""
    volume = next(
        v
        for vid, v in pad_b.topology.volumes.items()
        if vid in pad_b.routed.emitted_volumes and v.domain.name == "vacuum"
    )
    saved = volume.shells[0].faces
    volume.shells[0].faces = saved[1:]
    try:
        report = validate_model(pad_b.labelled, pad_b.topology, pad_b.routed, Runtime())
        assert any(c.name == "shells_closed_and_oriented" for c in report.failures)
        with pytest.raises(ValidationError):
            report.raise_if_failed()
    finally:
        volume.shells[0].faces = saved


def test_lineage_maps_every_edge_to_source_entities(pad_b):
    """Every edge's carriers resolve to a source entity of the catalogue."""
    topology = pad_b.topology
    names = {p.entity_id for p in pad_b.labelled.catalogue.prisms}
    entities = CarrierEntities(pad_b.labelled.catalogue)
    for sources in topology.lineage.values():
        assert sources
        assert set(entities.entities_of(sources)) <= names
    assert [e for e, s in topology.lineage.items() if "p1" in entities.entities_of(s)]


def test_semantic_names_do_not_depend_on_native_state(pad_a):
    """The group names are a function of the labels alone."""
    assert sorted(pad_a.routed.groups) == [
        "EXT::substrate::bottom", "EXT::substrate::sidewall", "EXT::vacuum::sidewall", "EXT::vacuum::top",
        "PORT::p1", "SA::substrate|vacuum::interface", "SHEET::ground::sheet", "SHEET::pad::sheet",
        "VOL::substrate", "VOL::vacuum",
    ]  # fmt: skip


def test_arrangements_are_local_not_global():
    """The deck exists only in one slab: no other slab or level arrangement contains its curves."""
    model = build_model_from_catalogue(airbridge(), "B")
    for layer in (*model.labelled.slabs, *model.labelled.level_layers):
        has_deck = any(s.startswith("deck/") for src in layer.edge_sources for s in src)
        touches_deck_slab = layer.plane in ("slab[3,33/10]", "level[3]", "level[33/10]")
        assert has_deck == touches_deck_slab, layer.plane


def test_curve_families_required_and_refinement_only():
    """A required family splits a cell into faces of equal label; a refinement-only curve is embedded."""
    from sgb_extreme_cases import refinement  # noqa: PLC0415 -- a test helper imported where it is used

    base = pad_in_ground("A")
    ring = [
        (-75_000, -20_000),
        (-55_000, -20_000),
        (-55_000, 20_000),
        (-75_000, 20_000),
    ]
    required = CurveFamilyRecord(
        "cap",
        "port_footprint",
        tuple(_family_segment(f"cap{i}", ring[i], ring[(i + 1) % 4]) for i in range(4)),
        (Fraction(0),),
        "required",
    )
    refine = refinement("ref", (-40_000, 55_000), (40_000, 55_000), Fraction(0))
    plain = build_model_from_catalogue(base, "A")
    families = build_model_from_catalogue(
        PrismCatalogue(base.grid, base.prisms, (required, refine)), "A"
    )

    def gap_faces(model):
        """Gap faces."""
        return len(model.routed.groups["SA::substrate|vacuum::interface"].ids)

    assert (
        gap_faces(families) == gap_faces(plain) + 1
    )  # the required family splits one cell in two
    assert len([f for f in families.topology.faces.values() if f.refinement_edges]) == 1


def _family_segment(name, p, q):
    from scgsim.geometry.sgb.records.carriers import Lineage, LineCarrier  # noqa: PLC0415 -- used once

    return LineCarrier(
        name, p, q, Lineage("family_member", family_id="fam", member=name)
    )


def test_a_refinement_curve_in_a_cell_without_a_face_is_an_error_not_a_silent_drop():
    """Inside the pad metal at the port mid-plane the cell is inside an excluded conductor: no face to embed in."""
    from sgb_extreme_cases import refinement  # noqa: PLC0415 -- a test helper imported where it is used

    base = pad_in_ground("B")
    inside_pad = refinement(
        "ref", (-40_000, 20_000), (-30_000, 20_000), Fraction("0.1")
    )
    with pytest.raises(SgbError, match="no face"):
        build_model_from_catalogue(
            PrismCatalogue(base.grid, base.prisms, (inside_pad,)), "B"
        )
    in_the_gap = refinement("ref", (60_000, 40_000), (70_000, 40_000), Fraction(0))
    model = build_model_from_catalogue(
        PrismCatalogue(base.grid, base.prisms, (in_the_gap,)), "B"
    )
    hosts = [f for f in model.topology.faces.values() if f.refinement_edges]
    assert (
        len(hosts) == 1
        and model.routed.face_group[hosts[0].id] == "SA::substrate|vacuum::interface"
    )


def test_no_entity_is_in_two_physical_groups_and_every_emitted_entity_is_in_one(
    pad_a, pad_b, bridge
):
    """Primary groups partition each dimension."""
    for model in (pad_a, pad_b, bridge):
        seen = Counter((g.dim, i) for g in model.routed.groups.values() for i in g.ids)
        assert all(n == 1 for n in seen.values())
        for face in model.topology.faces.values():
            emitted = model.routed.is_emitted_face(face.neg_volume, face.pos_volume)
            assert ((2, face.id) in seen) == emitted
        assert all(
            ((3, v) in seen) == (v in model.routed.emitted_volumes)
            for v in model.topology.volumes
        )


def test_route_a_net_union_lists_the_entity_sheet_groups(pad_a):
    """Net unions list the primary groups of the Net's exposed surfaces."""
    assert pad_a.routed.unions["NET::Ground"] == ["SHEET::ground::sheet"]
    assert pad_a.routed.unions["NET::pad"] == ["SHEET::pad::sheet"]
    assert pad_a.routed.unions["SHEET"] == ["SHEET::ground::sheet", "SHEET::pad::sheet"]
    assert (
        pad_a.routed.unions["MA"]
        == pad_a.routed.unions["MS"]
        == pad_a.routed.unions["SHEET"]
    )  # a sheet carries both rows


# --------------------------------------------------------------------------- wall coalescing


def wall_ranges(model):
    """Wall faces by their Z range."""
    return Counter(f.z_range for f in model.topology.faces.values() if f.kind == "wall")


def with_port(x0: float, x1: float) -> PrismCatalogue:
    """Route B pad-in-ground whose port rectangle spans x0..x1 at the metal mid-plane."""
    base = pad_in_ground("B", with_port=False)
    port = lumped_port("p1", rect_um(x0, -5, x1, 5), "B")
    return PrismCatalogue(base.grid, (*base.prisms, port), name="port_variant")


def test_port_edge_touching_a_metal_sidewall_cuts_that_wall_at_the_port_level():
    """A level edge on a sidewall cuts the wall there; everything else stays one face."""
    model = build_model_from_catalogue(
        with_port(50, 80), "B"
    )  # reaches the pad wall x = 50 and the cutout wall x = 80
    ranges = wall_ranges(model)
    halves = (Fraction(0), Fraction(1, 10)), (Fraction(1, 10), Fraction(1, 5))
    assert (
        ranges[halves[0]] == 2 and ranges[halves[1]] == 2
    )  # a pad wall and a ground wall, each cut in two
    # 22 metal walls (14 pad: 9 lines, 1 corner arc, 4 hole arcs; 8 ground); all but the two touched stay single
    assert ranges[(Fraction(0), Fraction(1, 5))] == 22 - 2


def test_port_sheet_that_touches_no_sidewall_does_not_cut_any_wall():
    """A port floating in the gap leaves every wall whole."""
    model = build_model_from_catalogue(with_port(55, 75), "B")
    ranges = wall_ranges(model)
    assert (Fraction(0), Fraction(1, 10)) not in ranges and (
        Fraction(1, 10),
        Fraction(1, 5),
    ) not in ranges
    assert ranges[(Fraction(0), Fraction(1, 5))] == 22
    assert len(model.topology.volumes) == 4 and not model.report.failures


def test_wall_coalescing_reduces_the_polygon_pad_and_the_arc_pad_face_counts(pad_b):
    """The true circle is four arc walls instead of 32 chord walls: 41 faces against 68."""
    polygon_pad = build_model_from_catalogue(
        pad_in_ground("B", hole_sides=32, rounded_corner=False), "B"
    )
    assert polygon_pad.counts["faces"] == 68
    assert pad_b.counts["faces"] == 41
