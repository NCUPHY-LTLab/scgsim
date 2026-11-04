"""End-to-end derivation on the running examples (stages S2-S9)."""

from __future__ import annotations

from collections import Counter
from fractions import Fraction

import pytest
import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union

from sgbproto.errors import RouteError, ValidationError
from sgbproto.pipeline import build
from sgbproto.records import Footprint, ModelSpec, PrismRecord, z_value
from sgbproto.scenes import airbridge, pad_in_ground, random_layout
from sgbproto.validate import validate


@pytest.fixture(scope="module")
def pad_a():
    return build(pad_in_ground("A"))


@pytest.fixture(scope="module")
def pad_b():
    return build(pad_in_ground("B"))


@pytest.fixture(scope="module")
def bridge():
    return build(airbridge())


def volume_by_name(result):
    return {v.domain.name: v for v in result.topology.volumes.values()}


def total_volume(result):
    return sum((s.signed_volume for v in result.topology.volumes.values() for s in v.shells), Fraction(0))


def test_all_invariants_pass(pad_a, pad_b, bridge):
    for result in (pad_a, pad_b, bridge):
        failed = [c.name for c in result.report.checks if not c.passed and not c.warning]
        assert not failed


def test_route_a_has_two_volumes_and_pec_sheet_faces(pad_a):
    groups = pad_a.topology.groups
    assert set(volume_by_name(pad_a)) == {"substrate", "vacuum"}
    assert len(groups["PEC::Ground"]["ids"]) == 1
    assert len(groups["PEC::pad"]["ids"]) == 1
    assert len(groups["PORT::p1"]["ids"]) == 1
    # gap region and the hole in the pad are the two plain substrate/vacuum interface cells
    assert len(groups["SA::substrate|vacuum"]["ids"]) == 2


def test_route_b_excludes_conductor_volumes_but_keeps_their_boundary(pad_b):
    vols = volume_by_name(pad_b)
    assert {n for n, v in vols.items() if v.emitted} == {"substrate", "vacuum"}
    assert {n for n, v in vols.items() if not v.emitted} == {"Ground", "pad"}
    groups = pad_b.topology.groups
    assert "PEC::Ground" not in groups and "PEC::pad" not in groups  # unions are not Gmsh groups
    assert {"MA::pad", "MS::pad", "MA::Ground", "MS::Ground"} <= set(groups)
    assert pad_b.topology.unions["PEC::pad"] == ["MA::pad", "MS::pad"]
    assert pad_b.topology.unions["PEC::Ground"] == ["MA::Ground", "MS::Ground"]
    # lumped port is a sheet inside the vacuum volume: an embedded face
    assert len(vols["vacuum"].embedded_faces) == 1


def test_volumes_are_exactly_conserved(pad_a, pad_b):
    side, depth = 300_000, 200  # grid units, micrometres (z is not scaled)
    assert total_volume(pad_a) == side * side * depth
    assert total_volume(pad_b) == side * side * depth
    vols = volume_by_name(pad_b)
    ground_area = side * side - 160_000 * 120_000
    assert vols["Ground"].shells[0].signed_volume == Fraction(ground_area) * Fraction(1, 5)


def test_airbridge_is_one_connected_ground_volume_across_three_slabs(bridge):
    vols = volume_by_name(bridge)
    assert len([v for v in bridge.topology.volumes.values() if v.domain.name == "Ground"]) == 1
    assert total_volume(bridge) == 300_000 * 300_000 * 200
    assert len(bridge.slabs) == 5
    assert vols["Ground"].shells[0].role == "outer"


def test_route_a_rejects_finite_conductors():
    with pytest.raises(RouteError):
        build(pad_in_ground("B"), route="A")


def test_sheet_conductor_overlap_of_different_nets_fails_closed():
    spec = pad_in_ground("A")
    box = Footprint.from_rings([(-1000, -1000), (1000, -1000), (1000, 1000), (-1000, 1000)])
    extra = PrismRecord("short", box, z_value(0), z_value(0), "Al", "conductor", net_id="other")
    bad = ModelSpec(spec.grid, spec.prisms + (extra,), name="bad", meta=spec.meta)
    with pytest.raises(Exception, match="overlap"):
        build(bad)


def test_conductor_areas_match_shapely_union_of_the_source_polygons():
    spec = random_layout(40, seed=11, route="B")
    result = build(spec)
    slab = next(s for s in result.slabs if s.plane == "slab[0,1/5]")
    arr = slab.arr
    by_net: dict[str, int] = {}
    for fi in range(1, len(arr.faces)):
        d = slab.labels[fi]
        if d is not None and d.kind == "conductor":
            by_net[d.net] = by_net.get(d.net, 0) + arr.face_twice_area(fi)
    for net, area2 in by_net.items():
        polygons = [p for p in spec.prisms if p.net_id == net and p.entity_id != "ground"] if net != "Ground" else [
            p for p in spec.prisms if p.entity_id == "ground"
        ]
        union = unary_union([
            Polygon(p.footprint.outer, p.footprint.holes) for p in polygons
        ])
        perimeter = sum(
            abs(r[(i + 1) % len(r)][0] - r[i][0]) + abs(r[(i + 1) % len(r)][1] - r[i][1])
            for p in polygons for r in p.footprint.rings() for i in range(len(r))
        )  # grid units; snapping moves a boundary by < 1 unit
        assert area2 / 2 == pytest.approx(union.area, abs=perimeter), net


def test_tampered_shell_is_rejected_by_validation(pad_b):
    topo = pad_b.topology
    vol = next(v for v in topo.volumes.values() if v.emitted and v.domain.name == "vacuum")
    saved = vol.shells[0].faces
    vol.shells[0].faces = saved[1:]
    try:
        report = validate(pad_b.spec, pad_b.levels, pad_b.slabs, pad_b.level_layers, topo, check_planarity=False)
        assert any(c.name == "shells_closed_and_oriented" for c in report.failures)
        with pytest.raises(ValidationError):
            report.raise_if_failed()
    finally:
        vol.shells[0].faces = saved


def test_lineage_maps_every_edge_to_source_entities(pad_b):
    topo = pad_b.topology
    names = {p.entity_id for p in pad_b.spec.prisms}
    for eid, sources in topo.lineage.items():
        assert sources
        assert {s.split("/")[0] for s in sources} <= names
    port_edges = [e for e, s in topo.lineage.items() if any(x.startswith("p1/") for x in s)]
    assert port_edges


def test_semantic_names_do_not_depend_on_native_state(pad_a):
    assert sorted(pad_a.topology.groups) == [
        "EXT::substrate", "EXT::vacuum", "PEC::Ground", "PEC::pad", "PORT::p1", "SA::substrate|vacuum",
        "VOL::substrate", "VOL::vacuum",
    ]


def test_arrangements_are_local_not_global():
    """The deck exists only in one slab: no other slab or level arrangement contains its curves."""
    result = build(airbridge())
    for layer in result.slabs + result.level_layers:
        has_deck = any(s.startswith("deck/") for src in layer.edge_sources for s in src)
        touches_deck_slab = layer.plane in ("slab[3,33/10]", "level[3]", "level[33/10]")
        assert has_deck == touches_deck_slab, layer.plane


def test_curve_families_required_and_refinement_only(tmp_path):
    from sgbproto.emit import emit
    from sgbproto.records import CurveFamilyRecord, LineCarrier, Lineage

    base = pad_in_ground("A")

    def carrier(cid, p, q):
        return LineCarrier(cid, p, q, Lineage("family_member", family_id="fam", member=cid))

    ring = [(-75_000, -20_000), (-55_000, -20_000), (-55_000, 20_000), (-75_000, 20_000)]
    port_like = CurveFamilyRecord(
        "cap", "port_footprint",
        tuple(carrier(f"cap{i}", ring[i], ring[(i + 1) % 4]) for i in range(4)),
        (Fraction(0),), "required",
    )
    refine = CurveFamilyRecord(
        "ref", "refinement", (carrier("ref0", (-40_000, 55_000), (40_000, 55_000)),), (Fraction(0),), "refinement_only"
    )
    spec = ModelSpec(base.grid, base.prisms, (port_like, refine), name="with_families", meta=base.meta)
    plain, result = build(base), build(spec)
    gap = lambda r: len(r.topology.groups["SA::substrate|vacuum"]["ids"])
    assert gap(result) == gap(plain) + 1  # the required family splits one cell into two faces of equal label
    face_with_curve = [f for f in result.topology.faces.values() if f.refinement_edges]
    assert len(face_with_curve) == 1
    out = emit(result.topology, tmp_path, "fam", mesh_size=40.0)
    assert out.mesh["conformal"], out.mesh["problems"]


def test_no_entity_is_in_two_physical_groups_and_every_emitted_entity_is_in_one(pad_a, pad_b, bridge):
    for result in (pad_a, pad_b, bridge):
        topo = result.topology
        seen = Counter((g["dim"], i) for g in topo.groups.values() for i in g["ids"])
        assert all(n == 1 for n in seen.values())
        for f in topo.faces.values():
            emitted = any(v is not None and topo.volumes[v].emitted for v in (f.neg_volume, f.pos_volume))
            assert ((2, f.id) in seen) == emitted
        assert all(((3, v.id) in seen) == v.emitted for v in topo.volumes.values())


def test_route_a_keeps_pec_as_a_primary_group_and_its_union_is_itself(pad_a):
    assert pad_a.topology.unions == {"PEC::Ground": ["PEC::Ground"], "PEC::pad": ["PEC::pad"]}


def wall_ranges(result):
    return Counter(f.z_range for f in result.topology.faces.values() if f.kind == "wall")


def with_port(x0: float, x1: float):
    """Route B pad-in-ground whose port rectangle spans x0..x1 at the metal mid-plane."""
    from sgbproto.scenes import rect_um, to_grid

    base = pad_in_ground("B")
    port = PrismRecord(
        "p1", Footprint.from_rings(to_grid(base.grid, rect_um(x0, -5, x1, 5))), z_value("0.1"), z_value("0.1"),
        "lumped_port", "dielectric", role="port",
    )
    prisms = tuple(p for p in base.prisms if p.entity_id != "p1") + (port,)
    return ModelSpec(base.grid, prisms, name="port_variant", meta=base.meta)


def test_port_edge_touching_a_metal_sidewall_cuts_that_wall_at_the_port_level():
    result = build(with_port(50, 80))  # reaches the pad wall (x = 50) and the cutout wall (x = 80)
    ranges = wall_ranges(result)
    half = (Fraction(0), Fraction(1, 10)), (Fraction(1, 10), Fraction(1, 5))
    assert ranges[half[0]] == 2 and ranges[half[1]] == 2  # one pad wall and one ground wall, each cut in two
    assert ranges[(Fraction(0), Fraction(1, 5))] == 22 - 2  # 22 metal walls (14 pad: 9 lines, 1 corner arc, 4 hole arcs; 8 ground); all but the two touched stay single


def test_port_sheet_that_touches_no_sidewall_does_not_cut_any_wall():
    result = build(with_port(55, 75))  # floating in the gap: no wall is touched
    ranges = wall_ranges(result)
    assert (Fraction(0), Fraction(1, 10)) not in ranges and (Fraction(1, 10), Fraction(1, 5)) not in ranges
    assert ranges[(Fraction(0), Fraction(1, 5))] == 22  # every metal wall is one face from z = 0 to 0.2
    assert len(result.topology.volumes) == 4 and not result.report.failures


def test_wall_coalescing_reduces_the_polygon_pad_to_68_faces_and_the_arc_pad_to_41(pad_b):
    polygon_pad = build(pad_in_ground("B", hole_sides=32, rounded_corner=False))
    assert polygon_pad.counts["faces"] == 68 < 115  # phase 1b number: 32-gon hole, no rounded corner
    assert pad_b.counts["faces"] == 41  # the true circle is 4 arc walls instead of 32 chord walls
    assert not pad_b.report.failures and not polygon_pad.report.failures


def test_coalesced_walls_stay_conformal_in_the_mesh(tmp_path):
    from sgbproto.emit import emit

    for spec in (with_port(50, 80), with_port(55, 75), airbridge()):
        out = emit(build(spec).topology, tmp_path, "m", mesh_size=60.0, mesh_size_min=2.0)
        assert out.mesh["conformal"], out.mesh["problems"]


def test_a_refinement_piece_in_no_emitted_face_is_an_error_not_a_silent_drop(tmp_path):
    from sgbproto.records import CurveFamilyRecord, LineCarrier, Lineage
    import json

    from sgbproto.errors import SgbError

    base = pad_in_ground("B")

    def curve(x0, y0, x1, y1, z):
        carrier = LineCarrier("ref0", (x0, y0), (x1, y1), Lineage("family_member", family_id="ref", member="ref0"))
        return ModelSpec(base.grid, base.prisms, (CurveFamilyRecord("ref", "refinement", (carrier,), (z,), "refinement_only"),),
                         name="refine", meta=base.meta)

    # Inside the pad metal at the port mid-plane: the cell has no face (it is inside the excluded conductor).
    with pytest.raises(SgbError, match="no face"):
        build(curve(-40_000, 20_000, -30_000, 20_000, Fraction("0.1")))
    # In the vacuum gap at z = 0 the curve lands in the substrate/vacuum interface: fine, and the embedding is
    # recorded in the gmsh-API sidecar as (edge, host face).
    from sgbproto.emit import emit

    result = build(curve(60_000, 40_000, 70_000, 40_000, Fraction(0)))
    out = emit(result.topology, tmp_path, "m")
    side = json.loads(out.sidecar_path.read_text())
    assert len(side["refinement_edges"]) == 1
    assert side["refinement_edges"][0]["host_face"] in result.topology.faces
