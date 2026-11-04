"""Routes A, B and C as labelling policies; decision D5 naming and the same-Net overlap priority."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest
from sgb_inputs import gds_pad_in_ground, rectangular_junction_pair, thin_film_pair
from sgb_scenes import (
    GRID,
    airbridge,
    chip_prisms,
    conductor,
    pad_in_ground,
    rect_um,
)

from scgsim.geometry.sgb.errors import InputError, OverlapError, RouteError
from scgsim.geometry.sgb.model import build_conformal_model, build_model_from_catalogue
from scgsim.geometry.sgb.records.labels import Domain, SheetOwner
from scgsim.geometry.sgb.records.prisms import PrismCatalogue
from scgsim.geometry.sgb.routes.naming import (
    check_entity_names,
    interface_type_of,
    surface_epr_row_ids,
    surface_group,
    volume_group,
)

# --------------------------------------------------------------------------- Route C


@pytest.fixture(scope="module")
def bridge_c():
    """The airbridge with its conductors retained and meshed."""
    return build_model_from_catalogue(airbridge(), "C")


def test_route_c_emits_every_volume_including_the_conductors(bridge_c):
    """Retained conductor volumes: each Entity is an emitted volume with a VOL group of its own."""
    topology = bridge_c.topology
    assert bridge_c.routed.emitted_volumes == frozenset(topology.volumes)
    groups = bridge_c.routed.groups
    for entity in ("deck", "ground_left", "ground_right", "post_left", "post_right"):
        assert groups[entity].dim == 3 and len(groups[entity].ids) == 1
    assert {"substrate", "vacuum"} <= set(groups)
    total = sum(
        (s.signed_volume for v in topology.volumes.values() for s in v.shells),
        Fraction(0),
    )
    assert total == 300_000 * 300_000 * 200


def test_route_c_conductor_interfaces_are_shared_faces_with_per_entity_names(bridge_c):
    """Conductor-vacuum, conductor-substrate and conductor-conductor faces each have two emitted sides."""
    topology, routed = bridge_c.topology, bridge_c.routed
    groups = routed.groups
    assert {
        "MA::deck::top",
        "MA::post_left::sidewall",
        "MS::ground_left::bottom",
        "MM::deck|post_left::interface",
        "MM::ground_left|post_left::interface",
    } <= set(groups)
    for name in (
        "MA::deck::top",
        "MS::ground_left::bottom",
        "MM::deck|post_left::interface",
    ):
        for face_id in groups[name].ids:
            face = topology.faces[face_id]
            assert (
                face.neg_volume in routed.emitted_volumes
                and face.pos_volume in routed.emitted_volumes
            )
    # the Net union lists the exposed conductor surfaces, never the internal MM faces
    union = routed.unions["NET::Ground"]
    assert "MA::deck::top" in union and not any(
        name.startswith("MM::") for name in union
    )
    assert routed.unions["MM"] == sorted(n for n in groups if n.startswith("MM::"))


def test_route_c_conductor_reaching_the_model_boundary_has_an_ext_group(bridge_c):
    """The ground halves run to the chip edge: those faces are EXT::<entity>::<face kind>."""
    assert {"EXT::ground_left::sidewall", "EXT::ground_right::sidewall"} <= set(
        bridge_c.routed.groups
    )


def test_route_c_rejects_conductor_sheets_but_supports_lumped_port_sheets():
    """Route C keeps conductor volumes: sheets have no meaning, a port between two retained conductors is normal."""
    with pytest.raises(
        RouteError, match=r"Route C keeps conductors finite.*ground, pad"
    ):
        build_model_from_catalogue(pad_in_ground("A", with_port=False), "C")
    model = build_model_from_catalogue(pad_in_ground("B"), "C")
    assert not model.report.failures
    (port_id,) = model.routed.groups["PORT::p1"].ids
    port = model.topology.faces[port_id]
    # an embedded face in the vacuum gap at the port plane, not a boundary of any shell
    assert port.neg_volume == port.pos_volume is not None
    assert port.neg_volume in model.topology.volumes
    assert [
        fid for v in model.topology.volumes.values() for fid in v.embedded_faces
    ] == [port_id]
    # its edges are shared with the conductor walls cut at the port plane
    walls = [
        f
        for f in model.topology.faces.values()
        if f.kind == "wall" and f.z_range[1] == Fraction(1, 10)
    ]
    port_edges = {eid for loop in port.loops for eid, _ in loop}
    assert any(port_edges & {eid for loop in w.loops for eid, _ in loop} for w in walls)


def test_route_b_rejects_conductor_sheets_but_accepts_lumped_port_sheets():
    """Today's contract: Route B knows only cutout shells; its ports are mid-plane embedded faces."""
    with pytest.raises(
        RouteError, match=r"Route B keeps conductors finite.*ground, pad"
    ):
        build_model_from_catalogue(pad_in_ground("A", with_port=False), "B")
    assert (
        "PORT::p1" in build_model_from_catalogue(pad_in_ground("B"), "B").routed.groups
    )


def test_route_c_from_a_real_input_names_the_gds_entities(tmp_path):
    """Real source input (S0 and S1), Route C: groups carry the qualified entity ids."""
    effective = gds_pad_in_ground(tmp_path, with_port=False)
    model = build_conformal_model(effective.build_input, effective.representation, "C")
    groups = set(model.routed.groups)
    assert {
        "ground_0000",
        "pad",
        "MA::pad::top",
        "MS::pad::bottom",
        "MA::ground_0000::top",
    } <= groups
    assert model.routed.unions["NET::P"] == [
        "MA::pad::sidewall",
        "MA::pad::top",
        "MS::pad::bottom",
    ]
    assert model.routed.unions["NET::G"] == [
        "MA::ground_0000::sidewall",
        "MA::ground_0000::top",
        "MS::ground_0000::bottom",
    ]


def test_route_a_accepts_the_undeclared_finite_conductors_of_a_real_solid_input(
    tmp_path,
):
    """No physical declaration: today's contract lets Route A keep finite conductors as cutout shells."""
    effective = gds_pad_in_ground(tmp_path)
    model = build_conformal_model(effective.build_input, effective.representation, "A")
    emitted = {
        model.topology.volumes[v].domain.name for v in model.routed.emitted_volumes
    }
    assert emitted == {"substrate", "VACUUM_REGION"}
    assert {"MA::pad::top", "MS::pad::bottom", "MA::ground_0000::top"} <= set(
        model.routed.groups
    )


def test_route_a_rejects_declared_films_that_the_solid_profile_left_finite():
    """The declaration says the layer is a film; ``solid`` keeps it 0.25 um thick, so Route A refuses."""
    effective = thin_film_pair(0.5, profile="solid")
    with pytest.raises(
        RouteError, match=r"films are still finite.*chip/left, chip/right"
    ):
        build_conformal_model(effective.build_input, effective.representation, "A")


def test_a_finite_bump_stands_on_a_thin_film_sheet_in_route_a():
    """Face-metal films are sheets, the ``finite_volume`` bump keeps its volume (excluded like Route B)."""
    effective = thin_film_pair(0.5, bump=True)
    model = build_conformal_model(effective.build_input, effective.representation, "A")
    assert not model.report.failures
    groups = model.routed.groups
    assert {
        "chip/left",
        "chip/right",
        "MA::chip/bump::top",
        "PORT::port_sheet",
    } <= set(groups)
    bump = next(
        v for v in model.topology.volumes.values() if v.domain.name == "chip/bump"
    )
    assert (
        bump.id not in model.routed.emitted_volumes
    )  # an excluded finite volume with PEC boundary faces
    assert model.routed.unions["NET::left"] == [
        "MA::chip/bump::sidewall",
        "MA::chip/bump::top",
        "chip/left",
    ]
    kinds = {
        p.entity_id: p.feature_kind
        for p in model.labelled.catalogue.prisms
        if p.material_kind == "conductor"
    }
    assert kinds == {
        "chip/left": "stack_film",
        "chip/right": "stack_film",
        "chip/bump": "finite_volume",
    }


def test_the_bump_without_the_film_collapse_is_a_finite_input_for_routes_b_and_c():
    """Profile ``solid``: films and bump are all finite; Routes B and C accept them and keep the port."""
    effective = thin_film_pair(0.5, profile="solid", bump=True)
    for route in ("B", "C"):
        model = build_conformal_model(
            effective.build_input, effective.representation, route
        )
        assert {"MA::chip/bump::top", "PORT::port_sheet"} <= set(model.routed.groups)
        assert not model.report.failures


def test_route_b_from_a_real_input_excludes_conductors_and_embeds_the_port(tmp_path):
    """Route B on the GDS input: the port at the 0.1 um mid-plane is an embedded sheet of the vacuum."""
    effective = gds_pad_in_ground(tmp_path)
    model = build_conformal_model(effective.build_input, effective.representation, "B")
    volumes = {v.domain.name: v for v in model.topology.volumes.values()}
    emitted = {
        model.topology.volumes[v].domain.name for v in model.routed.emitted_volumes
    }
    assert emitted == {"substrate", "VACUUM_REGION"}
    assert "PORT::PORT_SHEET__P1__0000" in model.routed.groups
    assert len(volumes["VACUUM_REGION"].embedded_faces) == 1
    assert not model.report.failures


# --------------------------------------------------------------------------- real thin-film input


def test_thin_film_sheets_build_under_route_a_only():
    """Sheets come from the profile: Route A accepts them, B and C have no conductor sheets."""
    effective = thin_film_pair(0.5)
    model_a = build_conformal_model(
        effective.build_input, effective.representation, "A"
    )
    assert {
        "chip/left",
        "chip/right",
        "PORT::port_sheet",
    } <= set(model_a.routed.groups)
    assert model_a.routed.unions["NET::left"] == ["chip/left"]
    for route in ("B", "C"):
        with pytest.raises(
            RouteError, match=rf"Route {route} keeps conductors finite.*chip/left"
        ):
            build_conformal_model(
                effective.build_input, effective.representation, route
            )


def test_a_rectangular_junction_builds_with_a_conformal_port_between_its_terminals():
    """Rectangular-junction fixture: the central port rectangle touches the terminal-owned ends."""
    effective = rectangular_junction_pair()
    model = build_conformal_model(effective.build_input, effective.representation, "A")
    assert not model.report.failures
    groups = model.routed.groups
    assert len(groups["PORT::sheet"].ids) == 1
    assert {"chip/left", "chip/right"} <= set(groups)
    port = model.topology.faces[groups["PORT::sheet"].ids[0]]
    assert (
        port.twice_xy_area == 2 * 32
    )  # 8 um^2 on the 0.5 um grid (0.25 um^2 per cell)


# --------------------------------------------------------------------------- decision D5


def adjoining_pads(
    route: str, *, priorities=(0, 0), overlap: float = 0.0
) -> PrismCatalogue:
    """Two pads of one Net that share an edge (or overlap by ``overlap`` um)."""
    z1 = 0 if route == "A" else "0.2"
    prisms = [
        *chip_prisms(GRID, 40.0),
        conductor(
            "padL", rect_um(-10, 0, 0 + overlap, 10), 0, z1, "N", priority=priorities[0]
        ),
        conductor("padR", rect_um(0, 0, 10, 10), 0, z1, "N", priority=priorities[1]),
    ]
    return PrismCatalogue(GRID, tuple(prisms))


@pytest.mark.parametrize("route", ["A", "B", "C"])
def test_seams_between_entities_of_one_net_stay_required_edges(route):
    """Two Entities on one Net keep their seam: two faces per side, each with its own group."""
    catalogue = adjoining_pads(route)
    model = build_model_from_catalogue(catalogue, route)
    groups = model.routed.groups
    if route == "A":
        assert len(groups["padL"].ids) == 1 and len(groups["padR"].ids) == 1
    else:
        assert {
            "MA::padL::top",
            "MA::padR::top",
            "MS::padL::bottom",
            "MS::padR::bottom",
        } <= set(groups)
    # the Net union is the only place the two Entities meet
    net_union = model.routed.unions["NET::N"]
    assert len(net_union) == (
        2 if route == "A" else 6
    )  # per Entity: top, bottom, sidewall


def test_the_overlap_of_two_entities_of_one_net_belongs_to_the_higher_priority():
    """Route A: the overlap strip is part of the SHEET face of the entity that declares the higher priority."""
    model = build_model_from_catalogue(
        adjoining_pads("A", priorities=(1, 2), overlap=2.0), "A"
    )
    level = model.labelled.level_layers[1]
    owned = {
        level.labels[f].sheet.name: level.arr.face_twice_area(f)
        for f in range(1, len(level.arr.faces))
        if level.labels[f].sheet is not None
    }
    um2 = 1_000_000  # grid^2 per um^2
    assert (
        owned["padL"] == 2 * (12 * 10 - 2 * 10) * um2
    )  # padL is 12 x 10 um, minus the 2 x 10 um overlap
    assert owned["padR"] == 2 * 100 * um2


def test_face_lineage_records_the_overlap_share_of_the_entity_that_lost_it():
    """Membership travels with the face: the winning face lists the share of the loser it covers."""
    model = build_model_from_catalogue(
        adjoining_pads("A", priorities=(1, 2), overlap=2.0), "A"
    )
    topology = model.topology
    face = topology.faces[model.routed.groups["padR"].ids[0]]
    assert face.membership["padR"] == pytest.approx(1.0)
    assert face.membership["padL"] == pytest.approx(
        0.2
    )  # the 2 x 10 um overlap is a fifth of padR's 100 um^2
    assert face.sources == ("padL", "padR")
    assert topology.faces[model.routed.groups["padL"].ids[0]].membership[
        "padL"
    ] == pytest.approx(1.0)


def test_equal_priority_overlap_of_two_entities_of_one_net_fails_closed_naming_both():
    """No silent winner."""
    with pytest.raises(
        OverlapError,
        match=r"same-Net entities padL and padR overlap near \(.*\) um .*declare distinct entity priorities",
    ):
        build_model_from_catalogue(adjoining_pads("B", overlap=2.0), "B")


def test_route_b_volumes_of_overlapping_entities_are_separate_with_a_shared_excluded_face():
    """The face between the two excluded conductor volumes is not emitted (no group)."""
    model = build_model_from_catalogue(
        adjoining_pads("B", priorities=(1, 2), overlap=2.0), "B"
    )
    internal = [
        f
        for f in model.topology.faces.values()
        if f.neg_domain is not None
        and f.pos_domain is not None
        and f.neg_domain.kind == f.pos_domain.kind == "conductor"
    ]
    assert internal and all(
        model.routed.face_group[f.id].startswith("MM::") for f in internal
    )
    assert not any(name.startswith("MM::") for name in model.routed.groups)


# --------------------------------------------------------------------------- names


V = Domain("vacuum", "vacuum", owner="air")
S = Domain("dielectric", "Si", owner="substrate")
M = Domain("conductor", "Al", "N", "chip/q0/pad")
M2 = Domain("conductor", "Al", "N", "chip/q0/arm")


@pytest.mark.parametrize(
    ("neg", "pos", "owner", "bands", "name"),
    [
        (V, S, None, (), "SA::substrate|air::interface"),
        (S, V, None, (), "SA::substrate|air::interface"),
        (M, V, None, (), "MA::chip/q0/pad::top"),
        (V, M, None, (), "MA::chip/q0/pad::bottom"),
        (S, M, None, (), "MS::chip/q0/pad::bottom"),
        (M, S, None, (), "MS::chip/q0/pad::top"),
        (M, V, None, (("fam", 2),), "MA::chip/q0/pad::top::band=2"),
        (M2, M, None, (), "MM::chip/q0/arm|chip/q0/pad::interface"),
        (
            V,
            Domain("vacuum", "vacuum", owner="air2"),
            None,
            (),
            "AA::air|air2::interface",
        ),
        (
            S,
            Domain("dielectric", "SiO2", owner="oxide"),
            None,
            (),
            "SS::oxide|substrate::interface",
        ),
        (
            V,
            V,
            SheetOwner("conductor", "chip/q0/pad", "N"),
            (),
            "chip/q0/pad",
        ),
        (
            V,
            S,
            SheetOwner("conductor", "chip/q0/pad", "N"),
            (("fam", 1),),
            "chip/q0/pad::band=1",
        ),
        (V, V, SheetOwner("port", "chip/q0/jj"), (("fam", 1),), "PORT::chip/q0/jj"),
        (None, V, None, (), "EXT::air::bottom"),
        (S, None, None, (), "EXT::substrate::top"),
        (None, None, None, (), None),
    ],
)
def test_primary_group_names_follow_kind_double_colon_qualified_entity(
    neg, pos, owner, bands, name
):
    """``<kind>::<qualified entity id>::<face kind>[::band=k]``, or the bare entity id of a Route A sheet: a face's name from the two sides alone."""
    assert surface_group(neg, pos, owner, bands) == name


def test_a_vertical_face_is_a_sidewall_whatever_its_sides():
    """Walls carry the face kind ``sidewall``; a conductor wall against vacuum is MA, a sheet has no wall."""
    assert surface_group(M, V, None, vertical=True) == "MA::chip/q0/pad::sidewall"
    assert surface_group(S, V, None, vertical=True) == "SA::substrate|air::sidewall"
    assert surface_group(None, V, None, vertical=True) == "EXT::air::sidewall"
    assert surface_group(M2, M, None, (("fam", 1),), vertical=True) == (
        "MM::chip/q0/arm|chip/q0/pad::sidewall::band=1"
    )


def test_volume_groups_and_interface_types_and_surface_epr_rows():
    """A volume is its bare entity id; the interface type comes from the structure, never from the name."""
    assert volume_group(M) == "chip/q0/pad"
    sheet = SheetOwner("conductor", "pad", "N")
    assert interface_type_of(M, V, None) == "MA"
    assert interface_type_of(V, V, sheet) == "SHEET"
    assert interface_type_of(V, V, SheetOwner("port", "p1")) == "PORT"
    assert interface_type_of(None, V, None) == "EXT"
    assert interface_type_of(None, None, None) is None
    # a banded Route A sheet carries both rows; every other banded group carries its own name
    assert surface_epr_row_ids("pad::band=1", "SHEET", "pad", 1) == [
        "MA::pad::sheet::band=1",
        "MS::pad::sheet::band=1",
    ]
    assert surface_epr_row_ids("SA::a|b::interface::band=0", "SA", "", 0) == [
        "SA::a|b::interface::band=0"
    ]
    assert surface_epr_row_ids("pad", "SHEET", "pad", None) == []  # unbanded: no row
    assert surface_epr_row_ids("PORT::p1", "PORT", "p1", 1) == []


@pytest.mark.parametrize(
    "entity_id",
    ["MA", "MS", "MM", "SA", "AA", "SS", "EXT", "PORT", "a::b", "chip/MA::x"],
)
def test_an_entity_id_that_would_make_a_group_name_ambiguous_is_rejected(entity_id):
    """An id equal to an interface prefix, or containing ``::``, fails closed with the id in the message."""
    with pytest.raises(InputError, match="entity id"):
        check_entity_names(["substrate", entity_id])


def test_ordinary_entity_ids_are_accepted():
    """Qualified ids with slashes, and ids that merely start with a prefix, are fine."""
    check_entity_names(["chip/left", "MA_pad", "SAx", "ground_0000", "vacuum"])


def test_the_model_build_rejects_a_reserved_entity_id():
    """The check runs on the catalogue, so hand-made prisms and real input alike are covered."""
    catalogue = pad_in_ground("C")
    prisms = tuple(
        replace(p, entity_id="MA") if p.entity_id == "pad" else p
        for p in catalogue.prisms
    )
    with pytest.raises(InputError, match="'MA'"):
        build_model_from_catalogue(PrismCatalogue(catalogue.grid, prisms), "C")
