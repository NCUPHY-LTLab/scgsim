"""Routes A, B and C as labelling policies; decision D5 naming and the same-Net overlap priority."""

from __future__ import annotations

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

from scgsim.geometry.sgb.errors import OverlapError, RouteError
from scgsim.geometry.sgb.model import build_conformal_model, build_model_from_catalogue
from scgsim.geometry.sgb.records.labels import Domain, SheetOwner
from scgsim.geometry.sgb.records.prisms import PrismCatalogue
from scgsim.geometry.sgb.routes.naming import (
    interface_type,
    surface_epr_rows,
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
        assert (
            groups[f"VOL::{entity}"].dim == 3 and len(groups[f"VOL::{entity}"].ids) == 1
        )
    assert {"VOL::substrate", "VOL::vacuum"} <= set(groups)
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
        "MA::deck",
        "MA::post_left",
        "MS::ground_left",
        "MM::deck|post_left",
        "MM::ground_left|post_left",
    } <= set(groups)
    for name in ("MA::deck", "MS::ground_left", "MM::deck|post_left"):
        for face_id in groups[name].ids:
            face = topology.faces[face_id]
            assert (
                face.neg_volume in routed.emitted_volumes
                and face.pos_volume in routed.emitted_volumes
            )
    # the Net union lists the exposed conductor surfaces, never the internal MM faces
    union = routed.unions["NET::Ground"]
    assert "MA::deck" in union and not any(name.startswith("MM::") for name in union)
    assert routed.unions["MM"] == sorted(n for n in groups if n.startswith("MM::"))


def test_route_c_conductor_reaching_the_model_boundary_has_an_ext_group(bridge_c):
    """The ground halves run to the chip edge: those faces are EXT::<entity>."""
    assert {"EXT::ground_left", "EXT::ground_right"} <= set(bridge_c.routed.groups)


def test_route_c_rejects_sheets_and_lumped_port_sheets():
    """Route C keeps conductor volumes: no sheets, no port sheets."""
    with pytest.raises(
        RouteError,
        match=r"Route C keeps conductor volumes.*lumped-port sheets \['p1'\]",
    ):
        build_model_from_catalogue(pad_in_ground("B"), "C")
    with pytest.raises(RouteError, match=r"sheet conductors \['ground', 'pad'\]"):
        build_model_from_catalogue(pad_in_ground("A", with_port=False), "C")


def test_route_c_from_a_real_input_names_the_gds_entities(tmp_path):
    """Real source input (S0 and S1), Route C: groups carry the qualified entity ids."""
    effective = gds_pad_in_ground(tmp_path, with_port=False)
    model = build_conformal_model(effective.build_input, effective.representation, "C")
    groups = set(model.routed.groups)
    assert {
        "VOL::ground_0000",
        "VOL::pad",
        "MA::pad",
        "MS::pad",
        "MA::ground_0000",
    } <= groups
    assert model.routed.unions["NET::P"] == ["MA::pad", "MS::pad"]
    assert model.routed.unions["NET::G"] == ["MA::ground_0000", "MS::ground_0000"]


def test_route_a_rejects_the_finite_conductors_of_a_real_solid_input(tmp_path):
    """The solid profile leaves conductors finite after S1, so Route A names them."""
    effective = gds_pad_in_ground(tmp_path)
    with pytest.raises(RouteError, match=r"still finite.*ground_0000, pad"):
        build_conformal_model(effective.build_input, effective.representation, "A")


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


def test_thin_film_input_builds_under_route_a_and_b_and_is_rejected_by_route_c():
    """Sheets come from the profile: A and B accept them, C has no sheets."""
    effective = thin_film_pair(0.5)
    model_a = build_conformal_model(
        effective.build_input, effective.representation, "A"
    )
    assert {"SHEET::chip/left", "SHEET::chip/right", "PORT::port_sheet"} <= set(
        model_a.routed.groups
    )
    assert model_a.routed.unions["NET::left"] == ["SHEET::chip/left"]
    model_b = build_conformal_model(
        effective.build_input, effective.representation, "B"
    )
    assert (
        model_b.routed.groups.keys() == model_a.routed.groups.keys()
    )  # sheets are PEC faces in both routes
    with pytest.raises(RouteError, match="Route C keeps conductor volumes"):
        build_conformal_model(effective.build_input, effective.representation, "C")


def test_a_rectangular_junction_builds_with_a_conformal_port_between_its_terminals():
    """Rectangular-junction fixture: the central port rectangle touches the terminal-owned ends."""
    effective = rectangular_junction_pair()
    model = build_conformal_model(effective.build_input, effective.representation, "A")
    assert not model.report.failures
    groups = model.routed.groups
    assert len(groups["PORT::sheet"].ids) == 1
    assert {"SHEET::chip/left", "SHEET::chip/right"} <= set(groups)
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
        assert (
            len(groups["SHEET::padL"].ids) == 1 and len(groups["SHEET::padR"].ids) == 1
        )
    else:
        assert {"MA::padL", "MA::padR", "MS::padL", "MS::padR"} <= set(groups)
    # the Net union is the only place the two Entities meet
    net_union = model.routed.unions["NET::N"]
    assert len(net_union) == (2 if route == "A" else 4)


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
    face = topology.faces[model.routed.groups["SHEET::padR"].ids[0]]
    assert face.membership["padR"] == pytest.approx(1.0)
    assert face.membership["padL"] == pytest.approx(
        0.2
    )  # the 2 x 10 um overlap is a fifth of padR's 100 um^2
    assert face.sources == ("padL", "padR")
    assert topology.faces[model.routed.groups["SHEET::padL"].ids[0]].membership[
        "padL"
    ] == pytest.approx(1.0)


def test_equal_priority_overlap_of_two_entities_of_one_net_fails_closed_naming_both():
    """No silent winner."""
    with pytest.raises(
        OverlapError, match=r"same-Net entities padL, padR .*equal priority"
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
        (V, S, None, (), "SA::substrate|air"),
        (S, V, None, (), "SA::substrate|air"),
        (M, V, None, (), "MA::chip/q0/pad"),
        (S, M, None, (), "MS::chip/q0/pad"),
        (M, V, None, (("fam", 2),), "MA::chip/q0/pad::band=2"),
        (M2, M, None, (), "MM::chip/q0/arm|chip/q0/pad"),
        (V, Domain("vacuum", "vacuum", owner="air2"), None, (), "AA::air|air2"),
        (
            S,
            Domain("dielectric", "SiO2", owner="oxide"),
            None,
            (),
            "SS::oxide|substrate",
        ),
        (V, V, SheetOwner("conductor", "chip/q0/pad", "N"), (), "SHEET::chip/q0/pad"),
        (
            V,
            S,
            SheetOwner("conductor", "chip/q0/pad", "N"),
            (("fam", 1),),
            "SHEET::chip/q0/pad::band=1",
        ),
        (V, V, SheetOwner("port", "chip/q0/jj"), (("fam", 1),), "PORT::chip/q0/jj"),
        (None, V, None, (), "EXT::air"),
        (S, None, None, (), "EXT::substrate"),
        (None, None, None, (), None),
    ],
)
def test_primary_group_names_follow_kind_double_colon_qualified_entity(
    neg, pos, owner, bands, name
):
    """``<kind>::<qualified entity id>[::band=k]``: a face's name from the two sides alone."""
    assert surface_group(neg, pos, owner, bands) == name


def test_volume_groups_and_interface_types_and_surface_epr_rows():
    """VOL names, interface-type prefixes, and the rows a banded Route A sheet carries."""
    assert volume_group(M) == "VOL::chip/q0/pad"
    assert interface_type("MA::chip/q0/pad::band=2") == "MA"
    rows = surface_epr_rows(
        [
            "SHEET::pad::band=0",
            "SHEET::pad::band=1",
            "SA::a|b::band=0",
            "MA::x",
            "EXT::air",
        ]
    )
    assert rows == {
        "MA::pad::band=0": "SHEET::pad::band=0",
        "MS::pad::band=0": "SHEET::pad::band=0",
        "MA::pad::band=1": "SHEET::pad::band=1",
        "MS::pad::band=1": "SHEET::pad::band=1",
        "SA::a|b::band=0": "SA::a|b::band=0",
    }
