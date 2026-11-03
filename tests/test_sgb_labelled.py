"""Levels, slabs, active sets, area-weighted membership and contacts (design brief sections 2 and 14)."""

from __future__ import annotations

from fractions import Fraction

import pytest
from sgb_scenes import (
    GRID,
    airbridge,
    chip_prisms,
    conductor,
    footprint,
    pad_in_ground,
    rect_um,
    slab_model,
)

from scgsim.geometry.sgb.derivation.labelled import derive_from_catalogue
from scgsim.geometry.sgb.errors import ContactError, NonManifoldError, OverlapError
from scgsim.geometry.sgb.prisms.levels import build_levels
from scgsim.geometry.sgb.records.grid import Grid, exact_z
from scgsim.geometry.sgb.records.prisms import Footprint, PrismCatalogue, PrismRecord

# --------------------------------------------------------------------------- levels


def test_route_b_levels_include_the_port_mid_plane_and_sheets_sit_on_levels():
    """Route b levels include the port mid plane and sheets sit on levels."""
    catalogue = pad_in_ground("B")
    levels = build_levels(catalogue.prisms)
    assert levels.levels == (
        Fraction(-100),
        Fraction(0),
        Fraction(1, 10),
        Fraction(1, 5),
        Fraction(100),
    )
    names = {k: p.entity_id for k, p in enumerate(catalogue.prisms)}
    port_level = levels.levels.index(Fraction(1, 10))
    assert [names[k] for k in levels.level_sheets[port_level]] == ["p1"]
    # ground and pad are active in both slabs of the metal layer, vacuum in all three above the substrate
    assert {names[k] for k in levels.slab_active[1]} == {"vacuum", "ground", "pad"}
    assert {names[k] for k in levels.slab_active[2]} == {"vacuum", "ground", "pad"}
    assert {names[k] for k in levels.slab_active[0]} == {"substrate"}


def test_route_a_sheets_are_never_active_in_a_slab():
    """Route a sheets are never active in a slab."""
    catalogue = pad_in_ground("A")
    levels = build_levels(catalogue.prisms)
    assert levels.levels == (Fraction(-100), Fraction(0), Fraction(100))
    assert all(
        not catalogue.prisms[k].is_sheet for slab in levels.slab_active for k in slab
    )
    assert len(levels.level_sheets[1]) == 3  # ground, pad, port


def test_airbridge_has_five_slabs():
    """Airbridge has five slabs."""
    levels = build_levels(airbridge().prisms)
    assert levels.levels == tuple(
        Fraction(x) for x in ("-100", "0", "0.2", "3", "3.3", "100")
    )


def test_slabs_thinner_than_the_grid_are_kept_as_declared():
    """Levels are the exact declared Z values: no threshold drops a 0.4 nm slab."""
    prisms = chip_prisms(GRID, 40.0)
    prisms.append(conductor("thin", rect_um(-5, -5, 5, 5), 0, "0.0004", "T"))
    result = derive_from_catalogue(PrismCatalogue(GRID, tuple(prisms)))
    assert Fraction(1, 2500) in result.levels.levels
    assert result.levels.slab_count == 3
    thin_slab = result.slabs[1]
    assert thin_slab.z == (Fraction(0), Fraction(1, 2500))
    assert any(d is not None and d.kind == "conductor" for d in thin_slab.labels)


def test_a_model_of_sheets_only_has_one_level_and_no_slab():
    """Without bodies there are no slabs; the sheets live on their single level."""
    sheet = conductor("a", rect_um(0, 0, 10, 10), 0, 0, "N")
    result = derive_from_catalogue(PrismCatalogue(GRID, (sheet,)))
    assert result.levels.levels == (Fraction(0),) and result.slabs == []
    assert len(result.level_layers) == 1 and len(result.level_layers[0].arr.faces) == 2


# --------------------------------------------------------------------------- membership

UM2 = 1_000_000  # grid^2 per um^2 on the 1 nm grid


def tab_model(route: str) -> PrismCatalogue:
    """A 20x20 um pad with an overlapping same-Net 10x6 um tab (it sticks out 5 um on the right).

    The tab entity declares the higher priority, so it owns the 5 x 6 um overlap (decision D5).
    """
    z1 = 0 if route == "A" else "0.2"
    prisms = chip_prisms(GRID, 40.0)
    prisms.append(conductor("pad", rect_um(-10, -10, 10, 10), 0, z1, "P"))
    prisms.append(conductor("pad_tab", rect_um(5, -3, 15, 3), 0, z1, "P", priority=1))
    return PrismCatalogue(GRID, tuple(prisms))


def test_each_entity_keeps_its_own_face_and_membership_records_the_overlap():
    """Per-Entity attribution: two faces; the face that lost the overlap still lists the tab's share."""
    result = derive_from_catalogue(tab_model("A"))
    layer = result.level_layers[1]
    owned = {
        layer.labels[fi].sheet.name: fi
        for fi in range(1, len(layer.arr.faces))
        if layer.labels[fi].sheet is not None
    }
    assert set(owned) == {"pad", "pad_tab"}
    pad_face, tab_face = (
        layer.membership[owned["pad"]],
        layer.membership[owned["pad_tab"]],
    )
    assert layer.arr.face_twice_area(owned["pad"]) == 2 * (400 - 30) * UM2
    assert layer.arr.face_twice_area(owned["pad_tab"]) == 2 * 60 * UM2
    assert pad_face["pad"] == pytest.approx(1.0) and "pad_tab" not in pad_face
    assert tab_face["pad_tab"] == pytest.approx(1.0)
    assert tab_face["pad"] == pytest.approx(
        30 / 60, abs=1e-9
    )  # half of the tab lies over the pad
    assert tab_face["vacuum"] == pytest.approx(1.0)
    assert sum(tab_face.values()) > 1.0  # non-exclusive


def test_slab_cells_carry_the_union_of_their_raw_cells():
    """Dissolution unions the raw coverage: the pad cell is the pad minus the overlap, still all pad."""
    result = derive_from_catalogue(tab_model("B"))
    slab = result.slabs[1]
    metal = {
        slab.labels[fi].owner: fi
        for fi in range(1, len(slab.arr.faces))
        if slab.labels[fi] is not None and slab.labels[fi].kind == "conductor"
    }
    assert slab.membership[metal["pad_tab"]]["pad"] == pytest.approx(0.5, abs=1e-9)
    assert slab.membership[metal["pad_tab"]]["pad_tab"] == pytest.approx(1.0)
    assert slab.membership[metal["pad"]]["pad"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- contacts


@pytest.mark.parametrize("route", ["A", "B"])
def test_two_nets_sharing_an_edge_fail_naming_both_entities_and_the_location(route):
    """Two nets sharing an edge fail naming both entities and the location."""
    catalogue = slab_model(
        route, [("padL", "N1", (0, 0, 10, 10)), ("padR", "N2", (10, 0, 20, 10))]
    )
    with pytest.raises(ContactError) as error:
        derive_from_catalogue(catalogue)
    text = str(error.value)
    assert "padL" in text and "padR" in text and "N1" in text and "N2" in text
    assert "10.0000" in text  # x of the shared edge's end points


@pytest.mark.parametrize("route", ["A", "B"])
def test_two_nets_sharing_only_a_vertex_fail(route):
    """Two nets sharing only a vertex fail."""
    catalogue = slab_model(
        route, [("padL", "N1", (0, 0, 10, 10)), ("padR", "N2", (10, 10, 20, 20))]
    )
    with pytest.raises(ContactError, match=r"padL.*padR|padR.*padL"):
        derive_from_catalogue(catalogue)
    with pytest.raises(ContactError, match=r"\(10\.0000, 10\.0000\)"):
        derive_from_catalogue(catalogue)


def test_a_source_gap_closed_by_a_coarse_grid_fails_but_the_fine_grid_passes():
    """A source gap closed by a coarse grid fails but the fine grid passes."""
    rects = [
        ("padL", "N1", (0, 0, 10, 10)),
        ("padR", "N2", (10.004, 0, 20, 10)),
    ]  # a 4 nm gap
    assert derive_from_catalogue(slab_model("A", rects, Grid(0.001))).level_layers
    with pytest.raises(ContactError, match=r"grid g = 0.01"):
        derive_from_catalogue(
            slab_model("A", rects, Grid(0.01))
        )  # both edges snap to x = 10


def test_a_post_of_another_net_standing_on_metal_is_a_contact():
    """A post of another net standing on metal is a contact."""
    catalogue = slab_model("B", [("ground", "G", (-30, -30, 30, 30))])
    post = conductor("post", rect_um(-5, -5, 5, 5), "0.2", 3, "X")
    with pytest.raises(ContactError, match=r"ground.*post|post.*ground"):
        derive_from_catalogue(PrismCatalogue(GRID, (*catalogue.prisms, post)))


def test_different_net_conductors_that_overlap_fail_while_labelling():
    """Different net conductors that overlap fail while labelling."""
    catalogue = slab_model(
        "B", [("a", "N1", (0, 0, 10, 10)), ("b", "N2", (5, 5, 15, 15))]
    )
    with pytest.raises(OverlapError, match=r"a, b"):
        derive_from_catalogue(catalogue)


@pytest.mark.parametrize("route", ["A", "B"])
def test_same_net_contact_passes(route):
    """Same net contact passes."""
    catalogue = slab_model(
        route, [("padL", "N1", (0, 0, 10, 10)), ("padR", "N1", (10, 0, 20, 10))]
    )
    assert derive_from_catalogue(catalogue).level_layers


def test_same_net_vertex_contact_passes_for_sheets():
    """Two Route-A sheets of one Net meeting at a corner are two faces sharing a vertex: legal."""
    catalogue = slab_model(
        "A", [("padL", "N1", (0, 0, 10, 10)), ("padR", "N1", (10, 10, 20, 20))]
    )
    assert derive_from_catalogue(catalogue).level_layers


def test_a_same_net_vertex_contact_of_finite_metal_is_a_dedicated_nonmanifold_error():
    """A same net vertex contact of finite metal is a dedicated nonmanifold error."""
    catalogue = slab_model(
        "B", [("padL", "N1", (0, 0, 10, 10)), ("padR", "N1", (10, 10, 20, 20))]
    )
    with pytest.raises(NonManifoldError) as error:
        derive_from_catalogue(catalogue)
    text = str(error.value)
    for needle in ("N1", "padL", "padR", "(10.0000, 10.0000)", "vertical edge"):
        assert needle in text


def test_a_hole_touching_its_outer_boundary_is_a_nonmanifold_pinch_for_finite_metal_but_fine_for_a_sheet():
    """A hole touching its outer boundary is a nonmanifold pinch for finite metal but fine for a sheet."""

    def catalogue(route: str) -> PrismCatalogue:
        """Catalogue."""
        z1 = 0 if route == "A" else "0.2"
        ring = footprint(
            rect_um(-10, -10, 10, 10),
            [[(10.0, 0.0), (5.0, -5.0), (0.0, 0.0), (5.0, 5.0)]],
        )
        prisms = chip_prisms(GRID, 40.0) + [
            PrismRecord(
                "ring", ring, exact_z(0), exact_z(z1), "Al", "conductor", net_id="N"
            )
        ]
        return PrismCatalogue(GRID, tuple(prisms))

    with pytest.raises(NonManifoldError, match=r"ring.*\(10\.0000, 0\.0000\)"):
        derive_from_catalogue(catalogue("B"))
    assert derive_from_catalogue(catalogue("A")).level_layers


def test_a_conductor_prism_needs_a_net():
    """A conductor prism needs a net."""
    with pytest.raises(ValueError, match="needs a net_id"):
        PrismRecord(
            "x",
            Footprint.from_rings([(0, 0), (1, 0), (1, 1)]),
            exact_z(0),
            exact_z(1),
            "Al",
            "conductor",
        )
