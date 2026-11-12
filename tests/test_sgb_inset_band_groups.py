"""A band group never mixes two families: the dielectric-side qualifier on a collision, an error on a second collision.

A band index is local to a band family, so a group named by the index alone and holding the faces of two
families would let a consumer who selects one family's band select the other's.  A conductor face is named by its
entity and side, which already identify the support; an interface support is a pair of solution regions, and the
same pair meets at several levels.  Only when two families would share a banded interface group do the faces of
that group gain the side of the dielectric they lie on (``__interface__top`` / ``__interface__bottom``); a group
that still holds two families (two concrete requests on one face) is an error.  A name never carries a height.
"""

from __future__ import annotations

import jsonschema
import pytest
from sgb_scenes import GRID, conductor, footprint, rect_um
from test_sgb_sidecar import sidecar_of

from scgsim.contracts.sgb_sidecar import sidecar_schema, views
from scgsim.geometry.sgb.cutfamilies.requests import (
    ConductorSurface,
    InsetRequest,
    InterfaceSurfaces,
    SolutionInterface,
)
from scgsim.geometry.sgb.errors import InputError
from scgsim.geometry.sgb.model import build_model_from_catalogue
from scgsim.geometry.sgb.records.grid import exact_z
from scgsim.geometry.sgb.records.prisms import (
    PortIdentity,
    PrismCatalogue,
    PrismRecord,
)

HALF = 20.0


def two_level_catalogue() -> PrismCatalogue:
    """A 40 x 40 um substrate z = -100..0 inside a larger vacuum, a 10 x 10 um sheet on each face of it."""
    box = footprint(rect_um(-HALF, -HALF, HALF, HALF))
    pad = rect_um(-5, -5, 5, 5)
    prisms = (
        PrismRecord(
            "substrate", box, exact_z(-100), exact_z(0), "substrate", "dielectric"
        ),
        PrismRecord("vacuum", box, exact_z(-200), exact_z(100), "vacuum", "vacuum"),
        conductor("top_pad", pad, 0, 0, "N_top"),
        conductor("bottom_pad", pad, -100, -100, "N_bottom"),
    )
    return PrismCatalogue(GRID, prisms, name="two_levels")


def two_level_requests() -> tuple[InsetRequest, ...]:
    """One concrete request per level, with different margins (each level is its own face and family)."""
    return (
        InsetRequest(
            "top_sa", SolutionInterface("substrate", "vacuum", 0.0), (0.5, 1.0)
        ),
        InsetRequest(
            "bottom_sa", SolutionInterface("substrate", "vacuum", -100.0), (0.5,)
        ),
    )


@pytest.fixture(scope="module")
def two_level_model():
    """The reviewer's repro, built once."""
    return build_model_from_catalogue(
        two_level_catalogue(), "A", insets=two_level_requests()
    )


def level_of(model, group: str) -> set[float]:
    """The Z of every face of a group."""
    return {
        float(model.topology.faces[fid].z_range[0])
        for fid in model.routed.groups[group].ids
    }


TOP = "SA::substrate|vacuum__interface__top"
BOTTOM = "SA::substrate|vacuum__interface__bottom"


PLAIN = "SA::substrate|vacuum__interface"


def test_the_reviewers_two_level_model_gets_one_band_group_per_dielectric_side(
    two_level_model,
):
    """The same pair at z = 0 (substrate below) and z = -100 (substrate above).

    The qualifier applies to a whole family: band 2 exists on the top level only and collides with nothing, but
    its family collides in bands 0 and 1, so all three bands of the top family carry ``__top``.
    """
    groups = {n for n in two_level_model.routed.groups if "__interface" in n}
    assert groups == {
        *(f"{TOP}::band={k}" for k in range(3)),
        *(f"{BOTTOM}::band={k}" for k in range(2)),
    }
    assert not [n for n in two_level_model.routed.groups if n.startswith(f"{PLAIN}::")]
    for name in groups:
        expected = -100.0 if name.startswith(BOTTOM) else 0.0
        assert level_of(two_level_model, name) == {expected}, name
        assert "z=" not in name


def test_the_qualifier_is_the_side_of_the_dielectric_not_the_height(two_level_model):
    """At z = 0 the substrate lies below the face (the face is its top); at z = -100 above it (its bottom)."""
    for name, group in two_level_model.routed.groups.items():
        if group.dim != 2 or "__interface__" not in name:
            continue
        for fid in group.ids:
            face = two_level_model.topology.faces[fid]
            substrate_below = face.neg_domain.name == "substrate"
            assert name.startswith(TOP if substrate_below else BOTTOM), name


def test_an_unbanded_interface_keeps_one_group_across_levels():
    """No inset request: both levels are one ``SA::...__interface`` group, with no qualifier and no height."""
    model = build_model_from_catalogue(two_level_catalogue(), "A")
    assert level_of(model, "SA::substrate|vacuum__interface") == {0.0, -100.0}
    assert not [n for n in model.routed.groups if "z=" in n or "__interface__" in n]


def test_a_single_level_band_model_has_no_qualifier():
    """One family on the pair: its bands are ``SA::...__interface::band=k``; the other level stays unbanded."""
    model = build_model_from_catalogue(
        two_level_catalogue(),
        "A",
        insets=(
            InsetRequest(
                "top_sa", SolutionInterface("substrate", "vacuum", 0.0), (0.5, 1.0)
            ),
        ),
    )
    groups = {n for n in model.routed.groups if "__interface" in n}
    assert groups == {
        "SA::substrate|vacuum__interface",
        *(f"SA::substrate|vacuum__interface::band={k}" for k in range(3)),
    }
    assert level_of(model, "SA::substrate|vacuum__interface") == {-100.0}
    assert {
        level_of(model, f"SA::substrate|vacuum__interface::band={k}").pop()
        for k in range(3)
    } == {0.0}


def test_each_level_has_its_own_families_margins_and_band_groups(two_level_model):
    """The sidecar gives a family its own margins and bands, each family on its own qualified groups."""
    side = sidecar_of(two_level_model)
    assert not list(jsonschema.Draft202012Validator(sidecar_schema()).iter_errors(side))
    families = side["band_family"]
    top, bottom = families["top_sa"], families["bottom_sa"]
    members = views.group_members(side)
    assert top["margins_um"] == [0.5, 1.0] and bottom["margins_um"] == [0.5]
    assert top["bands"] == {str(k): [f"{TOP}::band={k}"] for k in range(3)}
    assert bottom["bands"] == {str(k): [f"{BOTTOM}::band={k}"] for k in range(2)}
    for name, group in side["group"].items():
        if "__interface" not in name or "::band=" not in name:
            continue
        family = "bottom_sa" if name.startswith(BOTTOM) else "top_sa"
        assert group["face_kind"] == "interface"
        assert group["band"] == {"family": family, "index": group["band"]["index"]}
        assert {side["face"][fid]["z_um"][0] for fid in members[name]} == {
            0.0 if family == "top_sa" else -100.0
        }
        assert {side["face"][fid]["face_kind"] for fid in members[name]} == {
            "interface"
        }
        assert side["surface_epr_row"][name]["group"] == name
        assert side["surface_epr_row"][name]["band"] == group["band"]["index"]
    assert {n for n in side["surface_epr_row"] if n.startswith("SA::")} == {
        n for n in side["group"] if n.startswith("SA::") and "::band=" in n
    }
    assert sorted(
        n for n, g in side["group"].items() if g["interface_type"] == "SA"
    ) == sorted(n for n in side["group"] if n.startswith("SA::"))


def test_two_concrete_requests_on_one_face_are_refused():
    """A face has one band family: two concrete requests naming it are refused at planning, naming both."""
    pad = ConductorSurface("top_pad", "sheet")
    with pytest.raises(InputError, match="name the same face") as raised:
        build_model_from_catalogue(
            two_level_catalogue(),
            "A",
            insets=(InsetRequest("one", pad, (0.5,)), InsetRequest("two", pad, (1.0,))),
        )
    message = str(raised.value)
    assert "'one'" in message and "'two'" in message


def test_two_concrete_interface_requests_on_one_face_are_refused():
    """The same holds for a solution interface; a request on another level is not part of the error."""
    top = SolutionInterface("substrate", "vacuum", 0.0)
    bottom = SolutionInterface("substrate", "vacuum", -100.0)
    with pytest.raises(InputError, match="name the same face") as raised:
        build_model_from_catalogue(
            two_level_catalogue(),
            "A",
            insets=(
                InsetRequest("one", top, (0.5,)),
                InsetRequest("two", top, (1.0,)),
                InsetRequest("below", bottom, (0.5,)),
            ),
        )
    message = str(raised.value)
    assert "'one'" in message and "'two'" in message and "'below'" not in message


# --------------------------------------------------------------------------- expansion.banded


def covered_gap_catalogue() -> PrismCatalogue:
    """Two sheets with a 4 um gap between them and a lumped port covering the whole gap (0.5 um landing on each)."""
    box = footprint(rect_um(-HALF, -HALF, HALF, HALF))
    identity = PortIdentity(
        "PORT_SHEET__P1__0000",
        "P1",
        1,
        (1.0, 0.0, 0.0),
        ("left", "right"),
        "metal",
        "port",
    )
    prisms = (
        PrismRecord(
            "substrate", box, exact_z(-100), exact_z(0), "substrate", "dielectric"
        ),
        PrismRecord("vacuum", box, exact_z(0), exact_z(100), "vacuum", "vacuum"),
        conductor("left", rect_um(-HALF, -HALF, -2, HALF), 0, 0, "N_left"),
        conductor("right", rect_um(2, -HALF, HALF, HALF), 0, 0, "N_right"),
        PrismRecord(
            "port_p1",
            footprint(rect_um(-2.5, -HALF, 2.5, HALF)),
            exact_z(0),
            exact_z(0),
            "lumped_port",
            "port",
            port=identity,
        ),
    )
    return PrismCatalogue(GRID, prisms, name="covered_gap")


def test_a_family_the_pre_pass_banded_may_have_no_band_once_a_port_covers_its_face():
    """``expansion.banded`` lists the families the physical pre-pass created; ports placed later may cover one."""
    request = InsetRequest("sa", InterfaceSurfaces("SA"), (0.5, 1.0))
    model = build_model_from_catalogue(covered_gap_catalogue(), "A", insets=(request,))
    side = sidecar_of(model)
    assert not list(jsonschema.Draft202012Validator(sidecar_schema()).iter_errors(side))
    (record,) = side["inset_request"].values()
    (family_id,) = record["families"]
    family = side["band_family"][family_id]
    assert family_id == "sa::substrate|vacuum::z=0.0"
    assert family["margins_um"] == [0.5, 1.0]
    assert family["bands"] == {}
    assert record["skipped"] == []
    assert not [
        row for row in side["surface_epr_row"].values() if row["interface_type"] == "SA"
    ]


def test_two_concrete_requests_on_a_face_a_port_covers_are_refused():
    """The refusal does not depend on emitted bands: a port covering the whole face leaves none, and it still fails."""
    gap = SolutionInterface("substrate", "vacuum", 0.0)
    with pytest.raises(InputError, match="name the same face") as raised:
        build_model_from_catalogue(
            covered_gap_catalogue(),
            "A",
            insets=(InsetRequest("one", gap, (0.5,)), InsetRequest("two", gap, (1.0,))),
        )
    message = str(raised.value)
    assert "'one'" in message and "'two'" in message
