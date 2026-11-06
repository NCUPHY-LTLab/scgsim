"""Band groups name their support: a banded interface carries its level, and a group never mixes two families.

A band index is local to a band family, so a group named by the index alone and holding the faces of two
families would let a consumer who selects one family's band select the other's.  A conductor face is named by its
entity and side, which already identify the support; an interface support is a pair of solution regions at one Z,
so a banded interface group carries ``z=<z>``.  What is left (two concrete requests on one face) is an error.
"""

from __future__ import annotations

import jsonschema
import pytest
from sgb_scenes import GRID, conductor, footprint, rect_um
from test_sgb_sidecar import sidecar_of

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
from scgsim.geometry.sgb.sidecar.files import sidecar_schema

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
    """A type-level request (both levels) and a concrete one on the top level, with different margins."""
    return (
        InsetRequest("sa", InterfaceSurfaces("SA"), (0.5,)),
        InsetRequest("top_sa", SolutionInterface("substrate", "vacuum", 0.0), (1.0,)),
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


def test_a_banded_interface_group_carries_the_level_it_lies_on(two_level_model):
    """The same substrate-vacuum pair at z = 0 and z = -100 gives distinct band groups, each on one level."""
    groups = {n for n in two_level_model.routed.groups if "interface::" in n}
    assert groups == {
        f"SA::substrate|vacuum::interface::z={z}::band={k}"
        for z, bands in ((0.0, (0, 1, 2)), (-100.0, (0, 1)))
        for k in bands
    }
    for name in groups:
        expected = float(name.split("::z=")[1].split("::")[0])
        assert level_of(two_level_model, name) == {expected}, name


def test_the_group_name_agrees_with_the_family_id_suffix(two_level_model):
    """``z=<z>`` of a group is the ``z=<z>`` of the expanded request id of the same face."""
    (expanded,) = (
        r.request_id
        for r in two_level_model.inset_plan.requests
        if r.request_id.startswith("sa::")
    )
    suffix = expanded.split("::")[-1]
    assert suffix == "z=-100.0"
    assert f"SA::substrate|vacuum::interface::{suffix}::band=0" in (
        two_level_model.routed.groups
    )


def test_an_unbanded_interface_keeps_one_group_across_levels():
    """No inset request: both levels are one ``SA::...::interface`` group, as before."""
    model = build_model_from_catalogue(two_level_catalogue(), "A")
    assert level_of(model, "SA::substrate|vacuum::interface") == {0.0, -100.0}
    assert not [n for n in model.routed.groups if "::z=" in n]


def test_each_level_has_its_own_families_margins_and_band_groups(two_level_model):
    """The sidecar gives a family its own margins and bands, and no band group holds faces of two levels."""
    side = sidecar_of(two_level_model)
    assert not list(jsonschema.Draft202012Validator(sidecar_schema()).iter_errors(side))
    insets = {r["request_id"]: r for r in side["insets"]}
    top = insets["top_sa"]
    (bottom,) = (
        f
        for f in insets["sa"]["expansion"]["banded"]
        if f["request_id"] == "sa::substrate|vacuum::z=-100.0"
    )
    assert top["margins_um"] == [0.5, 1.0] and bottom["margins_um"] == [0.5]
    prefix = "SA::substrate|vacuum::interface::"
    assert top["bands"] == {str(k): [f"{prefix}z=0.0::band={k}"] for k in range(3)}
    assert bottom["bands"] == {
        str(k): [f"{prefix}z=-100.0::band={k}"] for k in range(2)
    }
    for name, group in side["physical_groups"].items():
        if "interface::z=" not in name:
            continue
        family = top if "z=0.0" in name else bottom
        assert group["band"] == {
            "family": family["request_id"],
            "index": group["band"]["index"],
        }
        assert {side["faces"][fid]["z_um"][0] for fid in group["ids"]} == {
            float(name.split("::z=")[1].split("::")[0])
        }
        assert side["surface_epr_rows"][name]["group"] == name


def test_two_concrete_requests_on_one_face_are_refused():
    """Two families on one face would share band group names: fail closed, naming the group and both families."""
    pad = ConductorSurface("top_pad", "sheet")
    with pytest.raises(InputError) as raised:
        build_model_from_catalogue(
            two_level_catalogue(),
            "A",
            insets=(InsetRequest("one", pad, (0.5,)), InsetRequest("two", pad, (1.0,))),
        )
    message = str(raised.value)
    assert "physical group 'top_pad::band=" in message
    assert "'one'" in message and "'two'" in message


def test_two_concrete_interface_requests_on_one_face_are_refused():
    """The same holds for an interface: the level in the name does not separate two families on one face."""
    top = SolutionInterface("substrate", "vacuum", 0.0)
    with pytest.raises(InputError, match="two band families"):
        build_model_from_catalogue(
            two_level_catalogue(),
            "A",
            insets=(InsetRequest("one", top, (0.5,)), InsetRequest("two", top, (1.0,))),
        )


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
    (record,) = side["insets"]
    (family,) = record["expansion"]["banded"]
    assert family["request_id"] == "sa::substrate|vacuum::z=0.0"
    assert family["margins_um"] == [0.5, 1.0]
    assert family["bands"] == {}
    assert record["expansion"]["skipped"] == []
    assert not [
        row
        for row in side["surface_epr_rows"].values()
        if row["interface_type"] == "SA"
    ]
