"""Surface-EPR rows over the attributes of SGB's rows and bands: baselines, margins as band unions, sides."""

from __future__ import annotations

import pytest
from palace_sgb_support import (
    effective_input,
    lower_surface_epr,
    sidecar_of,
    surface_epr_specs,
)
from test_sgb_band_labels import PORT_CASES, all_interface_insets

from scgsim.geometry.surface_epr.insets import inset_requests, needs_unbanded_build
from scgsim.geometry.surface_epr.specs import normalize_surface_epr_specs
from scgsim.palace.mesh.physical_groups import groups_from_sidecar


def banded_sidecar(route, directory, specs):
    """The sidecar of a build that carries the bands ``specs`` ask for, as ``mesh()`` builds it."""
    effective = effective_input(route, directory)
    unbanded = (
        sidecar_of(effective, route, directory) if needs_unbanded_build(specs) else None
    )
    insets = inset_requests(specs, unbanded)
    return sidecar_of(effective, route, directory / "b", insets)


def banded_groups(route, directory, specs):
    """The group dict of a build that carries the bands ``specs`` ask for."""
    return groups_from_sidecar(banded_sidecar(route, directory, specs))


@pytest.mark.parametrize("l0", [1e-6, 1e-3])
def test_margins_are_unions_of_bands_and_scale_only_the_thickness(tmp_path, l0):
    """Margin m_k selects the attributes of bands k..K; no row carries a Mask, whatever Model.L0."""
    specs = surface_epr_specs(MA=[0.5, 1.0])
    groups = banded_groups("B", tmp_path, specs)
    rows, entries = lower_surface_epr(groups, specs, model_l0_m=l0)
    assert all("Mask" not in row for row in rows)
    assert all(row["Thickness"] == pytest.approx(2e-9 / l0) for row in rows)
    by_name = {entry["entry_name"]: entry for entry in entries}
    top = groups["boundary_surfaces"]
    bands = [top[f"MA::pad__top::band={k}"]["attributes"][0] for k in (0, 1, 2)]
    assert by_name["MA::pad__top"]["attributes"] == sorted(bands)
    assert by_name["MA::pad__top__inset_0"]["attributes"] == bands[1:]
    assert by_name["MA::pad__top__inset_1"]["attributes"] == bands[2:]
    assert by_name["MA::pad__top__inset_1"]["inset"] == {
        "margin_um": 1.0,
        "margin_index": 1,
        "first_band": 2,
        "bands": [2],
    }
    assert (
        by_name["MA::pad__top__inset_1"]["baseline_index"]
        == by_name["MA::pad__top"]["index"]
    )


def test_a_sidewall_has_a_baseline_row_but_no_margin_rows(tmp_path):
    """Sidewalls are unbanded: they lie at the gap edge, so no margin selects them."""
    specs = surface_epr_specs(MA=[0.5])
    groups = banded_groups("B", tmp_path, specs)
    _, entries = lower_surface_epr(groups, specs)
    names = {entry["entry_name"] for entry in entries}
    assert "MA::pad__sidewall" in names
    assert not any(name.startswith("MA::pad__sidewall__inset") for name in names)


def test_each_margin_row_names_its_bands_and_lineage_in_the_index(tmp_path):
    """Index metadata: the row, group, type, band, sides, entities, Net, attributes and lineage."""
    specs = surface_epr_specs(MA=[0.5])
    groups = banded_groups("B", tmp_path, specs)
    _, entries = lower_surface_epr(groups, specs)
    entry = next(e for e in entries if e["entry_name"] == "MA::pad__top__inset_0")
    assert [r["row"] for r in entry["metadata"]["rows"]] == ["MA::pad__top::band=1"]
    row = entry["metadata"]["rows"][0]
    assert set(row) == {
        "row",
        "group",
        "interface_type",
        "face_kind",
        "band",
        "side",
        "face_sides",
        "shared_face_group",
        "entities",
        "net",
        "attributes",
        "lineage",
    }
    assert (row["interface_type"], row["entities"], row["net"]) == ("MA", ["pad"], "P")
    assert set(row["lineage"]) == {"membership"}


def test_route_a_sheet_rows_share_attributes_and_keep_their_sides(tmp_path):
    """MA and MS rows of one sheet band sit on the same attributes; the index keeps the side of each."""
    specs = surface_epr_specs(MA=[0.5, 1.0], MS=[0.5, 1.0])
    groups = banded_groups("A", tmp_path, specs)
    rows, entries = lower_surface_epr(groups, specs)
    by_name = {entry["entry_name"]: entry for entry in entries}
    ma = by_name["MA::chip/left__sheet"]
    ms = by_name["MS::chip/left__sheet"]
    assert ma["attributes"] == ms["attributes"]
    assert {r["side"] for r in ma["metadata"]["rows"]} == {"above"}
    assert {r["side"] for r in ms["metadata"]["rows"]} == {"below"}
    # Both specs ask the sheet's one set of margins (0.5, 1.0): margin 0.5 selects bands 1 and 2, 1.0 band 2.
    assert by_name["MS::chip/left__sheet__inset_0"]["inset"]["first_band"] == 1
    assert by_name["MA::chip/left__sheet__inset_1"]["inset"]["first_band"] == 2
    assert {row["Type"] for row in rows} == {"MA", "MS"}


def test_a_required_type_without_rows_is_an_error(tmp_path):
    """A required spec with no matching row fails; an optional one is skipped."""
    groups = groups_from_sidecar(
        sidecar_of(effective_input("B", tmp_path), "B", tmp_path)
    )
    specs = surface_epr_specs(SA=None)
    specs["SA"]["face_kinds"] = ["top"]
    with pytest.raises(ValueError, match="Required Surface EPR SA"):
        lower_surface_epr(groups, specs)
    specs["SA"]["required"] = False
    assert lower_surface_epr(groups, specs) == ([], [])


def test_face_kinds_select_the_rows_of_those_kinds(tmp_path):
    """``face_kinds`` keeps the groups of the named face kind only."""
    groups = groups_from_sidecar(
        sidecar_of(effective_input("B", tmp_path), "B", tmp_path)
    )
    specs = surface_epr_specs(MA=None)
    specs["MA"]["face_kinds"] = ["sidewall"]
    _, entries = lower_surface_epr(groups, specs)
    assert {e["entry_name"] for e in entries} == {
        "MA::ground_0000__sidewall",
        "MA::pad__sidewall",
    }


def test_a_margin_the_support_was_not_banded_with_is_an_error(tmp_path):
    """Config margins must be those the geometry was built with."""
    groups = banded_groups("B", tmp_path, surface_epr_specs(MA=[0.5]))
    with pytest.raises(ValueError, match="not among the margins"):
        lower_surface_epr(groups, surface_epr_specs(MA=[2.0]))


def test_margins_must_be_positive_and_strictly_increasing():
    """Zero is the whole surface (the baseline row); the margins of one spec grow."""
    film = {"film_thickness_m": 2e-9, "permittivity": 4}
    for margins, message in (
        ([0.0, 0.5], "finite positive"),
        ([-1.0], "finite positive"),
        ([1.0, 0.5], "strictly increasing"),
        ([0.5, 0.5], "strictly increasing"),
    ):
        with pytest.raises(ValueError, match=message):
            normalize_surface_epr_specs(
                {"MA": {**film, "inset_margins_um": margins}}, materials={}
            )
    ok = normalize_surface_epr_specs(
        {"MA": {**film, "inset_margins_um": [0.5, 1]}}, materials={}
    )
    assert ok["MA"]["inset_margins_um"] == [0.5, 1.0]


def test_the_type_level_requests_band_the_supports_of_the_all_interface_requests(
    tmp_path,
):
    """SGB expands the type-level requests into the conductor supports the all-interface requests name."""
    effective = effective_input("B", tmp_path)
    _, sa_pair, *_ = PORT_CASES["B"]
    reference = {
        (r.support.entity_id, r.support.side)
        for r in all_interface_insets(effective, "B", sa_pair)
        if hasattr(r.support, "entity_id")
    }
    sidecar = banded_sidecar("B", tmp_path, surface_epr_specs(MA=[0.5], MS=[0.5]))
    families = sidecar["band_family"]
    expanded = {
        (families[f]["support"]["entity"], families[f]["support"]["side"])
        for request in sidecar["inset_request"].values()
        for f in request["families"]
    }
    assert expanded == reference


def band_attributes(groups, bands):
    """The attributes of the named band groups."""
    return sorted(
        a for name in bands for a in groups["boundary_surfaces"][name]["attributes"]
    )


def test_margin_rows_take_their_bands_from_the_expanded_families(tmp_path):
    """A type-level request lists the families it became; margin ``m_k`` unions bands k..K of each."""
    specs = surface_epr_specs(MA=[0.5, 1.0])
    sidecar = banded_sidecar("B", tmp_path, specs)
    groups = groups_from_sidecar(sidecar)
    _, entries = lower_surface_epr(groups, specs)
    by_name = {entry["entry_name"]: entry for entry in entries}
    (request,) = sidecar["inset_request"].values()
    for family in (sidecar["band_family"][f] for f in request["families"]):
        (entity,) = [family["support"]["entity"]]
        for margin_index, first in enumerate((1, 2)):
            wanted = [
                name
                for band, names in family["bands"].items()
                if int(band) >= first
                for name in names
            ]
            entry = by_name[f"MA::{entity}__top__inset_{margin_index}"]
            assert entry["attributes"] == band_attributes(groups, wanted)
            assert entry["physical_names"] == sorted(wanted)


def test_margin_rows_take_their_bands_from_the_bands_of_concrete_requests(tmp_path):
    """A concrete request is its own family: its ``bands`` give the groups of each margin row."""
    effective = effective_input("B", tmp_path)
    _, sa_pair, *_ = PORT_CASES["B"]
    sidecar = sidecar_of(
        effective, "B", tmp_path, all_interface_insets(effective, "B", sa_pair)
    )
    assert all(r["scope"] == "face" for r in sidecar["inset_request"].values())
    groups = groups_from_sidecar(sidecar)
    specs = surface_epr_specs(MA=[0.5, 1.0], SA=[0.5])
    _, entries = lower_surface_epr(groups, specs)
    by_name = {entry["entry_name"]: entry for entry in entries}
    pad = sidecar["band_family"]["band_pad_top"]
    assert by_name["MA::pad__top__inset_1"]["physical_names"] == sorted(
        pad["bands"]["2"]
    )
    sa = sidecar["band_family"]["band_sa"]
    assert by_name["SA::substrate|VACUUM_REGION__interface__inset_0"][
        "physical_names"
    ] == sorted(sa["bands"]["1"] + sa["bands"]["2"])


def test_a_family_with_no_bands_contributes_nothing(tmp_path):
    """A port can cover a whole face: its family lists ``bands == {}`` and carries no row."""
    specs = surface_epr_specs(MA=[0.5, 1.0])
    sidecar = banded_sidecar("B", tmp_path, specs)
    reference = lower_surface_epr(groups_from_sidecar(sidecar), specs)
    (request,) = sidecar["inset_request"].values()
    first = sidecar["band_family"][request["families"][0]]
    sidecar["band_family"]["epr_MA::covered::top"] = {**first, "bands": {}}
    request["families"].append("epr_MA::covered::top")
    assert lower_surface_epr(groups_from_sidecar(sidecar), specs) == reference


def test_faces_sgb_skipped_have_a_baseline_row_and_no_margin_rows(tmp_path):
    """The substrate underside has no gap edge: SGB skips it, so only the baseline row covers it."""
    specs = surface_epr_specs(SA=[0.5])
    sidecar = banded_sidecar("B", tmp_path, specs)
    (request,) = sidecar["inset_request"].values()
    assert [s["support"]["z_um"] for s in request["skipped"]] == [-10.0]
    _, entries = lower_surface_epr(groups_from_sidecar(sidecar), specs)
    names = [entry["entry_name"] for entry in entries]
    # Two interfaces of one pair share a name; the unbanded one keeps its baseline row and says so.
    assert names.count("SA::substrate|VACUUM_REGION__interface") == 1
    assert names.count("SA::substrate|VACUUM_REGION__interface__unbanded") == 1
    assert len(names) == len(set(names))
    assert [n for n in names if "__inset_" in n] == [
        "SA::substrate|VACUUM_REGION__interface__inset_0"
    ]


def test_the_custom_path_gives_a_conductor_its_own_margin_rows(tmp_path):
    """The named conductor's rows use its margins; the type-level request leaves it out for the rest."""
    specs = surface_epr_specs(MA=[0.5, 1.0])
    specs["MA"]["inset_margins_by_entity_um"] = {"pad": [0.25, 0.5, 0.75]}
    sidecar = banded_sidecar("B", tmp_path, specs)
    assert set(sidecar["inset_request"]) == {"epr::pad::top", "epr_MA"}
    _, entries = lower_surface_epr(groups_from_sidecar(sidecar), specs)
    insets = {e["entry_name"]: e["inset"] for e in entries if "inset" in e}
    assert [
        (
            insets[f"MA::pad__top__inset_{k}"]["margin_um"],
            insets[f"MA::pad__top__inset_{k}"]["first_band"],
        )
        for k in range(3)
    ] == [(0.25, 1), (0.5, 2), (0.75, 3)]
    assert [
        (
            insets[f"MA::ground_0000__top__inset_{k}"]["margin_um"],
            insets[f"MA::ground_0000__top__inset_{k}"]["first_band"],
        )
        for k in range(2)
    ] == [(0.5, 1), (1.0, 2)]


def test_a_customised_sheet_serves_both_specs_from_its_one_set_of_margins(tmp_path):
    """Route A: MA and MS customise ``chip/left`` alike; each spec selects its rows from the one family."""
    specs = surface_epr_specs(MA=[1.0], MS=[1.0])
    specs["MA"]["inset_margins_by_entity_um"] = {"chip/left": [0.5]}
    specs["MS"]["inset_margins_by_entity_um"] = {"chip/left": [0.5]}
    sidecar = banded_sidecar("A", tmp_path, specs)
    groups = groups_from_sidecar(sidecar)
    _, entries = lower_surface_epr(groups, specs)
    insets = {e["entry_name"]: e["inset"] for e in entries if "inset" in e}
    # chip/left: margins (0.5) -> both specs select bands 1..; chip/right: margins (1.0) -> bands 1..
    for interface_type in ("MA", "MS"):
        assert insets[f"{interface_type}::chip/left__sheet__inset_0"]["first_band"] == 1
        assert (
            insets[f"{interface_type}::chip/right__sheet__inset_0"]["first_band"] == 1
        )


def test_one_sheet_takes_one_set_of_margins_however_they_are_written(tmp_path):
    """MA writes 0.3 and MS 0.1 + 0.2: the same multiple of the grid, so one request, not two values."""
    specs = surface_epr_specs(MA=None, MS=None)
    specs["MA"]["inset_margins_by_entity_um"] = {"chip/left": [0.3]}
    specs["MS"]["inset_margins_by_entity_um"] = {"chip/left": [0.1 + 0.2]}
    sidecar = sidecar_of(effective_input("A", tmp_path), "A", tmp_path)
    (request,) = inset_requests(specs, sidecar)
    assert request.request_id == "epr::chip/left::sheet"
    assert request.margins_um == (0.3,)


@pytest.mark.parametrize(
    ("ma", "ms"),
    [
        (
            {"inset_margins_by_entity_um": {"chip/left": [0.3]}},
            {"inset_margins_by_entity_um": {"chip/left": [0.2]}},
        ),
        (
            {"inset_margins_by_entity_um": {"chip/left": [0.3]}},
            {"inset_margins_um": [0.2]},
        ),
        (
            {"inset_margins_um": [0.3]},
            {"inset_margins_by_entity_um": {"chip/left": [0.2]}},
        ),
    ],
    ids=["both customise", "MA customises, MS by type", "MS customises, MA by type"],
)
def test_differing_margins_on_one_sheet_are_refused_naming_both(tmp_path, ma, ms):
    """A sheet is one band family: MA and MS may not give it different margins, whichever key they use."""
    specs = surface_epr_specs(MA=None, MS=None)
    specs["MA"].update(ma)
    specs["MS"].update(ms)
    sidecar = sidecar_of(effective_input("A", tmp_path), "A", tmp_path)
    with pytest.raises(ValueError, match="takes one set of margins") as raised:
        inset_requests(specs, sidecar)
    message = str(raised.value)
    assert "'chip/left'" in message and "sheet" in message
    assert "0.3" in message and "0.2" in message
    assert "MA" in message and "MS" in message


def test_a_type_level_spec_alone_on_a_customised_conductor_gets_no_request_of_its_own(
    tmp_path,
):
    """Route B: pad top is MA's and pad bottom MS's face, so MS by type does not meet MA's customised margins."""
    specs = surface_epr_specs(MA=[1.0], MS=[0.5])
    specs["MA"]["inset_margins_by_entity_um"] = {"pad": [0.25]}
    sidecar = sidecar_of(effective_input("B", tmp_path), "B", tmp_path)
    named = [
        r for r in inset_requests(specs, sidecar) if r.request_id.startswith("epr::")
    ]
    assert [(r.request_id, r.margins_um) for r in named] == [("epr::pad::top", (0.25,))]


def test_a_margin_is_found_among_the_bands_by_grid_identity(tmp_path):
    """The family records 0.3; a spec that writes 0.1 + 0.2 selects the same band."""
    groups = banded_groups("B", tmp_path, surface_epr_specs(MA=[0.3]))
    specs = surface_epr_specs(MA=[0.1 + 0.2])
    _, entries = lower_surface_epr(groups, specs)
    (entry,) = (e for e in entries if e["entry_name"] == "MA::pad__top__inset_0")
    assert entry["inset"]["first_band"] == 1


def test_a_customised_margin_the_conductor_was_not_banded_with_is_an_error(tmp_path):
    """Config margins must be those the geometry was built with, per conductor."""
    built = surface_epr_specs(MA=[1.0])
    built["MA"]["inset_margins_by_entity_um"] = {"pad": [0.25]}
    groups = groups_from_sidecar(banded_sidecar("B", tmp_path, built))
    changed = surface_epr_specs(MA=[1.0])
    changed["MA"]["inset_margins_by_entity_um"] = {"pad": [0.5]}
    with pytest.raises(ValueError, match="not among the margins"):
        lower_surface_epr(groups, changed)
