"""Surface-EPR rows over the attributes of SGB's rows and bands: baselines, margins as band unions, sides."""

from __future__ import annotations

import pytest
from palace_sgb_support import effective_input, sidecar_of, surface_epr_specs
from test_sgb_band_labels import PORT_CASES, all_interface_insets

from scgsim.palace.mesh.insets import inset_requests
from scgsim.palace.mesh.physical_groups import groups_from_sidecar
from scgsim.palace.preparation.surface_epr import (
    build_surface_epr_postprocessing,
    normalize_surface_epr_specs,
)


def banded_groups(route, directory, specs):
    """The group dict of a build that carries the bands ``specs`` ask for."""
    effective = effective_input(route, directory)
    unbanded = sidecar_of(effective, route, directory)
    insets = inset_requests(specs, unbanded)
    return groups_from_sidecar(sidecar_of(effective, route, directory / "b", insets))


@pytest.mark.parametrize("l0", [1e-6, 1e-3])
def test_margins_are_unions_of_bands_and_scale_only_the_thickness(tmp_path, l0):
    """Margin m_k selects the attributes of bands k..K; no row carries a Mask, whatever Model.L0."""
    specs = surface_epr_specs(MA=[0.5, 1.0])
    groups = banded_groups("B", tmp_path, specs)
    rows, entries = build_surface_epr_postprocessing(groups, specs, model_l0_m=l0)
    assert all("Mask" not in row for row in rows)
    assert all(row["Thickness"] == pytest.approx(2e-9 / l0) for row in rows)
    by_name = {entry["entry_name"]: entry for entry in entries}
    top = groups["boundary_surfaces"]
    bands = [top[f"MA::pad::top::band={k}"]["attributes"][0] for k in (0, 1, 2)]
    assert by_name["MA::pad::top"]["attributes"] == sorted(bands)
    assert by_name["MA::pad::top__inset_0"]["attributes"] == bands[1:]
    assert by_name["MA::pad::top__inset_1"]["attributes"] == bands[2:]
    assert by_name["MA::pad::top__inset_1"]["inset"] == {
        "margin_um": 1.0,
        "margin_index": 1,
        "first_band": 2,
        "bands": [2],
    }
    assert (
        by_name["MA::pad::top__inset_1"]["baseline_index"]
        == by_name["MA::pad::top"]["index"]
    )


def test_a_sidewall_has_a_baseline_row_but_no_margin_rows(tmp_path):
    """Sidewalls are unbanded: they lie at the gap edge, so no margin selects them."""
    specs = surface_epr_specs(MA=[0.5])
    groups = banded_groups("B", tmp_path, specs)
    _, entries = build_surface_epr_postprocessing(groups, specs)
    names = {entry["entry_name"] for entry in entries}
    assert "MA::pad::sidewall" in names
    assert not any(name.startswith("MA::pad::sidewall__") for name in names)


def test_each_margin_row_names_its_bands_and_lineage_in_the_index(tmp_path):
    """Index metadata: the row, group, type, band, sides, entities, Net, attributes and lineage."""
    specs = surface_epr_specs(MA=[0.5])
    groups = banded_groups("B", tmp_path, specs)
    _, entries = build_surface_epr_postprocessing(groups, specs)
    entry = next(e for e in entries if e["entry_name"] == "MA::pad::top__inset_0")
    assert [r["row"] for r in entry["metadata"]["rows"]] == ["MA::pad::top::band=1"]
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
    specs = surface_epr_specs(MA=[0.5, 1.0], MS=[0.75])
    groups = banded_groups("A", tmp_path, specs)
    rows, entries = build_surface_epr_postprocessing(groups, specs)
    by_name = {entry["entry_name"]: entry for entry in entries}
    ma = by_name["MA::chip/left::sheet"]
    ms = by_name["MS::chip/left::sheet"]
    assert ma["attributes"] == ms["attributes"]
    assert {r["side"] for r in ma["metadata"]["rows"]} == {"above"}
    assert {r["side"] for r in ms["metadata"]["rows"]} == {"below"}
    # The MS margin 0.75 is the second margin of the support (0.5, 0.75, 1.0): bands 2 and 3 are selected.
    assert by_name["MS::chip/left::sheet__inset_0"]["inset"]["first_band"] == 2
    assert by_name["MA::chip/left::sheet__inset_1"]["inset"]["first_band"] == 3
    assert {row["Type"] for row in rows} == {"MA", "MS"}


def test_a_required_type_without_rows_is_an_error(tmp_path):
    """A required spec with no matching row fails; an optional one is skipped."""
    groups = groups_from_sidecar(
        sidecar_of(effective_input("B", tmp_path), "B", tmp_path)
    )
    specs = surface_epr_specs(SA=None)
    specs["SA"]["face_kinds"] = ["top"]
    with pytest.raises(ValueError, match="Required Surface EPR SA"):
        build_surface_epr_postprocessing(groups, specs)
    specs["SA"]["required"] = False
    assert build_surface_epr_postprocessing(groups, specs) == ([], [])


def test_face_kinds_select_the_rows_of_those_kinds(tmp_path):
    """``face_kinds`` keeps the groups of the named face kind only."""
    groups = groups_from_sidecar(
        sidecar_of(effective_input("B", tmp_path), "B", tmp_path)
    )
    specs = surface_epr_specs(MA=None)
    specs["MA"]["face_kinds"] = ["sidewall"]
    _, entries = build_surface_epr_postprocessing(groups, specs)
    assert {e["entry_name"] for e in entries} == {
        "MA::ground_0000::sidewall",
        "MA::pad::sidewall",
    }


def test_a_margin_the_support_was_not_banded_with_is_an_error(tmp_path):
    """Config margins must be those the geometry was built with."""
    groups = banded_groups("B", tmp_path, surface_epr_specs(MA=[0.5]))
    with pytest.raises(ValueError, match="not among the margins"):
        build_surface_epr_postprocessing(groups, surface_epr_specs(MA=[2.0]))


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


def test_the_benchmark_all_interface_bands_agree_with_the_spec_derived_requests(
    tmp_path,
):
    """Spec-derived requests band the same conductor supports as the all-interface requests of the tests."""
    effective = effective_input("B", tmp_path)
    sidecar = sidecar_of(effective, "B", tmp_path)
    _, sa_pair, *_ = PORT_CASES["B"]
    reference = {
        (r.support.entity_id, r.support.side)
        for r in all_interface_insets(effective, "B", sa_pair)
        if hasattr(r.support, "entity_id")
    }
    derived = {
        (r.support.entity_id, r.support.side)
        for r in inset_requests(surface_epr_specs(MA=[0.5], MS=[0.5]), sidecar)
    }
    assert derived == reference
