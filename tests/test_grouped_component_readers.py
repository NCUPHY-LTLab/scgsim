"""Exact grouped conductor identities survive mesh lowering and boundary mapping."""

import copy

import pytest

from scgsim.palace.mesh.build import (
    _conductor_component_ids,
    _structured_surface_info,
)
from scgsim.palace.preparation.config import TerminalBinding, _electrostatic_boundaries


def grouped_record():
    return {
        "conductor_component_id": None,
        "conductor_component_ids": ("bump/1", "bump/2"),
        "conductor_component_ids_by_source": {"face/1": "bump/1", "face/2": "bump/2"},
        "net_id": "signal",
        "equipotential_id": "signal-equipotential",
        "metadata": {"source_record_ids": ("face/1", "face/2")},
        "source_provenance": {"source_record_ids": ("face/1", "face/2")},
        "solver_use": "solver_active",
    }


def surface(record=None, *, attr=7):
    return _structured_surface_info(
        name="bumps",
        phys_group=attr,
        entity_tags=(11, 12),
        record=record or grouped_record(),
        interface_type="MA",
        surface_epr=True,
    )


def boundaries(
    records, *, terminals=(TerminalBinding("Signal", "signal"),), grounds=()
):
    return _electrostatic_boundaries(
        groups={"boundary_surfaces": records},
        terminals=terminals,
        ground_net_ids=grounds,
        unassigned_conductor_policy="error",
        exterior_boundary_policy="none",
    )


def test_grouped_surface_keeps_exact_members_and_deduplicates_terminal_attribute():
    record = grouped_record()
    info = surface(record)
    assert (
        info["conductor_component_ids_by_source"]
        == record["conductor_component_ids_by_source"]
    )
    assert (
        info["conductor_component_ids_by_source"]
        is not record["conductor_component_ids_by_source"]
    )
    result, index, _ = boundaries(
        {"bumps__MA": info, "bumps__MS": {**info, "interface_type": "MS"}}
    )
    assert result == {"Terminal": [{"Index": 1, "Attributes": [7]}]}
    assert index[0]["conductor_component_ids"] == ["bump/1", "bump/2"]
    assert index[0]["physical_names"] == ["bumps__MA", "bumps__MS"]


def test_legacy_singleton_and_additive_singleton_preserve_boundary_mapping():
    legacy = {
        "conductor_component_id": "bump/1",
        "net_id": "signal",
        "solver_use": "solver_active",
    }
    additive = {
        **legacy,
        "conductor_component_ids": ("bump/1",),
        "conductor_component_ids_by_source": {"face/1": "bump/1"},
        "metadata": {"source_record_ids": ("face/1",)},
    }
    assert (
        boundaries({"legacy": surface(legacy)})[0]
        == boundaries({"additive": surface(additive)})[0]
    )


@pytest.mark.parametrize(
    "change",
    [
        {"conductor_component_id": "bump/1"},
        {"conductor_component_ids": ("bump/2", "bump/1")},
        {"conductor_component_ids": ("bump/1", "bump/1")},
        {"conductor_component_ids_by_source": {"face/1": "bump/1"}},
        {
            "conductor_component_ids_by_source": {
                "face/1": "bump/1",
                "foreign": "bump/2",
            }
        },
        {"conductor_component_ids_by_source": {"face/1": "bump/1", "face/2": None}},
    ],
)
def test_membership_rejects_contradictory_or_incomplete_source_bindings(change):
    with pytest.raises(ValueError):
        _conductor_component_ids({**grouped_record(), **change}, name="bumps")


def test_shared_attribute_rejects_cross_net_and_equipotential_assignment():
    first = surface()
    for field, value in (("net_id", "ground"), ("equipotential_id", "other")):
        other = copy.deepcopy(first)
        other[field] = value
        with pytest.raises(ValueError, match="conflicting"):
            boundaries({"first": first, "other": other})


def test_grouped_ground_keeps_one_attribute_and_every_component():
    result, _, resolution = boundaries(
        {"bumps": surface()}, terminals=(), grounds=("signal",)
    )
    assert result == {"Terminal": [], "Ground": {"Attributes": [7]}}
    assert resolution["conductor_component_ids"] == ["bump/1", "bump/2"]


def test_identity_free_additive_record_stays_empty():
    record = {
        "conductor_component_id": None,
        "conductor_component_ids": (),
        "conductor_component_ids_by_source": {},
    }
    assert _conductor_component_ids(record, name="air") == ()


def test_provenance_source_coverage_survives_lowering_and_epr_metadata():
    from scgsim.palace.preparation.surface_epr import _structured_metadata

    record = grouped_record()
    del record["metadata"]
    info = surface(record)
    assert _conductor_component_ids(info, name="bumps") == ("bump/1", "bump/2")
    metadata = _structured_metadata(info)
    assert metadata["conductor_component_ids"] == ("bump/1", "bump/2")
    assert (
        metadata["conductor_component_ids_by_source"]
        == record["conductor_component_ids_by_source"]
    )
    assert metadata["source_provenance"] == record["source_provenance"]


def test_ms_ma_group_lowers_to_both_aliases_and_pec_without_losing_members():
    from scgsim.palace.mesh.build import (
        _add_surface_group_records,
        _build_groups_from_sgb_records,
    )

    record = {
        **grouped_record(),
        "surface_id": "shared-bumps",
        "route": "A",
        "representation": "surface_sheet",
        "interface_type": "MS_MA",
        "face_kind": "interface",
        "contact_kind": "coincident",
        "owner_semantic_ids": ("bump/1", "bump/2"),
        "adjacent_solution_volume_ids": ("substrate", "air"),
        "physical_attribute": {},
    }
    record["source_provenance"]["conductor_source_layer_name"] = "metal"
    groups = _build_groups_from_sgb_records()
    _add_surface_group_records(
        groups=groups,
        name="shared-bumps",
        phys_group=7,
        entity_tags=(11, 12),
        record=record,
        route="A",
        solution_material_kinds={"substrate": "dielectric", "air": "vacuum"},
    )
    assert set(groups["pec_surfaces"]) == {"shared-bumps__MA", "shared-bumps__MS"}
    for alias in groups["boundary_surfaces"].values():
        assert _conductor_component_ids(alias, name="alias") == ("bump/1", "bump/2")
    assert boundaries(groups["boundary_surfaces"])[0]["Terminal"][0]["Attributes"] == [
        7
    ]


def test_conflicting_provenance_and_metadata_source_lists_are_rejected():
    record = grouped_record()
    record["source_provenance"]["source_record_ids"] = ("face/1", "foreign")
    with pytest.raises(ValueError, match="exact per-source"):
        _conductor_component_ids(record, name="bumps")


@pytest.mark.parametrize("value", [None, "", " ", 7])
def test_plural_readers_require_typed_net(value):
    record = {**grouped_record(), "net_id": value}
    with pytest.raises(ValueError, match="exact net_id"):
        _conductor_component_ids(record, name="bumps")
    with pytest.raises(ValueError, match="exact net_id"):
        boundaries({"bumps": {**surface(), "net_id": value}})


def test_plural_readers_reject_missing_net():
    record = grouped_record()
    del record["net_id"]
    with pytest.raises(ValueError, match="exact net_id"):
        _conductor_component_ids(record, name="bumps")
    info = surface()
    del info["net_id"]
    with pytest.raises(ValueError, match="exact net_id"):
        boundaries({"bumps": info})


@pytest.mark.parametrize("value", ["", " ", 7])
def test_plural_readers_reject_malformed_optional_equipotential(value):
    record = {**grouped_record(), "equipotential_id": value}
    with pytest.raises(ValueError, match="optional equipotential_id"):
        _conductor_component_ids(record, name="bumps")
    with pytest.raises(ValueError, match="optional equipotential_id"):
        boundaries({"bumps": {**surface(), "equipotential_id": value}})


@pytest.mark.parametrize("omitted", [False, True])
def test_plural_readers_accept_absent_optional_equipotential(omitted):
    record = {**grouped_record(), "equipotential_id": None}
    if omitted:
        del record["equipotential_id"]
    assert _conductor_component_ids(record, name="bumps") == ("bump/1", "bump/2")
    info = surface(record)
    assert boundaries({"first": info, "second": dict(info)})[0] == {
        "Terminal": [{"Index": 1, "Attributes": [7]}]
    }


def test_shared_attribute_rejects_mixed_known_and_absent_equipotential():
    first = surface()
    other = {**first, "equipotential_id": None}
    with pytest.raises(ValueError, match="conflicting"):
        boundaries({"first": first, "other": other})
