"""Official integration config/index contracts; these tests do not run Palace."""

from copy import deepcopy

import pytest

from scgsim.palace.preparation.surface_epr import (
    build_surface_epr_postprocessing,
    normalize_surface_epr_specs,
)
from scgsim.palace.results.resolve import _validate_surface_mask_config


def _group(surface_id, attributes, interface="MA"):
    return {
        "surface_epr": True,
        "interface_type": interface,
        "surface_id": surface_id,
        "phys_group": attributes,
        "face_kind": "top",
        "owner_semantic_ids": ["owner"],
        "physical_attribute": {"groups": attributes},
        "source_provenance": {"surface_id": surface_id},
    }


def _produce(groups=None, interfaces=("MA",)):
    if groups is None:
        groups = {"patch": _group("patch", [1, 2])}
    specs = normalize_surface_epr_specs(
        {
            kind: {"film_thickness_m": 2e-9, "permittivity": 4, "loss_tangent": 0.125}
            for kind in interfaces
        },
        materials={},
    )
    rows, entries = build_surface_epr_postprocessing(
        {"boundary_surfaces": groups}, specs
    )
    return {
        "Model": {"L0": 1e-6},
        "Boundaries": {"Postprocessing": {"Dielectric": rows}},
    }, {"entries": entries}


@pytest.mark.parametrize("interface", ("MA", "MS", "SA"))
def test_official_interface_type_and_film_parameters_are_real_config_fields(interface):
    config, index = _produce(
        {"patch": _group("patch", [1, 2], interface)}, (interface,)
    )
    row = config["Boundaries"]["Postprocessing"]["Dielectric"][0]
    assert row == {
        "Index": 1,
        "Attributes": [1, 2],
        "Type": interface,
        "Thickness": 2e-9 / 1e-6,
        "Permittivity": 4,
        "LossTan": 0.125,
    }
    assert "Mask" not in row
    assert (
        index["entries"][0]["epr_spec"]["film_thickness_m"] == row["Thickness"] * 1e-6
    )
    _validate_surface_mask_config(config, index)


@pytest.mark.parametrize("attributes", ([], [1, 1], [1, "2"], [0], [True]))
def test_producer_refuses_missing_invalid_or_duplicate_attributes(attributes):
    with pytest.raises(ValueError, match="attribute"):
        _produce({"patch": _group("patch", attributes)})


@pytest.mark.parametrize("defect", ("attribute", "surface"))
def test_producer_counts_each_exact_fragment_once(defect):
    groups = {
        "patch_a": _group("first", [1]),
        "patch_b": _group(
            "second" if defect == "attribute" else "first",
            [1] if defect == "attribute" else [2],
        ),
    }
    with pytest.raises(ValueError, match="duplicated"):
        _produce(groups)


@pytest.mark.parametrize(
    "field,value",
    (
        ("Type", "MS"),
        ("Thickness", 0.5),
        ("Permittivity", 8),
        ("LossTan", 0.25),
        ("Attributes", [3]),
    ),
)
def test_current_unmasked_config_must_match_the_producer_index(field, value):
    config, index = _produce()
    config["Boundaries"]["Postprocessing"]["Dielectric"][0][field] = value
    with pytest.raises(ValueError, match="match"):
        _validate_surface_mask_config(config, index)


@pytest.mark.parametrize(
    "defect",
    ("missing_row", "missing_index", "duplicate_attribute", "duplicate_surface"),
)
def test_current_returned_inventory_cannot_omit_or_duplicate_fragments(defect):
    config, index = _produce({"a": _group("first", [1]), "b": _group("second", [2])})
    rows = config["Boundaries"]["Postprocessing"]["Dielectric"]
    if defect == "missing_row":
        rows.pop()
    elif defect == "missing_index":
        index["entries"].pop()
    elif defect == "duplicate_attribute":
        rows[1]["Attributes"] = [1]
        index["entries"][1]["attributes"] = [1]
    elif defect == "duplicate_surface":
        index["entries"][1]["metadata"]["surface_id"] = "first"
    with pytest.raises(ValueError, match="disagree|duplicated"):
        _validate_surface_mask_config(config, index)


def test_distinct_interfaces_remain_distinct_native_integrands():
    groups = {kind: _group(kind, [1], kind) for kind in ("MA", "MS", "SA")}
    config, index = _produce(groups, ("MA", "MS", "SA"))
    _validate_surface_mask_config(config, index)
    assert [
        row["Type"] for row in config["Boundaries"]["Postprocessing"]["Dielectric"]
    ] == ["MA", "MS", "SA"]


def test_historical_unmasked_reader_contract_remains_unchanged():
    config, index = _produce()
    historical = deepcopy(index)
    historical["entries"][0]["epr_spec"].pop("schema_version")
    # Historical records are not relabelled as the current producer contract.
    config["Boundaries"]["Postprocessing"]["Dielectric"][0].pop("Type")
    _validate_surface_mask_config(config, historical)
