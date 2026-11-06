"""Official integration config/index contracts of Surface EPR; these tests do not run Palace."""

from copy import deepcopy

import pytest
from palace_sgb_support import effective_input, sidecar_of, surface_epr_specs

from scgsim.palace.mesh.physical_groups import groups_from_sidecar
from scgsim.palace.preparation.surface_epr import build_surface_epr_postprocessing
from scgsim.palace.results.surface_contract import validate_surface_epr_config


@pytest.fixture(scope="module")
def groups(tmp_path_factory):
    """The Route B group dict of the pad-in-ground build."""
    directory = tmp_path_factory.mktemp("surface_integration")
    return groups_from_sidecar(
        sidecar_of(effective_input("B", directory), "B", directory)
    )


def _produce(groups, interfaces=("MA",)):
    specs = surface_epr_specs(**{kind: None for kind in interfaces})
    rows, entries = build_surface_epr_postprocessing(groups, specs)
    return {
        "Model": {"L0": 1e-6},
        "Boundaries": {"Postprocessing": {"Dielectric": rows}},
    }, {"entries": entries}


@pytest.mark.parametrize("interface", ("MA", "MS", "SA"))
def test_official_interface_type_and_film_parameters_are_real_config_fields(
    groups, interface
):
    """Official interface type and film parameters are real config fields."""
    config, index = _produce(groups, (interface,))
    row = config["Boundaries"]["Postprocessing"]["Dielectric"][0]
    assert set(row) == {
        "Index",
        "Attributes",
        "Type",
        "Thickness",
        "Permittivity",
        "LossTan",
    }
    assert row["Type"] == interface
    assert (row["Thickness"], row["Permittivity"], row["LossTan"]) == (
        2e-9 / 1e-6,
        4,
        0.0,
    )
    assert "Mask" not in row
    assert (
        index["entries"][0]["epr_spec"]["film_thickness_m"] == row["Thickness"] * 1e-6
    )
    validate_surface_epr_config(config, index)


@pytest.mark.parametrize(
    "field,value",
    (
        ("Type", "SA"),
        ("Thickness", 0.5),
        ("Permittivity", 8),
        ("LossTan", 0.25),
        ("Attributes", [99]),
    ),
)
def test_current_config_must_match_the_producer_index(groups, field, value):
    """Current config must match the producer index."""
    config, index = _produce(groups)
    config["Boundaries"]["Postprocessing"]["Dielectric"][0][field] = value
    with pytest.raises(ValueError, match="match"):
        validate_surface_epr_config(config, index)


@pytest.mark.parametrize(
    "defect",
    ("missing_row", "missing_index", "duplicate_attribute", "duplicate_surface"),
)
def test_current_returned_inventory_cannot_omit_or_duplicate_fragments(groups, defect):
    """Current returned inventory cannot omit or duplicate fragments."""
    config, index = _produce(groups)
    rows = config["Boundaries"]["Postprocessing"]["Dielectric"]
    assert len(rows) >= 2
    if defect == "missing_row":
        rows.pop()
    elif defect == "missing_index":
        index["entries"].pop()
    elif defect == "duplicate_attribute":
        rows[1]["Attributes"] = rows[0]["Attributes"]
        index["entries"][1]["attributes"] = rows[0]["Attributes"]
    elif defect == "duplicate_surface":
        index["entries"][1]["metadata"] = deepcopy(index["entries"][0]["metadata"])
    with pytest.raises(ValueError, match="disagree|duplicated"):
        validate_surface_epr_config(config, index)


def test_distinct_interfaces_remain_distinct_native_integrands(groups):
    """Distinct interfaces remain distinct native integrands."""
    config, index = _produce(groups, ("MA", "MS", "SA"))
    validate_surface_epr_config(config, index)
    types = [
        row["Type"] for row in config["Boundaries"]["Postprocessing"]["Dielectric"]
    ]
    assert types == sorted(types)
    assert set(types) == {"MA", "MS", "SA"}


def test_native_mask_rows_are_not_accepted(groups):
    """Native Mask/Inset is not part of the stock integration, so a Mask row is rejected."""
    config, index = _produce(groups)
    config["Boundaries"]["Postprocessing"]["Dielectric"][0]["Mask"] = {
        "Type": "Inset",
        "Margin": 0.5,
    }
    with pytest.raises(ValueError, match="Mask"):
        validate_surface_epr_config(config, index)


def test_a_film_without_the_current_schema_is_rejected(groups):
    """The film snapshot has one schema; a snapshot without it is not read."""
    config, index = _produce(groups)
    index["entries"][0]["epr_spec"].pop("schema_version")
    with pytest.raises(ValueError, match="schema"):
        validate_surface_epr_config(config, index)
