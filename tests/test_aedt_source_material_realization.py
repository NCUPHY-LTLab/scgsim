"""Offline source-material contracts; no native application or simulated readback."""

import pytest

from scgsim.aedt.runtime.native.common import source_material_properties
from scgsim.geometry.source.normalization import normalize_material_records


def test_projects_only_declared_properties_without_material_defaults():
    """Project only declared scalar material properties without adding library defaults."""
    record = {
        "permittivity": 11.45,
        "permeability": 1,
        "loss_tangent": 0.0,
        "kind": "dielectric",
        "aedt_library_name": "Silicon",
        "metadata": {"note": "kept"},
    }
    assert source_material_properties(record) == {
        "permittivity": 11.45,
        "permeability": 1.0,
        "dielectric_loss_tangent": 0.0,
    }
    assert record["metadata"] == {"note": "kept"}
    assert source_material_properties({"kind": "vacuum"}) == {}


@pytest.mark.parametrize(
    "record, error",
    [
        ({"conductivity": 1.0}, TypeError),
        ({"permittivity": True}, TypeError),
        ({"permittivity": "11.45"}, TypeError),
        ({"permittivity": [11, 12, 13]}, TypeError),
        ({"permittivity": 0}, ValueError),
        ({"permittivity": float("nan")}, ValueError),
        ({"loss_tangent": -0.1}, ValueError),
        ({"loss_tangent": None}, TypeError),
    ],
)
def test_unsupported_declared_properties_fail_instead_of_using_library(record, error):
    """Reject unsupported declared property shapes before library substitution."""
    with pytest.raises(error):
        source_material_properties(record)


def test_source_normalization_preserves_declared_scalars_template_and_raw_conductivity():
    """Preserve declared scalars, library templates and raw conductivity during normalization."""

    declared = {
        "dielectric": {
            "material_kind": "dielectric",
            "relative_permittivity": 11.45,
            "permeability": 1.2,
            "aedt_library_name": "Silicon",
        },
        "conductor": {
            "material_kind": "conductive",
            "conductivity": {"value": 5.8e7, "unit": "S/m"},
        },
    }
    normalized = normalize_material_records(declared)
    assert normalized["dielectric"] == {
        "kind": "dielectric",
        "permittivity": 11.45,
        "permeability": 1.2,
        "aedt_library_name": "Silicon",
    }
    assert source_material_properties(normalized["dielectric"]) == {
        "permittivity": 11.45,
        "permeability": 1.2,
    }
    assert normalized["conductor"]["conductivity"] == {
        "value": 5.8e7,
        "unit": "S/m",
    }
    assert (
        normalized["conductor"]["conductivity"]
        is not declared["conductor"]["conductivity"]
    )
    with pytest.raises(TypeError, match="no established scalar unit contract"):
        source_material_properties(normalized["conductor"])


@pytest.mark.parametrize("permittivity", [True, [11.45, 12.0, 13.0]])
def test_source_normalization_keeps_unsupported_permittivity_for_backend_failure(
    permittivity,
):
    """Retain unsupported permittivity shapes for explicit backend rejection."""

    normalized = normalize_material_records(
        {
            "dielectric": {
                "kind": "dielectric",
                "relative_permittivity": permittivity,
                "aedt_library_name": "Silicon",
            },
        }
    )["dielectric"]
    assert normalized["permittivity"] == permittivity
    assert type(normalized["permittivity"]) is type(permittivity)
    assert normalized["aedt_library_name"] == "Silicon"
    with pytest.raises(TypeError, match="requires a scalar number"):
        source_material_properties(normalized)
