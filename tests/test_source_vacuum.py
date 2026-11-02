"""Contracts for deriving the explicit source vacuum envelope."""

from scgsim.geometry.models.input import VacuumRegionSpec
from scgsim.geometry.source.vacuum import apply_vacuum_region_to_stack


def test_vacuum_envelope_pads_region_xy_and_includes_stack_layer_z() -> None:
    """Padding expands source XY bounds and stack layers extend its Z range."""
    stack = {
        "solution_regions": {
            "substrate": {
                "material_id": "silicon",
                "geometry": {
                    "domain_bounds_um": {
                        "x_min_um": 0.0,
                        "x_max_um": 10.0,
                        "y_min_um": 2.0,
                        "y_max_um": 8.0,
                    },
                    "z_min_um": 0.0,
                    "thickness_um": 1.0,
                },
            }
        },
        "materials": {"silicon": {"kind": "dielectric"}},
        "layers": [
            {"semantic_id": "film", "geometry": {"z_min_um": -2.0, "z_max_um": 3.0}}
        ],
    }

    prepared = apply_vacuum_region_to_stack(
        stack, VacuumRegionSpec.from_padding((1.0, 2.0, 3.0))
    )

    vacuum = prepared["solution_regions"]["VACUUM_REGION"]
    assert vacuum["geometry"] == {
        "domain": "VACUUM_REGION",
        "domain_bounds_um": {
            "x_min_um": -1.0,
            "x_max_um": 11.0,
            "y_min_um": 0.0,
            "y_max_um": 10.0,
        },
        "z_min_um": -5.0,
        "z_max_um": 6.0,
    }
    assert prepared["materials"]["vacuum"] == {
        "kind": "vacuum",
        "permittivity": 1.0,
        "loss_tangent": 0.0,
    }
    assert "VACUUM_REGION" not in stack["solution_regions"]
    assert "vacuum" not in stack["materials"]
