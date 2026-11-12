"""PDK `mesh_order` (lower wins) must become internal `priority` (higher wins)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scgsim.geometry.models.physical import PhysicalLayerDeclaration
from scgsim.geometry.source.stack import build_component_stack

MATERIALS = {
    "PEC": {"kind": "conductor"},
    "vacuum": {"kind": "vacuum"},
    "silicon": {"kind": "dielectric", "permittivity": 11.7},
    "sapphire": {"kind": "dielectric", "permittivity": 9.3},
}


def _level(name, material, interval, role, mesh_order):
    declaration = PhysicalLayerDeclaration(
        name,
        name,
        {"level": name},
        interval,
        material,
        "finite_volume",
        {},
        {},
    )
    return SimpleNamespace(
        material=material,
        zmin=interval[0],
        thickness=interval[1] - interval[0],
        mesh_order=mesh_order,
        info={
            "simulation_role": role,
            "include_in_component_simulation": True,
            "part_role": "signal",
            "physical_layer": declaration.to_payload(),
        },
    )


def _layers(order_a, order_b, metal_order=1, vacuum_order=5):
    return {
        "VACUUM_REGION": _level(
            "VACUUM_REGION", "vacuum", (-2, 3), "solution_region", vacuum_order
        ),
        "diel_a": _level("diel_a", "silicon", (-2, 0), "solution_region", order_a),
        "diel_b": _level("diel_b", "sapphire", (-2, 0), "solution_region", order_b),
        "metal": _level("metal", "PEC", (0, 1), "conductor", metal_order),
    }


def _component(schema_version=2):
    return SimpleNamespace(
        dbbox=lambda: SimpleNamespace(left=0, bottom=0, right=4, top=4),
        info={
            "component_semantics": {
                "schema_version": schema_version,
                "conductor_regions": [
                    {
                        "semantic_id": "metal",
                        "level": "metal",
                        "gds_layer": [1, 0],
                        "net_id": "N1",
                    }
                ],
            }
        },
    )


@pytest.mark.parametrize("orders", ((1, 2), (2, 1), (0, 3)))
def test_stack_priority_is_negated_mesh_order_for_all_level_kinds(orders):
    """Lower `mesh_order` yields the higher internal priority, on every record."""
    order_a, order_b = orders
    stack = build_component_stack(
        component=_component(schema_version=1),
        layer_stack=SimpleNamespace(layers=_layers(order_a, order_b, metal_order=4)),
        material_records=MATERIALS,
        coupon_padding_um=1,
    )

    regions = stack["solution_regions"]
    assert regions["diel_a"]["priority"] == -order_a
    assert regions["diel_b"]["priority"] == -order_b
    assert (regions["diel_a"]["priority"] > regions["diel_b"]["priority"]) == (
        order_a < order_b
    )
    assert stack["layers"][0]["priority"] == -4
