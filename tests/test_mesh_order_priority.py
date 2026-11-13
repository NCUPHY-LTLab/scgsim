"""PDK `mesh_order` (lower wins) must become internal `priority` (higher wins)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scgsim.geometry.models.physical import PhysicalLayerDeclaration

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


def _plan(monkeypatch, layers):
    """A real ``GeometryPlan`` with one metal pad on the given PDK levels."""
    import gdsfactory as gf  # noqa: PLC0415 -- optional layout stack, imported only by the tests that need it

    from scgsim.geometry import GeometryPlan  # noqa: PLC0415 -- the plan needs the optional layout stack above

    # Restore process-local PDK state after this test.
    monkeypatch.setattr(
        gf.pdk,
        "_ACTIVE_PDK",
        gf.pdk._ACTIVE_PDK,  # noqa: SLF001 -- preserve the active PDK during test setup
    )
    gf.gpdk.PDK.activate()
    component = gf.Component()
    component.add_polygon(((0, 0), (4, 0), (4, 4), (0, 4)), layer=(1, 0))
    component.info["component_semantics"] = {
        "schema_version": 2,
        "conductor_regions": [
            {"semantic_id": "metal", "level": "metal", "gds_layer": [1, 0]}
        ],
    }
    plan = GeometryPlan(
        component,
        root_id="chip",
        layer_stack=SimpleNamespace(layers=layers),
        material_records=MATERIALS,
        coupon_padding_um=1,
        grid_size_um=0.001,
    )
    plan.set_nets({"N1": ("chip/metal",)})
    return plan


@pytest.mark.parametrize("orders", ((1, 2), (2, 1), (0, 3)))
def test_stack_priority_is_negated_mesh_order_for_all_level_kinds(monkeypatch, orders):
    """Lower `mesh_order` yields the higher internal priority, on every record."""
    order_a, order_b = orders
    plan = _plan(monkeypatch, _layers(order_a, order_b, metal_order=4))
    stack = plan.prepare().stack

    regions = stack["solution_regions"]
    assert regions["diel_a"]["priority"] == -order_a
    assert regions["diel_b"]["priority"] == -order_b
    assert (regions["diel_a"]["priority"] > regions["diel_b"]["priority"]) == (
        order_a < order_b
    )
    assert stack["layers"][0]["priority"] == -4
