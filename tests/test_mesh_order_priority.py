"""PDK `mesh_order` (lower wins) must become internal `priority` (higher wins)."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scgsim.geometry import (
    ApproximationProfile,
    GeometryPlan,
    interpret_physical_geometry,
)
from scgsim.geometry.compiler import build_route_construction_plan
from scgsim.geometry.construction.curved_geometry import arrangement
from scgsim.geometry.models.input import BoundaryCurveChainSpec, SourceCurveSpec
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


@pytest.mark.parametrize(
    ("order_a", "order_b", "vacuum_order", "winner"),
    ((1, 2, 5, "diel_a"), (2, 1, 5, "diel_b"), (1, 2, 0, "diel_a")),
)
def test_overlapping_dielectrics_resolve_to_lower_mesh_order(
    order_a, order_b, vacuum_order, winner
):
    """Within one material kind the lower `mesh_order` wins; material kind decides first.

    The last case gives vacuum the lowest `mesh_order`; a dielectric still owns
    the overlap because dielectric beats vacuum before priority is compared.
    """
    pytest.importorskip("gmsh")
    pytest.importorskip("gdsfactory")
    environment = dict(os.environ)
    source = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        (source, environment.get("PYTHONPATH", ""))
    )
    completed = subprocess.run(
        [sys.executable, __file__, str(order_a), str(order_b), str(vacuum_order)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.split()[-1] == winner, completed.stdout


def _curved_overlap_owner(order_a: int, order_b: int, vacuum_order: int) -> str:
    # Native Gmsh state is process-global, so this runs in a fresh subprocess.
    import gdsfactory as gf  # noqa: PLC0415 - optional dependency, subprocess-local

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
        root_id="coupon",
        layer_stack=SimpleNamespace(
            layers=_layers(order_a, order_b, vacuum_order=vacuum_order)
        ),
        material_records=MATERIALS,
        coupon_padding_um=1,
        grid_size_um=0.001,
    )
    snapshot = plan.prepare()
    metal = next(
        entity.semantic_id
        for entity in snapshot.geometry_input.entities
        if entity.role != "solution_region"
    )
    source = replace(
        snapshot.geometry_input,
        boundary_curves=(
            BoundaryCurveChainSpec(
                "boundary",
                metal,
                (2, 2),
                (
                    SourceCurveSpec("circular_arc", ((0, 2), (2, 0), (4, 2))),
                    SourceCurveSpec("circular_arc", ((4, 2), (2, 4), (0, 2))),
                ),
            ),
        ),
    )
    effective, _, physical = interpret_physical_geometry(
        source, snapshot.stack, ApproximationProfile("route_b", "v1", {})
    )
    owners: set[str] = set()
    original = arrangement.PlanarArrangement.connect_solution_cells

    def record_owners(self):
        owners.update(self.owners.values())
        return original(self)

    arrangement.PlanarArrangement.connect_solution_cells = record_owners
    build_route_construction_plan(
        effective, route="B", representation_context=physical.context
    )
    dielectrics = owners & {"diel_a", "diel_b"}
    assert len(dielectrics) == 1, owners
    return next(iter(dielectrics))


if __name__ == "__main__":
    print(_curved_overlap_owner(*(int(value) for value in sys.argv[1:4])))
