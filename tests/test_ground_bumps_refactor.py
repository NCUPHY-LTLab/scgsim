"""Direct behavior checks for the source ground-bump preparation stages."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import gdsfactory as gf
import gdstk
import pytest

from scgsim.geometry.source.ground_bump_geometry.lattice import (
    LatticeSelectionContext,
    select_lattice_sites,
)
from scgsim.geometry.source.ground_bumps import prepare_indium_ground_bump_fill


def _square(x_min: float, y_min: float, x_max: float, y_max: float) -> gdstk.Polygon:
    return gdstk.Polygon(
        [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
    )


def test_no_fill_without_authored_sites_returns_detached_stack() -> None:
    """The no-op path keeps the cell and returns an independent stack."""
    component = object()
    stack: dict[str, Any] = {
        "layers": [],
        "metadata": {"nested": {"values": ["source"]}},
    }

    result = prepare_indium_ground_bump_fill(
        component=component,
        stack=stack,
        fill=False,
        fill_pitch_um=1.0,
        fill_clearance_um=0.0,
    )

    assert result["component"] is component
    assert result["stack"] == {
        "layers": [],
        "metadata": {
            "nested": {"values": ["source"]},
            "indium_ground_bumps": {
                "controls": {
                    "fill": False,
                    "fill_pitch_um": 1.0,
                    "fill_clearance_um": 0.0,
                },
                "counts": {"authored": 0, "generated": 0, "rejected": 0},
                "source": "scgsim.sgb.ground_bumps._prepare_indium_ground_bump_fill",
            },
        },
    }
    assert result["stack"] is not stack
    assert result["stack"]["metadata"]["nested"] is not stack["metadata"]["nested"]
    assert result["injected_entities"] == ()
    assert result["injected_polygons"] == ()
    assert result["receipt"]["counts"] == {
        "authored": 0,
        "generated": 0,
        "rejected": 0,
    }


def test_authored_template_error_precedes_component_gds_access() -> None:
    """The existing template validation runs before authored GDS access."""
    with pytest.raises(
        ValueError,
        match="ground-bump fill requires exactly one PDK ground_bump_fill_spec",
    ):
        prepare_indium_ground_bump_fill(
            component=object(),
            stack={"layers": [{"part_role": "bump_body"}], "metadata": {}},
            fill=False,
            fill_pitch_um=1.0,
            fill_clearance_um=0.0,
        )


def test_lattice_selection_preserves_order_and_reports_blockers() -> None:
    """Lattice rows remain ordered and rejected sites retain blocker IDs."""
    context = LatticeSelectionContext(
        gdstk=gdstk,
        origin=(0.0, 0.0),
        pitch=1.0,
        clearance=0.0,
        body_polygon=((-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1)),
        contact_polygon=((-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1)),
        fill_prefix="BUMP_GROUND_FILL",
        geometry_digest="geometry",
        semantic_contract={
            "owner_semantic_ids": ["D0", "D1"],
            "net_id": "GND",
            "equipotential_id": "GROUND",
            "host_solution_volume_id": "VACUUM",
        },
        ground=(_square(-1.0, -1.0, 2.0, 2.0),),
        blockers=[("pdk_keepout:fixture", (_square(-0.2, -0.2, 0.2, 0.2),))],
    )

    accepted, rejected = select_lattice_sites(
        context,
        {"x_min_um": 0.0, "x_max_um": 1.0, "y_min_um": 0.0, "y_max_um": 1.0},
    )

    assert accepted == [(0, 1, 0.0, 1.0), (1, 0, 1.0, 0.0), (1, 1, 1.0, 1.0)]
    assert [record["lattice_index"] for record in rejected] == [[0, 0]]
    assert rejected[0]["reason"] == "intersects_blocking_footprint"
    assert rejected[0]["blocker_source_ids"] == ["pdk_keepout:fixture"]
    assert [source_id for source_id, _ in context.blockers[-3:]] == [
        "generated_site:BUMP_GROUND_FILL__I+0000__J+0001",
        "generated_site:BUMP_GROUND_FILL__I+0001__J+0000",
        "generated_site:BUMP_GROUND_FILL__I+0001__J+0001",
    ]


@pytest.mark.parametrize("fill", [False, True])
def test_authored_fill_path_audits_real_source_records(
    fill: bool,
) -> None:
    """Both fill modes audit real GDS ownership before returning records."""
    gf.gpdk.PDK.activate()
    component = gf.Component(f"GROUND_BUMP_AUDIT_{fill}")
    component.add_polygon(
        [(-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)], layer=(1, 0)
    )
    component.add_polygon(
        [(-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)], layer=(2, 0)
    )
    spec = {
        "schema_version": 1,
        "body_layer": [3, 0],
        "body_polygon_um": [[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]],
        "contact_layer": [4, 0],
        "contact_polygon_um": [
            [-0.15, -0.15],
            [0.15, -0.15],
            [0.15, 0.15],
            [-0.15, 0.15],
        ],
        "lattice_origin_um": [0.0, 0.0],
        "keepout_records": [],
    }
    layers = [
        {
            "semantic_id": semantic_id,
            "role": "metal",
            "material_id": "PEC",
            "layer": layer,
            "datatype": 0,
            "z_um": 0.0,
            "thickness_um": 0.1,
            "part_role": "face_metal",
            "net_id": "GND",
            "host_void_semantic_id": "VACUUM",
            "geometry": {"z_um": 0.0, "thickness_um": 0.1},
            "metadata": {"equipotential_id": "GROUND"},
        }
        for semantic_id, layer in (("D0", 1), ("D1", 2))
    ]
    layers.append(
        {
            "semantic_id": "BUMP_TEMPLATE",
            "role": "metal",
            "material_id": "PEC",
            "layer": 3,
            "datatype": 0,
            "z_um": 0.0,
            "thickness_um": 0.1,
            "part_role": "bump_body",
            "net_id": "GND",
            "host_void_semantic_id": "VACUUM",
            "geometry": {"z_um": 0.0, "thickness_um": 0.1},
            "metadata": {
                "owner_semantic_ids": ["D0", "D1"],
                "equipotential_id": "GROUND",
                "source_layer_name": "BUMP_TEMPLATE",
                "ground_bump_fill_spec": spec,
            },
        }
    )
    stack: dict[str, Any] = {
        "layers": layers,
        "solution_regions": {
            "VACUUM": {
                "role": "solution_region",
                "material_id": "VACUUM",
                "geometry_kind": "domain",
                "geometry": {
                    "z_min_um": -1.0,
                    "z_max_um": 1.0,
                    "domain_bounds_um": {
                        "x_min_um": -2.0,
                        "x_max_um": 2.0,
                        "y_min_um": -2.0,
                        "y_max_um": 2.0,
                    },
                },
            }
        },
        "materials": {"PEC": {"kind": "conductor"}, "VACUUM": {"kind": "dielectric"}},
        "metadata": {
            "top_cell_name": component.name,
            "coupon_domain_bounds_um": {
                "x_min_um": -1.0,
                "x_max_um": 1.0,
                "y_min_um": -1.0,
                "y_max_um": 1.0,
            },
        },
    }
    original_stack = deepcopy(stack)

    result = prepare_indium_ground_bump_fill(
        component=component,
        stack=stack,
        fill=fill,
        fill_pitch_um=1.0,
        fill_clearance_um=0.0,
    )

    assert result["receipt"]["counts"]["authored"] == 0
    assert result["receipt"]["semantic_contract"]["owner_semantic_ids"] == [
        "D0",
        "D1",
    ]
    assert stack == original_stack
    if fill:
        assert result["component"] is not component
        generated = result["receipt"]["counts"]["generated"]
        assert generated > 0
        assert len(result["injected_entities"]) == generated
        assert len(result["injected_polygons"]) == generated
        assert len(result["receipt"]["accepted"]) == generated
    else:
        assert result["component"] is component
        assert result["receipt"]["counts"] == {
            "authored": 0,
            "generated": 0,
            "rejected": 0,
        }
        assert result["injected_entities"] == ()
        assert result["injected_polygons"] == ()
    assert result["stack"] is not stack
    assert "indium_ground_bumps" not in stack["metadata"]
