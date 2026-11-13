"""Protect source-record normalization and handoff contracts."""

from types import SimpleNamespace

import pytest
from test_rectangular_junction_geometry import fixture

from scgsim.geometry import with_rectangular_junctions
from scgsim.geometry.models.input import (
    BoundaryCurveChainSpec,
    GdsBoundaryReconstructionSpec,
    LayoutPolygonSpec,
    SemanticEntitySpec,
)
from scgsim.geometry.models.physical import PhysicalLayerDeclaration
from scgsim.geometry.source.curves import (
    boundary_binding,
    source_curve,
    transform_boundary,
)
from scgsim.geometry.source.intents import (  # private-import: exercise the current plan-shared sheet projection.
    route_a_sheet_interfaces,
)


def test_curve_and_boundary_normalization_preserves_detached_metadata():
    """Normalize coordinates while keeping authored mapping copies detached."""
    source = {
        "boundary_id": "edge",
        "entity_id": "metal",
        "selector_point_um": [1, 2],
        "curves": [
            {
                "kind": "line_segment",
                "points_um": [[0, 0], [2, 0]],
            }
        ],
        "metadata": {"authored": {"labels": ["edge"]}},
    }

    curve = source_curve(source["curves"][0])
    binding = boundary_binding(source)
    transformed = transform_boundary(
        binding,
        path="root/child",
        transform=(0, -1, 1, 0, 10, 20),
        point=lambda matrix, xy: (
            matrix[0] * xy[0] + matrix[1] * xy[1] + matrix[4],
            matrix[2] * xy[0] + matrix[3] * xy[1] + matrix[5],
        ),
    )

    source["metadata"]["authored"]["labels"].append("caller-change")
    assert curve.points_um == ((0.0, 0.0), (2.0, 0.0))
    assert binding.metadata["authored"]["labels"] == ["edge"]
    assert isinstance(transformed, BoundaryCurveChainSpec)
    assert transformed.boundary_id == "root/child/edge"
    assert transformed.entity_id == "root/child/metal"
    assert transformed.selector_point_um == (8, 21)
    assert transformed.curves[0].points_um == ((10, 20), (10, 22))
    assert transformed.metadata["source_orientation"] == 1

    with pytest.raises(ValueError, match="unsupported source curve kind"):
        source_curve({"kind": "unknown", "points_um": []})


def test_reconstruction_binding_transforms_selector_without_curve_records():
    """Keep reconstruction selectors intact while qualifying local ownership."""
    source = {
        "boundary_id": "edge",
        "entity_id": "metal",
        "selector_point_um": [1, 2],
        "kind": "circular_arc",
        "segment_indices": [1, 3],
        "anchor_indices": [0, 2, 4],
    }

    normalized = boundary_binding(source, reconstruction=True)
    transformed = transform_boundary(
        normalized,
        path="root/child",
        transform=(-1, 0, 0, 1, 10, 0),
        point=lambda matrix, xy: (
            matrix[0] * xy[0] + matrix[1] * xy[1] + matrix[4],
            matrix[2] * xy[0] + matrix[3] * xy[1] + matrix[5],
        ),
        reconstruction=True,
    )

    assert isinstance(transformed, GdsBoundaryReconstructionSpec)
    assert transformed.selector_point_um == (9, 2)
    assert transformed.segment_indices == (1, 3)
    assert transformed.anchor_indices == (0, 2, 4)
    assert transformed.metadata["source_orientation"] == -1


def test_transform_boundary_preserves_selector_and_metadata_error_order():
    """Evaluate selector and metadata before invalid transform orientation access."""
    source = {
        "boundary_id": "edge",
        "entity_id": "metal",
        "selector_point_um": (0, 0),
        "curves": (),
        "metadata": None,
    }

    with pytest.raises(TypeError, match="mapping"):
        transform_boundary(
            source,
            path="chip",
            transform=(1,),
            point=lambda *_: (0, 0),
        )

    def fail_selector(*_):
        raise LookupError("selector transform failed")

    with pytest.raises(LookupError, match="selector transform failed"):
        transform_boundary(
            source,
            path="chip",
            transform=(1,),
            point=fail_selector,
        )


def test_generated_sheet_intent_keeps_source_identity_and_plane():
    """Emit one Route A sheet intent from its declared source polygon."""
    polygon = LayoutPolygonSpec("sheet", "metal", ((0, 0), (2, 0), (2, 1)))
    entity = SemanticEntitySpec(
        "root/metal",
        "conductor",
        "PEC",
        "conductor",
        1,
        "polygon",
        net_id="N1",
        polygon_ids=("sheet",),
        route_representations={"A": "surface_sheet"},
        geometry={"z_um": 1.5},
    )

    result = route_a_sheet_interfaces((entity,), (polygon,))

    assert result == {
        "interfaces": (
            {
                "interface_id": "MA__root/metal__AIR__0000",
                "kind": "MA",
                "owner_semantic_ids": ("root/metal", "AIR"),
                "interface_kinds": ("MS", "MA"),
                "recognition_rule": "route_a_surface_sheet_polygon",
                "intent_origin": "generated_route_a_surface_sheet",
                "source_polygon_ids": ("sheet",),
                "valid_routes": ("A",),
                "plane": {"axis": "z", "value_um": 1.5},
                "outer_loop": polygon.exterior,
                "hole_loops": (),
            },
        )
    }
    assert tuple(result["interfaces"][0]) == (
        "interface_id",
        "kind",
        "owner_semantic_ids",
        "interface_kinds",
        "recognition_rule",
        "intent_origin",
        "source_polygon_ids",
        "valid_routes",
        "plane",
        "outer_loop",
        "hole_loops",
    )


def test_plan_stack_keeps_pdk_authority_and_sizes_the_coupon(monkeypatch):
    """Build solution and conductor records from their owning source declarations."""
    import gdsfactory as gf  # noqa: PLC0415 -- optional layout stack, imported only by the tests that need it

    from scgsim.geometry import GeometryPlan  # noqa: PLC0415 -- the plan needs the optional layout stack above

    def physical_layer(name: str, material: str, interval: tuple[float, float]):
        return PhysicalLayerDeclaration(
            name,
            name,
            {"level": name},
            interval,
            material,
            "finite_volume",
            {},
            {},
        ).to_payload()

    # Restore process-local PDK state after this test.
    monkeypatch.setattr(
        gf.pdk,
        "_ACTIVE_PDK",
        gf.pdk._ACTIVE_PDK,  # noqa: SLF001 -- preserve the active PDK during test setup
    )
    gf.gpdk.PDK.activate()
    component = gf.Component()
    component.add_polygon(((0, 0), (4, 0), (4, 2), (0, 2)), layer=(1, 0))
    component.info["component_semantics"] = {
        "schema_version": 2,
        "conductor_regions": [
            {"semantic_id": "signal", "level": "metal", "gds_layer": [1, 0]}
        ],
    }
    levels = {
        "vacuum": SimpleNamespace(
            material="vacuum",
            zmin=-1,
            thickness=3,
            mesh_order=0,
            info={
                "simulation_role": "solution_region",
                "include_in_component_simulation": True,
                "physical_layer": physical_layer("vacuum", "vacuum", (-1, 2)),
            },
        ),
        "metal": SimpleNamespace(
            material="PEC",
            zmin=0,
            thickness=1,
            mesh_order=1,
            info={
                "simulation_role": "conductor",
                "part_role": "signal",
                "physical_layer": physical_layer("metal", "PEC", (0, 1)),
            },
        ),
    }
    plan = GeometryPlan(
        component,
        root_id="chip",
        layer_stack=SimpleNamespace(layers=levels),
        material_records={"PEC": {"kind": "conductor"}, "vacuum": {"kind": "vacuum"}},
        coupon_padding_um=1,
    )
    plan.set_nets({"N1": ("chip/signal",)})

    stack = plan.prepare().stack

    assert [record["semantic_id"] for record in stack["layers"]] == ["chip/signal"]
    assert (
        stack["solution_regions"]["vacuum"]["metadata"]["pdk_semantic_authority"]
        == "layer_stack"
    )
    assert stack["metadata"]["coupon_domain_bounds_um"] == {
        "x_min_um": -1.0,
        "y_min_um": -1.0,
        "x_max_um": 5.0,
        "y_max_um": 3.0,
    }


def test_rectangular_junction_binding_keeps_explicit_source_contract():
    """Bind only the two caller-selected terminal identities from the fixture."""
    build, request = fixture()

    result = with_rectangular_junctions(build, (request,))

    assert {
        overlap.host_semantic_id for overlap in result.port_sheet_regions[0].overlaps
    } == {request.terminal_a.source_entity_id, request.terminal_b.source_entity_id}
    assert result.metadata["rectangular_junctions"] == (request.to_payload(),)
    assert build.port_sheet_regions[0].overlaps == ()
