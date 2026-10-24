"""Conductor-bearing public polygon planning with the one physical context."""

import pytest

from scgsim.geometry import (
    ApproximationProfile,
    GeometryBuildInput,
    LayoutPolygonSpec,
    SemanticEntitySpec,
    interpret_physical_geometry,
    surface_distance_band_catalogue,
)
from scgsim.geometry.compiler import build_route_construction_plan


def conductor_input():
    metal_loop = ((0, 0), (4, 0), (4, 4), (0, 4))
    vacuum_loop = ((-2, -2), (6, -2), (6, 6), (-2, 6))
    polygon = LayoutPolygonSpec("metal_polygon", "metal", metal_loop)
    metal = SemanticEntitySpec(
        "metal",
        "conductor",
        "metal_material",
        "conductor",
        1,
        "polygon",
        net_id="signal",
        polygon_ids=(polygon.polygon_id,),
        host_void_semantic_id="vacuum",
        geometry={
            "outer_loop": metal_loop,
            "z_min_um": 0,
            "z_max_um": 1,
            "z_um": 0,
            "thickness_um": 1,
        },
    )
    vacuum = SemanticEntitySpec(
        "vacuum",
        "solution_region",
        "vacuum_material",
        "vacuum",
        0,
        "polygon",
        geometry={
            "outer_loop": vacuum_loop,
            "z_min_um": -2,
            "z_max_um": 6,
            "thickness_um": 8,
            "domain_bounds_um": {
                "x_min_um": -2,
                "x_max_um": 6,
                "y_min_um": -2,
                "y_max_um": 6,
                "z_min_um": -2,
                "z_max_um": 6,
            },
        },
    )
    stack = {
        "layers": [
            {"semantic_id": "metal", "geometry": dict(metal.geometry), "metadata": {}}
        ],
        "solution_regions": {
            "vacuum": {
                "semantic_id": "vacuum",
                "geometry": dict(vacuum.geometry),
                "metadata": {},
            }
        },
        "materials": {
            "metal_material": {"kind": "conductor"},
            "vacuum_material": {"kind": "vacuum", "permittivity": 1},
        },
        "metadata": {},
    }
    return GeometryBuildInput((polygon,), (metal, vacuum)), stack


@pytest.mark.parametrize("route,profile", (("B", "route_b"), ("C", "solid")))
def test_public_conductor_catalogue_forwards_existing_physical_context(route, profile):
    source, stack = conductor_input()
    effective, _, physical = interpret_physical_geometry(
        source, stack, ApproximationProfile(profile, "v1", {})
    )
    original_context = physical.context
    construction = build_route_construction_plan(
        effective, route=route, representation_context=original_context
    )
    rows = surface_distance_band_catalogue(construction, build_input=effective)
    assert rows
    assert construction.surfaces and construction.volumes and construction.tags
    assert any("metal" in row["source_entity_ids"] for row in rows)
    assert (
        tuple(map(tuple, effective.entities[0].geometry["outer_loop"]))
        == source.entities[0].geometry["outer_loop"]
    )
    assert (
        effective.entities[0].geometry["z_min_um"],
        effective.entities[0].geometry["z_max_um"],
    ) == (0, 1)
    assert original_context.entities["metal"].representation == "solid"
    if route == "B":
        assert construction.construction_bodies


def test_public_conductor_catalogue_still_rejects_missing_context():
    source, _ = conductor_input()
    with pytest.raises(ValueError, match="physical_geometry"):
        build_route_construction_plan(source, route="B")


def conductor_port_input():
    from dataclasses import replace
    from scgsim.geometry import PortSheetOverlapRecord, PortSheetRegionRecord

    source, stack = conductor_input()
    left = replace(
        source.entities[0],
        part_role="face_metal",
        metadata={"logical_layer_id": "metal"},
    )
    right_loop = ((6, 0), (10, 0), (10, 4), (6, 4))
    right = replace(
        left,
        semantic_id="right",
        net_id="other",
        polygon_ids=("right_polygon",),
        geometry={**left.geometry, "outer_loop": right_loop},
    )
    vacuum = replace(
        source.entities[1],
        geometry={
            **source.entities[1].geometry,
            "outer_loop": ((-2, -2), (12, -2), (12, 6), (-2, 6)),
            "domain_bounds_um": {
                **source.entities[1].geometry["domain_bounds_um"],
                "x_max_um": 12,
            },
        },
    )
    port_loop = ((3, 1), (7, 1), (7, 3), (3, 3))
    left_overlap = ((3, 1), (4, 1), (4, 3), (3, 3))
    right_overlap = ((6, 1), (7, 1), (7, 3), (6, 3))
    port = PortSheetRegionRecord(
        "port_sheet",
        "port",
        "port_polygon",
        port_loop,
        overlaps=(
            PortSheetOverlapRecord(
                "left_overlap",
                "port_sheet",
                "port_polygon",
                "metal",
                "metal_polygon",
                left_overlap,
            ),
            PortSheetOverlapRecord(
                "right_overlap",
                "port_sheet",
                "port_polygon",
                "right",
                "right_polygon",
                right_overlap,
            ),
        ),
        metadata={
            "source_name": "bridge",
            "target_layer": "metal",
            "port_index": 1,
            "direction": (1, 0, 0),
            "direction_raw": (1, 0, 0),
            "direction_sign_convention": "source_to_return",
        },
    )
    stack = {
        **stack,
        "layers": [
            {
                "semantic_id": entity.semantic_id,
                "geometry": dict(entity.geometry),
                "metadata": dict(entity.metadata),
            }
            for entity in (left, right)
        ],
        "solution_regions": {
            "vacuum": {
                "semantic_id": "vacuum",
                "geometry": dict(vacuum.geometry),
                "metadata": {},
            }
        },
    }
    return replace(
        source,
        entities=(left, right, vacuum),
        polygons=(
            *source.polygons,
            LayoutPolygonSpec("right_polygon", "metal", right_loop),
            LayoutPolygonSpec("port_polygon", "port", port_loop),
        ),
        port_sheet_regions=(port,),
    ), stack


def test_public_port_catalogue_accepts_selected_route_only():
    source, stack = conductor_port_input()
    effective, _, physical = interpret_physical_geometry(
        source, stack, ApproximationProfile("route_b", "v1", {})
    )
    construction = build_route_construction_plan(
        effective, route="B", representation_context=physical.context
    )
    ports = [
        surface
        for surface in construction.surfaces
        if surface.surface_role == "lumped_port"
    ]
    assert len(ports) == 1
    assert ports[0].metadata["owner_semantic_ids"] == ("metal", "right")
    assert ports[0].metadata["source_provenance"]["host_polygon_ids"] == (
        "metal_polygon",
        "right_polygon",
    )
    assert set(ports[0].geometry_ref["outer_loop"]) == {
        (4.0, 1.0),
        (6.0, 1.0),
        (6.0, 3.0),
        (4.0, 3.0),
    }


def test_route_c_ports_remain_explicitly_unsupported():
    source, stack = conductor_port_input()
    effective, _, physical = interpret_physical_geometry(
        source, stack, ApproximationProfile("solid", "v1", {})
    )
    with pytest.raises(ValueError, match="Route C does not support Palace lumped-port"):
        build_route_construction_plan(
            effective, route="C", representation_context=physical.context
        )


@pytest.mark.parametrize(
    "mutation,expected",
    (
        ("role", "active A/B face_metal"),
        ("layer", "active A/B face_metal"),
        ("plane", "same finite z_um"),
        ("thickness", "same finite positive thickness_um"),
    ),
)
def test_selected_route_keeps_existing_host_contracts(mutation, expected):
    from dataclasses import replace

    source, stack = conductor_port_input()
    right = source.entities[1]
    if mutation == "role":
        right = replace(right, part_role=None)
    elif mutation == "layer":
        right = replace(right, metadata={"logical_layer_id": "different_layer"})
    elif mutation == "plane":
        right = replace(
            right, geometry={**right.geometry, "z_um": 2, "z_min_um": 2, "z_max_um": 3}
        )
    else:
        right = replace(
            right, geometry={**right.geometry, "z_max_um": 2, "thickness_um": 2}
        )
    source = replace(source, entities=(source.entities[0], right, source.entities[2]))
    effective, _, physical = interpret_physical_geometry(
        source, stack, ApproximationProfile("route_b", "v1", {})
    )
    with pytest.raises(ValueError, match=expected):
        build_route_construction_plan(
            effective, route="B", representation_context=physical.context
        )


def test_polygonless_sa_catalogue_binding_partitions_actual_parent():
    from dataclasses import replace
    from scgsim.geometry import SurfaceDistanceBandSpec, with_surface_distance_bands

    source, stack = conductor_input()
    vacuum = source.entities[1]
    substrate = replace(
        vacuum, semantic_id="substrate", material_id="substrate_material",
        material_kind="dielectric",
        geometry={**vacuum.geometry, "z_min_um": -2, "z_max_um": 0,
                  "thickness_um": 2,
                  "domain_bounds_um": {**vacuum.geometry["domain_bounds_um"], "z_max_um": 0}},
    )
    vacuum = replace(
        vacuum, geometry={**vacuum.geometry, "z_min_um": 0, "thickness_um": 6,
                          "domain_bounds_um": {**vacuum.geometry["domain_bounds_um"], "z_min_um": 0}},
    )
    source = replace(source, polygons=(), entities=(substrate, vacuum))
    stack = {**stack, "layers": [], "materials": {
        **stack["materials"], "substrate_material": {"kind": "dielectric", "permittivity": 4}},
        "solution_regions": {entity.semantic_id: {
            "semantic_id": entity.semantic_id, "geometry": dict(entity.geometry), "metadata": {}}
            for entity in source.entities}}
    effective, _, physical = interpret_physical_geometry(
        source, stack, ApproximationProfile("route_b", "v1", {})
    )
    construction = build_route_construction_plan(
        effective, route="B", representation_context=physical.context
    )
    rows = surface_distance_band_catalogue(construction, build_input=effective)
    row = next(row for row in rows if row["source_entity_ids"] == ("substrate", "vacuum"))
    assert row["source_polygon_ids"] == ()
    assert all(item["source_bindings"] for item in rows if not item["source_polygon_ids"])
    assert {tuple(binding.items()) for binding in row["source_bindings"]} == {
        (("source_entity_id", "substrate"), ("source_polygon_id", None)),
        (("source_entity_id", "vacuum"), ("source_polygon_id", None)),
    }
    spec = SurfaceDistanceBandSpec(row["parent_surface_id"], "substrate", None, (1,), 0.001)
    assert "source_polygon_id" in spec.to_payload()
    assert SurfaceDistanceBandSpec.from_payload(spec.to_payload()) == spec
    result = build_route_construction_plan(
        with_surface_distance_bands(effective, (spec,)), route="B",
        representation_context=physical.context,
    )
    summary = result.metadata["surface_distance_bands"]
    assert summary and all(item["source_polygon_id"] is None for item in summary)
    assert all(item["parent_source_polygon_ids"] == () for item in summary)
    assert any(surface.metadata.get("surface_distance_band") for surface in result.surfaces)
    assert len(result.volumes) == len(construction.volumes)
    missing_key = spec.to_payload()
    del missing_key["source_polygon_id"]
    with pytest.raises(KeyError, match="source_polygon_id"):
        SurfaceDistanceBandSpec.from_payload(missing_key)


def test_polygon_backed_parent_rejects_null_anchor():
    from scgsim.geometry import SurfaceDistanceBandSpec, with_surface_distance_bands

    source, stack = conductor_input()
    effective, _, physical = interpret_physical_geometry(
        source, stack, ApproximationProfile("route_b", "v1", {})
    )
    construction = build_route_construction_plan(effective, route="B", representation_context=physical.context)
    row = next(row for row in surface_distance_band_catalogue(construction, build_input=effective)
               if "metal_polygon" in row["source_polygon_ids"])
    spec = SurfaceDistanceBandSpec(row["parent_surface_id"], "metal", None, (0.25,), 0.001)
    with pytest.raises(ValueError, match="polygon-backed parent"):
        build_route_construction_plan(with_surface_distance_bands(effective, (spec,)), route="B",
                                      representation_context=physical.context)


def test_port_split_wall_retains_original_boundary_and_clips_each_child():
    from scgsim.geometry import SurfaceDistanceBandSpec, with_surface_distance_bands

    source, stack = conductor_port_input()
    effective, _, physical = interpret_physical_geometry(
        source, stack, ApproximationProfile("route_b", "v1", {})
    )
    construction = build_route_construction_plan(effective, route="B", representation_context=physical.context)
    rows = {row["parent_surface_id"]: row for row in
            surface_distance_band_catalogue(construction, build_input=effective)}
    walls = [surface for surface in construction.surfaces
             if surface.metadata.get("route_b_port_sheet_sidewall_partition")]
    assert len(walls) == 4
    for wall in walls:
        row = rows[wall.surface_id]
        assert row["distance_origin_status"] == "supported"
        assert {point[2] for ring in row["distance_origin_rings_3d_um"] for point in ring} == {0, 1}
        child_z = {point[2] for point in wall.geometry_ref["quad_points"]}
        assert 0.5 in child_z
        spec = SurfaceDistanceBandSpec(wall.surface_id, wall.owner_semantic_id,
                                       rows[wall.surface_id]["source_bindings"][0]["source_polygon_id"],
                                       (0.25,), 0.001)
        result = build_route_construction_plan(with_surface_distance_bands(effective, (spec,)), route="B",
                                              representation_context=physical.context)
        bands = [surface for surface in result.surfaces if surface.metadata.get("surface_distance_band")]
        assert bands
        # A seam treated as an origin would consume this half-height child
        # completely at cut 0.25; the original full wall leaves a live core.
        assert result.metadata["surface_distance_bands"][-1]["child_surface_ids"]
        assert all(min(child_z) <= point[2] <= max(child_z)
                   for surface in bands for ring in surface.geometry_ref["planar_loops_3d"] for point in ring)
        # An original full-height wall has no edge at the port's interior plane.
        assert all(not (a[2] == b[2] == 0.5)
                   for ring in row["distance_origin_rings_3d_um"]
                   for a, b in zip(ring, (*ring[1:], ring[0])))
        assert len(result.volumes) == len(construction.volumes)
