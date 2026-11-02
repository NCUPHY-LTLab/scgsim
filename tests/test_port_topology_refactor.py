"""Preserve port provenance and canonical ownership through ordered planning."""

from dataclasses import replace

import pytest
from test_geometry_context_propagation import conductor_port_input

from scgsim.geometry import ApproximationProfile, interpret_physical_geometry
from scgsim.geometry.compiler import build_route_construction_plan
from scgsim.geometry.models.topology import SurfacePlanRecord
from scgsim.geometry.planning.ports import lower_port_sheet_regions
from scgsim.geometry.planning.topology import plan_canonical_topology


def _surface(
    name: str, owner: str, loop: tuple[tuple[float, float], ...]
) -> SurfacePlanRecord:
    return SurfacePlanRecord(
        surface_id=name,
        owner_semantic_id=owner,
        surface_role="boundary",
        geometry_ref={
            "outer_loop": loop,
            "hole_loops": (),
            "plane": {"axis": "z", "value_um": 0.0},
        },
        metadata={"boundary_volume_ids": (f"volume-{owner}",)},
    )


def test_topology_splits_shared_edge_and_retains_opposite_loop_orientation() -> None:
    """Credit one atomic seam to both surfaces while preserving directed loops."""
    main = _surface("main", "a", ((0, 0), (4, 0), (4, 2), (0, 2)))
    neighbor = _surface("neighbor", "b", ((2, -1), (3, -1), (3, 0), (2, 0)))
    points, curves, loops, surfaces = plan_canonical_topology(surfaces=(main, neighbor))
    coordinates = {point.point_id: point.coordinate for point in points}
    shared = tuple(
        curve for curve in curves if curve.used_by_surface_ids == ("main", "neighbor")
    )
    assert len(shared) == 1
    seam = shared[0]
    assert {coordinates[seam.start_point_id], coordinates[seam.end_point_id]} == {
        (2.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
    }
    assert seam.owner_semantic_ids == ("a", "b")
    assert seam.boundary_volume_ids == ("volume-a", "volume-b")
    orientations = tuple(
        ref.orientation
        for loop in loops
        for ref in loop.curve_refs
        if ref.curve_id == seam.curve_id
    )
    assert set(orientations) == {-1, 1}
    assert tuple(surface.surface_id for surface in surfaces) == ("main", "neighbor")


def test_backtracking_residue_keeps_no_live_curve_claim() -> None:
    """Keep a canceled spur's point identity without exposing a live curve."""
    surface = _surface(
        "residual", "metal", ((0, 0), (2, 0), (2, 2), (1, 2), (1, 3), (1, 2), (0, 2))
    )
    points, curves, loops, _ = plan_canonical_topology(surfaces=(surface,))
    spur = next(point for point in points if point.coordinate == (1.0, 3.0, 0.0))
    assert spur.used_by_curve_ids == ()
    assert all(
        spur.point_id not in (curve.start_point_id, curve.end_point_id)
        for curve in curves
    )
    retained = {ref.curve_id for loop in loops for ref in loop.curve_refs}
    assert retained == {curve.curve_id for curve in curves}


def test_port_lowering_preserves_raw_direction_and_final_host_binding() -> None:
    """Normalize the live direction while retaining source direction and host IDs."""
    source, stack = conductor_port_input()
    region = source.port_sheet_regions[0]
    region = replace(
        region,
        metadata={
            **region.metadata,
            "direction": (3, 4, 0),
            "direction_raw": (3, 4, 0),
        },
    )
    source = replace(source, port_sheet_regions=(region,))
    effective, _, physical = interpret_physical_geometry(
        source, stack, ApproximationProfile("route_b", "v1", {})
    )
    construction = build_route_construction_plan(
        effective, route="B", representation_context=physical.context
    )
    (port,) = (
        surface
        for surface in construction.surfaces
        if surface.surface_role == "lumped_port"
    )
    assert port.metadata["physical_attribute"]["direction"] == (0.6, 0.8, 0.0)
    assert port.metadata["source_provenance"]["direction_raw"] == (3.0, 4.0, 0.0)
    assert port.metadata["source_provenance"]["host_polygon_ids"] == (
        "metal_polygon",
        "right_polygon",
    )
    assert port.metadata["embedded_volume_plan_ids"]
    assert source.port_sheet_regions[0].metadata["direction"] == (3, 4, 0)


def test_port_source_name_failure_precedes_direction_failure() -> None:
    """Read required source identity before interpreting malformed directions."""
    source, _ = conductor_port_input()
    region = source.port_sheet_regions[0]
    invalid = replace(
        region, metadata={**region.metadata, "source_name": "", "direction": ()}
    )
    source = replace(source, port_sheet_regions=(invalid,))
    with pytest.raises(ValueError, match="port_sheet requires source_name"):
        lower_port_sheet_regions(source, route="B", mm_contacts=())
