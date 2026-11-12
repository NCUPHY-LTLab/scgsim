"""Preserve port provenance and canonical ownership through ordered planning."""

from scgsim.geometry.models.topology import SurfacePlanRecord
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
