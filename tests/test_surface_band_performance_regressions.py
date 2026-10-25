"""Build-local work reuse preserves exact source-wall incidence and topology."""

from dataclasses import replace

import pytest

from scgsim.geometry import SurfaceDistanceBandSpec, with_surface_distance_bands
from scgsim.geometry.compiler import build_route_construction_plan
from scgsim.geometry.models.grid import GeometryGridContext
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.planning.surface_bands import (
    _WallConstructionIndex,
    _layout_band_construction,
    _wall_construction_chain,
    surface_distance_band_catalogue,
)
from scgsim.geometry.source.grid import apply_geometry_grid
from test_surface_distance_bands import SQUARE, fixture


class CountedLedger:
    """Observe iteration of actual producer rows without replacing their data."""

    def __init__(self, rows):
        self.rows = rows
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        yield from self.rows


def layout_fixture():
    build, parent = fixture()
    bottom = replace(
        parent,
        surface_id="bottom",
        geometry_ref={
            **parent.geometry_ref,
            "z_um": -8,
        },
    )
    neighbor = replace(
        parent,
        surface_id="neighbor",
        geometry_ref={
            **parent.geometry_ref,
            "outer_loop": tuple((x + 20, y) for x, y in SQUARE),
        },
    )
    spec = SurfaceDistanceBandSpec.from_payload(
        build.metadata["surface_distance_bands"][0]
    )
    return _layout_band_construction(
        (parent, bottom, neighbor),
        {parent.surface_id: spec},
        GeometryGridContext(0.001),
    )


def test_wall_index_scans_real_ledger_once_and_preserves_order():
    layout = layout_fixture()
    ledger = CountedLedger(layout["ledger"])
    indexed = {**layout, "ledger": ledger}
    index = _WallConstructionIndex(indexed)
    expected = tuple(
        row
        for row in layout["ledger"]
        if {tuple(point[:2]) for point in row["source_endpoints"]} == {(0, 0), (8, 0)}
    )
    assert len(expected) == 2  # Opposite caps retain both source origins.
    assert index.candidates((0, 0), (8, 0)) == expected
    assert index.candidates((8, 0), (0, 0)) == expected
    assert len(layout["ledger"]) == 12
    for wall in ("wall", "same_xy_at_other_z", "another_band"):
        assert _wall_construction_chain(
            ((0, 0), (8, 0)), indexed, wall, index=index
        ) == (
            (0, 0),
            (8, 0),
            ((0, 0), (8, 0)),
        )
    assert ledger.iterations == 1
    assert len(index.chains) == 1
    assert tuple(ledger.rows) == layout["ledger"]


def test_wall_cache_keeps_full_source_anchor_order_and_rejects_missing_incidence():
    layout = layout_fixture()
    index = _WallConstructionIndex(layout)
    first = _wall_construction_chain(((0, 0), (8, 0)), layout, "first", index=index)
    assert (
        _wall_construction_chain(((0, 0), (8, 0)), layout, "repeat", index=index)
        is first
    )
    retained = _wall_construction_chain(
        ((0, 0), (4, 0), (8, 0)), layout, "retained", index=index
    )
    assert retained[2] == ((0, 0), (4, 0), (8, 0))
    reverse = _wall_construction_chain(
        ((8, 0), (4, 0), (0, 0)), layout, "reverse", index=index
    )
    assert reverse == retained
    assert len(index.chains) == 3
    for name in ("missing_a", "missing_b"):
        with pytest.raises(
            ValueError, match=f"missing retained boundary-anchor incidence: {name}"
        ):
            _wall_construction_chain(((1, 0), (9, 0)), layout, name, index=index)
    assert len(index.chains) == 3  # Failures are not cached with a previous surface ID.


def test_index_retains_ambiguous_realized_chains_instead_of_selecting_one():
    layout = layout_fixture()
    original = next(
        row
        for row in layout["ledger"]
        if {tuple(point[:2]) for point in row["source_endpoints"]} == {(0, 0), (8, 0)}
    )
    fork = {**original, "realized_xy_chain": ((0, 0), (4, 1), (8, 0))}
    layout = {**layout, "ledger": (*layout["ledger"], fork)}
    index = _WallConstructionIndex(layout)
    assert len(index.candidates((0, 0), (8, 0))) == 3
    with pytest.raises(
        ValueError, match="ambiguous original-segment construction incidence"
    ):
        _wall_construction_chain(((0, 0), (8, 0)), layout, "fork", index=index)
    assert not index.chains


def test_compiler_reports_band_and_synchronization_timings_with_live_volume_closure():
    build, _ = fixture()
    entity = replace(
        build.entities[0],
        role="solution_region",
        geometry={
            "outer_loop": SQUARE,
            "z_min_um": 0,
            "z_max_um": 8,
            "thickness_um": 8,
            "domain_bounds_um": {
                "x_min_um": 0,
                "x_max_um": 8,
                "y_min_um": 0,
                "y_max_um": 8,
                "z_min_um": 0,
                "z_max_um": 8,
            },
        },
    )
    build = apply_geometry_grid(
        replace(build, entities=(entity,), metadata={}), GeometryGridContext(0.001)
    )
    context = RepresentationContext({})
    original = build_route_construction_plan(
        build, route="C", representation_context=context
    )
    parent = next(
        row
        for row in surface_distance_band_catalogue(original)
        if all(point[2] == 8 for point in row["rings_3d_um"][0])
    )
    spec = SurfaceDistanceBandSpec(
        parent["parent_surface_id"], "entity", "polygon", (1, 2), 0.001
    )
    plan = build_route_construction_plan(
        with_surface_distance_bands(build, (spec,)),
        route="C",
        representation_context=context,
    )
    assert len(plan.volumes) == 1
    assert plan.surface_partitions
    assert all(
        ref.surface_id != spec.parent_surface_id
        for volume in plan.volumes
        for ref in volume.surface_refs
    )
    for stage in (
        "partition_surface_distance_bands",
        "synchronize_distance_partitions",
    ):
        rows = [row for row in plan.metadata["timings"] if row["stage"] == stage]
        assert len(rows) == 1
        assert rows[0]["status"] == "done"
