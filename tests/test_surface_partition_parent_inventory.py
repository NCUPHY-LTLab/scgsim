"""Real retired surface parents remain distinct from semantic interfaces."""

from dataclasses import replace

import pytest

from test_geometry_context_propagation import conductor_input
from scgsim.geometry import (
    ApproximationProfile,
    SurfaceDistanceBandSpec,
    interpret_physical_geometry,
    surface_distance_band_catalogue,
    with_surface_distance_bands,
)
from scgsim.geometry.compiler import build_route_construction_plan
from scgsim.geometry.compiler.validation import (
    _validate_surface_partition_coverage,
    validate_surface_partition_coverage,
)
from scgsim.geometry.models.grid import GeometryGridContext
from scgsim.geometry.planning.surface_bands import (
    partition_surface_distance_bands,
    synchronize_distance_partitions,
)
from scgsim.geometry.source.grid import apply_geometry_grid


@pytest.fixture
def real_partition():
    source, stack = conductor_input()
    source = apply_geometry_grid(source, GeometryGridContext(0.001))
    effective, _, physical = interpret_physical_geometry(
        source, stack, ApproximationProfile("route_b", "v1", {})
    )
    baseline = build_route_construction_plan(
        effective, route="B", representation_context=physical.context
    )
    row = next(
        row
        for row in surface_distance_band_catalogue(baseline, build_input=effective)
        if row["source_bindings"]
        and row["source_bindings"][0]["source_entity_id"] == "metal"
        and len({p[2] for ring in row["rings_3d_um"] for p in ring}) == 1
    )
    binding = row["source_bindings"][0]
    request = SurfaceDistanceBandSpec(
        row["parent_surface_id"],
        binding["source_entity_id"],
        binding["source_polygon_id"],
        (0.5,),
        0.001,
    )
    banded = with_surface_distance_bands(effective, (request,))
    children, generated, _ = partition_surface_distance_bands(
        banded, surfaces=baseline.surfaces
    )
    return (
        baseline,
        banded,
        physical,
        children,
        synchronize_distance_partitions(baseline.surface_partitions, generated),
    )


def validate(baseline, children, records, *, parents=None, ancestors=None):
    return _validate_surface_partition_coverage(
        interfaces=baseline.interfaces,
        surfaces=children,
        surface_partitions=records,
        parent_surfaces=baseline.surfaces if parents is None else parents,
        parent_partitions=baseline.surface_partitions
        if ancestors is None
        else ancestors,
    )


def test_full_public_route_retains_exterior_wall_children_and_real_parents(
    real_partition,
):
    baseline, banded, physical, children, records = real_partition
    validate(baseline, children, records)
    result = build_route_construction_plan(
        banded, route="B", representation_context=physical.context
    )
    parents = {p.surface_id: p for p in baseline.surfaces}
    exteriors = [
        r
        for r in result.surface_partitions
        if r.metadata.get("layout_grid_realization")
        and parents[
            r.metadata["layout_grid_realization"]["parent_surface_id"]
        ].interface_id
        is None
    ]
    assert len(exteriors) == 4
    live = {s.surface_id: s for s in result.surfaces}
    for record in exteriors:
        parent = parents[record.parent_interface_id]
        child = live[record.child_surface_id]
        assert parent.interface_id is child.interface_id is None
        assert parent.surface_role == child.surface_role == "domain_boundary"
        assert parent.owner_semantic_id == child.owner_semantic_id == "vacuum"
        assert (
            child.metadata["surface_partition_origin"]["parent_surface_id"]
            == parent.surface_id
        )
        assert child.geometry_ref["planar_loops_3d"]
        assert parent.surface_id not in live
    assert len(result.volumes) == len(baseline.volumes)
    assert any(
        r.metadata.get("surface_distance_band") for r in result.surface_partitions
    )
    assert result.metadata["surface_distance_bands"]


@pytest.mark.parametrize("kind", ("layout_grid_realization", "surface_distance_band"))
@pytest.mark.parametrize(
    "change",
    (
        "missing",
        "unknown",
        "record_parent",
        "child_lineage",
        "owner",
        "role",
        "side",
        "routes",
        "interface",
        "ancestry",
        "source_ancestry",
        "identity_payload",
    ),
)
def test_generated_parent_identity_and_semantics_are_bound(
    real_partition, kind, change
):
    baseline, _, _, children, records = real_partition
    record = next(r for r in records if kind in r.metadata)
    child = next(c for c in children if c.surface_id == record.child_surface_id)
    identity = dict(record.metadata[kind])
    parent_id = identity["parent_surface_id"]
    parents = baseline.surfaces
    if change == "missing":
        parents = tuple(p for p in parents if p.surface_id != parent_id)
    elif change == "unknown":
        identity["parent_surface_id"] = "not-a-declared-parent"
        record = replace(record, metadata={**record.metadata, kind: identity})
        child = replace(child, metadata={**child.metadata, kind: identity})
    elif change == "record_parent":
        record = replace(record, parent_interface_id="not-the-parent")
    elif change == "child_lineage":
        child = replace(child, parent_surface_id="not-the-parent")
    elif change == "owner":
        child = replace(child, owner_semantic_id="not-the-owner")
    elif change == "role":
        child = replace(child, surface_role="not-the-role")
    elif change == "side":
        parent = next(p for p in parents if p.surface_id == parent_id)
        parents = tuple(
            replace(p, metadata={**p.metadata, "surface_side": "declared-side"})
            if p is parent
            else p
            for p in parents
        )
        child = replace(
            child, metadata={**child.metadata, "surface_side": "wrong-side"}
        )
    elif change == "routes":
        child = replace(child, valid_routes=("C",))
    elif change == "interface":
        child = replace(child, interface_id="MA__invented__interface")
    elif change == "ancestry":
        record = replace(
            record,
            metadata={
                **record.metadata,
                "ancestor_surface_partitions": ({"child_surface_id": parent_id},),
            },
        )
    elif change == "source_ancestry":
        if kind == "layout_grid_realization":
            child = replace(
                child,
                metadata={
                    **child.metadata,
                    "surface_partition_origin": {"parent_surface_id": "wrong-origin"},
                },
            )
        else:
            identity["distance_origin_parent_surface_id"] = "wrong-origin"
            record = replace(record, metadata={**record.metadata, kind: identity})
            child = replace(child, metadata={**child.metadata, kind: identity})
    elif change == "identity_payload":
        child = replace(
            child,
            metadata={
                **child.metadata,
                kind: {**identity, "parent_surface_id": "wrong-parent"},
            },
        )
    records = tuple(
        record if r.child_surface_id == record.child_surface_id else r for r in records
    )
    children = tuple(child if c.surface_id == child.surface_id else c for c in children)
    with pytest.raises(ValueError, match="Invalid surface partition plan"):
        validate(baseline, children, records, parents=parents)


@pytest.mark.parametrize("where", ("parent", "partition", "child"))
def test_conflicting_duplicate_records_are_rejected(real_partition, where):
    baseline, _, _, children, records = real_partition
    if where == "parent":
        parents = (
            *baseline.surfaces,
            replace(baseline.surfaces[0], owner_semantic_id="conflict"),
        )
    else:
        parents = baseline.surfaces
        if where == "partition":
            records = (*records, replace(records[0], parent_interface_id="conflict"))
        else:
            children = (*children, replace(children[0], owner_semantic_id="conflict"))
    with pytest.raises(ValueError, match="conflicting"):
        validate(baseline, children, records, parents=parents)


def test_public_validator_cannot_invent_inventory_from_layout_metadata(real_partition):
    baseline, _, _, children, records = real_partition
    with pytest.raises(ValueError, match="unknown parent interface"):
        validate_surface_partition_coverage(
            interfaces=baseline.interfaces,
            surfaces=children,
            surface_partitions=records,
        )


def test_actual_explicit_partition_ancestor_is_preserved_and_bound():
    from test_surface_distance_bands import fixture
    from scgsim.geometry.models.construction import SurfacePartitionRecord
    from scgsim.geometry.models.representation import RepresentationContext
    from scgsim.geometry.models.topology import InterfacePlanRecord
    from scgsim.geometry.planning.surfaces import plan_route_surfaces

    build, parent = fixture()
    interface = InterfacePlanRecord(
        parent.interface_id,
        "SA",
        ("entity", "vacuum"),
        "explicit",
        source_polygon_ids=("polygon",),
        metadata={
            "outer_loop": ((0, 0), (8, 0), (8, 8), (0, 8)),
            "plane": {"axis": "z", "value_um": 0},
        },
    )
    ancestor = SurfacePartitionRecord(
        "left-partition",
        parent.interface_id,
        "left",
        "left",
        valid_routes=("C",),
        metadata={
            "outer_loop": ((0, 0), (4, 0), (4, 8), (0, 8)),
            "source_fact": "retain",
        },
    )
    parents = plan_route_surfaces(
        build,
        route="C",
        interfaces=(interface,),
        surface_partitions=(ancestor,),
        construction_bodies=(),
        representation_context=RepresentationContext({}),
    )
    banded = with_surface_distance_bands(
        build, (SurfaceDistanceBandSpec("left", "entity", "polygon", (1,), 0.001),)
    )
    children, records, _ = partition_surface_distance_bands(banded, surfaces=parents)
    records = synchronize_distance_partitions((ancestor,), records)
    arguments = dict(
        interfaces=(interface,),
        surfaces=children,
        surface_partitions=records,
        parent_surfaces=parents,
        parent_partitions=(ancestor,),
    )
    _validate_surface_partition_coverage(**arguments)
    assert all(
        record.parent_interface_id == interface.interface_id for record in records
    )
    assert all(
        record.metadata["ancestor_surface_partitions"][0]["metadata"]
        == ancestor.metadata
        for record in records
    )
    forged = replace(
        records[0],
        metadata={
            **records[0].metadata,
            "ancestor_surface_partitions": (
                {
                    **records[0].metadata["ancestor_surface_partitions"][0],
                    "label": "forged",
                },
            ),
        },
    )
    with pytest.raises(ValueError, match="ancestor partition mismatch"):
        _validate_surface_partition_coverage(
            **{**arguments, "surface_partitions": (forged, *records[1:])}
        )


@pytest.mark.parametrize("kind", ("layout_grid_realization", "surface_distance_band"))
def test_removing_lineage_cannot_invent_an_interface_parent(real_partition, kind):
    baseline, _, _, children, records = real_partition
    record = next(r for r in records if kind in r.metadata)
    child = next(s for s in children if s.surface_id == record.child_surface_id)
    invented = "MA__not_declared__not_a_parent"
    record = replace(record, parent_interface_id=invented, metadata={})
    child = replace(child, interface_id=invented, metadata={})
    children = tuple(child if s.surface_id == child.surface_id else s for s in children)
    records = tuple(
        record if r.child_surface_id == record.child_surface_id else r for r in records
    )
    with pytest.raises(ValueError, match="lacks generated construction parent lineage"):
        validate(baseline, children, records)
