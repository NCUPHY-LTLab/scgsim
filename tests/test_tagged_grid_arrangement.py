"""Public constructive grid regressions; no native mesher or study geometry."""

from fractions import Fraction
import json
import threading

import pytest
from shapely import Polygon, Point, coverage_is_valid, union_all

from scgsim.geometry._primitives.grid import GridOperand, build_grid_arrangement
from scgsim.geometry._primitives._tagged_grid import (
    _PointIndex,
    _pixel_interval,
    _source_paths,
    build_tagged_subdivision,
)
from scgsim.geometry.models.grid import GeometryGridContext
from scgsim.preparation import preparation_resources


def rectangle(x0, y0, x1, y1):
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def operands_for(polygons):
    return tuple(
        GridOperand(
            str(index),
            tuple(p.exterior.coords)[:-1],
            tuple(tuple(r.coords)[:-1] for r in p.interiors),
            physical=index == 0,
            contact=False,
        )
        for index, p in enumerate(polygons)
    )


def construct(polygons, grid):
    values = operands_for(polygons)
    shapes = dict(zip((v.operand_id for v in values), polygons, strict=True))
    return values, build_tagged_subdivision(values, shapes, grid)


def disk_fixture():
    return (Polygon(rectangle(-5, -5, 5, 5)),) + tuple(
        Point(x, y).buffer(3.2, quad_segs=8) for x, y in ((-1, 0), (1, 0), (0, 1))
    )


def test_global_partition_preserves_parent_before_any_resnap_union():
    polygons = disk_fixture()
    values, result = construct(polygons, 0.1)
    parent_cells = [cell for cell, owners in result.cells if "0" in owners]
    joined = union_all(parent_cells, grid_size=0)
    assert joined.equals(polygons[0])
    assert joined.area == 100
    assert joined.geom_type == "Polygon"
    assert len(joined.interiors) == 0
    assert coverage_is_valid([cell for cell, _ in result.cells])
    graph = build_grid_arrangement(values, GeometryGridContext(0.1))
    regions = graph.operand_regions("0")
    assert len(regions) == 1 and regions[0][1] == ()
    assert Polygon(*regions[0]).equals(polygons[0])
    assert all(origins for _, origins in result.edge_origins)
    assert len(result.source_paths) == 100


def test_reversed_holes_and_opposed_coincident_origins_remain_distinct():
    outer, hole = rectangle(-3, -3, 3, 3), rectangle(-1, -1, 1, 1)
    polygons = (
        Polygon(rectangle(-5, -5, 5, 5)),
        Polygon(outer, (hole,)),
        Polygon(tuple(reversed(outer)), (tuple(reversed(hole)),)),
    )
    _, result = construct(polygons, 1)
    for owner in ("1", "2"):
        joined = union_all(
            [cell for cell, owners in result.cells if owner in owners], grid_size=0
        )
        assert joined.equals(polygons[int(owner)])
        assert joined.area == 32
        assert len(joined.interiors) == 1
    assert {(p.operand_index, p.ring_index) for p in result.source_paths} == {
        (0, 0),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
    }
    opposed = []
    for edge, origins in result.edge_origins:
        if len(origins) > 1:
            assert {
                result.source_paths[index].operand_index for index, _ in origins
            } == {1, 2}
            opposed.append(tuple(direction for _, direction in origins))
    assert len(opposed) == 8
    assert all(sorted(directions) == [-1, 1] for directions in opposed)


@pytest.mark.parametrize(
    "grid,triangle",
    (
        (1, ((-0.5, -0.5), (1.5, -0.5), (0.5, 1.5))),
        (0.1, ((0.35, 0.35), (1.35, 0.35), (0.85, 1.35))),
    ),
)
def test_native_half_ties_bind_endpoint_incidence(grid, triangle):
    polygons = (Polygon(rectangle(-5, -5, 5, 5)), Polygon(triangle))
    _, result = construct(polygons, grid)
    source = [path for path in result.source_paths if path.operand_index == 1]
    assert source
    for first, second in zip(source, (*source[1:], source[0])):
        assert first.nodes[-1] == second.nodes[0]
    joined = union_all(
        [cell for cell, owners in result.cells if "0" in owners], grid_size=0
    )
    assert joined.equals(polygons[0])


def test_contacts_and_zero_area_source_origins_are_not_deleted():
    polygons = (
        Polygon(rectangle(-5, -5, 5, 5)),
        Polygon(rectangle(0, 0, 0.25, 2)),
        Polygon(rectangle(2, 2, 2.25, 2.25)),
    )
    _, result = construct(polygons, 1)
    assert all(owners == ("0",) for _, owners in result.cells)
    assert any(
        shape.geom_type == "LineString" and owners == ("1",)
        for shape, owners in result.auxiliary
    )
    assert any(
        shape.geom_type == "Point" and "2" in owners
        for shape, owners in result.auxiliary
    )
    point_paths = [path for path in result.source_paths if path.operand_index == 2]
    assert len(point_paths) == 4
    assert all(path.nodes == ((2, 2),) for path in point_paths)
    # Ordinary exact line and corner contacts still pass the original DE9IM guard.
    values = (
        GridOperand("A", rectangle(-3, -1, 0, 1)),
        GridOperand("B", rectangle(0, -1, 3, 1)),
        GridOperand("C", rectangle(3, 1, 4, 2)),
    )
    build_grid_arrangement(values, GeometryGridContext(1))


def test_direction_and_missing_endpoint_context_are_explicit_refusals():
    bowtie = GridOperand("invalid", ((0, 0), (2, 2), (0, 2), (2, 0)))
    with pytest.raises(ValueError, match="source ring direction") as caught:
        build_grid_arrangement((bowtie,), GeometryGridContext(1))
    context = json.loads(caught.value.__notes__[0].split(": ", 1)[1])
    assert context == dict(operand_index=0, ring_index=0)
    row = ("owner", 0, 0, 0, (0, 0), (2, 0), ((0, 0), (2, 0)), 1)
    with pytest.raises(ValueError, match="source endpoint pixel incidence"):
        _source_paths((row,), _PointIndex.build(((0, 0),)), 1)
    # Exact half-open geometry: point-touching top/right is excluded, while
    # the lower-left corner belongs to the pixel. No numerical tolerance.
    assert (
        _pixel_interval(
            (Fraction(1, 2), Fraction(1, 2)), (Fraction(1), Fraction(1)), (0, 0)
        )
        is None
    )
    assert _pixel_interval(
        (Fraction(-1, 2), Fraction(-1, 2)), (Fraction(-1), Fraction(-1)), (0, 0)
    ) == (0, 0)


def test_parallel_source_paths_preserve_exact_graph_and_real_worker_execution():
    polygons = disk_fixture()
    values, serial = construct(polygons, 0.1)
    main = threading.get_ident()
    seen = set()
    lock = threading.Lock()
    old = threading.getprofile()

    def observe(frame, event, arg):
        if event == "call" and frame.f_code is _source_paths.__code__:
            with lock:
                seen.add(threading.get_ident())

    threading.setprofile(observe)
    try:
        with preparation_resources(max_workers=2):
            _, parallel = construct(polygons, 0.1)
            graph = build_grid_arrangement(values, GeometryGridContext(0.1))
    finally:
        threading.setprofile(old)
    assert seen and main not in seen
    assert parallel.source_paths == serial.source_paths
    assert parallel.edge_origins == serial.edge_origins
    assert [(p.wkb, owners) for p, owners in parallel.cells] == [
        (p.wkb, owners) for p, owners in serial.cells
    ]
    assert [(p.wkb, owners) for p, owners in parallel.auxiliary] == [
        (p.wkb, owners) for p, owners in serial.auxiliary
    ]
    assert graph == build_grid_arrangement(values, GeometryGridContext(0.1))


def test_cached_pure_partition_does_not_cache_or_skip_physical_failure_audits():
    from scgsim.geometry._primitives._grid_cache import _ACTIVE, grid_work_cache_scope

    thin = GridOperand("thin", rectangle(0, 0, 0.25, 2), contact=False)
    with grid_work_cache_scope():
        errors = []
        for _ in range(2):
            with pytest.raises(
                ValueError, match="physical geometry collapsed"
            ) as caught:
                build_grid_arrangement((thin,), GeometryGridContext(1))
            errors.append(caught.value)
            cached = tuple(_ACTIVE.get().values.values())
            assert len(cached) == 1
            assert len(cached[0].subdivision.source_paths) == 4
        assert errors[0] is not errors[1]
        assert errors[0].__notes__ == errors[1].__notes__
    assert _ACTIVE.get() is None


@pytest.mark.parametrize(
    "polygons",
    (
        (
            Polygon(((0, 0), (11, 1), (0, -2))),
            Polygon(((5.5, 0.499), (6, -1), (5, -1))),
        ),
        (
            Polygon(((0, 0), (101, 1), (0, -2))),
            Polygon(((100, 1), (99, 2), (100, 2))),
            Polygon(((50, 1), (49, 2), (50, 2))),
        ),
    ),
)
def test_presnap_raw_context_recovers_public_native_network(polygons):
    # These raw source edges need native pre-snap nodes before pixel traversal.
    # The second case has integer-grid input, including a remote affected pixel.
    values, result = construct(polygons, 1)
    assert all(origins for _, origins in result.edge_origins)
    assert len(result.source_paths) == sum(len(p.exterior.coords) - 1 for p in polygons)
    for owner in range(len(polygons)):
        paths = [p for p in result.source_paths if p.operand_index == owner]
        assert all(
            a.nodes[-1] == b.nodes[0] for a, b in zip(paths, (*paths[1:], paths[0]))
        )
    parent = union_all([cell for cell, owners in result.cells if "0" in owners])
    assert parent.geom_type == "Polygon" and not parent.interiors
    if len(polygons) == 3:
        assert parent.area == 125
        assert (50, 1) in result.source_paths[0].nodes
        assert (100, 1) in result.source_paths[0].nodes
    else:
        assert parent.area == 8
    with preparation_resources(max_workers=2):
        _, parallel = construct(polygons, 1)
    assert parallel.source_paths == result.source_paths
    assert parallel.edge_origins == result.edge_origins
    assert [(p.wkb, labels) for p, labels in parallel.cells] == [
        (p.wkb, labels) for p, labels in result.cells
    ]


def test_directed_offline_nodes_use_geometric_secondary_order():
    from scgsim.geometry._primitives._tagged_grid import _raw_chain

    # Equal dominant coordinates must use direction-aware geometry, not tag ID.
    nodes = {(0, 0), (10, 1), (5, 0.51), (5, 0.49)}
    assert _raw_chain((0, 0), (10, 1), nodes) == ((0, 0), (5, 0.49), (5, 0.51), (10, 1))
    assert _raw_chain((10, 1), (0, 0), nodes) == ((10, 1), (5, 0.51), (5, 0.49), (0, 0))


def test_cache_reuses_raw_polygons_but_repeats_contact_audit():
    from scgsim.geometry._primitives._grid_cache import _ACTIVE, grid_work_cache_scope

    values = (
        GridOperand("A", rectangle(0, 0, 1.1, 1)),
        GridOperand("B", rectangle(1.2, 0, 2, 1)),
    )
    with grid_work_cache_scope():
        work = None
        errors = []
        for _ in range(2):
            with pytest.raises(ValueError, match="contact changed") as caught:
                build_grid_arrangement(values, GeometryGridContext(1))
            errors.append(caught.value)
            current = next(iter(_ACTIVE.get().values.values()))
            if work is not None:
                assert current is work
                assert all(
                    a[1] is b[1]
                    for a, b in zip(current.raw_polygons, work.raw_polygons)
                )
            work = current
        assert errors[0] is not errors[1]
