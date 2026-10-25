"""Exact output and operation-count regressions; no wall-clock acceptance."""

from dataclasses import replace

import pytest
from shapely import Polygon, coverage_union_all, union_all

from scgsim.geometry._primitives import grid
from scgsim.geometry.models.grid import GeometryGridContext


def box(x0, y0, x1, y1):
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


CASES = [
    (box(0, 0, 1, 1), box(4, 0, 5, 1)),
    (box(0, 0, 3, 1), box(1, -1, 2, 2)),
    (box(0, 0, 1, 1), box(1, 0, 2, 1)),
    (box(0, 0, 1, 1), box(1, 1, 2, 2)),
    (((0, 0), (3, 1), (1, 3)), box(1, 0, 2, 3)),
    (box(0, 0, 0.96, 1), box(1.04, 0, 2, 1)),
]


@pytest.mark.parametrize("rings", CASES)
def test_existing_incremental_index_matches_exhaustive_candidates(monkeypatch, rings):
    import shapely

    values = tuple(
        grid.GridOperand(str(i), ring, physical=False, contact=False)
        for i, ring in enumerate(rings)
    )
    shapes = {v.operand_id: Polygon(v.exterior) for v in values}
    indexed = grid._subdivide_grid(values, shapes, 0.1)

    class ExhaustiveCandidates:
        """Reference broad phase: every real cell remains an exact candidate."""

        def __init__(self, geometries):
            self.count = len(geometries)

        def query(self, geometry):
            return tuple(range(self.count))

    monkeypatch.setattr(shapely, "STRtree", ExhaustiveCandidates)
    exhaustive = grid._subdivide_grid(values, shapes, 0.1)
    assert [(g.wkb, labels) for g, labels in indexed.cells] == [
        (g.wkb, labels) for g, labels in exhaustive.cells
    ]
    assert [(r.geometry.wkb, r.operand_ids) for r in indexed.residues] == [
        (r.geometry.wkb, r.operand_ids) for r in exhaustive.residues
    ]


def test_separated_incremental_inputs_do_no_pair_boolean_work(monkeypatch):
    values = tuple(
        grid.GridOperand(
            str(i), box(10 * i, 0, 10 * i + 1, 1), physical=False, contact=False
        )
        for i in range(12)
    )
    calls = []
    original = grid._overlay_call

    def observed(operation, left, right, precision, operand_index, cell_index):
        calls.append((operation, cell_index))
        return original(operation, left, right, precision, operand_index, cell_index)

    monkeypatch.setattr(grid, "_overlay_call", observed)
    result = grid._subdivide_grid(
        values, {v.operand_id: Polygon(v.exterior) for v in values}, 0.1
    )
    assert len(result.cells) == len(values)
    assert not any(cell is not None for _, cell in calls)


def test_shared_vertex_conversion_once_and_exact_diagonal_incidence(monkeypatch):
    vertices = (
        (0.0, 0.0),
        (1.0, 1.0),
        (2.0, 2.0),
        (3.0, 3.0),
        (0.0, 3.0),
        *[(10.0 + i, 20.0 + i) for i in range(30)],
    )
    calls = []
    original = grid._region_key

    def observed(point, precision):
        calls.append(tuple(point))
        return original(point, precision)

    monkeypatch.setattr(grid, "_region_key", observed)
    index = grid._region_vertex_index(vertices, 0.1)
    assert len(calls) == len(vertices)
    shape = Polygon(((0.0, 0.0), (3.0, 3.0), (0.0, 3.0)))
    expected = (((0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (0.0, 3.0)), ())
    assert grid._regions_indexed(shape, index) == (expected,)
    assert grid._regions_indexed(shape, index) == (expected,)
    assert len(calls) == len(vertices)


@pytest.mark.parametrize("apply_grid", (False, True))
def test_arrangement_region_reuse_preserves_holed_memberships(monkeypatch, apply_grid):
    outer = grid.GridOperand(
        "outer", box(0, 0, 6, 6), holes=(box(2, 2, 4, 4),), contact=False
    )
    cross = grid.GridOperand("cross", box(3, 0, 5, 6), physical=False, contact=False)
    arrangement = grid.build_grid_arrangement(
        (outer, cross), GeometryGridContext(0.1), apply_grid=apply_grid
    )
    selected_ids = tuple(
        f.face_id for f in arrangement.faces if "outer" in f.operand_ids
    )
    selected = [
        Polygon(f.exterior, f.holes)
        for f in arrangement.faces
        if f.face_id in selected_ids
    ]
    merged = coverage_union_all(selected) if apply_grid else union_all(selected)
    expected = grid._regions(merged, arrangement.vertices, 0.1 if apply_grid else None)
    calls = []
    original = grid._regions_indexed

    def observed(geometry, index):
        calls.append(1)
        return original(geometry, index)

    monkeypatch.setattr(grid, "_regions_indexed", observed)
    first = arrangement.operand_regions("outer")
    assert first == expected
    assert arrangement.face_regions(selected_ids) is first
    assert arrangement.operand_regions("outer") is first
    assert len(calls) == 1
    assert arrangement.operand_regions("cross") != first
    assert len(calls) == 2
    assert arrangement.operand_regions("unknown") == ()


def test_strict_contact_and_identity_failures_remain():
    values = (
        grid.GridOperand("A", box(0, 0, 0.96, 1)),
        grid.GridOperand("B", box(1.04, 0, 2, 1)),
    )
    with pytest.raises(ValueError, match="physical contact changed"):
        grid.build_grid_arrangement(values, GeometryGridContext(0.1))
    with pytest.raises(ValueError, match="identities are duplicated"):
        grid.build_grid_arrangement(
            (values[0], replace(values[1], operand_id="A")), GeometryGridContext(0.1)
        )
