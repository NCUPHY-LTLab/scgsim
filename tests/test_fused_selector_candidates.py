"""Candidate filtering preserves the exact existing fused selector predicates."""

from contextlib import nullcontext

import gdstk
import pytest

from scgsim.geometry._primitives.grid import (
    _SOURCE_INTENT,
    geometry_grid_scope,
    polygon_world_frame,
)
from scgsim.geometry.models.grid import GeometryGridContext
from scgsim.geometry.source import _polygon_ops as ops


def polygons(kind):
    if kind == "mixed":
        return [
            gdstk.rectangle((x, 0), (x + 1, 1))
            for x in (10, 2, 0, 20, 1, 11, 30, 40)
        ]
    if kind == "degenerate":
        return [
            gdstk.rectangle((0, 0), (1, 1)),
            gdstk.Polygon(((5, 0), (6, 0), (7, 0))),
        ]
    if kind == "hole":
        return [
            *gdstk.boolean(
                gdstk.rectangle((0, 0), (4, 4)), gdstk.rectangle((1, 1), (3, 3)), "not"
            ),
            gdstk.rectangle((5, 0), (6, 1)),
        ]
    rings = {
        "distant": [((0, 0), (1, 0), (1, 1), (0, 1)), ((5, 0), (6, 0), (6, 1), (5, 1))],
        "point": [((0, 0), (1, 0), (1, 1), (0, 1)), ((1, 1), (2, 1), (2, 2), (1, 2))],
        "edge": [((0, 0), (1, 0), (1, 1), (0, 1)), ((1, 0), (2, 0), (2, 1), (1, 1))],
        "crossing": [((0, 0), (3, 1), (1, 3)), ((1, 0), (2, 0), (2, 3), (1, 3))],
        "snap_contact": [
            ((0, 0), (0.96, 0), (0.96, 1), (0, 1)),
            ((1.04, 0), (2, 0), (2, 1), (1.04, 1)),
        ],
    }
    return [gdstk.Polygon(ring) for ring in rings[kind]]


def original_all_pairs(values):
    """Original ordered pair loop, with the unchanged component lowering."""
    parents = list(range(len(values)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for index, polygon in enumerate(values):
        for other_index, other in enumerate(values[index + 1 :], index + 1):
            overlap = ops.polygon_boolean(
                (polygon,), (other,), "and", precision=ops.current_grid_size_um()
            )
            if overlap or ops._rings_share_edge(
                ops._ring_from_gdstk_polygon(polygon),
                ops._ring_from_gdstk_polygon(other),
            ):
                left, right = find(index), find(other_index)
                parents[max(left, right)] = min(left, right)
    members = {}
    for index in range(len(values)):
        members.setdefault(find(index), []).append(index)
    result = []
    for indexes in members.values():
        regions = ops.polygon_boolean(
            tuple(values[index] for index in indexes),
            (),
            "or",
            precision=ops.current_grid_size_um(),
        )
        if ops.current_grid_context() is None:
            regions = gdstk.offset(
                regions or (), 0.0, join="miter", precision=ops.current_grid_size_um()
            )
        if not regions or len(regions) != 1:
            raise ValueError("fused selector component must lower to one polygon")
        outer, holes = ops._split_gdstk_cutline_loop(
            ops._ring_from_gdstk_polygon(regions[0])
        )
        result.append(
            (
                ops.LayoutPolygonSpec(
                    polygon_id="", layer="", exterior=outer, holes=holes
                ),
                tuple(indexes),
                sum(float(values[index].area()) for index in indexes),
            )
        )
    return tuple(
        sorted(
            result,
            key=lambda record: (
                min(point[0] for point in record[0].exterior),
                min(point[1] for point in record[0].exterior),
                record[2],
            ),
        )
    )


def outcome(values, implementation=ops._fused_gdstk_components):
    try:
        return ("result", implementation(values))
    except ValueError as error:
        return (
            "error",
            type(error),
            str(error),
            tuple(getattr(error, "__notes__", ())),
        )


@pytest.mark.parametrize(
    "kind",
    ("distant", "point", "edge", "crossing", "hole", "snap_contact", "degenerate", "mixed"),
)
@pytest.mark.parametrize("frame", (None, (0.8, -0.6, 0.6, 0.8, 7.0, -2.0)))
@pytest.mark.parametrize("intent", (False, True))
def test_candidates_equal_original_ordered_all_pairs(kind, frame, intent):
    with (
        geometry_grid_scope(GeometryGridContext(0.1)),
        polygon_world_frame(frame) if frame else nullcontext(),
    ):
        token = _SOURCE_INTENT.set(intent)
        try:
            actual = outcome(polygons(kind))
            assert actual == outcome(polygons(kind), original_all_pairs)
        finally:
            _SOURCE_INTENT.reset(token)


def test_distant_world_inputs_avoid_all_pair_predicates(monkeypatch):
    calls, edges = [], []
    original = ops.polygon_boolean
    shared = ops._edges_share_edge

    def observed(left, right, operation, **kwargs):
        calls.append(operation)
        return original(left, right, operation, **kwargs)

    def observed_edge(left, right):
        edges.append(1)
        return shared(left, right)

    monkeypatch.setattr(ops, "polygon_boolean", observed)
    monkeypatch.setattr(ops, "_edges_share_edge", observed_edge)
    values = [gdstk.rectangle((10 * i, 0), (10 * i + 1, 1)) for i in range(8)]
    with geometry_grid_scope(GeometryGridContext(0.1)):
        assert len(ops._fused_gdstk_components(values)) == 8
    assert calls == ["or"] * 8
    assert edges == []


def test_legacy_unbound_selector_keeps_all_pair_calls(monkeypatch):
    calls = []
    original = ops.polygon_boolean

    def observed(left, right, operation, **kwargs):
        calls.append(operation)
        return original(left, right, operation, **kwargs)

    monkeypatch.setattr(ops, "polygon_boolean", observed)
    with geometry_grid_scope(None):
        assert len(ops._fused_gdstk_components(polygons("distant"))) == 2
    assert calls == ["and", "or", "or"]


@pytest.mark.parametrize("length", (1.0, 1e6, 1e9))
@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("intent", (False, True))
@pytest.mark.parametrize("frame", (None, (0.8, -0.6, 0.6, 0.8, 7.0, -2.0)))
def test_long_near_parallel_raw_contacts_are_candidates(length, reverse, intent, frame):
    # The second edge starts just outside the first segment. Its projected
    # overlap is far from its start and may be much farther than 1e-9 away.
    left = gdstk.Polygon(((0, 0), (length, 0), (length, -1), (0, -1)))
    right = gdstk.Polygon(
        ((-length, 0.5e-9), (length, length * 1e-9 + 0.5e-9),
         (length, length * 1e-9 + 1), (-length, 1))
    )
    values = [right, left] if reverse else [left, right]
    with (
        geometry_grid_scope(GeometryGridContext(0.1)),
        polygon_world_frame(frame) if frame else nullcontext(),
    ):
        token = _SOURCE_INTENT.set(intent)
        try:
            raw = ops._fused_raw_candidates(values)
            assert raw is not None
            assert 1 in raw[1][0]
            if not reverse:
                assert ops._edges_share_edge(raw[0][0], raw[0][1])
                if intent and frame is None:
                    assert 1 not in ops._fused_boolean_candidates(values)[0]
            assert outcome(values) == outcome(values, original_all_pairs)
        finally:
            _SOURCE_INTENT.reset(token)


def test_invalid_ring_keeps_original_boolean_error_order(monkeypatch):
    values = [gdstk.rectangle((0, 0), (1, 1)), gdstk.Polygon(((10, 0), (11, 0)))]
    calls = []

    def fail_boolean(*args, **kwargs):
        calls.append(args[2])
        raise ValueError("boolean first")

    monkeypatch.setattr(ops, "polygon_boolean", fail_boolean)
    with geometry_grid_scope(GeometryGridContext(0.1)):
        assert outcome(values) == outcome(values, original_all_pairs)
    assert calls == ["and", "and"]


def test_cached_edges_are_built_once_for_each_valid_ring(monkeypatch):
    original = ops._ring_edges
    calls = []

    def observed(ring):
        calls.append(ring)
        return original(ring)

    monkeypatch.setattr(ops, "_ring_edges", observed)
    values = [gdstk.rectangle((i, 0), (i + 1, 1)) for i in range(8)]
    with geometry_grid_scope(GeometryGridContext(0.1)):
        assert len(ops._fused_gdstk_components(values)) == 1
    assert len(calls) == len(values)
