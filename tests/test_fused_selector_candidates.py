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


def outcome(values):
    try:
        return ("result", ops._fused_gdstk_components(values))
    except ValueError as error:
        return (
            "error",
            type(error),
            str(error),
            tuple(getattr(error, "__notes__", ())),
        )


@pytest.mark.parametrize(
    "kind",
    ("distant", "point", "edge", "crossing", "hole", "snap_contact", "degenerate"),
)
@pytest.mark.parametrize("frame", (None, (0.8, -0.6, 0.6, 0.8, 7.0, -2.0)))
@pytest.mark.parametrize("intent", (False, True))
def test_candidates_equal_original_ordered_all_pairs(monkeypatch, kind, frame, intent):
    with (
        geometry_grid_scope(GeometryGridContext(0.1)),
        polygon_world_frame(frame) if frame else nullcontext(),
    ):
        token = _SOURCE_INTENT.set(intent)
        try:
            actual = outcome(polygons(kind))
            monkeypatch.setattr(ops, "_fused_boolean_candidates", lambda values: None)
            assert actual == outcome(polygons(kind))
        finally:
            _SOURCE_INTENT.reset(token)


def test_distant_world_inputs_avoid_pair_booleans_but_keep_raw_edge_calls(monkeypatch):
    calls, edges = [], []
    original = ops.polygon_boolean
    shared = ops._rings_share_edge

    def observed(left, right, operation, **kwargs):
        calls.append(operation)
        return original(left, right, operation, **kwargs)

    def observed_edge(left, right):
        edges.append(1)
        return shared(left, right)

    monkeypatch.setattr(ops, "polygon_boolean", observed)
    monkeypatch.setattr(ops, "_rings_share_edge", observed_edge)
    values = [gdstk.rectangle((10 * i, 0), (10 * i + 1, 1)) for i in range(8)]
    with geometry_grid_scope(GeometryGridContext(0.1)):
        assert len(ops._fused_gdstk_components(values)) == 8
    assert calls == ["or"] * 8
    assert len(edges) == 8 * 7 // 2


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
