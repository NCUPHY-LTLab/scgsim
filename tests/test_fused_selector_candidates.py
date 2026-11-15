"""Candidate filtering preserves the exact existing fused selector predicates."""

import gdstk
import pytest

from scgsim.geometry.grid import polygon_ops as ops
from scgsim.geometry.grid.scope import GridScope
from scgsim.geometry.models.grid import GeometryGridContext


def polygons(kind):
    """Build native polygon fixtures for each fused-selector relationship."""
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


def outcome(scope, values):
    """Capture a real selector result or its complete raised error."""
    try:
        return ("result", ops.fused_gdstk_components(scope, values))
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
    """Compare candidate filtering with the original ordered all-pairs result."""
    scope = GridScope(GeometryGridContext(0.1))
    if frame:
        scope = scope.placed(frame)
    if intent:
        scope = scope.as_reference()
    actual = outcome(scope, polygons(kind))
    monkeypatch.setattr(ops, "_fused_boolean_candidates", lambda scope, values: None)
    assert actual == outcome(scope, polygons(kind))


def test_distant_world_inputs_avoid_pair_booleans_but_keep_raw_edge_calls(monkeypatch):
    """Check distant inputs skip pair booleans while retaining raw edge checks."""
    calls, edges = [], []
    original = ops.polygon_boolean
    # White-box observation verifies the candidate path still performs raw edge checks.
    shared = ops._rings_share_edge  # noqa: SLF001 - observe private edge predicate calls

    def observed(scope, left, right, operation, **kwargs):
        calls.append(operation)
        return original(scope, left, right, operation, **kwargs)

    def observed_edge(left, right):
        edges.append(1)
        return shared(left, right)

    monkeypatch.setattr(ops, "polygon_boolean", observed)
    monkeypatch.setattr(ops, "_rings_share_edge", observed_edge)
    values = [gdstk.rectangle((10 * i, 0), (10 * i + 1, 1)) for i in range(8)]
    scope = GridScope(GeometryGridContext(0.1))
    assert len(ops.fused_gdstk_components(scope, values)) == 8
    assert calls == ["or"] * 8
    assert len(edges) == 8 * 7 // 2
