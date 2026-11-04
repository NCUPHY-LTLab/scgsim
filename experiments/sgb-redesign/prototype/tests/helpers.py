"""Shared test helpers: random segment sets and exact brute-force geometry checks."""

from __future__ import annotations

import math
import random

from sgbproto.geometry import segment_point_distance_sq, segments_properly_cross
from sgbproto.snap_rounding import HotGrid, NodedEdge, Segment


def random_segments(seed: int, n: int, extent: int) -> list[Segment]:
    rng = random.Random(seed)
    return [
        Segment(
            (rng.randint(0, extent), rng.randint(0, extent)),
            (rng.randint(0, extent), rng.randint(0, extent)),
            frozenset([f"s{i}"]),
        )
        for i in range(n)
    ]


def crossing_pairs(edges: list[NodedEdge]) -> int:
    """Brute-force count of edge pairs crossing in both interiors (exact)."""
    return sum(
        1
        for i in range(len(edges))
        for j in range(i + 1, len(edges))
        if segments_properly_cross(edges[i].a, edges[i].b, edges[j].a, edges[j].b)
    )


def pass_through_vertices(edges: list[NodedEdge]) -> int:
    """Vertices lying inside the hot pixel of an edge that does not end there."""
    vertices = {p for e in edges for p in (e.a, e.b)}
    grid = HotGrid(vertices)
    return sum(1 for e in edges for c in grid.touching(e.a, e.b) if c not in (e.a, e.b))


def max_deviation(edges: list[NodedEdge], segments: list[Segment]) -> float:
    """Largest distance (grid units) from an output edge end to the nearest of its carriers."""
    by_id = {next(iter(s.carriers)): s for s in segments}
    worst = 0.0
    for e in edges:
        best = min(
            max(
                segment_point_distance_sq(by_id[c].a, by_id[c].b, e.a),
                segment_point_distance_sq(by_id[c].a, by_id[c].b, e.b),
            )
            for c in e.carriers
        )
        worst = max(worst, math.sqrt(float(best)))
    return worst
