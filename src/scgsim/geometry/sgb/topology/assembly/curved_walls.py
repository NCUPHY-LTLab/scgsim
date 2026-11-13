"""Curved walls split at every vertex on their arc (design brief section 14, after review R3).

A wall on an arc is a patch of one cylinder only if its bottom and top boundary are each
*one* arc on the same circle.  Where a vertex of another curve lies on the arc (a port edge
meeting a round pad, a post standing on a curved edge) one level has an extra vertex the
other level lacks, or both carry a split but through snapped points that give each piece
its own realised circle.  The fix is a vertical edge at every such vertex: the arc edges of
*both* levels are split at the same lattice points, so each wall piece is bounded by one
bottom arc and one top arc through the same end points with the same carrier radius -- an
exact cylinder.

Splitting changes level edges, so every other face that uses one of them (the horizontal
faces on both sides, the wall run stacked above or below) sees the pieces too; the plan is
computed once for the whole model and iterated until no further vertex is forced.  Straight
walls need none of this: a planar polygon may have any number of boundary edges.  Circles
approximated by polygons never produce cylinder walls either.
"""

from __future__ import annotations

from scgsim.geometry.sgb.topology.assembly.walls import WallRun
from scgsim.geometry.sgb.arrangement.derivation.layer import Layer
from scgsim.geometry.sgb.exact.plane import Point

EdgeAt = tuple[int, int]  # (level index, edge index in that level's arrangement)


def is_curved(run: WallRun, level_layers: list[Layer]) -> bool:
    """Whether the wall of a run stands on arcs."""
    layer = level_layers[run.k0]
    return any(layer.edge_curve[h >> 1] is not None for h in run.bottom)


def plan_curved_splits(
    level_layers: list[Layer], runs: list[WallRun]
) -> dict[EdgeAt, tuple[Point, ...]]:
    """Interior lattice points to add to level arc edges, in order along each edge from ``a`` to ``b``."""
    curved = [run for run in runs if is_curved(run, level_layers)]
    extra: dict[EdgeAt, set[Point]] = {}
    changed = True
    while changed:
        changed = False
        for run in curved:
            chains = _chains(run)
            points = {
                p
                for k, edge_indices in chains
                for p in _chain_points(level_layers, extra, k, edge_indices)
            }
            for k, edge_indices in chains:
                for ei in edge_indices:
                    edge = level_layers[k].arr.edges[ei]
                    wanted = {p for p in points if _strictly_inside(edge.a, edge.b, p)}
                    if not wanted <= extra.get((k, ei), set()):
                        extra.setdefault((k, ei), set()).update(wanted)
                        changed = True
    return {
        at: tuple(
            sorted(
                points,
                key=lambda p: _along(
                    level_layers[at[0]].arr.edges[at[1]].a,
                    level_layers[at[0]].arr.edges[at[1]].b,
                    p,
                ),
            )
        )
        for at, points in extra.items()
        if points
    }


def _chains(run: WallRun) -> list[tuple[int, list[int]]]:
    return [
        (run.k0, [h >> 1 for h in run.bottom]),
        (run.k1 + 1, [h >> 1 for h in run.top]),
    ]


def _chain_points(
    level_layers: list[Layer],
    extra: dict[EdgeAt, set[Point]],
    level: int,
    edge_indices: list[int],
) -> set[Point]:
    """Every vertex of a chain, including the vertices already planned on its edges."""
    points: set[Point] = set()
    for ei in edge_indices:
        edge = level_layers[level].arr.edges[ei]
        points.update((edge.a, edge.b))
        points.update(extra.get((level, ei), ()))
    return points


def _along(a: Point, b: Point, p: Point) -> int:
    """Position of ``p`` along the chord a -> b (scaled dot product, exact)."""
    return (p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])


def _strictly_inside(a: Point, b: Point, p: Point) -> bool:
    """Whether the projection of ``p`` falls strictly between ``a`` and ``b`` on the chord.

    Arc edges span at most about a quarter turn, so every point of the same arc projects
    monotonically onto the chord; a vertex beyond an end projects outside it.
    """
    t = _along(a, b, p)
    return 0 < t < _along(a, b, b)
