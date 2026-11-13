"""Required edges and dissolution (design brief section 4, stage S5 part 2).

Owns deciding which edges of a labelled arrangement survive and rebuilding the
minimal arrangement without the rest:

* an edge is *required* iff the labels on its two sides differ, or one of its
  carriers belongs to a cut family;
* every other edge dissolves: its two cells merge;
* afterwards degree-2 vertices whose two edges have the same carrier set are
  removed (this erases witness splits of one arc and other same-carrier chains);
* a merged chain whose chord would cross another edge or pass through a vertex
  is split again at one of its own vertices, so the rebuilt arrangement stays
  planar.

Does not own labelling (``prism_set``) or geometry changes: dissolution never
moves a vertex.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from scgsim.geometry.sgb.arrangement.dcel import Arrangement, build_arrangement
from scgsim.geometry.sgb.arrangement.noding.snap_rounding import (
    NodedEdge,
    crossing_pixels_kernel,
    hits_kernel,
)
from scgsim.geometry.sgb.exact.carriers import CarrierRole
from scgsim.geometry.sgb.exact.plane import Point, point_in_ring

EdgeSides = tuple[
    Any, Any
]  # (left label, right label) along an edge's a -> b direction


@dataclass
class Dissolved:
    """A labelled arrangement after dissolution plus the maps back to the raw one."""

    arrangement: Arrangement
    labels: list[
        Any
    ]  # label of each face of ``arrangement``; index 0 is the unbounded face
    face_map: list[int]  # old face index -> new face index
    stats: dict[str, int]
    removed: list[NodedEdge] = field(
        default_factory=list
    )  # raw edges that dissolved (membership-only carriers)


def dissolve(
    arr: Arrangement, labels: list[Any], family_curves: frozenset[str] = frozenset()
) -> Dissolved:
    """Drop non-required edges, merge cells, remove same-carrier degree-2 vertices.

    ``family_curves`` are the carrier ids of the cut families on the arrangement: their edges are required.
    """
    kept = _required_edges(arr, labels, family_curves)
    chains = _planar_chords(arr, _merge_degree_two(arr, kept, labels))
    new_arr = build_arrangement([chain.edge for chain in chains])
    new_labels = _face_labels(new_arr, chains)
    face_map = _old_to_new_faces(arr, new_arr, chains)
    stats = {
        "edges_before": len(arr.edges),
        "edges_after": len(new_arr.edges),
        "faces_before": len(arr.faces) - 1,
        "faces_after": len(new_arr.faces) - 1,
    }
    kept_set = set(kept)
    removed = [e for ei, e in enumerate(arr.edges) if ei not in kept_set]
    return Dissolved(new_arr, new_labels, face_map, stats, removed)


def _required_edges(
    arr: Arrangement, labels: list[Any], family_curves: frozenset[str]
) -> list[int]:
    """Indices of the required edges: different labels on the two sides, or a cut-family carrier."""
    kept: list[int] = []
    for ei, edge in enumerate(arr.edges):
        left, right = labels[arr.he_face[2 * ei]], labels[arr.he_face[2 * ei + 1]]
        curves = {
            c.curve_id for c in edge.carriers if c.role is not CarrierRole.MEMBERSHIP
        }
        if left != right or (curves & family_curves):
            kept.append(ei)
    return kept


def _face_labels(new_arr: Arrangement, chains: list[_Chain]) -> list[Any]:
    """Label of every face of the rebuilt arrangement, read from a boundary edge's recorded sides."""
    sides = {(c.edge.a, c.edge.b): c.sides for c in chains}
    labels: list[Any] = [None]
    for face_index in range(1, len(new_arr.faces)):
        outer = new_arr.faces[face_index].outer
        assert outer is not None
        half_edge = new_arr.cycles[outer].half_edges[0]
        edge = new_arr.edges[half_edge >> 1]
        left, right = sides[(edge.a, edge.b)]
        labels.append(right if half_edge & 1 else left)
    return labels


@dataclass
class _Chain:
    """A surviving edge: the merged geometry plus the original edges it replaced."""

    edge: NodedEdge
    sides: EdgeSides
    pieces: list[
        tuple[int, int]
    ]  # (original edge index, +1 if it runs the same way as edge.a -> edge.b)


def _merge_degree_two(
    arr: Arrangement,
    kept: list[int],
    labels: list[Any],
) -> list[_Chain]:
    """Remove vertices of degree 2 whose two edges carry the identical carrier set."""
    live: dict[tuple[Point, Point], _Chain] = {}
    for ei in kept:
        e = arr.edges[ei]
        sides = (labels[arr.he_face[2 * ei]], labels[arr.he_face[2 * ei + 1]])
        # The witness pieces of one arc carry the same whole carrier, so consecutive pieces can merge.
        real = [c for c in e.carriers if c.role is not CarrierRole.MEMBERSHIP] or list(
            e.carriers
        )
        normalised = NodedEdge(e.a, e.b, frozenset(c.whole() for c in real))
        live[(e.a, e.b)] = _Chain(normalised, sides, [(ei, 1)])
    incident: dict[Point, set[tuple[Point, Point]]] = defaultdict(set)
    for key in live:
        incident[key[0]].add(key)
        incident[key[1]].add(key)
    for vertex in sorted(incident):
        keys = incident[vertex]
        if len(keys) == 2:
            _try_join(live, incident, vertex, sorted(keys))
    return [chain for _, chain in sorted(live.items())]


def _try_join(
    live: dict[tuple[Point, Point], _Chain],
    incident: dict[Point, set[tuple[Point, Point]]],
    vertex: Point,
    keys: list[tuple[Point, Point]],
) -> None:
    """Join the two chains meeting at a degree-2 vertex when their carrier sets agree."""
    k1, k2 = keys
    c1, c2 = live[k1], live[k2]
    if c1.edge.carriers != c2.edge.carriers:
        return
    u = k1[0] if k1[1] == vertex else k1[1]
    w = k2[0] if k2[1] == vertex else k2[1]
    if u == w:
        return  # would collapse a 2-gon
    first = _oriented(c1, toward=vertex)  # runs u -> vertex
    second = _oriented(c2, toward=w)  # runs vertex -> w
    path = _Chain(
        NodedEdge(u, w, c1.edge.carriers), first.sides, first.pieces + second.pieces
    )
    if w < u:
        path = _reversed(path)
    new_key = (path.edge.a, path.edge.b)
    if new_key in live:
        return
    for key in (k1, k2):
        del live[key]
        for end in key:
            if end != vertex:
                incident[end].discard(key)
    live[new_key] = path
    incident[u].add(new_key)
    incident[w].add(new_key)
    del incident[vertex]


def _planar_chords(arr: Arrangement, chains: list[_Chain]) -> list[_Chain]:
    """Split merged chains until no chord crosses an edge, passes through a vertex or repeats an edge.

    The rebuilt arrangement draws every chain as one straight edge, its chord.  The pieces a chain
    replaced were noded, but the chord of a long arc run can leave them.  Where a port edge splits
    an arc and not the concentric band arc beside it, the split arc's new end lies between the
    band arc and its chord, and the band arc's chord crosses the split arc's chord and the port
    edge (Issue #108).  The test is the planarity check's own: exact crossings, rounded to their
    hot pixel, and the hot pixels an edge touches.  A chain is split only at one of its own
    vertices, a vertex of the noded arrangement, so no vertex moves; an arc chain keeps its one
    carrier and is realised piece by piece.
    """
    while True:
        keys = [(chain.edge.a, chain.edge.b) for chain in chains]
        merged = [len(chain.pieces) > 1 for chain in chains]
        if not any(merged):
            return chains
        ends = {p for key in keys for p in key}
        hot = ends | crossing_pixels_kernel(keys, merged)
        touched = hits_kernel(keys, merged, hot, set())
        repeated = {key for key, count in Counter(keys).items() if count > 1}
        blocked = [
            i
            for i, key in enumerate(keys)
            if merged[i] and (key in repeated or any(p not in key for p in touched[i]))
        ]
        if not blocked:
            return chains
        splits = _split_vertices(arr, chains, blocked, ends, touched)
        chains = sorted(
            (
                part
                for i, chain in enumerate(chains)
                for part in (
                    _split_chain(arr, chain, splits[i]) if i in splits else (chain,)
                )
            ),
            key=lambda chain: (chain.edge.a, chain.edge.b),
        )


def _split_vertices(
    arr: Arrangement,
    chains: list[_Chain],
    blocked: list[int],
    ends: set[Point],
    touched: list[list[Point]],
) -> dict[int, int]:
    """Where each blocked chain splits: the path index of its vertex nearest an end it encloses.

    An end of another chain between a chain's path and its chord, or on the chord, is what makes
    the chord leave the path, so the chain splits beside it; the other chain of a crossing, whose
    path encloses no foreign end, keeps its chord.  When no blocked chain encloses an end, each
    is split at its middle vertex, so every round makes progress.
    """
    splits: dict[int, int] = {}
    for i in blocked:
        path = _path(arr, chains[i])
        own = set(path)
        enclosed = [p for p in touched[i] if p in ends and p not in own] + [
            p for p in _ends_in_box(ends, path) if p not in own and _encloses(path, p)
        ]
        if enclosed:
            splits[i] = min(
                range(1, len(path) - 1),
                key=lambda k: min(_distance_sq(path[k], p) for p in enclosed),
            )
    return splits or {i: len(chains[i].pieces) // 2 for i in blocked}


def _path(arr: Arrangement, chain: _Chain) -> list[Point]:
    """The chain's vertices in order from ``edge.a`` to ``edge.b``."""
    path = [chain.edge.a]
    for index, sign in chain.pieces:
        piece = arr.edges[index]
        path.append(piece.b if sign > 0 else piece.a)
    return path


def _ends_in_box(ends: set[Point], path: list[Point]) -> list[Point]:
    """The chain ends inside the bounding box of a path."""
    xs, ys = [p[0] for p in path], [p[1] for p in path]
    x_lo, x_hi, y_lo, y_hi = min(xs), max(xs), min(ys), max(ys)
    return sorted(p for p in ends if x_lo <= p[0] <= x_hi and y_lo <= p[1] <= y_hi)


def _encloses(path: list[Point], point: Point) -> bool:
    """Whether ``point`` lies inside the ring the path closes with its chord (exact, off the ring)."""
    return point_in_ring(path, (point[0], point[1], 1))


def _distance_sq(p: Point, q: Point) -> int:
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2


def _split_chain(arr: Arrangement, chain: _Chain, at: int) -> tuple[_Chain, _Chain]:
    """The chain as two chains meeting at its ``at``-th path vertex, each with ``a < b``."""
    index, sign = chain.pieces[at - 1]
    piece = arr.edges[index]
    vertex = piece.b if sign > 0 else piece.a
    halves = (
        _Chain(
            NodedEdge(chain.edge.a, vertex, chain.edge.carriers),
            chain.sides,
            chain.pieces[:at],
        ),
        _Chain(
            NodedEdge(vertex, chain.edge.b, chain.edge.carriers),
            chain.sides,
            chain.pieces[at:],
        ),
    )
    first, second = (
        half if half.edge.a < half.edge.b else _reversed(half) for half in halves
    )
    return first, second


def _oriented(chain: _Chain, toward: Point) -> _Chain:
    """The chain re-expressed so it runs from its other end toward ``toward``."""
    return chain if chain.edge.b == toward else _reversed(chain)


def _reversed(chain: _Chain) -> _Chain:
    """The chain travelled the other way; ``a > b`` afterwards, so it is only used as an oriented path."""
    e = chain.edge
    return _Chain(
        NodedEdge(e.b, e.a, e.carriers),
        (chain.sides[1], chain.sides[0]),
        [(ei, -s) for ei, s in reversed(chain.pieces)],
    )


def _old_to_new_faces(
    arr: Arrangement, new_arr: Arrangement, chains: list[_Chain]
) -> list[int]:
    """Map each old face to the new face that absorbed it (union-find across removed edges)."""
    parent = list(range(len(arr.faces)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    piece_of = {ei: (ci, s) for ci, c in enumerate(chains) for ei, s in c.pieces}
    for ei in range(len(arr.edges)):
        if ei in piece_of:
            continue
        a, b = find(arr.he_face[2 * ei]), find(arr.he_face[2 * ei + 1])
        if a != b:
            parent[max(a, b)] = min(a, b)
    new_index = {(e.a, e.b): i for i, e in enumerate(new_arr.edges)}
    set_to_new: dict[int, int] = {}
    for ei, (ci, sign) in piece_of.items():
        ni = new_index[(chains[ci].edge.a, chains[ci].edge.b)]
        for side in (0, 1):
            old_face = arr.he_face[2 * ei + side]
            new_face = new_arr.he_face[2 * ni + (side if sign > 0 else 1 - side)]
            set_to_new.setdefault(find(old_face), new_face)
    return [set_to_new.get(find(f), 0) for f in range(len(arr.faces))]
