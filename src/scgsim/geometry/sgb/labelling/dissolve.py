"""Required edges and dissolution (design brief section 4, stage S5 part 2).

Owns deciding which edges of a labelled arrangement survive and rebuilding the
minimal arrangement without the rest:

* an edge is *required* iff the labels on its two sides differ, or one of its
  carriers belongs to a family whose edge policy is ``required``;
* ``refinement_only`` edges are not required but are kept aside as embedded
  curves of the face that contains them;
* every other edge dissolves: its two cells merge;
* afterwards degree-2 vertices whose two edges have the same carrier set are
  removed (this erases witness splits of one arc and other same-carrier chains),
  except where a refinement-only curve ends: that vertex stays on the boundary.

Does not own labelling (``prism_set``) or geometry changes: dissolution never
moves a vertex.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from scgsim.geometry.sgb.arrangement.dcel import Arrangement, build_arrangement
from scgsim.geometry.sgb.noding.snap_rounding import NodedEdge
from scgsim.geometry.sgb.plane import Point
from scgsim.geometry.sgb.records.carriers import MEMBER_ONLY, base_carrier_id

EdgeSides = tuple[
    Any, Any
]  # (left label, right label) along an edge's a -> b direction


@dataclass(frozen=True)
class FamilyPolicies:
    """Carrier ids grouped by their curve family's edge policy."""

    required: frozenset[str] = frozenset()
    refinement_only: frozenset[str] = frozenset()


@dataclass
class Dissolved:
    """A labelled arrangement after dissolution plus the maps back to the raw one."""

    arrangement: Arrangement
    labels: list[
        Any
    ]  # label of each face of ``arrangement``; index 0 is the unbounded face
    refinement: list[tuple[NodedEdge, int]]  # (edge, face index in ``arrangement``)
    face_map: list[int]  # old face index -> new face index
    stats: dict[str, int]
    removed: list[NodedEdge] = field(
        default_factory=list
    )  # raw edges that dissolved (membership-only carriers)


def dissolve(
    arr: Arrangement, labels: list[Any], policies: FamilyPolicies | None = None
) -> Dissolved:
    """Drop non-required edges, merge cells, remove same-carrier degree-2 vertices."""
    policies = policies or FamilyPolicies()
    kept, refinement_edges = _classify_edges(arr, labels, policies)
    # A refinement curve that ends on a boundary chain pins that vertex: the mesher embeds the
    # curve in the face and needs the boundary edge to end there.
    pinned = {p for ei in refinement_edges for p in (arr.edges[ei].a, arr.edges[ei].b)}
    chains = _merge_degree_two(arr, kept, labels, pinned)
    new_arr = build_arrangement([chain.edge for chain in chains])
    new_labels = _face_labels(new_arr, chains)
    face_map = _old_to_new_faces(arr, new_arr, chains)
    refinement = [
        (arr.edges[ei], face_map[arr.he_face[2 * ei]]) for ei in refinement_edges
    ]
    stats = {
        "edges_before": len(arr.edges),
        "edges_after": len(new_arr.edges),
        "faces_before": len(arr.faces) - 1,
        "faces_after": len(new_arr.faces) - 1,
        "refinement_edges": len(refinement),
    }
    kept_set = set(kept)
    removed = [e for ei, e in enumerate(arr.edges) if ei not in kept_set]
    return Dissolved(new_arr, new_labels, refinement, face_map, stats, removed)


def _classify_edges(
    arr: Arrangement, labels: list[Any], policies: FamilyPolicies
) -> tuple[list[int], list[int]]:
    """Indices of the required edges and of the refinement-only edges."""
    kept: list[int] = []
    refinement: list[int] = []
    for ei, edge in enumerate(arr.edges):
        left, right = labels[arr.he_face[2 * ei]], labels[arr.he_face[2 * ei + 1]]
        base = {
            base_carrier_id(c) for c in edge.carriers if not c.startswith(MEMBER_ONLY)
        }
        if left != right or (base & policies.required):
            kept.append(ei)
        elif base & policies.refinement_only:
            refinement.append(ei)
    return kept, refinement


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
    pinned: set[Point],
) -> list[_Chain]:
    """Remove vertices of degree 2 whose two edges carry the identical carrier set (never at a pinned vertex)."""
    live: dict[tuple[Point, Point], _Chain] = {}
    for ei in kept:
        e = arr.edges[ei]
        sides = (labels[arr.he_face[2 * ei]], labels[arr.he_face[2 * ei + 1]])
        # Witness pieces ``<arc>#k`` of one arc carry the same normalised id, so consecutive pieces can merge.
        real = [c for c in e.carriers if not c.startswith(MEMBER_ONLY)] or list(
            e.carriers
        )
        normalised = NodedEdge(e.a, e.b, frozenset(base_carrier_id(c) for c in real))
        live[(e.a, e.b)] = _Chain(normalised, sides, [(ei, 1)])
    incident: dict[Point, set[tuple[Point, Point]]] = defaultdict(set)
    for key in live:
        incident[key[0]].add(key)
        incident[key[1]].add(key)
    for vertex in sorted(incident):
        keys = incident[vertex]
        if len(keys) == 2 and vertex not in pinned:
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
