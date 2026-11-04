"""Required edges and dissolution (brief section 4, stage S5 part 2).

Owns deciding which edges of a labelled arrangement survive and rebuilding the
minimal arrangement without the rest:

* an edge is *required* iff the labels on its two sides differ, or one of its
  carriers belongs to a family whose edge policy is ``required``;
* ``refinement_only`` edges are not required but are kept aside as embedded
  curves of the face that contains them;
* every other edge dissolves: its two cells merge;
* afterwards degree-2 vertices whose two edges have the same carrier set are
  removed (this erases witness splits of one arc and other same-carrier chains),
  except where a refinement-only curve ends: that vertex stays on the boundary chain.

Does not own labelling (``labels``) or geometry changes: dissolution never
moves a vertex.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from sgbproto.arrangement import Arrangement, build_arrangement
from sgbproto.geometry import Point
from sgbproto.records import MEMBER_ONLY, base_carrier_id
from sgbproto.snap_rounding import NodedEdge


@dataclass(frozen=True)
class FamilyPolicies:
    """Carrier ids grouped by their curve family's edge policy."""

    required: frozenset[str] = frozenset()
    refinement_only: frozenset[str] = frozenset()


@dataclass
class Dissolved:
    arrangement: Arrangement
    labels: list  # label of each face of ``arrangement``; index 0 is the unbounded face
    refinement: list[tuple[NodedEdge, int]]  # (edge, face index in ``arrangement``)
    face_map: list[int]  # old face index -> new face index
    stats: dict
    removed: list[NodedEdge] = field(default_factory=list)  # raw edges that dissolved (for membership-only carriers)


def dissolve(arr: Arrangement, labels: list, policies: FamilyPolicies = FamilyPolicies()) -> Dissolved:
    """Drop non-required edges, merge cells, remove same-carrier degree-2 vertices."""
    kept: list[int] = []
    refinement_idx: list[int] = []
    for ei, e in enumerate(arr.edges):
        left, right = labels[arr.he_face[2 * ei]], labels[arr.he_face[2 * ei + 1]]
        base = {base_carrier_id(c) for c in e.carriers if not c.startswith(MEMBER_ONLY)}
        if left != right or (base & policies.required):
            kept.append(ei)
        elif base & policies.refinement_only:
            refinement_idx.append(ei)

    # a refinement curve that ends on a boundary chain pins that vertex: the mesher embeds the curve in the face and
    # needs the boundary edge to end there, otherwise the curve's end vertex sits unconnected on top of the edge
    pinned = {p for ei in refinement_idx for p in (arr.edges[ei].a, arr.edges[ei].b)}
    chains = _merge_degree_two(arr, kept, labels, pinned)
    new_arr = build_arrangement([c.edge for c in chains])
    sides = {(c.edge.a, c.edge.b): c.sides for c in chains}

    new_labels: list = [None]
    for fi in range(1, len(new_arr.faces)):
        h = new_arr.cycles[new_arr.faces[fi].outer].half_edges[0]
        e = new_arr.edges[h >> 1]
        left, right = sides[(e.a, e.b)]
        new_labels.append(right if h & 1 else left)

    face_map = _old_to_new_faces(arr, new_arr, chains)
    refinement = [(arr.edges[ei], face_map[arr.he_face[2 * ei]]) for ei in refinement_idx]
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


@dataclass
class _Chain:
    """A surviving edge: the merged geometry plus the original edges it replaced."""

    edge: NodedEdge
    sides: tuple  # (left label, right label) along edge.a -> edge.b
    pieces: list[tuple[int, int]]  # (original edge index, +1 if it runs the same way as edge.a -> edge.b)


def _merge_degree_two(arr: Arrangement, kept: list[int], labels: list, pinned: set[Point] = frozenset()) -> list[_Chain]:
    """Remove vertices of degree 2 whose two edges carry the identical carrier set (never at a ``pinned`` vertex)."""
    live: dict[tuple[Point, Point], _Chain] = {}
    for ei in kept:
        e = arr.edges[ei]
        sides = (labels[arr.he_face[2 * ei]], labels[arr.he_face[2 * ei + 1]])
        # Witness pieces ``<arc>#k`` of one arc carry the same normalised id, so consecutive pieces can merge.
        real = [c for c in e.carriers if not c.startswith(MEMBER_ONLY)] or list(e.carriers)
        normalised = NodedEdge(e.a, e.b, frozenset(base_carrier_id(c) for c in real))
        live[(e.a, e.b)] = _Chain(normalised, sides, [(ei, 1)])
    incident: dict[Point, set[tuple[Point, Point]]] = defaultdict(set)
    for key in live:
        incident[key[0]].add(key)
        incident[key[1]].add(key)
    for v in sorted(incident):
        keys = incident[v]
        if len(keys) != 2 or v in pinned:
            continue
        k1, k2 = sorted(keys)
        c1, c2 = live[k1], live[k2]
        if c1.edge.carriers != c2.edge.carriers:
            continue
        u = k1[0] if k1[1] == v else k1[1]
        w = k2[0] if k2[1] == v else k2[1]
        if u == w:
            continue  # would collapse a 2-gon
        first = _oriented(c1, toward=v)  # runs u -> v
        second = _oriented(c2, toward=w)  # runs v -> w
        path = _Chain(NodedEdge(u, w, c1.edge.carriers), first.sides, first.pieces + second.pieces)
        if w < u:
            path = _reversed(path)
        new_key = (path.edge.a, path.edge.b)
        if new_key in live:
            continue
        for k in (k1, k2):
            del live[k]
            for end in k:
                if end != v:
                    incident[end].discard(k)
        live[new_key] = path
        incident[u].add(new_key)
        incident[w].add(new_key)
        del incident[v]
    return [c for _, c in sorted(live.items())]


def _oriented(chain: _Chain, toward: Point) -> _Chain:
    """The chain re-expressed so it runs from its other end toward ``toward``."""
    return chain if chain.edge.b == toward else _reversed(chain)


def _reversed(chain: _Chain) -> _Chain:
    e = chain.edge
    return _Chain(
        NodedEdge(e.b, e.a, e.carriers),  # note: a > b now; only used as an oriented path
        (chain.sides[1], chain.sides[0]),
        [(ei, -s) for ei, s in reversed(chain.pieces)],
    )


def _old_to_new_faces(arr: Arrangement, new_arr: Arrangement, chains: list[_Chain]) -> list[int]:
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
