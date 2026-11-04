"""Tying a level arrangement to the slab arrangements it refines (brief section 2).

Owns two derived facts needed by labelling and assembly:

* **chains** -- the path of level edges that covers each slab edge. A level
  arrangement is built from slab edges tagged ``~<edge id>``, so the level
  edges carrying a tag form a simple path from one slab-edge endpoint to the
  other (the bottom or top boundary of that edge's wall).
* **links** -- which slab cell each level cell lies in. Level cells are joined
  across every edge that is *not* part of a slab-edge chain (those edges do not
  separate slab cells); each resulting group borders at least one chain, whose
  left/right sides name the slab cells.

Pure functions of two arrangements; no geometry is created.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from sgbproto.arrangement import Arrangement
from sgbproto.errors import SgbError
from sgbproto.records import base_carrier_id

TAG = "~"


@dataclass(frozen=True)
class SlabEdges:
    """What a slab layer exposes for linking: its arrangement and edge ids."""

    arr: Arrangement
    edge_ids: list[str]


def trace_chains(level_arr: Arrangement, slab: SlabEdges) -> dict[int, list[int]]:
    """For every slab edge index, the level half-edges from its ``a`` end to its ``b`` end."""
    tag_to_slab_edge = {TAG + eid: i for i, eid in enumerate(slab.edge_ids)}
    members: dict[int, list[int]] = defaultdict(list)
    for ei, e in enumerate(level_arr.edges):
        for c in e.carriers:
            si = tag_to_slab_edge.get(base_carrier_id(c))
            if si is not None:
                members[si].append(ei)
    out: dict[int, list[int]] = {}
    for si, sedge in enumerate(slab.arr.edges):
        eis = members.get(si)
        if not eis:
            raise SgbError(f"slab edge {slab.edge_ids[si]} has no chain in the level arrangement")
        at: dict[tuple[int, int], list[int]] = defaultdict(list)
        for ei in eis:
            e = level_arr.edges[ei]
            at[e.a].append(ei)
            at[e.b].append(ei)
        cur, path, used = sedge.a, [], set()
        while cur != sedge.b:
            options = [ei for ei in at[cur] if ei not in used]
            if len(options) != 1:
                raise SgbError(f"slab edge {slab.edge_ids[si]}: ambiguous or broken chain at {cur}")
            ei = options[0]
            used.add(ei)
            e = level_arr.edges[ei]
            path.append(2 * ei if e.a == cur else 2 * ei + 1)
            cur = e.b if e.a == cur else e.a
        if len(used) != len(eis):
            raise SgbError(f"slab edge {slab.edge_ids[si]}: chain does not use all tagged edges")
        out[si] = path
    return out


def link_faces(level_arr: Arrangement, slab: SlabEdges) -> tuple[list[int], dict[int, list[int]]]:
    """Slab face index for every face of ``level_arr``, plus the chains used to compute it."""
    chains = trace_chains(level_arr, slab)
    parent = list(range(len(level_arr.faces)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    chain_edges = {h >> 1 for path in chains.values() for h in path}
    for ei in range(len(level_arr.edges)):
        if ei in chain_edges:
            continue
        a, b = find(level_arr.he_face[2 * ei]), find(level_arr.he_face[2 * ei + 1])
        if a != b:
            parent[max(a, b)] = min(a, b)

    assigned: dict[int, int] = {}

    def assign(level_face: int, slab_face: int) -> None:
        root = find(level_face)
        if assigned.setdefault(root, slab_face) != slab_face:
            raise SgbError("level cells disagree on the slab cell they lie in")

    for si, path in chains.items():
        left, right = slab.arr.he_face[2 * si], slab.arr.he_face[2 * si + 1]
        for h in path:
            assign(level_arr.he_face[h], left)
            assign(level_arr.he_face[h ^ 1], right)
    return [assigned.get(find(f), 0) for f in range(len(level_arr.faces))], chains
