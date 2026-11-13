"""Tying a level arrangement to the slab arrangements it refines (design brief section 2).

Owns two derived facts needed by labelling and assembly:

* **chains** -- the path of level edges that covers each slab edge.  A level
  arrangement is built from slab edges carried with the role ``SLAB_EDGE`` under
  the id ``~<edge id>``, so the level edges carrying one form a simple path from
  one slab-edge endpoint to the other (the bottom or top boundary of that edge's
  wall).
* **links** -- which slab cell each level cell lies in.  Level cells are joined
  across every edge that is *not* part of a slab-edge chain (those edges do not
  separate slab cells); each resulting group borders at least one chain, whose
  left and right sides name the slab cells.

Pure functions of two arrangements; no geometry is created.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from scgsim.geometry.sgb.arrangement.dcel import Arrangement
from scgsim.geometry.sgb.errors import SgbError
from scgsim.geometry.sgb.exact.carriers import CarrierRole
from scgsim.geometry.sgb.exact.plane import Point

TAG = "~"
"""Prefix of the carrier id a slab edge gets when it enters a level arrangement; the id is written with it,
and the carrier's role (``CarrierRole.SLAB_EDGE``) says what it is."""


@dataclass(frozen=True)
class SlabEdges:
    """What a slab layer exposes for linking: its arrangement and edge ids."""

    arr: Arrangement
    edge_ids: list[str]


def trace_chains(level_arr: Arrangement, slab: SlabEdges) -> dict[int, list[int]]:
    """For every slab edge index, the level half-edges from its ``a`` end to its ``b`` end."""
    tag_to_slab_edge = {TAG + eid: i for i, eid in enumerate(slab.edge_ids)}
    members: dict[int, list[int]] = defaultdict(list)
    for ei, edge in enumerate(level_arr.edges):
        for carrier in edge.carriers:
            if carrier.role is not CarrierRole.SLAB_EDGE:
                continue
            slab_edge = tag_to_slab_edge.get(carrier.curve_id)
            if slab_edge is not None:
                members[slab_edge].append(ei)
    chains: dict[int, list[int]] = {}
    for si, slab_edge in enumerate(slab.arr.edges):
        edge_indices = members.get(si)
        if not edge_indices:
            raise SgbError(
                f"internal: slab edge {slab.edge_ids[si]} has no chain in the level arrangement"
            )
        chains[si] = walk_chain(
            level_arr, edge_indices, slab_edge.a, slab_edge.b, slab.edge_ids[si]
        )
    return chains


def walk_chain(
    level_arr: Arrangement,
    edge_indices: list[int],
    start: Point,
    end: Point,
    slab_edge_id: str,
) -> list[int]:
    """The half-edges of the simple path of the given level edges from ``start`` to ``end``.

    ``slab_edge_id`` names the slab edge in the error when the path is broken or ambiguous.
    """
    at: dict[Point, list[int]] = defaultdict(list)
    for ei in edge_indices:
        e = level_arr.edges[ei]
        at[e.a].append(ei)
        at[e.b].append(ei)
    current, path, used = start, [], set()
    while current != end:
        options = [ei for ei in at[current] if ei not in used]
        if len(options) != 1:
            raise SgbError(
                f"internal: slab edge {slab_edge_id}: ambiguous or broken chain at {current}"
            )
        ei = options[0]
        used.add(ei)
        e = level_arr.edges[ei]
        path.append(2 * ei if e.a == current else 2 * ei + 1)
        current = e.b if e.a == current else e.a
    if len(used) != len(edge_indices):
        raise SgbError(
            f"internal: slab edge {slab_edge_id}: chain does not use all tagged edges"
        )
    return path


def link_faces(
    level_arr: Arrangement, slab: SlabEdges
) -> tuple[list[int], dict[int, list[int]]]:
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
            raise SgbError(
                "internal: level cells disagree on the slab cell they lie in"
            )

    for si, path in chains.items():
        left, right = slab.arr.he_face[2 * si], slab.arr.he_face[2 * si + 1]
        for h in path:
            assign(level_arr.he_face[h], left)
            assign(level_arr.he_face[h ^ 1], right)
    return [assigned.get(find(f), 0) for f in range(len(level_arr.faces))], chains
