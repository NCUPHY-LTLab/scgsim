"""Atomic content-addressed ids of a layer's vertices, edges and faces (stage S8 part 1).

Owns computing every id of a batch of layers as one parallel stage: the items are
vertices, edges and faces, each owned by the tile of its anchor point.  An edge
of a level arrangement that covers a slab edge carries the slab edge's tagged id
as a carrier; its source set expands that tag back to the polygon-edge carriers,
so ids stay stable when slab edges are renumbered.
"""

from __future__ import annotations

from typing import Any

from scgsim.geometry.sgb import identity
from scgsim.geometry.sgb.derivation.layer import Layer
from scgsim.geometry.sgb.labelling.linking import TAG
from scgsim.geometry.sgb.parallel import Runtime


def assign_ids(
    layers: list[Layer], tag_sources: list[dict[str, tuple[str, ...]]], rt: Runtime
) -> None:
    """Vertex and edge ids, then face ids, for every layer (one parallel stage each)."""
    _assign_vertex_and_edge_ids(layers, tag_sources, rt)
    _assign_face_ids(layers, rt)


def _assign_vertex_and_edge_ids(
    layers: list[Layer], tag_sources: list[dict[str, tuple[str, ...]]], rt: Runtime
) -> None:
    arrangements = [layer.arr for layer in layers]
    planes = [layer.plane for layer in layers]
    tasks: list[tuple[int, int, int]] = []
    owners: list[int] = []
    for li, arr in enumerate(arrangements):
        for vi, p in enumerate(arr.vertices):
            tasks.append((0, li, vi))
            owners.append(rt.owner(p))
        for ei, e in enumerate(arr.edges):
            tasks.append((1, li, ei))
            owners.append(rt.owner(e.a))
    out = rt.run(
        "atomic_ids",
        _vertex_edge_id_job,
        (arrangements, planes, tag_sources, tasks),
        owners,
    )
    for layer in layers:
        layer.vertex_ids = [""] * len(layer.arr.vertices)
        layer.edge_ids = [""] * len(layer.arr.edges)
        layer.edge_sources = [()] * len(layer.arr.edges)
    for (kind, li, index), value in zip(tasks, out, strict=False):
        if kind == 0:
            layers[li].vertex_ids[index] = value
        else:
            layers[li].edge_sources[index], layers[li].edge_ids[index] = value


def _assign_face_ids(layers: list[Layer], rt: Runtime) -> None:
    arrangements = [layer.arr for layer in layers]
    tasks = [
        (li, fi)
        for li, arr in enumerate(arrangements)
        for fi in range(1, len(arr.faces))
    ]
    owners = [rt.owner(arrangements[li].face_anchor(fi)) for li, fi in tasks]
    shared = (
        arrangements,
        [layer.edge_ids for layer in layers],
        [layer.labels for layer in layers],
        [layer.plane for layer in layers],
        [layer.kind for layer in layers],
        tasks,
    )
    out = rt.run("atomic_ids", _face_id_job, shared, owners)
    for layer in layers:
        layer.face_ids = [""] * len(layer.arr.faces)
    for (li, fi), face in zip(tasks, out, strict=False):
        layers[li].face_ids[fi] = face


def _vertex_edge_id_job(shared: Any, items: list[int]) -> list[Any]:
    arrangements, planes, tag_sources, tasks = shared
    out: list[Any] = []
    for i in items:
        kind, li, index = tasks[i]
        arr, plane = arrangements[li], planes[li]
        if kind == 0:
            out.append(identity.vertex_id(plane, arr.vertices[index]))
            continue
        edge = arr.edges[index]
        sources: set[str] = set()
        for carrier in edge.carriers:
            sources.update(
                tag_sources[li][carrier] if carrier.startswith(TAG) else (carrier,)
            )
        ordered = tuple(sorted(sources))
        out.append((ordered, identity.edge_id(plane, edge.a, edge.b, ordered)))
    return out


def _face_id_job(shared: Any, items: list[int]) -> list[str]:
    arrangements, edge_ids, labels, planes, kinds, tasks = shared
    out = []
    for i in items:
        li, fi = tasks[i]
        arr = arrangements[li]
        boundary = [
            edge_ids[li][h >> 1]
            for c in arr.face_cycles(fi)
            for h in arr.cycles[c].half_edges
        ]
        if kinds[li] == "slab":
            out.append(identity.cell_id(planes[li], labels[li][fi], boundary))
        else:
            out.append(
                identity.face_id("horizontal", planes[li], labels[li][fi], boundary)
            )
    return out
