"""Atomic content-addressed ids of a layer's vertices, edges and faces (stage S8 part 1).

Owns computing every id of a batch of layers.  An edge of a level arrangement that
covers a slab edge carries the slab edge's tagged id as a carrier; its source set
expands that tag back to the polygon-edge carriers, so ids stay stable when slab
edges are renumbered.
"""

from __future__ import annotations

from scgsim.geometry.sgb import identity
from scgsim.geometry.sgb.derivation.layer import Layer
from scgsim.geometry.sgb.labelling.linking import TAG


def assign_ids(
    layers: list[Layer], tag_sources: list[dict[str, tuple[str, ...]]]
) -> None:
    """Vertex and edge ids, then face ids (which hash the edge ids), for every layer."""
    for layer, sources in zip(layers, tag_sources, strict=True):
        _assign_vertex_and_edge_ids(layer, sources)
    for layer in layers:
        _assign_face_ids(layer)


def _assign_vertex_and_edge_ids(
    layer: Layer, tag_sources: dict[str, tuple[str, ...]]
) -> None:
    arr, plane = layer.arr, layer.plane
    layer.vertex_ids = [identity.vertex_id(plane, p) for p in arr.vertices]
    layer.edge_ids = []
    layer.edge_sources = []
    for edge in arr.edges:
        sources: set[str] = set()
        for carrier in edge.carriers:
            sources.update(
                tag_sources[carrier] if carrier.startswith(TAG) else (carrier,)
            )
        ordered = tuple(sorted(sources))
        layer.edge_sources.append(ordered)
        layer.edge_ids.append(identity.edge_id(plane, edge.a, edge.b, ordered))


def _assign_face_ids(layer: Layer) -> None:
    arr = layer.arr
    layer.face_ids = [""]
    for fi in range(1, len(arr.faces)):
        boundary = [
            layer.edge_ids[h >> 1]
            for c in arr.face_cycles(fi)
            for h in arr.cycles[c].half_edges
        ]
        if layer.kind == "slab":
            layer.face_ids.append(
                identity.cell_id(layer.plane, layer.labels[fi], boundary)
            )
        else:
            layer.face_ids.append(
                identity.face_id("horizontal", layer.plane, layer.labels[fi], boundary)
            )
