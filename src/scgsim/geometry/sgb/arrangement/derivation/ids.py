"""Atomic content-addressed ids of a layer's vertices, edges and faces (stage S8 part 1).

Owns computing every id of a batch of layers.  An edge of a level arrangement that
covers a slab edge carries that slab edge (role ``SLAB_EDGE``); its source set
expands it back to the slab edge's polygon-edge carriers, so ids stay stable when
slab edges are renumbered.
"""

from __future__ import annotations

from scgsim.geometry.sgb.exact import identity
from scgsim.geometry.sgb.arrangement.derivation.layer import Layer
from scgsim.geometry.sgb.errors import SgbError
from scgsim.geometry.sgb.exact.carriers import CarrierRole, EdgeCarrier


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
        sources = {
            s for carrier in edge.carriers for s in _sources(carrier, tag_sources)
        }
        ordered = tuple(sorted(sources))
        layer.edge_sources.append(ordered)
        layer.edge_ids.append(identity.edge_id(plane, edge.a, edge.b, ordered))


def _sources(
    carrier: EdgeCarrier, tag_sources: dict[str, tuple[str, ...]]
) -> tuple[str, ...]:
    """The source carrier ids an edge's carrier stands for: a slab edge's sources, else the curve itself.

    Only curves and slab edges name a kept edge's sources; a membership carrier never decides that an edge is
    kept, so one alone on a kept edge is an SGB defect.
    """
    if carrier.role is CarrierRole.SLAB_EDGE:
        return tag_sources[carrier.id]
    if carrier.role is not CarrierRole.CURVE:
        raise SgbError(
            f"internal: the {carrier.role.value} carrier {carrier.id!r} names the sources of a kept edge"
        )
    return (carrier.id,)


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
