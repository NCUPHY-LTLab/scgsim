"""Realisation of the arc edges of a layer (design brief section 3, decision D3).

An edge whose carriers are all pieces of one source arc is an arc edge: its
realised arc passes through the two snapped end points with the carrier's radius.
Slab arc edges are registered as derived carriers, so the level arrangements above
and below can node them again as arcs.
"""

from __future__ import annotations

from scgsim.geometry.sgb.arrangement.arcs import derived_arc, realise
from scgsim.geometry.sgb.derivation.layer import Layer
from scgsim.geometry.sgb.labelling.linking import TAG
from scgsim.geometry.sgb.records.carriers import ArcCarrier


def realise_arc_edges(
    layer: Layer, arcs: dict[str, ArcCarrier], *, register: bool
) -> None:
    """Give every arc edge its realised arc; with ``register``, slab arcs become carriers of ``arcs``."""
    layer.edge_curve = []
    for edge, edge_id in zip(layer.arr.edges, layer.edge_ids, strict=False):
        carriers = sorted(c.id for c in edge.carriers)
        roots = {arcs[c].root_id for c in carriers if c in arcs}
        if not carriers or not all(c in arcs for c in carriers) or len(roots) != 1:
            layer.edge_curve.append(None)
            continue
        arc = arcs[carriers[0]]
        root = arcs.get(arc.root_id, arc)
        realised = realise(arc, root, edge.a, edge.b)
        layer.edge_curve.append(realised)
        if register:
            tag = TAG + edge_id
            arcs[tag] = derived_arc(tag, root, realised, (edge.a, edge.b), arc.lineage)
