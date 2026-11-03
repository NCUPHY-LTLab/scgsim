"""Lineage records of the sidecar (design brief section 7): every edge and face back to its source.

Owns the pointers that let a consumer answer "which source curve is this edge" and "which Entities
cover this face" without geometry: ``edges`` maps a horizontal or refinement edge id to the carrier ids that produced it,
``carriers`` describes each such carrier (source polygon edge, or cut-family member),
``vertical_edges`` maps a wall column to the Entities whose walls stand on it, ``faces`` lists the
source Entities of a face and ``membership`` the area fraction each covers (non-exclusive, so the
fractions of a face may sum to more than 1; walls use the share of their bottom boundary length).

Carriers that exist only to cut cells for the membership computation never appear on an emitted edge.
"""

from __future__ import annotations

from typing import Any

from scgsim.geometry.sgb.assembly.records import Topology
from scgsim.geometry.sgb.emission.measures import rounded
from scgsim.geometry.sgb.emission.plan import BrepPlan
from scgsim.geometry.sgb.model import ConformalModel
from scgsim.geometry.sgb.records.carriers import Lineage, base_carrier_id


def lineage_record(model: ConformalModel, plan: BrepPlan) -> dict[str, Any]:
    """The ``lineage`` block: edge carriers, carrier descriptions, face sources and area-weighted membership."""
    topology = model.topology
    sources = {
        eid: list(topology.lineage.get(eid) or topology.edges[eid].sources)
        for eid in plan.edge_ids
    }
    edges = {
        eid: s for eid, s in sources.items() if topology.edges[eid].kind != "vertical"
    }
    vertical = {
        eid: s for eid, s in sources.items() if topology.edges[eid].kind == "vertical"
    }
    known = _carrier_lineages(model)
    carriers = {
        cid: _describe(known.get(cid) or known.get(base_carrier_id(cid)))
        for cid in sorted({c for sources in edges.values() for c in sources})
    }
    return {
        "edges": edges,
        "carriers": carriers,
        "vertical_edges": vertical,
        "faces": {fid: list(topology.faces[fid].sources) for fid in plan.face_ids},
        "membership": {
            fid: {
                entity: rounded(share)
                for entity, share in sorted(topology.faces[fid].membership.items())
            }
            for fid in plan.face_ids
        },
    }


def edge_family(
    topology: Topology, edge_id: str, carriers: dict[str, dict[str, Any]]
) -> str | None:
    """The cut family a refinement or band edge belongs to, from the first family-member carrier on it."""
    for carrier_id in topology.lineage.get(edge_id) or topology.edges[edge_id].sources:
        record = carriers.get(carrier_id)
        if record and record["kind"] == "family_member":
            return str(record["family"])
    return None


def _carrier_lineages(model: ConformalModel) -> dict[str, Lineage]:
    """Lineage of every carrier id the model knows: source line carriers and arcs, family members, derived arcs."""
    catalogue = model.labelled.catalogue
    known: dict[str, Lineage] = {}
    for prism in catalogue.prisms:
        for carrier in prism.carriers():
            known[carrier.id] = carrier.lineage
        for arc in prism.arcs:
            known[arc.id] = arc.lineage
    for family in catalogue.families:
        for carrier in family.carriers:
            known[carrier.id] = carrier.lineage
        for arc in family.arcs:
            known[arc.id] = arc.lineage
    for arc_id, arc in model.labelled.arcs.items():
        known.setdefault(arc_id, arc.lineage)
    return known


def _describe(lineage: Lineage | None) -> dict[str, Any]:
    if lineage is None:
        return {"kind": "unknown"}
    if lineage.kind == "polygon_edge":
        return {
            "kind": "polygon_edge",
            "entity": lineage.entity_id,
            "polygon": lineage.polygon,
            "ring": lineage.ring,
            "edge": lineage.edge,
        }
    return {
        "kind": "family_member",
        "family": lineage.family_id,
        "member": lineage.member,
    }
