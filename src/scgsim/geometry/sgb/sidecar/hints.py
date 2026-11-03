"""Mesh hints of the sidecar: which edges the mesher should resolve, never how finely (design brief section 12).

Owns the ``mesh_hints`` list.  Each entry names a kind, its source and the edges, by planned id and by native
tag (tag = 1 + position in ``ids_by_tag["1"]``).  There are no sizes: the mesher decides them.

* ``band_boundary`` -- the edges of an inset family (``source`` = the request id): where the Surface-EPR bands
  meet, so the elements should follow them;
* ``refinement_curve`` -- the refinement-only curves of a refinement request (``source`` = the request id);
* ``curved_edge`` -- every circular-arc edge (``source`` is ``None``): their chords need resolving on the mesh.
"""

from __future__ import annotations

from typing import Any

from scgsim.geometry.sgb.emission.plan import BrepPlan
from scgsim.geometry.sgb.model import ConformalModel

FAMILY_ROLE_HINT = {"inset_band": "band_boundary", "refinement": "refinement_curve"}


def mesh_hints(
    model: ConformalModel, plan: BrepPlan, carriers: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Mesh hints of the emitted edges, ordered by kind and source."""
    topology = model.topology
    hint_of = {
        family.family_id: FAMILY_ROLE_HINT[family.role]
        for family in model.labelled.catalogue.families
        if family.role in FAMILY_ROLE_HINT
    }
    by_source: dict[tuple[str, str | None], list[int]] = {}
    for tag, edge_id in enumerate(plan.edge_ids, start=1):
        edge = topology.edges[edge_id]
        sources = topology.lineage.get(edge_id) or edge.sources
        families = {
            str(carriers[c]["family"])
            for c in sources
            if c in carriers and carriers[c]["kind"] == "family_member"
        }
        for family in sorted(families & hint_of.keys()):
            by_source.setdefault((hint_of[family], family), []).append(tag)
        if edge.curve is not None:
            by_source.setdefault(("curved_edge", None), []).append(tag)
    return [
        {
            "kind": kind,
            "source": source,
            "edges": [plan.edge_ids[t - 1] for t in tags],
            "native_tags": tags,
        }
        for (kind, source), tags in sorted(
            by_source.items(), key=lambda i: (i[0][0], i[0][1] or "")
        )
    ]
