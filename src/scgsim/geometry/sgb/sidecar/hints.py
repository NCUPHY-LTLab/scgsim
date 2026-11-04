"""Mesh hints of the sidecar: which edges the mesher should resolve, never how finely (design brief section 12).

Owns the ``mesh_hints`` list.  Each entry names a kind, its source and the edges, by planned id and by native
tag (tag = 1 + position in ``ids_by_tag["1"]``).  There are no sizes: the mesher decides them (see "Sizing guidance" below).

* ``band_boundary`` -- the edges of an inset family (``source`` = the request id): where the Surface-EPR bands
  meet, so the elements should follow them;
* ``refinement_curve`` -- the refinement-only curves of a refinement request (``source`` = the request id);
* ``curved_edge`` -- every circular-arc edge (``source`` is ``None``): their chords need resolving on the mesh.

Sizing guidance
---------------
A band can be far narrower than the model (``pad_in_ground_B_bands``: 0.1 um strips in a 300 um chip), and
a mesh size that ignores the hints meshes those strips with elements hundreds of times wider than they
are.  Then the mesh can contain a tetrahedron of volume zero (``minSICN`` 0) with no fault in the geometry:
in ``pad_b_bands`` at a uniform 60 um, four parallel band-boundary edges symmetric about x = 0 each get a
node at x = 0, and 3D Delaunay joins those four collinear nodes.  Every size from 50 um upwards gives the
same mesh and the same tetrahedron; the 3D algorithms Frontal-Delaunay (4) and HXT (10) avoid it here, MMG3D
(7) stops on it.  What the mesher needs:

* **Near a ``band_boundary`` edge**: an element size of the order of the narrowest band margin, 0.1 to 0.2
  um for margins from 0.1 um (0.5 um also avoided it in the test case).
* **Growth**: a size that grows from there to the coarse size over distance, for example a Threshold field
  on the Distance to the hinted edges with ``DistMax = (coarse - near) / growth``.  Growth of 2 per
  micrometre gives ``minSICN`` 0.02 (320 thousand tetrahedra, minutes); growth of 8 gives about 0.001
  (40 thousand, 20 s), which is what the test uses; growth of 20 or more degrades it.
* Take the edges from ``native_tags`` (tag = 1 + position in ``ids_by_tag["1"]``) and turn off the size
  sources that would override the field (sizes from points, from curvature, extension from the boundary).
* ``curved_edge`` hints are not needed to avoid flat elements; use them to set the chord length.

The reference implementation is ``size_from_mesh_hints`` in the test helper ``sgb_gmsh_oracle``.
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
