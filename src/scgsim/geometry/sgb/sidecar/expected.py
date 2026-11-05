"""What a loaded model must look like: the expectations of the load-side check (design brief section 14).

Owns the ``expected`` block of the sidecar.  The tests load the XAO and call
:func:`~scgsim.geometry.sgb.verification.loaded.verify_loaded` with the sidecar alone, so the block carries
everything that check compares, by native tag (tag = 1 + position in ``ids_by_tag``):

* ``vertices_um`` -- where each vertex must sit (this proves the numbering; every other check relies on it);
* ``edge_vertices``, ``face_edges``, ``solid_faces`` -- the boundary of each edge, face and solid as sorted tag lists;
* ``face_area_um2``, ``solid_volume_um3`` -- planned measures (arcs and cylinder bulges included; an inside-out
  shell shows as a negative or wrong volume).
"""

from __future__ import annotations

from typing import Any

from scgsim.geometry.sgb.assembly.records import Topology
from scgsim.geometry.sgb.emission.measures import (
    planned_face_area_um2,
    planned_volume_um3,
    rounded,
)
from scgsim.geometry.sgb.emission.plan import BrepPlan


def expected_record(topology: Topology, plan: BrepPlan) -> dict[str, Any]:
    """The ``expected`` block for the plan's shapes."""
    found = plan.found
    face_edges = []
    for i in range(len(found.face_ids)):
        first = found.face_first_wire[i]
        wires = found.wires[first : first + found.face_wire_count[i]]
        face_edges.append(sorted({abs(s) for wire in wires for s in wire}))
    solid_faces = []
    for i in range(len(found.solid_ids)):
        first = found.solid_first_shell[i]
        shells = found.shells[first : first + found.solid_shell_count[i]]
        solid_faces.append(sorted({abs(s) for shell in shells for s in shell}))
    return {
        "vertices_um": [[rounded(x) for x in row] for row in plan.verts.tolist()],
        "edge_vertices": (plan.edge_v + 1).tolist(),
        "face_edges": face_edges,
        "solid_faces": solid_faces,
        "face_area_um2": [
            rounded(planned_face_area_um2(topology, topology.faces[f]))
            for f in plan.face_ids
        ],
        "solid_volume_um3": [
            rounded(planned_volume_um3(topology, topology.volumes[v]))
            for v in plan.solid_ids
        ],
    }
