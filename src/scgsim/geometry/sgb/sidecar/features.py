"""Mesh features of the sidecar: facts about the geometry that a mesher needs, never a mesh size (design brief section 12).

Owns the ``mesh_feature`` table.  SGB knows the exact topology, so it states what is true of an edge set
(this edge set is the outline of a 0.2 um film wall; these are the boundaries of bands 0.5 and 1 um wide)
and leaves every size to the mesher.  Each feature names a ``kind``, its ``source``, the edges by planned id
(in tag order; readers compute the native tags), and the ``facts`` of that kind:

* ``film_sidewall`` -- every edge of the sidewall faces of a finite body (``source`` = the entity id; facts
  ``thickness_um``).  A body is a conductor, in Route B too where its volume is not emitted, or a dielectric
  that does not reach the model boundary; vacuum and the walls on the model boundary are not films;
* ``band_boundary`` -- the edges of an inset family (``source`` = the band family; facts ``margins_um`` and
  ``narrowest_band_um``);
* ``arc`` -- the circular-arc edges of one radius (``source`` = the entity that owns them, or ``None``; facts
  ``radius_um``);
* ``refinement_curve`` -- the refinement-only curves of a refinement request (``source`` = the request id);
* ``port_edge`` -- the boundary edges of the faces of a lumped port (``source`` = the port sheet id).

A feature's id is a digest of its kind, source and facts, which no two features share; readers take the
features in the order of ``(kind, source or "", canonical JSON of facts)``
(``scgsim.contracts.sgb_sidecar.views.mesh_features``).  Lengths are rounded like every other micrometre value of the sidecar
(``emission.measures.rounded``).

A wall of a film can be split in z by a level that crosses it (a port plane in the middle of a conductor).  The
thickness is that of the whole *column*: emitted walls of one body that stand on the same horizontal edge, one
on top of the other, are one column, and its thickness is the sum of their heights.  Only emitted walls count,
because the thickness describes the geometry the mesher receives, so it can differ between routes: in Route B
the wall between two touching conductors is not emitted, and the wall above it is a column of its own height.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Iterator
from fractions import Fraction
from typing import Any, cast

from scgsim.contracts.sgb_sidecar.records import MeshFeature, Port
from scgsim.geometry.sgb.assembly.records import TopoFace, Topology
from scgsim.geometry.sgb.cutfamilies.requests import ConductorSurface, InsetRequest
from scgsim.geometry.sgb.emission.measures import rounded
from scgsim.geometry.sgb.emission.plan import BrepPlan
from scgsim.geometry.sgb.errors import SgbError
from scgsim.geometry.sgb.model import ConformalModel
from scgsim.geometry.sgb.sidecar.derived import edge_sources
from scgsim.geometry.sgb.sidecar.groups import face_kind
from scgsim.geometry.sgb.sidecar.lineage import (
    BAND_FAMILY_ROLE,
    REFINEMENT_FAMILY_ROLE,
    CarrierLineage,
)

FEATURE_KINDS = (
    "film_sidewall",
    "band_boundary",
    "arc",
    "refinement_curve",
    "port_edge",
)

# A feature as a producer below states it: kind, source, edges and facts.
_Stated = tuple[str, str | None, set[str], dict[str, Any]]


def mesh_features(
    model: ConformalModel,
    plan: BrepPlan,
    lineage: CarrierLineage,
    ports: dict[str, Port],
) -> dict[str, MeshFeature]:
    """The mesh features of the emitted edges, by id."""
    tag_of = {edge_id: tag for tag, edge_id in enumerate(plan.edge_ids, start=1)}
    stated: list[_Stated] = [
        *_film_sidewalls(model.topology, plan),
        *_family_features(model, plan, lineage),
        *_arcs(model, plan, lineage),
        *_port_edges(model.topology, ports),
    ]
    out: dict[str, MeshFeature] = {}
    for kind, source, edges, facts in stated:
        feature_id = _feature_id(kind, source, facts)
        if feature_id in out:
            raise SgbError(f"internal: two {kind} features of {source!r} state {facts}")
        out[feature_id] = _feature(kind, source, edges, facts, tag_of)
    return dict(sorted(out.items()))


def _feature(
    kind: str,
    source: str | None,
    edge_ids: Iterable[str],
    facts: dict[str, Any],
    tag_of: dict[str, int],
) -> MeshFeature:
    edges = sorted(set(edge_ids), key=tag_of.__getitem__)
    # Each producer states the source and facts its kind has in the contract.
    return cast(
        MeshFeature, {"kind": kind, "source": source, "edges": edges, "facts": facts}
    )


def _feature_id(kind: str, source: str | None, facts: dict[str, Any]) -> str:
    """An id from what the feature states, so it does not change when another feature is added."""
    canonical = json.dumps([kind, source, facts], sort_keys=True, separators=(",", ":"))
    return "mf:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- film sidewalls


def _film_sidewalls(topology: Topology, plan: BrepPlan) -> Iterator[_Stated]:
    """One feature per (body, column thickness): all edges of the sidewall faces of that body's columns."""
    walls_of_body: dict[str, list[TopoFace]] = defaultdict(list)
    for face_id in plan.face_ids:
        face = topology.faces[face_id]
        if face_kind(face) == "sidewall":
            for body in _film_bodies(face):
                walls_of_body[body].append(face)
    for body, walls in sorted(walls_of_body.items()):
        edges_of_thickness: dict[float, set[str]] = defaultdict(set)
        for column in _wall_columns(topology, walls):
            thickness = rounded(
                float(
                    max(w.z_range[1] for w in column)
                    - min(w.z_range[0] for w in column)
                )
            )
            edges_of_thickness[thickness] |= {
                edge_id for wall in column for loop in wall.loops for edge_id, _ in loop
            }
        for thickness, edge_ids in sorted(edges_of_thickness.items()):
            yield "film_sidewall", body, edge_ids, {"thickness_um": thickness}


def _film_bodies(wall: TopoFace) -> list[str]:
    """The finite bodies a sidewall belongs to: its conductors, and its dielectrics when it is not on the model boundary.

    A wall shared by a conductor and a dielectric belongs to both, so each body keeps its whole column.
    """
    sides = [d for d in (wall.neg_domain, wall.pos_domain) if d is not None]
    bodies = {d.name for d in sides if d.kind == "conductor"}
    if len(sides) == 2:  # noqa: PLR2004 - a wall with one side missing lies on the model boundary
        bodies |= {d.name for d in sides if d.kind == "dielectric"}
    return sorted(bodies)


def _wall_columns(topology: Topology, walls: list[TopoFace]) -> list[list[TopoFace]]:
    """Group walls into columns: a wall stands on the horizontal edge that is the top of the wall below it."""
    parent = {wall.id: wall.id for wall in walls}

    def root(wall_id: str) -> str:
        while parent[wall_id] != wall_id:
            parent[wall_id] = parent[parent[wall_id]]
            wall_id = parent[wall_id]
        return wall_id

    top_of: dict[str, list[TopoFace]] = defaultdict(list)
    for wall in walls:
        for edge_id in _edges_at(topology, wall, wall.z_range[1]):
            top_of[edge_id].append(wall)
    for wall in walls:
        for edge_id in _edges_at(topology, wall, wall.z_range[0]):
            for below in top_of.get(edge_id, ()):
                parent[root(wall.id)] = root(below.id)
    columns: dict[str, list[TopoFace]] = defaultdict(list)
    for wall in walls:
        columns[root(wall.id)].append(wall)
    return [columns[key] for key in sorted(columns)]


def _edges_at(topology: Topology, wall: TopoFace, z: Fraction) -> Iterator[str]:
    """The horizontal edges of a wall that lie on the plane ``z``."""
    for loop in wall.loops:
        for edge_id, _ in loop:
            edge = topology.edges[edge_id]
            if edge.kind == "horizontal" and topology.vertices[edge.v0].z == z:
                yield edge_id


# --------------------------------------------------------------------------- cut families


def _family_features(
    model: ConformalModel, plan: BrepPlan, lineage: CarrierLineage
) -> Iterator[_Stated]:
    """Band boundaries (with the margins of their request) and refinement curves, by family."""
    edges_of_family: dict[str, set[str]] = defaultdict(set)
    for edge_id in plan.edge_ids:
        if model.topology.edges[edge_id].kind == "vertical":
            continue  # a wall column carries Entities, not carriers
        for carrier in edge_sources(model, edge_id):
            found = lineage.lineage(carrier)
            if found is not None and found.kind == "family_member":
                edges_of_family[found.family_id].add(edge_id)
    requests = {request.request_id: request for request in model.inset_plan.requests}
    for family, edge_ids in sorted(edges_of_family.items()):
        role = lineage.role(family)
        if role == BAND_FAMILY_ROLE:
            yield "band_boundary", family, edge_ids, _band_facts(requests, family)
        elif role == REFINEMENT_FAMILY_ROLE:
            yield "refinement_curve", family, edge_ids, {}


def _band_facts(requests: dict[str, InsetRequest], family: str) -> dict[str, Any]:
    """The margins of an inset request, ascending, and the narrowest band they leave (the first one starts at 0)."""
    if family not in requests:
        raise SgbError(f"band family {family!r} has no inset request in the model")
    margins = sorted(requests[family].margins_um)
    widths = [
        upper - lower for lower, upper in zip((0.0, *margins), margins, strict=False)
    ]
    return {
        "margins_um": [rounded(m) for m in margins],
        "narrowest_band_um": rounded(min(widths)),
    }


# --------------------------------------------------------------------------- arcs


def _arcs(
    model: ConformalModel, plan: BrepPlan, lineage: CarrierLineage
) -> Iterator[_Stated]:
    """Arc edges grouped by (owning entity if unique, exact carrier radius)."""
    topology = model.topology
    band_entity = {
        r.request_id: r.support.entity_id
        for r in model.inset_plan.requests
        if isinstance(r.support, ConductorSurface)
    }
    edges_of_arc: dict[tuple[str | None, float], set[str]] = defaultdict(set)
    for edge_id in plan.edge_ids:
        curve = topology.edges[edge_id].curve
        if curve is not None:
            owner = _arc_owner(model, edge_id, lineage, band_entity)
            edges_of_arc[owner, rounded(curve.radius * topology.grid_um)].add(edge_id)
    for (owner, radius), edge_ids in edges_of_arc.items():
        yield "arc", owner, edge_ids, {"radius_um": radius}


def _arc_owner(
    model: ConformalModel,
    edge_id: str,
    lineage: CarrierLineage,
    band_entity: dict[str, str],
) -> str | None:
    """The one entity whose outline (or whose band) carries the arc, or ``None`` when no single entity does."""
    owners: set[str | None] = set()
    for carrier in edge_sources(model, edge_id):
        found = lineage.lineage(carrier)
        if found is None:
            owners.add(None)
        elif found.kind == "polygon_edge":
            owners.add(found.entity_id)
        else:
            owners.add(band_entity.get(found.family_id))
    return owners.pop() if len(owners) == 1 else None


# --------------------------------------------------------------------------- ports


def _port_edges(topology: Topology, ports: dict[str, Port]) -> Iterator[_Stated]:
    """The boundary edges of the faces of each lumped-port sheet."""
    for port_id, port in ports.items():
        edge_ids = {
            edge_id
            for face_id in port["faces"]
            for loop in topology.faces[face_id].loops
            for edge_id, _ in loop
        }
        if edge_ids:
            yield "port_edge", port_id, edge_ids, {}
