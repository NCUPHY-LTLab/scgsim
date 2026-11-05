"""Mesh features of the sidecar: facts about the geometry that a mesher needs, never a mesh size (design brief section 12).

Owns the ``mesh_features`` list.  SGB knows the exact topology, so it states what is true of an edge set
(this edge set is the outline of a 0.2 um film wall; these are the boundaries of bands 0.5 and 1 um wide)
and leaves every size to the mesher.  Each entry names a ``kind``, its ``source``, the edges by planned id
and by native tag (tag = 1 + position in ``ids_by_tag["1"]``), and the ``facts`` of that kind:

* ``film_sidewall`` -- every edge of the sidewall faces of a finite body (``source`` = the entity id; facts
  ``thickness_um``).  A body is a conductor, in Route B too where its volume is not emitted, or a dielectric
  that does not reach the model boundary; vacuum and the walls on the model boundary are not films;
* ``band_boundary`` -- the edges of an inset family (``source`` = the request id; facts ``margins_um`` and
  ``narrowest_band_um``);
* ``arc`` -- the circular-arc edges of one radius (``source`` = the entity that owns them, or ``None``; facts
  ``radius_um``);
* ``refinement_curve`` -- the refinement-only curves of a refinement request (``source`` = the request id);
* ``port_edge`` -- the boundary edges of the faces of a lumped port (``source`` = the port sheet id).

Entries are sorted by ``(kind, source or "", canonical JSON of facts)`` and the edges of an entry by tag, so
the block is byte-identical for any worker count and tile layout.  Lengths are rounded like every other
micrometre value of the sidecar (``emission.measures.rounded``).

A wall of a film can be split in z by a level that crosses it (a port plane in the middle of a conductor).  The
thickness is that of the whole *column*: emitted walls of one body that stand on the same horizontal edge, one
on top of the other, are one column, and its thickness is the sum of their heights.  Only emitted walls count,
because the thickness describes the geometry the mesher receives, so it can differ between routes: in Route B
the wall between two touching conductors is not emitted, and the wall above it is a column of its own height.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Iterator
from fractions import Fraction
from typing import Any

from scgsim.geometry.sgb.assembly.records import TopoFace, Topology
from scgsim.geometry.sgb.cutfamilies.requests import ConductorSurface, InsetRequest
from scgsim.geometry.sgb.emission.measures import rounded
from scgsim.geometry.sgb.emission.plan import BrepPlan
from scgsim.geometry.sgb.errors import SgbError
from scgsim.geometry.sgb.model import ConformalModel
from scgsim.geometry.sgb.sidecar.groups import face_kind

FEATURE_KINDS = (
    "film_sidewall",
    "band_boundary",
    "arc",
    "refinement_curve",
    "port_edge",
)
BAND_FAMILY_ROLE = "inset_band"
REFINEMENT_FAMILY_ROLE = "refinement"

Feature = dict[str, Any]


def mesh_features(
    model: ConformalModel,
    plan: BrepPlan,
    carriers: dict[str, dict[str, Any]],
    ports: list[dict[str, Any]],
) -> list[Feature]:
    """The mesh features of the emitted edges, sorted by kind, source and facts."""
    tag_of = {edge_id: tag for tag, edge_id in enumerate(plan.edge_ids, start=1)}
    features = [
        *_film_sidewalls(model.topology, plan),
        *_family_features(model, plan, carriers),
        *_arcs(model, plan, carriers),
        *_port_edges(model.topology, ports),
    ]
    return sorted(
        (
            _feature(kind, source, edges, facts, tag_of)
            for kind, source, edges, facts in features
        ),
        key=_sort_key,
    )


def _feature(
    kind: str,
    source: str | None,
    edge_ids: Iterable[str],
    facts: dict[str, Any],
    tag_of: dict[str, int],
) -> Feature:
    edges = sorted(set(edge_ids), key=tag_of.__getitem__)
    return {
        "kind": kind,
        "source": source,
        "edges": edges,
        "native_tags": [tag_of[edge_id] for edge_id in edges],
        "facts": facts,
    }


def _sort_key(feature: Feature) -> tuple[str, str, str]:
    facts = json.dumps(feature["facts"], sort_keys=True, separators=(",", ":"))
    return feature["kind"], feature["source"] or "", facts


# --------------------------------------------------------------------------- film sidewalls


def _film_sidewalls(
    topology: Topology, plan: BrepPlan
) -> Iterator[tuple[str, str, set[str], dict[str, Any]]]:
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
    model: ConformalModel, plan: BrepPlan, carriers: dict[str, dict[str, Any]]
) -> Iterator[tuple[str, str, set[str], dict[str, Any]]]:
    """Band boundaries (with the margins of their request) and refinement curves, by family."""
    roles = {f.family_id: f.role for f in model.labelled.catalogue.families}
    edges_of_family: dict[str, set[str]] = defaultdict(set)
    for edge_id in plan.edge_ids:
        sources = (
            model.topology.lineage.get(edge_id) or model.topology.edges[edge_id].sources
        )
        for carrier in sources:
            record = carriers.get(carrier)
            if (
                record
                and record["kind"] == "family_member"
                and record["family"] in roles
            ):
                edges_of_family[str(record["family"])].add(edge_id)
    requests = {request.request_id: request for request in model.insets}
    for family, edge_ids in sorted(edges_of_family.items()):
        if roles[family] == BAND_FAMILY_ROLE:
            yield "band_boundary", family, edge_ids, _band_facts(requests, family)
        elif roles[family] == REFINEMENT_FAMILY_ROLE:
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
    model: ConformalModel, plan: BrepPlan, carriers: dict[str, dict[str, Any]]
) -> Iterator[tuple[str, str | None, set[str], dict[str, Any]]]:
    """Arc edges grouped by (owning entity if unique, exact carrier radius)."""
    topology = model.topology
    band_entity = {
        r.request_id: r.support.entity_id
        for r in model.insets
        if isinstance(r.support, ConductorSurface)
    }
    edges_of_arc: dict[tuple[str | None, float], set[str]] = defaultdict(set)
    for edge_id in plan.edge_ids:
        curve = topology.edges[edge_id].curve
        if curve is not None:
            owner = _arc_owner(topology, edge_id, carriers, band_entity)
            edges_of_arc[owner, rounded(curve.radius * topology.grid_um)].add(edge_id)
    for (owner, radius), edge_ids in edges_of_arc.items():
        yield "arc", owner, edge_ids, {"radius_um": radius}


def _arc_owner(
    topology: Topology,
    edge_id: str,
    carriers: dict[str, dict[str, Any]],
    band_entity: dict[str, str],
) -> str | None:
    """The one entity whose outline (or whose band) carries the arc, or ``None`` when no single entity does."""
    owners: set[str | None] = set()
    for carrier in topology.lineage.get(edge_id) or topology.edges[edge_id].sources:
        record = carriers.get(carrier, {"kind": "unknown"})
        if record["kind"] == "polygon_edge":
            owners.add(record["entity"])
        elif record["kind"] == "family_member":
            owners.add(band_entity.get(record["family"]))
        else:
            owners.add(None)
    return owners.pop() if len(owners) == 1 else None


# --------------------------------------------------------------------------- ports


def _port_edges(
    topology: Topology, ports: list[dict[str, Any]]
) -> Iterator[tuple[str, str, set[str], dict[str, Any]]]:
    """The boundary edges of the faces of each lumped-port sheet."""
    for port in ports:
        edge_ids = {
            edge_id
            for face_id in port["faces"]
            for loop in topology.faces[face_id].loops
            for edge_id, _ in loop
        }
        if edge_ids:
            yield "port_edge", port["port_sheet_id"], edge_ids, {}
