"""The emitted shapes of the sidecar: every written vertex, edge, face and volume (design brief sections 7, 11, 14).

Owns the ``vertex``, ``edge``, ``face`` and ``volume`` tables.  They describe what the XAO holds, as the plan
numbered it: a vertex's position; an edge's kind, two vertices in written order, arc, length and what it comes
from (carriers, or the Entities of a wall column); a face's group, kind, side volumes, written wires as oriented
edges, planned area, the Entities, band families and refinement requests it comes from, the area each Entity
covers, its band labels and the refinement curves embedded in it, and for a face on the model's outer boundary
the side of the model's box it lies on and the domain it bounds; a volume's domain, group, planned volume, AEDT name, the faces of the
written solid and the sheets embedded in it.  The load-side expectations by native tag are computed from these
tables (``scgsim.contracts.sgb_sidecar.views.expected``), so every planned measure is stored once.

Measures are the plan's: a face's area has its arcs realised and a volume its cylinder bulges
(``emission.measures``), so an inside-out shell shows as a negative or wrong volume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from scgsim.contracts.sgb_sidecar.records import (
    BandLabel,
    BoxSide,
    Carrier,
    Edge,
    EdgeArc,
    Face,
    FaceKind,
    Loop,
    OuterBoundary,
    Vertex,
    Volume,
    VolumeGroup,
    VolumeKind,
)
from scgsim.geometry.sgb.assembly.records import TopoFace
from scgsim.geometry.sgb.emission.measures import (
    planned_face_area_um2,
    planned_volume_um3,
    rounded,
)
from scgsim.geometry.sgb.emission.plan import ARC, BrepPlan
from scgsim.geometry.sgb.model import ConformalModel
from scgsim.geometry.sgb.sidecar.aedt import aedt_names
from scgsim.geometry.sgb.sidecar.derived import edge_sources
from scgsim.geometry.sgb.sidecar.lineage import CarrierLineage


def vertex_records(plan: BrepPlan) -> dict[str, Vertex]:
    """Every written vertex and where it sits."""
    return {
        vid: Vertex(position_um=[rounded(x) for x in row])
        for vid, row in zip(plan.vertex_ids, plan.verts.tolist(), strict=True)
    }


def edge_records(model: ConformalModel, plan: BrepPlan) -> dict[str, Edge]:
    """Every written edge: kind, vertices, arc, length and what it comes from."""
    topology = model.topology
    free = {plan.edge_ids[i] for i in plan.found.free_edges}
    out: dict[str, Edge] = {}
    for number, edge_id in enumerate(plan.edge_ids):
        edge = topology.edges[edge_id]
        start, end = plan.edge_v[number].tolist()
        sources = list(edge_sources(model, edge_id))
        vertical = edge.kind == "vertical"
        out[edge_id] = Edge(
            kind=edge.kind,
            vertices=[plan.vertex_ids[start], plan.vertex_ids[end]],
            arc=_arc(model, edge_id),
            length_um=_length_um(plan, number),
            carriers=[] if vertical else sources,
            entities=sources if vertical else [],
            written_as_free_edge=edge_id in free,
        )
    return out


def _arc(model: ConformalModel, edge_id: str) -> EdgeArc | None:
    """The realised circle of an arc edge, in micrometres; the radius as the ``arc`` mesh features state it."""
    curve = model.topology.edges[edge_id].curve
    if curve is None:
        return None
    grid = model.topology.grid_um
    return EdgeArc(
        centre_um=[rounded(curve.centre[0] * grid), rounded(curve.centre[1] * grid)],
        radius_um=rounded(curve.radius * grid),
        ccw=curve.ccw,
    )


def _length_um(plan: BrepPlan, number: int) -> float:
    """The written length: a line's parameter range, or an arc's radius times its angle."""
    span = float(plan.edge_span[number])
    if plan.edge_kind[number] == ARC:
        return rounded(float(plan.arc_radius[number]) * span)
    return rounded(span)


def face_records(
    model: ConformalModel,
    plan: BrepPlan,
    lineage: CarrierLineage,
    carriers: dict[str, Carrier],
) -> dict[str, Face]:
    """Every written face: group, kind, side volumes, written wires, area and what it comes from.

    ``carriers`` are the carrier records of the written edges (``derived.carrier_records``).
    """
    topology = model.topology
    found = plan.found
    box = _Box.of(plan)
    out: dict[str, Face] = {}
    for number, face_id in enumerate(plan.face_ids):
        face = topology.faces[face_id]
        first = found.face_first_wire[number]
        wires = found.wires[first : first + found.face_wire_count[number]]
        sources = _face_sources(model, lineage, carriers, face)
        out[face_id] = Face(
            group=model.routed.face_group.get(face_id),
            kind=face.kind,
            face_kind=cast(FaceKind, model.routed.faces[face_id].face_kind),
            z_um=[float(face.z_range[0]), float(face.z_range[1])],
            negative_volume=face.neg_volume,
            positive_volume=face.pos_volume,
            loops=[_loop(plan, wire) for wire in wires],
            area_um2=rounded(planned_face_area_um2(topology, face)),
            curved=face.curved,
            entities=sources.entities,
            families=sources.families,
            refinements=sources.refinements,
            cut_families=sources.cut_families,
            membership={
                entity: rounded(share)
                for entity, share in sorted(face.membership.items())
            },
            bands=[BandLabel(family=f, index=i) for f, i in face.bands],
            embedded_edges=list(face.refinement_edges),
            outer_boundary=_outer_boundary(model, plan, box, face, wires),
        )
    return out


@dataclass(frozen=True)
class _Box:
    """The extent of the written vertices: the lowest and highest coordinate on each axis, in micrometres."""

    low: tuple[float, float, float]
    high: tuple[float, float, float]

    @classmethod
    def of(cls, plan: BrepPlan) -> _Box:
        """The box of ``plan``'s vertices."""
        low, high = plan.verts.min(axis=0).tolist(), plan.verts.max(axis=0).tolist()
        return cls((low[0], low[1], low[2]), (high[0], high[1], high[2]))


# The planes of the box in the order a face is tested against them: (axis, low or high end, side).
_BOX_PLANES: tuple[tuple[int, bool, BoxSide], ...] = (
    (0, False, "-x"),
    (0, True, "+x"),
    (1, False, "-y"),
    (1, True, "+y"),
    (2, False, "bottom"),
    (2, True, "top"),
)


def _outer_boundary(
    model: ConformalModel,
    plan: BrepPlan,
    box: _Box,
    face: TopoFace,
    wires: list[list[int]],
) -> OuterBoundary | None:
    """Where a face on the outer boundary lies (one side has no volume): the box side and the domain inside."""
    if (face.neg_volume is None) == (face.pos_volume is None):
        return None
    inside = face.pos_volume if face.neg_volume is None else face.neg_volume
    assert inside is not None
    domain = model.topology.volumes[inside].domain.name
    return OuterBoundary(side=_box_side(plan, box, face, wires), domain=domain)


def _box_side(
    plan: BrepPlan, box: _Box, face: TopoFace, wires: list[list[int]]
) -> BoxSide | None:
    """The plane of the box all of a planar face's written vertices lie on, or ``None``.

    The vertices are the written ones, compared with the box of the same written coordinates, so a vertex on a
    plane has exactly that plane's coordinate.  A curved wall has its vertices on a plane only at its ends.
    """
    if face.curved:
        return None
    vertices = {int(v) for wire in wires for s in wire for v in plan.edge_v[abs(s) - 1]}
    points = plan.verts[sorted(vertices)]
    for axis, high, side in _BOX_PLANES:
        bound = box.high[axis] if high else box.low[axis]
        if bool((points[:, axis] == bound).all()):
            return side
    return None


@dataclass(frozen=True)
class _Sources:
    """What the carriers that bound a face come from, by kind."""

    entities: list[str]
    families: list[str]
    refinements: list[str]
    cut_families: list[str]


def _face_sources(
    model: ConformalModel,
    lineage: CarrierLineage,
    carriers: dict[str, Carrier],
    face: TopoFace,
) -> _Sources:
    """The Entities, band families, refinement requests and other cut families of the carriers on a face's
    horizontal edges.

    These are the sources SGB gave the face (``TopoFace.sources``), each classified by the carrier's record
    (``derived.carrier_record``): an id of one kind may equal an id of another (an Entity named like a request),
    so the kind is never read from the id.  A wall column carries the Entities of its walls, not carriers, and
    an unknown carrier comes from none.
    """
    topology = model.topology
    found: dict[str, set[str]] = {
        "source": set(),
        "band_member": set(),
        "refinement": set(),
        "family_member": set(),
    }
    for loop in face.loops:
        for edge_id, _ in loop:
            if topology.edges[edge_id].kind == "vertical":
                continue
            for carrier in topology.lineage.get(edge_id, ()):
                record = carriers[carrier]
                origin = lineage.lineage(carrier)
                if record["kind"] == "unknown" or origin is None:
                    continue
                found[record["kind"]].add(
                    origin.entity_id if record["kind"] == "source" else origin.family_id
                )
    return _Sources(
        sorted(found["source"]),
        sorted(found["band_member"]),
        sorted(found["refinement"]),
        sorted(found["family_member"]),
    )


def _loop(plan: BrepPlan, wire: list[int]) -> Loop:
    """A written wire (signed 1-based edge numbers) as edges and their directions."""
    return Loop(
        edges=[plan.edge_ids[abs(s) - 1] for s in wire],
        reversed=[s < 0 for s in wire],
    )


def volume_records(
    model: ConformalModel, plan: BrepPlan, volume_groups: dict[str, VolumeGroup]
) -> dict[str, Volume]:
    """Every planned volume; for an emitted solid its group, AEDT name, faces and embedded sheets."""
    topology = model.topology
    found = plan.found
    group_of = {
        vid: name
        for name, number in _volume_group_members(plan, volume_groups).items()
        for vid in number
    }
    names = aedt_names([(vid, group_of[vid]) for vid in plan.solid_ids])
    solid_faces = {}
    for number, vid in enumerate(plan.solid_ids):
        first = found.solid_first_shell[number]
        shells = found.shells[first : first + found.solid_shell_count[number]]
        tags = sorted({abs(s) for shell in shells for s in shell})
        solid_faces[vid] = [plan.face_ids[t - 1] for t in tags]
    out: dict[str, Volume] = {}
    for vid, volume in sorted(topology.volumes.items()):
        emitted = vid in model.routed.emitted_volumes
        out[vid] = Volume(
            entity=volume.domain.name,
            kind=cast(VolumeKind, volume.domain.kind),
            material=volume.domain.material,
            net=volume.domain.net or None,
            emitted=emitted,
            group=group_of.get(vid),
            shells=len(volume.shells),
            volume_um3=rounded(planned_volume_um3(topology, volume)),
            aedt_name=names.get(vid),
            faces=solid_faces.get(vid, []),
            embedded_faces=list(volume.embedded_faces) if vid in solid_faces else [],
        )
    return out


def _volume_group_members(
    plan: BrepPlan, volume_groups: dict[str, VolumeGroup]
) -> dict[str, list[str]]:
    """The solids of each volume group, from the plan's groups."""
    return {
        group.name: [plan.solid_ids[i] for i in group.elements]
        for group in plan.groups
        if group.dim == 3 and group.name in volume_groups
    }
