"""Physical groups and Surface-EPR rows as the sidecar describes them (design brief sections 7 and 14).

Owns the facts a solver adapter needs about each primary physical group beyond its members: which
Entity or Entities it belongs to, the Net, whether it is a top, bottom, sidewall or sheet surface,
the volumes and solution regions on its sides, its inset band, its area -- so today's group record
(owner ids, adjacent solution volumes, face kind, Net) can be rebuilt without reading geometry.

Primary groups partition each dimension.  A face or volume names its group in its own record, so a group
record lists no members: readers compute them (``scgsim.contracts.sgb_sidecar.views.group_members``).
"""

from __future__ import annotations

from typing import cast

from scgsim.contracts.sgb_sidecar.records import (
    AdjacentDomain,
    FaceKind,
    GroupBand,
    InterfaceType,
    RowSide,
    SheetSide,
    SolutionKind,
    SurfaceEprRow,
    SurfaceEprType,
    SurfaceGroup,
    VolumeGroup,
    VolumeKind,
)
from scgsim.geometry.sgb.assembly.records import TopoFace, Topology
from scgsim.geometry.sgb.emission.measures import planned_face_area_um2, rounded
from scgsim.geometry.sgb.emission.plan import BrepPlan
from scgsim.geometry.sgb.errors import SgbError
from scgsim.geometry.sgb.model import ConformalModel
from scgsim.geometry.sgb.records.labels import Domain
from scgsim.geometry.sgb.routes.naming import (
    SHEET,
    interface_type_of,
    net_of_surface,
    sheet_face_sides,
    sheet_row_id,
)
from scgsim.geometry.sgb.routes.naming import face_kind as naming_face_kind

SURFACE_EPR_TYPES = (
    "MA",
    "MS",
    "SA",
)  # interface types that are Surface-EPR contributions


def face_kind(face: TopoFace) -> FaceKind:
    """The face kind of a planned face (``routes.naming.face_kind``): the one a group name carries, or ``port``."""
    return cast(
        FaceKind,
        naming_face_kind(
            face.neg_domain, face.pos_domain, face.owner, vertical=face.kind == "wall"
        ),
    )


def group_members(plan: BrepPlan) -> dict[str, list[str]]:
    """The written faces or solids of each primary group, by planned id, in the plan's order."""
    ids = {2: plan.face_ids, 3: plan.solid_ids}
    return {
        group.name: [ids[group.dim][i] for i in group.elements] for group in plan.groups
    }


def surface_group_records(
    model: ConformalModel, plan: BrepPlan
) -> dict[str, SurfaceGroup]:
    """Every primary surface group of the plan with the facts derived from its member faces."""
    topology = model.topology
    records: dict[str, SurfaceGroup] = {}
    for group in plan.groups:
        if group.dim != 2:
            continue
        faces = [topology.faces[plan.face_ids[i]] for i in group.elements]
        interface = _group_interface_type(group.name, faces)
        records[group.name] = SurfaceGroup(
            dim=2,
            tag=group.tag,
            interface_type=interface,
            entities=_owners(interface, faces),
            net=_net(faces),
            face_kind=_group_face_kind(group.name, faces),
            adjacent_domains=_adjacent_domains(faces),
            adjacent_volumes=_adjacent_volumes(model, faces),
            band=_band(faces),
            area_um2=rounded(sum(planned_face_area_um2(topology, f) for f in faces)),
        )
    return records


def volume_group_records(
    model: ConformalModel, plan: BrepPlan
) -> dict[str, VolumeGroup]:
    """Every primary volume group: the kind (vacuum, dielectric, conductor) and material of its domain."""
    records: dict[str, VolumeGroup] = {}
    for group in plan.groups:
        if group.dim != 3:
            continue
        volumes = [model.topology.volumes[plan.solid_ids[i]] for i in group.elements]
        domain = volumes[0].domain
        records[group.name] = VolumeGroup(
            dim=3,
            tag=group.tag,
            interface_type=None,
            entities=[domain.name],
            kind=cast(VolumeKind, domain.kind),
            material=domain.material,
            net=domain.net or None,
        )
    return records


def surface_epr_rows(
    model: ConformalModel,
    surface_groups: dict[str, SurfaceGroup],
    members: dict[str, list[str]],
) -> dict[str, SurfaceEprRow]:
    """Every Surface-EPR row and the primary group that carries it.

    ``MA``, ``MS`` and ``SA`` groups carry their own row.  A Route A sheet group carries the metal-air
    row of its Entity (``MA::<entity>__sheet[::band=k]``) when a face of it has vacuum on some side, and
    the metal-substrate row (``MS::...``) when a face has a dielectric on some side; a row with no such
    face does not exist.  Both rows of one sheet group integrate the same face, once per side, so they
    name the same ``shared_face_group``.  Each row carries:

    * ``face_sides`` -- for a sheet row, face id -> ``above``, ``below`` or ``both``: the side of that
      face whose neighbour the row integrates, listing exactly the faces that have such a neighbour;
      ``None`` for every other row;
    * ``side`` -- for a sheet row, the common ``face_sides`` value, or ``mixed`` when the faces
      disagree; ``None`` for every other row;
    * ``shared_face_group`` -- the sheet group for a sheet row, else ``None``.  A consumer that treats
      an internal two-sided face (cracking it, or counting the face once) decides from this field and
      ``face_sides``;
    * ``band`` -- the band index, or ``None`` for an unbanded group.

    ``members`` are the faces of each group (:func:`group_members`).
    """
    rows: dict[str, SurfaceEprRow] = {}
    for name, record in sorted(surface_groups.items()):
        kind = record["interface_type"]
        band = record["band"]["index"] if record["band"] else None
        if kind == SHEET:
            (entity,) = record["entities"]  # a sheet group is the sheet of one Entity
            for row_type, face_sides in _face_sides_by_row_type(
                model, members[name]
            ).items():
                rows[sheet_row_id(row_type, entity, band)] = _row(
                    name, row_type, record, band, face_sides
                )
        elif kind in SURFACE_EPR_TYPES:
            rows[name] = _row(name, cast(SurfaceEprType, kind), record, band, None)
    return rows


def _face_sides_by_row_type(
    model: ConformalModel, faces: list[str]
) -> dict[SurfaceEprType, dict[str, SheetSide]]:
    """Row type -> face id -> side, over the faces of a sheet group; a row type no face has a neighbour for is absent."""
    out: dict[SurfaceEprType, dict[str, SheetSide]] = {}
    for face_id in faces:
        face = model.topology.faces[face_id]
        for row_type, side in sheet_face_sides(
            face.neg_domain, face.pos_domain
        ).items():
            out.setdefault(cast(SurfaceEprType, row_type), {})[face_id] = cast(
                SheetSide, side
            )
    return out


def _row(
    group: str,
    row_type: SurfaceEprType,
    record: SurfaceGroup,
    band: int | None,
    face_sides: dict[str, SheetSide] | None,
) -> SurfaceEprRow:
    return SurfaceEprRow(
        group=group,
        interface_type=row_type,
        entities=record["entities"],
        band=band,
        side=None if face_sides is None else _common_side(face_sides),
        face_sides=face_sides,
        shared_face_group=None if face_sides is None else group,
    )


def _common_side(face_sides: dict[str, SheetSide]) -> RowSide:
    """The side every face agrees on, else ``mixed``."""
    sides: set[RowSide] = set(face_sides.values())
    return sides.pop() if len(sides) == 1 else "mixed"


def _group_face_kind(name: str, faces: list[TopoFace]) -> FaceKind:
    """The one face kind of a group; a group that mixed two would be a naming defect, so it stops the build."""
    kinds: set[FaceKind] = {face_kind(f) for f in faces}
    if len(kinds) != 1:
        raise SgbError(f"physical group {name} mixes face kinds {sorted(kinds)}")
    return kinds.pop()


def _group_interface_type(name: str, faces: list[TopoFace]) -> InterfaceType:
    """The one interface type of a group, from the structure of its faces (never from the name)."""
    types = {interface_type_of(f.neg_domain, f.pos_domain, f.owner) for f in faces}
    found = sorted(t for t in types if t is not None)
    if len(types) != 1 or len(found) != 1:
        raise SgbError(
            f"physical group {name} mixes interface types {sorted(map(str, types))}"
        )
    return cast(InterfaceType, found[0])


def _owners(kind: str, faces: list[TopoFace]) -> list[str]:
    """The Entities a surface belongs to: the conductor, the sheet or port owner, or the solution regions it separates."""
    names: set[str] = set()
    for face in faces:
        if face.owner is not None:
            names.add(face.owner.name)
            continue
        for domain in (face.neg_domain, face.pos_domain):
            if domain is None:
                continue
            if kind in ("MA", "MS") and domain.kind != "conductor":
                continue
            names.add(domain.name)
    return sorted(names)


def _net(faces: list[TopoFace]) -> str | None:
    nets = {
        net
        for f in faces
        if (net := net_of_surface(f.neg_domain, f.pos_domain, f.owner)) is not None
    }
    return min(nets) if len(nets) == 1 else None


def _adjacent_domains(faces: list[TopoFace]) -> list[AdjacentDomain]:
    """The solution regions (vacuum or dielectric) on any side of the group's faces."""
    found: dict[str, Domain] = {}
    for face in faces:
        for domain in (face.neg_domain, face.pos_domain):
            if domain is not None and domain.kind != "conductor":
                found[domain.name] = domain
    return [
        AdjacentDomain(name=name, kind=cast(SolutionKind, d.kind), material=d.material)
        for name, d in sorted(found.items())
    ]


def _adjacent_volumes(model: ConformalModel, faces: list[TopoFace]) -> list[str]:
    emitted = model.routed.emitted_volumes
    return sorted(
        {
            v
            for f in faces
            for v in (f.neg_volume, f.pos_volume)
            if v is not None and v in emitted
        }
    )


def _band(faces: list[TopoFace]) -> GroupBand | None:
    labels = {f.bands[0] for f in faces if f.bands}
    if not labels:
        return None
    family, index = min(labels)
    return GroupBand(family=family, index=index)


def band_groups(
    topology: Topology, face_group: dict[str, str | None]
) -> dict[str, dict[int, list[str]]]:
    """Family id -> band index -> sorted primary groups, over every face that carries a band label."""
    out: dict[str, dict[int, set[str]]] = {}
    for fid, face in topology.faces.items():
        group = face_group.get(fid)
        if (
            group is None
            or not face.bands
            or (face.owner and face.owner.kind == "port")
        ):
            continue
        family, index = face.bands[0]
        out.setdefault(family, {}).setdefault(index, set()).add(group)
    return {
        family: {index: sorted(names) for index, names in sorted(bands.items())}
        for family, bands in sorted(out.items())
    }
