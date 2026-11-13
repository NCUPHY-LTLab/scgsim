"""Physical groups and Surface-EPR rows as the sidecar describes them (design brief sections 7 and 14).

Owns the facts a solver adapter needs about each primary physical group beyond its members: which
Entity or Entities it belongs to, the Net, whether it is a top, bottom, sidewall or sheet surface,
the volumes and solution regions on its sides, its inset band, its area -- so a group's record
(owner ids, adjacent solution volumes, face kind, Net) can be rebuilt without reading geometry.

Primary groups partition each dimension.  A face or volume names its group in its own record, so a group
record lists no members: readers compute them (``scgsim.contracts.sgb_sidecar.views.group_members``).

Routing decided each face's interface type, Net, face kind and sheet sides (``topology.routes.groups.RoutedFace``);
a group's facts are read from those of its faces, never derived again from the domains or the name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from scgsim.contracts.sgb_sidecar.records import (
    AdjacentDomain,
    FaceKind,
    GroupBand,
    HiddenContact,
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
from scgsim.geometry.models.refusals import internal
from scgsim.geometry.sgb.errors import SgbError
from scgsim.geometry.sgb.exact.interfaces import SHEET, SURFACE_EPR_TYPES
from scgsim.geometry.sgb.exact.labels import Domain
from scgsim.geometry.sgb.model import ConformalModel
from scgsim.geometry.sgb.output.emission.measures import planned_face_area_um2, rounded
from scgsim.geometry.sgb.output.emission.plan import BrepPlan
from scgsim.geometry.sgb.topology.assembly.records import TopoFace, Topology
from scgsim.geometry.sgb.topology.routes.groups import RoutedFace, RoutedModel
from scgsim.geometry.sgb.topology.routes.naming import sheet_row_id


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
        routed = [model.routed.faces[f.id] for f in faces]
        interface = _group_interface_type(group.name, routed)
        records[group.name] = SurfaceGroup(
            dim=2,
            tag=group.tag,
            interface_type=interface,
            entities=_owners(interface, faces),
            net=_net(routed),
            face_kind=_group_face_kind(group.name, routed),
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
    * ``band`` -- the band index, or ``None`` for an unbanded group (the row's Entities are its group's);
    * ``banded_surface`` -- for a band row, the surface it is a band of, named as the row would be without its
      band (``MA::pad__top`` for ``MA::pad__top::band=2``, ``MA::pad__sheet`` for ``MA::pad__sheet::band=2``):
      the routed model's name, shared by every band row of that surface; ``None`` for an unbanded row.

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
                row_id = sheet_row_id(row_type, entity, band)
                rows[row_id] = _row(
                    name, row_type, record, band, face_sides, _surface(model, row_id)
                )
        elif kind in SURFACE_EPR_TYPES:
            rows[name] = _row(
                name,
                cast(SurfaceEprType, kind),
                record,
                band,
                None,
                _surface(model, name),
            )
    return rows


def _surface(model: ConformalModel, row_id: str) -> str | None:
    """The surface a band row is a band of, as routing named it; ``None`` for an unbanded row."""
    return model.routed.banded_surfaces.get(row_id)


def _face_sides_by_row_type(
    model: ConformalModel, faces: list[str]
) -> dict[SurfaceEprType, dict[str, SheetSide]]:
    """Row type -> face id -> side, over the faces of a sheet group; a row type no face has a neighbour for is absent."""
    out: dict[SurfaceEprType, dict[str, SheetSide]] = {}
    for face_id in faces:
        for row_type, side in model.routed.faces[face_id].sheet_sides.items():
            out.setdefault(cast(SurfaceEprType, row_type), {})[face_id] = cast(
                SheetSide, side
            )
    return out


def _row(  # noqa: PLR0913 - the group, the row type and record, the band, the sheet sides and the surface
    group: str,
    row_type: SurfaceEprType,
    record: SurfaceGroup,
    band: int | None,
    face_sides: dict[str, SheetSide] | None,
    banded_surface: str | None,
) -> SurfaceEprRow:
    if (band is None) != (banded_surface is None):
        raise SgbError(
            internal(
                f"row of group {group!r} has band {band} but banded surface {banded_surface!r}"
            )
        )
    return SurfaceEprRow(
        group=group,
        interface_type=row_type,
        band=band,
        banded_surface=banded_surface,
        side=None if face_sides is None else _common_side(face_sides),
        face_sides=face_sides,
        shared_face_group=None if face_sides is None else group,
    )


def _common_side(face_sides: dict[str, SheetSide]) -> RowSide:
    """The side every face agrees on, else ``mixed``."""
    sides: set[RowSide] = set(face_sides.values())
    return sides.pop() if len(sides) == 1 else "mixed"


def _group_face_kind(name: str, faces: list[RoutedFace]) -> FaceKind:
    """The one face kind of a group; a group that mixed two would be a naming defect, so it stops the build."""
    kinds = {f.face_kind for f in faces}
    if len(kinds) != 1:
        raise SgbError(
            internal(f"physical group {name} mixes face kinds {sorted(kinds)}")
        )
    return cast(FaceKind, kinds.pop())


def _group_interface_type(name: str, faces: list[RoutedFace]) -> InterfaceType:
    """The one interface type of a group, from the routed facts of its faces (never from the name)."""
    types = {f.interface_type for f in faces}
    found = sorted(t for t in types if t is not None)
    if len(types) != 1 or len(found) != 1:
        raise SgbError(
            internal(
                f"physical group {name} mixes interface types {sorted(map(str, types))}"
            )
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


def _net(faces: list[RoutedFace]) -> str | None:
    nets = {f.net for f in faces if f.net is not None}
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


@dataclass(frozen=True)
class FamilyBands:
    """What lies in each band of one family: band index -> the written groups, and the contacts no group carries."""

    groups: dict[int, list[str]]
    hidden_contacts: dict[int, list[HiddenContact]]


def band_groups(topology: Topology, routed: RoutedModel) -> dict[str, FamilyBands]:
    """Family id -> what lies in each band, over every band label of every named face that carries one.

    A face that bounds a written volume is in its group.  A named face that bounds none is a contact between two
    conductors in a route that writes no conductor volume (Route B): no group carries it, so it is a hidden
    contact, by its two conductors and face kind.  A port face carries the nominal bands of the place it lies
    in, and is in no band group.
    """
    groups: dict[str, dict[int, set[str]]] = {}
    contacts: dict[str, dict[int, set[tuple[str, str, str]]]] = {}
    for fid, face in topology.faces.items():
        group = routed.face_group.get(fid)
        routed_face = routed.faces[fid]
        if group is None or not face.bands or routed_face.face_kind == "port":
            continue
        written = routed.is_emitted_face(face.neg_volume, face.pos_volume)
        contact = None if written else _hidden_contact(fid, face, routed_face)
        for family, index in face.bands:
            if contact is None:
                groups.setdefault(family, {}).setdefault(index, set()).add(group)
            else:
                contacts.setdefault(family, {}).setdefault(index, set()).add(contact)
    return {
        family: FamilyBands(
            groups={i: sorted(g) for i, g in sorted(groups.get(family, {}).items())},
            hidden_contacts={
                i: [
                    HiddenContact(entities=[a, b], face_kind=cast(FaceKind, kind))
                    for a, b, kind in sorted(found)
                ]
                for i, found in sorted(contacts.get(family, {}).items())
            },
        )
        for family in sorted({*groups, *contacts})
    }


def _hidden_contact(
    face_id: str, face: TopoFace, routed: RoutedFace
) -> tuple[str, str, str]:
    """The two conductors and the face kind of a named face no written volume bounds."""
    if (
        routed.interface_type != "MM"
        or face.neg_domain is None
        or face.pos_domain is None
    ):
        raise SgbError(
            internal(
                f"the banded face {face_id} bounds no written volume and is no metal-metal contact"
            )
        )
    first, second = sorted((face.neg_domain.name, face.pos_domain.name))
    return first, second, routed.face_kind
