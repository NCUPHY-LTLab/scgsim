"""Physical-group plan of a route (design brief sections 7 and 14, stages S7 and S8).

Owns applying a route policy to the assembled topology: which volumes are emitted, the
single primary physical group of every emitted face and volume, and the unions that
backends need.  Primary groups *partition* each dimension -- an entity belongs to exactly
one -- so the unions are lists of primary group names, never overlapping Gmsh groups:

* interface-type unions: ``MA``, ``MS``, ``MM``, ``SA``, ``AA``, ``SS``, ``SHEET``, ``PORT``,
  ``EXT`` list every primary group of that type (reports aggregate by interface type; the type is
  derived from the face structure, never parsed from a name -- a Route A conductor sheet group is
  named by its bare entity id and is of type ``SHEET``); a ``SHEET`` group is also listed under ``MA``
  when a face of it has vacuum on some side and under ``MS`` when a face has a dielectric on some side,
  because those are the Surface-EPR rows it carries;
* Net unions: ``NET::<net>`` lists the primary groups of the Net's exposed conductor
  surfaces (Electrostatic terminals and grounds), whichever route produced them.

The face names, with the qualifier a band collision needs, come from
:func:`~scgsim.geometry.sgb.topology.routes.face_names.name_faces`; everything below reads those final names.
What routing decides about each face besides its name -- interface type, Net, face kind and the sides
of a sheet's Surface-EPR rows -- is a :class:`RoutedFace`, decided here once; the sidecar reads it.

Native tags are never part of this plan; they follow sorted group names and are only
recorded by emission.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from scgsim.geometry.models.refusals import internal
from scgsim.geometry.sgb.errors import SgbError
from scgsim.geometry.sgb.exact.interfaces import (
    SHEET,
    interface_type_of,
    sheet_face_sides,
)
from scgsim.geometry.sgb.topology.assembly.records import TopoFace, Topology
from scgsim.geometry.sgb.topology.routes.face_names import FaceNames, name_faces
from scgsim.geometry.sgb.topology.routes.naming import (
    banded_rows,
    face_kind,
    net_of_surface,
    volume_group,
)
from scgsim.geometry.sgb.topology.routes.policy import RoutePolicy


@dataclass
class PhysicalGroup:
    """One primary physical group: a dimension (2 faces, 3 volumes) and its entity ids."""

    dim: int
    ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RoutedFace:
    """What routing decides about one face, from the domains on its sides and its sheet owner.

    ``interface_type`` is ``MA``, ``MS``, ``MM``, ``SA``, ``AA``, ``SS``, ``EXT``, ``PORT`` or ``SHEET``
    (``None`` for a face without a name); ``net`` the Net whose exposed conductor surface the face is;
    ``face_kind`` ``top``, ``bottom``, ``sidewall``, ``sheet``, ``interface`` or ``port``; ``sheet_sides``, for
    a ``SHEET`` face, the Surface-EPR row types of the face and the side each integrates
    (:func:`~scgsim.geometry.sgb.exact.interfaces.sheet_face_sides`), empty for every other face.  The face's
    group is ``RoutedModel.face_group``.
    """

    interface_type: str | None
    net: str | None
    face_kind: str
    sheet_sides: dict[str, str]


def route_face(face: TopoFace) -> RoutedFace:
    """The facts of one face (:class:`RoutedFace`)."""
    interface = interface_type_of(face.neg_domain, face.pos_domain, face.owner)
    return RoutedFace(
        interface_type=interface,
        net=net_of_surface(face.neg_domain, face.pos_domain, face.owner),
        face_kind=face_kind(
            face.neg_domain, face.pos_domain, face.owner, vertical=face.kind == "wall"
        ),
        sheet_sides=(
            sheet_face_sides(face.neg_domain, face.pos_domain)
            if interface == SHEET
            else {}
        ),
    )


@dataclass
class RoutedModel:
    """The topology seen through one route."""

    policy: RoutePolicy
    emitted_volumes: frozenset[str]
    face_group: dict[
        str, str | None
    ]  # every face -> its primary group (None for a face without a name)
    faces: dict[str, RoutedFace]  # every face -> what routing decided about it
    groups: dict[
        str, PhysicalGroup
    ]  # primary groups of emitted entities; they partition each dimension
    unions: dict[str, list[str]]
    surface_epr_rows: dict[
        str, str
    ]  # banded Surface-EPR row ("MA::pad__top::band=2") -> the primary group that carries it
    banded_surfaces: dict[
        str, str
    ]  # banded Surface-EPR row -> the surface it is a band of ("MA::pad__top"), named without the band

    def is_emitted_face(self, face_neg: str | None, face_pos: str | None) -> bool:
        """Whether a face with these two side volumes bounds an emitted volume."""
        return any(
            v is not None and v in self.emitted_volumes for v in (face_neg, face_pos)
        )


def apply_route(topology: Topology, policy: RoutePolicy) -> RoutedModel:
    """Decide emission and group membership of every face and volume under a route."""
    emitted = frozenset(
        vid for vid, v in topology.volumes.items() if policy.is_emitted(v.domain)
    )
    names = name_faces(topology, emitted)
    face_group = names.group
    faces = {fid: route_face(face) for fid, face in topology.faces.items()}
    surfaces = _surface_groups(topology, emitted, names, faces)
    groups = _add_volume_groups(topology, emitted, surfaces.groups)
    return RoutedModel(
        policy,
        emitted,
        face_group,
        faces,
        dict(sorted(groups.items())),
        _unions(surfaces),
        dict(sorted(surfaces.epr_rows.items())),
        dict(sorted(surfaces.banded_surfaces.items())),
    )


@dataclass
class _Surfaces:
    """The primary surface groups of a route, with what the unions and the Surface-EPR rows are made of."""

    groups: dict[str, PhysicalGroup] = field(default_factory=dict)
    types: dict[str, str] = field(default_factory=dict)  # group -> interface type
    epr_rows: dict[str, str] = field(default_factory=dict)  # row -> group
    banded_surfaces: dict[str, str] = field(default_factory=dict)  # row -> surface
    # sheet group -> the row types (MA, MS) some face of it has a neighbour for
    sheet_row_types: dict[str, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    net_groups: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))


def _surface_groups(
    topology: Topology,
    emitted: frozenset[str],
    names: FaceNames,
    faces: dict[str, RoutedFace],
) -> _Surfaces:
    """Every named face that bounds an emitted volume, in its group, with its group's facts."""
    out = _Surfaces()
    for fid, face in sorted(topology.faces.items()):
        group, unbanded = names.group[fid], names.unbanded[fid]
        bounds_emitted = any(
            v is not None and v in emitted for v in (face.neg_volume, face.pos_volume)
        )
        if group is None or not bounds_emitted:
            continue
        out.groups.setdefault(group, PhysicalGroup(2)).ids.append(fid)
        routed = faces[fid]
        interface = routed.interface_type
        if interface is not None:
            out.types[group] = interface
            entity = face.owner.name if face.owner is not None else ""
            band = face.bands[0][1] if face.bands else None
            out.sheet_row_types[group].update(routed.sheet_sides)
            assert (
                unbanded is not None
            )  # a face with a group has a name without its band
            for row, surface in banded_rows(
                group, unbanded, interface, entity, band, routed.sheet_sides
            ).items():
                out.epr_rows[row] = group
                out.banded_surfaces[row] = surface
        if routed.net is not None:
            out.net_groups[f"NET::{routed.net}"].add(group)
    return out


def _add_volume_groups(
    topology: Topology, emitted: frozenset[str], groups: dict[str, PhysicalGroup]
) -> dict[str, PhysicalGroup]:
    """The groups with every emitted volume added to its own (a name may not be both a surface and a volume)."""
    for vid, volume in sorted(topology.volumes.items()):
        if vid in emitted:
            name = volume_group(volume.domain)
            group = groups.setdefault(name, PhysicalGroup(3))
            if group.dim != 3:
                raise SgbError(
                    internal(
                        f"physical-group name {name!r} names both a volume and a surface"
                    )
                )
            group.ids.append(vid)
    return groups


def _unions(surfaces: _Surfaces) -> dict[str, list[str]]:
    by_type: dict[str, list[str]] = defaultdict(list)
    for name, interface in surfaces.types.items():
        by_type[interface].append(name)
    # a Route A sheet group belongs to the MA / MS union exactly when it carries that Surface-EPR row
    for sheet in list(by_type.get(SHEET, ())):
        for row_type in surfaces.sheet_row_types[sheet]:
            by_type[row_type].append(sheet)
    unions = {kind: sorted(names) for kind, names in by_type.items()}
    unions.update(
        {name: sorted(members) for name, members in surfaces.net_groups.items()}
    )
    return dict(sorted(unions.items()))
