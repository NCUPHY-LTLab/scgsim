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
:func:`~scgsim.geometry.sgb.routes.face_names.name_faces`; everything below reads those final names.

Native tags are never part of this plan; they follow sorted group names and are only
recorded by emission.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from scgsim.geometry.sgb.assembly.records import Topology
from scgsim.geometry.sgb.errors import SgbError
from scgsim.geometry.sgb.routes.face_names import name_faces
from scgsim.geometry.sgb.routes.naming import (
    SHEET,
    interface_type_of,
    net_of_surface,
    sheet_face_sides,
    surface_epr_row_ids,
    volume_group,
)
from scgsim.geometry.sgb.routes.policy import RoutePolicy


@dataclass
class PhysicalGroup:
    """One primary physical group: a dimension (2 faces, 3 volumes) and its entity ids."""

    dim: int
    ids: list[str] = field(default_factory=list)


@dataclass
class RoutedModel:
    """The topology seen through one route."""

    policy: RoutePolicy
    emitted_volumes: frozenset[str]
    face_group: dict[
        str, str | None
    ]  # every face -> its primary group (None for a face without a name)
    groups: dict[
        str, PhysicalGroup
    ]  # primary groups of emitted entities; they partition each dimension
    unions: dict[str, list[str]]
    surface_epr_rows: dict[
        str, str
    ]  # Surface-EPR row ("MA::pad__top::band=2") -> the primary group that carries it

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
    face_group = name_faces(topology, emitted)
    groups: dict[str, PhysicalGroup] = {}
    # surface group -> its interface type, from the face structure (never from the name)
    group_types: dict[str, str] = {}
    epr_rows: dict[str, str] = {}
    # sheet group -> the row types (MA, MS) some face of it has a neighbour for
    sheet_row_types: dict[str, set[str]] = defaultdict(set)
    net_groups: dict[str, set[str]] = defaultdict(set)
    for fid, face in sorted(topology.faces.items()):
        group = face_group[fid]
        bounds_emitted = any(
            v is not None and v in emitted for v in (face.neg_volume, face.pos_volume)
        )
        if group is None or not bounds_emitted:
            continue
        groups.setdefault(group, PhysicalGroup(2)).ids.append(fid)
        interface = interface_type_of(face.neg_domain, face.pos_domain, face.owner)
        if interface is not None:
            group_types[group] = interface
            entity = face.owner.name if face.owner is not None else ""
            band = face.bands[0][1] if face.bands else None
            sheet_sides = (
                sheet_face_sides(face.neg_domain, face.pos_domain)
                if interface == SHEET
                else None
            )
            if sheet_sides is not None:
                sheet_row_types[group].update(sheet_sides)
            for row in surface_epr_row_ids(group, interface, entity, band, sheet_sides):
                epr_rows[row] = group
        net = net_of_surface(face.neg_domain, face.pos_domain, face.owner)
        if net is not None:
            net_groups[f"NET::{net}"].add(group)
    for vid, volume in sorted(topology.volumes.items()):
        if vid in emitted:
            name = volume_group(volume.domain)
            group = groups.setdefault(name, PhysicalGroup(3))
            if group.dim != 3:
                raise SgbError(
                    f"physical-group name {name!r} names both a volume and a surface"
                )
            group.ids.append(vid)
    _check_refinement_hosts(topology, emitted)
    ordered = dict(sorted(groups.items()))
    return RoutedModel(
        policy,
        emitted,
        face_group,
        ordered,
        _unions(group_types, sheet_row_types, net_groups),
        dict(sorted(epr_rows.items())),
    )


def _unions(
    group_types: dict[str, str],
    sheet_row_types: dict[str, set[str]],
    net_groups: dict[str, set[str]],
) -> dict[str, list[str]]:
    by_type: dict[str, list[str]] = defaultdict(list)
    for name, interface in group_types.items():
        by_type[interface].append(name)
    # a Route A sheet group belongs to the MA / MS union exactly when it carries that Surface-EPR row
    for sheet in list(by_type.get(SHEET, ())):
        for row_type in sheet_row_types[sheet]:
            by_type[row_type].append(sheet)
    unions = {kind: sorted(names) for kind, names in by_type.items()}
    unions.update({name: sorted(members) for name, members in net_groups.items()})
    return dict(sorted(unions.items()))


def _check_refinement_hosts(topology: Topology, emitted: frozenset[str]) -> None:
    """A refinement curve must lie in a face of an emitted volume: it is an error, never dropped silently."""
    for face in topology.faces.values():
        if not face.refinement_edges:
            continue
        if not any(
            v is not None and v in emitted for v in (face.neg_volume, face.pos_volume)
        ):
            raise SgbError(
                f"refinement curve(s) {', '.join(sorted(face.refinement_edges))} lie in face {face.id} "
                "of no emitted volume"
            )
