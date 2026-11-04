"""Physical-group plan of a route (design brief sections 7 and 14, stages S7 and S8).

Owns applying a route policy to the assembled topology: which volumes are emitted, the
single primary physical group of every emitted face and volume, and the unions that
backends need.  Primary groups *partition* each dimension -- an entity belongs to exactly
one -- so the unions are lists of primary group names, never overlapping Gmsh groups:

* interface-type unions: ``MA``, ``MS``, ``MM``, ``SA``, ``AA``, ``SS``, ``SHEET``, ``PORT``,
  ``EXT`` list every primary group of that type (reports aggregate by interface type; the type is
  derived from the face structure, never parsed from a name -- a Route A conductor sheet group is
  named by its bare entity id and is of type ``SHEET``); a ``SHEET`` group is also listed under ``MA``
  and ``MS``, because it carries both Surface-EPR rows;
* Net unions: ``NET::<net>`` lists the primary groups of the Net's exposed conductor
  surfaces (Electrostatic terminals and grounds), whichever route produced them.

Native tags are never part of this plan; they follow sorted group names and are only
recorded by emission.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from scgsim.geometry.sgb.assembly.records import Topology
from scgsim.geometry.sgb.errors import SgbError
from scgsim.geometry.sgb.routes.naming import (
    SHEET,
    interface_type_of,
    net_of_surface,
    surface_epr_row_ids,
    surface_group,
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
    ]  # Surface-EPR row ("MA::pad::top::band=2") -> the primary group that carries it

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
    face_group: dict[str, str | None] = {}
    groups: dict[str, PhysicalGroup] = {}
    # surface group -> its interface type, from the face structure (never from the name)
    group_types: dict[str, str] = {}
    epr_rows: dict[str, str] = {}
    net_groups: dict[str, set[str]] = defaultdict(set)
    for fid, face in sorted(topology.faces.items()):
        group = surface_group(
            face.neg_domain,
            face.pos_domain,
            face.owner,
            face.bands,
            vertical=face.kind == "wall",
        )
        face_group[fid] = group
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
            for row in surface_epr_row_ids(group, interface, entity, band):
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
        _unions(group_types, net_groups),
        dict(sorted(epr_rows.items())),
    )


def _unions(
    group_types: dict[str, str], net_groups: dict[str, set[str]]
) -> dict[str, list[str]]:
    by_type: dict[str, list[str]] = defaultdict(list)
    for name, interface in group_types.items():
        by_type[interface].append(name)
    # a Route A sheet group carries both the metal-air and the metal-substrate Surface-EPR row, so it
    # belongs to the MA and MS unions as well as to SHEET
    sheets = list(by_type.get(SHEET, ()))
    if sheets:
        by_type["MA"].extend(sheets)
        by_type["MS"].extend(sheets)
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
