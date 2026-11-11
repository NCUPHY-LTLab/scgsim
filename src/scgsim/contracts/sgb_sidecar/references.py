"""The references of a v2 sidecar that resolve against a table in the same file, checked after its structure.

Owns the cross-section checks v2 allows: every reference names a record that exists, of the kind it
needs, and every table v2 keys by written shape covers them all.  ``binding.ids_by_tag`` is the table of
planned ids: every planned id a section names must be in it, in its dimension, and every native tag must
name the id written beside it.  ``physical_groups`` is the table of groups by dimension, each tag once per
dimension: a reference to a surface group must name a group of dimension 2, one to a volume group a group
of dimension 3.  ``faces`` describes exactly the written faces, and ``volumes`` every planned volume, the
written solids exactly those it marks emitted.  ``lineage.carriers`` is the table of carriers; ``counts``
and ``expected`` must have the binding's sizes.

v2 states some facts twice (a face's group and the group's members, a port's faces and its group's), and
the two copies are not compared here: v3 writes each fact once (#170).

v2 also refers to things it keeps no table of, so these stay unchecked until slice C of #170 gives each one
a table: Entity ids (volume entities, face sources and membership, group and row entities, port hosts,
mesh-feature sources, wall columns), cut-family ids (group bands, refinement-edge and carrier families) and
members, and request ids.  The band groups of an inset family are unchecked too, because v2 writes groups
there that do not exist (see :func:`_group_lists`).
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from .errors import SidecarError
from .records import Sidecar

_DIMENSIONS = ("vertex", "edge", "face", "volume")
_SHOWN = 5  # unresolved references named in the error; the rest are counted


def require_references(sidecar: Sidecar) -> None:
    """Raise :class:`SidecarError` naming the first references of ``sidecar`` that do not resolve."""
    index = _Index.of(sidecar)
    problems = [
        *_binding_sizes(sidecar, index),
        *_group_members(sidecar, index),
        *_mesh_feature_edges(sidecar, index),
        *_embeddings(sidecar, index),
        *_planned_faces_and_volumes(sidecar, index),
        *_surface_epr_rows(sidecar, index),
        *_ports(sidecar, index),
        *_lineage(sidecar, index),
        *_group_lists(sidecar, index),
        *_expected(sidecar, index),
    ]
    if problems:
        more = f" (and {len(problems) - _SHOWN} more)" if len(problems) > _SHOWN else ""
        raise SidecarError(
            f"sidecar references do not resolve: {'; '.join(problems[:_SHOWN])}{more}"
        )


@dataclass(frozen=True)
class _Index:
    """The tables of one sidecar that references resolve against."""

    ids: tuple[list[str], ...]  # planned ids by dimension, in native-tag order
    known: tuple[frozenset[str], ...]
    groups: dict[int, frozenset[str]]  # primary group names by dimension, 2 and 3
    volumes: frozenset[str]

    @classmethod
    def of(cls, sidecar: Sidecar) -> "_Index":
        """The tables of ``sidecar``."""
        by_tag = sidecar["binding"]["ids_by_tag"]
        ids = (by_tag["0"], by_tag["1"], by_tag["2"], by_tag["3"])
        groups = sidecar["physical_groups"]
        return cls(
            ids,
            tuple(frozenset(dimension) for dimension in ids),
            {
                dim: frozenset(n for n, g in groups.items() if g["dim"] == dim)
                for dim in (2, 3)
            },
            frozenset(sidecar["volumes"]),
        )

    def unknown(self, dimension: int, ids: Iterable[str], where: str) -> Iterator[str]:
        """A problem for each of ``ids`` that is no planned id of ``dimension``."""
        for planned in ids:
            if planned not in self.known[dimension]:
                yield f"{where} names {_DIMENSIONS[dimension]} {planned!r}, which binding.ids_by_tag lacks"

    def mistagged(
        self, dimension: int, ids: list[str], tags: list[int], where: str
    ) -> Iterator[str]:
        """A problem for each native tag that does not name the planned id beside it."""
        if len(ids) != len(tags):
            yield f"{where} has {len(ids)} ids but {len(tags)} native tags"
            return
        table = self.ids[dimension]
        for planned, tag in zip(ids, tags, strict=True):
            if not (1 <= tag <= len(table)) or table[tag - 1] != planned:
                yield f"{where}: native tag {tag} is not {_DIMENSIONS[dimension]} {planned!r}"

    def unknown_groups(
        self, dimension: int, names: Iterable[str | None], where: str
    ) -> Iterator[str]:
        """A problem for each name that is no primary group of ``dimension`` (``None`` is no group, and resolves)."""
        for name in names:
            if name is None or name in self.groups[dimension]:
                continue
            if any(name in groups for groups in self.groups.values()):
                yield f"{where} names the group {name!r}, which is not a {_DIMENSIONS[dimension]} group"
            else:
                yield f"{where} names the group {name!r}, which physical_groups lacks"


def _binding_sizes(sidecar: Sidecar, index: _Index) -> Iterator[str]:
    """Each planned id once per dimension, as many as ``counts`` says were written, and as many planned
    volumes as ``counts.volumes_total``."""
    counts = sidecar["counts"]
    if counts["volumes_total"] != len(sidecar["volumes"]):
        yield f"counts.volumes_total is {counts['volumes_total']}, volumes has {len(sidecar['volumes'])}"
    written = (counts["vertices"], counts["edges"], counts["faces"], counts["solids"])
    for dimension, (ids, count) in enumerate(zip(index.ids, written, strict=True)):
        if len(index.known[dimension]) != len(ids):
            yield f"binding.ids_by_tag['{dimension}'] names a planned id more than once"
        if count != len(ids):
            yield f"counts has {count} of dimension {dimension}, binding.ids_by_tag['{dimension}'] {len(ids)}"


def _group_members(sidecar: Sidecar, index: _Index) -> Iterator[str]:
    """Each group's members and, for a surface group, the written volumes beside it; each tag once per dimension."""
    named: dict[tuple[int, int], str] = {}
    for name, group in sidecar["physical_groups"].items():
        where = f"physical group {name!r}"
        yield from index.mistagged(
            group["dim"], group["ids"], group["native_tags"], where
        )
        if group["dim"] == 2:
            yield from index.unknown(3, group["adjacent_volumes"], where)
        key = (group["dim"], group["tag"])
        if key in named:
            yield f"{where} has the tag {group['tag']} of physical group {named[key]!r}"
        named.setdefault(key, name)


def _mesh_feature_edges(sidecar: Sidecar, index: _Index) -> Iterator[str]:
    for position, feature in enumerate(sidecar["mesh_features"]):
        where = f"mesh_features[{position}] ({feature['kind']})"
        yield from index.mistagged(1, feature["edges"], feature["native_tags"], where)


def _embeddings(sidecar: Sidecar, index: _Index) -> Iterator[str]:
    for position, face in enumerate(sidecar["embedded_faces"]):
        where = f"embedded_faces[{position}]"
        yield from index.unknown(2, [face["face"]], where)
        yield from index.unknown(3, [face["host_volume"]], where)
        yield from index.unknown_groups(2, [face["group"]], where)
    for position, edge in enumerate(sidecar["refinement_edges"]):
        where = f"refinement_edges[{position}]"
        yield from index.unknown(1, [edge["edge"]], where)
        yield from index.unknown(2, [edge["host_face"]], where)


def _planned_faces_and_volumes(sidecar: Sidecar, index: _Index) -> Iterator[str]:
    """``faces`` describes exactly the written faces; ``volumes`` every planned volume, the written solids
    exactly the emitted ones."""
    yield from index.unknown(2, sidecar["faces"], "faces")
    for face_id in index.ids[2]:
        if face_id not in sidecar["faces"]:
            yield f"binding.ids_by_tag['2'] names {face_id!r}, which faces lacks"
    for face_id, face in sidecar["faces"].items():
        yield from index.unknown_groups(2, [face["group"]], f"faces[{face_id!r}]")
        for side in (face["negative_volume"], face["positive_volume"]):
            if side is not None and side not in index.volumes:
                yield f"faces[{face_id!r}] names the volume {side!r}, which volumes lacks"
    for solid in index.ids[3]:
        if solid not in index.volumes:
            yield f"binding.ids_by_tag['3'] names {solid!r}, which volumes lacks"
    for volume_id, volume in sidecar["volumes"].items():
        yield from index.unknown_groups(3, [volume["group"]], f"volumes[{volume_id!r}]")
        if volume["emitted"] != (volume_id in index.known[3]):
            state = "emitted" if volume["emitted"] else "not emitted"
            yield f"volumes[{volume_id!r}] is {state}, which binding.ids_by_tag['3'] contradicts"


def _surface_epr_rows(sidecar: Sidecar, index: _Index) -> Iterator[str]:
    for row_id, row in sidecar["surface_epr_rows"].items():
        where = f"surface_epr_rows[{row_id!r}]"
        yield from index.unknown_groups(
            2, [row["group"], row["shared_face_group"]], where
        )
        yield from index.unknown(2, row["face_sides"] or {}, where)


def _ports(sidecar: Sidecar, index: _Index) -> Iterator[str]:
    for port in sidecar["ports"]:
        where = f"port {port['port_sheet_id']!r}"
        yield from index.unknown_groups(2, [port["group"]], where)
        yield from index.unknown(2, port["faces"], where)
        yield from index.unknown(
            3, [*port["host_volumes"], *port["adjacent_volumes"]], where
        )


def _lineage(sidecar: Sidecar, index: _Index) -> Iterator[str]:
    lineage = sidecar["lineage"]
    yield from index.unknown(1, lineage["edges"], "lineage.edges")
    yield from index.unknown(1, lineage["vertical_edges"], "lineage.vertical_edges")
    yield from index.unknown(2, lineage["faces"], "lineage.faces")
    yield from index.unknown(2, lineage["membership"], "lineage.membership")
    carriers = lineage["carriers"]
    for edge_id, sources in lineage["edges"].items():
        for carrier in sources:
            if carrier not in carriers:
                yield f"lineage.edges[{edge_id!r}] names the carrier {carrier!r}, which lineage.carriers lacks"
    for position, refinement in enumerate(sidecar["refinements"]):
        yield from index.unknown(1, refinement["edges"], f"refinements[{position}]")
    yield from index.unknown(2, sidecar["curved_wall_faces"], "curved_wall_faces")


def _group_lists(sidecar: Sidecar, index: _Index) -> Iterator[str]:
    """Unions name primary surface groups: interface types are surface types, and a Net's union lists its
    exposed conductor surfaces.

    The ``bands`` of an inset family do not always: SGB lists the band group of every face that carries the
    band, emitted or not, so in Route B a family also names the ``MM`` group of a metal-metal contact, which
    has no physical group (``cell11_bumps_flip_chip``, ``cell22_tsv_signal``, ``cell23_airbridge``).
    """
    unions = sidecar["unions"]
    for kind in ("interface_types", "nets"):
        for key, names in unions[kind].items():
            yield from index.unknown_groups(2, names, f"unions.{kind}[{key!r}]")


def _expected(sidecar: Sidecar, index: _Index) -> Iterator[str]:
    """One expectation per written shape, and boundaries that name written shapes."""
    expected = sidecar["expected"]
    sizes = {
        "vertices_um": 0,
        "edge_vertices": 1,
        "face_edges": 2,
        "face_area_um2": 2,
        "solid_faces": 3,
        "solid_volume_um3": 3,
    }
    for name, dimension in sizes.items():
        if len(expected[name]) != len(index.ids[dimension]):
            yield f"expected.{name} has {len(expected[name])} entries for {len(index.ids[dimension])} shapes"
    boundaries = {"edge_vertices": 0, "face_edges": 1, "solid_faces": 2}
    for name, dimension in boundaries.items():
        written = len(index.ids[dimension])
        if any(tag > written for tags in expected[name] for tag in tags):
            yield f"expected.{name} names a tag beyond the {written} written shapes of dimension {dimension}"
