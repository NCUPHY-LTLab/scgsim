"""The primary group of every face, with the band qualifier (design brief section 14, naming rule in :mod:`.naming`).

Owns the two-pass naming of faces.  The first pass names each face from its two sides
(:func:`~scgsim.geometry.sgb.routes.naming.surface_group`).  A band index is local to its band family, so
when that name would put the banded faces of two families into one group, the second pass names *every band
group of those families* again with the side of the dielectric they lie on (``...__interface__top::band=1`` and
``...__interface__bottom::band=1``).  A group that still holds two families after that is an error: two concrete
requests on one face cannot both be banded.

Both passes read the sorted faces of the topology and nothing else, so the final names do not depend on
hash seeds, and every consumer (the routed model, the sidecar, the inset records)
reads the final names from the one :func:`name_faces` result.
"""

from __future__ import annotations

from scgsim.geometry.sgb.assembly.records import TopoFace, Topology
from scgsim.geometry.sgb.errors import InputError
from scgsim.geometry.sgb.routes.naming import surface_group


def name_faces(topology: Topology, emitted: frozenset[str]) -> dict[str, str | None]:
    """Face id -> its final primary group (``None`` for a face without a name).

    Only faces that bound an emitted volume can share a group, so only they decide a collision.  The unit of
    qualification is the band family: when any band group of a family is shared with another family, *every*
    band group of each family involved gains the side of its dielectric, so one family never carries two
    name forms.
    """
    faces = sorted(topology.faces.items())
    first = {fid: _name(face, qualified=False) for fid, face in faces}
    colliding = {
        family
        for families in _band_families(faces, first, emitted).values()
        if len(families) > 1
        for family in families
    }
    final = {
        fid: _name(face, qualified=_has_family(face, colliding)) for fid, face in faces
    }
    _check_one_band_family(_band_families(faces, final, emitted))
    return final


def _has_family(face: TopoFace, families: set[str]) -> bool:
    return not _is_port(face) and any(family in families for family, _ in face.bands)


def _name(face: TopoFace, *, qualified: bool) -> str | None:
    return surface_group(
        face.neg_domain,
        face.pos_domain,
        face.owner,
        face.bands,
        vertical=face.kind == "wall",
        qualified=qualified,
    )


def _band_families(
    faces: list[tuple[str, TopoFace]],
    names: dict[str, str | None],
    emitted: frozenset[str],
) -> dict[str, set[str]]:
    """Banded surface group -> the band families its emitted faces carry (a port face carries none)."""
    out: dict[str, set[str]] = {}
    for fid, face in faces:
        group = names[fid]
        if group is None or not face.bands or _is_port(face):
            continue
        if not any(
            v is not None and v in emitted for v in (face.neg_volume, face.pos_volume)
        ):
            continue
        out.setdefault(group, set()).update(family for family, _ in face.bands)
    return out


def _is_port(face: TopoFace) -> bool:
    return face.owner is not None and face.owner.kind == "port"


def _check_one_band_family(families_of: dict[str, set[str]]) -> None:
    """Fail closed when a group still holds the faces of two band families after qualifying.

    A consumer who selects one family's band by the group would select the other's faces too.  What is left
    after the qualifier is two requests on one face, whose cells carry a label of each family.
    """
    for group, families in sorted(families_of.items()):
        if len(families) > 1:
            first, second = sorted(families)[:2]
            raise InputError(
                f"physical group {group!r} would hold the faces of two band families, {first!r} and {second!r}; "
                "a band index belongs to one family, so ask for each face once in the inset requests "
                "(two requests on the same face cannot both be banded)"
            )
