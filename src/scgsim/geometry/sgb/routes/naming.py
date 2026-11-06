"""Semantic names of surfaces and volumes: decision D5, per-Entity attribution (design brief section 14).

Owns the primary physical-group names.  Surfaces are attributed **per Entity**, so every
Entity's Surface-EPR contribution is separable; reports aggregate by interface type and
Net is used only where a backend needs electrical identity.

**A bare entity id names the object itself; a name with an interface prefix names a face between two
things.**  Gmsh's Type column already says whether a group is a volume or a surface, so no prefix
restates the dimension:

* ``<entity>`` -- a volume (also the interior of a Route C conductor Entity), or, for a Route A
  conductor sheet, the sheet itself: ``<entity>[::band=k]``, whose two sides are the metal-air and the
  metal-substrate surface of that Entity wherever a vacuum or a dielectric lies against them (it carries no
  face kind, because it is always a ``sheet``);
* ``MA::<entity>__<face kind>`` / ``MS::<entity>__<face kind>`` -- a conductor surface against vacuum
  (metal-air) or against a dielectric (metal-substrate); ``MM::<a>|<b>__<face kind>`` between two
  conductor Entities;
* ``SA::<dielectric>|<vacuum>__<face kind>``, ``AA::<a>|<b>__...``, ``SS::<a>|<b>__...`` -- interfaces
  between solution regions (substrate-air, air-air, substrate-substrate);
* ``EXT::<domain>__<face kind>`` -- the model boundary;
* ``PORT::<port sheet id>`` -- a lumped-port sheet (always one sheet, so it carries no face kind).

The interface prefixes stay because they carry the physics: they say which Surface-EPR loss parameters
apply to the face.  An entity id equal to a prefix (:data:`RESERVED_PREFIXES`), or containing ``::`` or
``__``, would make a name ambiguous, so :func:`check_entity_names` rejects it.

The interface type of a face (``MA``, ``MS`` ..., or ``SHEET`` for a Route A conductor sheet) is never
read back from a name: :func:`interface_type_of` derives it from the domains and the sheet owner.

**Face kind** (:data:`FACE_KINDS`) is part of the identity of a prefixed name, attached to the entity part
with a double underscore (``MA::D1_pad__bottom``) -- every other separator is ``::``, and ``__`` sets the
face kind apart from the single underscores common in entity ids (``D1_pad``, ``ground_0000``).  So one
primary group never mixes physically different interfaces and a consumer can filter by the ``face_kind`` field
of the sidecar without reading geometry:

* ``top`` / ``bottom`` -- a horizontal face with a conductor below / above it (the conductor's top or
  bottom); at the model boundary, the top or bottom of the domain;
* ``sidewall`` -- a vertical face (a wall);
* ``sheet`` -- a zero-thickness conductor face (Route A);
* ``interface`` -- a horizontal face between two solution regions, or two conductors.

A face inside an inset band carries ``::band=k`` (the innermost band index of its label);
band seams stop at the boundary of a port, so a port is one face and carries no band.  Names never carry
native tags and never carry a height.

**Qualifier, only on a band collision.**  A band index is local to one band family.  A conductor face is
named by its entity and side, which already identify the support; an ``interface`` support is a pair of
solution regions, and the same pair meets at several levels, so two families can end up with the same
``<base>__interface::band=k``.  Only then does every band group of each family involved (all its bands, so
one family never carries two name forms) gain the side of the dielectric its faces lie on, again with ``__``: ``SA::substrate|vacuum__interface__top::band=1`` (the dielectric is below
the face) and ``...__interface__bottom::band=1`` (it is above).  The pass is
:func:`~scgsim.geometry.sgb.routes.face_names.name_faces`'s, after the first naming; the ``face_kind`` of
the group stays ``interface``.  A group that still holds two families after qualifying is an error.

Surface-EPR rows keep the prefixed form for a sheet: its group ``pad::band=1`` carries the row
``MA::pad__sheet::band=1`` when a face of the group has vacuum on some side, and ``MS::pad__sheet::band=1``
when a face has a dielectric on some side (see :func:`sheet_row_id` and :func:`sheet_face_sides`).  A sheet
with vacuum on both sides has no ``MS`` row.
"""

from __future__ import annotations

from collections.abc import Iterable

from scgsim.geometry.sgb.errors import InputError
from scgsim.geometry.sgb.records.labels import Domain, SheetOwner

RESERVED_PREFIXES = ("MA", "MS", "MM", "SA", "AA", "SS", "EXT", "PORT")
"""The interface prefixes of a prefixed group name; no entity id may equal one of them."""
SHEET = "SHEET"
"""Interface type of a Route A conductor sheet (a record field and a union key, never part of a group name)."""
SHEET_ROW_NEIGHBOURS = {"MA": "vacuum", "MS": "dielectric"}
"""The Surface-EPR row types of a sheet face and the kind of neighbour each one integrates."""
FACE_KINDS = ("top", "bottom", "sidewall", "sheet", "interface")
"""The face kinds a primary surface group name can carry (a port face is ``port`` and is not in a name)."""
FACE_KIND_SEPARATOR = "__"
"""Attaches the face kind (and a qualifier) to the entity part of a name; an entity id may not contain it."""


def surface_group(  # noqa: PLR0913 - the two sides, the owner, the bands, the orientation and the qualifier
    neg: Domain | None,
    pos: Domain | None,
    owner: SheetOwner | None,
    bands: tuple[tuple[str, int], ...] = (),
    *,
    vertical: bool = False,
    qualified: bool = False,
) -> str | None:
    """The single primary physical-group name of a face from the domains on its two sides.

    ``vertical`` says the face is a wall (``sidewall``); otherwise it is horizontal.  ``qualified`` adds
    :func:`interface_side` to the name of an ``interface`` face: the second naming of a band group that two
    band families would share (see the module docstring).
    """
    named = _prefix_and_entity(neg, pos, owner)
    if named is None:
        return None
    prefix, entity = named
    if prefix == "PORT":
        return f"PORT::{entity}"  # one sheet: no face kind, no band
    if prefix == SHEET:
        return _banded(entity, bands)  # the entity id; its face kind is always sheet
    kind = face_kind(neg, pos, owner, vertical=vertical)
    name = f"{prefix}::{entity}{FACE_KIND_SEPARATOR}{kind}"
    if qualified and kind == "interface":
        name = f"{name}{FACE_KIND_SEPARATOR}{interface_side(neg, pos)}"
    return _banded(name, bands)


def interface_side(neg: Domain | None, pos: Domain | None) -> str:
    """Which side of the first-named region of an interface face the face lies on: ``top`` or ``bottom``.

    ``top`` when that region (the dielectric of an ``SA`` face) is below the face, so the face is its top;
    ``bottom`` when it is above.  Both domains must exist.
    """
    if neg is None or pos is None:
        raise ValueError("an interface face has a domain on each side")
    first, _ = _ordered_pair(neg, pos)
    return "top" if first == neg else "bottom"


def _banded(name: str, bands: tuple[tuple[str, int], ...]) -> str:
    return f"{name}::band={bands[0][1]}" if bands else name


def face_kind(
    neg: Domain | None,
    pos: Domain | None,
    owner: SheetOwner | None,
    *,
    vertical: bool = False,
) -> str:
    """The face kind of a face (see the module docstring): ``top``, ``bottom``, ``sidewall``, ``sheet``, ``interface`` or ``port``."""
    if owner is not None:
        return "port" if owner.kind == "port" else "sheet"
    if vertical:
        return "sidewall"
    if neg is None:
        return "bottom"  # the lowest face of the domain above it
    if pos is None:
        return "top"
    conductors = [d for d in (neg, pos) if d.kind == "conductor"]
    if len(conductors) == 1:
        return "top" if neg.kind == "conductor" else "bottom"
    return "interface"


def _prefix_and_entity(
    neg: Domain | None, pos: Domain | None, owner: SheetOwner | None
) -> tuple[str, str] | None:
    """The interface type of a face and the entity part of its name (``("MA", "D1_pad")``), or ``None`` for no name.

    A Route A conductor sheet is ``("SHEET", <entity>)`` and a port ``("PORT", <port sheet id>)``.
    """
    if owner is not None:
        return ("PORT" if owner.kind == "port" else SHEET, owner.name)
    if neg is None:
        return None if pos is None else ("EXT", pos.name)
    if pos is None:
        return ("EXT", neg.name)
    conductors = [d for d in (neg, pos) if d.kind == "conductor"]
    if len(conductors) == 2:
        first, second = _ordered_pair(neg, pos)
        return ("MM", f"{first.name}|{second.name}")
    if conductors:
        metal = conductors[0]
        other = pos if metal == neg else neg
        return ("MA" if other.kind == "vacuum" else "MS", metal.name)
    return _solution_interface(neg, pos)


def _ordered_pair(a: Domain, b: Domain) -> tuple[Domain, Domain]:
    """The two domains of an interface in the order of its name: dielectric before vacuum, then by name."""
    first, second = sorted((a, b), key=lambda d: (d.kind == "vacuum", d.name))
    return first, second


def _solution_interface(a: Domain, b: Domain) -> tuple[str, str]:
    """Substrate-air, air-air or substrate-substrate, dielectric first."""
    first, second = _ordered_pair(a, b)
    kinds = {a.kind, b.kind}
    kind = (
        "SA"
        if kinds == {"dielectric", "vacuum"}
        else "AA"
        if kinds == {"vacuum"}
        else "SS"
    )
    return kind, f"{first.name}|{second.name}"


def volume_group(domain: Domain) -> str:
    """The primary group of a volume: the entity id itself."""
    return domain.name


def interface_type_of(
    neg: Domain | None, pos: Domain | None, owner: SheetOwner | None
) -> str | None:
    """The interface type of a face (``MA``, ``MS`` ... ``PORT``, ``EXT``, or ``SHEET``), from its structure.

    A Route A conductor sheet is ``SHEET`` (its ``MA`` and ``MS`` rows follow the neighbours of its faces,
    see :func:`sheet_face_sides`); a face with no name is ``None``.  Nothing is parsed from a group name.
    """
    named = _prefix_and_entity(neg, pos, owner)
    return None if named is None else named[0]


def sheet_row_id(row_type: str, entity: str, band: int | None) -> str:
    """The Surface-EPR row of a Route A sheet: ``MA::<entity>__sheet[::band=k]`` or the ``MS`` one."""
    row = f"{row_type}::{entity}{FACE_KIND_SEPARATOR}sheet"
    return row if band is None else f"{row}::band={band}"


def check_entity_names(entity_ids: Iterable[str]) -> None:
    """Fail closed on an entity id that would make a group name ambiguous.

    A bare entity id names the object itself and a prefixed name a face, so an id equal to an
    interface prefix, or one containing the ``::`` separator or the ``__`` that attaches a face kind,
    is rejected.
    """
    for entity_id in sorted(set(entity_ids)):
        if entity_id in RESERVED_PREFIXES:
            raise InputError(
                f"entity id {entity_id!r} is an interface prefix ({', '.join(RESERVED_PREFIXES)}), "
                "so its physical-group name would be ambiguous; rename the entity"
            )
        if "::" in entity_id:
            raise InputError(
                f"entity id {entity_id!r} contains '::', the separator of physical-group names; "
                "rename the entity"
            )
        if FACE_KIND_SEPARATOR in entity_id:
            raise InputError(
                f"entity id {entity_id!r} contains '{FACE_KIND_SEPARATOR}', which attaches the face kind "
                "to the entity in a physical-group name (MA::D1_pad__bottom); use single underscores "
                "(D1_pad) and rename the entity"
            )


def net_of_surface(
    neg: Domain | None, pos: Domain | None, owner: SheetOwner | None
) -> str | None:
    """The Net whose *exposed conductor surface* a face is (sheets, and conductor-to-non-conductor faces)."""
    if owner is not None:
        return owner.net if owner.kind == "conductor" else None
    if neg is None or pos is None:
        return None
    conductors = [d for d in (neg, pos) if d.kind == "conductor"]
    return conductors[0].net if len(conductors) == 1 else None


def sheet_face_sides(neg: Domain | None, pos: Domain | None) -> dict[str, str]:
    """The Surface-EPR row types of one sheet face and the side of the face each one integrates.

    ``MA`` is the neighbour of kind vacuum, ``MS`` the one of kind dielectric (:data:`SHEET_ROW_NEIGHBOURS`).
    A side is ``above`` (the neighbour is the ``pos`` domain), ``below`` (the ``neg`` domain) or ``both``.
    A row type the face has no neighbour for is absent.
    """
    out: dict[str, str] = {}
    for row_type, kind in SHEET_ROW_NEIGHBOURS.items():
        above = pos is not None and pos.kind == kind
        below = neg is not None and neg.kind == kind
        if above or below:
            out[row_type] = "both" if above and below else "above" if above else "below"
    return out


def surface_epr_row_ids(
    group: str,
    interface: str,
    entity: str,
    band: int | None,
    sheet_sides: dict[str, str] | None = None,
) -> list[str]:
    """The Surface-EPR rows a banded group carries, from facts the caller already has (never from the name).

    A Route A sheet group carries the ``MA`` and ``MS`` rows of its Entity that ``sheet_sides`` (the
    :func:`sheet_face_sides` of one of its faces) names; every other banded group carries its own name as
    its row; a port carries none.
    """
    if band is None or interface == "PORT":
        return []
    if interface == SHEET:
        return [sheet_row_id(row_type, entity, band) for row_type in sheet_sides or ()]
    return [group]
