"""Semantic names of surfaces and volumes: decision D5, per-Entity attribution (design brief section 14).

Owns the primary physical-group names.  Surfaces are attributed **per Entity**, so every
Entity's Surface-EPR contribution is separable; reports aggregate by interface type and
Net is used only where a backend needs electrical identity.

**A bare entity id names the object itself; a name with an interface prefix names a face between two
things.**  Gmsh's Type column already says whether a group is a volume or a surface, so no prefix
restates the dimension:

* ``<entity>`` -- a volume (also the interior of a Route C conductor Entity), or, for a Route A
  conductor sheet, the sheet itself: ``<entity>[::band=k]``, both its sides being the metal-air and the
  metal-substrate surface of that Entity (it carries no face kind, because it is always a ``sheet``);
* ``MA::<entity>::<face kind>`` / ``MS::<entity>::<face kind>`` -- a conductor surface against vacuum
  (metal-air) or against a dielectric (metal-substrate); ``MM::<a>|<b>::<face kind>`` between two
  conductor Entities;
* ``SA::<dielectric>|<vacuum>::<face kind>``, ``AA::<a>|<b>::...``, ``SS::<a>|<b>::...`` -- interfaces
  between solution regions (substrate-air, air-air, substrate-substrate);
* ``EXT::<domain>::<face kind>`` -- the model boundary;
* ``PORT::<port sheet id>`` -- a lumped-port sheet (always one sheet, so it carries no face kind).

The interface prefixes stay because they carry the physics: they say which Surface-EPR loss parameters
apply to the face.  An entity id equal to a prefix (:data:`RESERVED_PREFIXES`), or containing ``::``,
would make a name ambiguous, so :func:`check_entity_names` rejects it.

The interface type of a face (``MA``, ``MS`` ..., or ``SHEET`` for a Route A conductor sheet) is never
read back from a name: :func:`interface_type_of` derives it from the domains and the sheet owner.

**Face kind** (:data:`FACE_KINDS`) is part of the identity of a prefixed name, so one primary group never
mixes physically different interfaces and a consumer can filter by it without reading geometry:

* ``top`` / ``bottom`` -- a horizontal face with a conductor below / above it (the conductor's top or
  bottom); at the model boundary, the top or bottom of the domain;
* ``sidewall`` -- a vertical face (a wall);
* ``sheet`` -- a zero-thickness conductor face (Route A);
* ``interface`` -- a horizontal face between two solution regions, or two conductors.

A face inside an inset band carries ``::band=k`` (the innermost band index of its label);
a port stays one surface even where bands cut it.  Names never carry native tags.

Surface-EPR rows keep the prefixed form for a sheet: its group ``pad::band=1`` carries the rows
``MA::pad::sheet::band=1`` and ``MS::pad::sheet::band=1`` (see :func:`sheet_row_id`).
"""

from __future__ import annotations

from collections.abc import Iterable

from scgsim.geometry.sgb.errors import InputError
from scgsim.geometry.sgb.records.labels import Domain, SheetOwner

RESERVED_PREFIXES = ("MA", "MS", "MM", "SA", "AA", "SS", "EXT", "PORT")
"""The interface prefixes of a prefixed group name; no entity id may equal one of them."""
SHEET = "SHEET"
"""Interface type of a Route A conductor sheet (a record field and a union key, never part of a group name)."""
FACE_KINDS = ("top", "bottom", "sidewall", "sheet", "interface")
"""The face kinds a primary surface group name can carry (a port face is ``port`` and is not in a name)."""


def surface_group(  # noqa: PLR0913 - the two sides, the owner, the bands and the orientation
    neg: Domain | None,
    pos: Domain | None,
    owner: SheetOwner | None,
    bands: tuple[tuple[str, int], ...] = (),
    *,
    vertical: bool = False,
) -> str | None:
    """The single primary physical-group name of a face from the domains on its two sides.

    ``vertical`` says the face is a wall (``sidewall``); otherwise it is horizontal.
    """
    base = _base_group(neg, pos, owner)
    if base is None:
        return None
    if owner is not None and owner.kind == "port":
        return base  # one sheet: no face kind, no band
    if owner is not None:
        # a conductor sheet: the entity id; its face kind is always sheet
        return _banded(base, bands)
    name = f"{base}::{face_kind(neg, pos, owner, vertical=vertical)}"
    return _banded(name, bands)


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


def _base_group(
    neg: Domain | None, pos: Domain | None, owner: SheetOwner | None
) -> str | None:
    if owner is not None:
        return f"PORT::{owner.name}" if owner.kind == "port" else owner.name
    if neg is None:
        return None if pos is None else f"EXT::{pos.name}"
    if pos is None:
        return f"EXT::{neg.name}"
    conductors = [d for d in (neg, pos) if d.kind == "conductor"]
    if len(conductors) == 2:
        first, second = sorted((neg.name, pos.name))
        return f"MM::{first}|{second}"
    if conductors:
        metal = conductors[0]
        other = pos if metal == neg else neg
        return f"{'MA' if other.kind == 'vacuum' else 'MS'}::{metal.name}"
    return _solution_interface(neg, pos)


def _solution_interface(a: Domain, b: Domain) -> str:
    """Substrate-air, air-air or substrate-substrate, dielectric first."""
    first, second = sorted((a, b), key=lambda d: (d.kind == "vacuum", d.name))
    kinds = {a.kind, b.kind}
    kind = (
        "SA"
        if kinds == {"dielectric", "vacuum"}
        else "AA"
        if kinds == {"vacuum"}
        else "SS"
    )
    return f"{kind}::{first.name}|{second.name}"


def volume_group(domain: Domain) -> str:
    """The primary group of a volume: the entity id itself."""
    return domain.name


def interface_type_of(
    neg: Domain | None, pos: Domain | None, owner: SheetOwner | None
) -> str | None:
    """The interface type of a face (``MA``, ``MS`` ... ``PORT``, ``EXT``, or ``SHEET``), from its structure.

    A Route A conductor sheet is ``SHEET`` (it carries both the metal-air and the metal-substrate row);
    a face with no name is ``None``.  Nothing is parsed from a group name.
    """
    if owner is not None:
        return "PORT" if owner.kind == "port" else SHEET
    base = _base_group(neg, pos, owner)
    return None if base is None else base.split("::", 1)[0]


def sheet_row_id(row_type: str, entity: str, band: int | None) -> str:
    """The Surface-EPR row of a Route A sheet: ``MA::<entity>::sheet[::band=k]`` or the ``MS`` one."""
    row = f"{row_type}::{entity}::sheet"
    return row if band is None else f"{row}::band={band}"


def check_entity_names(entity_ids: Iterable[str]) -> None:
    """Fail closed on an entity id that would make a group name ambiguous.

    A bare entity id names the object itself and a prefixed name a face, so an id equal to an
    interface prefix, or one containing the ``::`` separator, is rejected.
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


def surface_epr_row_ids(
    group: str, interface: str, entity: str, band: int | None
) -> list[str]:
    """The Surface-EPR rows a banded group carries, from facts the caller already has (never from the name).

    A Route A sheet group carries both the ``MA`` and the ``MS`` row of its Entity; every other banded
    group carries its own name as its row; a port carries none.
    """
    if band is None or interface == "PORT":
        return []
    if interface == SHEET:
        return [sheet_row_id("MA", entity, band), sheet_row_id("MS", entity, band)]
    return [group]
