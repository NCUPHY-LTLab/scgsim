"""Semantic names of surfaces and volumes: decision D5, per-Entity attribution (design brief section 14).

Owns the primary physical-group names.  Surfaces are attributed **per Entity**, so every
Entity's Surface-EPR contribution is separable; reports aggregate by interface type and
Net is used only where a backend needs electrical identity.

``<kind>::<qualified entity id>::<face kind>[::band=k]``

* ``SHEET::<entity>::sheet`` -- a Route A conductor face; both its sides are the metal-air and the
  metal-substrate surface of that Entity;
* ``MA::<entity>::<face kind>`` / ``MS::<entity>::<face kind>`` -- a conductor surface against vacuum
  (metal-air) or against a dielectric (metal-substrate); ``MM::<a>|<b>::<face kind>`` between two
  conductor Entities;
* ``SA::<dielectric>|<vacuum>::<face kind>``, ``AA::<a>|<b>::...``, ``SS::<a>|<b>::...`` -- interfaces
  between solution regions (substrate-air, air-air, substrate-substrate);
* ``EXT::<domain>::<face kind>`` -- the model boundary;
* ``PORT::<port sheet id>`` -- a lumped-port sheet (always one sheet, so it carries no face kind);
* ``VOL::<entity>`` -- a volume (also the interior of a Route C conductor Entity).

**Face kind** (:data:`FACE_KINDS`) is part of the identity, so one primary group never mixes physically
different interfaces and a consumer can filter by it without reading geometry:

* ``top`` / ``bottom`` -- a horizontal face with a conductor below / above it (the conductor's top or
  bottom); at the model boundary, the top or bottom of the domain;
* ``sidewall`` -- a vertical face (a wall);
* ``sheet`` -- a zero-thickness conductor face (Route A);
* ``interface`` -- a horizontal face between two solution regions, or two conductors.

A face inside an inset band carries ``::band=k`` (the innermost band index of its label);
a port stays one surface even where bands cut it.  Names never carry native tags.
"""

from __future__ import annotations

from collections.abc import Iterable

from scgsim.geometry.sgb.records.labels import Domain, SheetOwner

INTERFACE_KINDS = ("MA", "MS", "MM", "SA", "AA", "SS", "SHEET", "PORT", "EXT")
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
    name = f"{base}::{face_kind(neg, pos, owner, vertical=vertical)}"
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
        return f"PORT::{owner.name}" if owner.kind == "port" else f"SHEET::{owner.name}"
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
    """The primary group of a volume: ``VOL::<entity>``."""
    return f"VOL::{domain.name}"


def interface_type(group: str) -> str:
    """The interface-type prefix of a primary group name (``MA`` for ``MA::pad::band=1``)."""
    return group.split("::", 1)[0]


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


def surface_epr_rows(group_names: Iterable[str]) -> dict[str, str]:
    """Surface-EPR row -> the primary group that carries it, for every banded group.

    A Route A sheet group (``SHEET::<entity>::sheet::band=k``) carries both the
    ``MA::<entity>::sheet::band=k`` and the ``MS::<entity>::sheet::band=k`` row; every other banded
    group (``MA``, ``MS``, ``SA``) carries its own row.
    """
    rows: dict[str, str] = {}
    for name in sorted(group_names):
        if "::band=" not in name:
            continue
        kind, rest = name.split("::", 1)
        if kind == "SHEET":
            rows[f"MA::{rest}"] = name
            rows[f"MS::{rest}"] = name
        else:
            rows[name] = name
    return rows
