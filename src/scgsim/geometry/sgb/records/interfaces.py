"""The interface type of a face, from the domains on its two sides and its sheet owner.

One rule, read by two stages: ``bands`` finds the faces a type-level request asks for and checks the type a
request expects, and ``topology.routes`` names the faces and gives them their Surface-EPR rows.  The type is
never read back from a group name.
"""

from __future__ import annotations

from scgsim.geometry.sgb.records.labels import Domain, SheetOwner

SHEET = "SHEET"
"""Interface type of a Route A conductor sheet (a record field and a union key, never part of a group name)."""
INTERFACE_TYPES = ("MA", "MS", "MM", "SA", "AA", "SS", SHEET)
"""The interface types of a face between two regions or of a conductor sheet: what an inset request may state
it expects its face to be (a port and the model boundary are not)."""
SURFACE_EPR_TYPES = ("MA", "MS", "SA")
"""The interface types that are Surface-EPR contributions: metal-air, metal-substrate and substrate-air.  They are
also the types a band can be asked for by: an ``AA`` surface has no conductor next to it, so it never has a gap
edge, and ``MM`` and ``SS`` are not Surface-EPR contributions."""
SHEET_ROW_NEIGHBOURS = {"MA": "vacuum", "MS": "dielectric"}
"""The Surface-EPR row types of a sheet face and the kind of neighbour each one integrates."""


def interface_type_of(
    neg: Domain | None, pos: Domain | None, owner: SheetOwner | None
) -> str | None:
    """The interface type of a face (``MA``, ``MS`` ... ``PORT``, ``EXT``, or ``SHEET``), from its structure.

    A Route A conductor sheet is ``SHEET`` (its ``MA`` and ``MS`` rows follow the neighbours of its faces,
    see :func:`sheet_face_sides`); a face with no domain on either side is ``None``.
    """
    if owner is not None:
        return "PORT" if owner.kind == "port" else SHEET
    if neg is None and pos is None:
        return None
    if neg is None or pos is None:
        return "EXT"
    conductors = [d for d in (neg, pos) if d.kind == "conductor"]
    if len(conductors) == 2:
        return "MM"
    if conductors:
        other = pos if conductors[0] == neg else neg
        return "MA" if other.kind == "vacuum" else "MS"
    kinds = {neg.kind, pos.kind}
    if kinds == {"dielectric", "vacuum"}:
        return "SA"
    return "AA" if kinds == {"vacuum"} else "SS"


def ordered_pair(a: Domain, b: Domain) -> tuple[Domain, Domain]:
    """The two domains of an interface in the order of its name: dielectric before vacuum, then by name."""
    first, second = sorted((a, b), key=lambda d: (d.kind == "vacuum", d.name))
    return first, second


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
