"""Typed requests that add cut families to a model (design brief section 5).

Owns the public vocabulary for asking SGB to partition a surface: *which* surface (a support selection by
entity and face, or by the two solution regions an interface lies between) and *how* (margins in
micrometres, or a curve to embed).  A request is plain data; ``generate`` turns it into a
``CurveFamilyRecord`` before the arrangements are built.  Nothing here refers to a solver: the bands are
geometry, and each is its own primary physical group (``MA::<entity>__<face kind>::band=k`` and so on), so a backend
integrates a plain surface integral over the union of the band groups it needs.

Inset margins
-------------
``InsetRequest(request_id, support, margins_um)`` asks for the bands ``B_0 .. B_K`` of one support face, or of every
face of a Surface-EPR type.  There are two forms of support, and they serve different uses.

*Concrete* supports name one face.  Use them to band chosen faces only (one pad's top, one gap):

* ``ConductorSurface(entity_id, side)`` -- a face of one conductor Entity.  ``side="sheet"`` is the face
  of a sheet conductor (a Route A sheet: one face whose metal-air and metal-substrate rows follow its
  neighbours, group ``<entity>[::band=k]``, interface type ``SHEET``); ``"top"`` and ``"bottom"`` are the faces at the
  upper and lower Z of a finite conductor (groups ``MA::<entity>__<face kind>[::band=k]`` against vacuum
  and ``MS::<entity>__<face kind>[::band=k]`` against a dielectric).  Sidewall bands are not part of version 1.
* ``SolutionInterface(first_entity_id, second_entity_id)`` -- the interface between two solution regions
  where no conductor lies (the gap between a pad and its ground, ``SA::<a>|<b>__interface[::band=k]``).  ``z_um``
  picks the level when the pair meets at several.

A concrete request is exact: it fails closed on a face that is not there, has no gap edge or whose
representation does not fit.

*Type-level* support names no face.  ``InterfaceSurfaces(interface_type)`` asks for "every face of this
Surface-EPR type" (``"MA"``, ``"MS"`` or ``"SA"``) and SGB finds the faces itself, from the same labels and
interface-type rules that give the sidecar its ``surface_epr_rows``:

* ``"MA"`` / ``"MS"`` -- the sheet of a Route A conductor that has a row of that type (a sheet with vacuum on
  one side and a dielectric on the other carries both), and the top and bottom faces of a finite conductor that
  have a cell of that type.  Sidewalls are never banded.
* ``"SA"`` -- each solution interface between a dielectric and a vacuum, on each level where it lies.

``face_kinds`` narrows the faces to some of ``"sheet"``, ``"top"``, ``"bottom"`` and ``"interface"``;
``exclude_entity_ids`` leaves out the faces of the named entities (a conductor face by its entity, a solution
interface when either side is that entity), so concrete requests for those entities never meet the type-level one.
A face with no gap edge is skipped and recorded in the sidecar, never an error.  Each face becomes one family with the
id ``<request id>::<entity>::<side>`` (``<request id>::<first>|<second>::z=<z>`` for an interface); a face that several
requests land on is one family, so they must ask the same margins on the model grid (``cutfamilies.inset_plan``).

``margins_um`` are strictly increasing positive lengths on the model grid.  ``interface`` optionally
states the interface type the face is expected to have (``"MA"``, ``"MS"``, ``"SA"``, ``"SHEET"`` ...);
the build fails if the selected face has no cell of that type.

Distance origins.  A band is the set of points of the support face whose distance to the *gap edges* lies
between two margins; farther than the last margin is the last band.  A gap edge is a part of a conductor
footprint's boundary with an exposed dielectric surface (an SA or SS interface) on its other side, in the plane
the conductor is attached to (its sheet plane, or the bottom of a finite body), with lumped ports ignored.  The
Surface-EPR correction factor of a band comes from the metal-gap-metal cross-section, so an edge with no gap
before it is no origin: a conductor edge on the die or substrate outer edge or the model boundary, an edge
shared with another conductor, an edge facing only vacuum.  The die outer edge, ports, other bands and seams
therefore never move a band.  Ports also cut no band: a band seam stops at the boundary of a lumped port on the
support's level, so the port is one face (a rectangular port stays one rectangle) and a band's geometry outside the
port is what it would be without it.  Only the cuts that must exist split a port: a material or entity boundary, a
host overlap, a junction partition, or a ``RefinementRequest`` curve crossing it.  A conductor face is measured from the gap edges of its footprint (the top and
bottom faces of a finite body have the same footprint); a solution interface from the gap edges on its own
boundary.  A support with no gap edge is an ``InputError``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Side = Literal["sheet", "top", "bottom"]
INTERFACE_TYPES = ("MA", "MS", "MM", "SA", "AA", "SS", "SHEET")
BAND_TYPES = ("MA", "MS", "SA")
"""The Surface-EPR types a type-level request can ask for: metal-air, metal-substrate and substrate-air.  The other
interface types have no band: an ``AA`` surface has no conductor next to it, so it never has a gap edge; ``MM`` and ``SS`` are
not Surface-EPR contributions."""
FaceKind = Literal["sheet", "top", "bottom", "interface"]
FACE_KINDS: tuple[FaceKind, ...] = ("sheet", "top", "bottom", "interface")
"""The face kinds a band can lie on: a conductor sheet, the top or bottom of a finite conductor, a solution interface."""


@dataclass(frozen=True)
class ConductorSurface:
    """A face of one conductor Entity: its sheet, or the top or bottom face of its finite body."""

    entity_id: str
    side: Side


@dataclass(frozen=True)
class SolutionInterface:
    """The interface between two solution regions, where no conductor lies (an SA, AA or SS surface)."""

    first_entity_id: str
    second_entity_id: str
    z_um: float | None = None


@dataclass(frozen=True)
class InterfaceSurfaces:
    """Every face of one Surface-EPR type (``MA``, ``MS`` or ``SA``), found by SGB.

    ``face_kinds`` keeps only those kinds of face (``None``: every bandable kind) and ``exclude_entity_ids`` drops the
    faces of the named entities.
    """

    interface_type: str
    face_kinds: tuple[str, ...] | None = None
    exclude_entity_ids: tuple[str, ...] = ()


ConcreteSupport = ConductorSurface | SolutionInterface
"""A support that names one face."""


@dataclass(frozen=True)
class InsetRequest:
    """Bands of one support face (or of every face of a type) at the given margins."""

    request_id: str  # the family id of a face; also the lineage prefix ``<request id>/ring<r>/edge<k>``
    support: ConductorSurface | SolutionInterface | InterfaceSurfaces
    margins_um: tuple[float, ...]
    interface: str | None = (
        None  # expected interface type of the face, checked against its labels (a concrete support only)
    )


@dataclass(frozen=True)
class RefinementRequest:
    """A polyline of mesh-size curves at height ``z_um`` that the mesher embeds in the face containing it.

    The pieces are ``refinement_only``: they never change a label, they are embedded in the face
    that contains them, and a piece that lands in no emitted face is an error, never dropped.
    """

    request_id: str
    z_um: float
    points_um: tuple[tuple[float, float], ...]
