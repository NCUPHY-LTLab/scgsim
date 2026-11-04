"""Typed requests that add cut families to a model (design brief section 5).

Owns the public vocabulary for asking SGB to partition a surface: *which* surface (a support selection by
entity and face, or by the two solution regions an interface lies between) and *how* (margins in
micrometres, or a curve to embed).  A request is plain data; ``generate`` turns it into a
``CurveFamilyRecord`` before the arrangements are built.  Nothing here refers to a solver: the bands are
geometry, and each is its own primary physical group (``MA::<entity>::<face kind>::band=k`` and so on), so a backend
integrates a plain surface integral over the union of the band groups it needs.

Inset margins
-------------
``InsetRequest(request_id, support, margins_um)`` asks for the bands ``B_0 .. B_K`` of one support face:

* ``ConductorSurface(entity_id, side)`` -- a face of one conductor Entity.  ``side="sheet"`` is the face
  of a sheet conductor (a Route A sheet: one face carrying both the metal-air and the metal-substrate
  surface, group ``<entity>[::band=k]``, interface type ``SHEET``); ``"top"`` and ``"bottom"`` are the faces at the
  upper and lower Z of a finite conductor (groups ``MA::<entity>::<face kind>[::band=k]`` against vacuum
  and ``MS::<entity>::<face kind>[::band=k]`` against a dielectric).  Sidewall bands are not part of version 1.
* ``SolutionInterface(first_entity_id, second_entity_id)`` -- the interface between two solution regions
  where no conductor lies (the gap between a pad and its ground, ``SA::<a>|<b>::interface[::band=k]``).  ``z_um``
  picks the level when the pair meets at several.

``margins_um`` are strictly increasing positive lengths on the model grid.  ``interface`` optionally
states the interface type the face is expected to have (``"MA"``, ``"MS"``, ``"SA"``, ``"SHEET"`` ...);
the build fails if the selected face has no cell of that type.  The distance origin is always the physical
boundary of the support face itself: ports, other bands and seams never move a band.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Side = Literal["sheet", "top", "bottom"]
INTERFACE_TYPES = ("MA", "MS", "MM", "SA", "AA", "SS", "SHEET")


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
class InsetRequest:
    """Bands of one support face at the given margins."""

    request_id: (
        str  # the family id; also the lineage prefix ``<request id>/ring<r>/edge<k>``
    )
    support: ConductorSurface | SolutionInterface
    margins_um: tuple[float, ...]
    interface: str | None = (
        None  # expected interface type of the face, checked against its labels
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
