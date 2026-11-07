"""Exceptions raised by the SGB pipeline.  Every one stops the build (fail closed).

No fallback geometry is ever produced: a source or planning defect surfaces as
one of these errors, naming the entities and the location involved.
"""

from __future__ import annotations


class SgbError(Exception):
    """Base class of every SGB failure.

    ``location_um`` is the ``(x, y)`` position in micrometres of the defect when the raising stage knows it,
    so a caller can show it without reading the message.
    """

    def __init__(
        self, message: str = "", *, location_um: tuple[float, float] | None = None
    ) -> None:
        super().__init__(message)
        self.location_um = location_um


class InputError(SgbError):
    """The effective input cannot be turned into prisms (names the entity)."""


class NoGapEdgeError(InputError):
    """An inset support face has no gap edge to measure its bands from (a type-level request skips it)."""


class UnsupportedCurveError(InputError):
    """A source boundary uses a curve class this version does not carry (splines)."""


class RouteError(SgbError):
    """The prisms are not valid for the chosen route (names the entities)."""


class OverlapError(SgbError):
    """Two different-Net conductors, or ambiguous dielectrics, occupy the same place."""


class ContactError(OverlapError):
    """Conductors of different Nets touch, on the source grid or after snapping."""


class NonManifoldError(SgbError):
    """A volume touches itself along a vertical edge (two cells of one domain meet at a vertex)."""


class ValidationError(SgbError):
    """A structural invariant of the design failed."""
