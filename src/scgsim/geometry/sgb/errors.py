"""The refusals of a geometry build: the one module a caller imports them from.  Every one stops the build (fail closed).

No fallback geometry is ever produced: a source or planning defect surfaces as
one of these errors, naming the entities and the location involved, and saying who
must change what.  A caller catches any of the nine classes of ``__all__`` from this
module (``docs/supported-scope.qmd`` names it).  Six are SGB's own and are defined here.
The base class and the two refusals of the exact kernel (``SgbError``, ``InputError``,
``UnsupportedCurveError``) are defined in ``scgsim.geometry.exact.errors``, the earliest
folder that raises them, and listed here for the caller; code inside ``scgsim`` imports
each class from the module that defines it.  An invariant SGB itself broke on an input it
accepted is worded by ``internal`` in ``scgsim.geometry.models.refusals``, which owns that wording.
"""

from __future__ import annotations

from scgsim.geometry.exact.errors import InputError, SgbError, UnsupportedCurveError

__all__ = [
    "ContactError",
    "InputError",
    "NoGapEdgeError",
    "NonManifoldError",
    "OverlapError",
    "RouteError",
    "SgbError",
    "UnsupportedCurveError",
    "ValidationError",
]


class NoGapEdgeError(InputError):
    """An inset support face has no gap edge to measure its bands from (a type-level request skips it)."""


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
