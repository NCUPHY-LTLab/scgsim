"""SGB's own refusals: the ones its stages raise.  Every one stops the build (fail closed).

No fallback geometry is ever produced: a source or planning defect surfaces as
one of these errors, naming the entities and the location involved, and saying who
must change what.  The base class and the refusals of the exact kernel (``SgbError``,
``InputError``, ``UnsupportedCurveError``) live in ``scgsim.geometry.exact.errors``, the
earliest folder that raises them.  An invariant SGB itself broke on an input it accepted
is worded by ``internal`` in ``scgsim.geometry.models.refusals``, which owns that wording.
"""

from __future__ import annotations

from scgsim.geometry.exact.errors import InputError, SgbError


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
