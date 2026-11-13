"""The refusals the exact kernel raises, and the base class of every SGB refusal.

Every one stops the build (fail closed): no fallback geometry is ever produced.  A refusal names the
entities and the location involved and says who must change what.  An invariant the code itself broke on
an input it accepted is worded by ``internal`` in ``scgsim.geometry.models.refusals``, which owns that
wording.  SGB's own refusals (``scgsim.geometry.sgb.errors``) derive from these classes.
"""

from __future__ import annotations


class SgbError(Exception):
    """Base class of every SGB failure."""


class InputError(SgbError):
    """The effective input cannot be turned into prisms (names the entity)."""


class UnsupportedCurveError(InputError):
    """A source boundary uses a curve class this version does not carry (splines)."""
