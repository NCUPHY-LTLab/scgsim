"""Exceptions raised by the pipeline. Every one stops the build (fail closed)."""

from __future__ import annotations


class SgbError(Exception):
    """Base class: a source or planning defect; no fallback geometry is ever produced."""


class OverlapError(SgbError):
    """Two different-Net conductors (or ambiguous dielectrics) occupy the same place."""


class RouteError(SgbError):
    """The prisms are not valid for the chosen route."""


class ValidationError(SgbError):
    """A structural invariant of brief section 11 failed."""


class ContactError(OverlapError):
    """Conductors of different Nets touch (share an edge or a vertex), on the source grid or after snapping."""


class NonManifoldError(SgbError):
    """A volume touches itself along a vertical edge (two cells of one domain meet only at a vertex of a slab)."""
