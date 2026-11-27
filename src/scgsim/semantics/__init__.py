"""Solver-independent, revision-bound semantic facts.

This internal package is standard-library only.  It holds plain, already-valid facts and never imports SGB,
Palace, AEDT, native backends, or callbacks: the revision-bound snapshot (``snapshot``), the Route A
projection (``route_a``), the EPR records (``epr``) and the S-parameter record every driven solver returns,
with its Touchstone writer (``s_parameters``).
"""

from .snapshot import (
    SemanticFactsSnapshot,
    SemanticSnapshotReference,
    canonical_sha256,
    create_snapshot,
)

__all__ = [
    "SemanticFactsSnapshot",
    "SemanticSnapshotReference",
    "canonical_sha256",
    "create_snapshot",
]
