"""The tolerances with which a loaded model is compared with the sidecar's ``expected`` block.

One set for every loader: Gmsh (:func:`~scgsim.geometry.sgb.checks.verification.loaded.verify_loaded`) and OCP
(:func:`~scgsim.geometry.sgb.checks.verification.occt.sidecar_mismatches`) accept the same difference, so a file
that passes one passes the other.  This module imports neither, so each can use it without the other's
optional dependency.
"""

from __future__ import annotations

VERTEX_TOLERANCE_UM = 1e-9
"""Largest absolute difference, in micrometres, of any coordinate of a vertex from its planned position."""
RELATIVE_TOLERANCE = 1e-9
"""Largest relative difference of a face area or solid volume from its planned value."""


def relative_error(measured: float, planned: float) -> float:
    """``|measured - planned|`` relative to the planned value (which may be tiny, never zero-divides)."""
    return abs(measured - planned) / max(abs(planned), 1e-30)
