"""Platform-independent floating point for the BREP bytes (design brief decision D8).

Owns the one rule that keeps the written file byte-identical on every machine: every number
in it comes from IEEE-754 ``+ - * /`` and ``sqrt`` (correctly rounded on every platform) and is
printed by Python's own ``%`` formatting (also exact on every platform).  The C library's ``sin``,
``cos`` and ``atan2`` are *not* correctly rounded and differ in the last digit between libm
versions, so the writer never calls them: angles come from :func:`angle_of`, which is built from
the basic operations only, and axes are taken as unit vectors rather than as ``(cos, sin)``.

The vectorised helpers work on numpy arrays; numpy's elementwise ``+ - * /`` and ``sqrt`` are
IEEE operations whichever SIMD path the CPU selects.
"""

from __future__ import annotations

import numpy as np

PI = 3.141592653589793
HALF_PI = PI / 2
_HALVINGS = (
    3  # atan(t) = 2 atan(t / (1 + sqrt(1 + t^2))); three halvings bring t below 0.1
)
_SERIES_TERMS = 9  # (0.1)^(2k) / (2k + 1) < 1e-17 for k = 9


def _atan_unit(t: np.ndarray) -> np.ndarray:
    """Arctangent of ``t`` in [0, 1] from argument halving and a Taylor series (basic operations only)."""
    for _ in range(_HALVINGS):
        t = t / (1.0 + np.sqrt(1.0 + t * t))
    square = t * t
    series = np.zeros_like(t)
    for k in range(
        _SERIES_TERMS, -1, -1
    ):  # alternating series 1/1 - u/3 + u^2/5 - ..., Horner form
        series = 1.0 / (2 * k + 1) - square * series
    return t * series * float(2**_HALVINGS)


def angle_of(cross: np.ndarray, dot: np.ndarray) -> np.ndarray:
    """The angle in (-pi, pi] of the vector ``(dot, cross)``: ``atan2(cross, dot)`` without libm.

    Accurate to a few units in the last place, identical on every platform.
    """
    cross, dot = np.asarray(cross, dtype=float), np.asarray(dot, dtype=float)
    abs_cross, abs_dot = np.abs(cross), np.abs(dot)
    larger = np.maximum(abs_cross, abs_dot)
    smaller = np.minimum(abs_cross, abs_dot)
    ratio = np.divide(smaller, larger, out=np.zeros_like(larger), where=larger > 0)
    base = _atan_unit(ratio)
    angle = np.where(abs_cross > abs_dot, HALF_PI - base, base)
    angle = np.where(dot < 0, PI - angle, angle)
    return np.where(cross < 0, -angle, angle)


def positive_zero(values: np.ndarray) -> np.ndarray:
    """``-0.0`` turned into ``0.0`` (``x + 0.0``), so a sign of zero never reaches the file."""
    return values + 0.0


def rows(template: str, array: np.ndarray) -> list[str]:
    """One string per row of ``array``, printed with ``template`` (``%`` formatting; ``-0`` normalised)."""
    return list(map(template.__mod__, map(tuple, positive_zero(array).tolist())))
