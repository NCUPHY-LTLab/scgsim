"""The model grid ``g`` and the exact conversion of declared lengths onto it.

Owns the one lattice every XY coordinate of the pipeline lives on.  The grid is
declared once per model (the S0 stage binds it to the source database unit) and
is recorded in every hand-off; Z values are never snapped by it.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

LATTICE_TOLERANCE = Fraction(1, 1_000_000)
"""Largest distance, in grid units, between a declared coordinate and its lattice point.

Source coordinates were projected onto the grid by the source stage, so they
differ from a lattice point only by binary floating-point representation.
"""


@dataclass(frozen=True)
class Grid:
    """The model grid in micrometres (declared once; typically the source database unit)."""

    size_um: float

    def __post_init__(self) -> None:
        if not self.size_um > 0:
            raise ValueError("grid size must be a positive length in micrometres")

    def to_um(self, units: int | Fraction | float) -> float:
        """Length in micrometres of a coordinate expressed in grid units."""
        return float(units) * self.size_um

    def exact_units(self, value_um: float) -> Fraction:
        """Exact grid-unit value of a declared micrometre length.

        The decimal text of the float is converted, not its binary expansion, so
        ``40.001`` on a 1 nm grid is exactly 40001 units.
        """
        return Fraction(repr(float(value_um))) / Fraction(repr(self.size_um))

    def lattice_units(self, value_um: float) -> int | None:
        """Lattice coordinate of a micrometre length, or ``None`` when it is off the lattice."""
        exact = self.exact_units(value_um)
        nearest = round(exact)
        return nearest if abs(exact - nearest) <= LATTICE_TOLERANCE else None


Z_DECIMALS = 12


def exact_z(value_um: float | int | str | Fraction) -> Fraction:
    """Exact Z of a declared micrometre value (decimal text, never the binary expansion).

    A float that carries binary noise from upstream arithmetic (``0.30000000000000004``)
    is first rounded to ``Z_DECIMALS`` decimals, six orders of magnitude below a
    nanometre, so noise never creates a spurious thin slab.  This is representation
    hygiene, not a Z threshold: every declared value stays its own level.
    """
    if isinstance(value_um, Fraction):
        return value_um
    if isinstance(value_um, str):
        return Fraction(value_um)
    return Fraction(repr(round(float(value_um), Z_DECIMALS)))
