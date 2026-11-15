"""Grid policy for assembled-layout XY coordinates, separate from physical Z and source DBU."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from math import isfinite
from typing import Any, Mapping

from scgsim.geometry.models.input import GeometryBuildInput
from scgsim.refusal import Declared

# The class A wall-column limit, in grid steps (decided on issue 125): a side wall standing on an
# outline edge shorter than this is refused by the meshing stage, and a junction partition keeps
# every edge its end pieces add zero or at least this long.  The meshing stage's
# FineStructureLimits reads this number; it is defined only here.
WALL_COLUMN_LIMIT_STEPS = 10


@dataclass(frozen=True)
class GeometryGridContext:
    """Positive XY snap spacing and the assembled-layout frame it applies to."""

    grid_size_um: float
    coordinate_frame: str = "assembly_xy"

    def __post_init__(self) -> None:
        if isinstance(self.grid_size_um, bool) or not isinstance(
            self.grid_size_um, (int, float)
        ):
            raise TypeError(
                Declared(
                    "the geometry grid",
                    "the layout's GDS database unit, or the input's metadata 'geometry_grid'",
                ).expected(
                    "grid_size_um",
                    "a positive length in micrometres",
                    self.grid_size_um,
                )
            )
        if not isfinite(self.grid_size_um) or self.grid_size_um <= 0:
            raise ValueError(
                Declared(
                    "the geometry grid",
                    "the layout's GDS database unit, or the input's metadata 'geometry_grid'",
                ).expected(
                    "grid_size_um", "a positive finite length", self.grid_size_um
                )
            )
        if self.coordinate_frame != "assembly_xy":
            raise ValueError(
                Declared(
                    "the geometry grid",
                    "the layout's GDS database unit, or the input's metadata 'geometry_grid'",
                ).expected(
                    "coordinate_frame",
                    "the assembly_xy layout frame",
                    self.coordinate_frame,
                )
            )

    def to_payload(self) -> dict[str, str | float]:
        """Return the canonical versioned metadata representation."""
        return {
            "schema_version": "scgsim.geometry.grid.v1",
            "grid_size_um": self.grid_size_um,
            "coordinate_frame": self.coordinate_frame,
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> GeometryGridContext:
        """Validate canonical metadata and reconstruct its grid policy."""
        if set(value) != {"schema_version", "grid_size_um", "coordinate_frame"}:
            raise ValueError(
                Declared(
                    "the geometry grid",
                    "the layout's GDS database unit, or the input's metadata 'geometry_grid'",
                ).expected(
                    "payload",
                    "exactly schema_version, grid_size_um and coordinate_frame",
                    sorted(value),
                )
            )
        if value["schema_version"] != "scgsim.geometry.grid.v1":
            raise ValueError(
                Declared(
                    "the geometry grid",
                    "the layout's GDS database unit, or the input's metadata 'geometry_grid'",
                ).expected(
                    "schema_version", "scgsim.geometry.grid.v1", value["schema_version"]
                )
            )
        return cls(value["grid_size_um"], value["coordinate_frame"])


def geometry_grid_context(build_input: GeometryBuildInput) -> GeometryGridContext:
    """The model grid the input declares; an input without one is refused, never defaulted."""
    value = build_input.metadata.get("geometry_grid")
    if value is None:
        raise ValueError(
            "the geometry input declares no geometry grid (metadata 'geometry_grid'). "
            "Build it with GeometryPlan.prepare() or build_gds_stack_geometry_input, "
            "which put its XY coordinates on the layout's database-unit grid."
        )
    return GeometryGridContext.from_payload(value)


NM_PER_UM = 1000
# Decimals a sub-grid length keeps beyond the grid's own (a gap of 0.2 nm on a
# 1 nm grid must not print as 0); presentation only, never a tolerance.
SUB_GRID_DECIMALS = 3


def format_um(value: float, grid_um: float | None) -> str:
    """A coordinate in micrometres with the decimals of the grid size (in full without a grid)."""
    if grid_um is None:
        return repr(float(value))
    decimals = grid_decimals(grid_um)
    # A coordinate a hair below zero rounds to zero; it prints without a sign.
    return f"{round(value, decimals) + 0.0:.{decimals}f}"


def format_point(x: float, y: float, grid_um: float | None) -> str:
    """A point of the plane in micrometres, for error messages."""
    return f"({format_um(x, grid_um)}, {format_um(y, grid_um)}) um"


def format_box(bounds: Sequence[float], grid_um: float | None) -> str:
    """A bounding box ``(x0, y0, x1, y1)`` in micrometres, or its point when it has no extent."""
    x0, y0, x1, y1 = bounds
    corner = format_point(x0, y0, grid_um)
    other = format_point(x1, y1, grid_um)
    return corner if corner == other else f"{corner.removesuffix(' um')}..{other}"


def format_extent(points: Sequence[Sequence[float]], grid_um: float | None) -> str:
    """Where points of the plane lie: the box around them, as :func:`format_box` prints it."""
    xs, ys = [x for x, _ in points], [y for _, y in points]
    return format_box((min(xs), min(ys), max(xs), max(ys)), grid_um)


def format_length(length_um: float, grid_um: float | None) -> str:
    """A length in nanometres, and in grid steps when there is a grid."""
    if grid_um is None:
        return f"{repr(float(length_um * NM_PER_UM))} nm"
    nm_decimals = max(grid_decimals(grid_um) - 3, 0) + SUB_GRID_DECIMALS
    nm = _trimmed(f"{length_um * NM_PER_UM:.{nm_decimals}f}")
    steps = _trimmed(f"{length_um / grid_um:.{SUB_GRID_DECIMALS}f}")
    return f"{nm} nm = {steps} grid steps"


def grid_decimals(grid_um: float) -> int:
    """Decimals of the grid size in micrometres, so a lattice coordinate prints exactly."""
    exponent = Decimal(repr(float(grid_um))).normalize().as_tuple().exponent
    assert isinstance(exponent, int)  # a finite positive grid size
    return max(0, -exponent)


def _trimmed(number: str) -> str:
    return number.rstrip("0").rstrip(".") if "." in number else number
