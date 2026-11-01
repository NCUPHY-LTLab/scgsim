"""Grid policy for assembled-layout XY coordinates, separate from physical Z and source DBU."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

from scgsim.geometry.models.input import GeometryBuildInput


@dataclass(frozen=True)
class GeometryGridContext:
    """Positive XY snap spacing and the assembled-layout frame it applies to."""

    grid_size_um: float
    coordinate_frame: str = "assembly_xy"

    def __post_init__(self) -> None:
        if isinstance(self.grid_size_um, bool) or not isinstance(
            self.grid_size_um, (int, float)
        ):
            raise TypeError("grid_size_um must be a positive length in micrometres")
        if not isfinite(self.grid_size_um) or self.grid_size_um <= 0:
            raise ValueError("grid_size_um must be a positive finite length")
        if self.coordinate_frame != "assembly_xy":
            raise ValueError("geometry grid requires the assembly_xy layout frame")

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
            raise ValueError("geometry grid payload members are not canonical")
        if value["schema_version"] != "scgsim.geometry.grid.v1":
            raise ValueError("unsupported geometry grid schema")
        return cls(value["grid_size_um"], value["coordinate_frame"])


def geometry_grid_context(
    build_input: GeometryBuildInput,
) -> GeometryGridContext | None:
    """Read the optional grid policy without inferring one for older inputs."""
    value = build_input.metadata.get("geometry_grid")
    return None if value is None else GeometryGridContext.from_payload(value)
