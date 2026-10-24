"""Caller-owned polygon boundary identities and distance partition controls."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SourceBoundarySegment:
    """Original parent-ring edge i runs from vertex i to (i + 1) modulo length."""

    ring: Literal["exterior", "hole"]
    segment_index: int
    hole_index: int | None = None

    def __post_init__(self) -> None:
        if self.ring not in ("exterior", "hole"):
            raise ValueError("ring must be exterior or hole")
        if self.segment_index < 0:
            raise ValueError("segment_index must be nonnegative")
        if self.ring == "exterior" and self.hole_index is not None:
            raise ValueError("exterior boundaries have no hole_index")
        if self.ring == "hole" and (self.hole_index is None or self.hole_index < 0):
            raise ValueError("hole boundaries require a nonnegative hole_index")


@dataclass(frozen=True)
class SurfaceDistanceBandSpec:
    """Partition one planar parent by distance from its original boundary edges.

    precision_um is the declared layout Boolean grid. chord_error_um controls
    round-cap approximation; omitted chord_error_um retains precision_um.
    For an existing partition, boundary indices address the retained original
    rings exposed as catalogue distance_origin_rings_3d_um. Missing lineage is
    unsupported; child seams never become distance origins.
    source_polygon_id must be explicit None only for a parent without polygon
    provenance; source_entity_id still binds an actual parent owner.
    """

    parent_surface_id: str
    source_entity_id: str
    source_polygon_id: str | None
    distances_um: tuple[float, ...]
    precision_um: float
    boundary_segments: tuple[SourceBoundarySegment, ...] | None = None
    chord_error_um: float | None = None

    def __post_init__(self) -> None:
        if self.chord_error_um is not None:
            from math import isfinite
            if not isfinite(self.chord_error_um) or self.chord_error_um <= 0:
                raise ValueError("chord_error_um must be positive and finite")
        object.__setattr__(self, "distances_um", tuple(self.distances_um))
        if self.boundary_segments is not None:
            object.__setattr__(self, "boundary_segments", tuple(self.boundary_segments))
        if not self.parent_surface_id or not self.source_entity_id:
            raise ValueError("parent surface and source entity identities are required")
        if self.source_polygon_id is not None and (
            not isinstance(self.source_polygon_id, str) or not self.source_polygon_id
        ):
            raise ValueError("source_polygon_id must be a nonempty string or explicit None")
        if not self.precision_um > 0:
            raise ValueError("precision_um must be positive")
        if not self.distances_um or any(not value > 0 for value in self.distances_um):
            raise ValueError("distances_um must contain positive cuts")
        if any(a >= b for a, b in zip(self.distances_um, self.distances_um[1:])):
            raise ValueError("distance cuts must be strictly increasing")
        if self.boundary_segments == ():
            raise ValueError("explicit boundary subset must not be empty")
        if self.boundary_segments is not None and len(
            set(self.boundary_segments)
        ) != len(self.boundary_segments):
            raise ValueError("boundary subset contains duplicate edges")

    def to_payload(self) -> dict:
        return {
            "parent_surface_id": self.parent_surface_id,
            "source_entity_id": self.source_entity_id,
            "source_polygon_id": self.source_polygon_id,
            "distances_um": list(self.distances_um),
            "precision_um": self.precision_um,
            **({"chord_error_um": self.chord_error_um} if self.chord_error_um is not None else {}),
            "boundary_segments": None
            if self.boundary_segments is None
            else [
                {
                    "ring": edge.ring,
                    "segment_index": edge.segment_index,
                    "hole_index": edge.hole_index,
                }
                for edge in self.boundary_segments
            ],
        }

    @classmethod
    def from_payload(cls, value: dict) -> "SurfaceDistanceBandSpec":
        edges = value["boundary_segments"]
        return cls(
            parent_surface_id=value["parent_surface_id"],
            source_entity_id=value["source_entity_id"],
            source_polygon_id=value["source_polygon_id"],
            distances_um=tuple(value["distances_um"]),
            precision_um=value["precision_um"],
            chord_error_um=value.get("chord_error_um"),
            boundary_segments=None
            if edges is None
            else tuple(SourceBoundarySegment(**edge) for edge in edges),
        )
