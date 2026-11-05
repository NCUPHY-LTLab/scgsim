"""What the quality audit found in one mesh: counts, and the bad elements with their SGB attribution.

Owns the records that cross the process boundary between the Gmsh worker (``audit``) and the repair loop
(``repair``).  They are plain frozen data with a JSON form, so the parent process never imports Gmsh to
read them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

# The element-quality measure the threshold applies to: Gmsh's minimum signed inscribed-circumscribed
# radius ratio (1 is a regular tetrahedron, 0 a flat one, negative an inverted one).
QUALITY_MEASURE = "minSICN"

# Where a bad element touches the model's entities, from the most to the least specific.  Rules/1 refines
# curves, so an element that touches only a face or nothing is reported but cannot be repaired.
LOCATIONS = ("point", "curve", "surface", "interior")
REPAIRABLE_LOCATIONS = ("point", "curve")


@dataclass(frozen=True)
class CurveCandidate:
    """A curve a bad element could be repaired through: its length and its edge length next to the element."""

    tag: int
    local_edge: (
        float  # longest 1D mesh edge of this curve that shares a node with the element
    )
    length: float  # total length of the curve


@dataclass(frozen=True)
class BadElement:
    """A tetrahedron below the threshold and everything the audit could say about where it is."""

    element: int  # Gmsh element tag
    quality: float
    edge_min: float
    edge_max: float
    centroid: tuple[float, float, float]
    location: str  # one of LOCATIONS
    points: tuple[int, ...]  # native tags of the entities the element's nodes sit on
    curves: tuple[int, ...]
    surfaces: tuple[int, ...]
    volumes: tuple[int, ...]
    sgb_ids: tuple[str, ...]  # planned ids of those entities
    groups: tuple[str, ...]  # physical-group names of the faces around the element
    feature_kinds: tuple[str, ...]  # kinds of the sidecar features covering its curves
    candidates: tuple[
        CurveCandidate, ...
    ]  # curves the repair loop may refine for this element

    @property
    def repairable(self) -> bool:
        """Whether the element touches a curve or a point, which is all rules/1 can refine."""
        return self.location in REPAIRABLE_LOCATIONS

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> BadElement:
        """Rebuild an element from its JSON form (``asdict`` of the record)."""
        fields = dict(raw)
        fields["centroid"] = tuple(fields["centroid"])
        for name in (
            "points",
            "curves",
            "surfaces",
            "volumes",
            "sgb_ids",
            "groups",
            "feature_kinds",
        ):
            fields[name] = tuple(fields[name])
        fields["candidates"] = tuple(CurveCandidate(**c) for c in fields["candidates"])
        return cls(**fields)


@dataclass(frozen=True)
class MeshAudit:
    """Quality of one generated mesh."""

    tetrahedra: int
    nodes: int
    min_quality: float  # minimum of the thresholded measure over all tetrahedra
    min_gamma: float  # inscribed-to-circumscribed radius ratio, reported beside it
    bad_count: int
    bad: tuple[
        BadElement, ...
    ]  # sorted by element tag; empty when attribution was not requested
    mesh_seconds: float
    audit_seconds: float

    def to_json(self) -> dict[str, Any]:
        """The JSON form of the audit."""
        return asdict(self)

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> MeshAudit:
        """Rebuild an audit from its JSON form."""
        fields = dict(raw)
        fields["bad"] = tuple(BadElement.from_json(b) for b in fields["bad"])
        return cls(**fields)
