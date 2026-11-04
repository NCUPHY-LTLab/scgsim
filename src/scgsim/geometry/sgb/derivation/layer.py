"""One dissolved, labelled and identified planar arrangement (a slab or a level).

Owns the record every later stage reads: the minimal arrangement after
dissolution, its labels and content-addressed ids, the realised arc of every arc
edge, area-weighted membership, and the raw measurements validation compares
against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Literal

from scgsim.geometry.sgb.arrangement.arcs import RealisedArc
from scgsim.geometry.sgb.arrangement.dcel import Arrangement
from scgsim.geometry.sgb.labelling.dissolve import Dissolved
from scgsim.geometry.sgb.labelling.prism_set import PrismSet
from scgsim.geometry.sgb.noding.snap_rounding import NodedEdge


@dataclass
class RawMeasurements:
    """Pre-dissolution measurements (integer, exact) that validation compares against."""

    total_twice_area: int
    top_twice_area: int
    prism_twice_area: dict[int, int] | None = (
        None  # slab layers: prism index -> covered area
    )


@dataclass
class Layer:
    """One dissolved, labelled and identified planar arrangement."""

    kind: Literal["slab", "level"]
    index: int
    plane: str  # content key, e.g. "slab[0,100]" or "level[0]"
    z: Fraction | tuple[Fraction, Fraction]
    dissolved: Dissolved
    cells: PrismSet | None  # prisms of a slab layer
    raw: RawMeasurements
    vertex_ids: list[str] = field(default_factory=list)
    edge_ids: list[str] = field(default_factory=list)
    edge_sources: list[tuple[str, ...]] = field(default_factory=list)
    face_ids: list[str] = field(default_factory=list)
    membership: list[dict[str, float]] = field(
        default_factory=list
    )  # faces/cells: entity -> covered area fraction
    removed_edges: list[NodedEdge] = field(
        default_factory=list
    )  # slab layers: raw edges that dissolved (they cut cells at levels)
    edge_curve: list[RealisedArc | None] = field(
        default_factory=list
    )  # realised arc of each arc edge, else None
    below_cell: list[int] = field(
        default_factory=list
    )  # level layers: slab cell below each face
    above_cell: list[int] = field(
        default_factory=list
    )  # level layers: slab cell above each face
    stats: dict[str, int] = field(default_factory=dict)
    raw_arrangement: Arrangement | None = None  # kept only when the runtime asks for it
    raw_labels: list[Any] | None = None

    @property
    def level_z(self) -> Fraction:
        """The Z of a level layer (a slab layer has a Z interval, see ``z``)."""
        assert not isinstance(self.z, tuple), "a slab layer has no single Z"
        return self.z

    @property
    def arr(self) -> Arrangement:
        """The dissolved arrangement."""
        return self.dissolved.arrangement

    @property
    def labels(self) -> list[Any]:
        """Label of every face of the dissolved arrangement (index 0: unbounded face)."""
        return self.dissolved.labels
