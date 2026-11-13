"""The shared, conformal boundary representation that assembly produces (design brief section 2).

Owns the topology records: vertices, edges, faces, shells and volumes with their
identities, planned for one model before any CAD kernel exists.  The records are
route independent: Route A, B and C only decide which volumes are emitted and which
primary group each face joins (``topology.routes``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Literal

from scgsim.geometry.sgb.arrangement.arcs import RealisedArc
from scgsim.geometry.sgb.exact.labels import Domain, SheetOwner

Loop = tuple[
    tuple[str, int], ...
]  # ((edge id, +1 / -1 relative to the edge's v0 -> v1 direction), ...)


@dataclass
class TopoVertex:
    """A vertex of a level plane, or the planned centre point of an arc edge."""

    id: str
    x: float  # grid units; an int for arrangement vertices, a float for the centre point of an arc edge
    y: float
    z: Fraction
    zs: int  # z times the common level denominator (exact integer for volume arithmetic)


@dataclass
class TopoEdge:
    """An edge: horizontal (on a level plane) or vertical (a wall column)."""

    id: str
    kind: Literal["horizontal", "vertical"]
    v0: str
    v1: str
    plane: str
    sources: tuple[str, ...]  # carrier ids
    curve: RealisedArc | None = None  # arc edges: the realised circle arc v0 -> v1
    centre_vertex: str | None = (
        None  # arc edges: the planned point carrying the arc's centre
    )


@dataclass
class TopoFace:
    """A horizontal face of a level or a vertical wall, with the volumes on its two sides."""

    id: str
    kind: Literal["horizontal", "wall"]
    plane: str
    loops: tuple[Loop, ...]  # the first loop is the outer boundary
    neg_domain: Domain | None  # domain on the side opposite the face normal
    pos_domain: Domain | None
    owner: SheetOwner | None
    neg_volume: str | None
    pos_volume: str | None
    planar: bool
    z_range: tuple[Fraction, Fraction]
    twice_xy_area: int = 0
    sources: tuple[str, ...] = ()  # source entity ids (lineage)
    membership: dict[str, float] = field(
        default_factory=dict
    )  # entity -> covered fraction (non-exclusive)
    bands: tuple[
        tuple[str, int], ...
    ] = ()  # inset band labels ((family id, band index), ...)
    cylinder: tuple[str, str] | None = (
        None  # curved wall: (bottom arc edge id, top arc edge id) of its one cylinder
    )
    bulge_volume: float = 0.0  # walls: sum of (arc segment area x height), + when the arcs bulge into the pos side

    @property
    def curved(self) -> bool:
        """Whether this is a wall on a cylinder."""
        return self.cylinder is not None


@dataclass
class Shell:
    """One closed surface bounding a volume (an outer shell or an inner void)."""

    faces: tuple[
        tuple[str, int], ...
    ]  # (face id, +1 if its normal points out of the volume)
    signed_volume: Fraction  # grid^2 * um
    role: Literal["outer", "inner"]


@dataclass
class TopoVolume:
    """A connected volume of one domain."""

    id: str
    domain: Domain
    cells: tuple[str, ...]
    shells: list[Shell] = field(default_factory=list)
    embedded_faces: list[str] = field(
        default_factory=list
    )  # sheets inside the volume that bound no shell


@dataclass
class Topology:
    """Vertices, edges, faces and volumes of one model, all with content-addressed ids."""

    grid_um: float
    levels: tuple[Fraction, ...]
    vertices: dict[str, TopoVertex]
    edges: dict[str, TopoEdge]
    faces: dict[str, TopoFace]
    volumes: dict[str, TopoVolume]
    lineage: dict[str, list[str]]  # edge id -> source carrier ids

    def counts(self) -> dict[str, int]:
        """Entity counts by kind."""
        return {
            "vertices": len(self.vertices),
            "edges": len(self.edges),
            "faces": len(self.faces),
            "volumes": len(self.volumes),
        }
