"""The traces of the 3D model's emitted faces on the cut: horizontal faces as stretches, walls as points.

The cut crosses a face where it crosses the face's boundary curves (``rules.cross_curves`` returns them, each
crossed at a right angle).  In a face's loop the face lies on the left of every edge in its travel direction,
and a wall's positive side lies on the left of its top chain, which its loop travels from the run's start to
its end (``topology.assembly.builder``).  So each crossing says on which side of it, along the cut, the face or the
volume lies: a horizontal face's stretches begin and end at crossings, and a wall's two volumes stand on the
two sides of its point.

With both ends of the cut on the edge of the model, every stretch begins and ends at a crossing; a face whose
crossings do not pair up, or a wall whose top and bottom chains are crossed at different places, is a
structural failure the section's validation reports.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from fractions import Fraction

from scgsim.geometry.exact.plane import Point
from scgsim.geometry.sgb.model import ConformalModel
from scgsim.geometry.sgb.section.frame import CutFrame
from scgsim.geometry.sgb.topology.assembly.records import TopoEdge, TopoFace, Topology


@dataclass(frozen=True)
class HorizontalTrace:
    """The stretch ``u0 .. u1`` (grid units) of the cut that a horizontal face covers, at level index ``level``."""

    face: str
    level: int
    u0: int
    u1: int


@dataclass(frozen=True)
class WallTrace:
    """Where the cut crosses a wall: at ``u`` (grid units), from level index ``levels[0]`` to ``levels[1]``.

    ``positive_after`` says that the wall's positive side (``pos_volume``, ``pos_domain``) lies at larger ``u``.
    ``curves`` are the carriers of the chain edges the cut crosses there.
    """

    face: str
    u: int
    levels: tuple[int, int]
    positive_after: bool
    curves: tuple[str, ...]


@dataclass
class Traces:
    """Every trace of the emitted faces, and the structural failures found while tracing them."""

    horizontal: list[HorizontalTrace] = field(default_factory=list)
    walls: list[WallTrace] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def trace_faces(
    frame: CutFrame, model: ConformalModel, crossings: dict[str, int]
) -> Traces:
    """The traces of every emitted face of ``model`` whose boundary the cut crosses."""
    topology, routed = model.topology, model.routed
    level_of = {z: k for k, z in enumerate(topology.levels)}
    traces = Traces()
    for _, face in sorted(topology.faces.items()):
        if not routed.is_emitted_face(face.neg_volume, face.pos_volume):
            continue
        if face.kind == "wall":
            _trace_wall(frame, topology, face, crossings, level_of, traces)
        else:
            _trace_horizontal(frame, topology, face, crossings, level_of, traces)
    return traces


def _trace_horizontal(  # noqa: PLR0913 - the cut, the topology, the face and the tables it is traced with
    frame: CutFrame,
    topology: Topology,
    face: TopoFace,
    crossings: dict[str, int],
    level_of: dict[Fraction, int],
    traces: Traces,
) -> None:
    """The stretches of one horizontal face: from each crossing it lies after to the next it lies before."""
    events: list[tuple[int, bool]] = []  # (u, whether the face lies after it)
    for loop in face.loops:
        for edge_id, sign in loop:
            u = crossings.get(edge_id)
            if u is not None:
                direction = _travel(topology, topology.edges[edge_id], sign)
                events.append((u, frame.left_is_after(direction)))
    if not events:
        return
    # A stretch ends where the next one starts when an edge of the face has the face on both sides.
    events.sort(key=lambda event: (event[0], event[1]))
    paired = len(events) % 2 == 0 and all(
        after == (k % 2 == 0) for k, (_, after) in enumerate(events)
    )
    if not paired:
        traces.problems.append(
            f"horizontal face {face.id}: the crossings of its boundary do not pair into stretches "
            f"({[(u, 'starts' if after else 'ends') for u, after in events]})"
        )
        return
    level = level_of[face.z_range[0]]
    for (u0, _), (u1, _) in zip(events[0::2], events[1::2], strict=True):
        traces.horizontal.append(HorizontalTrace(face.id, level, u0, u1))


def _trace_wall(  # noqa: PLR0913 - the cut, the topology, the wall and the tables it is traced with
    frame: CutFrame,
    topology: Topology,
    wall: TopoFace,
    crossings: dict[str, int],
    level_of: dict[Fraction, int],
    traces: Traces,
) -> None:
    """The points where the cut crosses one wall, read from its top chain and checked against its bottom chain."""
    z_lo, z_hi = wall.z_range
    positive_after: dict[int, bool] = {}
    bottom: set[int] = set()
    curves: dict[int, set[str]] = defaultdict(set)
    for edge_id, sign in wall.loops[0]:
        u = crossings.get(edge_id)
        if u is None:
            continue
        edge = topology.edges[edge_id]
        curves[u].update(edge.sources)
        if topology.vertices[edge.v0].z == z_hi:
            # The loop travels the top chain from the run's start to its end; the positive side is its left.
            positive_after[u] = frame.left_is_after(_travel(topology, edge, sign))
        else:
            bottom.add(u)
    if set(positive_after) != bottom:
        traces.problems.append(
            f"wall {wall.id}: the cut crosses its top chain at u = {sorted(positive_after)} and its bottom "
            f"chain at u = {sorted(bottom)} grid units"
        )
        return
    levels = (level_of[z_lo], level_of[z_hi])
    for u, after in sorted(positive_after.items()):
        traces.walls.append(
            WallTrace(wall.id, u, levels, after, tuple(sorted(curves[u])))
        )


def _travel(topology: Topology, edge: TopoEdge, sign: int) -> Point:
    """The plan direction of an edge travelled with ``sign`` (+1 from ``v0`` to ``v1``)."""
    v0, v1 = topology.vertices[edge.v0], topology.vertices[edge.v1]
    dx, dy = int(v1.x) - int(v0.x), int(v1.y) - int(v0.y)
    return (dx, dy) if sign > 0 else (-dx, -dy)
