"""Track lattice-edge provenance and certify polygon-face winding for grid geometry."""

from __future__ import annotations

import json
from bisect import bisect_left, bisect_right
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from scgsim.diagnostics import diagnostic_stage
from scgsim.geometry.grid.tagged_grid_operations import ShapelyGridOperations

if TYPE_CHECKING:
    from shapely.geometry import Polygon

LatticePoint = tuple[int, ...]
LatticeEdge = tuple[LatticePoint, LatticePoint]
EdgeOrigin = tuple[int, int]


@dataclass(frozen=True)
class SourcePath:
    """Carry one directed source segment and its arrangement nodes."""

    operand_id: str
    operand_index: int
    ring_index: int
    segment_index: int
    nodes: tuple[LatticePoint, ...]
    orientation: int


@dataclass(frozen=True)
class PointIndex:
    """Keep source points indexed along both coordinate axes."""

    by_x: tuple[LatticePoint, ...]
    by_y: tuple[LatticePoint, ...]
    xs: tuple[int, ...]
    ys: tuple[int, ...]

    @classmethod
    def build(cls, points: Iterable[LatticePoint]) -> PointIndex:
        """Deduplicate source points and index them in both axis orders."""
        by_x, by_y = (
            tuple(sorted(set(points))),
            tuple(sorted(set(points), key=lambda p: (p[1], p[0]))),
        )
        return cls(by_x, by_y, tuple(p[0] for p in by_x), tuple(p[1] for p in by_y))

    def candidates(
        self, xmin: int, ymin: int, xmax: int, ymax: int
    ) -> tuple[LatticePoint, ...]:
        """Select indexed source points inside the requested closed bounds."""
        xl, xr = bisect_left(self.xs, xmin), bisect_right(self.xs, xmax)
        yl, yr = bisect_left(self.ys, ymin), bisect_right(self.ys, ymax)
        rows = self.by_x[xl:xr] if xr - xl <= yr - yl else self.by_y[yl:yr]
        return tuple(p for p in rows if xmin <= p[0] <= xmax and ymin <= p[1] <= ymax)


@dataclass(frozen=True)
class EdgeProvenance:
    """Record native edges, their directed origins and winding transitions."""

    native_atoms: set[LatticeEdge]
    origins: dict[LatticeEdge, list[EdgeOrigin]]
    transitions: Mapping[LatticeEdge, Mapping[int, int]]
    grid_um: float


@dataclass(frozen=True)
class FaceArrangement:
    """Keep integer face loops, edge adjacency and operand winding together."""

    loops: tuple[tuple[tuple[LatticePoint, ...], ...], ...]
    left: dict[LatticeEdge, int]
    winding: dict[int, dict[int, int]]


def signed_ring_area(ring):
    """Compute the directed doubled area of a closed lattice ring."""
    return sum(
        a[0] * b[1] - a[1] * b[0]
        for a, b in zip(ring, (*ring[1:], ring[0]), strict=True)
    )


def edge_atoms(
    a: LatticePoint, b: LatticePoint, index: PointIndex
) -> tuple[LatticeEdge, ...]:
    """Split a collinear segment at indexed lattice points."""
    if a == b:
        return ()
    rows = [
        p
        for p in index.candidates(
            min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])
        )
        if (p[0] - a[0]) * (b[1] - a[1]) == (p[1] - a[1]) * (b[0] - a[0])
    ]
    axis = 0 if a[0] != b[0] else 1
    rows.sort(key=lambda p: p[axis], reverse=b[axis] < a[axis])
    return tuple((u, v) for u, v in zip(rows, rows[1:], strict=False) if u != v)


def refuse_graph(reason, **context):
    """Raise the established graph-construction error with structured context."""
    error = ValueError("tagged grid construction cannot establish " + reason)
    error.add_note("scgsim.grid.tagged_context: " + json.dumps(context, sort_keys=True))
    raise error


def build_edge_provenance(
    paths: Sequence[SourcePath],
    native_edges: Sequence[LatticeEdge],
    point_index: PointIndex,
    grid: float,
) -> EdgeProvenance:
    """Match native atomic edges to source paths and winding changes."""
    native_atoms: set[LatticeEdge] = {
        cast("LatticeEdge", tuple(sorted(edge)))
        for a, b in native_edges
        for edge in edge_atoms(a, b, point_index)
    }
    origins = defaultdict(list)
    transitions = defaultdict(lambda: defaultdict(int))
    for path_index, path in enumerate(paths):
        for a, b in zip(path.nodes, path.nodes[1:], strict=False):
            for u, v in edge_atoms(a, b, point_index):
                edge = tuple(sorted((u, v)))
                direction = 1 if (u, v) == edge else -1
                origins[edge].append((path_index, direction))
                transitions[edge][path.operand_index] += direction * path.orientation
    if set(origins) != native_atoms:
        refuse_graph(
            "exact native network provenance",
            missing_edges=len(native_atoms - set(origins)),
            extra_edges=len(set(origins) - native_atoms),
            grid_size_um=grid,
        )
    return EdgeProvenance(native_atoms, origins, transitions, grid)


def _face_rings(integer_faces):
    """Return exact lattice loops for native polygonized faces."""
    return tuple(
        tuple(
            tuple(tuple(round(value) for value in point) for point in ring.coords)[:-1]
            for ring in (face.exterior, *face.interiors)
        )
        for face in integer_faces
    )


def _face_edges(face_rings, native_atoms, point_index):
    """Map directed face edges to their unique adjacent face index."""
    left = {}
    for face_index, loops in enumerate(face_rings):
        for ring_index, ring in enumerate(loops):
            if (signed_ring_area(ring) > 0) != (ring_index == 0):
                ring = tuple(reversed(ring))
            for a, b in zip(ring, (*ring[1:], ring[0]), strict=True):
                for edge in edge_atoms(a, b, point_index):
                    if tuple(sorted(edge)) not in native_atoms or (
                        edge in left and left[edge] != face_index
                    ):
                        refuse_graph(
                            "unique face-edge incidence", face_index=face_index
                        )
                    left[edge] = face_index
    return left


def _face_links(native_atoms, left, origins, transitions):
    """Build neighboring-face winding transitions from directed edge ownership."""
    links = defaultdict(list)
    for edge in sorted(native_atoms):
        a, b = edge
        left_face, right_face = left.get((a, b), -1), left.get((b, a), -1)
        change = tuple(
            (owner, amount)
            for owner, amount in sorted(transitions[edge].items())
            if amount
        )
        if left_face == right_face:
            if change:
                refuse_graph(
                    "zero-area directed winding conservation",
                    origin_count=len(origins[edge]),
                )
            continue
        links[right_face].append((left_face, change))
        links[left_face].append(
            (right_face, tuple((owner, -amount) for owner, amount in change))
        )
    return links


def _face_windings(integer_faces, links):
    """Propagate source polygon membership from the unbounded face."""
    winding = {-1: {}}
    queue = deque((-1,))
    while queue:
        face_index = queue.popleft()
        for target, change in links[face_index]:
            state = dict(winding[face_index])
            for owner, amount in change:
                state[owner] = state.get(owner, 0) + amount
                if not state[owner]:
                    del state[owner]
            if any(amount != 1 for amount in state.values()):
                refuse_graph("binary source polygon winding", face_index=target)
            if target in winding:
                if winding[target] != state:
                    refuse_graph("consistent operand winding", face_index=target)
            else:
                winding[target] = state
                queue.append(target)
    if set(winding) != set(range(len(integer_faces))) | {-1}:
        refuse_graph("unbounded-face reachability", face_count=len(integer_faces))
    return winding


def build_face_arrangement(
    native_atoms,
    point_index,
    origins,
    transitions,
    operations: ShapelyGridOperations,
) -> FaceArrangement:
    """Polygonize the atomic graph and certify its face winding."""
    with diagnostic_stage(
        "geometry.grid.tagged_faces", counts={"edges": len(native_atoms)}
    ):
        # Polygonize the same atomic integer graph. This adds no new noding or
        # precision overlay; every face reuses these exact lattice identities.
        native_faces = tuple(
            operations.polygonize(
                [operations.line_string(edge) for edge in sorted(native_atoms)]
            ).geoms
        )
        face_loops = _face_rings(tuple(cast("Polygon", face) for face in native_faces))
        left = _face_edges(face_loops, native_atoms, point_index)
        links = _face_links(native_atoms, left, origins, transitions)
        winding = _face_windings(native_faces, links)
    return FaceArrangement(face_loops, left, winding)
