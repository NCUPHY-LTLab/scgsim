"""Half-edge structure and face extraction for a noded planar graph (stage S4).

Owns turning snap-rounded edges into a doubly-connected edge list (DCEL):
vertices, half-edges, boundary cycles, faces with holes and the nesting of
connected components inside faces.  Everything is exact integer arithmetic.

Conventions
-----------
* Half-edge ``2*e`` runs ``a -> b`` of edge ``e`` (``a < b``), ``2*e + 1`` runs back.
* A half-edge's face is on its *left*.  Bounded faces are bounded by
  counter-clockwise cycles of positive twice-area; every connected component
  also has exactly one cycle of non-positive area (its outer boundary seen from
  outside), which is a *hole* of the face that encloses the component.
* Face 0 is always the unbounded face.

Does not own labels (``arrangement.labelling``), dissolution or snapping.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from functools import cmp_to_key

from scgsim.geometry.models.refusals import internal
from scgsim.geometry.sgb.arrangement.noding.snap_rounding import NodedEdge
from scgsim.geometry.sgb.arrangement.noding.spatial_hash import cell_size_for
from scgsim.geometry.sgb.exact.plane import (
    Point,
    RepPoint,
    fraction_to_rep,
    point_in_ring,
)


@dataclass(frozen=True)
class Cycle:
    """One boundary cycle of half-edges with its signed area and connected component."""

    half_edges: tuple[int, ...]
    twice_area: int
    component: int


@dataclass(frozen=True)
class FaceRec:
    """A face: its outer cycle (``None`` for the unbounded face) and its hole cycles."""

    outer: int | None
    holes: tuple[int, ...]


class Arrangement:
    """A noded planar graph with faces; build with :func:`build_arrangement`."""

    def __init__(  # noqa: PLR0913 - the DCEL tables travel together
        self,
        vertices: list[Point],
        edges: list[NodedEdge],
        edge_ends: list[tuple[int, int]],
        cycles: list[Cycle],
        faces: list[FaceRec],
        he_face: list[int],
        he_cycle: list[int],
    ) -> None:
        self.vertices = vertices
        self.edges = edges
        self.edge_ends = edge_ends
        self.cycles = cycles
        self.faces = faces
        self.he_face = he_face
        self.he_cycle = he_cycle

    # -- half-edge accessors
    def origin(self, half_edge: int) -> Point:
        """Start point of a half-edge."""
        a, b = self.edge_ends[half_edge >> 1]
        return self.vertices[b if half_edge & 1 else a]

    def target(self, half_edge: int) -> Point:
        """End point of a half-edge."""
        a, b = self.edge_ends[half_edge >> 1]
        return self.vertices[a if half_edge & 1 else b]

    def cycle_points(self, cycle_index: int) -> list[Point]:
        """Vertices of a cycle in travel order."""
        return [self.origin(h) for h in self.cycles[cycle_index].half_edges]

    def face_cycles(self, face_index: int) -> list[int]:
        """Cycle indices bounding a face (outer first, then holes)."""
        face = self.faces[face_index]
        return ([face.outer] if face.outer is not None else []) + list(face.holes)

    def face_twice_area(self, face_index: int) -> int:
        """Twice the area of the face (outer minus holes); 0 for the unbounded face."""
        face = self.faces[face_index]
        if face.outer is None:
            return 0
        return self.cycles[face.outer].twice_area + sum(
            self.cycles[c].twice_area for c in face.holes
        )

    def face_segments(self, face_index: int) -> list[tuple[Point, Point]]:
        """All boundary segments of a face, travelled with the face on the left."""
        return [
            (self.origin(h), self.target(h))
            for c in self.face_cycles(face_index)
            for h in self.cycles[c].half_edges
        ]

    def total_bounded_twice_area(self) -> int:
        """Sum of twice the area of every bounded face."""
        return sum(self.face_twice_area(i) for i in range(1, len(self.faces)))

    def top_level_twice_area(self) -> int:
        """Area enclosed by components not nested in any bounded face (independent of nesting)."""
        return -sum(self.cycles[c].twice_area for c in self.faces[0].holes)

    def face_anchor(self, face_index: int) -> Point:
        """A point on the face's boundary that locates it in a message: the origin of its outer cycle's first half-edge."""
        outer = self.faces[face_index].outer
        assert outer is not None
        return self.origin(self.cycles[outer].half_edges[0])


# --------------------------------------------------------------------------- construction


def _half(dx: int, dy: int) -> int:
    return 0 if (dy > 0 or (dy == 0 and dx > 0)) else 1


def _angle_compare(d1: tuple[int, int], d2: tuple[int, int]) -> int:
    h1, h2 = _half(*d1), _half(*d2)
    if h1 != h2:
        return h1 - h2
    turn = d1[0] * d2[1] - d1[1] * d2[0]
    return -1 if turn > 0 else (1 if turn < 0 else 0)


def _compare_by_direction(
    first: tuple[tuple[int, int], int], second: tuple[tuple[int, int], int]
) -> int:
    """Order (direction, half-edge) pairs by the counter-clockwise angle of their direction."""
    return _angle_compare(first[0], second[0])


_ANGLE_KEY = cmp_to_key(_compare_by_direction)


def build_arrangement(edges: list[NodedEdge]) -> Arrangement:
    """DCEL, cycles, faces and component nesting for noded edges (``a < b`` each)."""
    edges = sorted(edges, key=lambda e: (e.a, e.b))
    vertices = sorted({p for e in edges for p in (e.a, e.b)})
    vertex_index = {p: i for i, p in enumerate(vertices)}
    edge_ends = [(vertex_index[e.a], vertex_index[e.b]) for e in edges]
    next_half_edge = _next_half_edges(vertices, edge_ends)
    component = _components(len(vertices), edge_ends)
    he_cycle, cycles = _trace_cycles(vertices, edge_ends, next_half_edge, component)
    faces, he_face = _faces_from_cycles(vertices, edge_ends, cycles, he_cycle)
    return Arrangement(vertices, edges, edge_ends, cycles, faces, he_face, he_cycle)


def _next_half_edges(
    vertices: list[Point], edge_ends: list[tuple[int, int]]
) -> list[int]:
    """For each half-edge, the next half-edge of its face boundary (face on the left)."""
    outgoing: list[list[int]] = [[] for _ in vertices]
    for edge_index, (a, b) in enumerate(edge_ends):
        outgoing[a].append(2 * edge_index)
        outgoing[b].append(2 * edge_index + 1)

    def direction(half_edge: int) -> tuple[int, int]:
        a, b = edge_ends[half_edge >> 1]
        pa, pb = vertices[a], vertices[b]
        if half_edge % 2 == 0:
            return (pb[0] - pa[0], pb[1] - pa[1])
        return (pa[0] - pb[0], pa[1] - pb[1])

    position: dict[int, int] = {}
    for fan in outgoing:
        if len(fan) > 2:
            decorated = sorted(((direction(h), h) for h in fan), key=_ANGLE_KEY)
            fan[:] = [h for _, h in decorated]
        for i, h in enumerate(fan):
            position[h] = i

    next_half_edge = [0] * (2 * len(edge_ends))
    for h in range(len(next_half_edge)):
        head = edge_ends[h >> 1][0 if h & 1 else 1]
        fan = outgoing[head]
        next_half_edge[h] = fan[(position[h ^ 1] - 1) % len(fan)]
    return next_half_edge


def _trace_cycles(
    vertices: list[Point],
    edge_ends: list[tuple[int, int]],
    next_half_edge: list[int],
    component: list[int],
) -> tuple[list[int], list[Cycle]]:
    """Follow ``next`` from every unvisited half-edge; returns (cycle of each half-edge, cycles)."""
    he_cycle = [-1] * len(next_half_edge)
    cycles: list[Cycle] = []
    for start in range(len(next_half_edge)):
        if he_cycle[start] >= 0:
            continue
        sequence: list[int] = []
        area = 0
        half_edge = start
        while he_cycle[half_edge] < 0:
            he_cycle[half_edge] = len(cycles)
            sequence.append(half_edge)
            a, b = edge_ends[half_edge >> 1]
            origin, target = (b, a) if half_edge & 1 else (a, b)
            po, pt = vertices[origin], vertices[target]
            area += po[0] * pt[1] - pt[0] * po[1]
            half_edge = next_half_edge[half_edge]
        cycles.append(Cycle(tuple(sequence), area, component[edge_ends[start >> 1][0]]))
    return he_cycle, cycles


def _components(count: int, edge_ends: list[tuple[int, int]]) -> list[int]:
    parent = list(range(count))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edge_ends:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)
    return [find(i) for i in range(count)]


class BoxIndex:
    """Uniform hash of bounding boxes for point-stabbing queries."""

    def __init__(self, boxes: list[tuple[int, int, int, int]]) -> None:
        extent = max([1] + [max(b[2] - b[0], b[3] - b[1]) for b in boxes])
        x_min = min([b[0] for b in boxes] + [0])
        x_max = max([b[2] for b in boxes] + [1])
        y_min = min([b[1] for b in boxes] + [0])
        y_max = max([b[3] for b in boxes] + [1])
        self.size = cell_size_for(max(x_max - x_min, y_max - y_min, extent), len(boxes))
        self.boxes = boxes
        self.cells: dict[tuple[int, int], list[int]] = defaultdict(list)
        self.large: list[int] = []
        size = self.size
        for i, (x0, y0, x1, y1) in enumerate(boxes):
            nx, ny = x1 // size - x0 // size + 1, y1 // size - y0 // size + 1
            if nx * ny > 64:
                self.large.append(i)
                continue
            for cx in range(x0 // size, x1 // size + 1):
                for cy in range(y0 // size, y1 // size + 1):
                    self.cells[(cx, cy)].append(i)

    def stab(self, x: int, y: int) -> list[int]:
        """Indices of the boxes that contain the point."""
        found = []
        for i in self.cells.get((x // self.size, y // self.size), ()):
            b = self.boxes[i]
            if b[0] <= x <= b[2] and b[1] <= y <= b[3]:
                found.append(i)
        for i in self.large:
            b = self.boxes[i]
            if b[0] <= x <= b[2] and b[1] <= y <= b[3]:
                found.append(i)
        return found


def _faces_from_cycles(
    vertices: list[Point],
    edge_ends: list[tuple[int, int]],
    cycles: list[Cycle],
    he_cycle: list[int],
) -> tuple[list[FaceRec], list[int]]:
    """Faces are the positive cycles; every component's outer cycle becomes a hole of its parent face."""
    positive = [i for i, c in enumerate(cycles) if c.twice_area > 0]
    outer_of_component: dict[int, int] = {}
    for i, cycle in enumerate(cycles):
        if cycle.twice_area <= 0:
            outer_of_component[cycle.component] = i

    def ring_points(cycle_index: int) -> list[Point]:
        points = []
        for h in cycles[cycle_index].half_edges:
            a, b = edge_ends[h >> 1]
            points.append(vertices[b if h & 1 else a])
        return points

    rings = {ci: ring_points(ci) for ci in positive}
    boxes = []
    for ci in positive:
        xs = [p[0] for p in rings[ci]]
        ys = [p[1] for p in rings[ci]]
        boxes.append((min(xs), min(ys), max(xs), max(ys)))
    index = BoxIndex(boxes)

    face_of_positive = {ci: k + 1 for k, ci in enumerate(positive)}
    holes: dict[int, list[int]] = defaultdict(list)
    for comp in sorted(outer_of_component):
        outer_cycle = outer_of_component[comp]
        probe = min(ring_points(outer_cycle))
        best = None
        for k in index.stab(*probe):
            candidate = positive[k]
            if cycles[candidate].component == comp:
                continue
            if not point_in_ring(rings[candidate], (probe[0], probe[1], 1)):
                continue
            if best is None or cycles[candidate].twice_area < cycles[best].twice_area:
                best = candidate
        parent_face = 0 if best is None else face_of_positive[best]
        holes[parent_face].append(outer_cycle)

    faces = [FaceRec(None, tuple(holes[0]))]
    for ci in positive:
        faces.append(FaceRec(ci, tuple(holes[face_of_positive[ci]])))
    he_face = [0] * len(he_cycle)
    for face_index, face in enumerate(faces):
        for ci in ([face.outer] if face.outer is not None else []) + list(face.holes):
            for h in cycles[ci].half_edges:
                he_face[h] = face_index
    return faces, he_face


# --------------------------------------------------------------------------- representative point


def interior_point(arr: Arrangement, face_index: int) -> RepPoint:
    """Deterministic exact point strictly inside a bounded face.

    Scan the lowest horizontal strip between distinct vertex heights of the
    face's boundary: no boundary vertex lies inside the strip, so every spanning
    edge crosses it.  Even-odd over the sorted crossings gives the interior
    intervals; the widest one (leftmost on ties) gives the point.
    """
    for _, mid, intervals in _strip_intervals(arr.face_segments(face_index)):
        if intervals:
            left, right = max(intervals, key=lambda i: i[1] - i[0])
            return fraction_to_rep((left + right) / 2, mid)
    raise ValueError(internal("face has no interior point (degenerate face)"))


# Where, across a cell's width, its sample points lie (fractions of the width from the left).
SAMPLE_FRACTIONS = (Fraction(1, 2), Fraction(1, 4), Fraction(3, 4), Fraction(1, 8))


@dataclass(frozen=True)
class StripCell:
    """One trapezoid of a face: a strip between two vertex heights, cut to one interior interval."""

    area: Fraction  # exact
    points: tuple[RepPoint, ...]  # strictly inside, spread over the width at mid height


def strip_cells(arr: Arrangement, face_index: int) -> list[StripCell]:
    """The face cut into trapezoids whose exact areas add up to the face's area."""
    cells = []
    for height, mid, intervals in _strip_intervals(arr.face_segments(face_index)):
        for left, right in intervals:
            width = right - left
            points = tuple(
                fraction_to_rep(left + width * fraction, mid)
                for fraction in SAMPLE_FRACTIONS
            )
            cells.append(StripCell(width * height, points))
    return cells


def sample_points(arr: Arrangement, face_index: int) -> Iterator[RepPoint]:
    """Exact points strictly inside a face, bottom strip first, produced only as far as they are read.

    A caller that needs one point with a property (clear of a band boundary) stops at the first that has it,
    so a face is cut into strips only as far as that.
    """
    for _, mid, intervals in _strip_intervals(arr.face_segments(face_index)):
        for left, right in intervals:
            width = right - left
            for fraction in SAMPLE_FRACTIONS:
                yield fraction_to_rep(left + width * fraction, mid)


def _strip_intervals(
    segments: list[tuple[Point, Point]],
) -> Iterator[tuple[Fraction, Fraction, list[tuple[Fraction, Fraction]]]]:
    """Per strip between distinct vertex heights, bottom up: its height, mid height and interior intervals.

    No boundary vertex lies inside a strip, so every spanning edge crosses it; even-odd over the sorted
    crossings at mid height gives the intervals of positive width (left to right).
    """
    heights = sorted({p[1] for s in segments for p in s})
    for lo, hi in zip(heights, heights[1:], strict=False):
        mid = Fraction(lo + hi, 2)
        crossings: list[Fraction] = []
        for (px, py), (qx, qy) in segments:
            if py == qy:
                continue
            if min(py, qy) <= lo and max(py, qy) >= hi:
                crossings.append(px + Fraction((qx - px) * (mid - py), qy - py))
        crossings.sort()
        intervals = [
            (crossings[i], crossings[i + 1])
            for i in range(0, len(crossings) - 1, 2)
            if crossings[i + 1] > crossings[i]
        ]
        yield Fraction(hi - lo), mid, intervals
