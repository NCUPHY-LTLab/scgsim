"""Iterated snap rounding on the integer grid (design brief section 3, stage S4 noding).

Owns turning a set of exact integer line segments into a *noded* set of edges
whose vertices are all grid points, with no two edges crossing and no edge
passing through a hot pixel it does not end in.

Algorithm (Hobby 1999; Guibas and Marimont 1998; Halperin and Packer 2002):

* A *hot pixel* is the half-open unit square ``[c-1/2, c+1/2)^2`` centred on a
  grid point ``c`` that holds a segment endpoint or a segment crossing.
* Crossings are computed exactly with integer arithmetic and rounded to the
  pixel that contains them (half rounds up).
* Every segment is re-routed through the centres of all hot pixels it touches,
  ordered along the segment.  Re-routed chains can touch hot pixels the original
  segment missed, so the process is iterated until nothing changes.  Each round
  only re-examines edges that changed in the previous round or that touch a
  newly hot pixel.
* Every output edge keeps the set of carrier ids of the input segments that
  produced it; coincident output edges merge their carrier sets.

The two heavy kernels (crossing detection, hot-pixel hit search) are pure
functions over lists, so ``noding.tiling`` can run them per tile and union the
results without changing the answer.

Does not own face extraction (``arrangement``) or any labelling.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import NamedTuple, Protocol

from scgsim.geometry.sgb.noding.spatial_hash import (
    HotGrid,
    cell_size_for,
    cells_along,
    points_extent,
)
from scgsim.geometry.sgb.plane import Point
from scgsim.geometry.sgb.stopwatch import Stopwatch, elapsed_seconds

EdgeKey = tuple[Point, Point]  # canonical: a < b lexicographically
Crossing = tuple[int, int, int]  # exact point (X / den, Y / den) stored as (X, Y, den)
TileBox = tuple[
    int, int, int, int
]  # x_lo, y_lo, x_hi, y_hi; pixel centres with lo <= c < hi
MAX_ROUNDS = 500
COORDINATE_LIMIT = (
    1 << 40
)  # clamp for the hash-cell index of a crossing, whatever its magnitude


class Segment(NamedTuple):
    """An input segment with the ids of the carriers it came from."""

    a: Point
    b: Point
    carriers: frozenset[str]


class NodedEdge(NamedTuple):
    """An output edge between two grid points (``a < b``) with its carrier ids."""

    a: Point
    b: Point
    carriers: frozenset[str]


@dataclass(frozen=True)
class SnapResult:
    """Noded edges plus how much work the rounding took."""

    edges: list[NodedEdge]
    hot_pixels: int
    rounds: int


class RoundEngine(Protocol):
    """Where the two kernels run: in-process or tile-parallel (see ``noding.tiling``)."""

    def crossing_pixels(self, keys: list[EdgeKey], changed: list[bool]) -> set[Point]:
        """Pixels containing proper crossings of pairs with at least one changed edge."""
        ...

    def hits(
        self,
        keys: list[EdgeKey],
        changed: list[bool],
        hot: set[Point],
        delta: set[Point],
    ) -> list[list[Point]]:
        """Hot pixel centres each edge touches."""
        ...


# --------------------------------------------------------------------------- kernels


def proper_crossing(a1: Point, b1: Point, a2: Point, b2: Point) -> Crossing | None:
    """Exact proper-crossing point ``(X, Y, den)`` (point = X/den, Y/den) or ``None``."""
    d1x, d1y = b1[0] - a1[0], b1[1] - a1[1]
    d2x, d2y = b2[0] - a2[0], b2[1] - a2[1]
    den = d1x * d2y - d1y * d2x
    if den == 0:
        return None
    ex, ey = a2[0] - a1[0], a2[1] - a1[1]
    t_num = ex * d2y - ey * d2x
    u_num = ex * d1y - ey * d1x
    if den < 0:
        den, t_num, u_num = -den, -t_num, -u_num
    if not (0 < t_num < den and 0 < u_num < den):
        return None
    return a1[0] * den + d1x * t_num, a1[1] * den + d1y * t_num, den


def crossing_pixel(crossing: Crossing) -> Point:
    """The hot pixel that holds an exact crossing point: each coordinate rounded half up."""
    x_num, y_num, den = crossing
    return (2 * x_num + den) // (2 * den), (2 * y_num + den) // (2 * den)


def _clamped_cell(value: int, size: int) -> int:
    return max(-COORDINATE_LIMIT, min(COORDINATE_LIMIT, value)) // size


def crossing_pixels_kernel(
    keys: Sequence[EdgeKey],
    changed: Sequence[bool],
    box: TileBox | None = None,
) -> set[Point]:
    """Pixels containing proper crossings of pairs with at least one changed edge.

    ``box`` keeps only pixels whose centre satisfies ``x_lo <= x < x_hi`` and
    ``y_lo <= y < y_hi`` (tile ownership); ``None`` keeps all.
    """
    count = len(keys)
    if count < 2:
        return set()
    size = cell_size_for(points_extent(p for key in keys for p in key), count)
    bounds = None
    if box is not None:
        # One cell of slack: a crossing may lie up to half a pixel outside its pixel's tile.
        bounds = (
            _clamped_cell(box[0], size) - 1,
            _clamped_cell(box[1], size) - 1,
            _clamped_cell(box[2], size) + 1,
            _clamped_cell(box[3], size) + 1,
        )
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, (a, b) in enumerate(keys):
        for cell in cells_along(a, b, size, bounds):
            grid[cell].append(i)
    found: set[Point] = set()
    for cell, members in grid.items():
        if len(members) >= 2:
            found |= _crossings_in_cell(keys, changed, members, (cell, size), box)
    return found


def _crossings_in_cell(
    keys: Sequence[EdgeKey],
    changed: Sequence[bool],
    members: list[int],
    hash_cell: tuple[tuple[int, int], int],
    box: TileBox | None,
) -> set[Point]:
    """Crossing pixels of the pairs whose crossing point lies in this hash cell."""
    cell, size = hash_cell
    found: set[Point] = set()
    for ii, i in enumerate(members):
        a1, b1 = keys[i]
        for j in members[ii + 1 :]:
            if not (changed[i] or changed[j]):
                continue
            a2, b2 = keys[j]
            hit = proper_crossing(a1, b1, a2, b2)
            if hit is None:
                continue
            x_num, y_num, den = hit
            # Each pair is processed only in the hash cell that contains its crossing point.
            if (x_num // (den * size), y_num // (den * size)) != cell:
                continue
            pixel = crossing_pixel(hit)
            if box is None or (
                box[0] <= pixel[0] < box[2] and box[1] <= pixel[1] < box[3]
            ):
                found.add(pixel)
    return found


def hits_kernel(
    keys: Sequence[EdgeKey],
    changed: Sequence[bool],
    hot: Iterable[Point],
    delta: Iterable[Point],
) -> list[list[Point]]:
    """Hot pixel centres each edge touches.

    Changed edges are tested against all hot pixels; unchanged edges only against
    the pixels that became hot since they were last routed.
    """
    hot_grid = HotGrid(hot) if any(changed) else None
    delta_points = list(delta)
    delta_grid = HotGrid(delta_points) if delta_points else None
    hits: list[list[Point]] = []
    for (a, b), is_changed in zip(keys, changed, strict=False):
        grid = hot_grid if is_changed else delta_grid
        hits.append(grid.touching(a, b) if grid is not None else [])
    return hits


def assemble_chain(a: Point, b: Point, hits: Iterable[Point]) -> tuple[Point, ...]:
    """Order hot pixel centres along segment ab (unique by construction, see module docs)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    points = set(hits)
    points.add(a)
    points.add(b)
    return tuple(
        sorted(points, key=lambda c: ((c[0] - a[0]) * dx + (c[1] - a[1]) * dy, c))
    )


class SerialEngine:
    """Runs the kernels in the calling process (optionally booking their time on a stopwatch)."""

    def __init__(self, watch: Stopwatch | None = None) -> None:
        self.watch = watch

    def crossing_pixels(self, keys: list[EdgeKey], changed: list[bool]) -> set[Point]:
        """Run the crossing kernel over all edges."""
        started = time.perf_counter()
        pixels = crossing_pixels_kernel(keys, changed)
        self._book(started)
        return pixels

    def hits(
        self,
        keys: list[EdgeKey],
        changed: list[bool],
        hot: set[Point],
        delta: set[Point],
    ) -> list[list[Point]]:
        """Run the hit kernel over all edges."""
        started = time.perf_counter()
        hits = hits_kernel(keys, changed, hot, delta)
        self._book(started)
        return hits

    def _book(self, started: float) -> None:
        if self.watch is not None:
            seconds = elapsed_seconds(started)
            self.watch.add("noding_kernels", seconds, parallel=True)
            self.watch.add_work("noding_kernels", seconds)


# --------------------------------------------------------------------------- driver


def canonical_key(a: Point, b: Point) -> EdgeKey:
    """The edge between two points with its endpoints in lexicographic order."""
    return (a, b) if a < b else (b, a)


def snap_round(
    segments: Iterable[Segment],
    engine: RoundEngine | None = None,
    max_rounds: int = MAX_ROUNDS,
) -> SnapResult:
    """Iterated snap rounding of integer segments; see the module docstring."""
    return _IteratedRounding(segments, engine or SerialEngine(), max_rounds).run()


class _IteratedRounding:
    """The state that travels through the rounds: edges, hot pixels, what changed."""

    def __init__(
        self, segments: Iterable[Segment], engine: RoundEngine, max_rounds: int
    ) -> None:
        self.engine = engine
        self.max_rounds = max_rounds
        self.edges: dict[EdgeKey, set[str]] = {}
        for seg in segments:
            if seg.a != seg.b:
                self.edges.setdefault(canonical_key(seg.a, seg.b), set()).update(
                    seg.carriers
                )
        self.hot: set[Point] = {p for key in self.edges for p in key}
        self.changed: set[EdgeKey] = set(self.edges)
        self.delta: set[Point] = set(self.hot)

    def run(self) -> SnapResult:
        """Repeat rounds until no edge and no hot pixel changes."""
        rounds = 0
        while self.changed or self.delta:
            rounds += 1
            if rounds > self.max_rounds:
                raise RuntimeError("snap rounding did not converge")
            self._round()
        edges = [
            NodedEdge(a, b, frozenset(carriers))
            for (a, b), carriers in sorted(self.edges.items())
        ]
        return SnapResult(edges, len(self.hot), rounds)

    def _round(self) -> None:
        keys = sorted(self.edges)
        flags = [key in self.changed for key in keys]
        new_hot = self.engine.crossing_pixels(keys, flags) - self.hot
        self.hot |= new_hot
        self.delta |= new_hot
        chain_hits = self.engine.hits(keys, flags, self.hot, self.delta)
        rerouted: dict[EdgeKey, set[str]] = {}
        verified: set[EdgeKey] = (
            set()
        )  # edges whose route was re-checked and left unchanged
        for key, hits in zip(keys, chain_hits, strict=False):
            chain = assemble_chain(key[0], key[1], hits)
            if len(chain) == 2:
                verified.add(key)
            for u, v in zip(chain, chain[1:], strict=False):
                rerouted.setdefault(canonical_key(u, v), set()).update(self.edges[key])
        # An edge that is only produced as a piece of another chain has not been checked
        # against the hot pixels yet, even if an edge with the same endpoints existed before.
        self.changed = set(rerouted) - verified
        self.delta = set()
        self.edges = rerouted
