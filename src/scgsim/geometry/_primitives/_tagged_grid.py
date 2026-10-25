"""One native grid network with independently constructed directed provenance.

Pixel incidence is an exact half-open slab construction, not a replay of GEOS
private segment history. Missing context is refused, never assigned by samples.
"""

from bisect import bisect_left, bisect_right
from collections import defaultdict, deque
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
import json

from scgsim.diagnostics import _progress, _stage
from scgsim.preparation import _ordered_map, current_preparation_workers


@dataclass(frozen=True)
class SourcePath:
    operand_id: str
    operand_index: int
    ring_index: int
    segment_index: int
    nodes: tuple
    orientation: int


@dataclass(frozen=True)
class TaggedSubdivision:
    cells: tuple
    boundary: tuple
    auxiliary: tuple
    source_paths: tuple[SourcePath, ...]
    edge_origins: tuple


@dataclass(frozen=True)
class _PointIndex:
    by_x: tuple
    by_y: tuple
    xs: tuple
    ys: tuple

    @classmethod
    def build(cls, points):
        by_x, by_y = (
            tuple(sorted(set(points))),
            tuple(sorted(set(points), key=lambda p: (p[1], p[0]))),
        )
        return cls(by_x, by_y, tuple(p[0] for p in by_x), tuple(p[1] for p in by_y))

    def candidates(self, xmin, ymin, xmax, ymax):
        xl, xr = bisect_left(self.xs, xmin), bisect_right(self.xs, xmax)
        yl, yr = bisect_left(self.ys, ymin), bisect_right(self.ys, ymax)
        rows = self.by_x[xl:xr] if xr - xl <= yr - yl else self.by_y[yl:yr]
        return tuple(p for p in rows if xmin <= p[0] <= xmax and ymin <= p[1] <= ymax)


def _line_parts(value):
    if value.is_empty:
        return ()
    if value.geom_type in {"LineString", "LinearRing"}:
        return (value,)
    if value.geom_type in {"MultiLineString", "GeometryCollection"}:
        return tuple(p for child in value.geoms for p in _line_parts(child))
    if value.geom_type in {"Point", "MultiPoint"}:
        return ()
    raise ValueError("tagged grid native boundary network is not lineal")


def _signed(ring):
    return sum(a[0] * b[1] - a[1] * b[0] for a, b in zip(ring, (*ring[1:], ring[0])))


def _pixel_interval(a, b, pixel):
    """Intersect [0,1] with two exact left/bottom-closed, right/top-open slabs."""
    low, high, low_open, high_open = Fraction(0), Fraction(1), False, False
    for start, end, center in zip(a, b, pixel, strict=True):
        direction = end - start
        left, right = Fraction(2 * center - 1, 2), Fraction(2 * center + 1, 2)
        if not direction:
            if not left <= start < right:
                return None
            continue
        lower, upper = (left - start) / direction, (right - start) / direction
        lower_open, upper_open = False, True
        if direction < 0:
            lower, upper, lower_open, upper_open = upper, lower, upper_open, lower_open
        if lower > low:
            low, low_open = lower, lower_open
        elif lower == low:
            low_open = low_open or lower_open
        if upper < high:
            high, high_open = upper, upper_open
        elif upper == high:
            high_open = high_open or upper_open
        if low > high or (low == high and (low_open or high_open)):
            return None
    return low, high


def _edge_atoms(a, b, index):
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
    return tuple((u, v) for u, v in zip(rows, rows[1:]) if u != v)


def _refuse(reason, **context):
    error = ValueError("tagged grid construction cannot establish " + reason)
    error.add_note("scgsim.grid.tagged_context: " + json.dumps(context, sort_keys=True))
    raise error


def _source_paths(rows, index, scale):
    half = Fraction(1, 2)
    output = []
    for (
        value,
        operand_index,
        ring_index,
        segment_index,
        a,
        b,
        endpoints,
        orientation,
    ) in rows:
        # Bind the native scale-first binary64 operation; exact predicates start
        # after this operation, including decimal half ties.
        start, end = (
            tuple(Fraction(v * scale) for v in a),
            tuple(Fraction(v * scale) for v in b),
        )
        candidates = index.candidates(
            min(start[0], end[0]) - half,
            min(start[1], end[1]) - half,
            max(start[0], end[0]) + half,
            max(start[1], end[1]) + half,
        )
        hits = []
        for pixel in candidates:
            interval = _pixel_interval(start, end, pixel)
            if interval is not None:
                hits.append((*interval, pixel))
        hits.sort()
        path = tuple(pixel for _, _, pixel in hits)
        context = dict(
            operand_index=operand_index,
            ring_index=ring_index,
            segment_index=segment_index,
        )
        if not path or (path[0], path[-1]) != endpoints:
            _refuse("source endpoint pixel incidence", **context)
        if any(second[0] < first[1] for first, second in zip(hits, hits[1:])):
            _refuse("unique source pixel order", **context)
        output.append(
            SourcePath(
                value, operand_index, ring_index, segment_index, path, orientation
            )
        )
    return tuple(output)


def build_tagged_subdivision(values, geometries, grid):
    from shapely import (
        LineString,
        Point,
        Polygon,
        get_coordinates,
        get_precision,
        points as make_points,
        polygonize,
        set_precision,
        union_all,
    )

    raw_rows, vertices = [], []
    for operand_index, value in enumerate(values):
        shape = geometries[value.operand_id]
        for ring_index, ring in enumerate((shape.exterior, *shape.interiors)):
            points = tuple(tuple(p) for p in ring.coords)[:-1]
            if not points:
                continue
            area = _signed(tuple(tuple(Fraction(v) for v in point) for point in points))
            if not area:
                _refuse(
                    "nondegenerate source ring direction",
                    operand_index=operand_index,
                    ring_index=ring_index,
                )
            orientation = (1 if ring_index == 0 else -1) * (1 if area > 0 else -1)
            for segment_index, (a, b) in enumerate(
                zip(points, (*points[1:], points[0]))
            ):
                raw_rows.append(
                    (
                        value.operand_id,
                        operand_index,
                        ring_index,
                        segment_index,
                        a,
                        b,
                        orientation,
                    )
                )
                vertices.extend((a, b))
    if not raw_rows:
        return TaggedSubdivision((), (), (), (), ())
    with _stage("geometry.grid.tagged_network", counts={"segments": len(raw_rows)}):
        try:
            native = union_all(
                [LineString((row[4], row[5])) for row in raw_rows], grid_size=grid
            )
        except Exception as error:
            try:
                error.add_note(
                    "scgsim.grid.tagged_network: "
                    + json.dumps(
                        dict(
                            operation="union_all",
                            input_dimension=1,
                            segment_count=len(raw_rows),
                            grid_size_um=grid,
                        ),
                        sort_keys=True,
                    )
                )
            except Exception:
                pass
            raise
        # Native point rounding is observed AFTER the global network is built.
        # It supplies endpoint metadata/ties, never independently rounded inputs.
        rounded = set_precision(make_points(vertices), grid, mode="pointwise")
        native_grid = get_precision(rounded[0])
        scale = 1 / native_grid
        key = lambda point: tuple(round(v * scale) for v in point)
        endpoint_keys = tuple(key(tuple(p.coords[0])) for p in rounded)
        native_edges = tuple(
            (key(a), key(b))
            for line in _line_parts(native)
            for a, b in zip(line.coords, tuple(line.coords)[1:])
            if key(a) != key(b)
        )
        pixels = {key(tuple(p)) for p in get_coordinates(native)} | set(endpoint_keys)
    point_index = _PointIndex.build(pixels)
    rows = tuple(
        (*row[:6], (endpoint_keys[2 * i], endpoint_keys[2 * i + 1]), row[6])
        for i, row in enumerate(raw_rows)
    )
    workers = current_preparation_workers() or 1
    size = (len(rows) + workers - 1) // workers
    chunks = tuple(rows[start : start + size] for start in range(0, len(rows), size))
    with _stage(
        "geometry.grid.tagged_paths",
        counts={"segments": len(rows), "pixels": len(pixels)},
    ):
        paths = tuple(
            path
            for chunk in _ordered_map(
                lambda chunk: _source_paths(chunk, point_index, scale), chunks
            )
            for path in chunk
        )
    rings = defaultdict(list)
    for path in paths:
        rings[(path.operand_index, path.ring_index)].append(path)
    for (operand_index, ring_index), ring in rings.items():
        for first, second in zip(ring, (*ring[1:], ring[0])):
            if first.nodes[-1] != second.nodes[0]:
                _refuse(
                    "closed source ring order",
                    operand_index=operand_index,
                    ring_index=ring_index,
                )
    native_atoms = {
        tuple(sorted(edge))
        for a, b in native_edges
        for edge in _edge_atoms(a, b, point_index)
    }
    origins = defaultdict(list)
    transitions = defaultdict(lambda: defaultdict(int))
    for path_index, path in enumerate(paths):
        for a, b in zip(path.nodes, path.nodes[1:]):
            for u, v in _edge_atoms(a, b, point_index):
                edge = tuple(sorted((u, v)))
                direction = 1 if (u, v) == edge else -1
                origins[edge].append((path_index, direction))
                transitions[edge][path.operand_index] += direction * path.orientation
    if set(origins) != native_atoms:
        _refuse(
            "exact native network provenance",
            missing_edges=len(native_atoms - set(origins)),
            extra_edges=len(set(origins) - native_atoms),
            grid_size_um=grid,
        )
    with _stage("geometry.grid.tagged_faces", counts={"edges": len(native_atoms)}):
        # Polygonize the same atomic integer graph. This adds no new noding or
        # precision overlay; every face reuses these exact lattice identities.
        integer_faces = tuple(
            polygonize([LineString(edge) for edge in sorted(native_atoms)]).geoms
        )
        left = {}
        face_rings = []
        for face_index, face in enumerate(integer_faces):
            loops = tuple(
                tuple(tuple(round(v) for v in p) for p in ring.coords)[:-1]
                for ring in (face.exterior, *face.interiors)
            )
            face_rings.append(loops)
            for ring_index, ring in enumerate(loops):
                if (_signed(ring) > 0) != (ring_index == 0):
                    ring = tuple(reversed(ring))
                for a, b in zip(ring, (*ring[1:], ring[0])):
                    for edge in _edge_atoms(a, b, point_index):
                        if tuple(sorted(edge)) not in native_atoms or (
                            edge in left and left[edge] != face_index
                        ):
                            _refuse("unique face-edge incidence", face_index=face_index)
                        left[edge] = face_index
        links = defaultdict(list)
        for edge in sorted(native_atoms):
            a, b = edge
            l, r = left.get((a, b), -1), left.get((b, a), -1)
            change = tuple(
                (owner, amount)
                for owner, amount in sorted(transitions[edge].items())
                if amount
            )
            if l == r:
                if change:
                    _refuse(
                        "zero-area directed winding conservation",
                        origin_count=len(origins[edge]),
                    )
                continue
            links[r].append((l, change))
            links[l].append((r, tuple((owner, -amount) for owner, amount in change)))
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
                    _refuse("binary source polygon winding", face_index=target)
                if target in winding:
                    if winding[target] != state:
                        _refuse("consistent operand winding", face_index=target)
                else:
                    winding[target] = state
                    queue.append(target)
        if set(winding) != set(range(len(integer_faces))) | {-1}:
            _refuse("unbounded-face reachability", face_count=len(integer_faces))
    step = Decimal(str(grid))

    def world(point):
        return tuple(float(value * step) for value in point)

    cells = tuple(
        (
            Polygon(
                tuple(map(world, loops[0])),
                tuple(tuple(map(world, r)) for r in loops[1:]),
            ),
            tuple(values[owner].operand_id for owner in sorted(winding[index])),
        )
        for index, loops in enumerate(face_rings)
        if winding[index]
    )
    boundary, auxiliary = [], []
    active_edges = {edge for edge, face_index in left.items() if winding[face_index]}
    boundary_nodes = {point for edge in active_edges for point in edge}
    for edge in sorted(native_atoms):
        owners = tuple(
            values[owner].operand_id
            for owner in sorted(
                {paths[index].operand_index for index, _ in origins[edge]}
            )
        )
        row = (LineString(tuple(map(world, edge))), owners)
        (
            boundary
            if edge in active_edges or tuple(reversed(edge)) in active_edges
            else auxiliary
        ).append(row)
    collapsed = defaultdict(set)
    for path in paths:
        if len(path.nodes) == 1:
            collapsed[path.nodes[0]].add(path.operand_index)
    for point, owners in sorted(collapsed.items()):
        row = (
            Point(world(point)),
            tuple(values[owner].operand_id for owner in sorted(owners)),
        )
        (boundary if point in boundary_nodes else auxiliary).append(row)
    _progress(len(paths), len(paths))
    return TaggedSubdivision(
        cells,
        tuple(boundary),
        tuple(auxiliary),
        paths,
        tuple((edge, tuple(origins[edge])) for edge in sorted(native_atoms)),
    )
