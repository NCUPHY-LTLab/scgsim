"""Build native grid networks with directed source provenance.
Pixel incidence uses half-open slabs; missing context is refused, never sampled."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import TYPE_CHECKING, NamedTuple, cast

from scgsim.diagnostics import diagnostic_stage, progress
from scgsim.geometry.grid.grid_overlay import LabelledCell, ZeroAreaResidue
from scgsim.geometry.grid.tagged_grid_faces import (
    EdgeProvenance,
    FaceArrangement,
    LatticeEdge,
    LatticePoint,
    PointIndex,
    SourcePath,
    build_edge_provenance,
    build_face_arrangement,
    refuse_graph,
    signed_ring_area,
)
from scgsim.geometry.grid.tagged_grid_operations import (
    ShapelyGridOperations,
    load_shapely_grid_operations,
)
from scgsim.geometry.models.refusals import internal
from scgsim.preparation import current_preparation_workers, ordered_map

if TYPE_CHECKING:
    from shapely.geometry import Point, Polygon
    from shapely.geometry.base import BaseGeometry

    from scgsim.geometry.grid.grid_regions import GridOperand


@dataclass(frozen=True)
class TaggedSubdivision:
    """Bundle subdivision cells, boundaries and their source-segment provenance."""

    cells: tuple
    boundary: tuple
    auxiliary: tuple
    source_paths: tuple[SourcePath, ...]
    edge_origins: tuple[tuple[LatticeEdge, tuple[tuple[int, int], ...]], ...]


@dataclass(frozen=True)
class _NativeGridNetwork:
    """Keep native snapped edges and their endpoint lattice identities together."""

    edges: tuple[LatticeEdge, ...]
    endpoint_keys: tuple[LatticePoint, ...]
    native_grid: float
    scale: float
    pixels: frozenset[LatticePoint]


@dataclass(frozen=True)
class _SourcePathArrangement:
    """Bundle directed paths with the lattice index used to derive them."""

    paths: tuple[SourcePath, ...]
    point_index: PointIndex


@dataclass(frozen=True)
class _PixelPathContext:
    """Carry the index and refusal context used for one directed path."""

    index: PointIndex
    scale: float
    refusal_context: dict[str, int]


def _line_parts(value):
    if value.is_empty:
        return ()
    if value.geom_type in {"LineString", "LinearRing"}:
        return (value,)
    if value.geom_type in {"MultiLineString", "GeometryCollection"}:
        return tuple(p for child in value.geoms for p in _line_parts(child))
    if value.geom_type in {"Point", "MultiPoint"}:
        return ()
    raise ValueError(
        internal(
            f"the tagged grid boundary network holds a {value.geom_type}, not lines"
        )
    )


def pixel_interval(a, b, pixel):
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


def raw_chain(a, b, nodes):
    """Order raw source nodes along their directed segment."""
    # Segment direction defines a geometric lexicographic order, including
    # off-line near vertices with equal dominant coordinate. No identity tie.
    dx, dy = b[0] - a[0], b[1] - a[1]
    axis = 0 if abs(dx) >= abs(dy) else 1
    signs = (1 if dx >= 0 else -1, 1 if dy >= 0 else -1)
    ordered = tuple(
        sorted(
            nodes, key=lambda p: (signs[axis] * p[axis], signs[1 - axis] * p[1 - axis])
        )
    )
    if not ordered or (ordered[0], ordered[-1]) != (a, b):
        refuse_graph("directed raw source node order")
    return ordered


def _intersection_nodes(value):
    if value.is_empty:
        return ()
    if value.geom_type == "Point":
        return (tuple(value.coords[0]),)
    if value.geom_type in {"LineString", "LinearRing"}:
        return (tuple(value.coords[0]), tuple(value.coords[-1]))
    return tuple(point for child in value.geoms for point in _intersection_nodes(child))


def _segment_intersection(lines, i, j, grid, intersection):
    """The nodes where source segments ``i`` and ``j`` meet; a native failure carries their ordinals."""
    try:
        return _intersection_nodes(intersection(lines[i], lines[j], grid_size=0))
    except Exception as error:  # noqa: BLE001 -- annotate and re-raise any native failure unchanged.
        try:
            error.add_note(
                "scgsim.grid.tagged_raw_context: "
                + json.dumps(
                    dict(
                        operation="intersection",
                        input_dimension=1,
                        segment_ordinals=(i, j),
                        grid_size_um=grid,
                    ),
                    sort_keys=True,
                )
            )
        except Exception:  # noqa: BLE001 -- preserve the native error if its note cannot be attached.
            pass
        raise


def _raw_context(segments, grid, operations: ShapelyGridOperations):
    """Recover pair-known pre-snap nodes without native-edge attribution.
    Snap width is context, not a physical tolerance; graph certification binds."""
    from shapely import STRtree, box, intersection  # noqa: PLC0415 -- tree queries and exact intersections are needed only for source context.

    lines = tuple(operations.line_string(edge) for edge in segments)
    index = STRtree(lines)
    width = grid / 100
    nodes = [set(edge) for edge in segments]

    def events(i):
        a, b = segments[i]
        xmin, ymin, xmax, ymax = lines[i].bounds
        candidates = sorted(
            int(j)
            for j in index.query(
                box(xmin - width, ymin - width, xmax + width, ymax + width)
            )
            if int(j) > i
        )
        output = []
        for j in candidates:
            c, d = segments[j]
            common = _segment_intersection(lines, i, j, grid, intersection)
            if any(p not in (a, b) or p not in (c, d) for p in common):
                output.extend((owner, p) for owner in (i, j) for p in common)
                continue
            for p, target, ends in (
                (a, j, (c, d)),
                (b, j, (c, d)),
                (c, i, (a, b)),
                (d, i, (a, b)),
            ):
                point = operations.point(p)
                if (
                    all(operations.point(end).distance(point) >= width for end in ends)
                    and point.distance(lines[target]) < width
                ):
                    output.append((target, p))
        return tuple(output)

    workers = current_preparation_workers() or 1
    size = max(1, (len(segments) + workers - 1) // workers)
    chunks = tuple(
        range(start, min(start + size, len(segments)))
        for start in range(0, len(segments), size)
    )

    def chunk_events(chunk):
        return tuple(row for i in chunk for row in events(i))

    with diagnostic_stage(
        "geometry.grid.tagged_raw_context", counts={"segments": len(segments)}
    ):
        for rows in ordered_map(chunk_events, chunks):
            for owner, point in rows:
                nodes[owner].add(point)
    return tuple(
        raw_chain(a, b, points) for (a, b), points in zip(segments, nodes, strict=True)
    )


def _pixel_path(a, b, endpoints, context):
    if endpoints[0] == endpoints[1]:
        return (endpoints[0],)
    half = Fraction(1, 2)
    # Bind native scale-first binary64 arithmetic before exact predicates.
    start, end = (
        tuple(Fraction(value * context.scale) for value in point) for point in (a, b)
    )
    candidates = context.index.candidates(
        min(start[0], end[0]) - half,
        min(start[1], end[1]) - half,
        max(start[0], end[0]) + half,
        max(start[1], end[1]) + half,
    )
    hits = []
    for pixel in candidates:
        interval = pixel_interval(start, end, pixel)
        if interval is not None:
            hits.append((*interval, pixel))
    hits.sort()
    path = tuple(pixel for _, _, pixel in hits)
    if not path or (path[0], path[-1]) != endpoints:
        refuse_graph("source endpoint pixel incidence", **context.refusal_context)
    if any(second[0] < first[1] for first, second in zip(hits, hits[1:], strict=False)):
        refuse_graph("unique source pixel order", **context.refusal_context)
    return path


class SourceSegment(NamedTuple):
    """One directed exterior or hole edge of an operand, in operand order, with its ring's orientation."""

    operand_id: str
    operand_index: int
    ring_index: int
    segment_index: int
    start: tuple
    end: tuple
    orientation: int


class LocatedSegment(NamedTuple):
    """A source segment with its ends' arrangement keys and the raw chain it splits into.

    Without a ``chain`` the segment runs straight from end to end.
    """

    segment: SourceSegment
    endpoint_keys: tuple
    chain: tuple | None = None


def source_paths(rows: Sequence[LocatedSegment], index, scale):
    """Build source-segment paths through indexed arrangement nodes."""
    output = []
    for row in rows:
        segment, endpoints = row.segment, row.endpoint_keys
        value, operand_index = segment.operand_id, segment.operand_index
        ring_index, segment_index = segment.ring_index, segment.segment_index
        a, b, orientation = segment.start, segment.end, segment.orientation
        chain = (
            row.chain
            if row.chain is not None
            else ((a, endpoints[0]), (b, endpoints[1]))
        )
        context = dict(
            operand_index=operand_index,
            ring_index=ring_index,
            segment_index=segment_index,
        )
        pixel_context = _PixelPathContext(index, scale, context)
        path = [chain[0][1]] if len(chain) == 1 else []
        for (start, first), (end, last) in zip(chain, chain[1:], strict=False):
            piece = _pixel_path(start, end, (first, last), pixel_context)
            if path and path[-1] != piece[0]:
                refuse_graph(
                    "continuous raw split source path", **pixel_context.refusal_context
                )
            path.extend(piece[1:] if path else piece)
        if not path:
            refuse_graph(
                "source endpoint pixel incidence", **pixel_context.refusal_context
            )
        output.append(
            SourcePath(
                value,
                operand_index,
                ring_index,
                segment_index,
                tuple(path),
                orientation,
            )
        )
    return tuple(output)


def _source_segments(
    values: Sequence[GridOperand], geometries: Mapping[str, BaseGeometry]
):
    """Record each directed exterior and hole edge in operand order."""
    raw_rows, vertices = [], []
    for operand_index, value in enumerate(values):
        shape = cast("Polygon", geometries[value.operand_id])
        for ring_index, ring in enumerate((shape.exterior, *shape.interiors)):
            ring_points = tuple(tuple(point) for point in ring.coords)[:-1]
            if not ring_points:
                continue
            area = signed_ring_area(
                tuple(
                    tuple(Fraction(value) for value in point) for point in ring_points
                )
            )
            if not area:
                refuse_graph(
                    "nondegenerate source ring direction",
                    operand_index=operand_index,
                    ring_index=ring_index,
                )
            orientation = (1 if ring_index == 0 else -1) * (1 if area > 0 else -1)
            for segment_index, (a, b) in enumerate(
                zip(ring_points, (*ring_points[1:], ring_points[0]), strict=True)
            ):
                raw_rows.append(
                    SourceSegment(
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
    return raw_rows, vertices


class _NativeNetwork(NamedTuple):
    """The unique source edges, and the snapped line network built from them."""

    unique_edges: tuple
    network: _NativeGridNetwork


def _native_grid_network(
    raw_rows, vertices, grid, operations: ShapelyGridOperations
) -> _NativeNetwork:
    """Build one global snapped line network before observing rounded points."""
    unique_edges = tuple(
        dict.fromkeys(tuple(sorted((row.start, row.end))) for row in raw_rows)
    )
    with diagnostic_stage(
        "geometry.grid.tagged_network", counts={"segments": len(raw_rows)}
    ):
        try:
            native = operations.union_all(
                [operations.line_string(edge) for edge in unique_edges],
                grid_size=grid,
            )
        except Exception as error:  # noqa: BLE001 -- annotate and re-raise any native failure unchanged.
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
            except Exception:  # noqa: BLE001 -- preserve the native error if its note cannot be attached.
                pass
            raise
        # Observe native point rounding after global noding; these points only
        # provide endpoint metadata and ties, never independently rounded inputs.
        rounded = operations.set_precision(
            operations.points(vertices), grid, mode="pointwise"
        )
        native_grid = operations.get_precision(cast("Sequence[Point]", rounded)[0])
        scale = 1 / native_grid

        def key(point):
            return tuple(round(value * scale) for value in point)

        endpoint_keys = tuple(
            key(cast("tuple[float, ...]", point.coords[0])) for point in rounded
        )
        edges = tuple(
            (key(a), key(b))
            for line in _line_parts(native)
            for a, b in zip(line.coords, tuple(line.coords)[1:], strict=False)
            if key(a) != key(b)
        )
        pixels = frozenset(
            {key(point) for point in operations.get_coordinates(native)}
            | set(endpoint_keys)
        )
    return _NativeNetwork(
        unique_edges,
        _NativeGridNetwork(edges, endpoint_keys, native_grid, scale, pixels),
    )


def _context_keys(chains, network, operations: ShapelyGridOperations) -> dict:
    """Each raw context point of the chains, keyed by where the native grid rounds it."""
    context_points = tuple(
        dict.fromkeys(point for chain in chains.values() for point in chain)
    )
    context_rounded = operations.set_precision(
        operations.points(context_points), network.native_grid, mode="pointwise"
    )

    def key(point):
        return tuple(round(value * network.scale) for value in point)

    context_keys = dict(
        zip(
            context_points,
            (
                key(cast("tuple[float, ...]", point.coords[0]))
                for point in cast("Sequence[Point]", context_rounded)
            ),
            strict=True,
        )
    )
    return context_keys


def _source_path_arrangement(
    raw_rows,
    unique_edges,
    network,
    operations: ShapelyGridOperations,
) -> _SourcePathArrangement:
    """Recover source context and rasterize each directed source segment."""
    chains = dict(
        zip(
            unique_edges,
            _raw_context(unique_edges, network.native_grid, operations),
            strict=True,
        )
    )
    context_keys = _context_keys(chains, network, operations)
    point_index = PointIndex.build((*network.pixels, *context_keys.values()))

    def source_chain(row):
        edge = tuple(sorted((row.start, row.end)))
        chain = raw_chain(row.start, row.end, chains[edge])
        return tuple((point, context_keys[point]) for point in chain)

    rows = tuple(
        LocatedSegment(
            row,
            (network.endpoint_keys[2 * index], network.endpoint_keys[2 * index + 1]),
            source_chain(row),
        )
        for index, row in enumerate(raw_rows)
    )
    workers = current_preparation_workers() or 1
    size = (len(rows) + workers - 1) // workers
    chunks = tuple(rows[start : start + size] for start in range(0, len(rows), size))
    with diagnostic_stage(
        "geometry.grid.tagged_paths",
        counts={"segments": len(rows), "pixels": len(point_index.by_x)},
    ):
        paths = tuple(
            path
            for chunk in ordered_map(
                lambda chunk: source_paths(chunk, point_index, network.scale), chunks
            )
            for path in chunk
        )
    for (operand_index, ring_index), ring in _source_rings(paths).items():
        for first, second in zip(ring, (*ring[1:], ring[0]), strict=True):
            if first.nodes[-1] != second.nodes[0]:
                refuse_graph(
                    "closed source ring order",
                    operand_index=operand_index,
                    ring_index=ring_index,
                )
    return _SourcePathArrangement(paths, point_index)


def _source_rings(paths):
    """Group directed segments by their originating polygon ring."""
    rings = defaultdict(list)
    for path in paths:
        rings[(path.operand_index, path.ring_index)].append(path)
    return rings


def _world_point(point: LatticePoint, grid_step: Decimal) -> tuple[float, ...]:
    """Convert integer lattice coordinates back to physical coordinates."""
    return tuple(float(value * grid_step) for value in point)


def _tagged_cells(
    values: Sequence[GridOperand],
    arrangement: FaceArrangement,
    step: Decimal,
    operations: ShapelyGridOperations,
) -> tuple[LabelledCell, ...]:
    """Each face some operand covers, as a polygon in the layout with the ids of the operands that cover it."""

    def world(point):
        return _world_point(point, step)

    return tuple(
        (
            operations.polygon(
                tuple(map(world, loops[0])),
                tuple(tuple(map(world, ring)) for ring in loops[1:]),
            ),
            tuple(
                values[owner].operand_id for owner in sorted(arrangement.winding[index])
            ),
        )
        for index, loops in enumerate(arrangement.loops)
        if arrangement.winding[index]
    )


def _subdivision_output(
    values: Sequence[GridOperand],
    paths: tuple[SourcePath, ...],
    provenance: EdgeProvenance,
    arrangement: FaceArrangement,
    operations: ShapelyGridOperations,
) -> TaggedSubdivision:
    """Assemble cells and boundary/auxiliary provenance from certified faces."""
    step = Decimal(str(provenance.grid_um))

    def world(point):
        return _world_point(point, step)

    cells = _tagged_cells(values, arrangement, step, operations)
    boundary: list[ZeroAreaResidue] = []
    auxiliary: list[ZeroAreaResidue] = []
    active_edges = {
        edge
        for edge, face_index in arrangement.left.items()
        if arrangement.winding[face_index]
    }
    boundary_nodes = {point for edge in active_edges for point in edge}
    for edge in sorted(provenance.native_atoms):
        owners = tuple(
            values[owner].operand_id
            for owner in sorted(
                {paths[index].operand_index for index, _ in provenance.origins[edge]}
            )
        )
        row = ZeroAreaResidue(operations.line_string(tuple(map(world, edge))), owners)
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
        row = ZeroAreaResidue(
            operations.point(_world_point(point, step)),
            tuple(values[owner].operand_id for owner in sorted(owners)),
        )
        (boundary if point in boundary_nodes else auxiliary).append(row)
    return TaggedSubdivision(
        cells,
        tuple(boundary),
        tuple(auxiliary),
        paths,
        tuple(
            (edge, tuple(provenance.origins[edge]))
            for edge in sorted(provenance.native_atoms)
        ),
    )


def build_tagged_subdivision(values, geometries, grid):
    """Build a grid-aligned subdivision while retaining source edge provenance."""
    operations = load_shapely_grid_operations()
    raw_rows, vertices = _source_segments(values, geometries)
    if not raw_rows:
        return TaggedSubdivision((), (), (), (), ())
    unique_edges, network = _native_grid_network(raw_rows, vertices, grid, operations)
    arrangement = _source_path_arrangement(raw_rows, unique_edges, network, operations)
    paths = arrangement.paths
    point_index = arrangement.point_index
    provenance = build_edge_provenance(paths, network.edges, point_index, grid)
    arrangement = build_face_arrangement(
        provenance.native_atoms,
        point_index,
        provenance.origins,
        provenance.transitions,
        operations,
    )
    result = _subdivision_output(values, paths, provenance, arrangement, operations)
    progress(len(paths), len(paths))
    return result
