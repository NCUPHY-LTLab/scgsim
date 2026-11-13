"""Polygon conversion and selection that the source stage's Ground recipe and occurrence selectors share.

This leaf has no adapter, occurrence, planning, or backend dependency. Keep one
implementation of source-ring and connected-polygon mechanics for the Ground
recipe and occurrence selectors.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import degrees, isfinite
from typing import Any, cast

from scgsim.geometry.grid.grid import (
    FIX_EMPTY,
    SOURCE_INTENT,
    bound_grid_context,
    frame_point,
    polygon_boolean,
    source_place,
    source_refusal,
    split_gdstk_cutline_loop,
)
from scgsim.geometry.grid.grid_regions import GridPoint, canonical_grid_point
from scgsim.geometry.models.input import LayoutPolygonSpec


def _fused_boolean_candidates(candidates):
    """Conservative world-frame broad phase for the expensive overlap test.

    Raw shared-edge recognition still runs in original local coordinates for
    every pair.
    """
    from shapely import STRtree, box  # noqa: PLC0415 -- only the bound-grid broad phase needs this optional native dependency

    # Native near-vertex insertion is within grid/100 and node rounding within
    # half a grid step. A full step on EACH bound is a conservative candidate
    # envelope, never an overlap/contact acceptance rule.
    padding = 0.0 if SOURCE_INTENT.get() else bound_grid_context().grid_size_um
    envelopes = []
    for polygon in candidates:
        points = tuple(frame_point(tuple(map(float, p))) for p in polygon.points)
        if not points:
            return None
        # Each comprehension enumerates exactly the X and Y axes.
        low = cast(
            tuple[float, float],
            tuple(min(p[axis] for p in points) - padding for axis in (0, 1)),
        )
        high = cast(
            tuple[float, float],
            tuple(max(p[axis] for p in points) + padding for axis in (0, 1)),
        )
        if not all(isfinite(v) for v in (*low, *high)):
            # An envelope that cannot be bounded takes the unfiltered path; no input is refused here.
            return None
        envelopes.append(box(*low, *high))
    tree = STRtree(envelopes)
    return tuple(
        frozenset(int(other) for other in tree.query(envelope) if int(other) > index)
        for index, envelope in enumerate(envelopes)
    )


def fused_gdstk_components(
    candidates: Sequence[Any],
) -> tuple[tuple[LayoutPolygonSpec, tuple[int, ...], float], ...]:
    """Return stable occupied components for explicit source selectors."""

    overlap_candidates = _fused_boolean_candidates(candidates)
    parents = list(range(len(candidates)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parents[max(first_root, second_root)] = min(first_root, second_root)

    for index, polygon in enumerate(candidates):
        for other_index, other in enumerate(candidates[index + 1 :], index + 1):
            overlap = ()
            if overlap_candidates is None or other_index in overlap_candidates[index]:
                overlap = polygon_boolean(
                    (polygon,),
                    (other,),
                    "and",
                    precision=bound_grid_context().grid_size_um,
                )
            if overlap or _rings_share_edge(
                ring_from_gdstk_polygon(polygon), ring_from_gdstk_polygon(other)
            ):
                union(index, other_index)

    members: dict[int, list[int]] = {}
    for index in range(len(candidates)):
        members.setdefault(find(index), []).append(index)
    result: list[tuple[LayoutPolygonSpec, tuple[int, ...], float]] = []
    for indexes in members.values():
        regions = polygon_boolean(
            tuple(candidates[index] for index in indexes),
            (),
            "or",
            precision=bound_grid_context().grid_size_um,
        )
        if not regions or len(regions) != 1:
            raise ValueError("fused selector component must lower to one polygon")
        outer, holes = split_gdstk_cutline_loop(ring_from_gdstk_polygon(regions[0]))
        source_indexes = tuple(indexes)
        result.append(
            (
                LayoutPolygonSpec(
                    polygon_id="",
                    layer="",
                    exterior=outer,
                    holes=holes,
                ),
                source_indexes,
                sum(float(candidates[index].area()) for index in source_indexes),
            )
        )
    return tuple(
        sorted(
            result,
            key=lambda record: (
                min(point[0] for point in record[0].exterior),
                min(point[1] for point in record[0].exterior),
                record[2],
            ),
        )
    )


def flattened_polygons(
    cell: Any, *, excluded_reference_indexes: Sequence[int] = ()
) -> list[Any]:
    """Flatten a GDS cell's polygons, placing quarter-turn references exactly.

    gdstk places a reference with the cosine and sine of its rotation in
    radians, and cos(pi/2) is 6e-17, not 0. A cell placed at 90, 180 or 270
    degrees therefore lands up to about 1e-14 um off the lattice it is drawn
    on, and two of its shapes that met exactly no longer do (Issue #112). A
    whole number of quarter turns, as GDS stores the angle in degrees, maps
    the lattice onto itself, so such a reference is placed by swapping and
    negating coordinates. Any other rotation, and a reference to an unparsed
    raw cell, keeps gdstk's placement: a rotated point is off the lattice as
    authored. The order is gdstk's: the cell's own shapes, then each
    reference in turn, each shape with its repetitions. References at
    ``excluded_reference_indexes`` of the top cell are left out.
    """
    polygons = list(cell.get_polygons(apply_repetitions=True, depth=0))
    excluded = set(excluded_reference_indexes)
    for index, reference in enumerate(cell.references):
        if index not in excluded:
            polygons.extend(_placed_reference_polygons(reference))
    return polygons


def _placed_reference_polygons(reference: Any) -> list[Any]:
    import gdstk  # noqa: PLC0415 -- load the optional GDS kernel only when source polygons are read

    turns = _quarter_turns(reference.rotation)
    if turns is None or not isinstance(reference.cell, gdstk.Cell):
        return list(reference.get_polygons(apply_repetitions=True, depth=None))
    origin_x, origin_y = (float(value) for value in reference.origin)
    offsets = (
        [tuple(map(float, offset)) for offset in reference.repetition.get_offsets()]
        if reference.repetition.size
        else [(0.0, 0.0)]
    )
    placed = []
    for polygon in flattened_polygons(reference.cell):
        points = [
            _quarter_turned(
                (float(x), float(y)),
                turns,
                reference.magnification,
                reference.x_reflection,
            )
            for x, y in polygon.points
        ]
        for offset_x, offset_y in offsets:
            placed.append(
                gdstk.Polygon(
                    [
                        (x + origin_x + offset_x, y + origin_y + offset_y)
                        for x, y in points
                    ],
                    polygon.layer,
                    polygon.datatype,
                )
            )
    return placed


def _quarter_turns(rotation: float) -> int | None:
    # gdstk reads the GDS angle in degrees and stores radians; converting back
    # recovers 90, 180 or 270 exactly, so no angle tolerance is involved.
    angle = degrees(rotation)
    if angle % 90 != 0:
        return None
    return int(angle // 90) % 4


def _quarter_turned(
    point: tuple[float, float], turns: int, magnification: float, x_reflection: bool
) -> tuple[float, float]:
    # gdstk's order: reflect across X, scale, then rotate about the origin.
    x, y = point[0] * magnification, point[1] * magnification
    if x_reflection:
        y = -y
    return ((x, y), (-y, x), (-x, -y), (y, -x))[turns]


def without_repeated_vertices(polygons: Iterable[Any]) -> list[Any]:
    """Drop exactly repeated consecutive vertices from GDS polygons as they are read.

    gdsfactory also stores a circle's closing point, so a 32-gon arrives as 33
    points with one vertex listed twice in a row; KLayout starts the stored walk
    at its lowest point, so the repeat usually sits mid-walk (Issue #107). A
    repeated vertex is a zero-length edge and carries no geometry, and the
    cutline split cannot order points along it. Only exactly equal neighbours,
    the last and first included, are dropped; near points stay, so no
    tolerance enters here. A polygon without a repeat is returned as the same
    object.
    """
    import gdstk  # noqa: PLC0415 -- load the optional GDS kernel only when source polygons are read

    result = []
    for polygon in polygons:
        points = [(float(x), float(y)) for x, y in polygon.points]
        # Keep the first of each run, so a repeated closing point goes, not the start.
        kept = [
            point
            for index, point in enumerate(points)
            if index == 0 or point != points[index - 1]
        ]
        if len(kept) > 1 and kept[-1] == kept[0]:
            kept.pop()
        if len(kept) == len(points):
            result.append(polygon)
            continue
        result.append(gdstk.Polygon(kept, polygon.layer, polygon.datatype))
    return result


def ring_from_gdstk_polygon(polygon: Any) -> tuple[tuple[float, float], ...]:
    """Read an open coordinate ring from a GDS polygon."""
    points = tuple((float(x), float(y)) for x, y in polygon.points)
    if len(points) > 1 and points[0] == points[-1]:
        points = points[:-1]
    if len(points) < 3:
        raise ValueError(
            source_refusal(
                "GDS polygon requires at least 3 unique points",
                [f"the polygon at {source_place(points)} has only {len(points)} points"]
                if points
                else [],
                FIX_EMPTY,
            )
        )
    return points


def realized_on_lattice(polygons: Sequence[Any]) -> Sequence[Any]:
    """Round a source recipe's flattened polygons to the bound grid when realizing it.

    gdstk applies a reference's rotation in floating point, so a rotated
    instance's points fall up to half a grid step off the lattice the cell is
    stored on. A copy of the same shape drawn directly stays on it, and a union
    of the two then keeps both outlines and their crossings (Issue #104).
    Rounding makes identical copies identical before the recipe's Booleans.

    The preservation reference keeps the points as authored: its result is the
    original geometry the source and contact audits compare against, and
    rounding can make two disjoint polygons touch. A polygon that rounds to no
    area is left out, so the audit reports the lost component or hole.

    The lattice is in assembly XY. Inside an occurrence frame each point is
    rounded where the frame places it and mapped back into the cell, so the
    recipe keeps working in local coordinates; rounding the local point instead
    would land on a different assembly node whenever the frame rotates.
    """
    if SOURCE_INTENT.get():
        return polygons
    context = bound_grid_context()
    import gdstk  # noqa: PLC0415 -- the optional native GDS kernel, needed only when realizing a bound source recipe

    def on_lattice(polygon: Any) -> list[tuple[float, float]]:
        nodes: list[GridPoint] = []
        for point in polygon.points:
            node = canonical_grid_point(frame_point(point), context.grid_size_um)
            # Two points that round to one node become one vertex, not a zero-length edge.
            if not nodes or nodes[-1] != node:
                nodes.append(node)
        if len(nodes) > 1 and nodes[0] == nodes[-1]:
            nodes.pop()
        local = (frame_point(node, inverse=True) for node in nodes)
        return [(x, y) for x, y in local]

    rounded = (
        gdstk.Polygon(on_lattice(polygon), polygon.layer, polygon.datatype)
        for polygon in polygons
    )
    return tuple(polygon for polygon in rounded if polygon.area() > 0)


def layout_polygon_region(gdstk: Any, polygon: LayoutPolygonSpec) -> tuple[Any, ...]:
    """Convert a layout polygon and its holes into GDS polygon pieces."""
    outer = gdstk.Polygon(polygon.exterior)
    holes = tuple(gdstk.Polygon(hole) for hole in polygon.holes)
    if not holes:
        return (outer,)
    return tuple(polygon_boolean((outer,), holes, "not", precision=1e-9) or ())


def _ring_edges(
    loop: Sequence[tuple[float, float]],
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    return tuple(zip(loop, (*loop[1:], loop[0]), strict=True))


def _rings_share_edge(
    left: Sequence[tuple[float, float]], right: Sequence[tuple[float, float]]
) -> bool:
    return any(
        _shared_segment_length(left_start, left_end, right_start, right_end) > 1e-9
        for left_start, left_end in _ring_edges(left)
        for right_start, right_end in _ring_edges(right)
    )


def _shared_segment_length(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
) -> float:
    first_dx, first_dy = (
        first_end[0] - first_start[0],
        first_end[1] - first_start[1],
    )
    second_dx, second_dy = (
        second_end[0] - second_start[0],
        second_end[1] - second_start[1],
    )
    first_length = (first_dx * first_dx + first_dy * first_dy) ** 0.5
    second_length = (second_dx * second_dx + second_dy * second_dy) ** 0.5
    if first_length <= 1e-9 or second_length <= 1e-9:
        return 0.0
    if (
        abs(
            (second_start[0] - first_start[0]) * first_dy
            - (second_start[1] - first_start[1]) * first_dx
        )
        > 1e-9 * first_length
        or abs(first_dx * second_dy - first_dy * second_dx)
        > 1e-9 * first_length * second_length
    ):
        return 0.0
    axis = 0 if abs(first_dx) >= abs(first_dy) else 1
    first_low, first_high = sorted((first_start[axis], first_end[axis]))
    second_low, second_high = sorted((second_start[axis], second_end[axis]))
    overlap = min(first_high, second_high) - max(first_low, second_low)
    if overlap <= 1e-9:
        return 0.0
    return overlap * first_length / abs((first_dx, first_dy)[axis])


def point_in_layout_polygon(
    point: tuple[float, float], polygon: LayoutPolygonSpec
) -> bool:
    """Test whether a point lies in the exterior and outside all holes."""
    return point_in_ring(point, polygon.exterior) and not any(
        point_in_ring(point, hole) for hole in polygon.holes
    )


def point_in_ring(
    point: Sequence[float],
    ring: Sequence[tuple[float, float]],
) -> bool:
    """Test ring membership using horizontal ray crossings."""
    x, y = point
    inside = False
    j = len(ring) - 1
    for i, (xi, yi) in enumerate(ring):
        xj, yj = ring[j]
        crosses = (yi > y) != (yj > y)
        if crosses:
            x_intersect = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


FIX_SELECTOR = (
    "Fix the layout or the declaration: draw one region under the selector point on this "
    "layer, or move the point into the region it names."
)


def selector_place(
    point: Sequence[float],
    matched: Sequence[Sequence[Sequence[float]]],
    drawn: Sequence[Sequence[Sequence[float]]],
    *,
    what: str = "the selector point",
) -> str:
    """Where a selector point lies in the layout and which drawn regions hold it, for its refusal.

    ``matched`` and ``drawn`` are the outer rings of the regions that hold the point and of
    every region on the layer, in the polygon frame (``source_place`` maps them to the layout).
    """
    at = source_place([point])
    if matched:
        boxes = ", ".join(source_place(ring) for ring in matched)
        return f"{what} lies at {at}, inside each of the regions at {boxes}"
    if not drawn:
        return f"{what} lies at {at}, and nothing is drawn on the layer"
    extent = source_place([p for ring in drawn for p in ring])
    return (
        f"{what} lies at {at}, outside every region drawn on the layer, "
        f"which lie within {extent}"
    )
