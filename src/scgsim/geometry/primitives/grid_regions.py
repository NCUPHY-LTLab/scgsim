"""Represent grid operands/results, reconstruct atomic polygon boundaries, and locate refused changes."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from functools import cached_property
from types import MappingProxyType
from typing import TYPE_CHECKING, cast, overload

from scgsim.diagnostics import (
    diagnostic_stage,
)
from scgsim.geometry.models.grid import GeometryGridContext, format_box, format_length

if TYPE_CHECKING:
    from collections.abc import Collection

    from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
    from shapely.geometry.base import BaseGeometry

# Native grid coordinates retain their supplied width; no coordinate cleaning
# or validation is introduced at this representation boundary.
GridPoint = tuple[float, ...]
GridRing = tuple[GridPoint, ...]
GridEdge = tuple[GridPoint, GridPoint]
GridRegion = tuple[GridRing, tuple[GridRing, ...]]
RegionKey = tuple[int, ...] | tuple[Fraction, ...]


@dataclass(frozen=True)
class RegionVertex:
    """A realized coordinate and exact key in original vertex order."""

    index: int
    point: GridPoint
    key: RegionKey


@dataclass
class _RegionCache:
    polygons: dict[int, Polygon] = field(default_factory=dict)
    regions: dict[tuple[int, ...], tuple[GridRegion, ...]] = field(default_factory=dict)


def polygon_components(geometry: BaseGeometry) -> tuple[Polygon, ...]:
    """Return only polygon components from a GEOS polygon or collection."""
    if geometry.is_empty:
        return ()
    if geometry.geom_type == "Polygon":
        # The native geom_type discriminator selects Polygon without repeating a
        # runtime check or changing the native attribute/error path.
        return (cast("Polygon", geometry),)
    if geometry.geom_type not in {"MultiPolygon", "GeometryCollection"}:
        return ()
    # Only these native collection types expose geoms on this branch.
    collection = cast("GeometryCollection | MultiPolygon", geometry)
    return tuple(p for g in collection.geoms for p in polygon_components(g))


def _ring(coords: Iterable[Sequence[float]]) -> GridRing:
    points = tuple(tuple(p) for p in coords)[:-1]
    start = min(range(len(points)), key=points.__getitem__)
    return points[start:] + points[:start]


@dataclass(frozen=True)
class RegionVertexIndex:
    """Coordinate indices and exact keys shared during atomic ring reconstruction."""

    grid_size: float | None
    by_x: tuple[RegionVertex, ...]
    by_y: tuple[RegionVertex, ...]
    xs: tuple[float, ...]
    ys: tuple[float, ...]
    coordinates: Mapping[RegionKey, GridPoint]
    point_keys: Mapping[GridPoint, RegionKey]


@overload
def region_key(point: Sequence[float], grid_size: float) -> tuple[int, ...]: ...


@overload
def region_key(point: Sequence[float], grid_size: None) -> tuple[Fraction, ...]: ...


def region_key(point: Sequence[float], grid_size: float | None) -> RegionKey:
    """Map a point to its exact lattice key or ungridded rational coordinates."""

    return (
        tuple(round(value / grid_size) for value in point)
        if grid_size is not None
        else tuple(Fraction(value) for value in point)
    )


def canonical_grid_point(point: Sequence[float], grid_size: float | None) -> GridPoint:
    """Return the lattice node nearest a point, spelled as the grid spells it."""
    if grid_size is None:
        return tuple(point)

    # Decimal spelling is the declared layout step; avoid introducing a second
    # binary multiplication spelling for an otherwise identical lattice node.
    step = Decimal(str(grid_size))
    return tuple(float(value * step) for value in region_key(point, grid_size))


def region_vertex_index(
    vertices: Collection[GridPoint], grid_size: float | None
) -> RegionVertexIndex:
    """Index realized vertices once for exact segment-incidence queries."""
    # Only realized lattice nodes share one key-to-coordinate identity. Parent
    # planes without a layout grid retain their original coordinates/occurrences.

    coordinates: dict[RegionKey, GridPoint] = {}
    point_keys: dict[GridPoint, RegionKey] = {}
    if grid_size is not None:
        step = Decimal(str(grid_size))
        for point in vertices:
            key = region_key(point, grid_size)
            if key not in coordinates:
                canonical = tuple(float(value * step) for value in key)
                coordinates[key] = canonical
                point_keys[canonical] = key
        vertices = tuple(coordinates.values())
    else:
        point_keys = {tuple(point): region_key(point, None) for point in vertices}
    rows = tuple(
        RegionVertex(index, point, point_keys[tuple(point)])
        for index, point in enumerate(vertices)
    )
    by_x = tuple(sorted(rows, key=lambda row: row.point[0]))
    by_y = tuple(sorted(rows, key=lambda row: row.point[1]))
    return RegionVertexIndex(
        grid_size,
        by_x,
        by_y,
        tuple(row.point[0] for row in by_x),
        tuple(row.point[1] for row in by_y),
        MappingProxyType(coordinates),
        MappingProxyType(point_keys),
    )


def regions(
    geometry: BaseGeometry,
    vertices: Collection[GridPoint] = (),
    grid_size: float | None = None,
) -> tuple[GridRegion, ...]:
    """Reconstruct atomic polygon rings using a shared realized-vertex index."""
    with diagnostic_stage("geometry.grid.regions"):
        return regions_indexed(geometry, region_vertex_index(vertices, grid_size))


def regions_indexed(
    geometry: BaseGeometry, vertex_index: RegionVertexIndex
) -> tuple[GridRegion, ...]:
    """Insert realized collinear vertices into each polygon boundary ring."""
    return _RingReconstructor(vertex_index).regions(geometry)


@dataclass
class _RingReconstructor:
    vertex_index: RegionVertexIndex
    extra_keys: dict[GridPoint, RegionKey] = field(default_factory=dict)

    def point_key(self, point: Sequence[float]) -> RegionKey:
        point = tuple(point)
        key = self.vertex_index.point_keys.get(point)
        if key is None:
            key = self.extra_keys.get(point)
            if key is None:
                key = region_key(point, self.vertex_index.grid_size)
                self.extra_keys[point] = key
        return key

    def atomic(self, coords: Iterable[Sequence[float]]) -> GridRing:
        by_x, by_y = self.vertex_index.by_x, self.vertex_index.by_y
        xs, ys = self.vertex_index.xs, self.vertex_index.ys
        if self.vertex_index.grid_size is not None:
            points = []
            for point in coords:
                key = self.point_key(point)
                canonical = self.vertex_index.coordinates.get(key)
                points.append(
                    canonical_grid_point(point, self.vertex_index.grid_size)
                    if canonical is None
                    else canonical
                )
            points = tuple(points)
            coords = tuple(
                point
                for index, point in enumerate(points)
                if index == 0 or point != points[index - 1]
            )
        ring = _ring(coords)
        if not by_x:
            return ring
        ring_keys = tuple(self.point_key(point) for point in ring)
        output = []
        for index, (a, b) in enumerate(zip(ring, (*ring[1:], ring[0]), strict=False)):
            output.append(a)
            ak, bk = ring_keys[index], ring_keys[(index + 1) % len(ring)]
            xmin, xmax = min(a[0], b[0]), max(a[0], b[0])
            ymin, ymax = min(a[1], b[1]), max(a[1], b[1])
            xlo, xhi = bisect_left(xs, xmin), bisect_right(xs, xmax)
            ylo, yhi = bisect_left(ys, ymin), bisect_right(ys, ymax)
            candidates = by_x[xlo:xhi] if xhi - xlo <= yhi - ylo else by_y[ylo:yhi]
            along = [
                point
                for vertex in sorted(candidates, key=lambda row: row.index)
                for point, candidate_key in ((vertex.point, vertex.key),)
                if point != a
                and point != b
                and (candidate_key[0] - ak[0]) * (bk[1] - ak[1])
                == (candidate_key[1] - ak[1]) * (bk[0] - ak[0])
                and xmin <= point[0] <= xmax
                and ymin <= point[1] <= ymax
            ]
            axis = 0 if a[0] != b[0] else 1
            output.extend(
                sorted(along, key=lambda p: p[axis], reverse=b[axis] < a[axis])
            )
        return tuple(output)

    def regions(self, geometry: BaseGeometry) -> tuple[GridRegion, ...]:
        return tuple(
            sorted(
                (
                    self.atomic(p.exterior.coords),
                    tuple(sorted(self.atomic(h.coords) for h in p.interiors)),
                )
                for p in polygon_components(geometry)
            )
        )


@dataclass(frozen=True)
class GridOperand:
    """One declared polygon operand and its physical/contact preservation policy."""

    operand_id: str
    exterior: GridRing
    holes: tuple[GridRing, ...] = ()
    physical: bool = True
    contact: bool = True
    # The source layer, named by a refusal only; it never changes the grid work.
    layer: str | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "exterior", tuple(map(tuple, self.exterior)))
        object.__setattr__(
            self, "holes", tuple(tuple(map(tuple, r)) for r in self.holes)
        )


@dataclass(frozen=True)
class GridFace:
    """One atomic polygon face and its complete operand membership."""

    face_id: str
    exterior: GridRing
    holes: tuple[GridRing, ...]
    operand_ids: tuple[str, ...]


@dataclass(frozen=True)
class GridOperandAudit:
    """Requested and realized area and source-vertex preservation observations."""

    operand_id: str
    source_area_um2: float
    effective_area_um2: float
    geometry_changed: bool
    duplicate_vertex_count: int


@dataclass(frozen=True)
class GridArrangement:
    """Immutable labelled faces with per-arrangement pure region-result caches."""

    context: GeometryGridContext
    faces: tuple[GridFace, ...]
    vertices: tuple[GridPoint, ...]
    edges: tuple[GridEdge, ...]
    audits: tuple[GridOperandAudit, ...]
    grid_applied: bool = True

    @cached_property
    def _vertex_index(self) -> RegionVertexIndex:
        grid = self.context.grid_size_um if self.grid_applied else None
        return region_vertex_index(self.vertices, grid)

    @cached_property
    def _operand_faces(self) -> Mapping[str, tuple[int, ...]]:

        selected: dict[str, list[int]] = {}
        for index, face in enumerate(self.faces):
            for operand_id in face.operand_ids:
                selected.setdefault(operand_id, []).append(index)
        return MappingProxyType({key: tuple(value) for key, value in selected.items()})

    @cached_property
    def _region_work(self) -> _RegionCache:
        # Per-arrangement successful pure results only. Concurrent readers may
        # compute the same miss; immutable geometry/results are never mutated.
        return _RegionCache()

    def _regions_for_faces(self, indices: tuple[int, ...]) -> tuple[GridRegion, ...]:
        from shapely import (  # noqa: PLC0415 - defer native geometry loading until this operation requires it
            Polygon,
            coverage_union_all,
            union_all,
        )  # optional geometry dependency  # noqa: PLC0415 - optional geometry dependency

        polygons, regions = self._region_work.polygons, self._region_work.regions
        if indices in regions:
            return regions[indices]
        selected = []
        for index in indices:
            polygon = polygons.get(index)
            if polygon is None:
                face = self.faces[index]
                polygon = Polygon(face.exterior, face.holes)
                polygons[index] = polygon
            selected.append(polygon)
        with diagnostic_stage("geometry.grid.regions"):
            merged = (
                coverage_union_all(selected)
                if self.grid_applied
                else union_all(selected)
            )
            result = regions_indexed(merged, self._vertex_index)
        regions[indices] = result
        return result

    def face_regions(self, face_ids: Collection[str]) -> tuple[GridRegion, ...]:
        """Return regions for selected faces, reusing exact prior results."""
        return self._regions_for_faces(
            tuple(
                index
                for index, face in enumerate(self.faces)
                if face.face_id in face_ids
            )
        )

    def operand_regions(self, operand_id: str) -> tuple[GridRegion, ...]:
        """Return regions for one operand under the arrangement grid policy."""
        return self._regions_for_faces(self._operand_faces.get(operand_id, ()))

    @property
    def metadata(self) -> dict[str, object]:
        """Describe the realized grid, memberships and operand preservation audits."""
        return {
            "geometry_grid": self.context.to_payload(),
            "method": "geos_labelled_polygon_overlay",
            "grid_scope": "assembly_xy"
            if self.grid_applied
            else "parent_plane_unquantized",
            "face_memberships": [
                {"face_id": f.face_id, "operand_ids": list(f.operand_ids)}
                for f in self.faces
            ],
            "vertex_count": len(self.vertices),
            "edge_count": len(self.edges),
            "operands": [
                {
                    "operand_id": a.operand_id,
                    "source_area_um2": a.source_area_um2,
                    "effective_area_um2": a.effective_area_um2,
                    "geometry_changed": a.geometry_changed,
                    "duplicate_vertex_count": a.duplicate_vertex_count,
                }
                for a in self.audits
            ],
        }


@dataclass(frozen=True)
class PolygonRegion:
    """Simple exterior/holes; cutline walk is mechanical legacy serialization."""

    exterior: GridRing
    holes: tuple[GridRing, ...] = ()

    @property
    def points(self) -> GridRing:
        """Serialize paired cutline connectors without altering physical boundaries."""

        def area(ring):
            return sum(
                a[0] * b[1] - a[1] * b[0]
                for a, b in zip(ring, (*ring[1:], ring[0]), strict=False)
            )

        outer = (
            self.exterior if area(self.exterior) > 0 else tuple(reversed(self.exterior))
        )
        walk = [outer[0]]
        for hole in self.holes:
            ring = hole if area(hole) < 0 else tuple(reversed(hole))
            # Both copies of this serialization connector cancel exactly. It
            # has no physical boundary/origin identity and performs no clipping.
            walk.extend((*ring, ring[0], outer[0]))
        return (*walk, *outer[1:])

    def area(self) -> float:
        """Return exterior area minus the sum of hole areas."""

        def area(ring):
            return (
                abs(
                    sum(
                        a[0] * b[1] - a[1] * b[0]
                        for a, b in zip(ring, (*ring[1:], ring[0]), strict=False)
                    )
                )
                / 2
            )

        return area(self.exterior) - sum(area(r) for r in self.holes)


# -- Locating a refused grid change -------------------------------------------------
#
# The grid audits compare a requested (unsnapped) geometry with its realized
# (snapped) one and refuse when a count or a contact relation differs.  These
# helpers find the places behind that difference, so a refusal can name where in
# micrometres the layout must change.  They add no tolerance: a contact change is
# read entry by entry from the point sets behind the DE-9IM relation; components
# and holes are matched by shared area, and a merge or a split is located on the
# drawn part snapped to the same grid.  Where a place cannot be isolated, the
# finding says so and names only what is certainly true, never a nearby box.
# Coordinates and lengths are printed by the grid's own formatters (``models.grid``).


def topology_changes(
    requested: BaseGeometry, realized: BaseGeometry, grid_um: float | None
) -> tuple[str, ...]:
    """Each place where snapping merged, split, removed or added a component or a hole.

    When the component or hole counts differ, at least one finding is returned:
    a count can only change through one of these four events.  A geometry GEOS
    cannot process gives no finding; the refusal itself still stands.
    """
    from shapely import Polygon  # noqa: PLC0415 - optional geometry dependency, loaded on the refusal path only
    from shapely.errors import GEOSException  # noqa: PLC0415 - optional geometry dependency, loaded on the refusal path only

    before = polygon_components(requested)
    after = polygon_components(realized)

    def holes(components: Sequence[Polygon]) -> list[Polygon]:
        return [Polygon(ring) for p in components for ring in p.interiors]

    try:
        return (
            *_matching_changes("component", before, after, grid_um),
            *_matching_changes("hole", holes(before), holes(after), grid_um),
        )
    except GEOSException:
        return ()


def contact_changes(
    requested: tuple[BaseGeometry, BaseGeometry],
    realized: tuple[BaseGeometry, BaseGeometry],
    grid_um: float | None,
) -> tuple[str, ...]:
    """Where the contact relation of two polygonal geometries differs before and after snapping.

    Each entry of the DE-9IM relation, except exterior with exterior, is the
    dimension of one point set: the interiors' overlap, the boundaries' contact,
    one boundary inside or outside the other geometry, one interior outside the
    other.  The relation differs only if one of these sets changes dimension, so
    the parts of the higher-dimension side are the places, and every one of
    them is reported.  A geometry GEOS cannot process gives no finding; the
    refusal itself still stands.
    """
    from shapely.errors import GEOSException  # noqa: PLC0415 - optional geometry dependency, loaded on the refusal path only

    try:
        return _contact_changes(requested, realized, grid_um)
    except GEOSException:
        return ()


def _contact_changes(
    requested: tuple[BaseGeometry, BaseGeometry],
    realized: tuple[BaseGeometry, BaseGeometry],
    grid_um: float | None,
) -> tuple[str, ...]:
    before, after = _relation_sets(*requested), _relation_sets(*realized)
    findings: list[str] = []
    for (appears, lost), drawn, snapped in zip(
        _RELATION_WORDS, before, after, strict=True
    ):
        dim_before, dim_after = _dimension(drawn), _dimension(snapped)
        if dim_after > dim_before:
            findings.extend(
                f"{appears or _TOUCH[_part_dimension(part)]} after snapping at "
                f"{format_place(part, grid_um)}"
                for part in _parts_above(snapped, dim_before)
            )
        elif dim_before > dim_after:
            findings.extend(
                f"{lost} at {format_place(part, grid_um)} is lost after snapping"
                for part in _parts_above(drawn, dim_after)
            )
    first, second = requested
    if findings and not first.intersects(second):
        gap = format_length(first.distance(second), grid_um)
        findings.append(f"the narrowest drawn gap between them is {gap}")
    return tuple(findings)


# What is said when a set of ``_relation_sets`` appears, and when it is lost, in
# the order of that function; a new boundary contact is told by its dimension.
_TOUCH = {0: "they touch at a point", 1: "they touch along an edge"}
_RELATION_WORDS: tuple[tuple[str | None, str], ...] = (
    ("they overlap", "their drawn overlap"),
    (None, "the drawn contact of their boundaries"),
    (
        "the first one's boundary runs inside the second",
        "the drawn run of the first one's boundary inside the second",
    ),
    (
        "the second one's boundary runs inside the first",
        "the drawn run of the second one's boundary inside the first",
    ),
    (
        "the first reaches outside the second",
        "the drawn part of the first outside the second",
    ),
    (
        "the second reaches outside the first",
        "the drawn part of the second outside the first",
    ),
    (
        "the first one's boundary runs outside the second",
        "the drawn run of the first one's boundary outside the second",
    ),
    (
        "the second one's boundary runs outside the first",
        "the drawn run of the second one's boundary outside the first",
    ),
)


def _relation_sets(first: BaseGeometry, second: BaseGeometry) -> list[BaseGeometry]:
    """The point sets whose dimensions are the DE-9IM entries II, BB, BI, IB, IE, EI, BE, EB.

    An open interior is reached as the closed geometry minus its boundary; for
    polygons an interior set is two-dimensional or empty, so its polygonal
    parts stand for it.
    """
    edge_1, edge_2 = first.boundary, second.boundary
    return [
        _polygonal(first.intersection(second)),
        edge_1.intersection(edge_2),
        edge_1.intersection(second).difference(edge_2),
        edge_2.intersection(first).difference(edge_1),
        _polygonal(first.difference(second)),
        _polygonal(second.difference(first)),
        edge_1.difference(second),
        edge_2.difference(first),
    ]


def _polygonal(geometry: BaseGeometry) -> BaseGeometry:
    from shapely import MultiPolygon  # noqa: PLC0415 - optional geometry dependency, loaded on the refusal path only

    return MultiPolygon(list(polygon_components(geometry)))


FIX_SHAPE = "Fix the layout: draw each gap, neck and hole named here on the grid, widening or closing it."
FIX_CONTACT = "Fix the layout: draw the gap or contact between them on the grid, widening or closing it."


def grid_change_message(
    lead: str,
    layers: Sequence[str],
    grid_um: float | None,
    findings: Sequence[str],
    action: str,
) -> str:
    """One refusal: what changed, on which layers, where (each finding) and who must act."""
    names = sorted(set(layers))
    on = f" on layer{'s' if len(names) > 1 else ''} {', '.join(names)}" if names else ""
    grid = "" if grid_um is None else f" (grid g = {grid_um} um)"
    where = "; ".join(findings) if findings else "the place could not be located"
    return f"{lead}{on}{grid}: {where}. {action}"


def format_place(geometry: BaseGeometry, grid_um: float | None) -> str:
    """Where a geometry is: its point, or its bounding box."""
    return format_box(geometry.bounds, grid_um)


def _matching_changes(
    kind: str,
    before: Sequence[Polygon],
    after: Sequence[Polygon],
    grid_um: float | None,
) -> list[str]:
    """Match parts before and after snapping by shared area and report the ones that do not pair."""
    shared = _shared_areas(before, after)
    image = _largest_partner(shared)
    origin = _largest_partner({(j, i): a for (i, j), a in shared.items()})
    findings: list[str] = []
    for j, group in _groups(image):
        findings.extend(
            _merge_findings(kind, [before[i] for i in group], after[j], grid_um)
        )
    for i, group in _groups(origin):
        findings.extend(
            _split_findings(kind, before[i], [after[j] for j in group], grid_um)
        )
    findings.extend(
        f"a drawn {kind} disappears at {format_place(before[i], grid_um)} "
        f"({_size(before[i], grid_um)})"
        for i in range(len(before))
        if i not in image
    )
    findings.extend(
        f"a {kind} that is not drawn appears at {format_place(after[j], grid_um)} "
        f"({_size(after[j], grid_um)})"
        for j in range(len(after))
        if j not in origin
    )
    return findings


def _merge_findings(
    kind: str, drawn: Sequence[Polygon], merged: Polygon, grid_um: float | None
) -> list[str]:
    """Where drawn parts meet once each is snapped to the grid on its own.

    Where two snapped drawn parts intersect is where they merged.  The
    narrowest drawn gap between them is a true measure, though not necessarily
    of that place.
    """
    from shapely import union_all  # noqa: PLC0415 - optional geometry dependency, loaded on the refusal path only

    gap = format_length(
        min(a.distance(b) for k, a in enumerate(drawn) for b in drawn[k + 1 :]),
        grid_um,
    )
    snapped = [_snapped(part, grid_um) for part in drawn]
    meetings = _parts_above(
        union_all(
            [
                a.intersection(b)
                for k, a in enumerate(snapped)
                for b in snapped[k + 1 :]
                if a is not None and b is not None
            ]
        ),
        -1,
    )
    if not meetings:
        return [
            f"{len(drawn)} drawn {kind}s merge into the one at "
            f"{format_place(merged, grid_um)}, at a place that could not be isolated; "
            f"the narrowest drawn gap between them is {gap}"
        ]
    return [
        *(f"two drawn {kind}s merge at {format_place(m, grid_um)}" for m in meetings),
        f"the narrowest drawn gap between the merged {kind}s is {gap}",
    ]


def _split_findings(
    kind: str, drawn: Polygon, parts: Sequence[Polygon], grid_um: float | None
) -> list[str]:
    """Where a drawn part collapses between its snapped parts.

    Snapped on its own, the drawn part keeps a neck narrower than the grid as a
    line or a point; the ones that touch two of the snapped parts are where it
    split.  An edge that snapping only moved leaves no such residue, so it is
    never taken for the neck.
    """
    from shapely import union_all  # noqa: PLC0415 - optional geometry dependency, loaded on the refusal path only

    snapped = _snapped(drawn, grid_um)
    residues = [] if snapped is None else _flat_parts(snapped)
    necks = [
        residue
        for residue in residues
        if _part_dimension(residue) < 2
        and sum(residue.intersects(part) for part in parts) > 1
    ]
    if not necks:
        where = ", ".join(format_place(part, grid_um) for part in parts)
        return [
            f"a drawn {kind} splits into the parts at {where}, at a place that could "
            "not be isolated"
        ]
    return [
        f"a drawn {kind} splits in two at {format_place(neck, grid_um)}, where it is "
        f"narrower than the grid along {format_length(neck.length, grid_um)}"
        for neck in _parts_above(union_all(necks), -1)
    ]


def _snapped(part: Polygon, grid_um: float | None) -> BaseGeometry | None:
    """The part snapped to the grid on its own, keeping collapsed pieces as lines and points."""
    from shapely import intersection  # noqa: PLC0415 - optional geometry dependency, loaded on the refusal path only

    return None if grid_um is None else intersection(part, part, grid_size=grid_um)


def _shared_areas(
    before: Sequence[Polygon], after: Sequence[Polygon]
) -> dict[tuple[int, int], float]:
    from shapely import STRtree  # noqa: PLC0415 - optional geometry dependency, loaded on the refusal path only

    if not before or not after:
        return {}
    tree = STRtree(list(after))
    shared: dict[tuple[int, int], float] = {}
    for i, part in enumerate(before):
        for j in tree.query(part):
            area = part.intersection(after[int(j)]).area
            if area > 0:
                shared[(i, int(j))] = area
    return shared


def _largest_partner(shared: dict[tuple[int, int], float]) -> dict[int, int]:
    """For each part, the part on the other side it shares the most area with."""
    best: dict[int, tuple[float, int]] = {}
    for (i, j), area in sorted(shared.items()):
        if i not in best or area > best[i][0]:
            best[i] = (area, j)
    return {i: j for i, (_, j) in best.items()}


def _groups(partner: dict[int, int]) -> list[tuple[int, list[int]]]:
    """Each partner shared by more than one part, with those parts, in partner order."""
    by_partner: dict[int, list[int]] = defaultdict(list)
    for part, other in sorted(partner.items()):
        by_partner[other].append(part)
    return [
        (other, parts) for other, parts in sorted(by_partner.items()) if len(parts) > 1
    ]


def _size(geometry: BaseGeometry, grid_um: float | None) -> str:
    """The width by the height of a geometry's bounding box."""
    x0, y0, x1, y1 = geometry.bounds
    return f"{format_length(x1 - x0, grid_um)} by {format_length(y1 - y0, grid_um)}"


def _dimension(geometry: BaseGeometry) -> int:
    """The largest dimension of a geometry's parts: -1 when it is empty."""
    return max((_part_dimension(p) for p in _flat_parts(geometry)), default=-1)


def _part_dimension(part: BaseGeometry) -> int:
    return {"Point": 0, "LineString": 1, "LinearRing": 1, "Polygon": 2}[part.geom_type]


def _flat_parts(geometry: BaseGeometry) -> list[BaseGeometry]:
    from shapely import get_parts  # noqa: PLC0415 - optional geometry dependency, loaded on the refusal path only

    if geometry.is_empty:
        return []
    if geometry.geom_type in {"Point", "LineString", "LinearRing", "Polygon"}:
        return [geometry]
    return [q for p in get_parts(geometry) for q in _flat_parts(p)]


def _parts_above(geometry: BaseGeometry, dimension: int) -> list[BaseGeometry]:
    """The connected parts of a geometry whose dimension exceeds ``dimension``, in coordinate order.

    The union drops a lower-dimension part that lies on a higher one (the edge
    of an overlap), and touching edge pieces are merged into one line.
    """
    from shapely import line_merge, union_all  # noqa: PLC0415 - optional geometry dependency, loaded on the refusal path only

    kept = [p for p in _flat_parts(geometry) if _part_dimension(p) > dimension]
    if not kept:
        return []
    parts = _flat_parts(union_all(kept))
    lines = [p for p in parts if _part_dimension(p) == 1]
    joined = [p for p in parts if _part_dimension(p) != 1]
    if lines:
        joined.extend(_flat_parts(line_merge(union_all(lines))))
    return sorted(joined, key=lambda part: part.bounds)
