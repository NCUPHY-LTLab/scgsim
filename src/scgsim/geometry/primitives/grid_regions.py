"""Represent grid operands/results and reconstruct atomic polygon boundaries."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
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
from scgsim.geometry.models.grid import GeometryGridContext

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
