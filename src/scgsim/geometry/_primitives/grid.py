"""Shared fixed-precision noding, atomic faces and full operand membership."""

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from hashlib import sha256
import json

from scgsim.geometry.models.grid import GeometryGridContext
from scgsim.diagnostics import _progress, _stage
from scgsim.preparation import _ordered_map

_ACTIVE_GRID = ContextVar("scgsim_geometry_grid", default=None)
_SOURCE_INTENT = ContextVar("scgsim_source_intent_overlay", default=False)
_SOURCE_AUDITS = ContextVar("scgsim_source_grid_audits", default=None)


def current_grid_context():
    return _ACTIVE_GRID.get()


def with_input_grid(function):
    """Scope existing leaf operations to the explicit input policy only."""
    from functools import wraps
    from scgsim.geometry.models.grid import geometry_grid_context

    @wraps(function)
    def scoped(build_input, *args, **kwargs):
        with geometry_grid_scope(geometry_grid_context(build_input)):
            return function(build_input, *args, **kwargs)

    return scoped


def current_grid_size_um():
    context = _ACTIVE_GRID.get()
    # Retained only for historical inputs that have no declared grid context.
    return 1e-9 if context is None else context.grid_size_um


@contextmanager
def geometry_grid_scope(context):
    same = context is not None and context == _ACTIVE_GRID.get()
    token = _ACTIVE_GRID.set(context)
    audit_token = (
        None if same else _SOURCE_AUDITS.set([] if context is not None else None)
    )
    try:
        yield
    finally:
        if audit_token is not None:
            _SOURCE_AUDITS.reset(audit_token)
        _ACTIVE_GRID.reset(token)


@dataclass(frozen=True)
class GridOperand:
    operand_id: str
    exterior: tuple
    holes: tuple = ()
    physical: bool = True
    contact: bool = True

    def __post_init__(self):
        object.__setattr__(self, "exterior", tuple(map(tuple, self.exterior)))
        object.__setattr__(
            self, "holes", tuple(tuple(map(tuple, r)) for r in self.holes)
        )


@dataclass(frozen=True)
class GridFace:
    face_id: str
    exterior: tuple
    holes: tuple
    operand_ids: tuple[str, ...]


@dataclass(frozen=True)
class GridOperandAudit:
    operand_id: str
    source_area_um2: float
    effective_area_um2: float
    geometry_changed: bool
    duplicate_vertex_count: int


def _polygons(geometry):
    if geometry.is_empty:
        return ()
    if geometry.geom_type == "Polygon":
        return (geometry,)
    if geometry.geom_type not in {"MultiPolygon", "GeometryCollection"}:
        return ()
    return tuple(p for g in geometry.geoms for p in _polygons(g))


def _ring(coords):
    points = tuple(tuple(p) for p in coords)[:-1]
    start = min(range(len(points)), key=points.__getitem__)
    return points[start:] + points[:start]


@dataclass(frozen=True)
class _RegionVertexIndex:
    grid_size: float | None
    by_x: tuple
    by_y: tuple
    xs: tuple
    ys: tuple


def _region_key(point, grid_size):
    from fractions import Fraction

    return (
        tuple(round(value / grid_size) for value in point)
        if grid_size is not None
        else tuple(Fraction(value) for value in point)
    )


def _region_vertex_index(vertices, grid_size):
    # Original occurrence order and duplicate entries belong to the index.
    rows = tuple(
        (index, point, _region_key(point, grid_size))
        for index, point in enumerate(vertices)
    )
    by_x = tuple(sorted(rows, key=lambda row: row[1][0]))
    by_y = tuple(sorted(rows, key=lambda row: row[1][1]))
    return _RegionVertexIndex(
        grid_size,
        by_x,
        by_y,
        tuple(row[1][0] for row in by_x),
        tuple(row[1][1] for row in by_y),
    )


def _regions(geometry, vertices=(), grid_size=None):
    with _stage("geometry.grid.regions"):
        return _regions_indexed(geometry, _region_vertex_index(vertices, grid_size))


def _regions_indexed(geometry, vertex_index):
    from bisect import bisect_left, bisect_right

    by_x, by_y = vertex_index.by_x, vertex_index.by_y
    xs, ys = vertex_index.xs, vertex_index.ys

    def atomic(coords):
        ring = _ring(coords)
        if not by_x:
            return ring
        ring_keys = tuple(_region_key(point, vertex_index.grid_size) for point in ring)
        output = []
        for index, (a, b) in enumerate(zip(ring, (*ring[1:], ring[0]))):
            output.append(a)
            ak, bk = ring_keys[index], ring_keys[(index + 1) % len(ring)]
            xmin, xmax = min(a[0], b[0]), max(a[0], b[0])
            ymin, ymax = min(a[1], b[1]), max(a[1], b[1])
            xlo, xhi = bisect_left(xs, xmin), bisect_right(xs, xmax)
            ylo, yhi = bisect_left(ys, ymin), bisect_right(ys, ymax)
            candidates = by_x[xlo:xhi] if xhi - xlo <= yhi - ylo else by_y[ylo:yhi]
            along = [
                point
                for _, point, point_key in sorted(candidates, key=lambda row: row[0])
                if point != a
                and point != b
                and (point_key[0] - ak[0]) * (bk[1] - ak[1])
                == (point_key[1] - ak[1]) * (bk[0] - ak[0])
                and xmin <= point[0] <= xmax
                and ymin <= point[1] <= ymax
            ]
            axis = 0 if a[0] != b[0] else 1
            output.extend(
                sorted(along, key=lambda p: p[axis], reverse=b[axis] < a[axis])
            )
        return tuple(output)

    return tuple(
        sorted(
            (
                atomic(p.exterior.coords),
                tuple(sorted(atomic(h.coords) for h in p.interiors)),
            )
            for p in _polygons(geometry)
        )
    )


@dataclass(frozen=True)
class GridArrangement:
    context: GeometryGridContext
    faces: tuple[GridFace, ...]
    vertices: tuple
    edges: tuple
    audits: tuple[GridOperandAudit, ...]
    grid_applied: bool = True

    def face_regions(self, face_ids):
        from shapely import Polygon, union_all

        selected = [
            Polygon(f.exterior, f.holes) for f in self.faces if f.face_id in face_ids
        ]
        grid = self.context.grid_size_um if self.grid_applied else None
        return _regions(union_all(selected, grid_size=grid), self.vertices, grid)

    def operand_regions(self, operand_id):
        from shapely import Polygon, union_all

        selected = [
            Polygon(f.exterior, f.holes)
            for f in self.faces
            if operand_id in f.operand_ids
        ]
        grid = self.context.grid_size_um if self.grid_applied else None
        return _regions(union_all(selected, grid_size=grid), self.vertices, grid)

    @property
    def metadata(self):
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
                dict(
                    operand_id=a.operand_id,
                    source_area_um2=a.source_area_um2,
                    effective_area_um2=a.effective_area_um2,
                    geometry_changed=a.geometry_changed,
                    duplicate_vertex_count=a.duplicate_vertex_count,
                )
                for a in self.audits
            ],
        }


@dataclass(frozen=True)
class _OverlayParts:
    geometries: tuple


@dataclass(frozen=True)
class _ZeroAreaResidue:
    geometry: object
    operand_ids: tuple


@dataclass(frozen=True)
class _Subdivision:
    cells: tuple
    residues: tuple[_ZeroAreaResidue, ...]


@dataclass(frozen=True)
class _ZeroAreaAccounting:
    boundary: tuple[_ZeroAreaResidue, ...]
    auxiliary: tuple[_ZeroAreaResidue, ...]


def _overlay_parts(geometry):
    # A result collection is not necessarily a legal input to OverlayNG.
    # Keep every nonempty dimension; homogeneous multi-geometries stay intact.
    if geometry.is_empty:
        return _OverlayParts(())
    if geometry.geom_type == "GeometryCollection":
        return _OverlayParts(
            tuple(
                part
                for child in geometry.geoms
                for part in _overlay_parts(child).geometries
            )
        )
    return _OverlayParts((geometry,))


def _geometry_dimension(geometry):
    return {
        "Polygon": 2,
        "MultiPolygon": 2,
        "LineString": 1,
        "LinearRing": 1,
        "MultiLineString": 1,
        "Point": 0,
        "MultiPoint": 0,
    }[geometry.geom_type]


def _overlay_call(operation, left, right, grid, operand_index, cell_index):
    from shapely import difference, intersection, union_all

    try:
        if operation == "union_all":
            return union_all(left, grid_size=grid)
        function = {"intersection": intersection, "difference": difference}[operation]
        return function(left, right, grid_size=grid)
    except Exception as error:
        # Diagnostic failure must never replace the original native exception.
        try:

            def describe(geometry):
                def components(value):
                    if value.is_empty:
                        return ()
                    if value.geom_type in {
                        "GeometryCollection",
                        "MultiPolygon",
                        "MultiLineString",
                        "MultiPoint",
                    }:
                        return tuple(
                            part for child in value.geoms for part in components(child)
                        )
                    return (value,)

                parts = components(geometry)
                return {
                    "geom_type": geometry.geom_type,
                    "component_dimensions": [_geometry_dimension(p) for p in parts],
                    "component_count": len(parts),
                }

            note = {
                "operation": operation,
                "operand_index": operand_index,
                "cell_index": cell_index,
                "grid_size_um": grid,
                "left": [describe(p) for p in left]
                if operation == "union_all"
                else describe(left),
                "right": None if right is None else describe(right),
            }
            error.add_note(
                "scgsim.grid.overlay_failure: " + json.dumps(note, sort_keys=True)
            )
        except Exception:
            try:
                error.add_note(
                    "scgsim.grid.overlay_failure: diagnostic context unavailable"
                )
            except Exception:
                pass
        raise


def _bounds_disjoint(left, right):
    ax0, ay0, ax1, ay1 = left.bounds
    bx0, by0, bx1, by1 = right.bounds
    return ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0


def _deduplicate_residues(residues):
    # Canonical exact coordinates include every retained collinear vertex.
    # Equal support with different labels remains a distinct semantic record.
    seen = set()
    output = []
    for residue in residues:
        key = residue.operand_ids, residue.geometry.normalize().wkb
        if key not in seen:
            seen.add(key)
            output.append(residue)
    return tuple(output)


def _parts_overlay(operation, left, right, grid, operand_index, cell_index):
    if operation == "intersection":
        return _OverlayParts(
            tuple(
                part
                for a in left.geometries
                for b in right.geometries
                if not _bounds_disjoint(a, b)
                for part in _overlay_parts(
                    _overlay_call(operation, a, b, grid, operand_index, cell_index)
                ).geometries
            )
        )
    parts = left.geometries
    for b in right.geometries:
        next_parts = []
        for a in parts:
            if (
                _geometry_dimension(a) == 2
                and _geometry_dimension(b) < 2
                or _bounds_disjoint(a, b)
            ):
                # Zero-area support cannot subtract positive-area interior.
                # Its contacts/nodes are retained by the common sideband.
                next_parts.append(a)
            else:
                next_parts.extend(
                    _overlay_parts(
                        _overlay_call(operation, a, b, grid, operand_index, cell_index)
                    ).geometries
                )
        parts = tuple(next_parts)
    return _OverlayParts(parts)


def _parts_union(parts, grid, operand_index):
    # Unary union is homogeneous too; do not reassemble a mixed input.
    output = []
    dimensions = tuple(dict.fromkeys(_geometry_dimension(p) for p in parts.geometries))
    for dimension in dimensions:
        group = [p for p in parts.geometries if _geometry_dimension(p) == dimension]
        output.extend(
            _overlay_parts(
                _overlay_call("union_all", group, None, grid, operand_index, None)
            ).geometries
        )
    return _OverlayParts(tuple(output))


def _subdivide_grid(values, geometries, grid):
    from shapely import STRtree

    cells, residues = [], []
    candidate_count = skipped_count = residues_before_dedup = 0
    with _stage("geometry.grid.overlay", counts={"operands": len(values)}):
        for operand_index, value in enumerate(values):
            original = geometries[value.operand_id]
            operand = _overlay_parts(
                _overlay_call(
                    "intersection", original, original, grid, operand_index, None
                )
            )
            pending = operand
            divided, sideband = [], []

            def retain(parts, labels):
                for part in parts.geometries:
                    if _geometry_dimension(part) == 2:
                        divided.extend((p, labels) for p in _polygons(part))
                    else:
                        sideband.append(_ZeroAreaResidue(part, labels))

            # Area order is unchanged. Lower-dimensional cells are retained and
            # subdivided separately; they can never create positive-area labels.
            previous = (*cells, *((r.geometry, r.operand_ids) for r in residues))
            tree = STRtree([cell for cell, _ in previous])
            candidates = {
                int(index) for part in operand.geometries for index in tree.query(part)
            }
            candidate_count += len(candidates)
            skipped_count += len(previous) - len(candidates)
            for cell_index, (cell, labels) in enumerate(previous):
                if cell_index not in candidates:
                    retain(_OverlayParts((cell,)), labels)
                    continue
                cell_parts = _OverlayParts((cell,))
                common = _parts_overlay(
                    "intersection", cell_parts, operand, grid, operand_index, cell_index
                )
                residual = _parts_overlay(
                    "difference", cell_parts, operand, grid, operand_index, cell_index
                )
                pending = _parts_overlay(
                    "difference", pending, cell_parts, grid, operand_index, cell_index
                )
                retain(residual, labels)
                retain(common, (*labels, value.operand_id))
            retain(_parts_union(pending, grid, operand_index), (value.operand_id,))
            residues_before_dedup += len(sideband)
            cells, residues = divided, _deduplicate_residues(sideband)
            _progress(operand_index + 1, len(values))
    with _stage(
        "geometry.grid.overlay_work",
        counts={
            "candidate_cells": candidate_count,
            "disjoint_cells": skipped_count,
            "residues_before_dedup": residues_before_dedup,
            "final_residues": len(residues),
        },
    ):
        return _Subdivision(tuple(cells), tuple(residues))


def _residue_coordinates(geometry):
    if geometry.geom_type in {"LineString", "LinearRing", "Point"}:
        return tuple(tuple(p) for p in geometry.coords)
    return tuple(p for child in geometry.geoms for p in _residue_coordinates(child))


def _account_zero_area(residues, cells, grid):
    from shapely import LineString, STRtree

    if not residues:
        return _ZeroAreaAccounting((), ())
    # Exact face edges give a tight broad-phase instead of overlaying each
    # residue against the full all-face wire. Touching envelopes are included.
    edges = {}
    for cell, _ in cells:
        for ring in (cell.exterior, *cell.interiors):
            points = tuple(tuple(p) for p in ring.coords)
            for a, b in zip(points, points[1:]):
                if a != b:
                    edges.setdefault(tuple(sorted((a, b))), LineString((a, b)))
    boundaries = tuple(edges.values())
    tree = STRtree(boundaries)
    geometries = {}
    labeled = []
    for residue in _deduplicate_residues(residues):
        key = residue.geometry.normalize().wkb
        geometries.setdefault(key, residue.geometry)
        labeled.append((key, residue.operand_ids))

    def classify(geometry):
        indices = sorted(int(index) for index in tree.query(geometry))
        wire = _parts_union(
            _OverlayParts(tuple(boundaries[i] for i in indices)), grid, None
        )
        parts = _OverlayParts((geometry,))
        common = _parts_overlay("intersection", parts, wire, grid, None, None)
        remainder = _parts_overlay("difference", parts, wire, grid, None, None)
        return common, remainder

    # Classifications are geometric only. Reattach every exact semantic label
    # in original order after ordered independent work; never merge labels.
    classified = dict(
        zip(geometries, _ordered_map(classify, tuple(geometries.values())), strict=True)
    )
    incident, auxiliary = [], []
    for key, labels in labeled:
        common, remainder = classified[key]
        incident.extend(_ZeroAreaResidue(p, labels) for p in common.geometries)
        auxiliary.extend(_ZeroAreaResidue(p, labels) for p in remainder.geometries)
    return _ZeroAreaAccounting(
        _deduplicate_residues(incident), _deduplicate_residues(auxiliary)
    )


def build_grid_arrangement(operands, context, *, apply_grid=True):
    """Construct labelled cells by common-policy polygon overlay.

    Membership is carried by intersection/difference construction, never
    recovered from samples or independently rounded linework faces. Every
    existing cell and the new operand are updated in the same subdivision.
    """
    from shapely import Polygon, union_all

    grid = context.grid_size_um if apply_grid else None
    values = (
        tuple(operands.values()) if isinstance(operands, Mapping) else tuple(operands)
    )
    ids = [v.operand_id for v in values]
    if len(ids) != len(set(ids)):
        raise ValueError("common grid operand identities are duplicated")
    with _stage("geometry.grid.inputs", counts={"operands": len(values)}):
        geometries = {v.operand_id: Polygon(v.exterior, v.holes) for v in values}
    subdivision = _subdivide_grid(values, geometries, grid)
    cells = subdivision.cells
    with _stage(
        "geometry.grid.zero_area", counts={"residues": len(subdivision.residues)}
    ):
        zero_area = _account_zero_area(subdivision.residues, cells, grid)
    residue_points = {
        point
        for residue in (*zero_area.boundary, *zero_area.auxiliary)
        for point in _residue_coordinates(residue.geometry)
    }
    with _stage("geometry.grid.vertices", counts={"cells": len(cells)}):
        vertices = tuple(
            sorted(
                {
                    tuple(p)
                    for cell, _ in cells
                    for ring in (cell.exterior, *cell.interiors)
                    for p in ring.coords
                }
                | residue_points
            )
        )
    with _stage(
        "geometry.grid.cell_regions",
        counts={"cells": len(cells), "vertices": len(vertices)},
    ):
        faces = []
        vertex_index = _region_vertex_index(vertices, grid)
        for cell_index, (cell, members) in enumerate(cells):
            outer, holes = _regions_indexed(cell, vertex_index)[0]
            identity = sha256(
                json.dumps([outer, holes], separators=(",", ":")).encode()
            ).hexdigest()
            faces.append(
                GridFace(
                    "GRID_FACE__" + identity[:24], outer, holes, tuple(sorted(members))
                )
            )
            _progress(cell_index + 1, len(cells))
        faces = tuple(sorted(faces, key=lambda f: f.face_id))
        vertices = tuple(
            sorted(
                {
                    p
                    for face in faces
                    for ring in (face.exterior, *face.holes)
                    for p in ring
                }
            )
        )
    # Boundary and auxiliary residues remain inspectable internally through
    # the exact physical audits; neither can create an orphan area-graph edge.
    with _stage("geometry.grid.effective", counts={"operands": len(values)}):

        def realize(value):
            return union_all(
                [
                    Polygon(f.exterior, f.holes)
                    for f in faces
                    if value.operand_id in f.operand_ids
                ],
                grid_size=grid,
            )

        effective = dict(zip(ids, _ordered_map(realize, values), strict=True))
    with _stage("geometry.grid.physical_audit", counts={"operands": len(values)}):
        audits = []
        for operand_index, value in enumerate(values):
            raw, final = geometries[value.operand_id], effective[value.operand_id]
            if value.physical and (
                final.is_empty
                or len(_polygons(raw)) != len(_polygons(final))
                or sum(len(p.interiors) for p in _polygons(raw))
                != sum(len(p.interiors) for p in _polygons(final))
            ):
                error = ValueError(
                    f"physical geometry collapsed or changed component/hole topology at grid: {value.operand_id}"
                )
                try:

                    def summary(shape):
                        polygons = _polygons(shape)
                        return {
                            "empty": bool(shape.is_empty),
                            "is_valid": bool(shape.is_valid),
                            "components": len(polygons),
                            "holes": sum(len(p.interiors) for p in polygons),
                            "area_um2": float(shape.area),
                        }

                    error.add_note(
                        "scgsim.grid.physical_topology: "
                        + json.dumps(
                            {
                                "operand_index": operand_index,
                                "grid_size_um": grid,
                                "requested": summary(raw),
                                "realized": summary(final),
                            },
                            sort_keys=True,
                        )
                    )
                except Exception:
                    error.add_note("scgsim.grid.physical_topology: metrics unavailable")
                raise error
            audits.append(
                GridOperandAudit(
                    value.operand_id,
                    raw.area,
                    final.area,
                    not raw.equals(final),
                    sum(len(r) - len(set(r)) for r in (value.exterior, *value.holes)),
                )
            )
            _progress(operand_index + 1, len(values))
    contacts = [v for v in values if v.contact]
    with _stage("geometry.grid.contact_audit", counts={"operands": len(contacts)}):
        for i, first in enumerate(contacts):
            for second in contacts[i + 1 :]:
                before = geometries[first.operand_id].relate(
                    geometries[second.operand_id]
                )
                after = effective[first.operand_id].relate(effective[second.operand_id])
                if before != after:
                    raise ValueError(
                        f"physical contact changed at grid: {first.operand_id}, {second.operand_id}"
                    )
            _progress(i + 1, len(contacts))
    with _stage("geometry.grid.edges", counts={"faces": len(faces)}):
        edges = tuple(
            sorted(
                {
                    tuple(sorted((a, b)))
                    for f in faces
                    for ring in (f.exterior, *f.holes)
                    for a, b in zip(ring, (*ring[1:], ring[0]))
                }
            )
        )
    return GridArrangement(context, faces, vertices, edges, tuple(audits), apply_grid)


@dataclass(frozen=True)
class PolygonRegion:
    """Simple exterior/holes; cutline walk is mechanical legacy serialization."""

    exterior: tuple
    holes: tuple = ()

    @property
    def points(self):
        def area(ring):
            return sum(
                a[0] * b[1] - a[1] * b[0] for a, b in zip(ring, (*ring[1:], ring[0]))
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

    def area(self):
        def area(ring):
            return (
                abs(
                    sum(
                        a[0] * b[1] - a[1] * b[0]
                        for a, b in zip(ring, (*ring[1:], ring[0]))
                    )
                )
                / 2
            )

        return area(self.exterior) - sum(area(r) for r in self.holes)


def polygon_rings(polygon):
    if isinstance(polygon, PolygonRegion):
        return polygon.exterior, polygon.holes
    from scgsim.geometry._primitives.loops import _split_gdstk_cutline_loop

    return _split_gdstk_cutline_loop(
        tuple(tuple(map(float, p)) for p in polygon.points)
    )


_POLYGON_FRAME = ContextVar("scgsim_polygon_world_frame", default=None)


@contextmanager
def polygon_world_frame(transform):
    token = _POLYGON_FRAME.set(tuple(transform))
    try:
        yield
    finally:
        _POLYGON_FRAME.reset(token)


def in_occurrence_frame(function):
    from functools import wraps

    @wraps(function)
    def framed(*args, transform, **kwargs):
        with polygon_world_frame(transform):
            return function(*args, transform=transform, **kwargs)

    return framed


def _frame_point(point, inverse=False):
    frame = _POLYGON_FRAME.get()
    if frame is None:
        return tuple(point)
    a, b, c, d, x, y = frame
    u, v = point
    if inverse:
        determinant = a * d - b * c
        return (
            (d * (u - x) - b * (v - y)) / determinant,
            (-c * (u - x) + a * (v - y)) / determinant,
        )
    return a * u + b * v + x, c * u + d * v + y


def polygon_boolean(left, right, operation, *, precision=1e-9):
    """One GEOS labelled overlay for bound data; historical gdstk unchanged."""
    context = current_grid_context()
    if context is None:
        import gdstk

        return gdstk.boolean(left, right, operation, precision=precision)
    values = []
    for label, polygons in (("L", left), ("R", right)):
        for index, polygon in enumerate(polygons):
            outer, holes = polygon_rings(polygon)
            values.append(
                GridOperand(
                    f"{label}{index}",
                    tuple(_frame_point(p) for p in outer),
                    tuple(tuple(_frame_point(p) for p in ring) for ring in holes),
                    physical=False,
                    contact=False,
                )
            )
    arrangement = build_grid_arrangement(
        values, context, apply_grid=not _SOURCE_INTENT.get()
    )
    selected = []
    for face in arrangement.faces:
        first = any(name.startswith("L") for name in face.operand_ids)
        second = any(name.startswith("R") for name in face.operand_ids)
        retain = {
            "and": first and second,
            "or": first or second,
            "not": first and not second,
            "xor": first != second,
        }[operation]
        if retain:
            selected.append(face.face_id)
    return [
        PolygonRegion(
            tuple(_frame_point(p, True) for p in outer),
            tuple(tuple(_frame_point(p, True) for p in ring) for ring in holes),
        )
        for outer, holes in arrangement.face_regions(selected)
    ]


def source_grid_audits():
    return tuple(_SOURCE_AUDITS.get() or ())


def preserve_declared_source(function):
    """Compare declared physical source result before destructive quantization.

    The reference evaluates the SAME declared mask/include/selector recipe in
    unquantized coordinates solely for its preservation audit. It is never
    substituted for the actual shared-grid geometry.
    """
    from functools import wraps
    from dataclasses import replace

    @wraps(function)
    def preserved(entity, *args, **kwargs):
        context = current_grid_context()
        if context is None or _SOURCE_INTENT.get():
            return function(entity, *args, **kwargs)
        token = _SOURCE_INTENT.set(True)
        try:
            with _stage("geometry.grid.source_reference"):
                requested = function(entity, *args, **kwargs)
        finally:
            _SOURCE_INTENT.reset(token)
        with _stage("geometry.grid.source_realization"):
            realized = function(entity, *args, **kwargs)
        from shapely import Polygon, union_all

        def geometry(polygons):
            return union_all(
                [
                    Polygon(
                        tuple(_frame_point(p) for p in v.exterior),
                        tuple(tuple(_frame_point(p) for p in r) for r in v.holes),
                    )
                    for v in polygons
                ]
            )

        with _stage("geometry.grid.source_audit"):
            raw, final = geometry(requested), geometry(realized)
            before = (
                len(_polygons(raw)),
                sum(len(p.interiors) for p in _polygons(raw)),
            )
            after = (
                len(_polygons(final)),
                sum(len(p.interiors) for p in _polygons(final)),
            )
            if before != after:
                raise ValueError(
                    f"declared physical source component/hole topology changed at grid: {entity.semantic_id}"
                )
            raw_regions = _regions(raw)
            requested_by_id = {p.polygon_id: p for p in requested}
            if set(requested_by_id) != {p.polygon_id for p in realized}:
                raise ValueError(
                    f"declared source polygon identity changed at grid: {entity.semantic_id}"
                )
            requested_polygons = [
                {
                    "source_polygon_id": p.polygon_id,
                    "exterior": tuple(_frame_point(v) for v in p.exterior),
                    "holes": tuple(tuple(_frame_point(v) for v in r) for r in p.holes),
                }
                for p in requested
            ]
            ledger = {
                "source_entity_id": entity.semantic_id,
                "operation": function.__name__,
                "geometry_grid": context.to_payload(),
                "requested_regions": raw_regions,
                "requested_polygons": requested_polygons,
                "requested_geometry_sha256": sha256(
                    json.dumps(raw_regions, separators=(",", ":")).encode()
                ).hexdigest(),
                "requested_area_um2": raw.area,
                "realized_area_um2": final.area,
                "requested_components": before[0],
                "realized_components": after[0],
                "requested_holes": before[1],
                "realized_holes": after[1],
                "geometry_changed": not raw.equals(final),
            }
            audits = _SOURCE_AUDITS.get()
            if audits is not None and ledger not in audits:
                audits.append(ledger)
            return tuple(
                replace(
                    p,
                    metadata={
                        **p.metadata,
                        "geometry_grid_source_identity": {
                            "source_entity_id": entity.semantic_id,
                            "source_polygon_id": p.polygon_id,
                        },
                    },
                )
                for p in realized
            )

    return preserved
