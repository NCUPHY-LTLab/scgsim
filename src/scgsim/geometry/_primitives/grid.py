"""Shared fixed-precision noding, atomic faces and full operand membership."""

import json
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import cached_property, wraps
from hashlib import sha256
from types import MappingProxyType
from typing import Any

from scgsim.diagnostics import _progress, _stage
from scgsim.geometry._primitives._grid_cache import cached_subdivision
from scgsim.geometry._primitives._tagged_grid import build_tagged_subdivision
from scgsim.geometry._primitives.constants import _TOPOLOGY_EPS_UM
from scgsim.geometry._primitives.grid_overlay import (
    OverlayParts,
    Subdivision,
    ZeroAreaAccounting,
    ZeroAreaResidue,
    account_zero_area,
    cell_overlay_chunk,
    deduplicate_residues,
    overlay_call,
    parts_overlay,
    residue_coordinates,
    subdivide_grid,
)
from scgsim.geometry._primitives.grid_regions import (
    polygon_components,
    region_key,
    region_vertex_index,
    regions,
    regions_indexed,
)
from scgsim.geometry._primitives.planar_loops import (
    clean_loop,
    cutline_boundary_loops,
    filter_gdstk_polygons,
    polygon_area,
)
from scgsim.geometry.models.grid import GeometryGridContext, geometry_grid_context
from scgsim.preparation import _ordered_map

_ACTIVE_GRID = ContextVar("scgsim_geometry_grid", default=None)
_SOURCE_INTENT = ContextVar("scgsim_source_intent_overlay", default=False)
_SOURCE_AUDITS = ContextVar("scgsim_source_grid_audits", default=None)


def current_grid_context():
    return _ACTIVE_GRID.get()


def with_input_grid(function):
    """Scope existing leaf operations to the explicit input policy only."""

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


@dataclass(frozen=True)
class GridArrangement:
    context: GeometryGridContext
    faces: tuple[GridFace, ...]
    vertices: tuple
    edges: tuple
    audits: tuple[GridOperandAudit, ...]
    grid_applied: bool = True

    @cached_property
    def _vertex_index(self):
        grid = self.context.grid_size_um if self.grid_applied else None
        return region_vertex_index(self.vertices, grid)

    @cached_property
    def _operand_faces(self):

        selected = {}
        for index, face in enumerate(self.faces):
            for operand_id in face.operand_ids:
                selected.setdefault(operand_id, []).append(index)
        return MappingProxyType({key: tuple(value) for key, value in selected.items()})

    @cached_property
    def _region_work(self):
        # Per-arrangement successful pure results only. Concurrent readers may
        # compute the same miss; immutable geometry/results are never mutated.
        return {}, {}

    def _regions_for_faces(self, indices):
        from shapely import (
            Polygon,
            coverage_union_all,
            union_all,
        )  # optional geometry dependency  # noqa: PLC0415 - optional geometry dependency

        polygons, regions = self._region_work
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
        with _stage("geometry.grid.regions"):
            merged = (
                coverage_union_all(selected)
                if self.grid_applied
                else union_all(selected)
            )
            result = regions_indexed(merged, self._vertex_index)
        regions[indices] = result
        return result

    def face_regions(self, face_ids):
        return self._regions_for_faces(
            tuple(
                index
                for index, face in enumerate(self.faces)
                if face.face_id in face_ids
            )
        )

    def operand_regions(self, operand_id):
        return self._regions_for_faces(self._operand_faces.get(operand_id, ()))

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
class _GridWork:
    raw_polygons: tuple
    subdivision: Subdivision


def build_grid_arrangement(operands, context, *, apply_grid=True):
    """Construct one labelled fixed-grid network or an ungridded plane overlay.

    Fixed-grid membership follows directed source paths and face transitions.
    Ungridded parent planes retain their distinct existing overlay contract.
    """
    from shapely import (
        Polygon,
        coverage_union_all,
        union_all,
    )  # optional geometry dependency

    grid = context.grid_size_um if apply_grid else None
    values = (
        tuple(operands.values()) if isinstance(operands, Mapping) else tuple(operands)
    )
    ids = [v.operand_id for v in values]
    if len(ids) != len(set(ids)):
        raise ValueError("common grid operand identities are duplicated")

    def construct():
        with _stage("geometry.grid.inputs", counts={"operands": len(values)}):
            raw_polygons = tuple(
                (v.operand_id, Polygon(v.exterior, v.holes)) for v in values
            )
        geometries = dict(raw_polygons)
        if not apply_grid:
            return _GridWork(raw_polygons, subdivide_grid(values, geometries, None))

        result = build_tagged_subdivision(values, geometries, grid)
        accounting = ZeroAreaAccounting(
            tuple(ZeroAreaResidue(*row) for row in result.boundary),
            tuple(ZeroAreaResidue(*row) for row in result.auxiliary),
        )
        return _GridWork(
            raw_polygons,
            Subdivision(
                result.cells,
                (*accounting.boundary, *accounting.auxiliary),
                accounting,
                result.source_paths,
                result.edge_origins,
            ),
        )

    work = cached_subdivision(values, context, apply_grid=apply_grid, build=construct)
    geometries = dict(work.raw_polygons)
    subdivision = work.subdivision
    cells = subdivision.cells
    with _stage(
        "geometry.grid.zero_area", counts={"residues": len(subdivision.residues)}
    ):
        zero_area = subdivision.accounting
        if zero_area is None:
            zero_area = account_zero_area(subdivision.residues, cells, grid)
    residue_points = {
        point
        for residue in (*zero_area.boundary, *zero_area.auxiliary)
        for point in residue_coordinates(residue.geometry)
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
        vertex_index = region_vertex_index(vertices, grid)
        for cell_index, (cell, members) in enumerate(cells):
            outer, holes = regions_indexed(cell, vertex_index)[0]
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
        # One immutable polygon per atomic face, including multiply-labelled
        # cells. Preserve face order within each operand's exact union.
        face_polygons = tuple(Polygon(face.exterior, face.holes) for face in faces)
        operand_faces = {operand_id: [] for operand_id in ids}
        for index, face in enumerate(faces):
            for operand_id in face.operand_ids:
                operand_faces[operand_id].append(index)

        def realize(value):
            selected = [
                face_polygons[index] for index in operand_faces[value.operand_id]
            ]
            return coverage_union_all(selected) if apply_grid else union_all(selected)

        effective = dict(zip(ids, _ordered_map(realize, values), strict=True))
    audits = _audit_physical_operands(values, geometries, effective, cells, faces, grid)
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
                    for a, b in zip(ring, (*ring[1:], ring[0]), strict=False)
                }
            )
        )
    return GridArrangement(context, faces, vertices, edges, tuple(audits), apply_grid)


def _audit_physical_operands(values, geometries, effective, cells, faces, grid):
    """Preserve source topology and attach the existing error-only witnesses."""
    from shapely import Polygon, union_all  # optional geometry dependency

    with _stage("geometry.grid.physical_audit", counts={"operands": len(values)}):
        audits = []
        for operand_index, value in enumerate(values):
            raw, final = geometries[value.operand_id], effective[value.operand_id]
            if value.physical and (
                final.is_empty
                or len(polygon_components(raw)) != len(polygon_components(final))
                or sum(len(p.interiors) for p in polygon_components(raw))
                != sum(len(p.interiors) for p in polygon_components(final))
            ):
                error = ValueError(
                    f"physical geometry collapsed or changed component/hole topology at grid: {value.operand_id}"
                )
                try:

                    def summary(shape):
                        polygons = polygon_components(shape)
                        return {
                            "empty": bool(shape.is_empty),
                            "is_valid": bool(shape.is_valid),
                            "components": len(polygons),
                            "holes": sum(len(p.interiors) for p in polygons),
                            "area_um2": float(shape.area),
                        }

                    metrics = {
                        "operand_index": operand_index,
                        "grid_size_um": grid,
                        "requested": summary(raw),
                        "realized": summary(final),
                    }
                    # Error-only witnesses separate input rounding, subdivision
                    # coverage, and final re-union. They never enter the output
                    # or change the original physical preservation decision.
                    stages = (
                        (
                            "initial_grid_realization",
                            lambda raw=raw, operand_index=operand_index: overlay_call(
                                "intersection", raw, raw, grid, operand_index, None
                            ),
                        ),
                        (
                            "labelled_cells_without_resnap",
                            lambda value=value: union_all(
                                [
                                    cell
                                    for cell, labels in cells
                                    if value.operand_id in labels
                                ],
                                grid_size=0,
                            ),
                        ),
                        (
                            "reconstructed_faces_without_resnap",
                            lambda value=value: union_all(
                                [
                                    Polygon(face.exterior, face.holes)
                                    for face in faces
                                    if value.operand_id in face.operand_ids
                                ],
                                grid_size=0,
                            ),
                        ),
                    )
                    for name, operation in stages:
                        try:
                            metrics[name] = summary(operation())
                        except Exception as diagnostic_error:  # error-only witnesses preserve the original failure
                            metrics[name] = {
                                "unavailable": True,
                                "error_type": type(diagnostic_error).__name__,
                            }
                    error.add_note(
                        "scgsim.grid.physical_topology: "
                        + json.dumps(metrics, sort_keys=True)
                    )
                except (
                    Exception
                ):  # diagnostic failure cannot replace the topology error
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
    return audits


@dataclass(frozen=True)
class PolygonRegion:
    """Simple exterior/holes; cutline walk is mechanical legacy serialization."""

    exterior: tuple
    holes: tuple = ()

    @property
    def points(self):
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

    def area(self):
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


def polygon_rings(polygon):
    if isinstance(polygon, PolygonRegion):
        return polygon.exterior, polygon.holes

    return split_gdstk_cutline_loop(tuple(tuple(map(float, p)) for p in polygon.points))


_POLYGON_FRAME = ContextVar("scgsim_polygon_world_frame", default=None)


@contextmanager
def polygon_world_frame(transform):
    token = _POLYGON_FRAME.set(tuple(transform))
    try:
        yield
    finally:
        _POLYGON_FRAME.reset(token)


def in_occurrence_frame(function):
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
        import gdstk  # optional native geometry dependency

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


@dataclass(frozen=True)
class _SourceRecipeOutcome:
    result: object
    audits: tuple
    error: BaseException | None
    traceback: object


def _source_recipe_map(function, items):
    """Run independent recipes with isolated audits and source-order merging."""
    parent_audits = _SOURCE_AUDITS.get()

    def execute(item):
        local_audits = [] if parent_audits is not None else None
        token = _SOURCE_AUDITS.set(local_audits)
        try:
            try:
                result = function(item)
            except (
                BaseException
            ) as error:  # ordered worker failures must retain their original type
                return _SourceRecipeOutcome(
                    None, tuple(local_audits or ()), error, error.__traceback__
                )
            return _SourceRecipeOutcome(result, tuple(local_audits or ()), None, None)
        finally:
            _SOURCE_AUDITS.reset(token)

    outcomes = _ordered_map(execute, items)
    results = []
    for outcome in outcomes:
        if parent_audits is not None:
            for row in outcome.audits:
                if row not in parent_audits:
                    parent_audits.append(row)
        if outcome.error is not None:
            raise outcome.error.with_traceback(outcome.traceback)
        results.append(outcome.result)
    return results


def preserve_declared_source(function):
    """Compare declared physical source result before destructive quantization.

    The reference evaluates the SAME declared mask/include/selector recipe in
    unquantized coordinates solely for its preservation audit. It is never
    substituted for the actual shared-grid geometry.
    """

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
        from shapely import Polygon, union_all  # optional geometry dependency  # noqa: PLC0415 - optional geometry dependency

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
                len(polygon_components(raw)),
                sum(len(p.interiors) for p in polygon_components(raw)),
            )
            after = (
                len(polygon_components(final)),
                sum(len(p.interiors) for p in polygon_components(final)),
            )
            if before != after:
                raise ValueError(
                    f"declared physical source component/hole topology changed at grid: {entity.semantic_id}"
                )
            raw_regions = regions(raw)
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


def boolean_polygon_region(
    gdstk: Any,
    left: Sequence[Any],
    right: Sequence[Any],
    operation: str,
) -> tuple[Any, ...]:
    """Apply the scoped polygon Boolean and retain positive-area output polygons."""
    if not left:
        return ()
    if operation == "not" and not right:
        return filter_gdstk_polygons(left)
    result = polygon_boolean(
        left,
        right,
        operation,
        precision=1e-9,
    )
    return filter_gdstk_polygons(result or ())


def loop_inside_loop(loop: Any, container: Any) -> bool:
    """Test containment through the active grid-aware intersection area path."""
    import gdstk  # optional native geometry dependency

    loop_points = clean_loop(loop)
    clean_container = clean_loop(container)
    loop_area = abs(polygon_area(loop_points))
    intersection = boolean_polygon_region(
        gdstk,
        (gdstk.Polygon(loop_points),),
        (gdstk.Polygon(clean_container),),
        "and",
    )
    intersection_area = sum(
        abs(polygon_area(polygon.points)) for polygon in intersection
    )
    return loop_area - intersection_area <= max(_TOPOLOGY_EPS_UM, loop_area * 1e-9)


def split_gdstk_cutline_loop(
    loop: tuple[tuple[float, float], ...],
) -> tuple[
    tuple[tuple[float, float], ...], tuple[tuple[tuple[float, float], ...], ...]
]:
    """Rebuild cutline boundaries and verify holes under the active grid scope."""
    outer, holes = cutline_boundary_loops(loop)
    if any(not loop_inside_loop(hole, outer) for hole in holes):
        raise ValueError("gdstk cutline polygon has a non-interior boundary loop")
    return outer, holes


# Transitional aliases preserve callers during the coordinated internal migration.
_region_key = region_key
_regions = regions
_regions_indexed = regions_indexed
_OverlayParts = OverlayParts
_ZeroAreaResidue = ZeroAreaResidue
_overlay_call = overlay_call
_subdivide_grid = subdivide_grid
_account_zero_area = account_zero_area

_parts_overlay = parts_overlay
_cell_overlay_chunk = cell_overlay_chunk
_deduplicate_residues = deduplicate_residues
