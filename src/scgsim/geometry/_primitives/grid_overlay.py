"""Run labelled GEOS overlays and preserve area, zero-area support and contacts.

OverlayScope carries grid and failure-location context through native calls.
Legacy positional entry points retain existing callers during their migration.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cached_property
from itertools import pairwise
from types import TracebackType
from typing import TYPE_CHECKING, Literal, cast, overload

from scgsim.diagnostics import (  # private-import: diagnostics owner retains its shared stage and progress entry points
    _progress,
    _stage,
)
from scgsim.geometry._primitives.grid_regions import (  # private-import: primitive package-path migration awaits coordinated backend source lists
    GridOperandAudit,
    polygon_components,
)
from scgsim.preparation import (  # private-import: preparation owner retains the existing ordered-worker entry point
    _ordered_map,
    current_preparation_workers,
)

if TYPE_CHECKING:
    from shapely.geometry import (
        GeometryCollection,
        MultiLineString,
        MultiPoint,
        MultiPolygon,
        Polygon,
    )
    from shapely.geometry.base import BaseGeometry

    from scgsim.geometry._primitives._tagged_grid import (  # private-import: primitive package-path migration awaits coordinated backend source lists
        SourcePath,
    )
    from scgsim.geometry._primitives.grid_regions import (  # private-import: primitive package-path migration awaits coordinated backend source lists
        GridFace,
        GridOperand,
        GridPoint,
    )


type LabelledCell = tuple[Polygon, tuple[str, ...]]
type OverlayInput = BaseGeometry | Sequence[BaseGeometry]
type LatticePoint = tuple[int, ...]
type LatticeEdge = tuple[LatticePoint, LatticePoint]
# Each origin stores its directed source-path index and orientation.
type EdgeOrigin = tuple[int, int]
type SourceEdgeOrigins = tuple[LatticeEdge, tuple[EdgeOrigin, ...]]
type CellRows = tuple[tuple[int, BaseGeometry], ...]
type CellWork = tuple[CellRows, OverlayParts, float | None, int | None]


@dataclass(frozen=True)
class OverlayParts:
    """Nonempty native geometries kept separate for dimension-safe overlays."""

    geometries: tuple[BaseGeometry, ...]


@dataclass(frozen=True)
class OverlayScope:
    """Fixed precision and original operand/cell indices for native error notes."""

    grid: float | None
    operand_index: int | None
    cell_index: int | None = None

    @overload
    def call(
        self, operation: Literal["union_all"], left: Sequence[BaseGeometry], right: None
    ) -> BaseGeometry: ...

    @overload
    def call(
        self, operation: str, left: BaseGeometry, right: BaseGeometry
    ) -> BaseGeometry: ...

    @overload
    def call(
        self, operation: str, left: OverlayInput, right: BaseGeometry | None
    ) -> BaseGeometry | None: ...

    def call(
        self, operation: str, left: OverlayInput, right: BaseGeometry | None
    ) -> BaseGeometry | None:
        """Run one native operation and preserve its original failure and notes."""
        from shapely import difference, intersection, union_all  # noqa: PLC0415 - native GEOS loads only when an overlay is requested

        try:
            if operation == "union_all":
                # Operation selects the native input shape; these casts add no
                # checks ahead of GEOS and keep its error handling authoritative.
                return union_all(
                    cast("Sequence[BaseGeometry]", left), grid_size=self.grid
                )
            function = {"intersection": intersection, "difference": difference}[
                operation
            ]
            return function(cast("BaseGeometry", left), right, grid_size=self.grid)
        except Exception as error:
            self._annotate_failure(error, operation, left, right)
            raise

    def _annotate_failure(
        self,
        error: Exception,
        operation: str,
        left: OverlayInput,
        right: BaseGeometry | None,
    ) -> None:
        try:
            note = {
                "operation": operation,
                "operand_index": self.operand_index,
                "cell_index": self.cell_index,
                "grid_size_um": self.grid,
                "left": [
                    _describe_geometry(p) for p in cast("Iterable[BaseGeometry]", left)
                ]
                if operation == "union_all"
                else _describe_geometry(cast("BaseGeometry", left)),
                "right": None if right is None else _describe_geometry(right),
            }
            error.add_note(
                "scgsim.grid.overlay_failure: " + json.dumps(note, sort_keys=True)
            )
        except Exception:  # noqa: BLE001 - diagnostic failure cannot replace the original native exception
            try:
                error.add_note(
                    "scgsim.grid.overlay_failure: diagnostic context unavailable"
                )
            except Exception:  # noqa: BLE001 - even an annotation failure must preserve the original native exception
                pass  # noqa: S110 - logging here could replace the native failure

    def parts(
        self, operation: str, left: OverlayParts, right: OverlayParts
    ) -> OverlayParts:
        """Overlay dimension-safe parts in original operand and cell order."""
        if operation == "intersection":
            return OverlayParts(
                tuple(
                    part
                    for a in left.geometries
                    for b in right.geometries
                    if not _bounds_disjoint(a, b)
                    for part in _overlay_parts(self.call(operation, a, b)).geometries
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
                        _overlay_parts(self.call(operation, a, b)).geometries
                    )
            parts = tuple(next_parts)
        return OverlayParts(parts)

    def union(self, parts: OverlayParts) -> OverlayParts:
        """Union each native dimension separately without mixed collections."""
        output = []
        dimensions = tuple(
            dict.fromkeys(_geometry_dimension(p) for p in parts.geometries)
        )
        for dimension in dimensions:
            group = [p for p in parts.geometries if _geometry_dimension(p) == dimension]
            output.extend(
                _overlay_parts(self.call("union_all", group, None)).geometries
            )
        return OverlayParts(tuple(output))


def _describe_geometry(geometry: BaseGeometry) -> dict[str, object]:
    parts = _geometry_components(geometry)
    return {
        "geom_type": geometry.geom_type,
        "component_dimensions": [_geometry_dimension(p) for p in parts],
        "component_count": len(parts),
    }


def _geometry_components(geometry: BaseGeometry) -> tuple[BaseGeometry, ...]:
    if geometry.is_empty:
        return ()
    if geometry.geom_type in {
        "GeometryCollection",
        "MultiPolygon",
        "MultiLineString",
        "MultiPoint",
    }:
        # Native collection dispatch supplies geoms; no duplicate runtime check.
        collection = cast(
            "GeometryCollection | MultiLineString | MultiPoint | MultiPolygon", geometry
        )
        return tuple(
            part for child in collection.geoms for part in _geometry_components(child)
        )
    return (geometry,)


def _overlay_parts(geometry: BaseGeometry) -> OverlayParts:
    # Keep every nonempty dimension; homogeneous multi-geometries stay intact.
    if geometry.is_empty:
        return OverlayParts(())
    if geometry.geom_type == "GeometryCollection":
        collection = cast("GeometryCollection", geometry)
        return OverlayParts(
            tuple(
                part
                for child in collection.geoms
                for part in _overlay_parts(child).geometries
            )
        )
    return OverlayParts((geometry,))


def _geometry_dimension(geometry: BaseGeometry) -> int:
    return {
        "Polygon": 2,
        "MultiPolygon": 2,
        "LineString": 1,
        "LinearRing": 1,
        "MultiLineString": 1,
        "Point": 0,
        "MultiPoint": 0,
    }[geometry.geom_type]


def _bounds_disjoint(left: BaseGeometry, right: BaseGeometry) -> bool:
    ax0, ay0, ax1, ay1 = left.bounds
    bx0, by0, bx1, by1 = right.bounds
    return ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0


def overlay_call(  # noqa: PLR0913 - retain the positional boundary used by unmigrated callers
    operation: str,
    left: OverlayInput,
    right: BaseGeometry | None,
    grid: float | None,
    operand_index: int | None,
    cell_index: int | None,
) -> BaseGeometry | None:
    """Run the scoped native operation through the existing positional boundary."""
    return OverlayScope(grid, operand_index, cell_index).call(operation, left, right)


def parts_overlay(  # noqa: PLR0913 - retain the positional boundary used by unmigrated callers
    operation: str,
    left: OverlayParts,
    right: OverlayParts,
    grid: float | None,
    operand_index: int | None,
    cell_index: int | None,
) -> OverlayParts:
    """Overlay scoped parts through the existing positional boundary."""
    return OverlayScope(grid, operand_index, cell_index).parts(operation, left, right)


@dataclass(frozen=True)
class ZeroAreaResidue:
    """A zero-area geometry with its exact operand membership."""

    geometry: BaseGeometry
    operand_ids: tuple[str, ...]

    @cached_property
    def exact_key(self) -> tuple[tuple[str, ...], bytes]:
        # Equal support with different labels remains a distinct semantic record.
        return self.operand_ids, self.geometry.normalize().wkb


@dataclass(frozen=True)
class ZeroAreaAccounting:
    """Zero-area support classified as incident boundary or auxiliary geometry."""

    boundary: tuple[ZeroAreaResidue, ...]
    auxiliary: tuple[ZeroAreaResidue, ...]


@dataclass(frozen=True)
class Subdivision:
    """Labelled area cells and their zero-area support and source provenance."""

    cells: tuple[LabelledCell, ...]
    residues: tuple[ZeroAreaResidue, ...]
    accounting: ZeroAreaAccounting | None = None
    source_paths: tuple[SourcePath, ...] = ()
    edge_origins: tuple[SourceEdgeOrigins, ...] = ()


def deduplicate_residues(
    residues: Iterable[ZeroAreaResidue],
) -> tuple[ZeroAreaResidue, ...]:
    """Retain one exact geometry record for each distinct semantic label set."""
    seen = set()
    output = []
    for residue in residues:
        key = residue.exact_key
        if key not in seen:
            seen.add(key)
            output.append(residue)
    return tuple(output)


@dataclass(frozen=True)
class _CellOverlaySuccess:
    common: OverlayParts
    residual: OverlayParts
    error: None = None
    traceback: None = None


@dataclass(frozen=True)
class _CellOverlayFailure:
    common: None
    residual: None
    error: BaseException
    traceback: TracebackType | None


type _CellOverlayOutcome = _CellOverlaySuccess | _CellOverlayFailure


def _cell_overlay_pair(
    cell: BaseGeometry, operand: OverlayParts, scope: OverlayScope
) -> _CellOverlayOutcome:
    parts = OverlayParts((cell,))
    try:
        common = scope.parts("intersection", parts, operand)
        residual = scope.parts("difference", parts, operand)
        return _CellOverlaySuccess(common, residual)
    except BaseException as error:  # noqa: BLE001 - retain original worker failures until their source-order turn
        return _CellOverlayFailure(None, None, error, error.__traceback__)


def cell_overlay_chunk(work: CellWork) -> tuple[tuple[int, _CellOverlayOutcome], ...]:
    """Evaluate immutable cell pairs in original order for a worker chunk."""
    rows, operand, grid, operand_index = work
    scope = OverlayScope(grid, operand_index)
    return tuple(
        (index, _cell_overlay_pair(cell, operand, replace(scope, cell_index=index)))
        for index, cell in rows
    )


@dataclass
class _SubdivisionStep:
    scope: OverlayScope
    operand_id: str
    operand: OverlayParts
    previous_cells: Sequence[LabelledCell]
    previous_residues: Sequence[ZeroAreaResidue]
    divided: list[LabelledCell]
    sideband: list[ZeroAreaResidue]

    def retain(self, parts: OverlayParts, labels: tuple[str, ...]) -> None:
        for part in parts.geometries:
            if _geometry_dimension(part) == 2:
                self.divided.extend((p, labels) for p in polygon_components(part))
            else:
                self.sideband.append(ZeroAreaResidue(part, labels))

    def divide(
        self,
        previous: Sequence[tuple[BaseGeometry, tuple[str, ...]]],
        candidates: set[int],
    ) -> None:
        pairs = self._candidate_pairs(previous, candidates)
        pending = self.operand
        for cell_index, (cell, labels) in enumerate(previous):
            if cell_index not in candidates:
                if cell_index < len(self.previous_cells):
                    self.divided.append(self.previous_cells[cell_index])
                else:
                    # Preserve the immutable residue and its cached exact key.
                    self.sideband.append(
                        self.previous_residues[cell_index - len(self.previous_cells)]
                    )
                continue
            scope = replace(self.scope, cell_index=cell_index)
            pair = pairs.get(cell_index)
            if pair is None:
                pair = _cell_overlay_pair(cell, self.operand, scope)
            if pair.error is not None:
                raise pair.error.with_traceback(pair.traceback)
            pending = scope.parts("difference", pending, OverlayParts((cell,)))
            self.retain(pair.residual, labels)
            self.retain(pair.common, (*labels, self.operand_id))
        self.retain(self.scope.union(pending), (self.operand_id,))

    def _candidate_pairs(
        self,
        previous: Sequence[tuple[BaseGeometry, tuple[str, ...]]],
        candidates: set[int],
    ) -> dict[int, _CellOverlayOutcome]:
        workers = current_preparation_workers()
        if workers is None or workers <= 1 or not candidates:
            return {}
        rows = tuple((index, previous[index][0]) for index in sorted(candidates))
        chunk_size = (len(rows) + workers - 1) // workers
        work = tuple(
            (
                rows[start : start + chunk_size],
                self.operand,
                self.scope.grid,
                self.scope.operand_index,
            )
            for start in range(0, len(rows), chunk_size)
        )
        # Complete immutable pair work before serial pending updates; the caller
        # and worker pool never process these native overlays concurrently.
        return dict(
            row for chunk in _ordered_map(cell_overlay_chunk, work) for row in chunk
        )


def subdivide_grid(
    values: Sequence[GridOperand],
    geometries: Mapping[str, BaseGeometry],
    grid: float | None,
) -> Subdivision:
    """Incrementally subdivide operands while retaining every exact semantic label."""
    from shapely import STRtree  # noqa: PLC0415 - native GEOS loads only when subdivision is requested

    cells: Sequence[LabelledCell] = []
    residues: Sequence[ZeroAreaResidue] = []
    candidate_count = skipped_count = residues_before_dedup = 0
    with _stage("geometry.grid.overlay", counts={"operands": len(values)}):
        for operand_index, value in enumerate(values):
            scope = OverlayScope(grid, operand_index)
            original = geometries[value.operand_id]
            operand = _overlay_parts(scope.call("intersection", original, original))
            previous = (*cells, *((r.geometry, r.operand_ids) for r in residues))
            tree = STRtree([cell for cell, _ in previous])
            candidates = {
                int(index) for part in operand.geometries for index in tree.query(part)
            }
            candidate_count += len(candidates)
            skipped_count += len(previous) - len(candidates)
            step = _SubdivisionStep(
                scope, value.operand_id, operand, cells, residues, [], []
            )
            step.divide(previous, candidates)
            residues_before_dedup += len(step.sideband)
            cells, residues = step.divided, deduplicate_residues(step.sideband)
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
        return Subdivision(tuple(cells), tuple(residues))


def residue_coordinates(geometry: BaseGeometry) -> tuple[GridPoint, ...]:
    """Collect coordinates of point and line support recursively."""
    if geometry.geom_type in {"LineString", "LinearRing", "Point"}:
        return tuple(tuple(p) for p in geometry.coords)
    collection = cast(
        "GeometryCollection | MultiLineString | MultiPoint | MultiPolygon", geometry
    )
    return tuple(p for child in collection.geoms for p in residue_coordinates(child))


def account_zero_area(
    residues: Sequence[ZeroAreaResidue],
    cells: Sequence[LabelledCell],
    grid: float | None,
) -> ZeroAreaAccounting:
    """Classify zero-area support against exact cell boundaries, retaining labels."""
    from shapely import LineString, STRtree  # noqa: PLC0415 - native GEOS loads only when zero-area support is classified

    if not residues:
        return ZeroAreaAccounting((), ())
    edges = {}
    for cell, _ in cells:
        for ring in (cell.exterior, *cell.interiors):
            points = tuple(tuple(p) for p in ring.coords)
            for a, b in pairwise(points):
                if a != b:
                    edges.setdefault(tuple(sorted((a, b))), LineString((a, b)))
    boundaries = tuple(edges.values())
    tree = STRtree(boundaries)
    geometries = {}
    labeled = []
    for residue in deduplicate_residues(residues):
        key = residue.exact_key[1]
        geometries.setdefault(key, residue.geometry)
        labeled.append((key, residue.operand_ids))
    scope = OverlayScope(grid, None)

    def classify(geometry: BaseGeometry) -> tuple[OverlayParts, OverlayParts]:
        indices = sorted(int(index) for index in tree.query(geometry))
        wire = scope.union(OverlayParts(tuple(boundaries[i] for i in indices)))
        parts = OverlayParts((geometry,))
        return scope.parts("intersection", parts, wire), scope.parts(
            "difference", parts, wire
        )

    classified = dict(
        zip(geometries, _ordered_map(classify, tuple(geometries.values())), strict=True)
    )
    incident, auxiliary = [], []
    for key, labels in labeled:
        common, remainder = classified[key]
        incident.extend(ZeroAreaResidue(p, labels) for p in common.geometries)
        auxiliary.extend(ZeroAreaResidue(p, labels) for p in remainder.geometries)
    return ZeroAreaAccounting(
        deduplicate_residues(incident), deduplicate_residues(auxiliary)
    )


@dataclass(frozen=True)
class GridAuditor:
    """Shared source, realization and cell state for the ordered preservation audits."""

    values: tuple[GridOperand, ...]
    geometries: Mapping[str, Polygon]
    effective: Mapping[str, BaseGeometry]
    cells: tuple[LabelledCell, ...]
    faces: tuple[GridFace, ...]
    grid: float | None

    def physical_audits(self) -> list[GridOperandAudit]:
        """Check existing physical topology rules in source operand order."""
        with _stage(
            "geometry.grid.physical_audit", counts={"operands": len(self.values)}
        ):
            audits = []
            for operand_index, value in enumerate(self.values):
                raw, final = (
                    self.geometries[value.operand_id],
                    self.effective[value.operand_id],
                )
                if value.physical and (
                    final.is_empty
                    or len(polygon_components(raw)) != len(polygon_components(final))
                    or sum(len(p.interiors) for p in polygon_components(raw))
                    != sum(len(p.interiors) for p in polygon_components(final))
                ):
                    raise self._topology_error(operand_index, value, raw, final)
                audits.append(
                    GridOperandAudit(
                        value.operand_id,
                        raw.area,
                        final.area,
                        not raw.equals(final),
                        sum(
                            len(r) - len(set(r)) for r in (value.exterior, *value.holes)
                        ),
                    )
                )
                _progress(operand_index + 1, len(self.values))
        return audits

    def audit_contacts(self) -> None:
        """Retain the original pairwise contact relation of physical operands."""
        contacts = [v for v in self.values if v.contact]
        with _stage("geometry.grid.contact_audit", counts={"operands": len(contacts)}):
            for i, first in enumerate(contacts):
                for second in contacts[i + 1 :]:
                    before = self.geometries[first.operand_id].relate(
                        self.geometries[second.operand_id]
                    )
                    after = self.effective[first.operand_id].relate(
                        self.effective[second.operand_id]
                    )
                    if before != after:
                        raise ValueError(
                            f"physical contact changed at grid: {first.operand_id}, {second.operand_id}"
                        )
                _progress(i + 1, len(contacts))

    def _topology_error(
        self,
        operand_index: int,
        value: GridOperand,
        raw: BaseGeometry,
        final: BaseGeometry,
    ) -> ValueError:
        from shapely import Polygon, union_all  # noqa: PLC0415 - defer native geometry loading until this operation requires it

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
                "grid_size_um": self.grid,
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
                        "intersection", raw, raw, self.grid, operand_index, None
                    ),
                ),
                (
                    "labelled_cells_without_resnap",
                    lambda value=value: union_all(
                        [
                            cell
                            for cell, labels in self.cells
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
                            for face in self.faces
                            if value.operand_id in face.operand_ids
                        ],
                        grid_size=0,
                    ),
                ),
            )
            for name, operation in stages:
                try:
                    metrics[name] = summary(operation())
                except Exception as diagnostic_error:  # noqa: BLE001 - an error-only witness cannot replace the original topology failure
                    metrics[name] = {
                        "unavailable": True,
                        "error_type": type(diagnostic_error).__name__,
                    }
            error.add_note(
                "scgsim.grid.physical_topology: " + json.dumps(metrics, sort_keys=True)
            )
        except Exception:  # noqa: BLE001 - diagnostic failure cannot replace the original topology error
            error.add_note("scgsim.grid.physical_topology: metrics unavailable")
        return error
