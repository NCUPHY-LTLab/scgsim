"""Subdivide labelled GEOS operands and account for zero-area boundary residues."""

import json
from dataclasses import dataclass
from functools import cached_property
from itertools import pairwise

from scgsim.diagnostics import _progress, _stage
from scgsim.geometry._primitives.grid_regions import polygon_components
from scgsim.preparation import _ordered_map, current_preparation_workers


@dataclass(frozen=True)
class OverlayParts:
    """Nonempty native geometries kept separate for dimension-safe overlays."""

    geometries: tuple


@dataclass(frozen=True)
class ZeroAreaResidue:
    """A zero-area geometry with its exact operand membership."""

    geometry: object
    operand_ids: tuple

    @cached_property
    def exact_key(self):
        # Records survive disjoint cuts unchanged. Cache only their immutable
        # exact identity, never geometry from another arrangement or process.
        return self.operand_ids, self.geometry.normalize().wkb


@dataclass(frozen=True)
class ZeroAreaAccounting:
    """Zero-area support classified as incident boundary or auxiliary geometry."""

    boundary: tuple[ZeroAreaResidue, ...]
    auxiliary: tuple[ZeroAreaResidue, ...]


@dataclass(frozen=True)
class Subdivision:
    """Labelled area cells and their zero-area support and source provenance."""

    cells: tuple
    residues: tuple[ZeroAreaResidue, ...]
    accounting: ZeroAreaAccounting | None = None
    source_paths: tuple = ()
    edge_origins: tuple = ()


def _overlay_parts(geometry):
    # A result collection is not necessarily a legal input to OverlayNG.
    # Keep every nonempty dimension; homogeneous multi-geometries stay intact.
    if geometry.is_empty:
        return OverlayParts(())
    if geometry.geom_type == "GeometryCollection":
        return OverlayParts(
            tuple(
                part
                for child in geometry.geoms
                for part in _overlay_parts(child).geometries
            )
        )
    return OverlayParts((geometry,))


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


def overlay_call(operation, left, right, grid, operand_index, cell_index):
    """Run the native overlay and annotate failures without replacing the error."""
    from shapely import (
        difference,
        intersection,
        union_all,
    )  # optional geometry dependency

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
        except Exception:  # diagnostic failure cannot replace the native error
            try:
                error.add_note(
                    "scgsim.grid.overlay_failure: diagnostic context unavailable"
                )
            except Exception:  # retain the native error even if annotation fails
                pass  # logging could itself mask the native error
        raise


def _bounds_disjoint(left, right):
    ax0, ay0, ax1, ay1 = left.bounds
    bx0, by0, bx1, by1 = right.bounds
    return ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0


def deduplicate_residues(residues):
    """Retain one exact geometry record for each distinct semantic label set."""
    # Canonical exact coordinates include every retained collinear vertex.
    # Equal support with different labels remains a distinct semantic record.
    seen = set()
    output = []
    for residue in residues:
        key = residue.exact_key
        if key not in seen:
            seen.add(key)
            output.append(residue)
    return tuple(output)


def parts_overlay(operation, left, right, grid, operand_index, cell_index):
    """Overlay dimension-safe parts in the existing operand and cell order."""
    if operation == "intersection":
        return OverlayParts(
            tuple(
                part
                for a in left.geometries
                for b in right.geometries
                if not _bounds_disjoint(a, b)
                for part in _overlay_parts(
                    overlay_call(operation, a, b, grid, operand_index, cell_index)
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
                        overlay_call(operation, a, b, grid, operand_index, cell_index)
                    ).geometries
                )
        parts = tuple(next_parts)
    return OverlayParts(parts)


def _parts_union(parts, grid, operand_index):
    # Unary union is homogeneous too; do not reassemble a mixed input.
    output = []
    dimensions = tuple(dict.fromkeys(_geometry_dimension(p) for p in parts.geometries))
    for dimension in dimensions:
        group = [p for p in parts.geometries if _geometry_dimension(p) == dimension]
        output.extend(
            _overlay_parts(
                overlay_call("union_all", group, None, grid, operand_index, None)
            ).geometries
        )
    return OverlayParts(tuple(output))


@dataclass(frozen=True)
class _CellOverlayOutcome:
    common: object
    residual: object
    error: BaseException | None = None
    traceback: object = None


def _cell_overlay_pair(cell, operand, grid, operand_index, cell_index):
    parts = OverlayParts((cell,))
    try:
        common = parts_overlay(
            "intersection", parts, operand, grid, operand_index, cell_index
        )
        residual = parts_overlay(
            "difference", parts, operand, grid, operand_index, cell_index
        )
        return _CellOverlayOutcome(common, residual)
    except (
        BaseException
    ) as error:  # ordered worker failures must retain their original type
        # A later cell's failure must not overtake an earlier pending difference.
        return _CellOverlayOutcome(None, None, error, error.__traceback__)


def cell_overlay_chunk(work):
    """Evaluate immutable cell pairs in order for a preparation worker chunk."""
    rows, operand, grid, operand_index = work
    return tuple(
        (index, _cell_overlay_pair(cell, operand, grid, operand_index, index))
        for index, cell in rows
    )


def subdivide_grid(values, geometries, grid):
    """Incrementally subdivide operands while retaining every exact semantic label."""
    from shapely import STRtree  # optional geometry dependency

    cells, residues = [], []
    candidate_count = skipped_count = residues_before_dedup = 0
    with _stage("geometry.grid.overlay", counts={"operands": len(values)}):
        for operand_index, value in enumerate(values):
            original = geometries[value.operand_id]
            operand = _overlay_parts(
                overlay_call(
                    "intersection", original, original, grid, operand_index, None
                )
            )
            pending = operand
            divided, sideband = [], []

            def retain(parts, labels, divided=divided, sideband=sideband):
                for part in parts.geometries:
                    if _geometry_dimension(part) == 2:
                        divided.extend((p, labels) for p in polygon_components(part))
                    else:
                        sideband.append(ZeroAreaResidue(part, labels))

            # Area order is unchanged. Lower-dimensional cells are retained and
            # subdivided separately; they can never create positive-area labels.
            previous = (*cells, *((r.geometry, r.operand_ids) for r in residues))
            tree = STRtree([cell for cell, _ in previous])
            candidates = {
                int(index) for part in operand.geometries for index in tree.query(part)
            }
            candidate_count += len(candidates)
            skipped_count += len(previous) - len(candidates)
            workers = current_preparation_workers()
            pairs = {}
            if workers is not None and workers > 1 and candidates:
                rows = tuple(
                    (index, previous[index][0]) for index in sorted(candidates)
                )
                chunk_size = (len(rows) + workers - 1) // workers
                work = tuple(
                    (rows[start : start + chunk_size], operand, grid, operand_index)
                    for start in range(0, len(rows), chunk_size)
                )
                # Wait for all immutable pair work before doing serial pending
                # updates; caller CPU and pool never process overlay concurrently.
                pairs = dict(
                    row
                    for chunk in _ordered_map(cell_overlay_chunk, work)
                    for row in chunk
                )
            for cell_index, (cell, labels) in enumerate(previous):
                if cell_index not in candidates:
                    if cell_index < len(cells):
                        divided.append((cell, labels))
                    else:
                        # Preserve the immutable record and cached exact key;
                        # no dimension dispatch or GEOS normalization is needed.
                        sideband.append(residues[cell_index - len(cells)])
                    continue
                cell_parts = OverlayParts((cell,))
                pair = pairs.get(cell_index)
                if pair is None:
                    pair = _cell_overlay_pair(
                        cell, operand, grid, operand_index, cell_index
                    )
                if pair.error is not None:
                    raise pair.error.with_traceback(pair.traceback)
                common, residual = pair.common, pair.residual
                pending = parts_overlay(
                    "difference", pending, cell_parts, grid, operand_index, cell_index
                )
                retain(residual, labels)
                retain(common, (*labels, value.operand_id))
            retain(_parts_union(pending, grid, operand_index), (value.operand_id,))
            residues_before_dedup += len(sideband)
            cells, residues = divided, deduplicate_residues(sideband)
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


def residue_coordinates(geometry):
    """Collect coordinates of point and line support recursively."""
    if geometry.geom_type in {"LineString", "LinearRing", "Point"}:
        return tuple(tuple(p) for p in geometry.coords)
    return tuple(p for child in geometry.geoms for p in residue_coordinates(child))


def account_zero_area(residues, cells, grid):
    """Classify zero-area support against exact cell boundaries, retaining labels."""
    from shapely import LineString, STRtree  # optional geometry dependency

    if not residues:
        return ZeroAreaAccounting((), ())
    # Exact face edges give a tight broad-phase instead of overlaying each
    # residue against the full all-face wire. Touching envelopes are included.
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

    def classify(geometry):
        indices = sorted(int(index) for index in tree.query(geometry))
        wire = _parts_union(
            OverlayParts(tuple(boundaries[i] for i in indices)), grid, None
        )
        parts = OverlayParts((geometry,))
        common = parts_overlay("intersection", parts, wire, grid, None, None)
        remainder = parts_overlay("difference", parts, wire, grid, None, None)
        return common, remainder

    # Classifications are geometric only. Reattach every exact semantic label
    # in original order after ordered independent work; never merge labels.
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
