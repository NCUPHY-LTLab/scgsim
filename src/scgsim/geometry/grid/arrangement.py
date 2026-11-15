"""Build the labelled network of polygon operands on the model grid, or their exact overlay without one.

This is a planar arrangement, the same concept as ``geometry.sgb.arrangement``, built by a
different engine for a different stage: here GEOS nodes drawn polygons on the model grid,
for the source stage's projection and every grid Boolean; SGB's arrangement nodes the
prisms' curves, arcs included, in exact integer arithmetic for the build. A reader of the
source stage or of a Boolean wants this one; a reader of the solid model wants SGB's.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING

from scgsim.diagnostics import (
    diagnostic_stage,
    progress,
)
from scgsim.geometry.grid.grid_cache import cached_subdivision
from scgsim.geometry.grid.grid_overlay import (
    GridAuditor,
    Subdivision,
    ZeroAreaAccounting,
    account_zero_area,
    residue_coordinates,
    subdivide_grid,
)
from scgsim.geometry.grid.grid_regions import (
    GridArrangement,
    GridFace,
    GridOperand,
    region_vertex_index,
    regions_indexed,
)
from scgsim.geometry.grid.tagged_grid import build_tagged_subdivision
from scgsim.geometry.models.grid import GeometryGridContext
from scgsim.preparation import (
    ordered_map,
)
from scgsim.refusal import internal

if TYPE_CHECKING:
    from shapely.geometry import Polygon


@dataclass(frozen=True)
class _GridWork:
    raw_polygons: tuple[tuple[str, Polygon], ...]
    subdivision: Subdivision


def build_grid_arrangement(
    operands: Iterable[GridOperand] | Mapping[str, GridOperand],
    context: GeometryGridContext,
    *,
    apply_grid: bool = True,
) -> GridArrangement:
    """Construct one labelled fixed-grid network or an ungridded plane overlay.

    Fixed-grid membership follows directed source paths and face transitions.
    With ``apply_grid=False`` the overlay runs on the operands' own coordinates, not on the model grid.
    """
    from shapely import (  # noqa: PLC0415 - defer native geometry loading until this operation requires it
        Polygon,
    )

    grid = context.grid_size_um if apply_grid else None
    values = (
        tuple(operands.values()) if isinstance(operands, Mapping) else tuple(operands)
    )
    ids = [v.operand_id for v in values]
    if len(ids) != len(set(ids)):
        raise ValueError(
            internal(
                f"grid operand identities are duplicated: {sorted(set(i for i in ids if ids.count(i) > 1))}"
            )
        )

    def construct() -> _GridWork:
        return _construct_grid_work(values, grid, apply_grid, Polygon)

    work = cached_subdivision(values, context, apply_grid=apply_grid, build=construct)
    geometries = dict(work.raw_polygons)
    subdivision = work.subdivision
    cells = subdivision.cells
    with diagnostic_stage(
        "geometry.grid.zero_area", counts={"residues": len(subdivision.residues)}
    ):
        zero_area = subdivision.accounting
        if zero_area is None:
            zero_area = account_zero_area(subdivision.residues, cells, grid)
    vertices = _cell_vertices(cells, zero_area)
    faces, vertices = _atomic_grid_faces(cells, vertices, grid)
    # Boundary and auxiliary residues remain inspectable internally through
    # the exact physical audits; neither can create an orphan area-graph edge.
    effective = _effective_grid_operands(values, ids, faces, apply_grid)
    auditor = GridAuditor(values, geometries, effective, cells, faces, grid)
    audits = auditor.physical_audits()
    auditor.audit_contacts()
    return GridArrangement(
        context, faces, vertices, _face_edges(faces), tuple(audits), apply_grid
    )


def _cell_vertices(cells, zero_area: ZeroAreaAccounting) -> tuple:
    """Every vertex of the cells and of the zero-area residues, sorted."""
    residue_points = {
        point
        for residue in (*zero_area.boundary, *zero_area.auxiliary)
        for point in residue_coordinates(residue.geometry)
    }
    with diagnostic_stage("geometry.grid.vertices", counts={"cells": len(cells)}):
        return tuple(
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


def _face_edges(faces) -> tuple:
    """Every edge of the faces' rings once, as a sorted pair of its ends, sorted."""
    with diagnostic_stage("geometry.grid.edges", counts={"faces": len(faces)}):
        return tuple(
            sorted(
                {
                    tuple(sorted((a, b)))
                    for f in faces
                    for ring in (f.exterior, *f.holes)
                    for a, b in zip(ring, (*ring[1:], ring[0]), strict=False)
                }
            )
        )


def _atomic_grid_faces(cells, vertices, grid):
    with diagnostic_stage(
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
            progress(cell_index + 1, len(cells))
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
    return faces, vertices


def _effective_grid_operands(values, ids, faces, apply_grid):
    from shapely import (  # noqa: PLC0415 - defer native geometry loading until this operation requires it
        Polygon,
        coverage_union_all,
        union_all,
    )

    with diagnostic_stage("geometry.grid.effective", counts={"operands": len(values)}):
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

        effective = dict(zip(ids, ordered_map(realize, values), strict=True))
    return effective


def _construct_grid_work(
    values: tuple[GridOperand, ...],
    grid: float | None,
    apply_grid: bool,
    polygon_type: type[Polygon],
) -> _GridWork:
    with diagnostic_stage("geometry.grid.inputs", counts={"operands": len(values)}):
        raw_polygons = tuple(
            (v.operand_id, polygon_type(v.exterior, v.holes)) for v in values
        )
    geometries = dict(raw_polygons)
    if not apply_grid:
        return _GridWork(raw_polygons, subdivide_grid(values, geometries, None))

    result = build_tagged_subdivision(values, geometries, grid)
    accounting = ZeroAreaAccounting(tuple(result.boundary), tuple(result.auxiliary))
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
