"""Coordinate scoped grid construction, polygon Booleans and source-preservation audits."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import wraps
from hashlib import sha256
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Concatenate,
    Literal,
    ParamSpec,
    TypedDict,
    TypeVar,
    cast,
)

from scgsim.diagnostics import (
    diagnostic_stage,
    progress,
)
from scgsim.geometry.models.grid import (
    GeometryGridContext,
    format_box,
    geometry_grid_context,
)
from scgsim.geometry.primitives.constants import (
    TOPOLOGY_EPS_UM,
)
from scgsim.geometry.primitives.grid_cache import (
    cached_subdivision,
)
from scgsim.geometry.primitives.grid_overlay import (
    GridAuditor,
    Subdivision,
    ZeroAreaAccounting,
    ZeroAreaResidue,
    account_zero_area,
    residue_coordinates,
    subdivide_grid,
)
from scgsim.geometry.primitives.grid_overlay import (
    OverlayParts as OverlayParts,
)
from scgsim.geometry.primitives.grid_overlay import (
    cell_overlay_chunk as cell_overlay_chunk,
)
from scgsim.geometry.primitives.grid_overlay import (
    deduplicate_residues as deduplicate_residues,
)
from scgsim.geometry.primitives.grid_overlay import (
    overlay_call as overlay_call,
)
from scgsim.geometry.primitives.grid_overlay import (
    parts_overlay as parts_overlay,
)
from scgsim.geometry.primitives.grid_regions import (
    FIX_SHAPE,
    GridArrangement,
    GridFace,
    GridOperand,
    GridPoint,
    GridRegion,
    GridRing,
    PolygonRegion,
    grid_change_message,
    polygon_components,
    region_vertex_index,
    regions,
    regions_indexed,
    topology_changes,
)
from scgsim.geometry.primitives.grid_regions import (
    GridOperandAudit as GridOperandAudit,
)
from scgsim.geometry.primitives.grid_regions import (
    region_key as region_key,
)
from scgsim.geometry.primitives.planar_loops import (
    LoopCoordinates,
    PlanarLoop,
    PolygonCoordinates,
    boundary_meeting_points,
    clean_loop,
    cutline_boundary_edges,
    cutline_boundary_loops,
    filter_gdstk_polygons,
    polygon_area,
    repeated_boundary_edges,
)
from scgsim.geometry.primitives.tagged_grid import (
    build_tagged_subdivision,
)
from scgsim.preparation import (
    ordered_map,
)

if TYPE_CHECKING:
    from types import ModuleType

    from gdstk import Polygon as GdstkPolygon
    from shapely.geometry import Polygon

    from scgsim.geometry.models.input import (
        GeometryBuildInput,
        LayoutPolygonSpec,
        SemanticEntitySpec,
    )


class SourcePolygonAudit(TypedDict):
    """One requested source polygon in assembled coordinates."""

    source_polygon_id: str
    exterior: GridRing
    holes: tuple[GridRing, ...]


class SourceGridAudit(TypedDict):
    """Existing source preservation ledger stored within one active grid scope."""

    source_entity_id: str
    operation: str
    geometry_grid: Mapping[str, object]
    requested_regions: tuple[GridRegion, ...]
    requested_polygons: list[SourcePolygonAudit]
    requested_geometry_sha256: str
    requested_area_um2: float
    realized_area_um2: float
    requested_components: int
    realized_components: int
    requested_holes: int
    realized_holes: int
    geometry_changed: bool


_P = ParamSpec("_P")
_T = TypeVar("_T")

_ACTIVE_GRID: ContextVar[GeometryGridContext | None] = ContextVar(
    "scgsim_geometry_grid", default=None
)
SOURCE_INTENT: ContextVar[bool] = ContextVar(
    "scgsim_source_intent_overlay", default=False
)
SOURCE_AUDITS: ContextVar[list[SourceGridAudit] | None] = ContextVar(
    "scgsim_source_grid_audits", default=None
)
# The declared object and source layer a recipe is reading, as (object id, layer); named only
# by refusals, so a polygon converted deep inside a Boolean still says whose it is.
_NAMED_SOURCE: ContextVar[tuple[str, str] | None] = ContextVar(
    "scgsim_named_source", default=None
)


@contextmanager
def named_source(object_id: str, layer: str):
    """Name the declared object and source layer that refusals raised inside report."""
    token = _NAMED_SOURCE.set((object_id, layer))
    try:
        yield
    finally:
        _NAMED_SOURCE.reset(token)


def current_grid_context() -> GeometryGridContext | None:
    """Return the explicit grid context bound to this invocation."""
    return _ACTIVE_GRID.get()


def with_input_grid(
    function: Callable[Concatenate[GeometryBuildInput, _P], _T],
) -> Callable[Concatenate[GeometryBuildInput, _P], _T]:
    """Scope existing leaf operations to the explicit input policy only."""

    @wraps(function)
    def scoped(
        build_input: GeometryBuildInput, *args: _P.args, **kwargs: _P.kwargs
    ) -> _T:
        with geometry_grid_scope(geometry_grid_context(build_input)):
            return function(build_input, *args, **kwargs)

    return scoped


def current_grid_size_um() -> float:
    """Return the active grid step or the historical unbound precision."""
    context = _ACTIVE_GRID.get()
    # Retained only for historical inputs that have no declared grid context.
    return 1e-9 if context is None else context.grid_size_um


@contextmanager
def geometry_grid_scope(context: GeometryGridContext | None):
    """Bind grid policy and a scope-local source audit ledger, restoring both."""
    same = context is not None and context == _ACTIVE_GRID.get()
    token = _ACTIVE_GRID.set(context)
    audit_token = (
        None if same else SOURCE_AUDITS.set([] if context is not None else None)
    )
    try:
        yield
    finally:
        if audit_token is not None:
            SOURCE_AUDITS.reset(audit_token)
        _ACTIVE_GRID.reset(token)


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
    Ungridded parent planes retain their distinct existing overlay contract.
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
        raise ValueError("common grid operand identities are duplicated")

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
    residue_points = {
        point
        for residue in (*zero_area.boundary, *zero_area.auxiliary)
        for point in residue_coordinates(residue.geometry)
    }
    with diagnostic_stage("geometry.grid.vertices", counts={"cells": len(cells)}):
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
    faces, vertices = _atomic_grid_faces(cells, vertices, grid)
    # Boundary and auxiliary residues remain inspectable internally through
    # the exact physical audits; neither can create an orphan area-graph edge.
    effective = _effective_grid_operands(values, ids, faces, apply_grid)
    auditor = GridAuditor(values, geometries, effective, cells, faces, grid)
    audits = auditor.physical_audits()
    auditor.audit_contacts()
    with diagnostic_stage("geometry.grid.edges", counts={"faces": len(faces)}):
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


def polygon_rings(polygon: PolygonCoordinates) -> GridRegion:
    """Read stored regions or split the unchanged float-converted native walk."""
    if isinstance(polygon, PolygonRegion):
        return polygon.exterior, polygon.holes

    # Gdstk supplies XY pairs; this annotation records that native boundary
    # while leaving conversion, ordering and errors exactly as before.
    points = cast("PlanarLoop", tuple(tuple(map(float, p)) for p in polygon.points))
    return split_gdstk_cutline_loop(points)


_POLYGON_FRAME: ContextVar[tuple[float, ...] | None] = ContextVar(
    "scgsim_polygon_world_frame", default=None
)


@contextmanager
def polygon_world_frame(transform: Sequence[float]):
    """Bind the supplied affine occurrence transform for nested polygon work."""
    token = _POLYGON_FRAME.set(tuple(transform))
    try:
        yield
    finally:
        _POLYGON_FRAME.reset(token)


def in_occurrence_frame(function: Callable[..., _T]) -> Callable[..., _T]:
    """Preserve the callable while binding its required occurrence transform."""

    @wraps(function)
    def framed(*args: object, transform: Sequence[float], **kwargs: object) -> _T:
        with polygon_world_frame(transform):
            return function(*args, transform=transform, **kwargs)

    return framed


def frame_point(point: Sequence[float], inverse: bool = False) -> GridPoint:
    """Transform a point through the active polygon frame or its inverse."""
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


def polygon_boolean(
    left: Sequence[PolygonCoordinates],
    right: Sequence[PolygonCoordinates],
    operation: str,
    *,
    precision: float = 1e-9,
) -> Sequence[PolygonCoordinates]:
    """One GEOS labelled overlay for bound data; historical gdstk unchanged."""
    context = current_grid_context()
    if context is None:
        import gdstk  # noqa: PLC0415 - defer native geometry loading until this operation requires it

        # Unbound operands follow the native gdstk contract. These annotations
        # preserve direct native dispatch and its own malformed-input errors.
        return gdstk.boolean(
            cast("Sequence[GdstkPolygon]", left),
            cast("Sequence[GdstkPolygon]", right),
            cast("Literal['or', 'and', 'xor', 'not']", operation),
            precision=precision,
        )
    values = []
    for label, polygons in (("L", left), ("R", right)):
        for index, polygon in enumerate(polygons):
            outer, holes = polygon_rings(polygon)
            values.append(
                GridOperand(
                    f"{label}{index}",
                    tuple(frame_point(p) for p in outer),
                    tuple(tuple(frame_point(p) for p in ring) for ring in holes),
                    physical=False,
                    contact=False,
                )
            )
    arrangement = build_grid_arrangement(
        values, context, apply_grid=not SOURCE_INTENT.get()
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
            tuple(frame_point(p, True) for p in outer),
            tuple(tuple(frame_point(p, True) for p in ring) for ring in holes),
        )
        for outer, holes in arrangement.face_regions(selected)
    ]


def source_grid_audits() -> tuple[SourceGridAudit, ...]:
    """Read the current scope ledger without changing its ordering or records."""
    return tuple(SOURCE_AUDITS.get() or ())


@dataclass(frozen=True)
class _SourceRecipeSuccess[_Result]:
    result: _Result
    audits: tuple[SourceGridAudit, ...]
    error: None = None
    traceback: None = None


@dataclass(frozen=True)
class _SourceRecipeFailure:
    result: None
    audits: tuple[SourceGridAudit, ...]
    error: BaseException
    traceback: TracebackType | None


def source_recipe_map[_Item, _Result](
    function: Callable[[_Item], _Result], items: Iterable[_Item]
) -> list[_Result]:
    """Run independent recipes with isolated audits and source-order merging."""
    parent_audits = SOURCE_AUDITS.get()

    def execute(item: _Item) -> _SourceRecipeSuccess[_Result] | _SourceRecipeFailure:
        local_audits: list[SourceGridAudit] | None = (
            [] if parent_audits is not None else None
        )
        token = SOURCE_AUDITS.set(local_audits)
        try:
            try:
                result = function(item)
            except BaseException as error:  # noqa: BLE001 - preserve original worker failures and source-order rethrow
                return _SourceRecipeFailure(
                    None, tuple(local_audits or ()), error, error.__traceback__
                )
            return _SourceRecipeSuccess(result, tuple(local_audits or ()))
        finally:
            SOURCE_AUDITS.reset(token)

    outcomes = ordered_map(execute, items)
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


def preserve_declared_source(
    function: Callable[
        Concatenate[SemanticEntitySpec, _P], tuple[LayoutPolygonSpec, ...]
    ],
) -> Callable[Concatenate[SemanticEntitySpec, _P], tuple[LayoutPolygonSpec, ...]]:
    """Compare declared physical source result before destructive quantization.

    The reference evaluates the SAME declared mask/include/selector recipe in
    unquantized coordinates solely for its preservation audit. It is never
    substituted for the actual shared-grid geometry.
    """

    @wraps(function)
    def preserved(
        entity: SemanticEntitySpec, *args: _P.args, **kwargs: _P.kwargs
    ) -> tuple[LayoutPolygonSpec, ...]:
        context = current_grid_context()
        if context is None or SOURCE_INTENT.get():
            return function(entity, *args, **kwargs)
        token = SOURCE_INTENT.set(True)
        try:
            with diagnostic_stage("geometry.grid.source_reference"):
                requested = function(entity, *args, **kwargs)
        finally:
            SOURCE_INTENT.reset(token)
        with diagnostic_stage("geometry.grid.source_realization"):
            realized = function(entity, *args, **kwargs)
        return _audit_declared_source(
            entity, function.__name__, context, requested, realized
        )

    return preserved


def boolean_polygon_region(
    gdstk: ModuleType,
    left: Sequence[PolygonCoordinates],
    right: Sequence[PolygonCoordinates],
    operation: str,
) -> tuple[PolygonCoordinates, ...]:
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


def loop_inside_loop(loop: LoopCoordinates, container: LoopCoordinates) -> bool:
    """Test containment through the active grid-aware intersection area path."""
    import gdstk  # noqa: PLC0415 - defer native geometry loading until this operation requires it

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
    return loop_area - intersection_area <= max(TOPOLOGY_EPS_UM, loop_area * 1e-9)


def split_gdstk_cutline_loop(
    loop: tuple[tuple[float, float], ...],
) -> tuple[
    tuple[tuple[float, float], ...], tuple[tuple[tuple[float, float], ...], ...]
]:
    """Rebuild cutline boundaries and verify holes under the active grid scope."""
    try:
        outer, holes = cutline_boundary_loops(loop)
    except ValueError as error:
        # Every refusal of the walk is about the drawn boundary: keep its lead phrase
        # and locate it on the same walk, in the assembled layout frame.
        raise ValueError(_boundary_walk_refusal(str(error), loop)) from error
    stray = next((hole for hole in holes if not loop_inside_loop(hole, outer)), None)
    if stray is not None:
        raise ValueError(
            source_refusal(
                "gdstk cutline polygon has a non-interior boundary loop",
                [
                    f"its boundary loop at {source_place(stray)} lies outside its outer loop"
                ],
                _FIX_DISJOINT,
            )
        )
    return outer, holes


FIX_EMPTY = "Fix the layout: draw the polygon with a positive area, or remove it."
_FIX_MEETING = (
    "Fix the layout: where two parts of one region meet only at a point, "
    "give them a shared edge or separate them."
)
_FIX_RETRACED = "Fix the layout: draw the outline so that it runs along each edge once."
_FIX_DISJOINT = (
    "Fix the layout: draw separate parts as separate polygons, "
    "or join them by a shared edge."
)


def _boundary_walk_refusal(lead: str, loop: PlanarLoop) -> str:
    """The refusal of a drawn boundary that is not simple loops, at the edges or points that break it."""
    edges = cutline_boundary_edges(loop)
    if retraced := repeated_boundary_edges(edges):
        findings = [
            f"its boundary runs twice along {source_place(e)}" for e in retraced
        ]
        return source_refusal(lead, findings, _FIX_RETRACED)
    if meeting := boundary_meeting_points(edges):
        findings = [
            f"its boundary meets itself at {source_place((p,))}" for p in meeting
        ]
        return source_refusal(lead, findings, _FIX_MEETING)
    return source_refusal(
        lead, [f"its boundary at {source_place(loop)} encloses no area"], FIX_EMPTY
    )


def source_refusal(lead: str, findings: list[str], action: str) -> str:
    """A drawn-polygon refusal naming the source being read (:func:`named_source`), where and who acts."""
    named = _NAMED_SOURCE.get()
    context = current_grid_context()
    return grid_change_message(
        lead if named is None else f"{lead}: {named[0]}",
        [] if named is None else [named[1]],
        None if context is None else context.grid_size_um,
        findings,
        action,
    )


def source_place(points: Sequence[Sequence[float]]) -> str:
    """Where points of the polygon frame lie in the assembled layout, in micrometres."""
    world = [frame_point(point) for point in points]
    context = current_grid_context()
    return format_box(
        (
            min(x for x, _ in world),
            min(y for _, y in world),
            max(x for x, _ in world),
            max(y for _, y in world),
        ),
        None if context is None else context.grid_size_um,
    )


_regions_indexed = regions_indexed


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


def _audit_declared_source(
    entity: SemanticEntitySpec,
    operation: str,
    context: GeometryGridContext,
    requested: Sequence[LayoutPolygonSpec],
    realized: Sequence[LayoutPolygonSpec],
) -> tuple[LayoutPolygonSpec, ...]:
    from shapely import Polygon, union_all  # optional geometry dependency  # noqa: PLC0415 - optional geometry dependency

    def geometry(polygons):
        return union_all(
            [
                Polygon(
                    tuple(frame_point(p) for p in v.exterior),
                    tuple(tuple(frame_point(p) for p in r) for r in v.holes),
                )
                for v in polygons
            ]
        )

    with diagnostic_stage("geometry.grid.source_audit"):
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
                grid_change_message(
                    "declared physical source component/hole topology changed at grid: "
                    f"{entity.semantic_id}",
                    [p.layer for p in requested],
                    context.grid_size_um,
                    topology_changes(raw, final, context.grid_size_um),
                    FIX_SHAPE,
                )
            )
        raw_regions = regions(raw)
        requested_by_id = {p.polygon_id: p for p in requested}
        if set(requested_by_id) != {p.polygon_id for p in realized}:
            raise ValueError(
                f"declared source polygon identity changed at grid: {entity.semantic_id}"
            )
        requested_polygons: list[SourcePolygonAudit] = [
            {
                "source_polygon_id": p.polygon_id,
                "exterior": tuple(frame_point(v) for v in p.exterior),
                "holes": tuple(tuple(frame_point(v) for v in r) for r in p.holes),
            }
            for p in requested
        ]
        ledger: SourceGridAudit = {
            "source_entity_id": entity.semantic_id,
            "operation": operation,
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
        audits = SOURCE_AUDITS.get()
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
