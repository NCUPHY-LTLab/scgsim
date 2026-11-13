"""Source recipes: the jobs that read one Entity's or occurrence's polygons, run in source order, and the audit that gridding kept each declared source.

A recipe returns ``Audited``: its value and the preservation audits of the declared
sources it read. ``source_recipe_map`` runs recipes, possibly on worker threads, and
merges their values and audits in source order; the source stage hands the merged
audits to the grid projection (``grid_projection.apply_geometry_grid``).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from hashlib import sha256
from typing import Any, TypedDict

from scgsim.diagnostics import diagnostic_stage
from scgsim.geometry.grid.grid_regions import (
    FIX_SHAPE,
    GridRegion,
    GridRing,
    grid_change_message,
    polygon_components,
    regions,
    topology_changes,
)
from scgsim.geometry.grid.scope import GridScope
from scgsim.geometry.models.input import LayoutPolygonSpec, SemanticEntitySpec
from scgsim.preparation import ordered_map


class SourcePolygonAudit(TypedDict):
    """One requested source polygon in assembled coordinates."""

    source_polygon_id: str
    exterior: GridRing
    holes: tuple[GridRing, ...]


class SourceGridAudit(TypedDict):
    """What one declared source recipe requested and what the grid realized of it."""

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


@dataclass(frozen=True)
class Audited[_Value]:
    """A recipe's value and the audits of the declared sources it read, in reading order."""

    value: _Value
    audits: tuple[SourceGridAudit, ...] = ()


def merged_audits(*groups: Iterable[SourceGridAudit]) -> tuple[SourceGridAudit, ...]:
    """Audits of several recipes in order, each distinct audit once."""
    merged: list[SourceGridAudit] = []
    for group in groups:
        for audit in group:
            if audit not in merged:
                merged.append(audit)
    return tuple(merged)


DeclaredRecipe = Callable[..., tuple[LayoutPolygonSpec, ...]]


def audited_declared_source(
    recipe: DeclaredRecipe,
    scope: GridScope,
    entity: SemanticEntitySpec,
    **inputs: Any,
) -> Audited[tuple[LayoutPolygonSpec, ...]]:
    """Run a declared physical source recipe as authored and on the grid, and audit that the grid kept it.

    The reference evaluates the SAME declared mask/include/selector recipe in
    unquantized coordinates solely for its preservation audit. It is never
    substituted for the actual shared-grid geometry.
    """
    with diagnostic_stage("geometry.grid.source_reference"):
        requested = recipe(scope.as_reference(), entity, **inputs)
    with diagnostic_stage("geometry.grid.source_realization"):
        realized = recipe(scope, entity, **inputs)
    with diagnostic_stage("geometry.grid.source_audit"):
        audit = _declared_source_audit(
            scope, entity, recipe.__name__, requested, realized
        )
    return Audited(
        tuple(
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
        ),
        (audit,),
    )


def _declared_source_audit(
    scope: GridScope,
    entity: SemanticEntitySpec,
    operation: str,
    requested: Sequence[LayoutPolygonSpec],
    realized: Sequence[LayoutPolygonSpec],
) -> SourceGridAudit:
    from shapely import Polygon, union_all  # noqa: PLC0415 - optional geometry dependency

    context = scope.bound_grid()

    def geometry(polygons):
        return union_all(
            [
                Polygon(
                    tuple(scope.to_layout(p) for p in v.exterior),
                    tuple(tuple(scope.to_layout(p) for p in r) for r in v.holes),
                )
                for v in polygons
            ]
        )

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
    if {p.polygon_id for p in requested} != {p.polygon_id for p in realized}:
        raise ValueError(
            f"declared source polygon identity changed at grid: {entity.semantic_id}"
        )
    requested_polygons: list[SourcePolygonAudit] = [
        {
            "source_polygon_id": p.polygon_id,
            "exterior": tuple(scope.to_layout(v) for v in p.exterior),
            "holes": tuple(tuple(scope.to_layout(v) for v in r) for r in p.holes),
        }
        for p in requested
    ]
    return {
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


@dataclass(frozen=True)
class SourceRecipeJob:
    """Carry one source recipe or its deferred preparation failure."""

    function: Any = None
    args: tuple = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    error: Exception | None = None
    traceback: Any = None


def source_recipe_job(
    function,
    *args,
    source_polygons=None,
    **kwargs,
):
    """Prepare worker-owned inputs on the source thread, before pool dispatch.

    ``source_polygons`` (the drawn polygons by GDS layer) are copied into the job's
    ``polygons_by_layer``, so a worker never shares a native polygon with another.
    """
    copied = deepcopy(kwargs)
    if source_polygons is not None:
        copied["polygons_by_layer"] = {
            layer: tuple(polygon.copy() for polygon in polygons)
            for layer, polygons in source_polygons.items()
        }
    return SourceRecipeJob(function, deepcopy(args), copied)


def failed_source_recipe(error):
    # Preparation errors join the same source-order failure stream as recipes.
    """Capture a preparation failure for ordered source-recipe execution."""
    return SourceRecipeJob(error=error, traceback=error.__traceback__)


def run_source_recipe(job):
    """Execute one source recipe and preserve its ordered failure behavior."""
    if job.error is not None:
        raise job.error.with_traceback(job.traceback)
    return job.function(*job.args, **job.kwargs)


def source_recipe_map[_Item, _Result](
    function: Callable[[_Item], Audited[_Result]], items: Iterable[_Item]
) -> Audited[list[_Result]]:
    """Run independent recipes, possibly in parallel, and merge values and audits in source order.

    The first recipe in source order that fails raises its own error.
    """
    outcomes = ordered_map(function, items)
    return Audited(
        [outcome.value for outcome in outcomes],
        merged_audits(*(outcome.audits for outcome in outcomes)),
    )
