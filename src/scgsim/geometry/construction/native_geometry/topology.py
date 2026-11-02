"""Project canonical loops, surface directions and backend tag inventories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from scgsim.geometry.construction.native_types import GmshModelApi
from scgsim.geometry.models.common import GmshDimTag, TagSourceKindLiteral
from scgsim.geometry.models.construction import ConstructionPlanRecord
from scgsim.geometry.models.tags import BackendEntityTagRecord, TagPlanRecord
from scgsim.geometry.models.topology import (
    SurfaceLoopRecord,
    SurfacePlanRecord,
    SurfaceRefRecord,
)


def add_curve_loop_from_plan(
    gmsh: GmshModelApi,
    loop: SurfaceLoopRecord,
    curve_tags: Mapping[str, int],
    *,
    reverse: bool = False,
) -> int:
    """Lower the exact signed canonical edges and preserve loop failure context."""
    signed_curve_tags = [
        curve_tags[curve_ref.curve_id] * curve_ref.orientation * (-1 if reverse else 1)
        for curve_ref in loop.curve_refs
    ]
    try:
        return gmsh.model.occ.addCurveLoop(signed_curve_tags)
    except Exception as exc:
        raise ValueError(
            f"{loop.loop_id} could not be lowered to an OCC curve loop"
        ) from exc


def chosen_surface_orientations(plan: ConstructionPlanRecord) -> dict[str, str]:
    """Choose one planned direction per OCC surface before it is created."""
    uses: dict[str, set[str]] = {}
    for volume in plan.volumes:
        exterior_refs = (
            getattr(volume, "exterior_surface_refs", ()) or volume.surface_refs
        )
        for ref in exterior_refs:
            uses.setdefault(ref.surface_id, set()).add(ref.orientation)
        for void in getattr(volume, "inner_pec_void_shells", ()):
            for ref in void.surface_refs:
                uses.setdefault(ref.surface_id, set()).add(ref.orientation)
    for orientations in uses.values():
        if len(orientations) > 2:
            raise ValueError("surface has ambiguous planned volume orientations")
    # A shared interface may be opposite in two volumes; its plane is created
    # once and OCC sewing supplies the shell-local sense. Prefer forward
    # deterministically, otherwise lower its single reversed use.
    return {
        surface_id: "forward" if "forward" in orientations else "reversed"
        for surface_id, orientations in uses.items()
    }


def surface_edge_count(
    surface: SurfacePlanRecord,
    surface_loops: Sequence[SurfaceLoopRecord],
) -> int:
    """Count planned outer and hole edges for existing native diagnostics."""
    loops_by_id = {loop.loop_id: loop for loop in surface_loops}
    loop_ids = (
        *((surface.outer_loop_ref,) if surface.outer_loop_ref is not None else ()),
        *surface.hole_loop_refs,
    )
    return sum(len(loops_by_id[loop_id].curve_refs) for loop_id in loop_ids)


def surface_ref_tags(
    owner_id: str,
    surface_refs: Sequence[SurfaceRefRecord],
    source_tags: Mapping[tuple[str, str], Sequence[GmshDimTag]],
) -> list[int]:
    """Return OCC surface tags for `addSurfaceLoop()`.

    OCC's Python `addSurfaceLoop()` rejects negative tags.  Orientation is
    retained in the plan/audit ledger; lowering relies on each planned surface
    loop's canonical ring direction.
    """
    tags = [
        tag
        for surface_ref in surface_refs
        for dim, tag in source_tags.get(("surface", surface_ref.surface_id), ())
        if dim == 2
    ]
    if len(tags) != len(surface_refs):
        missing = [
            surface_ref.surface_id
            for surface_ref in surface_refs
            if not source_tags.get(("surface", surface_ref.surface_id), ())
        ]
        raise ValueError(f"{owner_id} missing surface tags: {missing!r}")
    return tags


def backend_entity_tags(
    source_tags: Mapping[tuple[str, str], Sequence[GmshDimTag]],
) -> tuple[BackendEntityTagRecord, ...]:
    """Detach native tags in the established source-insertion order."""
    return tuple(
        BackendEntityTagRecord(
            source_record_kind=cast(TagSourceKindLiteral, source_kind),
            source_record_id=source_id,
            dim_tag=dim_tag,
        )
        for (source_kind, source_id), dim_tags in source_tags.items()
        for dim_tag in dim_tags
    )


def physical_entity_tags(
    tag_plans: tuple[TagPlanRecord, ...],
    source_tags: Mapping[tuple[str, str], Sequence[GmshDimTag]],
) -> list[int]:
    """Deduplicate dimension-matching physical members without reordering them."""
    tags: list[int] = []
    seen: set[int] = set()
    for tag_plan in tag_plans:
        for dimension, tag in source_tags.get(
            (tag_plan.source_record_kind, tag_plan.source_record_id),
            (),
        ):
            if dimension != tag_plan.dimension or tag in seen:
                continue
            tags.append(tag)
            seen.add(tag)
    return tags
