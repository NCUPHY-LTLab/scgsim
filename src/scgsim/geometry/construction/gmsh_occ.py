"""Own the existing native model lifecycle while lowering canonical OCC plans."""

from __future__ import annotations

import os
from pathlib import Path

from scgsim.geometry.construction.native_geometry.lowering import OccPlanLowerer
from scgsim.geometry.construction.native_geometry.timing import debug_stage
from scgsim.geometry.construction.native_types import GmshKernelApi, NativeTimingRecord
from scgsim.geometry.models.construction import ConstructionPlanRecord


def write_occ_geometry_from_plan(
    plan: ConstructionPlanRecord,
    *,
    xao_path: Path,
) -> ConstructionPlanRecord:
    """Write one XAO and return the plan with backend tags attached.

    The implementation attaches physical groups before writing `xao_path` and
    returns a copy of `plan` with `backend_entity_tags` populated. It refuses
    any volume that lacks planned `surface_refs`, because constructing that
    volume directly would bypass semantic surface ownership.
    """
    _validate_conformal_plan(plan)
    xao_path.parent.mkdir(parents=True, exist_ok=True)
    timings: list[NativeTimingRecord] = []
    debug_logging = False
    gmsh = None
    was_initialized = True
    try:
        with debug_stage(
            debug_logging,
            "gmsh import/initialize/clear/model setup",
            timings,
        ):
            import gmsh as gmsh_module  # noqa: PLC0415 - optional native dependency and original initialization failure boundary

            gmsh = gmsh_module
            was_initialized = bool(gmsh.isInitialized())
            if not was_initialized:
                gmsh.initialize(readConfigFiles=False)
            gmsh.clear()
            gmsh.option.setNumber("General.Terminal", 1 if debug_logging else 0)
            _apply_gmsh_number_option(
                gmsh,
                "Geometry.OCCAutoFix",
                "SGB_GMSH_OCC_AUTO_FIX",
                default=0.0,
            )
            gmsh.model.add(f"semantic_geometry_route_{plan.route.lower()}")
        return OccPlanLowerer(plan, gmsh, timings, debug_logging).write(xao_path)
    finally:
        if gmsh is not None and not was_initialized:
            with debug_stage(debug_logging, "finalize gmsh", timings):
                gmsh.finalize()


def _apply_gmsh_number_option(
    gmsh: GmshKernelApi,
    option_name: str,
    environment_name: str,
    *,
    default: float,
) -> None:
    value = os.environ.get(environment_name)
    gmsh.option.setNumber(option_name, default if value is None else float(value))


def _validate_conformal_plan(plan: ConstructionPlanRecord) -> None:
    """Reject any backend-live geometry that lacks canonical topology refs."""
    if any(not surface.construction_only for surface in plan.surfaces) and (
        not plan.points or not plan.curves or not plan.surface_loops
    ):
        raise NotImplementedError(
            "nonconformal OCC build refused: backend-live surfaces require "
            "planned PointPlan/CurvePlan/SurfaceLoop records before lowering"
        )
    missing_loop_refs = [
        surface.surface_id
        for surface in plan.surfaces
        if not surface.construction_only and surface.outer_loop_ref is None
    ]
    if missing_loop_refs:
        raise NotImplementedError(
            "nonconformal OCC build refused: backend-live surfaces require "
            f"outer_loop_ref before lowering {missing_loop_refs!r}"
        )
    unpartitioned_volumes = [
        volume.volume_id
        for volume in plan.volumes
        if not volume.construction_only and not volume.surface_refs
    ]
    if unpartitioned_volumes:
        raise NotImplementedError(
            "nonconformal OCC build refused: backend-live volumes must be "
            "assembled from planned surface_refs with addSurfaceLoop/addVolume "
            f"{unpartitioned_volumes!r}. Building these directly would create "
            "standalone surfaces instead of shared conformal topology."
        )
