"""Compile exact source boundaries while preserving the native model lifecycle."""

from __future__ import annotations

from scgsim.geometry.construction.curved_geometry.arrangement import PlanarArrangement
from scgsim.geometry.construction.curved_geometry.boundaries import (
    source_boundaries,
    source_rings,
)
from scgsim.geometry.construction.curved_geometry.kernel import CurveKernel
from scgsim.geometry.construction.curved_geometry.records import CanonicalRecords
from scgsim.geometry.construction.curved_geometry.surfaces import SharedSurfaces
from scgsim.geometry.models.common import RouteLiteral
from scgsim.geometry.models.construction import ConstructionPlanRecord
from scgsim.geometry.models.input import GeometryBuildInput
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.planning.domain import prepare_auto_vacuum_solution_regions
from scgsim.geometry.planning.representation import validate_selected_route


def build_curved_construction_plan(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    representation_context: RepresentationContext | None = None,
) -> ConstructionPlanRecord:
    """Compile source curves globally before naming shared material surfaces."""
    # Native Gmsh is optional; importing the planning facade must not load it.
    import gmsh  # noqa: PLC0415 - optional native geometry dependency

    prepared = prepare_auto_vacuum_solution_regions(
        build_input,
        route=route,
        native_complement=True,
        representation_context=representation_context,
    )
    validate_selected_route(
        prepared, route, representation_context=representation_context
    )
    rings = source_rings(prepared)
    chains, operations = source_boundaries(prepared, rings)
    planar = PlanarArrangement(prepared, route, rings, chains)
    initialized = bool(gmsh.isInitialized())
    if not initialized:
        gmsh.initialize(readConfigFiles=False)
    previous = gmsh.model.getCurrent()
    gmsh.model.add("scgsim-curved-arrangement")
    terminal = gmsh.option.getNumber("General.Terminal")
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        kernel = CurveKernel(gmsh)
        planar.build(kernel)
        geometry = SharedSurfaces(planar).build()
        return CanonicalRecords(geometry, operations).build()
    finally:
        gmsh.model.remove()
        if previous:
            gmsh.model.setCurrent(previous)
        gmsh.option.setNumber("General.Terminal", terminal)
        if not initialized:
            gmsh.finalize()
