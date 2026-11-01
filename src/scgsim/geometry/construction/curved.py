"""Compile curved source geometry into a native construction plan.

Entry point for the curved (arc and spline) path: `build_curved_construction_plan`
runs the stages in a fixed order inside one temporary gmsh model, then validates
and returns the plan. Each stage lives in its own module:

- `curved_source`: authored rings and boundary curves (no gmsh)
- `curved_kernel`: native edges and footprint faces
- `curved_arrangement`: fragment footprints into shared planar cells
- `curved_ownership`: slab ownership and volume grouping (no gmsh)
- `curved_sweep`: sweep the curve complex along Z
- `curved_surfaces`: caps, walls and contacts as plan surfaces
- `curved_records`: curve, loop, surface and contact records
- `curved_volumes`: volumes and shell orientation

The stages create OCC and gmsh entities in a fixed order, and native tags, sort
order and set iteration feed identifiers, so stages must not be reordered even
where they look independent. Does not own the public compiler facade.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from scgsim.geometry.construction.curved_arrangement import build_planar_arrangement
from scgsim.geometry.construction.curved_context import BuildSession, CurvedContext
from scgsim.geometry.construction.curved_kernel import Kernel
from scgsim.geometry.construction.curved_ownership import build_slab_ownership
from scgsim.geometry.construction.curved_records import (
    apply_surface_contract,
    bind_port_conductor_components,
    bind_port_terminals,
    build_curve_table,
    build_mm_contacts,
    build_surface_table,
    finalize_curves_and_points,
)
from scgsim.geometry.construction.curved_source import boundaries, source_rings
from scgsim.geometry.construction.curved_surfaces import register_surfaces
from scgsim.geometry.construction.curved_sweep import sweep_curve_complex
from scgsim.geometry.construction.curved_volumes import build_volumes
from scgsim.geometry.models.construction import (
    ConstructionPlanRecord,
    RouteABConstructionPlanRecord,
)
from scgsim.geometry.planning.context.tags import plan_route_tags
from scgsim.geometry.planning.solutions.auto_vacuum import (
    prepare_auto_vacuum_solution_regions,
)
from scgsim.geometry.planning.validation import (
    validate_curve_plan_coverage,
    validate_selected_route,
    validate_tag_plan_coverage,
    validate_volume_surface_closure,
)


@dataclass(frozen=True)
class _PlanRecords:
    """Records gathered by the stages, ready to be assembled into a plan."""

    curves: tuple
    points: tuple
    surface_table: Any
    volumes: Any
    mm_contacts: tuple


def build_curved_construction_plan(build_input, *, route, representation_context=None):
    """Compile source curves globally before naming shared material surfaces."""
    import gmsh

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
    chains, operations = boundaries(prepared, rings)
    entities = {e.semantic_id: e for e in prepared.entities}
    initialized = bool(gmsh.isInitialized())
    if not initialized:
        gmsh.initialize(readConfigFiles=False)
    previous = gmsh.model.getCurrent()
    gmsh.model.add("scgsim-curved-arrangement")
    terminal = gmsh.option.getNumber("General.Terminal")
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        kernel = Kernel(gmsh)
        session = BuildSession(gmsh, kernel.occ, kernel, route, prepared, entities)
        return _build_plan(session, rings, chains, operations)
    finally:
        gmsh.model.remove()
        if previous:
            gmsh.model.setCurrent(previous)
        gmsh.option.setNumber("General.Terminal", terminal)
        if not initialized:
            gmsh.finalize()


def _build_plan(session, rings, chains, operations):
    """Run the construction stages in order and assemble the validated plan."""
    arrangement = build_planar_arrangement(session, rings, chains)
    ownership = build_slab_ownership(session, arrangement)
    sweep = sweep_curve_complex(session, arrangement, ownership.z_values)
    context = CurvedContext(session, arrangement, ownership, sweep)
    book = register_surfaces(context)
    curves = build_curve_table(context, book)
    table = build_surface_table(context, book, curves)
    mm_contacts = build_mm_contacts(context, book, curves)
    bind_port_conductor_components(table.surfaces, mm_contacts)
    apply_surface_contract(session, table, mm_contacts)
    bind_port_terminals(session, table)
    curve_records, point_records = finalize_curves_and_points(curves, table)
    volumes = build_volumes(context, book, curves, table)
    _keep_only_plan_entities(session, book, curves)
    payload = _write_boundary_xao(session.gmsh, book, curves)
    records = _PlanRecords(curve_records, point_records, table, volumes, mm_contacts)
    plan = _assemble_plan(context, curves, operations, payload, records)
    _validate_plan(plan)
    return plan


def _keep_only_plan_entities(session, book, curves):
    """Preserve exact shared boundary shapes in one detached XAO, before any volume exists."""
    gmsh = session.gmsh
    occ = session.occ
    active_faces = set(book.native_surfaces.values())
    occ.remove(
        [(2, t) for _, t in occ.getEntities(2) if t not in active_faces],
        recursive=False,
    )
    occ.remove(
        [(1, t) for _, t in occ.getEntities(1) if t not in curves.used_edges],
        recursive=False,
    )
    occ.synchronize()
    active_points = {
        abs(t)
        for edge in curves.used_edges
        for _, t in gmsh.model.getBoundary([(1, edge)])
    }
    occ.remove(
        [(0, t) for _, t in occ.getEntities(0) if t not in active_points],
        recursive=False,
    )
    occ.synchronize()


def _write_boundary_xao(gmsh, book, curves):
    """Name curves and surfaces by plan id, then export them as XAO bytes."""
    for edge, cid in curves.curve_ids.items():
        physical = gmsh.model.addPhysicalGroup(1, [edge])
        gmsh.model.setPhysicalName(1, physical, "SGB_CURVE::" + cid)
    for sid, tag in book.native_surfaces.items():
        physical = gmsh.model.addPhysicalGroup(2, [tag])
        gmsh.model.setPhysicalName(2, physical, "SGB_SURFACE::" + sid)
    with TemporaryDirectory(prefix="scgsim-curved-boundaries-") as temporary:
        path = Path(temporary) / "boundaries.xao"
        gmsh.write(str(path))
        return path.read_bytes()


def _assemble_plan(context, curves, operations, payload, records):

    session = context.session
    route = session.route
    arrangement = context.arrangement
    table = records.surface_table
    surfaces = table.surfaces
    volumes = records.volumes
    metadata = {
        "backend_strategy": "surface_plan_first_bottom_up_occ",
        "curved_boundary_xao": base64.b64encode(payload).decode("ascii"),
        "curved_boundary_xao_sha256": hashlib.sha256(payload).hexdigest(),
        "curved_arrangement": {
            "native_gmsh_version": session.gmsh.__version__,
            "source_boundary_bindings": [
                asdict(r) for r in session.prepared.boundary_curves
            ],
            "source_footprints": arrangement.source_footprints,
            "reconstruction_operations": operations,
            "planar_cells": [
                {
                    "cell_id": arrangement.cell_ids[c],
                    "source_membership": sorted(arrangement.membership[c]),
                }
                for c in sorted(arrangement.cells)
            ],
            "native_curve_intervals": [
                {
                    "curve_id": curves.curve_ids[e],
                    "geometry": arrangement.curve_descriptors[source],
                }
                for e, (level, source) in curves.lookup.lifted_inverse.items()
                if e in curves.curve_ids
            ],
            "z_levels_um": context.ownership.z_values,
            "shared_curves": len(records.curves),
            "shared_surfaces": len(surfaces),
            "solution_volumes": len(volumes),
            "sidewall_method": "native_collective_exact_edge_sweep_reusing_canonical_edges",
        },
    }
    plan_type = (
        RouteABConstructionPlanRecord if route in {"A", "B"} else ConstructionPlanRecord
    )
    return plan_type(
        route,
        interfaces=tuple(table.interfaces),
        points=records.points,
        curves=records.curves,
        surface_loops=tuple(table.loops),
        surfaces=tuple(surfaces),
        volumes=volumes,
        tags=plan_route_tags(route=route, surfaces=tuple(surfaces), volumes=volumes),
        metadata=metadata,
        port_sheet_regions=session.prepared.port_sheet_regions,
        mm_contacts=records.mm_contacts,
    )


def _validate_plan(plan):

    validate_curve_plan_coverage(
        points=plan.points,
        curves=plan.curves,
        surface_loops=plan.surface_loops,
        surfaces=plan.surfaces,
    )
    validate_volume_surface_closure(
        volumes=plan.volumes,
        surfaces=plan.surfaces,
        surface_loops=plan.surface_loops,
    )
    validate_tag_plan_coverage(
        surfaces=plan.surfaces, volumes=plan.volumes, tags=plan.tags
    )
