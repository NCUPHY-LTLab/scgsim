"""Plan solution-domain boundary surfaces and conductor exclusions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
from typing import Any, cast

from scgsim.geometry.grid.entities import entity_z_range_um
from scgsim.geometry.grid.entity_validation import is_solution_entity
from scgsim.geometry.grid.grid import boolean_polygon_region
from scgsim.geometry.grid.loops import (
    gdstk_surface_region,
    geometry_ref_from_gdstk_polygon,
    geometry_refs_from_gdstk_region,
    loop_centroid,
    loop_signature,
    simple_interior_hole_loops,
)
from scgsim.geometry.grid.planar_loops import clean_loop, polygon_area
from scgsim.geometry.grid.spatial import (
    geometry_ref_surface_z_um,
    intersect_bounds,
    same_coordinate_z,
)
from scgsim.geometry.interpretation.representation import (
    planned_conductor_representation,
)
from scgsim.geometry.models.common import RouteLiteral
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.models.topology import SurfacePlanRecord
from scgsim.geometry.planning.domain_planning.faces import (
    planar_side_solution_regions,
    route_a_sheet_boundary_volume_ids,
    route_a_sheet_plane_z_um,
)
from scgsim.geometry.planning.domain_planning.occupancy import (
    domain_bounds_loop,
    entity_occupied_region,
    route_conductor_contact_footprints,
    solution_bounds,
    solution_entity_xy_region,
    solution_region_entities,
)
from scgsim.geometry.planning.domain_planning.sidewalls import SolutionSidewallPlanner
from scgsim.geometry.planning.evidence import solution_solution_evidence
from scgsim.geometry.planning.surface_records import (
    RouteASheetPatch,
    surface_contribution_provenance,
)
from scgsim.semantics import SemanticEvidenceFacade


def plan_substrate_air_surfaces(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral | None,
    semantic_facts: SemanticEvidenceFacade | None = None,
    route_a_sheet_patches: Sequence[RouteASheetPatch] = (),
    representation_context: RepresentationContext | None = None,
) -> tuple[SurfacePlanRecord, ...]:
    """Plan solution boundaries and substrate-air surfaces for the route."""
    import gdstk  # noqa: PLC0415 -- defer the optional polygon engine until this operation reaches native geometry.

    records: list[SurfacePlanRecord] = list(
        _plan_solution_domain_boundary_surfaces(
            build_input, route=route, representation_context=representation_context
        )
    )
    surface_index = 0
    for lower, upper, z_um, _bounds in _solution_interface_planes(build_input):
        interface_region = boolean_polygon_region(
            gdstk,
            solution_entity_xy_region(gdstk, lower),
            solution_entity_xy_region(gdstk, upper),
            "and",
        )
        if not interface_region:
            continue
        material_pair = (lower.material_kind, upper.material_kind)
        if material_pair == ("dielectric", "vacuum"):
            physical_side = "top"
        elif material_pair == ("vacuum", "dielectric"):
            physical_side = "bottom"
        else:
            physical_side = "shared_plane"
        interface_evidence = solution_solution_evidence(
            cast(SemanticEvidenceFacade, semantic_facts),
            contribution_id=f"solution-solution:{lower.semantic_id}:{upper.semantic_id}:z={z_um}",
            patch_id=f"planned:{lower.semantic_id}:{upper.semantic_id}:z={z_um}",
            lower_id=lower.semantic_id,
            upper_id=upper.semantic_id,
            side=physical_side,
        )
        kind = interface_evidence.classification
        owner_ids = interface_evidence.source_owner_ids
        interface_region = tuple(
            sorted(
                interface_region,
                key=lambda polygon: (
                    round(loop_centroid(clean_loop(polygon.points))[0], 9),
                    round(loop_centroid(clean_loop(polygon.points))[1], 9),
                    round(abs(polygon_area(polygon.points)), 9),
                ),
            )
        )
        for patch in interface_region:
            base_geometry_ref = geometry_ref_from_gdstk_polygon(
                {"plane": {"axis": "z", "value_um": z_um}}, patch
            )
            plane_conductors = _conductor_entities_on_solution_plane(
                _SolutionPlaneRequest(
                    build_input,
                    route=route,
                    representation_context=representation_context,
                    lower=lower,
                    upper=upper,
                    z_um=z_um,
                    base_region=(patch,),
                )
            )
            sheet_geometry_refs = tuple(
                (
                    sheet_patch.geometry_ref
                    for sheet_patch in route_a_sheet_patches
                    if sheet_patch.boundary_volume_ids
                    == (lower.semantic_id, upper.semantic_id)
                    and same_coordinate_z(
                        geometry_ref_surface_z_um(sheet_patch.geometry_ref), z_um
                    )
                    and boolean_polygon_region(
                        gdstk,
                        gdstk_surface_region(sheet_patch.geometry_ref),
                        (patch,),
                        "and",
                    )
                )
            )
            solution_geometry_refs = _solution_interface_geometry_refs(
                base_geometry_ref,
                plane_conductors,
                conductor_geometry_refs=sheet_geometry_refs,
            )
            for geometry_ref in solution_geometry_refs:
                interface_id = (
                    f"{kind}__{owner_ids[0]}__{owner_ids[1]}__{surface_index:04d}"
                )
                records.append(
                    SurfacePlanRecord(
                        surface_id=f"SURF__{interface_id}",
                        owner_semantic_id=owner_ids[0],
                        surface_role="solution_interface",
                        geometry_ref=geometry_ref,
                        interface_id=interface_id,
                        valid_routes=() if route is None else (route,),
                        solver_use="solver_active",
                        metadata={
                            "interface_kinds": (kind,),
                            "owner_semantic_ids": owner_ids,
                            "boundary_volume_ids": (
                                lower.semantic_id,
                                upper.semantic_id,
                            ),
                            "source_provenance": surface_contribution_provenance(
                                parent_interface_id=f"solution-interface:{lower.semantic_id}:{upper.semantic_id}",
                                patch_id=interface_evidence.contribution_id,
                                contributions=(interface_evidence,),
                            ),
                        },
                    )
                )
                surface_index += 1
    return tuple(records)


def _solution_interface_geometry_refs(
    parent_geometry_ref: Mapping[str, Any],
    plane_conductors: Sequence[SemanticEntitySpec],
    *,
    conductor_geometry_refs: Sequence[Mapping[str, Any]] = (),
) -> tuple[dict[str, Any], ...]:
    """Create live solution-interface patches after removing conductors."""
    if not plane_conductors and (not conductor_geometry_refs):
        return (dict(parent_geometry_ref),)
    import gdstk  # noqa: PLC0415 -- defer the optional polygon engine until this operation reaches native geometry.

    hole_loops = simple_interior_hole_loops(
        parent_geometry_ref,
        plane_conductors,
        conductor_geometry_refs=conductor_geometry_refs,
    )
    if hole_loops is not None:
        return ({**dict(parent_geometry_ref), "hole_loops": hole_loops},)
    base_region = gdstk_surface_region(parent_geometry_ref)
    conductor_region = tuple(
        (
            polygon
            for entity in plane_conductors
            for polygon in entity_occupied_region(gdstk, entity)
        )
    ) + tuple(
        (
            polygon
            for geometry_ref in conductor_geometry_refs
            for polygon in gdstk_surface_region(geometry_ref)
        )
    )
    live_region = boolean_polygon_region(gdstk, base_region, conductor_region, "not")
    return geometry_refs_from_gdstk_region(parent_geometry_ref, live_region)


def _plan_solution_domain_boundary_surfaces(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral | None,
    representation_context: RepresentationContext | None = None,
) -> tuple[SurfacePlanRecord, ...]:
    records: list[SurfacePlanRecord] = []
    for entity in build_input.entities:
        if not is_solution_entity(entity):
            continue
        if "outer_loop" in entity.geometry:
            loop = clean_loop(entity.geometry["outer_loop"])
            z_min_um = float(entity.geometry["z_min_um"])
            z_max_um = float(entity.geometry["z_max_um"])
        else:
            bounds = entity.geometry.get("domain_bounds_um")
            if not isinstance(bounds, Mapping):
                continue
            loop = domain_bounds_loop(bounds)
            z_min_um = float(entity.geometry["z_min_um"])
            z_max_um = float(entity.geometry["z_max_um"])
        if len(loop) < 3:
            continue
        for boundary_role in ("bottom", "top"):
            geometry_refs = _solution_exterior_face_geometry_refs(
                build_input, entity, boundary_role
            )
            for component_index, geometry_ref in enumerate(geometry_refs):
                component_suffix = (
                    "" if len(geometry_refs) == 1 else f"__{component_index:04d}"
                )
                records.append(
                    SurfacePlanRecord(
                        surface_id=f"SURF__BOUNDARY__{entity.semantic_id}__{boundary_role.upper()}{component_suffix}",
                        owner_semantic_id=entity.semantic_id,
                        surface_role="domain_boundary",
                        geometry_ref=geometry_ref,
                        valid_routes=() if route is None else (route,),
                        metadata={
                            "owner_semantic_ids": (entity.semantic_id,),
                            "boundary_volume_ids": (entity.semantic_id,),
                            "boundary_role": boundary_role,
                        },
                    )
                )
        for edge_index, geometry_ref in enumerate(
            SolutionSidewallPlanner(
                build_input,
                route=route,
                solution=entity,
                outer_loop=loop,
                z_min_um=z_min_um,
                z_max_um=z_max_um,
                representation_context=representation_context,
            ).build()
        ):
            records.append(
                SurfacePlanRecord(
                    surface_id=f"SURF__BOUNDARY__{entity.semantic_id}__SIDEWALL__{edge_index:04d}",
                    owner_semantic_id=entity.semantic_id,
                    surface_role="domain_boundary",
                    geometry_ref=geometry_ref,
                    valid_routes=() if route is None else (route,),
                    metadata={
                        "owner_semantic_ids": (entity.semantic_id,),
                        "boundary_volume_ids": (entity.semantic_id,),
                        "boundary_role": "sidewall",
                    },
                )
            )
    return tuple(records)


def _solution_interface_planes(
    build_input: GeometryBuildInput,
) -> tuple[
    tuple[SemanticEntitySpec, SemanticEntitySpec, float, Mapping[str, float]], ...
]:
    records: list[
        tuple[SemanticEntitySpec, SemanticEntitySpec, float, Mapping[str, float]]
    ] = []
    solutions = solution_region_entities(build_input)
    for index, first in enumerate(solutions):
        for second in solutions[index + 1 :]:
            first_bounds = solution_bounds(first)
            second_bounds = solution_bounds(second)
            overlap = intersect_bounds(first_bounds, second_bounds)
            if overlap is None:
                continue
            if same_coordinate_z(
                float(first.geometry["z_max_um"]), float(second.geometry["z_min_um"])
            ):
                records.append(
                    (first, second, float(first.geometry["z_max_um"]), overlap)
                )
            elif same_coordinate_z(
                float(second.geometry["z_max_um"]), float(first.geometry["z_min_um"])
            ):
                records.append(
                    (second, first, float(second.geometry["z_max_um"]), overlap)
                )
    return tuple(records)


@dataclass(frozen=True)
class _SolutionPlaneRequest:
    """Carry the exact solution-plane patch and conductor selection context."""

    build_input: GeometryBuildInput
    _: KW_ONLY
    route: RouteLiteral | None
    lower: SemanticEntitySpec
    upper: SemanticEntitySpec
    z_um: float
    base_loop: tuple[tuple[float, float], ...] | None = None
    base_region: tuple[Any, ...] | None = None
    representation_context: RepresentationContext | None = None


def _conductor_entities_on_solution_plane(
    request: _SolutionPlaneRequest,
) -> tuple[SemanticEntitySpec, ...]:
    import gdstk  # noqa: PLC0415 -- defer the optional polygon engine until this operation reaches native geometry.

    pair_ids = {request.lower.semantic_id, request.upper.semantic_id}
    records: list[SemanticEntitySpec] = []
    if request.base_loop is not None and request.base_region is not None:
        raise ValueError("pass at most one of base_loop or base_region")
    if request.base_loop is None and request.base_region is None:
        raise ValueError("base_loop or base_region is required")
    patch_region = (
        request.base_region
        if request.base_region is not None
        else (
            gdstk.Polygon(
                clean_loop(cast(tuple[tuple[float, float], ...], request.base_loop))
            ),
        )
    )
    candidate_entities = (
        route_conductor_contact_footprints(
            request.build_input,
            request.route,
            representation_context=request.representation_context,
        )
        if request.route is None and request.representation_context is not None
        else request.build_input.entities
    )
    for entity in candidate_entities:
        if (
            is_solution_entity(entity)
            or planned_conductor_representation(
                entity, route=request.route, context=request.representation_context
            )
            is None
            or "outer_loop" not in entity.geometry
            or (
                not boolean_polygon_region(
                    gdstk, entity_occupied_region(gdstk, entity), patch_region, "and"
                )
            )
        ):
            continue
        representation = planned_conductor_representation(
            entity, route=request.route, context=request.representation_context
        )
        if representation == "surface_sheet":
            if (
                request.route != "A"
                and (
                    not (
                        request.route is None
                        and request.representation_context is not None
                    )
                )
                and (
                    set(route_a_sheet_boundary_volume_ids(request.build_input, entity))
                    == pair_ids
                )
                and same_coordinate_z(
                    route_a_sheet_plane_z_um(
                        request.build_input,
                        entity,
                        representation_context=request.representation_context,
                    ),
                    request.z_um,
                )
            ):
                records.append(entity)
            continue
        entity_region = entity_occupied_region(gdstk, entity)
        local_overlap = boolean_polygon_region(
            gdstk, entity_region, patch_region, "and"
        )
        if not local_overlap:
            continue
        for face, face_z in zip(
            ("bottom", "top"), entity_z_range_um(entity), strict=True
        ):
            if not same_coordinate_z(face_z, request.z_um):
                continue
            local_domains = planar_side_solution_regions(
                request.build_input,
                owner_id=entity.semantic_id,
                occupied_region=local_overlap,
                plane_z_um=face_z,
                side=face,
            )
            if any((domain_id in pair_ids for domain_id, _ in local_domains)):
                records.append(entity)
                break
    return tuple(records)


def _solution_exterior_face_geometry_refs(
    build_input: GeometryBuildInput, entity: SemanticEntitySpec, face: str
) -> tuple[dict[str, Any], ...]:
    import gdstk  # noqa: PLC0415 -- defer the optional polygon engine until this operation reaches native geometry.

    z_key = "z_max_um" if face == "top" else "z_min_um"
    z_um = float(entity.geometry[z_key])
    base_region = solution_entity_xy_region(gdstk, entity)
    if not base_region:
        return ()
    subtractor_region: tuple[Any, ...] = ()
    for other in solution_region_entities(build_input):
        if other.semantic_id == entity.semantic_id:
            continue
        touches_face = (
            face == "top"
            and same_coordinate_z(float(other.geometry["z_min_um"]), z_um)
            or (
                face == "bottom"
                and same_coordinate_z(float(other.geometry["z_max_um"]), z_um)
            )
        )
        if touches_face:
            candidate = solution_entity_xy_region(gdstk, other)
        else:
            continue
        if not candidate:
            continue
        if subtractor_region:
            subtractor_region = boolean_polygon_region(
                gdstk, subtractor_region, candidate, "or"
            )
        else:
            subtractor_region = candidate
    residual_region = boolean_polygon_region(
        gdstk, base_region, subtractor_region, "not"
    )
    if not residual_region:
        return ()
    geometry_refs = geometry_refs_from_gdstk_region(
        {"plane": {"axis": "z", "value_um": z_um}}, residual_region
    )
    return tuple(
        sorted(
            (
                geometry_ref
                for geometry_ref in geometry_refs
                if geometry_ref["outer_loop"]
            ),
            key=lambda geometry_ref: loop_signature(geometry_ref["outer_loop"]),
        )
    )
