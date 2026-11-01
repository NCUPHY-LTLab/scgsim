"""Plan the surfaces that bound solution entities: domain sidewalls, top and bottom faces.

Owns the exterior boundary of the simulation domain and the sidewall interfaces that
separate neighbouring solution entities. Does not plan conductor-owned surfaces.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from math import hypot
from typing import Any

from scgsim.geometry.models.common import RouteLiteral
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.models.topology import SurfacePlanRecord
from scgsim.geometry.planning.conductors.faces import active_route_conductor_entities
from scgsim.geometry.planning.context.evidence import solution_solution_evidence
from scgsim.geometry.planning.context.representation import (
    planned_conductor_representation,
)
from scgsim.geometry.planning.context.topology import surface_ring3d_specs
from scgsim.geometry.planning.solutions.occupancy import (
    domain_bounds_loop,
    list_solution_entities,
    solution_entity_xy_region,
)
from scgsim.geometry.primitives.constants import TOPOLOGY_EPS_UM
from scgsim.geometry.primitives.entities import entity_z_range_um, unique_ids
from scgsim.geometry.primitives.entity_validation import (
    is_solution_entity,
    is_vacuum_solution_entity,
)
from scgsim.geometry.primitives.loops import (
    boolean_gdstk_region,
    edge_is_covered_by_loop_edge,
    gdstk_surface_region,
    geometry_refs_from_gdstk_region,
    loop_signature,
    normalized_loop,
    ring_edges,
)
from scgsim.geometry.primitives.spatial import (
    geometry_ref_surface_z_um,
    interpolate_2d,
    is_same_z,
    segment_overlap_interval,
)
from scgsim.geometry.primitives.surface_records import (
    canonical_face_signature_3d,
    surface_boundary_volume_ids,
    surface_contribution_provenance,
)
from scgsim.semantics import SemanticEvidenceFacade


def plan_solution_domain_boundary_surfaces(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral | None,
    representation_context: RepresentationContext | None = None,
) -> tuple[SurfacePlanRecord, ...]:
    """Plan bottom, top and sidewall boundary surfaces for each solution entity."""
    records: list[SurfacePlanRecord] = []
    for entity in build_input.entities:
        if not is_solution_entity(entity):
            continue
        if "outer_loop" in entity.geometry:
            loop = normalized_loop(entity.geometry["outer_loop"])
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
                build_input,
                entity,
                boundary_role,
            )
            for component_index, geometry_ref in enumerate(geometry_refs):
                component_suffix = (
                    "" if len(geometry_refs) == 1 else f"__{component_index:04d}"
                )
                records.append(
                    SurfacePlanRecord(
                        surface_id=(
                            f"SURF__BOUNDARY__{entity.semantic_id}__"
                            f"{boundary_role.upper()}{component_suffix}"
                        ),
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
            _solution_domain_sidewall_geometry_refs(
                _SolutionBoundaryScope(
                    build_input, route, entity, representation_context
                ),
                outer_loop=loop,
                z_min_um=z_min_um,
                z_max_um=z_max_um,
            )
        ):
            records.append(
                SurfacePlanRecord(
                    surface_id=(
                        f"SURF__BOUNDARY__{entity.semantic_id}__"
                        f"SIDEWALL__{edge_index:04d}"
                    ),
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


@dataclass(frozen=True)
class _SolutionBoundaryScope:
    """The route, build input and solution entity one sidewall query runs against."""

    build_input: GeometryBuildInput
    route: RouteLiteral | None
    solution: SemanticEntitySpec
    representation_context: RepresentationContext | None = None


def _solution_domain_sidewall_geometry_refs(
    scope: _SolutionBoundaryScope,
    *,
    outer_loop: tuple[tuple[float, float], ...],
    z_min_um: float,
    z_max_um: float,
) -> tuple[dict[str, Any], ...]:
    solution = scope.solution
    refs: list[dict[str, Any]] = []
    for ring_role, ring_loop in (
        ("outer", normalized_loop(outer_loop)),
        *(
            (f"hole_{index:04d}", normalized_loop(hole_loop))
            for index, hole_loop in enumerate(solution.geometry.get("hole_loops", ()))
        ),
    ):
        for edge_index, (start, end) in enumerate(ring_edges(ring_loop)):
            parameters = _solution_boundary_edge_parameters(scope, start, end)
            for first, second in pairwise(parameters):
                segment_start = interpolate_2d(start, end, first)
                segment_end = interpolate_2d(start, end, second)
                if (
                    hypot(
                        segment_end[0] - segment_start[0],
                        segment_end[1] - segment_start[1],
                    )
                    <= TOPOLOGY_EPS_UM
                ):
                    continue
                edge_z_min_um = _solution_boundary_edge_z_min_um(
                    scope, segment_start, segment_end, default_z_min_um=z_min_um
                )
                edge_z_max_um = _solution_boundary_edge_z_max_um(
                    scope, segment_start, segment_end, default_z_max_um=z_max_um
                )
                if edge_z_max_um - edge_z_min_um <= TOPOLOGY_EPS_UM:
                    continue
                refs.append(
                    {
                        "quad_points": (
                            (*segment_start, edge_z_min_um),
                            (*segment_end, edge_z_min_um),
                            (*segment_end, edge_z_max_um),
                            (*segment_start, edge_z_max_um),
                        ),
                        "sidewall_ring_role": ring_role,
                        "sidewall_edge_index": edge_index,
                    }
                )
    return tuple(refs)


def _solution_boundary_edge_parameters(
    scope: _SolutionBoundaryScope,
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[float, ...]:
    """Split a vacuum exterior edge at exact finite-conductor endpoints."""
    build_input = scope.build_input
    route = scope.route
    solution = scope.solution
    representation_context = scope.representation_context
    if not (
        route == "B" or (route is None and representation_context is not None)
    ) or not is_vacuum_solution_entity(solution):
        return (0.0, 1.0)
    parameters = {0.0, 1.0}
    solution_z_min = float(solution.geometry["z_min_um"])
    solution_z_max = float(solution.geometry["z_max_um"])
    for entity in active_route_conductor_entities(
        build_input, route, representation_context=representation_context
    ):
        if planned_conductor_representation(
            entity, route=route, context=representation_context
        ) not in {
            "cutout_boundary_shell",
            "material_volume",
        }:
            continue
        entity_z_min, entity_z_max = entity_z_range_um(entity)
        if not (
            is_same_z(entity_z_min, solution_z_min)
            or is_same_z(entity_z_max, solution_z_max)
        ):
            continue
        for edge_start, edge_end in ring_edges(
            normalized_loop(entity.geometry["outer_loop"])
        ):
            interval = segment_overlap_interval(start, end, edge_start, edge_end)
            if interval is not None:
                parameters.update(interval)
    return tuple(sorted(parameters))


def _solution_boundary_edge_z_min_um(
    scope: _SolutionBoundaryScope,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    default_z_min_um: float,
) -> float:
    build_input = scope.build_input
    route = scope.route
    solution = scope.solution
    representation_context = scope.representation_context
    if not is_vacuum_solution_entity(solution):
        return default_z_min_um
    z_min_um = default_z_min_um
    for entity in active_route_conductor_entities(
        build_input, route, representation_context=representation_context
    ):
        if (
            is_solution_entity(entity)
            or planned_conductor_representation(
                entity, route=route, context=representation_context
            )
            not in {
                "cutout_boundary_shell",
                "material_volume",
            }
            or "outer_loop" not in entity.geometry
        ):
            continue
        entity_z_min_um, entity_z_max_um = entity_z_range_um(entity)
        if not is_same_z(entity_z_min_um, default_z_min_um):
            continue
        if edge_is_covered_by_loop_edge(start, end, entity.geometry["outer_loop"]):
            z_min_um = max(z_min_um, entity_z_max_um)
    return z_min_um


def _solution_boundary_edge_z_max_um(
    scope: _SolutionBoundaryScope,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    default_z_max_um: float,
) -> float:
    build_input = scope.build_input
    route = scope.route
    solution = scope.solution
    representation_context = scope.representation_context
    if not is_vacuum_solution_entity(solution):
        return default_z_max_um
    z_max_um = default_z_max_um
    for entity in active_route_conductor_entities(
        build_input, route, representation_context=representation_context
    ):
        if (
            is_solution_entity(entity)
            or planned_conductor_representation(
                entity, route=route, context=representation_context
            )
            not in {
                "cutout_boundary_shell",
                "material_volume",
            }
            or "outer_loop" not in entity.geometry
        ):
            continue
        entity_z_min_um, entity_z_max_um = entity_z_range_um(entity)
        if not is_same_z(entity_z_max_um, default_z_max_um):
            continue
        if edge_is_covered_by_loop_edge(start, end, entity.geometry["outer_loop"]):
            z_max_um = min(z_max_um, entity_z_min_um)
    return z_max_um


def merge_solution_sidewall_interfaces(
    build_input: GeometryBuildInput,
    *,
    surfaces: tuple[SurfacePlanRecord, ...],
    semantic_facts: SemanticEvidenceFacade | None = None,
) -> tuple[SurfacePlanRecord, ...]:
    """Merge coincident solution sidewalls into one shared interface surface.

    Two neighbouring solutions each plan their own wall on the face they share. Walls
    with the same canonical face are one physical interface, so they become one surface.
    Merged and unmerged walls come first, in group order, then every other surface.
    """
    entities = {entity.semantic_id: entity for entity in build_input.entities}
    solution_ids = {
        entity.semantic_id for entity in entities.values() if is_solution_entity(entity)
    }
    non_merge: list[SurfacePlanRecord] = []
    sidewall_groups: dict[
        tuple[tuple[float, float, float], ...],
        list[SurfacePlanRecord],
    ] = {}

    for surface in surfaces:
        signature = _mergeable_sidewall_signature(surface, entities, solution_ids)
        if signature is None:
            non_merge.append(surface)
        else:
            sidewall_groups.setdefault(signature, []).append(surface)

    grouped_surfaces: list[SurfacePlanRecord] = []
    for signature, group in sidewall_groups.items():
        if len(group) == 1:
            grouped_surfaces.append(group[0])
            continue
        grouped_surfaces.append(
            _merged_sidewall(
                group,
                signature=signature,
                entities=entities,
                solution_ids=solution_ids,
                semantic_facts=semantic_facts,
            )
        )
    grouped_surfaces.extend(non_merge)
    return tuple(grouped_surfaces)


def _mergeable_sidewall_signature(
    surface: SurfacePlanRecord,
    entities: Mapping[str, SemanticEntitySpec],
    solution_ids: set[str],
) -> tuple[tuple[float, float, float], ...] | None:
    """Return the canonical face signature of a one-solution quad sidewall, else None."""
    if surface.construction_only:
        return None
    if (
        surface.surface_role != "domain_boundary"
        or surface.metadata.get("boundary_role") != "sidewall"
    ):
        return None
    if "quad_points" not in surface.geometry_ref:
        return None
    boundary_ids = surface_boundary_volume_ids(
        surface,
        known_entity_ids=solution_ids,
    )
    if len(boundary_ids) != 1:
        return None
    owner = entities.get(boundary_ids[0])
    if owner is None or not is_solution_entity(owner):
        return None
    specs = surface_ring3d_specs(surface)
    if len(specs) != 1:
        return None
    return canonical_face_signature_3d(specs[0][2])


def _merged_sidewall(
    group: Sequence[SurfacePlanRecord],
    *,
    signature: tuple[tuple[float, float, float], ...],
    entities: Mapping[str, SemanticEntitySpec],
    solution_ids: set[str],
    semantic_facts: SemanticEvidenceFacade | None,
) -> SurfacePlanRecord:
    """Merge several walls on one face into the first, as a two-solution interface."""
    all_ids = unique_ids(
        value
        for surface in group
        for value in surface_boundary_volume_ids(
            surface,
            known_entity_ids=solution_ids,
        )
    )
    if len(all_ids) != 2:
        names = tuple(surface.surface_id for surface in group)
        raise ValueError(
            "duplicate solution sidewall cannot define a 2-owner interface: "
            + f"{signature} => {names!r}"
        )
    lower = entities[all_ids[0]]
    upper = entities[all_ids[1]]
    interface_evidence = solution_solution_evidence(
        semantic_facts,
        contribution_id=(
            f"solution-sidewall:{lower.semantic_id}:{upper.semantic_id}:"
            f"{group[0].surface_id.rsplit('__', maxsplit=1)[-1]}"
        ),
        patch_id=f"planned:merged:{group[0].surface_id}",
        lower_id=lower.semantic_id,
        upper_id=upper.semantic_id,
        side="sidewall",
    )
    kind = interface_evidence.classification
    owner_ids = interface_evidence.source_owner_ids
    edge_suffix = group[0].surface_id.rsplit("__", maxsplit=1)[-1]
    interface_id = f"{kind}__{owner_ids[0]}__{owner_ids[1]}__{edge_suffix}"
    return replace(
        group[0],
        owner_semantic_id=owner_ids[0],
        interface_id=interface_id,
        metadata={
            **group[0].metadata,
            "owner_semantic_ids": owner_ids,
            "boundary_volume_ids": owner_ids,
            "interface_kinds": (kind,),
            "interface_type": kind,
            "source_provenance": surface_contribution_provenance(
                parent_interface_id=(
                    f"solution-sidewall:{lower.semantic_id}:{upper.semantic_id}"
                ),
                patch_id=interface_evidence.contribution_id,
                contributions=(interface_evidence,),
            ),
        },
    )


def reconcile_solution_domain_boundaries(
    build_input: GeometryBuildInput,
    *,
    surfaces: tuple[SurfacePlanRecord, ...],
) -> tuple[SurfacePlanRecord, ...]:
    """Reuse live structured interfaces in place of duplicate domain faces.

    A domain boundary is only replaced by a surface that explicitly names the
    same solution volume in its structured boundary ownership. Horizontal
    regions are reconciled by exact planar Boolean residuals; sidewalls are
    removed only for one-for-one canonical face equality. Partial vertical
    overlap remains an error for the existing topology validators to expose.
    """
    import gdstk

    solution_ids = {
        entity.semantic_id
        for entity in build_input.entities
        if is_solution_entity(entity)
    }
    retained: list[SurfacePlanRecord] = []
    non_domain_surfaces = tuple(
        surface
        for surface in surfaces
        if not surface.construction_only
        and surface.solver_use == "solver_active"
        and surface.surface_role != "domain_boundary"
    )
    for surface in surfaces:
        retained.extend(
            _reconciled_domain_surface(
                gdstk, surface, non_domain_surfaces, solution_ids
            )
        )
    return tuple(retained)


def _reconciled_domain_surface(
    gdstk: Any,
    surface: SurfacePlanRecord,
    non_domain_surfaces: Sequence[SurfacePlanRecord],
    solution_ids: set[str],
) -> list[SurfacePlanRecord]:
    """Return what remains of one surface: itself, residual pieces, or nothing.

    Only a domain boundary that bounds exactly one solution, and that another live
    surface also bounds, can be replaced. Everything else is kept unchanged.
    """
    if surface.construction_only or surface.surface_role != "domain_boundary":
        return [surface]
    boundary_ids = surface_boundary_volume_ids(
        surface,
        known_entity_ids=solution_ids,
    )
    if len(boundary_ids) != 1:
        return [surface]
    owner_id = boundary_ids[0]
    candidates = tuple(
        candidate
        for candidate in non_domain_surfaces
        if owner_id
        in surface_boundary_volume_ids(
            candidate,
            known_entity_ids=solution_ids,
        )
    )
    if not candidates:
        return [surface]
    if "quad_points" in surface.geometry_ref:
        # A sidewall is dropped only when another surface is the identical quad.
        return [] if _has_identical_quad(surface, candidates) else [surface]
    if "outer_loop" not in surface.geometry_ref:
        return [surface]
    return _domain_face_residual_surfaces(gdstk, surface, candidates)


def _has_identical_quad(
    surface: SurfacePlanRecord, candidates: Sequence[SurfacePlanRecord]
) -> bool:
    """True when a candidate has the same single 3D face as the surface."""
    domain_specs = surface_ring3d_specs(surface)
    domain_signature = (
        canonical_face_signature_3d(domain_specs[0][2])
        if len(domain_specs) == 1
        else None
    )
    return domain_signature is not None and any(
        len(candidate_specs := surface_ring3d_specs(candidate)) == 1
        and canonical_face_signature_3d(candidate_specs[0][2]) == domain_signature
        for candidate in candidates
    )


def _domain_face_residual_surfaces(
    gdstk: Any,
    surface: SurfacePlanRecord,
    candidates: Sequence[SurfacePlanRecord],
) -> list[SurfacePlanRecord]:
    """Cut the horizontal candidates out of a domain face and keep what is left over."""
    plane_z_um = geometry_ref_surface_z_um(surface.geometry_ref)
    planar_candidates = tuple(
        candidate
        for candidate in candidates
        if "quad_points" not in candidate.geometry_ref
        and "outer_loop" in candidate.geometry_ref
        and is_same_z(
            geometry_ref_surface_z_um(candidate.geometry_ref),
            plane_z_um,
        )
    )
    if not planar_candidates:
        return [surface]
    replacement_region: tuple[Any, ...] = ()
    for candidate in sorted(planar_candidates, key=lambda item: item.surface_id):
        candidate_region = gdstk_surface_region(candidate.geometry_ref)
        replacement_region = (
            candidate_region
            if not replacement_region
            else boolean_gdstk_region(
                gdstk,
                replacement_region,
                candidate_region,
                "or",
            )
        )
    residual_region = boolean_gdstk_region(
        gdstk,
        gdstk_surface_region(surface.geometry_ref),
        replacement_region,
        "not",
    )
    residual_refs = tuple(
        sorted(
            geometry_refs_from_gdstk_region(surface.geometry_ref, residual_region),
            key=lambda geometry_ref: loop_signature(geometry_ref["outer_loop"]),
        )
    )
    return [
        replace(
            surface,
            surface_id=(
                surface.surface_id
                if len(residual_refs) == 1
                else f"{surface.surface_id}__R{index:04d}"
            ),
            geometry_ref=geometry_ref,
        )
        for index, geometry_ref in enumerate(residual_refs)
    ]


def _solution_exterior_face_geometry_refs(
    build_input: GeometryBuildInput,
    entity: SemanticEntitySpec,
    face: str,
) -> tuple[dict[str, Any], ...]:
    import gdstk

    z_key = "z_max_um" if face == "top" else "z_min_um"
    z_um = float(entity.geometry[z_key])
    base_region = solution_entity_xy_region(gdstk, entity)
    if not base_region:
        return ()
    subtractor_region: tuple[Any, ...] = ()
    for other in list_solution_entities(build_input):
        if other.semantic_id == entity.semantic_id:
            continue
        touches_face = (
            face == "top" and is_same_z(float(other.geometry["z_min_um"]), z_um)
        ) or (face == "bottom" and is_same_z(float(other.geometry["z_max_um"]), z_um))
        if touches_face:
            candidate = solution_entity_xy_region(gdstk, other)
        else:
            continue
        if not candidate:
            continue
        if subtractor_region:
            subtractor_region = boolean_gdstk_region(
                gdstk,
                subtractor_region,
                candidate,
                "or",
            )
        else:
            subtractor_region = candidate
    residual_region = boolean_gdstk_region(
        gdstk,
        base_region,
        subtractor_region,
        "not",
    )
    if not residual_region:
        return ()
    geometry_refs = geometry_refs_from_gdstk_region(
        {"plane": {"axis": "z", "value_um": z_um}},
        residual_region,
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
