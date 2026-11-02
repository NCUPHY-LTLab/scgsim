"""Merge and reconcile duplicate solution boundary surface records."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

from scgsim.geometry.models.input import GeometryBuildInput
from scgsim.geometry.models.topology import SurfacePlanRecord
from scgsim.geometry.planning.evidence import solution_solution_evidence
from scgsim.geometry.planning.topology import surface_ring3d_specs
from scgsim.geometry.primitives.entities import unique_ids
from scgsim.geometry.primitives.entity_validation import is_solution_entity
from scgsim.geometry.primitives.grid import boolean_polygon_region
from scgsim.geometry.primitives.loops import (
    gdstk_surface_region,
    geometry_refs_from_gdstk_region,
    loop_signature,
)
from scgsim.geometry.primitives.spatial import (
    geometry_ref_surface_z_um,
    same_coordinate_z,
)
from scgsim.geometry.primitives.surface_records import (
    canonical_face_signature_3d,
    surface_boundary_volume_ids,
    surface_contribution_provenance,
)
from scgsim.semantics import SemanticEvidenceFacade


def merge_solution_sidewall_interfaces(
    build_input: GeometryBuildInput,
    *,
    surfaces: tuple[SurfacePlanRecord, ...],
    semantic_facts: SemanticEvidenceFacade | None = None,
) -> tuple[SurfacePlanRecord, ...]:
    """Merge coincident solution sidewalls into one shared interface surface."""
    entities = {entity.semantic_id: entity for entity in build_input.entities}
    solution_ids = {
        entity.semantic_id for entity in entities.values() if is_solution_entity(entity)
    }
    non_merge: list[SurfacePlanRecord] = []
    sidewall_groups: dict[
        tuple[tuple[float, float, float], ...], list[SurfacePlanRecord]
    ] = {}
    for surface in surfaces:
        if surface.construction_only:
            non_merge.append(surface)
            continue
        if (
            surface.surface_role != "domain_boundary"
            or surface.metadata.get("boundary_role") != "sidewall"
        ):
            non_merge.append(surface)
            continue
        if "quad_points" not in surface.geometry_ref:
            non_merge.append(surface)
            continue
        boundary_ids = surface_boundary_volume_ids(
            surface, known_entity_ids=solution_ids
        )
        if len(boundary_ids) != 1:
            non_merge.append(surface)
            continue
        owner_id = boundary_ids[0]
        owner = entities.get(owner_id)
        if owner is None or not is_solution_entity(owner):
            non_merge.append(surface)
            continue
        specs = surface_ring3d_specs(surface)
        if len(specs) != 1:
            non_merge.append(surface)
            continue
        signature = canonical_face_signature_3d(specs[0][2])
        if signature is None:
            non_merge.append(surface)
            continue
        sidewall_groups.setdefault(signature, []).append(surface)
    grouped_surfaces: list[SurfacePlanRecord] = []
    for signature, group in sidewall_groups.items():
        if len(group) == 1:
            grouped_surfaces.append(group[0])
            continue
        all_ids = unique_ids(
            (
                value
                for surface in group
                for value in surface_boundary_volume_ids(
                    surface, known_entity_ids=solution_ids
                )
            )
        )
        if len(all_ids) != 2:
            names = tuple((surface.surface_id for surface in group))
            raise ValueError(
                "duplicate solution sidewall cannot define a 2-owner interface: "
                + f"{signature} => {names!r}"
            )
        lower = entities[all_ids[0]]
        upper = entities[all_ids[1]]
        interface_evidence = solution_solution_evidence(
            cast(SemanticEvidenceFacade, semantic_facts),
            contribution_id=f"solution-sidewall:{lower.semantic_id}:{upper.semantic_id}:{group[0].surface_id.rsplit('__', maxsplit=1)[-1]}",
            patch_id=f"planned:merged:{group[0].surface_id}",
            lower_id=lower.semantic_id,
            upper_id=upper.semantic_id,
            side="sidewall",
        )
        kind = interface_evidence.classification
        owner_ids = interface_evidence.source_owner_ids
        edge_suffix = group[0].surface_id.rsplit("__", maxsplit=1)[-1]
        interface_id = f"{kind}__{owner_ids[0]}__{owner_ids[1]}__{edge_suffix}"
        merged = replace(
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
                    parent_interface_id=f"solution-sidewall:{lower.semantic_id}:{upper.semantic_id}",
                    patch_id=interface_evidence.contribution_id,
                    contributions=(interface_evidence,),
                ),
            },
        )
        grouped_surfaces.append(merged)
    grouped_surfaces.extend(non_merge)
    return tuple(grouped_surfaces)


def reconcile_solution_domain_boundaries(
    build_input: GeometryBuildInput, *, surfaces: tuple[SurfacePlanRecord, ...]
) -> tuple[SurfacePlanRecord, ...]:
    """Reuse live structured interfaces in place of duplicate domain faces.

    A domain boundary is only replaced by a surface that explicitly names the
    same solution volume in its structured boundary ownership. Horizontal
    regions are reconciled by exact planar Boolean residuals; sidewalls are
    removed only for one-for-one canonical face equality. Partial vertical
    overlap remains an error for the existing topology validators to expose.
    """
    import gdstk  # noqa: PLC0415 -- defer the optional polygon engine until this operation reaches native geometry.

    solution_ids = {
        entity.semantic_id
        for entity in build_input.entities
        if is_solution_entity(entity)
    }
    retained: list[SurfacePlanRecord] = []
    non_domain_surfaces = tuple(
        (
            surface
            for surface in surfaces
            if not surface.construction_only
            and surface.solver_use == "solver_active"
            and (surface.surface_role != "domain_boundary")
        )
    )
    for surface in surfaces:
        if surface.construction_only or surface.surface_role != "domain_boundary":
            retained.append(surface)
            continue
        boundary_ids = surface_boundary_volume_ids(
            surface, known_entity_ids=solution_ids
        )
        if len(boundary_ids) != 1:
            retained.append(surface)
            continue
        owner_id = boundary_ids[0]
        candidates = tuple(
            (
                candidate
                for candidate in non_domain_surfaces
                if owner_id
                in surface_boundary_volume_ids(candidate, known_entity_ids=solution_ids)
            )
        )
        if not candidates:
            retained.append(surface)
            continue
        if "quad_points" in surface.geometry_ref:
            domain_specs = surface_ring3d_specs(surface)
            domain_signature = (
                canonical_face_signature_3d(domain_specs[0][2])
                if len(domain_specs) == 1
                else None
            )
            if domain_signature is not None and any(
                (
                    len((candidate_specs := surface_ring3d_specs(candidate))) == 1
                    and canonical_face_signature_3d(candidate_specs[0][2])
                    == domain_signature
                    for candidate in candidates
                )
            ):
                continue
            retained.append(surface)
            continue
        if "outer_loop" not in surface.geometry_ref:
            retained.append(surface)
            continue
        plane_z_um = geometry_ref_surface_z_um(surface.geometry_ref)
        planar_candidates = tuple(
            (
                candidate
                for candidate in candidates
                if "quad_points" not in candidate.geometry_ref
                and "outer_loop" in candidate.geometry_ref
                and same_coordinate_z(
                    geometry_ref_surface_z_um(candidate.geometry_ref), plane_z_um
                )
            )
        )
        if not planar_candidates:
            retained.append(surface)
            continue
        replacement_region: tuple[Any, ...] = ()
        for candidate in sorted(planar_candidates, key=lambda item: item.surface_id):
            candidate_region = gdstk_surface_region(candidate.geometry_ref)
            replacement_region = (
                candidate_region
                if not replacement_region
                else boolean_polygon_region(
                    gdstk, replacement_region, candidate_region, "or"
                )
            )
        residual_region = boolean_polygon_region(
            gdstk, gdstk_surface_region(surface.geometry_ref), replacement_region, "not"
        )
        residual_refs = tuple(
            sorted(
                geometry_refs_from_gdstk_region(surface.geometry_ref, residual_region),
                key=lambda geometry_ref: loop_signature(geometry_ref["outer_loop"]),
            )
        )
        if not residual_refs:
            continue
        for index, geometry_ref in enumerate(residual_refs):
            surface_id = (
                surface.surface_id
                if len(residual_refs) == 1
                else f"{surface.surface_id}__R{index:04d}"
            )
            retained.append(
                replace(surface, surface_id=surface_id, geometry_ref=geometry_ref)
            )
    return tuple(retained)
