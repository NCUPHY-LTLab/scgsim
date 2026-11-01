"""Split conductor top and bottom faces into pieces by the solution entity they touch.

Owns the per-face solution pieces and the contact footprints of conductors that a route
represents as a shell or material volume.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from scgsim.geometry._primitives.entities import _entity_z_range_um
from scgsim.geometry._primitives.entity_validation import _is_solution_entity
from scgsim.geometry._primitives.loops import (
    _boolean_gdstk_region,
    _clean_loop,
    _gdstk_surface_region,
    _geometry_refs_from_gdstk_region,
    _loop_signature,
)
from scgsim.geometry._primitives.spatial import _same_z
from scgsim.geometry.models.common import RouteLiteral
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.planning.representation import planned_conductor_representation
from scgsim.geometry.planning.solution_adjacency import (
    _conductor_face_adjacent_solution_id,
    _route_a_sheet_boundary_volume_ids,
    _route_a_sheet_plane_z_um,
)
from scgsim.geometry.planning.solution_geometry import (
    _entity_occupied_region,
    _solution_entities,
    _solution_entity_xy_region,
)


def _conductor_face_solution_pieces(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral | None,
    entity: SemanticEntitySpec,
    shell_part: str,
    geometry_refs: Sequence[Mapping[str, Any]],
    representation_context: RepresentationContext | None = None,
) -> tuple[tuple[str, dict[str, Any]], ...]:
    """Partition Route-A/B planar faces by exact local solution adjacency."""
    if route not in {"A", "B"} and not (
        route is None and representation_context is not None
    ):
        adjacent_id = _conductor_face_adjacent_solution_id(
            build_input,
            entity,
            shell_part,
        )
        return tuple(
            (adjacent_id, dict(geometry_ref)) for geometry_ref in geometry_refs
        )

    z_min_um, z_max_um = _entity_z_range_um(entity)
    plane_z_um = z_min_um if shell_part == "bottom" else z_max_um
    records: list[tuple[str, dict[str, Any]]] = []
    for geometry_ref in geometry_refs:
        occupied_region = _gdstk_surface_region(geometry_ref)
        adjacent_regions = _planar_side_solution_regions(
            build_input,
            owner_id=entity.semantic_id,
            occupied_region=occupied_region,
            plane_z_um=plane_z_um,
            side=shell_part,
        )
        for adjacent_id, region in adjacent_regions:
            records.extend(
                (adjacent_id, child)
                for child in _geometry_refs_from_gdstk_region(geometry_ref, region)
            )
    return tuple(
        sorted(
            records,
            key=lambda item: (
                item[0],
                _loop_signature(item[1]["outer_loop"]),
            ),
        )
    )


def _planar_side_solution_regions(
    build_input: GeometryBuildInput,
    *,
    owner_id: str,
    occupied_region: tuple[Any, ...],
    plane_z_um: float,
    side: str,
) -> tuple[tuple[str, tuple[Any, ...]], ...]:
    """Resolve exact nonoverlapping solution coverage on one planar side."""
    import gdstk

    records: list[tuple[str, tuple[Any, ...]]] = []
    for solution in _solution_entities(build_input):
        z_min_um = float(solution.geometry["z_min_um"])
        z_max_um = float(solution.geometry["z_max_um"])
        on_boundary = (
            _same_z(z_max_um, plane_z_um)
            if side == "bottom"
            else _same_z(z_min_um, plane_z_um)
        )
        # Endpoint equivalence and strict interior must be mutually exclusive.
        contains_plane = (
            z_min_um < plane_z_um < z_max_um
            and not _same_z(z_min_um, plane_z_um)
            and not _same_z(z_max_um, plane_z_um)
        )
        if not on_boundary and not contains_plane:
            continue
        overlap = _boolean_gdstk_region(
            gdstk,
            occupied_region,
            _solution_entity_xy_region(gdstk, solution),
            "and",
        )
        if overlap:
            records.append((solution.semantic_id, overlap))

    for index, (left_id, left_region) in enumerate(records):
        for right_id, right_region in records[index + 1 :]:
            if _boolean_gdstk_region(gdstk, left_region, right_region, "and"):
                raise ValueError(
                    f"{owner_id} {side} local adjacency is ambiguous "
                    f"between {left_id!r} and {right_id!r}."
                )
    covered: tuple[Any, ...] = ()
    for _, region in records:
        covered = (
            region
            if not covered
            else _boolean_gdstk_region(gdstk, covered, region, "or")
        )
    uncovered = _boolean_gdstk_region(gdstk, occupied_region, covered, "not")
    if uncovered:
        raise ValueError(f"{owner_id} {side} has no local solution coverage.")
    return tuple(records)


def _active_route_conductor_entities(
    build_input: GeometryBuildInput,
    route: RouteLiteral | None,
    *,
    representation_context: RepresentationContext | None = None,
) -> tuple[SemanticEntitySpec, ...]:
    if route is None and representation_context is not None:
        return _route_conductor_contact_footprints(
            build_input,
            route,
            representation_context=representation_context,
        )
    return tuple(
        entity
        for entity in build_input.entities
        if not _is_solution_entity(entity)
        and planned_conductor_representation(
            entity, route=route, context=representation_context
        )
        is not None
        and "outer_loop" in entity.geometry
    )


def _route_conductor_contact_footprints(
    build_input: GeometryBuildInput,
    route: RouteLiteral | None,
    *,
    representation_context: RepresentationContext | None = None,
) -> tuple[SemanticEntitySpec, ...]:
    """Recognize effective contacts on each declared source polygon.

    Temporary footprints retain parent Entity identity and original holes;
    they neither replace source bodies nor invent a bounding exterior.
    """
    if route is not None:
        return _active_route_conductor_entities(
            build_input, route, representation_context=representation_context
        )
    if representation_context is None:
        return ()
    polygons = {polygon.polygon_id: polygon for polygon in build_input.polygons}
    footprints: list[SemanticEntitySpec] = []
    for entity in build_input.entities:
        if (
            _is_solution_entity(entity)
            or planned_conductor_representation(
                entity, route=route, context=representation_context
            )
            is None
        ):
            continue
        if "outer_loop" in entity.geometry:
            footprints.append(entity)
            continue
        for polygon_id in entity.polygon_ids:
            polygon = polygons[polygon_id]
            footprints.append(
                replace(
                    entity,
                    polygon_ids=(polygon_id,),
                    geometry={
                        **entity.geometry,
                        "outer_loop": polygon.exterior,
                        "hole_loops": polygon.holes,
                    },
                )
            )
    return tuple(footprints)


def _conductor_entities_on_solution_plane(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral | None,
    lower: SemanticEntitySpec,
    upper: SemanticEntitySpec,
    z_um: float,
    base_loop: tuple[tuple[float, float], ...] | None = None,
    base_region: tuple[Any, ...] | None = None,
    representation_context: RepresentationContext | None = None,
) -> tuple[SemanticEntitySpec, ...]:
    import gdstk

    pair_ids = {lower.semantic_id, upper.semantic_id}
    records: list[SemanticEntitySpec] = []
    if base_loop is not None and base_region is not None:
        raise ValueError("pass at most one of base_loop or base_region")
    if base_loop is None and base_region is None:
        raise ValueError("base_loop or base_region is required")
    patch_region = (
        base_region
        if base_region is not None
        else (gdstk.Polygon(_clean_loop(base_loop)),)
    )
    candidate_entities = (
        _route_conductor_contact_footprints(
            build_input,
            route,
            representation_context=representation_context,
        )
        if route is None and representation_context is not None
        else build_input.entities
    )
    for entity in candidate_entities:
        if (
            _is_solution_entity(entity)
            or planned_conductor_representation(
                entity, route=route, context=representation_context
            )
            is None
            or "outer_loop" not in entity.geometry
            or not _boolean_gdstk_region(
                gdstk,
                _entity_occupied_region(gdstk, entity),
                patch_region,
                "and",
            )
        ):
            continue
        representation = planned_conductor_representation(
            entity, route=route, context=representation_context
        )
        if representation == "surface_sheet":
            # Route-A sheets are removed by exact local patch geometry in
            # `_plan_substrate_air_surfaces`; whole-entity coverage cannot
            # represent a sheet spanning supported and exposed regions.
            if (
                route != "A"
                and not (route is None and representation_context is not None)
                and set(_route_a_sheet_boundary_volume_ids(build_input, entity))
                == pair_ids
                and _same_z(
                    _route_a_sheet_plane_z_um(
                        build_input,
                        entity,
                        representation_context=representation_context,
                    ),
                    z_um,
                )
            ):
                records.append(entity)
            continue
        entity_region = _entity_occupied_region(gdstk, entity)
        local_overlap = _boolean_gdstk_region(gdstk, entity_region, patch_region, "and")
        if not local_overlap:
            continue
        for face, face_z in zip(
            ("bottom", "top"),
            _entity_z_range_um(entity),
            strict=True,
        ):
            if not _same_z(face_z, z_um):
                continue
            local_domains = _planar_side_solution_regions(
                build_input,
                owner_id=entity.semantic_id,
                occupied_region=local_overlap,
                plane_z_um=face_z,
                side=face,
            )
            if any(domain_id in pair_ids for domain_id, _ in local_domains):
                records.append(entity)
                break
    return tuple(records)
