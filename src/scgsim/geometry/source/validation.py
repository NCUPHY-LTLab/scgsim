"""Validate a ``GeometryBuildInput`` from its own contents: polygons, entities, attachments and port regions.

The entity predicates it uses are in ``models.entities``; the contact-pad attachment is in ``source.contact_pads``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import isfinite

from scgsim.geometry.models.entities import (
    is_solution_entity,
    is_vacuum_solution_entity,
    requires_route_representation,
)
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.refusals import Declared
from scgsim.geometry.models.regions import PortSheetRegionRecord
from scgsim.geometry.source.contact_pads import resolve_contact_pad_attachment


def has_vacuum_solution_region(build_input: GeometryBuildInput) -> bool:
    """Whether the input holds a vacuum solution region: the one rule validation and Palace preparation share."""
    return any(is_vacuum_solution_entity(entity) for entity in build_input.entities)


def validate_geometry_input(build_input: GeometryBuildInput) -> GeometryBuildInput:
    """Validate adapter-normalized input before route-aware planning."""
    state = _InputValidation(build_input)
    _validate_polygons(state)
    _validate_entities(state)
    _validate_attachments(state)
    _validate_port_regions(state)
    if not has_vacuum_solution_region(build_input):
        state.errors.append("GeometryBuildInput requires a vacuum solution_region")
    if state.errors:
        raise ValueError(
            Declared(
                "the geometry input",
                "the input: build it with GeometryPlan.prepare() or build_gds_stack_geometry_input",
            ).refusal("Invalid GeometryBuildInput: " + "; ".join(state.errors))
        )
    return build_input


@dataclass
class _InputValidation:
    """Carry source indexes and ordered errors through one validation call."""

    build_input: GeometryBuildInput
    errors: list[str] = field(default_factory=list)
    polygon_ids: set[str] = field(default_factory=set)
    entity_ids: set[str] = field(default_factory=set)
    entities_by_id: dict[str, SemanticEntitySpec] = field(default_factory=dict)


def _validate_polygons(state: _InputValidation) -> None:
    for polygon in state.build_input.polygons:
        if not polygon.polygon_id:
            state.errors.append("polygon_id must be non-empty")
        elif polygon.polygon_id in state.polygon_ids:
            state.errors.append(f"duplicate polygon_id: {polygon.polygon_id}")
        state.polygon_ids.add(polygon.polygon_id)
        if polygon.exterior and len(polygon.exterior) < 3:
            state.errors.append(
                f"{polygon.polygon_id} exterior requires at least 3 points"
            )


def _validate_entities(state: _InputValidation) -> None:
    for entity in state.build_input.entities:
        if not entity.semantic_id:
            state.errors.append("semantic_id must be non-empty")
        elif entity.semantic_id in state.entity_ids:
            state.errors.append(f"duplicate semantic_id: {entity.semantic_id}")
        state.entity_ids.add(entity.semantic_id)
        if not entity.material_id:
            state.errors.append(f"{entity.semantic_id} material_id must be non-empty")
        if entity.material_kind not in {"vacuum", "dielectric", "conductor"}:
            state.errors.append(
                f"{entity.semantic_id} material_kind must be vacuum, dielectric, or conductor"
            )
        if entity.role == "solution_region":
            if entity.material_kind not in {"vacuum", "dielectric"}:
                state.errors.append(
                    f"{entity.semantic_id} solution_region must be vacuum or dielectric"
                )
        elif entity.material_kind != "conductor":
            state.errors.append(
                f"{entity.semantic_id} non-solution entity must be conductor"
            )
        if entity.material_kind == "conductor" and (
            not isinstance((host_id := entity.host_void_semantic_id), str)
            or not host_id
        ):
            state.errors.append(
                f"{entity.semantic_id} conductor needs explicit host_void_semantic_id"
            )
        state.entities_by_id[entity.semantic_id] = entity
        for polygon_id in entity.polygon_ids:
            if polygon_id not in state.polygon_ids:
                state.errors.append(
                    f"{entity.semantic_id} references unknown polygon_id: {polygon_id}"
                )


def _validate_attachments(state: _InputValidation) -> None:
    for entity in state.build_input.entities:
        if entity.material_kind == "conductor" and entity.host_void_semantic_id:
            host = state.entities_by_id.get(entity.host_void_semantic_id)
            if host is None or not is_solution_entity(host):
                state.errors.append(
                    f"{entity.semantic_id} host_void_semantic_id must resolve a solution_region"
                )
        if entity.part_role != "contact_pad":
            continue
        try:
            resolve_contact_pad_attachment(
                entity,
                state.build_input.entities,
                defer_overlap=bool(
                    state.build_input.boundary_curves
                    or state.build_input.boundary_reconstruction
                ),
            )
        except ValueError as exc:
            state.errors.append(str(exc))


def _validate_port_regions(state: _InputValidation) -> None:
    port_sheet_ids: set[str] = set()
    for region in state.build_input.port_sheet_regions:
        if not region.port_sheet_id:
            state.errors.append("port_sheet_id must be non-empty")
        elif region.port_sheet_id in port_sheet_ids:
            state.errors.append(f"duplicate port_sheet_id: {region.port_sheet_id}")
        port_sheet_ids.add(region.port_sheet_id)
        if region.metadata.get("source") != "palace_lumped_port_sheet":
            state.errors.append(
                f"{region.port_sheet_id} must declare source 'palace_lumped_port_sheet'"
            )
        for key in ("source_name", "target_layer", "direction_sign_convention"):
            if (
                not isinstance(region.metadata.get(key), str)
                or not region.metadata[key]
            ):
                state.errors.append(f"{region.port_sheet_id} requires {key}")
        port_index = region.metadata.get("port_index")
        if (
            isinstance(port_index, bool)
            or not isinstance(port_index, int)
            or port_index < 1
        ):
            state.errors.append(f"{region.port_sheet_id} requires 1-based port_index")
        _validate_port_direction(state, region)
        _validate_port_overlaps(state, region)


def _validate_port_direction(
    state: _InputValidation, region: PortSheetRegionRecord
) -> None:
    direction = region.metadata.get("direction")
    direction_raw = region.metadata.get("direction_raw")
    if (
        isinstance(direction, str | bytes)
        or not isinstance(direction, Sequence)
        or len(direction) != 3
    ):
        state.errors.append(f"{region.port_sheet_id} requires a 3D direction")
    else:
        try:
            direction_values = tuple((float(value) for value in direction))
        except (TypeError, ValueError):
            state.errors.append(f"{region.port_sheet_id} direction must be numeric")
        else:
            if (
                not all((isfinite(value) for value in direction_values))
                or direction_values[2] != 0.0
                or direction_values[0] == direction_values[1] == 0.0
            ):
                state.errors.append(
                    f"{region.port_sheet_id} direction must be finite, XY, and nonzero"
                )
    if (
        isinstance(direction_raw, str | bytes)
        or not isinstance(direction_raw, Sequence)
        or len(direction_raw) != 3
    ):
        state.errors.append(f"{region.port_sheet_id} requires raw direction provenance")
    if len(region.exterior) < 3:
        state.errors.append(f"{region.port_sheet_id} exterior requires 3 points")


def _validate_port_overlaps(
    state: _InputValidation, region: PortSheetRegionRecord
) -> None:
    for overlap in region.overlaps:
        if overlap.port_polygon_id != region.source_polygon_id:
            state.errors.append(
                f"{overlap.overlap_id} port polygon {overlap.port_polygon_id!r} does not match region source {region.source_polygon_id!r}"
            )
        if overlap.operation != "local_fragment_required":
            state.errors.append(
                f"{overlap.overlap_id} unsupported port-sheet operation {overlap.operation!r}"
            )
        if overlap.port_sheet_id != region.port_sheet_id:
            state.errors.append(
                f"{overlap.overlap_id} references port sheet {overlap.port_sheet_id!r}, expected {region.port_sheet_id!r}"
            )
        if overlap.host_semantic_id not in state.entity_ids:
            state.errors.append(
                f"{overlap.overlap_id} references unknown host semantic id {overlap.host_semantic_id}"
            )
        else:
            host_entity = state.entities_by_id[overlap.host_semantic_id]
            host_is_solver_geometry = not is_solution_entity(
                host_entity
            ) and requires_route_representation(host_entity)
            if not host_is_solver_geometry:
                state.errors.append(
                    f"{overlap.overlap_id} host {overlap.host_semantic_id} is not solver-relevant geometry"
                )
            if overlap.host_polygon_id not in host_entity.polygon_ids:
                state.errors.append(
                    f"{overlap.overlap_id} host polygon {overlap.host_polygon_id} is not owned by {overlap.host_semantic_id}"
                )
        if overlap.host_polygon_id not in state.polygon_ids:
            state.errors.append(
                f"{overlap.overlap_id} references unknown host polygon id {overlap.host_polygon_id}"
            )
        if len(overlap.overlap_loop) < 3:
            state.errors.append(f"{overlap.overlap_id} overlap_loop requires 3 points")
