"""Compute conductor contact geometry, signatures and source-bound metadata."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from scgsim.geometry.grid.entities import unique_ids
from scgsim.geometry.grid.grid import boolean_polygon_region
from scgsim.geometry.grid.loops import canonical_loop_sort_key, loop_signature
from scgsim.geometry.grid.planar_loops import clean_loop, polygon_area, ring_edges
from scgsim.geometry.grid.spatial import coordinate_z_key, same_coordinate_z
from scgsim.geometry.interpretation.representation import (
    planned_conductor_representation,
)
from scgsim.geometry.models.common import RouteLiteral
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.representation import (
    EntityRepresentation,
    RepresentationContext,
)
from scgsim.geometry.models.topology import InterfacePlanRecord
from scgsim.geometry.planning.domain_planning.faces import (
    route_a_sheet_boundary_volume_ids_from_solutions,
    route_a_sheet_plane_z_um_from_solutions,
)
from scgsim.geometry.planning.domain_planning.occupancy import required_host_solution_id


@dataclass(frozen=True)
class ContactPatch:
    """Carry one ordered pair's contact loop, plane and deterministic index."""

    lower: SemanticEntitySpec
    upper: SemanticEntitySpec
    contact_z_um: float
    contact_loop: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class IndexedContactPatch(ContactPatch):
    """Add the deterministic contact index to one recognized patch."""

    contact_index: int


def same_declared_effective_sheet_plane(
    lower: SemanticEntitySpec,
    upper: SemanticEntitySpec,
    *,
    representation_context: RepresentationContext | None,
) -> bool:
    """Identify one declared source film, without inferring layer or connectivity."""
    if representation_context is None:
        return False
    records = tuple(
        (
            representation_context.entities.get(entity.semantic_id)
            for entity in (lower, upper)
        )
    )
    if any((record is None for record in records)):
        return False
    left, right = cast(tuple[EntityRepresentation, EntityRepresentation], records)
    identity = left.physical_layer_id
    if (
        not identity
        or identity != right.physical_layer_id
        or left.representation != "sheet"
        or (right.representation != "sheet")
        or (not lower.net_id)
        or (lower.net_id != upper.net_id)
    ):
        return False
    return (
        left.source_z_interval_um == right.source_z_interval_um
        and left.effective_z_interval_um == right.effective_z_interval_um
        and (left.effective_z_interval_um[0] == left.effective_z_interval_um[1])
    )


def sheet_loops_share_positive_edge(
    lower: SemanticEntitySpec, upper: SemanticEntitySpec
) -> bool:
    """Exact positive-length lateral incidence, excluding a point-only touch."""
    lower_loops = (lower.geometry["outer_loop"], *lower.geometry.get("hole_loops", ()))
    upper_loops = (upper.geometry["outer_loop"], *upper.geometry.get("hole_loops", ()))
    for left in lower_loops:
        for right in upper_loops:
            for start, end in ring_edges(left):
                dx, dy = (end[0] - start[0], end[1] - start[1])
                length_sq = dx * dx + dy * dy
                if length_sq == 0:
                    continue
                for first, second in ring_edges(right):
                    if any(
                        (
                            (point[0] - start[0]) * dy - (point[1] - start[1]) * dx != 0
                            for point in (first, second)
                        )
                    ):
                        continue
                    parameters = tuple(
                        (
                            ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy)
                            / length_sq
                            for point in (first, second)
                        )
                    )
                    if min(1.0, max(parameters)) > max(0.0, min(parameters)):
                        return True
    return False


def route_c_contact_patch_metadata(
    build_input: GeometryBuildInput,
    *,
    solution_entities: Sequence[SemanticEntitySpec],
    patch: ContactPatch,
) -> Mapping[str, Any]:
    """Exact pre-A/B Route-C contact metadata shape."""
    del build_input, solution_entities
    return {
        "recognition_rule": "coplanar_conductor_contact_patch",
        "contact_policy": "retained_material_contact",
        "export_surface": True,
        "lower_entity_id": patch.lower.semantic_id,
        "upper_entity_id": patch.upper.semantic_id,
        "lower_face": "top",
        "upper_face": "bottom",
        "contact_z_um": patch.contact_z_um,
        "contact_plane": {"axis": "z", "value_um": patch.contact_z_um},
        "plane": {"axis": "z", "value_um": patch.contact_z_um},
        "outer_loop": patch.contact_loop,
        "hole_loops": (),
        "interface_kinds": ("MM",),
        "surface_owner_semantic_ids": (
            patch.lower.semantic_id,
            patch.upper.semantic_id,
        ),
        "boundary_volume_ids": (patch.lower.semantic_id, patch.upper.semantic_id),
        "valid_routes": ("C",),
    }


def source_layer_provenance(entity: SemanticEntitySpec) -> dict[str, Any]:
    """Exact authored source layer identity; no layer/datatype collapsing."""
    return {
        "source_layer_name": entity.metadata.get("source_layer_name"),
        "gds_layer": entity.geometry.get("gds_layer"),
        "gds_datatype": entity.geometry.get("gds_datatype"),
    }


type ContactSignature = (
    tuple[()] | tuple[str, str, float, tuple[tuple[float, float], ...]]
)


def contact_signature(interface: InterfacePlanRecord) -> ContactSignature:
    """Return the canonical source identity of an existing conductor contact."""
    if interface.recognition_rule != "coplanar_conductor_contact_patch":
        return ()
    return (
        str(interface.metadata.get("lower_entity_id", "")),
        str(interface.metadata.get("upper_entity_id", "")),
        coordinate_z_key(float(interface.metadata.get("contact_z_um", 0.0))),
        loop_signature(interface.metadata.get("outer_loop", ())),
    )


def contact_patch_loops(
    overlap_region: Sequence[Any], lower_id: str, upper_id: str
) -> tuple[tuple[tuple[float, float], ...], ...]:
    """Extract and deterministically order finite conductor contact loops."""
    import gdstk  # noqa: PLC0415 -- defer the optional polygon engine until this operation reaches native geometry.

    polygons = boolean_polygon_region(gdstk, overlap_region, (), "or")
    if not polygons:
        raise ValueError(
            f"{lower_id} and {upper_id} contact must produce at least one polygon, got {len(polygons)}"
        )
    loops: list[tuple[tuple[float, float], ...]] = []
    for polygon in polygons:
        loop = clean_loop(polygon.points)
        loops.append(loop)
    return _sort_contact_loops(tuple(loops))


def single_rectangular_contact_loop(
    overlap_region: Sequence[Any], lower_id: str, upper_id: str
) -> tuple[tuple[float, float], ...]:
    """Route-C legacy contact recognizer: exactly one rectangular patch."""
    import gdstk  # noqa: PLC0415 -- defer the optional polygon engine until this operation reaches native geometry.

    polygons = boolean_polygon_region(gdstk, overlap_region, (), "or")
    if len(polygons) != 1:
        raise ValueError(
            f"{lower_id} and {upper_id} contact must produce one polygon, got {len(polygons)}"
        )
    loop = clean_loop(polygons[0].points)
    if not _is_axis_aligned_rectangle(loop):
        raise ValueError(
            f"{lower_id} and {upper_id} contact is not a rectangular patch"
        )
    return loop


def _is_axis_aligned_rectangle(loop: tuple[tuple[float, float], ...]) -> bool:
    return len(loop) == 4 and all(
        (
            same_coordinate_z(start[0], end[0]) or same_coordinate_z(start[1], end[1])
            for start, end in ring_edges(loop)
        )
    )


def _sort_contact_loops(
    loops: tuple[tuple[tuple[float, float], ...], ...],
) -> tuple[tuple[tuple[float, float], ...], ...]:

    def _stable_loop_key(loop: tuple[tuple[float, float], ...]) -> tuple:
        clean = clean_loop(loop)
        xs = tuple((point[0] for point in clean))
        ys = tuple((point[1] for point in clean))
        canonical = canonical_loop_sort_key(clean)
        return (
            min(xs),
            max(xs),
            min(ys),
            max(ys),
            round(abs(polygon_area(clean)), 12),
            canonical,
        )

    return tuple(
        sorted(
            {loop_signature(loop): loop for loop in loops}.values(),
            key=_stable_loop_key,
        )
    )


def contact_patch_metadata(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral | None,
    solution_entities: Sequence[SemanticEntitySpec],
    patch: IndexedContactPatch,
    representation_context: RepresentationContext | None = None,
) -> Mapping[str, Any]:
    """Bind one indexed contact patch to its route and incident solution volumes."""
    plane_z_um = patch.contact_z_um
    boundary_volume_ids: tuple[str, ...]
    interface_kinds: tuple[str, ...] = ("MM",)
    export_surface = route != "B"
    if route == "A" or (route is None and representation_context is not None):
        sheet = next(
            (
                entity
                for entity in (patch.lower, patch.upper)
                if planned_conductor_representation(
                    entity, route=route, context=representation_context
                )
                == "surface_sheet"
            ),
            None,
        )
        if sheet is not None:
            plane_z_um = route_a_sheet_plane_z_um_from_solutions(
                sheet, solution_entities
            )
            boundary_volume_ids = route_a_sheet_boundary_volume_ids_from_solutions(
                sheet, solution_entities
            )
            interface_kinds = ("MM", "MS")
        else:
            boundary_volume_ids = ()
            export_surface = False
    elif route == "C":
        boundary_volume_ids = (patch.lower.semantic_id, patch.upper.semantic_id)
    else:
        boundary_volume_ids = unique_ids(
            (
                required_host_solution_id(build_input, patch.lower),
                required_host_solution_id(build_input, patch.upper),
            )
        )
    contact_policy = (
        "representation_context_contact_patch"
        if route is None
        else {
            "A": "sheet_contact_patch",
            "B": "hidden_cutout_contact",
            "C": "retained_material_contact",
        }[route]
    )
    interface_type = "_".join(interface_kinds)
    return {
        "recognition_rule": "coplanar_conductor_contact_patch",
        "contact_policy": contact_policy,
        "export_surface": export_surface,
        "contact_kind": "MM",
        "contact_index": patch.contact_index,
        "lower_entity_id": patch.lower.semantic_id,
        "upper_entity_id": patch.upper.semantic_id,
        "lower_face": "top",
        "upper_face": "bottom",
        "contact_z_um": patch.contact_z_um,
        "face_kinds": ("top", "bottom"),
        "route": route,
        "contact_representation": "circuit_contact_patch",
        "contact_id": f"{patch.lower.semantic_id}__{patch.upper.semantic_id}__{contact_policy}__{patch.contact_index:04d}",
        "contact_plane": {"axis": "z", "value_um": plane_z_um},
        "plane": {"axis": "z", "value_um": plane_z_um},
        "outer_loop": patch.contact_loop,
        "hole_loops": (),
        "interface_type": interface_type,
        "interface_kinds": interface_kinds,
        "surface_owner_semantic_ids": (
            patch.lower.semantic_id,
            patch.upper.semantic_id,
        ),
        "boundary_volume_ids": boundary_volume_ids,
        "conductor_component_id": f"{patch.lower.semantic_id}__{patch.upper.semantic_id}",
        "valid_routes": (route,),
    }
