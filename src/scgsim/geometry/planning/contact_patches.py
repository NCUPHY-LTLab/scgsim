"""Plan the contact patches where two conductors overlap in the xy plane.

A contact patch is the positive-area loop shared by two touching conductors. Owns the
patch loops and their metadata; the metal-metal (MM) contact records built from them
live in ``mm_contacts``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

from scgsim.geometry._primitives.entities import _entity_z_range_um, _unique_ids
from scgsim.geometry._primitives.loops import (
    _boolean_gdstk_region,
    _canonical_loop_sort_key,
    _clean_loop,
    _loop_signature,
    _loops_touch_without_area,
    _polygon_area,
    _ring_edges,
)
from scgsim.geometry._primitives.spatial import _bounds_overlap, _same_z, _z_key
from scgsim.geometry.models.common import RouteLiteral
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.models.topology import InterfacePlanRecord
from scgsim.geometry.planning.conductor_faces import (
    _active_route_conductor_entities,
    _route_conductor_contact_footprints,
)
from scgsim.geometry.planning.representation import planned_conductor_representation
from scgsim.geometry.planning.solution_adjacency import (
    _required_host_solution_id,
    _route_a_sheet_boundary_volume_ids_from_solutions,
    _route_a_sheet_plane_z_um_from_solutions,
)
from scgsim.geometry.planning.solution_geometry import (
    _entity_loop_bounds,
    _entity_occupied_region,
    _solution_entities,
)


def plan_conductor_contact_patches(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral | None,
    interfaces: tuple[InterfacePlanRecord, ...],
    representation_context: RepresentationContext | None = None,
) -> tuple[InterfacePlanRecord, ...]:
    """Recognize coplanar conductor contact patches from planned conductors.

    This is the small v1 contact planner: it partitions each opposite-face
    overlap into connected conductor islands and emits one interface per
    validated contact component.

    Two conductors are in contact where the top face of one and the bottom face of
    the other lie at the same z and their xy footprints overlap with positive area.
    """
    if route == "C":
        return _plan_route_c_conductor_contact_patches(
            build_input, route=route, interfaces=interfaces
        )
    import gdstk

    generated: list[InterfacePlanRecord] = []
    seen = {_contact_signature(interface) for interface in interfaces}
    solution_entities = _solution_entities(build_input)
    top_faces, bottom_faces = _conductor_faces_by_z(
        gdstk,
        _route_conductor_contact_footprints(
            build_input, route, representation_context=representation_context
        ),
    )

    index = 0
    for z_key, lower_faces in top_faces.items():
        for lower_face in lower_faces:
            lower = lower_face.entity
            for upper_face in bottom_faces.get(z_key, ()):
                upper = upper_face.entity
                if lower.semantic_id == upper.semantic_id:
                    continue
                overlap_region = _contact_overlap_region(
                    gdstk,
                    lower_face,
                    upper_face,
                    route=route,
                    representation_context=representation_context,
                )
                if overlap_region is None:
                    continue
                contact_loops = _contact_patch_loops(
                    overlap_region,
                    lower.semantic_id,
                    upper.semantic_id,
                )
                # When the upper footprint is wholly the contact patch, keep
                # its authored ring verbatim.  The lower top hole and upper
                # authored sidewall then share canonical curves without
                # replacing either body's sidewall geometry.
                upper_remainder = _boolean_gdstk_region(
                    gdstk, upper_face.region, overlap_region, "not"
                )
                if not upper_remainder and not upper.geometry.get("hole_loops"):
                    contact_loops = (_clean_loop(upper.geometry["outer_loop"]),)
                for contact_loop in contact_loops:
                    signature = (
                        lower.semantic_id,
                        upper.semantic_id,
                        _z_key(float(z_key)),
                        _loop_signature(contact_loop),
                    )
                    if signature in seen:
                        continue
                    seen.add(signature)
                    metadata = _contact_patch_metadata(
                        _ContactPatchScope(
                            build_input,
                            route,
                            solution_entities,
                            representation_context,
                        ),
                        _ContactCandidate(
                            lower, upper, float(z_key), contact_loop, index
                        ),
                    )
                    generated.append(
                        InterfacePlanRecord(
                            interface_id=(
                                f"MM__CONTACT__{lower.semantic_id}__"
                                f"{upper.semantic_id}__{index:04d}"
                            ),
                            kind="MM",
                            owner_semantic_ids=(lower.semantic_id, upper.semantic_id),
                            recognition_rule="coplanar_conductor_contact_patch",
                            source_polygon_ids=(
                                *lower.polygon_ids,
                                *upper.polygon_ids,
                            ),
                            metadata=metadata,
                        )
                    )
                    index += 1
    return (*interfaces, *generated)


class _ConductorFace(NamedTuple):
    """One horizontal conductor face: its entity, occupied xy region and bounding box."""

    entity: SemanticEntitySpec
    region: Any
    bounds: Mapping[str, float]


def _conductor_faces_by_z(
    gdstk: Any, entities: Iterable[SemanticEntitySpec]
) -> tuple[dict[float, list[_ConductorFace]], dict[float, list[_ConductorFace]]]:
    """Index conductor top faces and bottom faces by their z, in entity order.

    Returns ``(top_faces, bottom_faces)``. A top face and a bottom face at the same z
    key are candidates for a contact.
    """
    top_faces: dict[float, list[_ConductorFace]] = {}
    bottom_faces: dict[float, list[_ConductorFace]] = {}
    for entity in entities:
        region = _entity_occupied_region(gdstk, entity)
        if not region:
            continue
        face = _ConductorFace(entity, region, _entity_loop_bounds(entity))
        z_min_um, z_max_um = _entity_z_range_um(entity)
        bottom_faces.setdefault(_z_key(z_min_um), []).append(face)
        top_faces.setdefault(_z_key(z_max_um), []).append(face)
    return top_faces, bottom_faces


def _contact_overlap_region(
    gdstk: Any,
    lower_face: _ConductorFace,
    upper_face: _ConductorFace,
    *,
    route: RouteLiteral | None,
    representation_context: RepresentationContext | None,
) -> Any | None:
    """Return the xy overlap of two faces at the same z, or None when they do not contact.

    Touching along an edge or at a point, without area, raises an error.
    """
    lower, upper = lower_face.entity, upper_face.entity
    overlap_region = None
    if (
        route is None
        and representation_context is not None
        and _same_declared_effective_sheet_plane(
            lower,
            upper,
            representation_context=representation_context,
        )
    ):
        overlap_region = _boolean_gdstk_region(
            gdstk, lower_face.region, upper_face.region, "and"
        )
        # One collapsed source film may have a lateral seam between
        # distinct same-Net bodies. It has no horizontal MM area;
        # keep both source bodies and their actual boundary rings.
        if not overlap_region and _sheet_loops_share_positive_edge(lower, upper):
            return None
    if not _bounds_overlap(lower_face.bounds, upper_face.bounds):
        _reject_edge_or_point_contact(lower, upper)
        return None
    if overlap_region is None:
        overlap_region = _boolean_gdstk_region(
            gdstk,
            lower_face.region,
            upper_face.region,
            "and",
        )
    if not overlap_region:
        _reject_edge_or_point_contact(lower, upper)
        return None
    return overlap_region


def _reject_edge_or_point_contact(
    lower: SemanticEntitySpec, upper: SemanticEntitySpec
) -> None:
    """Raise when the two outlines touch without sharing any area."""
    if _loops_touch_without_area(
        lower.geometry["outer_loop"], upper.geometry["outer_loop"]
    ):
        raise ValueError(
            f"{lower.semantic_id}/{upper.semantic_id} has "
            "edge/point-only conductor contact"
        )


def _plan_route_c_conductor_contact_patches(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    interfaces: tuple[InterfacePlanRecord, ...],
) -> tuple[InterfacePlanRecord, ...]:
    """Preserve the pre-A/B Route-C rectangular-contact behavior exactly."""
    import gdstk

    generated: list[InterfacePlanRecord] = []
    seen = {_contact_signature(interface) for interface in interfaces}
    top_faces, bottom_faces = _conductor_faces_by_z(
        gdstk, _active_route_conductor_entities(build_input, route)
    )

    index = 0
    for z_key, lower_faces in top_faces.items():
        for lower_face in lower_faces:
            lower = lower_face.entity
            for upper_face in bottom_faces.get(z_key, ()):
                upper = upper_face.entity
                if lower.semantic_id == upper.semantic_id or not _bounds_overlap(
                    lower_face.bounds, upper_face.bounds
                ):
                    continue
                overlap_region = _boolean_gdstk_region(
                    gdstk, lower_face.region, upper_face.region, "and"
                )
                if not overlap_region:
                    continue
                contact_loop = _single_rectangular_contact_loop(
                    overlap_region, lower.semantic_id, upper.semantic_id
                )
                signature = (
                    lower.semantic_id,
                    upper.semantic_id,
                    _z_key(float(z_key)),
                    _loop_signature(contact_loop),
                )
                if signature in seen:
                    continue
                seen.add(signature)
                metadata = _route_c_contact_patch_metadata(
                    lower=lower,
                    upper=upper,
                    contact_z_um=float(z_key),
                    contact_loop=contact_loop,
                )
                generated.append(
                    InterfacePlanRecord(
                        interface_id=(
                            f"MM__CONTACT__{lower.semantic_id}__"
                            f"{upper.semantic_id}__{index:04d}"
                        ),
                        kind="MM",
                        owner_semantic_ids=(lower.semantic_id, upper.semantic_id),
                        recognition_rule="coplanar_conductor_contact_patch",
                        source_polygon_ids=(*lower.polygon_ids, *upper.polygon_ids),
                        metadata=metadata,
                    )
                )
                index += 1
    return (*interfaces, *generated)


def _route_c_contact_patch_metadata(
    *,
    lower: SemanticEntitySpec,
    upper: SemanticEntitySpec,
    contact_z_um: float,
    contact_loop: tuple[tuple[float, float], ...],
) -> Mapping[str, Any]:
    """Exact pre-A/B Route-C contact metadata shape."""
    return {
        "recognition_rule": "coplanar_conductor_contact_patch",
        "contact_policy": "retained_material_contact",
        "export_surface": True,
        "lower_entity_id": lower.semantic_id,
        "upper_entity_id": upper.semantic_id,
        "lower_face": "top",
        "upper_face": "bottom",
        "contact_z_um": contact_z_um,
        "contact_plane": {"axis": "z", "value_um": contact_z_um},
        "plane": {"axis": "z", "value_um": contact_z_um},
        "outer_loop": contact_loop,
        "hole_loops": (),
        "interface_kinds": ("MM",),
        "surface_owner_semantic_ids": (lower.semantic_id, upper.semantic_id),
        "boundary_volume_ids": (lower.semantic_id, upper.semantic_id),
        "valid_routes": ("C",),
    }


def _same_declared_effective_sheet_plane(
    lower: SemanticEntitySpec,
    upper: SemanticEntitySpec,
    *,
    representation_context: RepresentationContext | None,
) -> bool:
    """Identify one declared source film, without inferring layer or connectivity."""
    if representation_context is None:
        return False
    records = tuple(
        representation_context.entities.get(entity.semantic_id)
        for entity in (lower, upper)
    )
    if any(record is None for record in records):
        return False
    left, right = records
    identity = left.physical_layer_id
    if (
        not identity
        or identity != right.physical_layer_id
        or left.representation != "sheet"
        or right.representation != "sheet"
        or not lower.net_id
        or lower.net_id != upper.net_id
    ):
        return False
    return (
        left.source_z_interval_um == right.source_z_interval_um
        and left.effective_z_interval_um == right.effective_z_interval_um
        and left.effective_z_interval_um[0] == left.effective_z_interval_um[1]
    )


def _sheet_loops_share_positive_edge(
    lower: SemanticEntitySpec, upper: SemanticEntitySpec
) -> bool:
    """Exact positive-length lateral incidence, excluding a point-only touch."""
    lower_loops = (lower.geometry["outer_loop"], *lower.geometry.get("hole_loops", ()))
    upper_loops = (upper.geometry["outer_loop"], *upper.geometry.get("hole_loops", ()))
    for left in lower_loops:
        for right in upper_loops:
            for start, end in _ring_edges(left):
                dx, dy = end[0] - start[0], end[1] - start[1]
                length_sq = dx * dx + dy * dy
                if length_sq == 0:
                    continue
                for first, second in _ring_edges(right):
                    if any(
                        (point[0] - start[0]) * dy - (point[1] - start[1]) * dx != 0
                        for point in (first, second)
                    ):
                        continue
                    parameters = tuple(
                        ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy)
                        / length_sq
                        for point in (first, second)
                    )
                    if min(1.0, max(parameters)) > max(0.0, min(parameters)):
                        return True
    return False


def _contact_signature(interface: InterfacePlanRecord) -> tuple[Any, ...]:
    if interface.recognition_rule != "coplanar_conductor_contact_patch":
        return ()
    return (
        str(interface.metadata.get("lower_entity_id", "")),
        str(interface.metadata.get("upper_entity_id", "")),
        _z_key(float(interface.metadata.get("contact_z_um", 0.0))),
        _loop_signature(interface.metadata.get("outer_loop", ())),
    )


def _contact_patch_loops(
    overlap_region: Sequence[Any],
    lower_id: str,
    upper_id: str,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    import gdstk

    polygons = _boolean_gdstk_region(gdstk, overlap_region, (), "or")
    if not polygons:
        raise ValueError(
            f"{lower_id} and {upper_id} contact must produce at least one "
            f"polygon, got {len(polygons)}"
        )
    loops: list[tuple[tuple[float, float], ...]] = []
    for polygon in polygons:
        loop = _clean_loop(polygon.points)
        loops.append(loop)
    return _sort_contact_loops(tuple(loops))


def _single_rectangular_contact_loop(
    overlap_region: Sequence[Any],
    lower_id: str,
    upper_id: str,
) -> tuple[tuple[float, float], ...]:
    """Route-C legacy contact recognizer: exactly one rectangular patch."""
    import gdstk

    polygons = _boolean_gdstk_region(gdstk, overlap_region, (), "or")
    if len(polygons) != 1:
        raise ValueError(
            f"{lower_id} and {upper_id} contact must produce one polygon, got "
            f"{len(polygons)}"
        )
    loop = _clean_loop(polygons[0].points)
    if not _is_axis_aligned_rectangle(loop):
        raise ValueError(
            f"{lower_id} and {upper_id} contact is not a rectangular patch"
        )
    return loop


def _is_axis_aligned_rectangle(loop: tuple[tuple[float, float], ...]) -> bool:
    return len(loop) == 4 and all(
        _same_z(start[0], end[0]) or _same_z(start[1], end[1])
        for start, end in _ring_edges(loop)
    )


def _sort_contact_loops(
    loops: tuple[tuple[tuple[float, float], ...], ...],
) -> tuple[tuple[tuple[float, float], ...], ...]:
    def _stable_loop_key(loop: tuple[tuple[float, float], ...]) -> tuple:
        clean = _clean_loop(loop)
        xs = tuple(point[0] for point in clean)
        ys = tuple(point[1] for point in clean)
        canonical = _canonical_loop_sort_key(clean)
        return (
            min(xs),
            max(xs),
            min(ys),
            max(ys),
            round(abs(_polygon_area(clean)), 12),
            canonical,
        )

    return tuple(
        sorted(
            {_loop_signature(loop): loop for loop in loops}.values(),
            key=_stable_loop_key,
        )
    )


@dataclass(frozen=True)
class _ContactPatchScope:
    """The build input and route a contact patch is planned under."""

    build_input: GeometryBuildInput
    route: RouteLiteral | None
    solution_entities: Sequence[SemanticEntitySpec]
    representation_context: RepresentationContext | None = None


@dataclass(frozen=True)
class _ContactCandidate:
    """One accepted contact: the touching conductors, their shared Z and loop."""

    lower: SemanticEntitySpec
    upper: SemanticEntitySpec
    contact_z_um: float
    contact_loop: tuple[tuple[float, float], ...]
    contact_index: int


def _contact_patch_metadata(
    scope: _ContactPatchScope, candidate: _ContactCandidate
) -> Mapping[str, Any]:
    build_input = scope.build_input
    route = scope.route
    solution_entities = scope.solution_entities
    representation_context = scope.representation_context
    lower = candidate.lower
    upper = candidate.upper
    contact_z_um = candidate.contact_z_um
    contact_loop = candidate.contact_loop
    contact_index = candidate.contact_index
    plane_z_um = contact_z_um
    boundary_volume_ids: tuple[str, ...]
    interface_kinds: tuple[str, ...] = ("MM",)
    export_surface = route != "B"

    if route == "A" or (route is None and representation_context is not None):
        sheet = next(
            (
                entity
                for entity in (lower, upper)
                if planned_conductor_representation(
                    entity, route=route, context=representation_context
                )
                == "surface_sheet"
            ),
            None,
        )
        if sheet is not None:
            plane_z_um = _route_a_sheet_plane_z_um_from_solutions(
                sheet,
                solution_entities,
            )
            boundary_volume_ids = _route_a_sheet_boundary_volume_ids_from_solutions(
                sheet,
                solution_entities,
            )
            interface_kinds = ("MM", "MS")
        else:
            boundary_volume_ids = ()
            export_surface = False
    elif route == "C":
        boundary_volume_ids = (lower.semantic_id, upper.semantic_id)
    else:
        boundary_volume_ids = _unique_ids(
            (
                _required_host_solution_id(build_input, lower),
                _required_host_solution_id(build_input, upper),
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
        "contact_index": contact_index,
        "lower_entity_id": lower.semantic_id,
        "upper_entity_id": upper.semantic_id,
        "lower_face": "top",
        "upper_face": "bottom",
        "contact_z_um": contact_z_um,
        "face_kinds": ("top", "bottom"),
        "route": route,
        "contact_representation": "circuit_contact_patch",
        "contact_id": (
            f"{lower.semantic_id}__{upper.semantic_id}__"
            f"{contact_policy}__{contact_index:04d}"
        ),
        "contact_plane": {"axis": "z", "value_um": plane_z_um},
        "plane": {"axis": "z", "value_um": plane_z_um},
        "outer_loop": contact_loop,
        "hole_loops": (),
        "interface_type": interface_type,
        "interface_kinds": interface_kinds,
        "surface_owner_semantic_ids": (lower.semantic_id, upper.semantic_id),
        "boundary_volume_ids": boundary_volume_ids,
        "conductor_component_id": f"{lower.semantic_id}__{upper.semantic_id}",
        "valid_routes": (route,),
    }
