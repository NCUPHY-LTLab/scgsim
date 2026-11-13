"""Recognize ordered coplanar conductor contact patches for each route."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import ModuleType

from scgsim.geometry.grid.entities import entity_z_range_um
from scgsim.geometry.grid.grid import boolean_polygon_region
from scgsim.geometry.grid.loops import loop_signature, loops_touch_without_area
from scgsim.geometry.grid.planar_loops import PolygonCoordinates, clean_loop
from scgsim.geometry.grid.spatial import bounds_overlap, coordinate_z_key
from scgsim.geometry.models.common import RouteLiteral
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.models.topology import InterfacePlanRecord
from scgsim.geometry.planning.contact_planning.geometry import (
    ContactPatch,
    ContactSignature,
    IndexedContactPatch,
    contact_patch_loops,
    contact_patch_metadata,
    contact_signature,
    route_c_contact_patch_metadata,
    same_declared_effective_sheet_plane,
    sheet_loops_share_positive_edge,
    single_rectangular_contact_loop,
)
from scgsim.geometry.planning.domain_planning.occupancy import (
    active_route_conductor_entities,
    entity_loop_bounds,
    entity_occupied_region,
    route_conductor_contact_footprints,
    solution_region_entities,
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
    """
    if route == "C":
        return _plan_route_c_conductor_contact_patches(
            build_input,
            route=route,
            interfaces=interfaces,
            representation_context=representation_context,
        )
    import gdstk  # noqa: PLC0415 -- defer the optional polygon engine until this operation reaches native geometry.

    return _ContactPatchPlanner(
        build_input, route, interfaces, representation_context, gdstk
    ).build()


type ContactFace = tuple[
    SemanticEntitySpec, tuple[PolygonCoordinates, ...], Mapping[str, float]
]


@dataclass
class _ContactPatchPlanner:
    """Carry face indexes, deduplication and contact order through one route."""

    build_input: GeometryBuildInput
    route: RouteLiteral | None
    interfaces: tuple[InterfacePlanRecord, ...]
    representation_context: RepresentationContext | None
    gdstk: ModuleType
    generated: list[InterfacePlanRecord] = field(init=False)
    seen: set[ContactSignature] = field(init=False)
    solution_entities: tuple[SemanticEntitySpec, ...] = field(init=False)
    top_faces: dict[float, list[ContactFace]] = field(init=False)
    bottom_faces: dict[float, list[ContactFace]] = field(init=False)
    index: int = field(init=False)

    def build(self) -> tuple[InterfacePlanRecord, ...]:
        """Index conductor faces and emit their ordered contact patches."""
        self.index_faces()
        for z_key, lower_faces in self.top_faces.items():
            for lower_face in lower_faces:
                for upper_face in self.bottom_faces.get(z_key, ()):
                    self.plan_pair(lower_face, upper_face, z_key)
        return (*self.interfaces, *self.generated)

    def index_faces(self) -> None:
        """Collect occupied top and bottom faces in source order."""
        self.generated: list[InterfacePlanRecord] = []
        self.seen = {contact_signature(interface) for interface in self.interfaces}
        self.solution_entities = solution_region_entities(self.build_input)
        self.top_faces: dict[float, list[ContactFace]] = {}
        self.bottom_faces: dict[float, list[ContactFace]] = {}
        for entity in route_conductor_contact_footprints(
            self.build_input,
            self.route,
            representation_context=self.representation_context,
        ):
            region = entity_occupied_region(self.gdstk, entity)
            if not region:
                continue
            bounds = entity_loop_bounds(entity)
            z_min_um, z_max_um = entity_z_range_um(entity)
            self.bottom_faces.setdefault(coordinate_z_key(z_min_um), []).append(
                (entity, region, bounds)
            )
            self.top_faces.setdefault(coordinate_z_key(z_max_um), []).append(
                (entity, region, bounds)
            )
        self.index = 0

    def overlap_region(
        self, lower_face: ContactFace, upper_face: ContactFace
    ) -> tuple[PolygonCoordinates, ...] | None:
        """Resolve finite contact area while retaining source sheet seam behavior."""
        lower, lower_region, lower_bounds = lower_face
        upper, upper_region, upper_bounds = upper_face
        if lower.semantic_id == upper.semantic_id:
            return None
        overlap_region = None
        if (
            self.route is None
            and self.representation_context is not None
            and same_declared_effective_sheet_plane(
                lower, upper, representation_context=self.representation_context
            )
        ):
            overlap_region = boolean_polygon_region(
                self.gdstk, lower_region, upper_region, "and"
            )
            # A collapsed source film can have a same-net lateral seam without
            # horizontal contact area; retain both source bodies and their rings.
            if not overlap_region and sheet_loops_share_positive_edge(lower, upper):
                return None
        if not bounds_overlap(lower_bounds, upper_bounds):
            if loops_touch_without_area(
                lower.geometry["outer_loop"], upper.geometry["outer_loop"]
            ):
                raise ValueError(
                    f"{lower.semantic_id}/{upper.semantic_id} has edge/point-only conductor contact"
                )
            return None
        if overlap_region is None:
            overlap_region = boolean_polygon_region(
                self.gdstk, lower_region, upper_region, "and"
            )
        if not overlap_region:
            if loops_touch_without_area(
                lower.geometry["outer_loop"], upper.geometry["outer_loop"]
            ):
                raise ValueError(
                    f"{lower.semantic_id}/{upper.semantic_id} has edge/point-only conductor contact"
                )
            return None
        return overlap_region

    def plan_pair(
        self, lower_face: ContactFace, upper_face: ContactFace, z_key: float
    ) -> None:
        """Emit the unique contact loops for one ordered pair of faces."""
        lower, lower_region, lower_bounds = lower_face
        upper, upper_region, upper_bounds = upper_face
        overlap_region = self.overlap_region(lower_face, upper_face)
        if not overlap_region:
            return
        contact_loops = contact_patch_loops(
            overlap_region, lower.semantic_id, upper.semantic_id
        )
        # A wholly covered upper footprint keeps its authored ring so that its
        # sidewall and the lower top hole share the same canonical curves.
        upper_remainder = boolean_polygon_region(
            self.gdstk, upper_region, overlap_region, "not"
        )
        if not upper_remainder and (not upper.geometry.get("hole_loops")):
            contact_loops = (clean_loop(upper.geometry["outer_loop"]),)
        for contact_loop in contact_loops:
            signature = (
                lower.semantic_id,
                upper.semantic_id,
                coordinate_z_key(float(z_key)),
                loop_signature(contact_loop),
            )
            if signature in self.seen:
                continue
            self.seen.add(signature)
            metadata = contact_patch_metadata(
                self.build_input,
                route=self.route,
                representation_context=self.representation_context,
                solution_entities=self.solution_entities,
                patch=IndexedContactPatch(
                    lower=lower,
                    upper=upper,
                    contact_z_um=float(z_key),
                    contact_loop=contact_loop,
                    contact_index=self.index,
                ),
            )
            self.generated.append(
                InterfacePlanRecord(
                    interface_id=f"MM__CONTACT__{lower.semantic_id}__{upper.semantic_id}__{self.index:04d}",
                    kind="MM",
                    owner_semantic_ids=(lower.semantic_id, upper.semantic_id),
                    recognition_rule="coplanar_conductor_contact_patch",
                    source_polygon_ids=(*lower.polygon_ids, *upper.polygon_ids),
                    metadata=metadata,
                )
            )
            self.index += 1


def _plan_route_c_conductor_contact_patches(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    interfaces: tuple[InterfacePlanRecord, ...],
    representation_context: RepresentationContext | None = None,
) -> tuple[InterfacePlanRecord, ...]:
    """Preserve the pre-A/B Route-C rectangular-contact behavior exactly."""
    import gdstk  # noqa: PLC0415 -- defer the optional polygon engine until this operation reaches native geometry.

    generated: list[InterfacePlanRecord] = []
    seen = {contact_signature(interface) for interface in interfaces}
    solution_entities = solution_region_entities(build_input)
    top_faces: dict[float, list[ContactFace]] = {}
    bottom_faces: dict[float, list[ContactFace]] = {}
    for entity in active_route_conductor_entities(
        build_input, route, representation_context=representation_context
    ):
        region = entity_occupied_region(gdstk, entity)
        if not region:
            continue
        bounds = entity_loop_bounds(entity)
        z_min_um, z_max_um = entity_z_range_um(entity)
        bottom_faces.setdefault(coordinate_z_key(z_min_um), []).append(
            (entity, region, bounds)
        )
        top_faces.setdefault(coordinate_z_key(z_max_um), []).append(
            (entity, region, bounds)
        )
    index = 0
    for z_key, lower_faces in top_faces.items():
        for lower, lower_region, lower_bounds in lower_faces:
            for upper, upper_region, upper_bounds in bottom_faces.get(z_key, ()):
                if lower.semantic_id == upper.semantic_id or not bounds_overlap(
                    lower_bounds, upper_bounds
                ):
                    continue
                overlap_region = boolean_polygon_region(
                    gdstk, lower_region, upper_region, "and"
                )
                if not overlap_region:
                    continue
                contact_loop = single_rectangular_contact_loop(
                    overlap_region, lower.semantic_id, upper.semantic_id
                )
                signature = (
                    lower.semantic_id,
                    upper.semantic_id,
                    coordinate_z_key(float(z_key)),
                    loop_signature(contact_loop),
                )
                if signature in seen:
                    continue
                seen.add(signature)
                metadata = route_c_contact_patch_metadata(
                    build_input,
                    solution_entities=solution_entities,
                    patch=ContactPatch(
                        lower=lower,
                        upper=upper,
                        contact_z_um=float(z_key),
                        contact_loop=contact_loop,
                    ),
                )
                generated.append(
                    InterfacePlanRecord(
                        interface_id=f"MM__CONTACT__{lower.semantic_id}__{upper.semantic_id}__{index:04d}",
                        kind="MM",
                        owner_semantic_ids=(lower.semantic_id, upper.semantic_id),
                        recognition_rule="coplanar_conductor_contact_patch",
                        source_polygon_ids=(*lower.polygon_ids, *upper.polygon_ids),
                        metadata=metadata,
                    )
                )
                index += 1
    return (*interfaces, *generated)
