"""Own the single polygon/curved route dispatcher and existing planning stage order. Native construction remains downstream."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

from scgsim.diagnostics import diagnostic_stage
from scgsim.geometry.compiler.stages import StageTimings
from scgsim.geometry.compiler.validation import (
    validate_curve_plan_coverage,
    validate_interface_surface_source_of_truth,
    validate_no_surface_overlap,
    validate_route_operation_coverage,
    validate_route_volume_surface_refs,
    validate_selected_route,
    validate_surface_deduplication,
    validate_surface_partition_coverage,
    validate_surface_sheet_interface_coverage,
    validate_surface_use_counts,
    validate_tag_plan_coverage,
    validate_volume_surface_closure,
)
from scgsim.geometry.construction.curved import build_curved_construction_plan
from scgsim.geometry.models.common import RouteLiteral
from scgsim.geometry.models.construction import (
    ConstructionBodyPlanRecord,
    ConstructionPlanRecord,
    CutHostOperationRecord,
    RouteABConstructionPlanRecord,
    SurfacePartitionRecord,
)
from scgsim.geometry.models.input import GeometryBuildInput
from scgsim.geometry.models.regions import PortSheetRegionRecord
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.models.tags import TagPlanRecord
from scgsim.geometry.models.topology import (
    CurvePlanRecord,
    InterfacePlanRecord,
    MMContactRecord,
    PointPlanRecord,
    SurfaceLoopRecord,
    SurfacePlanRecord,
    VolumePlanRecord,
)
from scgsim.geometry.planning.domain import (
    merge_solution_sidewall_interfaces,
    prepare_auto_vacuum_solution_regions,
    reconcile_solution_domain_boundaries,
)
from scgsim.geometry.planning.evidence import build_semantic_evidence_facade
from scgsim.geometry.planning.interfaces import (
    complete_interface_plan_from_surfaces,
    plan_conductor_contact_patches,
    plan_mm_contact_records,
    recognize_route_interfaces,
)
from scgsim.geometry.planning.ports import (
    bind_route_b_port_volume_plans,
    lower_port_sheet_regions,
    validate_route_b_port_sheet_sidewall_topology,
)
from scgsim.geometry.planning.representation import project_native_route_representations
from scgsim.geometry.planning.surfaces import (
    plan_route_surfaces,
    plan_surface_partitions,
)
from scgsim.geometry.planning.tags import plan_route_tags
from scgsim.geometry.planning.topology import plan_canonical_topology
from scgsim.geometry.planning.volumes import (
    plan_cut_host_operations,
    plan_route_construction_bodies,
    plan_route_volumes,
    reconcile_construction_body_surface_ids,
)
from scgsim.geometry.primitives.entities import component_metadata
from scgsim.geometry.primitives.grid import with_input_grid
from scgsim.semantics import SemanticEvidenceFacade


@with_input_grid
@diagnostic_stage("geometry.compiler.plan")
def build_route_construction_plan(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    representation_context=None,
) -> ConstructionPlanRecord:
    """Build the route-aware plan consumed by bottom-up OCC construction.

    The plan is the semantic center of v1. Interface recognition, surface
    ownership, partition ownership, and tag ownership are decided before OCC
    geometry exists.

    The backend must be able to build from this record without global
    `occ.fragment()`: surfaces carry loop geometry, volumes reference those
    surfaces, Route A/B cuts are explicit `CutHostOperationRecord`s, and
    `TagPlanRecord`s define the physical names before dim-tags exist.

    Tags are planned before backend construction. Every backend-live
    `SurfacePlanRecord` or `VolumePlanRecord` must either get a `TagPlanRecord`
    or be explicitly marked `construction_only`.
    """
    if build_input.metadata.get("surface_distance_bands"):
        raise ValueError(
            "SGB surface-distance geometry bands have been withdrawn; use Palace native inset_margins_um."
        )
    build_input = project_native_route_representations(
        build_input, route, representation_context
    )
    if build_input.boundary_curves or build_input.boundary_reconstruction:
        if route == "C":
            raise NotImplementedError(
                "active source curves support Palace Route A/B, not Route C"
            )
        return build_curved_construction_plan(
            build_input, route=route, representation_context=representation_context
        )
    timings = StageTimings(diagnostic_prefix="geometry.compiler.")
    build_input = prepare_auto_vacuum_solution_regions(
        build_input, route=route, representation_context=representation_context
    )
    with timings.measure("validate_selected_route"):
        validate_selected_route(
            build_input, route, representation_context=representation_context
        )
    semantic_facts = build_semantic_evidence_facade(
        build_input, route=route, representation_context=representation_context
    )
    planner = _PolygonPlanner(
        build_input, route, representation_context, semantic_facts, timings
    )
    return planner.build()


class _PlanArguments(TypedDict):
    route: RouteLiteral
    interfaces: tuple[InterfacePlanRecord, ...]
    surface_partitions: tuple[SurfacePartitionRecord, ...]
    construction_bodies: tuple[ConstructionBodyPlanRecord, ...]
    surfaces: tuple[SurfacePlanRecord, ...]
    points: tuple[PointPlanRecord, ...]
    curves: tuple[CurvePlanRecord, ...]
    surface_loops: tuple[SurfaceLoopRecord, ...]
    volumes: tuple[VolumePlanRecord, ...]
    cut_operations: tuple[CutHostOperationRecord, ...]
    tags: tuple[TagPlanRecord, ...]
    port_sheet_regions: tuple[PortSheetRegionRecord, ...]
    metadata: dict[str, Any]


@dataclass
class _PolygonPlanner:
    build_input: GeometryBuildInput
    route: RouteLiteral
    representation_context: RepresentationContext | None
    semantic_facts: SemanticEvidenceFacade
    timings: StageTimings
    interfaces: tuple[InterfacePlanRecord, ...] = ()
    surface_partitions: tuple[SurfacePartitionRecord, ...] = ()
    construction_bodies: tuple[ConstructionBodyPlanRecord, ...] = ()
    mm_contacts: tuple[MMContactRecord, ...] = ()
    surfaces: tuple[SurfacePlanRecord, ...] = ()
    points: tuple[PointPlanRecord, ...] = ()
    curves: tuple[CurvePlanRecord, ...] = ()
    surface_loops: tuple[SurfaceLoopRecord, ...] = ()
    volumes: tuple[VolumePlanRecord, ...] = ()
    cut_operations: tuple[CutHostOperationRecord, ...] = ()
    tags: tuple[TagPlanRecord, ...] = ()

    def build(self) -> ConstructionPlanRecord:
        """Run the existing planning and validation phases in source order."""
        self.plan_interfaces()
        self.plan_surfaces()
        self.plan_topology()
        self.plan_volumes()
        self.plan_cuts_and_tags()
        return self.assemble()

    def plan_interfaces(self) -> None:
        """Recognize interfaces and retain same-net metal contact records."""
        with self.timings.measure("recognize_route_interfaces"):
            self.interfaces = recognize_route_interfaces(
                self.build_input,
                route=self.route,
                representation_context=self.representation_context,
            )
        with self.timings.measure("plan_conductor_contact_patches"):
            self.interfaces = plan_conductor_contact_patches(
                self.build_input,
                route=self.route,
                representation_context=self.representation_context,
                interfaces=self.interfaces,
            )
        if self.route in {"A", "B"}:
            with self.timings.measure("plan_mm_contact_records"):
                contacts = plan_mm_contact_records(
                    self.build_input,
                    route=self.route,
                    representation_context=self.representation_context,
                    interfaces=self.interfaces,
                )
            self.interfaces, self.mm_contacts = contacts
        else:
            self.mm_contacts = ()

    def plan_surfaces(self) -> None:
        """Partition source interfaces and reconcile the final surface inventory."""
        with self.timings.measure("plan_surface_partitions"):
            self.surface_partitions = plan_surface_partitions(
                self.build_input, route=self.route, interfaces=self.interfaces
            )
        with self.timings.measure("plan_route_construction_bodies"):
            self.construction_bodies = plan_route_construction_bodies(
                self.build_input,
                route=self.route,
                interfaces=self.interfaces,
                representation_context=self.representation_context,
            )
        with self.timings.measure("plan_route_surfaces"):
            self.surfaces = plan_route_surfaces(
                self.build_input,
                route=self.route,
                representation_context=self.representation_context,
                interfaces=self.interfaces,
                surface_partitions=self.surface_partitions,
                construction_bodies=self.construction_bodies,
                mm_contacts=self.mm_contacts,
                semantic_facts=self.semantic_facts,
            )
        with self.timings.measure("reconcile_solution_domain_boundaries"):
            self.surfaces = reconcile_solution_domain_boundaries(
                self.build_input, surfaces=self.surfaces
            )
        if self.route in {"A", "B"} and self.build_input.port_sheet_regions:
            with self.timings.measure("lower_port_sheet_regions"):
                self.surfaces = lower_port_sheet_regions(
                    self.build_input,
                    route=self.route,
                    mm_contacts=self.mm_contacts,
                    planned_surfaces=self.surfaces,
                    representation_context=self.representation_context,
                )
        with self.timings.measure("validate_surface_sheet_interface_coverage"):
            validate_surface_sheet_interface_coverage(
                self.build_input, route=self.route, surfaces=self.surfaces
            )
        with self.timings.measure("merge_solution_sidewall_interfaces"):
            self.surfaces = merge_solution_sidewall_interfaces(
                self.build_input,
                surfaces=self.surfaces,
                semantic_facts=self.semantic_facts,
            )
        with self.timings.measure("complete_interface_plan_from_surfaces"):
            self.interfaces = complete_interface_plan_from_surfaces(
                interfaces=self.interfaces, surfaces=self.surfaces
            )

    def plan_topology(self) -> None:
        """Canonicalize shared topology and check its existing invariants."""
        with self.timings.measure("plan_canonical_topology"):
            topology = plan_canonical_topology(surfaces=self.surfaces)
        self.points, self.curves, self.surface_loops, self.surfaces = topology
        with self.timings.measure("validate_route_b_port_sheet_sidewall_topology"):
            validate_route_b_port_sheet_sidewall_topology(
                route=self.route,
                points=self.points,
                curves=self.curves,
                surface_loops=self.surface_loops,
                surfaces=self.surfaces,
            )
        with self.timings.measure("validate_curve_plan_coverage"):
            validate_curve_plan_coverage(
                points=self.points,
                curves=self.curves,
                surface_loops=self.surface_loops,
                surfaces=self.surfaces,
            )
        with self.timings.measure("validate_surface_deduplication"):
            validate_surface_deduplication(surfaces=self.surfaces)
        with self.timings.measure("validate_no_surface_overlap"):
            validate_no_surface_overlap(surfaces=self.surfaces)

    def plan_volumes(self) -> None:
        """Plan volumes, bind Route B ports and reconcile construction bodies."""
        with self.timings.measure("plan_route_volumes"):
            self.volumes = plan_route_volumes(
                self.build_input,
                route=self.route,
                surfaces=self.surfaces,
                mm_contacts=self.mm_contacts,
            )
        if self.route == "B":
            self.surfaces = bind_route_b_port_volume_plans(
                self.build_input, self.surfaces, self.volumes
            )
        with self.timings.measure("validate_route_volume_surface_refs"):
            validate_route_volume_surface_refs(route=self.route, volumes=self.volumes)
        with self.timings.measure("validate_volume_surface_closure"):
            validate_volume_surface_closure(
                volumes=self.volumes,
                surfaces=self.surfaces,
                surface_loops=self.surface_loops,
            )
        with self.timings.measure("validate_surface_use_counts"):
            validate_surface_use_counts(volumes=self.volumes, surfaces=self.surfaces)
        self.construction_bodies = reconcile_construction_body_surface_ids(
            self.construction_bodies, surfaces=self.surfaces
        )

    def plan_cuts_and_tags(self) -> None:
        """Plan host cuts and physical tags, then check ownership coverage."""
        with self.timings.measure("plan_cut_host_operations"):
            self.cut_operations = plan_cut_host_operations(
                route=self.route, construction_bodies=self.construction_bodies
            )
        with self.timings.measure("plan_route_tags"):
            self.tags = plan_route_tags(
                route=self.route, surfaces=self.surfaces, volumes=self.volumes
            )
        with self.timings.measure("validate_route_operation_coverage"):
            validate_route_operation_coverage(
                construction_bodies=self.construction_bodies,
                cut_operations=self.cut_operations,
                surfaces=self.surfaces,
            )
        with self.timings.measure("validate_surface_partition_coverage"):
            validate_surface_partition_coverage(
                interfaces=self.interfaces,
                surface_partitions=self.surface_partitions,
                surfaces=self.surfaces,
            )
        with self.timings.measure("validate_interface_surface_source_of_truth"):
            validate_interface_surface_source_of_truth(
                interfaces=self.interfaces, surfaces=self.surfaces
            )
        with self.timings.measure("validate_tag_plan_coverage"):
            validate_tag_plan_coverage(
                surfaces=self.surfaces, volumes=self.volumes, tags=self.tags
            )

    def assemble(self) -> ConstructionPlanRecord:
        """Assemble the route's record type without repeating planning decisions."""
        arguments: _PlanArguments = {
            "route": self.route,
            "interfaces": self.interfaces,
            "surface_partitions": self.surface_partitions,
            "points": self.points,
            "curves": self.curves,
            "surface_loops": self.surface_loops,
            "surfaces": self.surfaces,
            "volumes": self.volumes,
            "construction_bodies": self.construction_bodies,
            "cut_operations": self.cut_operations,
            "port_sheet_regions": self.build_input.port_sheet_regions,
            "tags": self.tags,
            "metadata": self.plan_metadata(),
        }
        if self.route in {"A", "B"}:
            return RouteABConstructionPlanRecord(
                **arguments, mm_contacts=self.mm_contacts
            )
        return ConstructionPlanRecord(**arguments)

    def plan_metadata(self) -> dict[str, Any]:
        """Retain the established grid, junction, port and timing projections."""
        return {
            "backend_strategy": "surface_plan_first_bottom_up_occ",
            **{
                key: self.build_input.metadata[key]
                for key in (
                    "geometry_grid",
                    "geometry_grid_arrangement",
                    "unquantized_source_sha256",
                )
                if key in self.build_input.metadata
            },
            **(
                {
                    "rectangular_junctions": self.build_input.metadata[
                        "rectangular_junctions"
                    ],
                    "junction_partitions": self.build_input.metadata[
                        "junction_partitions"
                    ],
                    "junction_authored_input_sha256": self.build_input.metadata[
                        "junction_authored_input_sha256"
                    ],
                }
                if self.build_input.metadata.get("rectangular_junctions")
                else {}
            ),
            "port_sheet_region_layer": {
                "source": "GeometryBuildInput.port_sheet_regions",
                "backend_live_surface_records": bool(
                    self.build_input.port_sheet_regions
                )
                and self.route in {"A", "B"},
                "allowed_overlap": "palace_lumped_port_sheet_only",
                "lowering_status": "lowered"
                if self.build_input.port_sheet_regions
                else "none",
            },
            "timings": self.timings.records,
            **(
                {"conductor_components": component_metadata(self.mm_contacts)}
                if self.route in {"A", "B"}
                else {}
            ),
        }
