"""Own the single polygon/curved route dispatcher and existing planning stage order. Native construction remains downstream."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from scgsim.geometry.compiler.surface_overlap import validate_no_surface_overlap
from scgsim.geometry.models.common import RouteLiteral
from scgsim.geometry.models.construction import (
    ConstructionPlanRecord,
    RouteABConstructionPlanRecord,
)
from scgsim.geometry.models.input import GeometryBuildInput
from scgsim.geometry.planning.conductors.contact_patches import (
    plan_conductor_contact_patches,
)
from scgsim.geometry.planning.conductors.mm_contacts import plan_mm_contact_records
from scgsim.geometry.planning.context.evidence import build_semantic_evidence_facade
from scgsim.geometry.planning.context.representation import (
    project_native_route_representations,
)
from scgsim.geometry.planning.context.tags import plan_route_tags
from scgsim.geometry.planning.context.topology import plan_canonical_topology
from scgsim.geometry.planning.domain.boundaries import (
    merge_solution_sidewall_interfaces,
    reconcile_solution_domain_boundaries,
)
from scgsim.geometry.planning.interfaces import (
    complete_interface_plan_from_surfaces,
    recognize_route_interfaces,
)
from scgsim.geometry.planning.ports.lowering import lower_port_sheet_regions
from scgsim.geometry.planning.ports.route_b import (
    bind_route_b_port_volume_plans,
    validate_route_b_port_sheet_sidewall_topology,
)
from scgsim.geometry.planning.solutions.auto_vacuum import (
    prepare_auto_vacuum_solution_regions,
)
from scgsim.geometry.planning.surfaces import (
    plan_route_surfaces,
    plan_surface_partitions,
)
from scgsim.geometry.planning.validation import (
    validate_curve_plan_coverage,
    validate_interface_surface_source_of_truth,
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
from scgsim.geometry.planning.volumes import (
    plan_cut_host_operations,
    plan_route_construction_bodies,
    plan_route_volumes,
    reconcile_construction_body_surface_ids,
)
from scgsim.geometry.primitives.entities import component_metadata


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
    build_input = project_native_route_representations(
        build_input, route, representation_context
    )
    if build_input.boundary_curves or build_input.boundary_reconstruction:
        if route == "C":
            raise NotImplementedError(
                "active source curves support Palace Route A/B, not Route C"
            )
        from scgsim.geometry.construction.curved import build_curved_construction_plan

        return build_curved_construction_plan(
            build_input, route=route, representation_context=representation_context
        )

    return _RoutePlanner(
        build_input=build_input,
        route=route,
        representation_context=representation_context,
    ).build()


@dataclass
class _RoutePlanner:
    """Plan one route, stage by stage; each stage reads what earlier stages stored.

    Fields after ``timings`` hold ``None`` until the stage that produces them runs.
    """

    build_input: GeometryBuildInput
    route: RouteLiteral
    representation_context: Any
    timings: list[dict[str, Any]] = field(default_factory=list)
    semantic_facts: Any = None
    interfaces: Any = None
    mm_contacts: Any = ()
    surface_partitions: Any = None
    construction_bodies: Any = None
    surfaces: Any = None
    points: Any = None
    curves: Any = None
    surface_loops: Any = None
    volumes: Any = None
    cut_operations: Any = None
    tags: Any = None

    @property
    def is_route_ab(self) -> bool:
        """Routes A and B share cut-based construction and carry MM contacts."""
        return self.route in {"A", "B"}

    def build(self) -> ConstructionPlanRecord:
        """Run the planning stages in their required order and bundle the results."""
        self.plan_interfaces()
        self.plan_surfaces()
        self.plan_topology()
        self.plan_volumes()
        self.plan_cuts_and_tags()
        return self.assemble()

    def plan_interfaces(self) -> None:
        """Validate the route, recognize interfaces, plan contact patches and MM contacts."""
        self.build_input = prepare_auto_vacuum_solution_regions(
            self.build_input,
            route=self.route,
            representation_context=self.representation_context,
        )
        self.timed(
            validate_selected_route,
            self.build_input,
            self.route,
            representation_context=self.representation_context,
        )
        self.semantic_facts = build_semantic_evidence_facade(
            self.build_input,
            route=self.route,
            representation_context=self.representation_context,
        )
        self.interfaces = self.timed(
            recognize_route_interfaces,
            self.build_input,
            route=self.route,
            representation_context=self.representation_context,
        )
        self.interfaces = self.timed(
            plan_conductor_contact_patches,
            self.build_input,
            route=self.route,
            representation_context=self.representation_context,
            interfaces=self.interfaces,
        )
        if self.is_route_ab:
            self.interfaces, self.mm_contacts = self.timed(
                plan_mm_contact_records,
                self.build_input,
                route=self.route,
                representation_context=self.representation_context,
                interfaces=self.interfaces,
            )
        else:
            self.mm_contacts = ()

    def plan_surfaces(self) -> None:
        """Plan partitions, construction bodies and surfaces, then fold in port sheets."""
        self.surface_partitions = self.timed(
            plan_surface_partitions,
            self.build_input,
            route=self.route,
            interfaces=self.interfaces,
        )
        self.construction_bodies = self.timed(
            plan_route_construction_bodies,
            self.build_input,
            route=self.route,
            interfaces=self.interfaces,
            representation_context=self.representation_context,
        )
        self.surfaces = self.timed(
            plan_route_surfaces,
            self.build_input,
            route=self.route,
            representation_context=self.representation_context,
            interfaces=self.interfaces,
            surface_partitions=self.surface_partitions,
            construction_bodies=self.construction_bodies,
            mm_contacts=self.mm_contacts,
            semantic_facts=self.semantic_facts,
        )
        self.surfaces = self.timed(
            reconcile_solution_domain_boundaries,
            self.build_input,
            surfaces=self.surfaces,
        )
        if self.is_route_ab and self.build_input.port_sheet_regions:
            self.surfaces = self.timed(
                lower_port_sheet_regions,
                self.build_input,
                route=self.route,
                mm_contacts=self.mm_contacts,
                planned_surfaces=self.surfaces,
            )
        self.timed(
            validate_surface_sheet_interface_coverage,
            self.build_input,
            route=self.route,
            surfaces=self.surfaces,
        )
        self.surfaces = self.timed(
            merge_solution_sidewall_interfaces,
            self.build_input,
            surfaces=self.surfaces,
            semantic_facts=self.semantic_facts,
        )
        self.interfaces = self.timed(
            complete_interface_plan_from_surfaces,
            interfaces=self.interfaces,
            surfaces=self.surfaces,
        )

    def plan_topology(self) -> None:
        """Derive points, curves and surface loops, then check they cover the surfaces."""
        self.points, self.curves, self.surface_loops, self.surfaces = self.timed(
            plan_canonical_topology, surfaces=self.surfaces
        )
        self.timed(
            validate_route_b_port_sheet_sidewall_topology,
            route=self.route,
            points=self.points,
            curves=self.curves,
            surface_loops=self.surface_loops,
            surfaces=self.surfaces,
        )
        self.timed(
            validate_curve_plan_coverage,
            points=self.points,
            curves=self.curves,
            surface_loops=self.surface_loops,
            surfaces=self.surfaces,
        )
        self.timed(validate_surface_deduplication, surfaces=self.surfaces)
        self.timed(validate_no_surface_overlap, surfaces=self.surfaces)

    def plan_volumes(self) -> None:
        """Plan volumes from the surfaces and check they close over them."""
        self.volumes = self.timed(
            plan_route_volumes,
            self.build_input,
            route=self.route,
            surfaces=self.surfaces,
            mm_contacts=self.mm_contacts,
        )
        if self.route == "B":
            self.surfaces = bind_route_b_port_volume_plans(
                self.build_input, self.surfaces, self.volumes
            )
        self.timed(
            validate_route_volume_surface_refs, route=self.route, volumes=self.volumes
        )
        self.timed(
            validate_volume_surface_closure,
            volumes=self.volumes,
            surfaces=self.surfaces,
            surface_loops=self.surface_loops,
        )
        self.timed(
            validate_surface_use_counts, volumes=self.volumes, surfaces=self.surfaces
        )

    def plan_cuts_and_tags(self) -> None:
        """Plan cut operations and physical tags, then run the final coverage checks."""
        self.construction_bodies = reconcile_construction_body_surface_ids(
            self.construction_bodies,
            surfaces=self.surfaces,
        )
        self.cut_operations = self.timed(
            plan_cut_host_operations,
            route=self.route,
            construction_bodies=self.construction_bodies,
        )
        self.tags = self.timed(
            plan_route_tags,
            route=self.route,
            surfaces=self.surfaces,
            volumes=self.volumes,
        )
        self.timed(
            validate_route_operation_coverage,
            construction_bodies=self.construction_bodies,
            cut_operations=self.cut_operations,
            surfaces=self.surfaces,
        )
        self.timed(
            validate_surface_partition_coverage,
            interfaces=self.interfaces,
            surface_partitions=self.surface_partitions,
            surfaces=self.surfaces,
        )
        self.timed(
            validate_interface_surface_source_of_truth,
            interfaces=self.interfaces,
            surfaces=self.surfaces,
        )
        self.timed(
            validate_tag_plan_coverage,
            surfaces=self.surfaces,
            volumes=self.volumes,
            tags=self.tags,
        )

    def assemble(self) -> ConstructionPlanRecord:
        """Bundle the staged results; only Route A/B plans carry MM contacts."""
        plan_type = (
            RouteABConstructionPlanRecord
            if self.is_route_ab
            else ConstructionPlanRecord
        )
        plan_kwargs = {
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
            "metadata": {
                "backend_strategy": "surface_plan_first_bottom_up_occ",
                "port_sheet_region_layer": {
                    "source": "GeometryBuildInput.port_sheet_regions",
                    "backend_live_surface_records": (
                        bool(self.build_input.port_sheet_regions) and self.is_route_ab
                    ),
                    "allowed_overlap": "palace_lumped_port_sheet_only",
                    "lowering_status": (
                        "lowered" if self.build_input.port_sheet_regions else "none"
                    ),
                },
                "timings": self.timings,
                **(
                    {"conductor_components": component_metadata(self.mm_contacts)}
                    if self.is_route_ab
                    else {}
                ),
            },
        }
        if self.is_route_ab:
            plan_kwargs["mm_contacts"] = self.mm_contacts
        return plan_type(**plan_kwargs)

    def timed(self, step: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run one planning step and record its duration under the step's name."""
        started = time.perf_counter()
        try:
            result = step(*args, **kwargs)
        except Exception:
            self.timings.append(
                {
                    "stage": step.__name__,
                    "seconds": round(time.perf_counter() - started, 6),
                    "status": "failed",
                }
            )
            raise
        self.timings.append(
            {
                "stage": step.__name__,
                "seconds": round(time.perf_counter() - started, 6),
                "status": "done",
            }
        )
        return result
