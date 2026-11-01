"""Own the single polygon/curved route dispatcher and existing planning stage order. Native construction remains downstream."""

from __future__ import annotations

import time
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

    state = _PlanState(
        build_input=build_input,
        route=route,
        representation_context=representation_context,
    )
    # The stage order is the order of the original single function. Each stage
    # reads what earlier stages stored on ``state``; the timings list records it.
    _plan_interfaces(state)
    _plan_surfaces(state)
    _plan_topology(state)
    _plan_volumes(state)
    _plan_cuts_and_tags(state)
    return _assemble_plan(state)


@dataclass
class _PlanState:
    """Intermediate planning results, filled in by the stage functions in order.

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


def _plan_interfaces(state: _PlanState) -> None:
    """Validate the route, recognize interfaces, plan contact patches and MM contacts."""
    route = state.route
    context = state.representation_context
    state.build_input = prepare_auto_vacuum_solution_regions(
        state.build_input, route=route, representation_context=context
    )
    build_input = state.build_input
    timings = state.timings
    _timed(
        timings,
        "validate_selected_route",
        lambda: validate_selected_route(
            build_input, route, representation_context=context
        ),
    )
    state.semantic_facts = build_semantic_evidence_facade(
        build_input, route=route, representation_context=context
    )
    state.interfaces = _timed(
        timings,
        "recognize_route_interfaces",
        lambda: recognize_route_interfaces(
            build_input, route=route, representation_context=context
        ),
    )
    state.interfaces = _timed(
        timings,
        "plan_conductor_contact_patches",
        lambda: plan_conductor_contact_patches(
            build_input,
            route=route,
            representation_context=context,
            interfaces=state.interfaces,
        ),
    )
    if state.is_route_ab:
        state.interfaces, state.mm_contacts = _timed(
            timings,
            "plan_mm_contact_records",
            lambda: plan_mm_contact_records(
                build_input,
                route=route,
                representation_context=context,
                interfaces=state.interfaces,
            ),
        )
    else:
        state.mm_contacts = ()


def _plan_surfaces(state: _PlanState) -> None:
    """Plan partitions, construction bodies and surfaces, then fold in port sheets."""
    route = state.route
    build_input = state.build_input
    timings = state.timings
    state.surface_partitions = _timed(
        timings,
        "plan_surface_partitions",
        lambda: plan_surface_partitions(
            build_input,
            route=route,
            interfaces=state.interfaces,
        ),
    )
    state.construction_bodies = _timed(
        timings,
        "plan_route_construction_bodies",
        lambda: plan_route_construction_bodies(
            build_input,
            route=route,
            interfaces=state.interfaces,
            representation_context=state.representation_context,
        ),
    )
    state.surfaces = _timed(
        timings,
        "plan_route_surfaces",
        lambda: plan_route_surfaces(
            build_input,
            route=route,
            representation_context=state.representation_context,
            interfaces=state.interfaces,
            surface_partitions=state.surface_partitions,
            construction_bodies=state.construction_bodies,
            mm_contacts=state.mm_contacts,
            semantic_facts=state.semantic_facts,
        ),
    )
    state.surfaces = _timed(
        timings,
        "reconcile_solution_domain_boundaries",
        lambda: reconcile_solution_domain_boundaries(
            build_input,
            surfaces=state.surfaces,
        ),
    )
    if state.is_route_ab and build_input.port_sheet_regions:
        state.surfaces = _timed(
            timings,
            "lower_port_sheet_regions",
            lambda: lower_port_sheet_regions(
                build_input,
                route=route,
                mm_contacts=state.mm_contacts,
                planned_surfaces=state.surfaces,
            ),
        )
    _timed(
        timings,
        "validate_surface_sheet_interface_coverage",
        lambda: validate_surface_sheet_interface_coverage(
            build_input,
            route=route,
            surfaces=state.surfaces,
        ),
    )
    state.surfaces = _timed(
        timings,
        "merge_solution_sidewall_interfaces",
        lambda: merge_solution_sidewall_interfaces(
            build_input,
            surfaces=state.surfaces,
            semantic_facts=state.semantic_facts,
        ),
    )
    state.interfaces = _timed(
        timings,
        "complete_interface_plan_from_surfaces",
        lambda: complete_interface_plan_from_surfaces(
            interfaces=state.interfaces,
            surfaces=state.surfaces,
        ),
    )


def _plan_topology(state: _PlanState) -> None:
    """Derive points, curves and surface loops, then check they cover the surfaces."""
    route = state.route
    timings = state.timings
    state.points, state.curves, state.surface_loops, state.surfaces = _timed(
        timings,
        "plan_canonical_topology",
        lambda: plan_canonical_topology(surfaces=state.surfaces),
    )
    _timed(
        timings,
        "validate_route_b_port_sheet_sidewall_topology",
        lambda: validate_route_b_port_sheet_sidewall_topology(
            route=route,
            points=state.points,
            curves=state.curves,
            surface_loops=state.surface_loops,
            surfaces=state.surfaces,
        ),
    )
    _timed(
        timings,
        "validate_curve_plan_coverage",
        lambda: validate_curve_plan_coverage(
            points=state.points,
            curves=state.curves,
            surface_loops=state.surface_loops,
            surfaces=state.surfaces,
        ),
    )
    _timed(
        timings,
        "validate_surface_deduplication",
        lambda: validate_surface_deduplication(surfaces=state.surfaces),
    )
    _timed(
        timings,
        "validate_no_surface_overlap",
        lambda: validate_no_surface_overlap(surfaces=state.surfaces),
    )


def _plan_volumes(state: _PlanState) -> None:
    """Plan volumes from the surfaces and check they close over them."""
    route = state.route
    timings = state.timings
    state.volumes = _timed(
        timings,
        "plan_route_volumes",
        lambda: plan_route_volumes(
            state.build_input,
            route=route,
            surfaces=state.surfaces,
            mm_contacts=state.mm_contacts,
        ),
    )
    if route == "B":
        state.surfaces = bind_route_b_port_volume_plans(
            state.build_input, state.surfaces, state.volumes
        )
    _timed(
        timings,
        "validate_route_volume_surface_refs",
        lambda: validate_route_volume_surface_refs(route=route, volumes=state.volumes),
    )
    _timed(
        timings,
        "validate_volume_surface_closure",
        lambda: validate_volume_surface_closure(
            volumes=state.volumes,
            surfaces=state.surfaces,
            surface_loops=state.surface_loops,
        ),
    )
    _timed(
        timings,
        "validate_surface_use_counts",
        lambda: validate_surface_use_counts(
            volumes=state.volumes, surfaces=state.surfaces
        ),
    )


def _plan_cuts_and_tags(state: _PlanState) -> None:
    """Plan cut operations and physical tags, then run the final coverage checks."""
    route = state.route
    timings = state.timings
    state.construction_bodies = reconcile_construction_body_surface_ids(
        state.construction_bodies,
        surfaces=state.surfaces,
    )
    state.cut_operations = _timed(
        timings,
        "plan_cut_host_operations",
        lambda: plan_cut_host_operations(
            route=route,
            construction_bodies=state.construction_bodies,
        ),
    )
    state.tags = _timed(
        timings,
        "plan_route_tags",
        lambda: plan_route_tags(
            route=route, surfaces=state.surfaces, volumes=state.volumes
        ),
    )
    _timed(
        timings,
        "validate_route_operation_coverage",
        lambda: validate_route_operation_coverage(
            construction_bodies=state.construction_bodies,
            cut_operations=state.cut_operations,
            surfaces=state.surfaces,
        ),
    )
    _timed(
        timings,
        "validate_surface_partition_coverage",
        lambda: validate_surface_partition_coverage(
            interfaces=state.interfaces,
            surface_partitions=state.surface_partitions,
            surfaces=state.surfaces,
        ),
    )
    _timed(
        timings,
        "validate_interface_surface_source_of_truth",
        lambda: validate_interface_surface_source_of_truth(
            interfaces=state.interfaces,
            surfaces=state.surfaces,
        ),
    )
    _timed(
        timings,
        "validate_tag_plan_coverage",
        lambda: validate_tag_plan_coverage(
            surfaces=state.surfaces,
            volumes=state.volumes,
            tags=state.tags,
        ),
    )


def _assemble_plan(state: _PlanState) -> ConstructionPlanRecord:
    """Bundle the staged results; only Route A/B plans carry MM contacts."""
    route = state.route
    build_input = state.build_input
    plan_type = (
        RouteABConstructionPlanRecord if state.is_route_ab else ConstructionPlanRecord
    )
    plan_kwargs = {
        "route": route,
        "interfaces": state.interfaces,
        "surface_partitions": state.surface_partitions,
        "points": state.points,
        "curves": state.curves,
        "surface_loops": state.surface_loops,
        "surfaces": state.surfaces,
        "volumes": state.volumes,
        "construction_bodies": state.construction_bodies,
        "cut_operations": state.cut_operations,
        "port_sheet_regions": build_input.port_sheet_regions,
        "tags": state.tags,
        "metadata": {
            "backend_strategy": "surface_plan_first_bottom_up_occ",
            "port_sheet_region_layer": {
                "source": "GeometryBuildInput.port_sheet_regions",
                "backend_live_surface_records": (
                    bool(build_input.port_sheet_regions) and state.is_route_ab
                ),
                "allowed_overlap": "palace_lumped_port_sheet_only",
                "lowering_status": (
                    "lowered" if build_input.port_sheet_regions else "none"
                ),
            },
            "timings": state.timings,
            **(
                {"conductor_components": component_metadata(state.mm_contacts)}
                if state.is_route_ab
                else {}
            ),
        },
    }
    if state.is_route_ab:
        plan_kwargs["mm_contacts"] = state.mm_contacts
    return plan_type(**plan_kwargs)


def _timed(
    timings: list[dict[str, Any]],
    stage: str,
    fn: Any,
) -> Any:
    started = time.perf_counter()
    try:
        result = fn()
    except Exception:
        timings.append(
            {
                "stage": stage,
                "seconds": round(time.perf_counter() - started, 6),
                "status": "failed",
            }
        )
        raise
    timings.append(
        {
            "stage": stage,
            "seconds": round(time.perf_counter() - started, 6),
            "status": "done",
        }
    )
    return result
