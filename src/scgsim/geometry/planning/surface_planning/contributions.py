"""Prepare source sheet intents and orchestrate route or neutral surface contributions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from scgsim.geometry.models.common import (
    RouteLiteral,
)
from scgsim.geometry.models.input import GeometryBuildInput
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.models.topology import (
    SurfacePlanRecord,
)
from scgsim.geometry.planning.contact_planning.intents import recognize_route_interfaces
from scgsim.geometry.planning.contact_planning.membership import (
    plan_mm_contact_records,
    plan_mm_contact_records_with_components,
)
from scgsim.geometry.planning.contact_planning.patches import (
    plan_conductor_contact_patches,
)
from scgsim.geometry.planning.domain_planning.reconciliation import (
    merge_solution_sidewall_interfaces,
    reconcile_solution_domain_boundaries,
)
from scgsim.geometry.planning.domain_planning.vacuum import (
    prepare_auto_vacuum_solution_regions,
)
from scgsim.geometry.planning.evidence import (
    build_semantic_evidence_facade,
)
from scgsim.geometry.planning.surface_planning.partitions import plan_surface_partitions
from scgsim.geometry.planning.surface_planning.planner import plan_route_surfaces
from scgsim.geometry.source.intents import route_a_sheet_interfaces


def plan_representation_contribution_geometry(
    build_input: GeometryBuildInput,
    representation_context: RepresentationContext,
) -> tuple[GeometryBuildInput, tuple[SurfacePlanRecord, ...]]:
    """Plan shared contribution surfaces from an explicit neutral context."""
    return _plan_surface_contribution_geometry(
        build_input,
        route=None,
        representation_context=representation_context,
    )


def _plan_surface_contribution_geometry(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral | None,
    representation_context: RepresentationContext | None = None,
) -> tuple[GeometryBuildInput, tuple[SurfacePlanRecord, ...]]:
    """Shared geometric contribution primitive with an explicit geometry authority."""
    if route is None or representation_context is not None:
        if representation_context is None:
            raise ValueError(
                "route-neutral contribution planning requires representation context"
            )
        build_input = prepare_auto_vacuum_solution_regions(
            build_input, route=route, representation_context=representation_context
        )
        build_input = refresh_generated_context_sheet_interfaces(
            build_input, representation_context=representation_context
        )
    elif route == "A":
        build_input = refresh_generated_route_a_sheet_interfaces(build_input)
    semantic_facts = build_semantic_evidence_facade(
        build_input,
        route=route,
        representation_context=representation_context,
    )
    interfaces = recognize_route_interfaces(
        build_input, route=route, representation_context=representation_context
    )
    interfaces = plan_conductor_contact_patches(
        build_input,
        route=route,
        interfaces=interfaces,
        representation_context=representation_context,
    )
    component_ledger = None
    if route is None and representation_context is not None:
        interfaces, mm_contacts, component_ledger = (
            plan_mm_contact_records_with_components(
                build_input,
                route=route,
                interfaces=interfaces,
                representation_context=representation_context,
            )
        )
    elif route in {"A", "B"}:
        interfaces, mm_contacts = plan_mm_contact_records(
            build_input,
            route=route,
            interfaces=interfaces,
            representation_context=representation_context,
        )
    else:
        mm_contacts = ()
    surfaces = plan_route_surfaces(
        build_input,
        route=route,
        interfaces=interfaces,
        surface_partitions=plan_surface_partitions(
            build_input, route=route, interfaces=interfaces
        ),
        construction_bodies=(),
        mm_contacts=mm_contacts,
        semantic_facts=semantic_facts,
        representation_context=representation_context,
        _conductor_component_ledger=component_ledger,
    )
    surfaces = reconcile_solution_domain_boundaries(build_input, surfaces=surfaces)
    return build_input, merge_solution_sidewall_interfaces(
        build_input,
        surfaces=surfaces,
        semantic_facts=semantic_facts,
    )


def refresh_generated_route_a_sheet_interfaces(
    build_input: GeometryBuildInput, *, representation_key: RouteLiteral = "A"
) -> GeometryBuildInput:
    """Rebuild adapter-generated Route A sheet footprints from final geometry.

    Junction partitioning can replace authored conductor polygons with PEC
    ends. Refresh only the generated sheet intents so their footprints follow
    that final geometry; caller-authored interface declarations stay intact.
    """
    intents_value = build_input.metadata.get("interface_intents_2d", {})
    if not isinstance(intents_value, Mapping):
        return build_input
    intents = dict(intents_value)
    existing = tuple(intents.get("interfaces", ()))
    generated_owner_ids = {
        str(owners[0])
        for intent in existing
        if isinstance(intent, Mapping)
        and intent.get("intent_origin") == "generated_route_a_surface_sheet"
        and isinstance((owners := intent.get("owner_semantic_ids")), Sequence)
        and not isinstance(owners, (str, bytes))
        and owners
    }
    explicit = tuple(
        intent
        for intent in existing
        if not (
            isinstance(intent, Mapping)
            and intent.get("intent_origin") == "generated_route_a_surface_sheet"
        )
    )
    generated = tuple(
        intent
        for intent in route_a_sheet_interfaces(
            build_input.entities,
            build_input.polygons,
            representation_key=representation_key,
        )["interfaces"]
        if intent["owner_semantic_ids"][0] in generated_owner_ids
    )
    intents["interfaces"] = (*explicit, *generated)
    return replace(
        build_input,
        metadata={**build_input.metadata, "interface_intents_2d": intents},
    )


def refresh_generated_context_sheet_interfaces(
    build_input: GeometryBuildInput,
    *,
    representation_context: RepresentationContext,
) -> GeometryBuildInput:
    """Refresh only generated context-sheet intents against final polygons."""
    intents_value = build_input.metadata.get("interface_intents_2d", {})
    if not isinstance(intents_value, Mapping):
        return build_input
    intents = dict(intents_value)
    existing = tuple(intents.get("interfaces", ()))
    context_origin = "generated_representation_context_surface_sheet"
    explicit = tuple(
        intent
        for intent in existing
        if not (
            isinstance(intent, Mapping)
            and intent.get("intent_origin") == context_origin
        )
    )
    generated = route_a_sheet_interfaces(
        build_input.entities,
        build_input.polygons,
        representation_context=representation_context,
    )["interfaces"]
    intents["interfaces"] = (*explicit, *generated)
    return replace(
        build_input,
        metadata={**build_input.metadata, "interface_intents_2d": intents},
    )
