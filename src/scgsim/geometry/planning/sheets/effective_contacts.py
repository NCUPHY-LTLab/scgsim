"""Route A sheets whose typed metal contacts hide one side, and sheet intent refresh.

Route A draws a thin conductor as a zero-thickness sheet; see ``docs/geometry-sgb.qmd``.
A sheet in typed contact with a finite shell (for example a bump) has one side covered by
that contact and the other side still exposed, so each side is resolved on its own here.
The covered footprint stays hidden as an MM (metal-metal) contact, and the paired exposed
patches are returned so substrate-air planning can subtract them.

Also refreshes generated sheet interface intents after junction partitioning changes the
conductor polygons. Ordinary sheet patches and surfaces are in ``sheet_patches`` and
``sheet_surfaces``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from scgsim.geometry.models.common import RouteLiteral
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.models.topology import (
    InterfacePlanRecord,
    MMContactRecord,
    SurfacePlanRecord,
)
from scgsim.geometry.planning.conductors.faces import planar_side_solution_regions
from scgsim.geometry.planning.context.evidence import conductor_solution_evidence
from scgsim.geometry.planning.context.representation import (
    planned_conductor_representation,
)
from scgsim.geometry.planning.context.scope import SurfaceScope
from scgsim.geometry.planning.context.tags import resolve_physical_group_owner_ids
from scgsim.geometry.planning.sheets.surfaces import route_a_sheet_patch_interface_kinds
from scgsim.geometry.planning.solutions.adjacency import route_a_sheet_plane_z_um
from scgsim.geometry.primitives.entities import (
    entity_by_id,
    geometry_ref_from_metadata,
)
from scgsim.geometry.primitives.loops import (
    boolean_gdstk_region,
    gdstk_surface_region,
    geometry_refs_from_gdstk_region,
    normalized_loop,
)
from scgsim.geometry.primitives.surface_records import (
    RouteASheetPatch,
    is_route_a_sheet_interface,
    surface_contribution_provenance,
)
from scgsim.geometry.source.intents import route_a_sheet_interfaces
from scgsim.semantics import EvidenceResult


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


def effective_sheet_contact_sides(
    build_input: GeometryBuildInput,
    contacts: Sequence[MMContactRecord],
    *,
    representation_context: RepresentationContext,
) -> dict[str, dict[str, list[MMContactRecord]]]:
    """Project typed opposite-face contacts using explicit representation facts."""
    entities = {entity.semantic_id: entity for entity in build_input.entities}
    result: dict[str, dict[str, list[MMContactRecord]]] = {}
    for contact in contacts:
        members = (
            (
                contact.lower_entity_id,
                contact.lower_source_face_id,
                contact.upper_entity_id,
            ),
            (
                contact.upper_entity_id,
                contact.upper_source_face_id,
                contact.lower_entity_id,
            ),
        )
        for entity_id, face_id, other_id in members:
            if (
                planned_conductor_representation(
                    entities[entity_id], route=None, context=representation_context
                )
                != "surface_sheet"
                or planned_conductor_representation(
                    entities[other_id], route=None, context=representation_context
                )
                != "cutout_boundary_shell"
            ):
                continue
            for side in ("bottom", "top"):
                if face_id == f"{entity_id}__{side}":
                    result.setdefault(entity_id, {}).setdefault(side, []).append(
                        contact
                    )
    return result


def plan_effective_contact_sheets(
    scope: SurfaceScope,
    effective_contacts: Mapping[str, Mapping[str, Sequence[MMContactRecord]]],
) -> tuple[
    dict[str, tuple[SurfacePlanRecord, ...]],
    list[RouteASheetPatch],
]:
    """Plan sheets whose typed contacts hide one side while the other stays exposed.

    Returns the planned surfaces keyed by interface id, and the paired exposed patches
    that the substrate-air planning subtracts as a mask.
    """
    effective_sheet_surfaces: dict[str, tuple[SurfacePlanRecord, ...]] = {}
    effective_mask_patches: list[RouteASheetPatch] = []
    for interface in scope.interfaces:
        if (
            is_route_a_sheet_interface(
                scope.route,
                interface,
                representation_context=scope.representation_context,
            )
            and interface.owner_semantic_ids[0] in effective_contacts
        ):
            patches, surfaces = _effective_contact_sheet_surfaces(
                scope,
                interface=interface,
                contact_sides=effective_contacts[interface.owner_semantic_ids[0]],
            )
            effective_mask_patches.extend(patches)
            effective_sheet_surfaces[interface.interface_id] = surfaces
    return effective_sheet_surfaces, effective_mask_patches


@dataclass(frozen=True)
class _EffectiveSheet:
    """One sheet interface being split into independently exposed sides."""

    scope: SurfaceScope
    interface: InterfacePlanRecord
    sheet: SemanticEntitySpec
    parent_ref: Mapping[str, Any]
    plane_z_um: float


def _effective_contact_sheet_surfaces(
    scope: SurfaceScope,
    *,
    interface: InterfacePlanRecord,
    contact_sides: Mapping[str, Sequence[MMContactRecord]],
) -> tuple[tuple[RouteASheetPatch, ...], tuple[SurfacePlanRecord, ...]]:
    """Resolve exposed sides independently while keeping MM footprints hidden.

    Paired exposed patches are also returned for the existing substrate/air mask
    subtraction. A contact on one side never removes the opposite exposed side.
    """
    import gdstk

    build_input = scope.build_input
    sheet = entity_by_id(build_input, interface.owner_semantic_ids[0])
    plane_z_um = route_a_sheet_plane_z_um(
        build_input, sheet, representation_context=scope.representation_context
    )
    parent_ref = {
        "from_interface_id": interface.interface_id,
        "source_polygon_ids": interface.source_polygon_ids,
        **geometry_ref_from_metadata(interface.metadata),
        "plane": {"axis": "z", "value_um": plane_z_um},
        "representation": "surface_sheet",
    }
    effective = _EffectiveSheet(
        scope=scope,
        interface=interface,
        sheet=sheet,
        parent_ref=parent_ref,
        plane_z_um=plane_z_um,
    )
    sheet_region = gdstk_surface_region(parent_ref)
    side_regions = {}
    records = []
    contributions = []
    for side in ("bottom", "top"):
        contacts = contact_sides.get(side, ())
        contact_region = tuple(
            gdstk.Polygon(normalized_loop(contact.outer_loop)) for contact in contacts
        )
        # What a contact covers on this side is hidden; only the rest is exposed.
        exposed = (
            boolean_gdstk_region(gdstk, sheet_region, contact_region, "not")
            if contact_region
            else sheet_region
        )
        adjacency = (
            planar_side_solution_regions(
                build_input,
                owner_id=sheet.semantic_id,
                occupied_region=exposed,
                plane_z_um=plane_z_um,
                side=side,
            )
            if exposed
            else ()
        )
        side_regions[side] = adjacency
        side_records, side_contributions = _exposed_side_surfaces(
            effective, side=side, adjacency=adjacency, contacts=contacts
        )
        records.extend(side_records)
        contributions.extend(side_contributions)
    # Preserve authored interface-kind agreement; generated kinds follow evidence.
    route_a_sheet_patch_interface_kinds(interface, contributions)
    paired = _paired_exposed_patches(effective, side_regions)
    return tuple(paired), tuple(records)


def _exposed_side_surfaces(
    effective: _EffectiveSheet,
    *,
    side: str,
    adjacency: Sequence[tuple[str, tuple[Any, ...]]],
    contacts: Sequence[MMContactRecord],
) -> tuple[list[SurfacePlanRecord], list[EvidenceResult]]:
    """Build one surface per exposed piece of one sheet side, with its evidence."""
    scope = effective.scope
    interface = effective.interface
    sheet = effective.sheet
    records: list[SurfacePlanRecord] = []
    contributions: list[EvidenceResult] = []
    for solution_id, region in adjacency:
        for index, geometry_ref in enumerate(
            geometry_refs_from_gdstk_region(effective.parent_ref, region)
        ):
            patch_id = (
                f"effective-local:{interface.interface_id}:{side}:"
                f"{solution_id}:{index:04d}"
            )
            evidence = conductor_solution_evidence(
                scope.semantic_facts,
                contribution_id=patch_id,
                patch_id=f"planned:{patch_id}",
                conductor_id=sheet.semantic_id,
                solution_id=solution_id,
                side=side,
            )
            contributions.append(evidence)
            records.append(
                SurfacePlanRecord(
                    surface_id=f"SURF__{patch_id}",
                    owner_semantic_id=sheet.semantic_id,
                    surface_role="effective_exposed_sheet_interface",
                    geometry_ref=geometry_ref,
                    interface_id=interface.interface_id,
                    normal_hint=(0.0, 0.0, -1.0 if side == "bottom" else 1.0),
                    solver_use=interface.solver_use or "solver_active",
                    metadata={
                        "interface_kinds": (evidence.classification,),
                        "owner_semantic_ids": evidence.source_owner_ids,
                        "physical_owner_semantic_ids": resolve_physical_group_owner_ids(
                            scope.build_input, evidence.source_owner_ids
                        ),
                        "boundary_volume_ids": (solution_id,),
                        "embedded_surface_sheet": True,
                        "excluded_contact_ids": tuple(c.contact_id for c in contacts),
                        "source_provenance": surface_contribution_provenance(
                            parent_interface_id=interface.interface_id,
                            patch_id=patch_id,
                            contributions=(evidence,),
                        ),
                    },
                )
            )
    return records, contributions


def _paired_exposed_patches(
    effective: _EffectiveSheet,
    side_regions: Mapping[str, Sequence[tuple[str, tuple[Any, ...]]]],
) -> list[RouteASheetPatch]:
    """Intersect the bottom and top exposed regions into patches for the mask subtraction."""
    import gdstk

    semantic_facts = effective.scope.semantic_facts
    interface = effective.interface
    sheet = effective.sheet
    paired: list[RouteASheetPatch] = []
    for bottom_id, bottom_region in side_regions["bottom"]:
        for top_id, top_region in side_regions["top"]:
            overlap = boolean_gdstk_region(gdstk, bottom_region, top_region, "and")
            for geometry_ref in geometry_refs_from_gdstk_region(
                effective.parent_ref, overlap
            ):
                patch_id = f"effective-mask:{interface.interface_id}:{len(paired):04d}"
                paired.append(
                    RouteASheetPatch(
                        parent_interface_id=interface.interface_id,
                        sheet_entity_id=sheet.semantic_id,
                        patch_id=patch_id,
                        geometry_ref=geometry_ref,
                        bottom=conductor_solution_evidence(
                            semantic_facts,
                            contribution_id=f"{patch_id}:bottom",
                            patch_id=f"planned:{patch_id}:bottom",
                            conductor_id=sheet.semantic_id,
                            solution_id=bottom_id,
                            side="bottom",
                        ),
                        top=conductor_solution_evidence(
                            semantic_facts,
                            contribution_id=f"{patch_id}:top",
                            patch_id=f"planned:{patch_id}:top",
                            conductor_id=sheet.semantic_id,
                            solution_id=top_id,
                            side="top",
                        ),
                    )
                )
    return paired
