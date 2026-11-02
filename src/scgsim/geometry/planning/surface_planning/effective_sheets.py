"""Resolve mixed sheet/solid contact exposure independently on each physical side."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from scgsim.geometry.models.common import PolygonRing
from scgsim.geometry.models.input import GeometryBuildInput
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.models.topology import (
    InterfacePlanRecord,
    MMContactRecord,
    SurfacePlanRecord,
)
from scgsim.geometry.planning.domain import (
    planar_side_solution_regions,
    route_a_sheet_plane_z_um,
)
from scgsim.geometry.planning.evidence import (
    conductor_solution_evidence,
)
from scgsim.geometry.planning.representation import (
    planned_conductor_representation,
)
from scgsim.geometry.planning.surface_planning.sheets import (
    route_a_sheet_patch_interface_kinds,
)
from scgsim.geometry.planning.tags import (
    physical_group_owner_ids,
)
from scgsim.geometry.primitives.entities import (
    entity_by_id,
    geometry_ref_from_metadata,
)
from scgsim.geometry.primitives.grid import boolean_polygon_region
from scgsim.geometry.primitives.loops import (
    gdstk_surface_region,
    geometry_refs_from_gdstk_region,
)
from scgsim.geometry.primitives.planar_loops import clean_loop
from scgsim.geometry.primitives.surface_records import (
    RouteASheetPatch,
    surface_contribution_provenance,
)
from scgsim.semantics import SemanticEvidenceFacade


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


def effective_contact_sheet_surfaces(
    build_input: GeometryBuildInput,
    *,
    interface: InterfacePlanRecord,
    contact_sides: Mapping[str, Sequence[MMContactRecord]],
    semantic_facts: SemanticEvidenceFacade,
    representation_context: RepresentationContext,
) -> tuple[tuple[RouteASheetPatch, ...], tuple[SurfacePlanRecord, ...]]:
    """Resolve exposed sides independently while keeping MM footprints hidden.

    Paired exposed patches are also returned for the existing substrate/air mask
    subtraction. A contact on one side never removes the opposite exposed side.
    """
    import gdstk  # noqa: PLC0415 - optional native polygon dependency

    sheet = entity_by_id(build_input, interface.owner_semantic_ids[0])
    plane_z_um = route_a_sheet_plane_z_um(
        build_input, sheet, representation_context=representation_context
    )
    parent_ref = {
        "from_interface_id": interface.interface_id,
        "source_polygon_ids": interface.source_polygon_ids,
        **geometry_ref_from_metadata(interface.metadata),
        "plane": {"axis": "z", "value_um": plane_z_um},
        "representation": "surface_sheet",
    }
    sheet_region = gdstk_surface_region(parent_ref)
    side_regions = {}
    records = []
    contributions = []
    for side in ("bottom", "top"):
        contacts = contact_sides.get(side, ())
        contact_region = tuple(
            gdstk.Polygon(clean_loop(cast(PolygonRing, contact.outer_loop)))
            for contact in contacts
        )
        exposed = (
            boolean_polygon_region(gdstk, sheet_region, contact_region, "not")
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
        for solution_id, region in adjacency:
            for index, geometry_ref in enumerate(
                geometry_refs_from_gdstk_region(parent_ref, region)
            ):
                patch_id = (
                    f"effective-local:{interface.interface_id}:{side}:"
                    f"{solution_id}:{index:04d}"
                )
                evidence = conductor_solution_evidence(
                    semantic_facts,
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
                            "physical_owner_semantic_ids": physical_group_owner_ids(
                                build_input, evidence.source_owner_ids
                            ),
                            "boundary_volume_ids": (solution_id,),
                            "embedded_surface_sheet": True,
                            "excluded_contact_ids": tuple(
                                c.contact_id for c in contacts
                            ),
                            "source_provenance": surface_contribution_provenance(
                                parent_interface_id=interface.interface_id,
                                patch_id=patch_id,
                                contributions=(evidence,),
                            ),
                        },
                    )
                )
    # Preserve authored interface-kind agreement; generated kinds follow evidence.
    route_a_sheet_patch_interface_kinds(interface, contributions)
    paired = []
    for bottom_id, bottom_region in side_regions["bottom"]:
        for top_id, top_region in side_regions["top"]:
            overlap = boolean_polygon_region(gdstk, bottom_region, top_region, "and")
            for geometry_ref in geometry_refs_from_gdstk_region(parent_ref, overlap):
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
    return tuple(paired), tuple(records)
