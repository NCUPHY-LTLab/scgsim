"""Partition Route A sheets by ordered local domains and detach their exposed surfaces."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any, cast

from scgsim.geometry.grid.grid import boolean_polygon_region
from scgsim.geometry.grid.loops import (
    gdstk_surface_region,
    geometry_refs_from_gdstk_region,
    loop_signature,
)
from scgsim.geometry.grid.planar_loops import clean_loop
from scgsim.geometry.models.common import (
    INTERFACE_KIND_ORDER,
    PolygonRing,
    RouteLiteral,
)
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.models.topology import (
    InterfacePlanRecord,
    MMContactRecord,
    SurfacePlanRecord,
)
from scgsim.geometry.planning.domain_planning.faces import (
    planar_side_solution_regions,
    route_a_sheet_plane_z_um,
)
from scgsim.geometry.planning.entities import (
    entity_by_id,
    geometry_ref_from_metadata,
    unique_ids,
)
from scgsim.geometry.planning.evidence import (
    conductor_solution_evidence,
)
from scgsim.geometry.planning.surface_planning.contacts import (
    route_a_sheet_contact_cap_face,
    subtract_contact_patches_from_face,
)
from scgsim.geometry.planning.surface_planning.scope import SurfaceScope
from scgsim.geometry.planning.surface_records import (
    RouteASheetPatch,
    is_route_a_sheet_interface,
    surface_contribution_provenance,
)
from scgsim.geometry.planning.tags import (
    entity_physical_group_id,
    physical_group_owner_ids,
)
from scgsim.semantics import EvidenceResult, SemanticEvidenceFacade


def plan_route_a_sheet_patches(
    build_input: GeometryBuildInput,
    *,
    interfaces: Sequence[InterfacePlanRecord],
    semantic_facts: SemanticEvidenceFacade,
    route: RouteLiteral | None = "A",
    representation_context: RepresentationContext | None = None,
) -> tuple[RouteASheetPatch, ...]:
    """Partition each Route-A sheet by its exact ordered local domains."""
    import gdstk  # noqa: PLC0415 - optional native polygon dependency

    raw_patches: list[
        tuple[
            InterfacePlanRecord,
            SemanticEntitySpec,
            str,
            str,
            dict[str, Any],
        ]
    ] = []
    for interface in interfaces:
        if not is_route_a_sheet_interface(
            route, interface, representation_context=representation_context
        ):
            continue
        sheet = entity_by_id(build_input, interface.owner_semantic_ids[0])
        plane_z_um = route_a_sheet_plane_z_um(
            build_input, sheet, representation_context=representation_context
        )
        parent_geometry_ref = {
            "from_interface_id": interface.interface_id,
            "source_polygon_ids": interface.source_polygon_ids,
            **geometry_ref_from_metadata(interface.metadata),
            "plane": {"axis": "z", "value_um": plane_z_um},
            "representation": "surface_sheet",
        }
        sheet_region = gdstk_surface_region(parent_geometry_ref)
        if not sheet_region:
            raise ValueError(
                f"{sheet.semantic_id} Route A sheet has no occupied region"
            )
        bottom_regions = _route_a_sheet_side_solution_regions(
            build_input,
            sheet=sheet,
            sheet_region=sheet_region,
            plane_z_um=plane_z_um,
            side="bottom",
        )
        top_regions = _route_a_sheet_side_solution_regions(
            build_input,
            sheet=sheet,
            sheet_region=sheet_region,
            plane_z_um=plane_z_um,
            side="top",
        )
        pair_region: tuple[Any, ...] = ()
        has_distinct_side_domains = False
        for bottom_id, bottom_region in bottom_regions:
            for top_id, top_region in top_regions:
                overlap = boolean_polygon_region(
                    gdstk,
                    bottom_region,
                    top_region,
                    "and",
                )
                if not overlap:
                    continue
                has_distinct_side_domains |= bottom_id != top_id
                refs = geometry_refs_from_gdstk_region(parent_geometry_ref, overlap)
                for geometry_ref in refs:
                    raw_patches.append(
                        (interface, sheet, bottom_id, top_id, geometry_ref)
                    )
                pair_region = (
                    overlap
                    if not pair_region
                    else boolean_polygon_region(gdstk, pair_region, overlap, "or")
                )
        uncovered = boolean_polygon_region(
            gdstk,
            sheet_region,
            pair_region,
            "not",
        )
        if uncovered:
            raise ValueError(
                f"{sheet.semantic_id} local Route A patches do not cover the full sheet."
            )
        if not has_distinct_side_domains:
            raise ValueError(
                f"{sheet.semantic_id} has no positive-area physical support from "
                "distinct side domains."
            )

    ordered = sorted(
        raw_patches,
        key=lambda item: (
            item[0].interface_id,
            item[2],
            item[3],
            loop_signature(item[4]["outer_loop"]),
        ),
    )
    counts: Counter[str] = Counter(interface.interface_id for interface, *_ in ordered)
    indexes: Counter[str] = Counter()
    records: list[RouteASheetPatch] = []
    for interface, sheet, bottom_id, top_id, geometry_ref in ordered:
        component_index = indexes[interface.interface_id]
        indexes[interface.interface_id] += 1
        patch_suffix = (
            "" if counts[interface.interface_id] == 1 else f":{component_index:04d}"
        )
        patch_prefix = "representation-local" if route is None else "route-a-local"
        patch_id = f"{patch_prefix}:{interface.interface_id}{patch_suffix}"
        bottom = conductor_solution_evidence(
            semantic_facts,
            contribution_id=f"{patch_id}:bottom:{bottom_id}",
            patch_id=f"planned:{patch_id}:bottom",
            conductor_id=sheet.semantic_id,
            solution_id=bottom_id,
            side="bottom",
        )
        top = conductor_solution_evidence(
            semantic_facts,
            contribution_id=f"{patch_id}:top:{top_id}",
            patch_id=f"planned:{patch_id}:top",
            conductor_id=sheet.semantic_id,
            solution_id=top_id,
            side="top",
        )
        records.append(
            RouteASheetPatch(
                parent_interface_id=interface.interface_id,
                sheet_entity_id=sheet.semantic_id,
                patch_id=patch_id,
                geometry_ref=geometry_ref,
                bottom=bottom,
                top=top,
            )
        )
    return tuple(records)


def _route_a_sheet_side_solution_regions(
    build_input: GeometryBuildInput,
    *,
    sheet: SemanticEntitySpec,
    sheet_region: tuple[Any, ...],
    plane_z_um: float,
    side: str,
) -> tuple[tuple[str, tuple[Any, ...]], ...]:
    """Return nonoverlapping exact solution coverage for one sheet side."""
    return planar_side_solution_regions(
        build_input,
        owner_id=sheet.semantic_id,
        occupied_region=sheet_region,
        plane_z_um=plane_z_um,
        side=side,
    )


class SheetSurfacePlanner:
    """Share one sheet's local patches and contact masks across live surface steps."""

    def __init__(
        self,
        scope: SurfaceScope,
        interface: InterfacePlanRecord,
        patches: Sequence[RouteASheetPatch],
        sheet_contacts: Sequence[MMContactRecord],
    ):
        self.scope = scope
        self.interface = interface
        self.patches = patches
        self.sheet_contacts = sheet_contacts

    def plan(self) -> tuple[SurfacePlanRecord, ...]:
        """Match contact caps before emitting remaining sheet contributions."""
        import gdstk  # noqa: PLC0415 - optional native polygon dependency

        self.gdstk = gdstk
        self.sheet = entity_by_id(
            self.scope.build_input, self.interface.owner_semantic_ids[0]
        )
        self.contact_loops_by_patch: dict[
            str, list[tuple[tuple[float, float], ...]]
        ] = {patch.patch_id: [] for patch in self.patches}
        self.records: list[SurfacePlanRecord] = []
        self.plan_contact_caps()
        self.plan_live_patches()
        return tuple(self.records)

    def plan_contact_caps(self):
        """Bind every typed contact to its one ordered local domain pair."""
        for contact_index, contact in enumerate(self.sheet_contacts):
            contact_region = (
                self.gdstk.Polygon(clean_loop(cast(PolygonRing, contact.outer_loop))),
            )
            matches: list[RouteASheetPatch] = []
            for patch in self.patches:
                overlap = boolean_polygon_region(
                    self.gdstk,
                    contact_region,
                    gdstk_surface_region(patch.geometry_ref),
                    "and",
                )
                if overlap:
                    matches.append(patch)
            if len(matches) != 1:
                raise ValueError(
                    f"{contact.contact_id} crosses {len(matches)} local Route A "
                    "sheet patches; one contact requires one ordered domain pair."
                )
            patch = matches[0]
            uncovered = boolean_polygon_region(
                self.gdstk,
                contact_region,
                gdstk_surface_region(patch.geometry_ref),
                "not",
            )
            if uncovered:
                raise ValueError(
                    f"{contact.contact_id} is not contained by its local Route A patch."
                )
            self.contact_loops_by_patch[patch.patch_id].append(
                clean_loop(cast(PolygonRing, contact.outer_loop))
            )
            cap_face = route_a_sheet_contact_cap_face(
                self.scope.build_input,
                sheet_entity=self.sheet,
                contact=contact,
            )
            parent_evidence = patch.bottom if cap_face == "bottom" else patch.top
            cap_evidence = _route_a_sheet_child_evidence(
                self.scope.semantic_facts,
                patch=patch,
                parent=parent_evidence,
                child_label=f"contact-cap:{contact.contact_id}",
            )
            cap_solution_id = _one_effective_domain(cap_evidence)
            cap_kind = cap_evidence.classification
            cap_owner_ids = cap_evidence.source_owner_ids
            cap_interface_id = f"{cap_kind}__{self.sheet.semantic_id}__SHEET_CONTACT_CAP__{contact_index:04d}"
            cap_geometry_ref = {
                **dict(patch.geometry_ref),
                "outer_loop": clean_loop(cast(PolygonRing, contact.outer_loop)),
                "hole_loops": (),
                "source_polygon_ids": unique_ids(
                    (
                        *contact.lower_source_fragment_ids,
                        *contact.upper_source_fragment_ids,
                    )
                ),
            }
            self.records.append(
                SurfacePlanRecord(
                    surface_id=f"SURF__{cap_interface_id}",
                    owner_semantic_id=self.sheet.semantic_id,
                    surface_role=(
                        "representation_context_sheet_contact_cap"
                        if self.scope.route is None
                        and self.scope.representation_context is not None
                        else "route_a_sheet_contact_cap"
                    ),
                    geometry_ref=cap_geometry_ref,
                    interface_id=cap_interface_id,
                    valid_routes=()
                    if self.scope.route is None
                    else (self.scope.route,),
                    metadata={
                        "physical_name": (
                            f"{cap_kind}__{entity_physical_group_id(self.sheet)}"
                            "__SHEET_CONTACT_CAP"
                        ),
                        "interface_kinds": (cap_kind,),
                        "owner_semantic_ids": cap_owner_ids,
                        "physical_owner_semantic_ids": physical_group_owner_ids(
                            self.scope.build_input, cap_owner_ids
                        ),
                        "boundary_volume_ids": (cap_solution_id,),
                        "embedded_surface_sheet": True,
                        "exposed_surface_role": "sheet_contact_cap",
                        "sheet_contact_cap": True,
                        "source_contact_id": contact.contact_id,
                        "source_contact_owner_semantic_ids": (
                            contact.lower_entity_id,
                            contact.upper_entity_id,
                        ),
                        "source_provenance": surface_contribution_provenance(
                            parent_interface_id=self.interface.interface_id,
                            patch_id=patch.patch_id,
                            contributions=(cap_evidence,),
                        ),
                    },
                )
            )

    def plan_live_patches(self):
        """Subtract contact caps and retain both side contributions on live children."""
        candidates: list[
            tuple[
                RouteASheetPatch, dict[str, Any], tuple[EvidenceResult, EvidenceResult]
            ]
        ] = []
        for patch in self.patches:
            geometry_refs = subtract_contact_patches_from_face(
                patch.geometry_ref,
                self.contact_loops_by_patch[patch.patch_id],
            )
            for child_index, geometry_ref in enumerate(geometry_refs):
                child_label = f"live:{child_index:04d}"
                bottom = _route_a_sheet_child_evidence(
                    self.scope.semantic_facts,
                    patch=patch,
                    parent=patch.bottom,
                    child_label=child_label,
                )
                top = _route_a_sheet_child_evidence(
                    self.scope.semantic_facts,
                    patch=patch,
                    parent=patch.top,
                    child_label=child_label,
                )
                candidates.append((patch, geometry_ref, (bottom, top)))
        interface_ids = [
            _route_a_sheet_patch_interface_id(self.interface, patch, contributions)
            for patch, _, contributions in candidates
        ]
        interface_counts = Counter(interface_ids)
        interface_indexes: Counter[str] = Counter()
        for (patch, geometry_ref, contributions), surface_interface_id in zip(
            candidates, interface_ids, strict=True
        ):
            child_index = interface_indexes[surface_interface_id]
            interface_indexes[surface_interface_id] += 1
            surface_id = f"SURF__{surface_interface_id}"
            parent_surface_id = None
            partition_label = None
            if interface_counts[surface_interface_id] > 1:
                parent_surface_id = surface_id
                partition_label = f"LOCAL_{child_index:04d}"
                surface_id = f"{surface_id}__{partition_label}"
            interface_kinds = route_a_sheet_patch_interface_kinds(
                self.interface,
                contributions,
            )
            owner_ids = unique_ids(
                owner_id
                for result in contributions
                for owner_id in result.source_owner_ids
            )
            self.records.append(
                SurfacePlanRecord(
                    surface_id=surface_id,
                    owner_semantic_id=self.sheet.semantic_id,
                    surface_role=(
                        "representation_context_sheet_interface"
                        if self.scope.route is None
                        and self.scope.representation_context is not None
                        else "A_planned_interface"
                    ),
                    geometry_ref=dict(geometry_ref),
                    interface_id=surface_interface_id,
                    parent_surface_id=parent_surface_id,
                    partition_label=partition_label,
                    solver_use=self.interface.solver_use or "solver_active",
                    valid_routes=()
                    if self.scope.route is None
                    else (self.scope.route,),
                    metadata={
                        "interface_kinds": interface_kinds,
                        "owner_semantic_ids": owner_ids,
                        "physical_owner_semantic_ids": physical_group_owner_ids(
                            self.scope.build_input, owner_ids
                        ),
                        "boundary_volume_ids": patch.boundary_volume_ids,
                        "embedded_surface_sheet": True,
                        "source_provenance": surface_contribution_provenance(
                            parent_interface_id=self.interface.interface_id,
                            patch_id=patch.patch_id,
                            contributions=contributions,
                        ),
                    },
                )
            )


def _route_a_sheet_child_evidence(
    semantic_facts: SemanticEvidenceFacade,
    *,
    patch: RouteASheetPatch,
    parent: EvidenceResult,
    child_label: str,
) -> EvidenceResult:
    solution_id = _one_effective_domain(parent)
    return conductor_solution_evidence(
        semantic_facts,
        contribution_id=(f"{patch.patch_id}:{child_label}:{parent.side}:{solution_id}"),
        patch_id=f"planned:{patch.patch_id}:{child_label}:{parent.side}",
        conductor_id=patch.sheet_entity_id,
        solution_id=solution_id,
        side=parent.side or "unknown",
    )


def _one_effective_domain(result: EvidenceResult) -> str:
    if result.effective_domain_ids is None or len(result.effective_domain_ids) != 1:
        raise ValueError(
            f"{result.contribution_id} requires one effective solution domain."
        )
    return result.effective_domain_ids[0]


def _route_a_sheet_patch_interface_id(
    interface: InterfacePlanRecord,
    patch: RouteASheetPatch,
    contributions: tuple[EvidenceResult, EvidenceResult],
) -> str:
    preferred_domain = interface.owner_semantic_ids[1]
    primary_kind = next(
        (
            result.classification
            for result in contributions
            if preferred_domain in (result.effective_domain_ids or ())
        ),
        contributions[0].classification,
    )
    suffix = interface.interface_id.rsplit("__", 1)[-1]
    return (
        f"{primary_kind}__{patch.sheet_entity_id}__"
        f"{'__'.join(patch.boundary_volume_ids)}__{suffix}"
    )


def route_a_sheet_patch_interface_kinds(
    interface: InterfacePlanRecord,
    contributions: Sequence[EvidenceResult],
) -> tuple[str, ...]:
    """Keep authored interface kinds consistent with both local side contributions."""
    derived = {result.classification for result in contributions}
    raw = interface.metadata.get("interface_kinds")
    if (
        interface.metadata.get("intent_origin")
        not in {
            "generated_route_a_surface_sheet",
            "generated_representation_context_surface_sheet",
        }
        and raw is not None
    ):
        declared = {str(raw)} if isinstance(raw, str) else {str(value) for value in raw}
        if declared != derived:
            raise ValueError(
                f"{interface.interface_id} explicit interface kinds "
                f"{sorted(declared)!r} contradict local evidence {sorted(derived)!r}."
            )
    return tuple(kind for kind in INTERFACE_KIND_ORDER if kind in derived)
