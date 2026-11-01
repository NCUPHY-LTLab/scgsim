"""Build the surfaces of a Route A sheet from its patches and metal-metal contacts.

Each contact on a sheet becomes a cap surface; the rest of every patch becomes a live
(exposed) surface classified by the solution entities on its two sides, as MS (metal
substrate), MA (metal air) and similar kinds. Does not partition the sheet; see
``sheet_patches``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any

from scgsim.geometry._primitives.constants import (
    _INTERFACE_KIND_ORDER,
    _TOPOLOGY_EPS_UM,
)
from scgsim.geometry._primitives.entities import _entity_by_id, _unique_ids
from scgsim.geometry._primitives.loops import (
    _boolean_gdstk_region,
    _clean_loop,
    _gdstk_surface_region,
)
from scgsim.geometry._primitives.surface_records import (
    _RouteASheetPatch,
    _surface_contribution_provenance,
)
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.topology import (
    InterfacePlanRecord,
    MMContactRecord,
    SurfacePlanRecord,
)
from scgsim.geometry.planning.evidence import conductor_solution_evidence
from scgsim.geometry.planning.surface_contacts import (
    _subtract_contact_patches_from_face,
)
from scgsim.geometry.planning.surface_scope import SurfaceScope
from scgsim.geometry.planning.tags import (
    _entity_physical_group_id,
    _physical_group_owner_ids,
)
from scgsim.semantics import EvidenceResult, SemanticEvidenceFacade


def _route_a_sheet_patch_surfaces(
    scope: SurfaceScope,
    *,
    interface: InterfacePlanRecord,
    patches: Sequence[_RouteASheetPatch],
    sheet_contacts: Sequence[MMContactRecord],
) -> tuple[SurfacePlanRecord, ...]:
    """Lower one sheet's local adjacency without assigning one pair to its whole area.

    Each metal-metal contact on the sheet becomes a cap surface on the one patch that
    contains it. The contact footprints are then cut out of their patches, and what
    remains of each patch becomes a live (exposed) surface.
    """
    sheet = _entity_by_id(scope.build_input, interface.owner_semantic_ids[0])
    contact_loops_by_patch: dict[str, list[tuple[tuple[float, float], ...]]] = {
        patch.patch_id: [] for patch in patches
    }
    records: list[SurfacePlanRecord] = []
    for contact_index, contact in enumerate(sheet_contacts):
        patch = _patch_containing_contact(contact=contact, patches=patches)
        contact_loops_by_patch[patch.patch_id].append(_clean_loop(contact.outer_loop))
        records.append(
            _sheet_contact_cap_record(
                scope,
                interface=interface,
                patch=patch,
                contact=contact,
                contact_index=contact_index,
            )
        )
    candidates = _live_sheet_patch_candidates(
        scope.semantic_facts,
        patches=patches,
        contact_loops_by_patch=contact_loops_by_patch,
    )
    records.extend(
        _live_sheet_patch_records(
            scope,
            interface=interface,
            sheet=sheet,
            candidates=candidates,
        )
    )
    return tuple(records)


def _patch_containing_contact(
    contact: MMContactRecord,
    patches: Sequence[_RouteASheetPatch],
) -> _RouteASheetPatch:
    """Return the one local patch that fully contains the contact footprint."""
    import gdstk

    contact_region = (gdstk.Polygon(_clean_loop(contact.outer_loop)),)
    matches: list[_RouteASheetPatch] = []
    for patch in patches:
        overlap = _boolean_gdstk_region(
            gdstk,
            contact_region,
            _gdstk_surface_region(patch.geometry_ref),
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
    uncovered = _boolean_gdstk_region(
        gdstk,
        contact_region,
        _gdstk_surface_region(patch.geometry_ref),
        "not",
    )
    if uncovered:
        raise ValueError(
            f"{contact.contact_id} is not contained by its local Route A patch."
        )
    return patch


def _sheet_contact_cap_record(
    scope: SurfaceScope,
    *,
    interface: InterfacePlanRecord,
    patch: _RouteASheetPatch,
    contact: MMContactRecord,
    contact_index: int,
) -> SurfacePlanRecord:
    """Build the cap surface that closes a sheet where a bump or pad contacts it."""
    sheet = _entity_by_id(scope.build_input, interface.owner_semantic_ids[0])
    cap_face = _route_a_sheet_contact_cap_face(
        scope.build_input,
        sheet_entity=sheet,
        contact=contact,
    )
    parent_evidence = patch.bottom if cap_face == "bottom" else patch.top
    cap_evidence = _route_a_sheet_child_evidence(
        scope.semantic_facts,
        patch=patch,
        parent=parent_evidence,
        child_label=f"contact-cap:{contact.contact_id}",
    )
    cap_solution_id = _one_effective_domain(cap_evidence)
    cap_kind = cap_evidence.classification
    cap_owner_ids = cap_evidence.source_owner_ids
    cap_interface_id = (
        f"{cap_kind}__{sheet.semantic_id}__SHEET_CONTACT_CAP__{contact_index:04d}"
    )
    cap_geometry_ref = {
        **dict(patch.geometry_ref),
        "outer_loop": _clean_loop(contact.outer_loop),
        "hole_loops": (),
        "source_polygon_ids": _unique_ids(
            (
                *contact.lower_source_fragment_ids,
                *contact.upper_source_fragment_ids,
            )
        ),
    }
    return SurfacePlanRecord(
        surface_id=f"SURF__{cap_interface_id}",
        owner_semantic_id=sheet.semantic_id,
        surface_role=(
            "representation_context_sheet_contact_cap"
            if scope.context_planning
            else "route_a_sheet_contact_cap"
        ),
        geometry_ref=cap_geometry_ref,
        interface_id=cap_interface_id,
        valid_routes=() if scope.route is None else (scope.route,),
        metadata={
            "physical_name": (
                f"{cap_kind}__{_entity_physical_group_id(sheet)}__SHEET_CONTACT_CAP"
            ),
            "interface_kinds": (cap_kind,),
            "owner_semantic_ids": cap_owner_ids,
            "physical_owner_semantic_ids": _physical_group_owner_ids(
                scope.build_input, cap_owner_ids
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
            "source_provenance": _surface_contribution_provenance(
                parent_interface_id=interface.interface_id,
                patch_id=patch.patch_id,
                contributions=(cap_evidence,),
            ),
        },
    )


def _live_sheet_patch_candidates(
    semantic_facts: SemanticEvidenceFacade,
    *,
    patches: Sequence[_RouteASheetPatch],
    contact_loops_by_patch: Mapping[str, Sequence[tuple[tuple[float, float], ...]]],
) -> list[
    tuple[_RouteASheetPatch, dict[str, Any], tuple[EvidenceResult, EvidenceResult]]
]:
    """Cut contact footprints out of every patch and classify each remaining piece."""
    candidates: list[
        tuple[_RouteASheetPatch, dict[str, Any], tuple[EvidenceResult, EvidenceResult]]
    ] = []
    for patch in patches:
        geometry_refs = _subtract_contact_patches_from_face(
            patch.geometry_ref,
            contact_loops_by_patch[patch.patch_id],
        )
        for child_index, geometry_ref in enumerate(geometry_refs):
            child_label = f"live:{child_index:04d}"
            bottom = _route_a_sheet_child_evidence(
                semantic_facts,
                patch=patch,
                parent=patch.bottom,
                child_label=child_label,
            )
            top = _route_a_sheet_child_evidence(
                semantic_facts,
                patch=patch,
                parent=patch.top,
                child_label=child_label,
            )
            candidates.append((patch, geometry_ref, (bottom, top)))
    return candidates


def _live_sheet_patch_records(
    scope: SurfaceScope,
    *,
    interface: InterfacePlanRecord,
    sheet: SemanticEntitySpec,
    candidates: Sequence[
        tuple[_RouteASheetPatch, dict[str, Any], tuple[EvidenceResult, EvidenceResult]]
    ],
) -> list[SurfacePlanRecord]:
    """Build the exposed sheet surfaces; pieces that share an interface id become partitions."""
    interface_ids = [
        _route_a_sheet_patch_interface_id(interface, patch, contributions)
        for patch, _, contributions in candidates
    ]
    interface_counts = Counter(interface_ids)
    interface_indexes: Counter[str] = Counter()
    records: list[SurfacePlanRecord] = []
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
        interface_kinds = _route_a_sheet_patch_interface_kinds(
            interface,
            contributions,
        )
        owner_ids = _unique_ids(
            owner_id for result in contributions for owner_id in result.source_owner_ids
        )
        records.append(
            SurfacePlanRecord(
                surface_id=surface_id,
                owner_semantic_id=sheet.semantic_id,
                surface_role=(
                    "representation_context_sheet_interface"
                    if scope.context_planning
                    else "A_planned_interface"
                ),
                geometry_ref=dict(geometry_ref),
                interface_id=surface_interface_id,
                parent_surface_id=parent_surface_id,
                partition_label=partition_label,
                solver_use=interface.solver_use or "solver_active",
                valid_routes=() if scope.route is None else (scope.route,),
                metadata={
                    "interface_kinds": interface_kinds,
                    "owner_semantic_ids": owner_ids,
                    "physical_owner_semantic_ids": _physical_group_owner_ids(
                        scope.build_input, owner_ids
                    ),
                    "boundary_volume_ids": patch.boundary_volume_ids,
                    "embedded_surface_sheet": True,
                    "source_provenance": _surface_contribution_provenance(
                        parent_interface_id=interface.interface_id,
                        patch_id=patch.patch_id,
                        contributions=contributions,
                    ),
                },
            )
        )
    return records


def _route_a_sheet_child_evidence(
    semantic_facts: SemanticEvidenceFacade,
    *,
    patch: _RouteASheetPatch,
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
    patch: _RouteASheetPatch,
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


def _route_a_sheet_patch_interface_kinds(
    interface: InterfacePlanRecord,
    contributions: Sequence[EvidenceResult],
) -> tuple[str, ...]:
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
    return tuple(kind for kind in _INTERFACE_KIND_ORDER if kind in derived)


def _route_a_sheet_contact_cap_face(
    build_input: GeometryBuildInput,
    *,
    sheet_entity: SemanticEntitySpec,
    contact: MMContactRecord,
) -> str:
    """Select the sheet side opposite its typed finite MM contact direction."""
    entities = {entity.semantic_id: entity for entity in build_input.entities}
    lower = entities.get(contact.lower_entity_id)
    upper = entities.get(contact.upper_entity_id)
    if (
        lower is None
        or upper is None
        or sheet_entity.semantic_id
        not in {
            contact.lower_entity_id,
            contact.upper_entity_id,
        }
    ):
        raise ValueError(f"{contact.contact_id} has ambiguous sheet contact members")
    normal = tuple(float(value) for value in contact.normal)
    if len(normal) != 3 or not all(isfinite(value) for value in normal):
        raise ValueError(f"{contact.contact_id} has ambiguous sheet contact normal")
    roles = {lower.part_role, upper.part_role}
    if roles == {"face_metal", "contact_pad"}:
        # A typed contact_pad is the same-z under-sheet normalization authority.
        return "bottom"
    if roles != {"face_metal", "bump_body"}:
        raise ValueError(f"{contact.contact_id} has incompatible sheet contact roles")
    if (
        abs(normal[0]) > _TOPOLOGY_EPS_UM
        or abs(normal[1]) > _TOPOLOGY_EPS_UM
        or abs(normal[2]) <= _TOPOLOGY_EPS_UM
    ):
        raise ValueError(f"{contact.contact_id} has ambiguous sheet contact normal")
    if sheet_entity.semantic_id == lower.semantic_id:
        contact_face = "top" if normal[2] > 0 else "bottom"
        source_face_id = contact.lower_source_face_id
        expected_source_face_id = f"{lower.semantic_id}__{contact_face}"
    else:
        contact_face = "bottom" if normal[2] > 0 else "top"
        source_face_id = contact.upper_source_face_id
        expected_source_face_id = f"{upper.semantic_id}__{contact_face}"
    expected_lower_face_id = (
        f"{lower.semantic_id}__{'top' if normal[2] > 0 else 'bottom'}"
    )
    expected_upper_face_id = (
        f"{upper.semantic_id}__{'bottom' if normal[2] > 0 else 'top'}"
    )
    if (
        source_face_id != expected_source_face_id
        or contact.lower_source_face_id != expected_lower_face_id
        or contact.upper_source_face_id != expected_upper_face_id
    ):
        raise ValueError(f"{contact.contact_id} has inconsistent sheet contact face")
    return "bottom" if contact_face == "top" else "top"
