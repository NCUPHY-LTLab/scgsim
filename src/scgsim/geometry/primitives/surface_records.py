"""Shared compiler surface-record projection and derived sheet-region facts. Canonical definitions prevent stage import cycles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from scgsim.geometry.models.common import RouteLiteral
from scgsim.geometry.models.input import SemanticEntitySpec
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.models.topology import InterfacePlanRecord, SurfacePlanRecord
from scgsim.geometry.primitives.constants import INTERFACE_KIND_ORDER
from scgsim.geometry.primitives.entities import unique_ids
from scgsim.geometry.primitives.spatial import coordinate_key
from scgsim.semantics import EvidenceResult


@dataclass(frozen=True, slots=True)
class RouteASheetPatch:
    """One exact sheet region with one ordered solution domain on each side."""

    parent_interface_id: str
    sheet_entity_id: str
    patch_id: str
    geometry_ref: Mapping[str, Any]
    bottom: EvidenceResult
    top: EvidenceResult

    @property
    def boundary_volume_ids(self) -> tuple[str, str]:
        """The single effective domain id of the bottom and of the top side."""
        bottom_ids = self.bottom.effective_domain_ids or ()
        top_ids = self.top.effective_domain_ids or ()
        if len(bottom_ids) != 1 or len(top_ids) != 1:
            raise ValueError(f"{self.patch_id} lacks one effective domain per side")
        return (bottom_ids[0], top_ids[0])

    @property
    def contributions(self) -> tuple[EvidenceResult, EvidenceResult]:
        """The (bottom, top) evidence results as a pair."""
        return (self.bottom, self.top)


def surface_contribution_provenance(
    *,
    parent_interface_id: str,
    patch_id: str,
    contributions: Sequence[EvidenceResult],
) -> dict[str, Any]:
    """Dict describing contribution evidence, native aggregation and mask support."""
    ledger = tuple(
        {
            "contribution_id": result.contribution_id,
            "patch_ids": result.patch_ids,
            "classification": result.classification,
            "source_owner_ids": result.source_owner_ids,
            "aggregate_owner_ids": result.aggregate_owner_ids,
            "material_id": result.material_id,
            "material_kind": result.material_kind,
            "effective_domain_ids": result.effective_domain_ids,
            "side": result.side,
            "sheet_owner_id": result.sheet_owner_id,
            "outer_source_ids": result.outer_source_ids,
            "hole_source_ids": result.hole_source_ids,
            "seam_source_ids": result.seam_source_ids,
            "evidence_stages": result.evidence_stages,
            "observation_hashes": result.observation_hashes,
            "snapshot_reference": result.snapshot_reference.detached(),
        }
        for result in contributions
    )
    classifications = tuple(result.classification for result in contributions)
    surface_epr_classifications = tuple(
        classification
        for classification in classifications
        if classification in {"MA", "MS", "SA"}
    )
    unique_surface_epr = tuple(dict.fromkeys(surface_epr_classifications))
    aggregate_once = (
        len(surface_epr_classifications) > 1 and len(unique_surface_epr) == 1
    )
    aggregation_policy = (
        "semantic_only_no_native_row"
        if not unique_surface_epr
        else "shared_native_surface_once"
        if aggregate_once
        else "one_row_per_interface_kind"
    )
    return {
        "surface_contribution_schema": "scgsim.surface-contributions.v1",
        "parent_interface_id": parent_interface_id,
        "local_patch_id": patch_id,
        "surface_contribution_ledger": ledger,
        "native_aggregation": {
            "policy": aggregation_policy,
            "interface_kinds": tuple(dict.fromkeys(classifications)),
            "surface_epr_interface_kinds": unique_surface_epr,
            "contribution_count": len(ledger),
            "native_row_count": 1 if aggregate_once else len(unique_surface_epr),
            "side_resolved_numeric_output": False,
        },
        "mask_support": {
            "zero_inset_supported": True,
            "positive_inset_supported": bool(unique_surface_epr) and not aggregate_once,
            "unsupported_reason": (
                "same-kind side contributions share one native aggregate surface"
                if aggregate_once
                else "semantic contribution has no Surface EPR native row"
                if not unique_surface_epr
                else None
            ),
            "sides": tuple(result.side for result in contributions),
        },
    }


def interface_kinds(interface: InterfacePlanRecord) -> tuple[str, ...]:
    """Declared plus own interface kinds, in INTERFACE_KIND_ORDER."""
    raw = interface.metadata.get("interface_kinds", (interface.kind,))
    if isinstance(raw, str):
        raw_values = (raw,)
    elif isinstance(raw, tuple | list | set):
        raw_values = raw
    else:
        raw_values = (interface.kind,)

    seen = {str(value) for value in raw_values}
    seen.add(interface.kind)
    return tuple(kind for kind in INTERFACE_KIND_ORDER if kind in seen)


def is_route_a_sheet_interface(
    route: RouteLiteral | None,
    interface: InterfacePlanRecord,
    *,
    representation_context: RepresentationContext | None = None,
) -> bool:
    """True when the interface is a Route A or representation-context sheet polygon."""
    recognition_rule = interface.metadata.get("recognition_rule")
    if route == "A" and recognition_rule == "route_a_surface_sheet_polygon":
        return True
    if route is None and representation_context is not None:
        generated = (
            interface.metadata.get("intent_origin")
            == "generated_representation_context_surface_sheet"
            and recognition_rule == "representation_context_surface_sheet_polygon"
        )
        owner_id = interface.owner_semantic_ids[0]
        owner = representation_context.entities.get(owner_id)
        return generated and owner is not None and owner.representation == "sheet"
    return False


def canonical_face_signature_3d(
    ring: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], ...] | None:
    """Canonical ring key: smallest rotation either way; None under 3 points."""
    if len(ring) < 3:
        return None
    normalized = tuple(coordinate_key(point) for point in ring)
    candidates: list[tuple[tuple[float, float, float], ...]] = []
    for candidate in (normalized, tuple(reversed(normalized))):
        rotations = tuple(
            candidate[index:] + candidate[:index] for index in range(len(candidate))
        )
        candidates.append(min(rotations))
    return min(candidates)


def conductor_boundary_surface_id(
    route: RouteLiteral | None,
    entity: SemanticEntitySpec,
    representation: str,
    shell_part: str,
) -> str:
    """Surface id built from route, SHELL or MAT kind, entity id and shell part."""
    surface_kind = "SHELL" if representation == "cutout_boundary_shell" else "MAT"
    route_id = route or "REPRESENTATION"
    return (
        f"SURF__{route_id}__{surface_kind}__{entity.semantic_id}__{shell_part.upper()}"
    )


def surface_boundary_volume_ids(
    surface: SurfacePlanRecord,
    *,
    known_entity_ids: set[str],
) -> tuple[str, ...]:
    """Known entity ids bounding a surface, at most two, or () for embedded sheets."""
    if surface.surface_role == "lumped_port" and not surface.metadata.get(
        "route_a_boundary_port"
    ):
        return ()
    if surface.metadata.get("embedded_surface_sheet") and not surface.metadata.get(
        "sheet_contact_cap"
    ):
        return ()
    raw = surface.metadata.get(
        "boundary_volume_ids",
        surface.metadata.get("owner_semantic_ids", (surface.owner_semantic_id,)),
    )
    if isinstance(raw, str):
        values = (raw,)
    else:
        values = tuple(str(value) for value in raw)
    result = unique_ids(value for value in values if value in known_entity_ids)
    if len(result) > 2:
        raise ValueError(
            f"{surface.surface_id} belongs to more than two volumes: {result!r}"
        )
    return result
