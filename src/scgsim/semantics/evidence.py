"""Validate sourced-patch evidence and deduplicate logical contributions.

Observation payloads stay dynamically typed at this serialized-data boundary;
``freeze_plain`` validates and detaches them before any result uses their hash.
The semantic identity tuple is heterogeneous because it compares all identity
fields together without changing their serialized or scalar types.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

from scgsim.refusal import refusal

from .interfaces import (
    conductor_solution_interface_kind,
    solution_interface_kind,
    solution_interface_owner_ids,
)
from .snapshot import (
    SemanticFactsSnapshot,
    SemanticSnapshotReference,
    canonical_sha256,
    freeze_plain,
)

EvidenceStage = Literal["declared", "planned", "observed"]
_STAGE_ORDER: tuple[EvidenceStage, ...] = ("declared", "planned", "observed")
_INTERFACE_INTENTS = (
    "conductor_solution",
    "solution_solution",
    "metal_metal_contact",
    "unknown",
)


@dataclass(frozen=True, slots=True)
class SourcedPatch:
    """Immutable patch evidence with its semantic identity and source facts."""

    patch_id: str
    contribution_id: str
    interface_intent: str
    source_owner_ids: tuple[str, ...]
    aggregate_owner_ids: tuple[str, ...]
    material_id: str | None
    material_kind: str | None
    effective_domain_ids: tuple[str, ...] | None
    side: str | None
    sheet_owner_id: str | None
    outer_source_ids: tuple[str, ...] | None
    hole_source_ids: tuple[str, ...] | None
    seam_source_ids: tuple[str, ...] | None
    evidence_stage: EvidenceStage
    observation: Mapping[str, Any] | None = None
    snapshot_reference: SemanticSnapshotReference | None = None

    def __post_init__(self) -> None:
        """Freeze sequence and observation inputs at the record boundary."""
        if self.evidence_stage not in _STAGE_ORDER:
            raise ValueError(
                refusal(
                    "evidence_stage must be declared, planned, or observed",
                    found=self.evidence_stage,
                )
            )
        for field_name in ("source_owner_ids", "aggregate_owner_ids"):
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))
        for field_name in (
            "effective_domain_ids",
            "outer_source_ids",
            "hole_source_ids",
            "seam_source_ids",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, tuple(value))
        if self.observation is not None:
            object.__setattr__(self, "observation", freeze_plain(self.observation))
        if self.snapshot_reference is not None and not isinstance(
            self.snapshot_reference, SemanticSnapshotReference
        ):
            raise TypeError(
                refusal(
                    "snapshot_reference must be a SemanticSnapshotReference or None",
                    found=type(self.snapshot_reference).__name__,
                )
            )

    def semantic_identity(self) -> tuple[Any, ...]:
        """Return fields that must agree for patches to share one contribution."""
        return (
            self.contribution_id,
            self.interface_intent,
            self.source_owner_ids,
            self.aggregate_owner_ids,
            self.material_id,
            self.material_kind,
            self.effective_domain_ids,
            self.side,
            self.sheet_owner_id,
            self.outer_source_ids,
            self.hole_source_ids,
            self.seam_source_ids,
        )


@dataclass(frozen=True, slots=True)
class EvidenceResult:
    """One classified contribution with ordered provenance and snapshot identity."""

    contribution_id: str
    patch_ids: tuple[str, ...]
    classification: str
    source_owner_ids: tuple[str, ...]
    aggregate_owner_ids: tuple[str, ...]
    material_id: str | None
    material_kind: str | None
    effective_domain_ids: tuple[str, ...] | None
    side: str | None
    sheet_owner_id: str | None
    outer_source_ids: tuple[str, ...] | None
    hole_source_ids: tuple[str, ...] | None
    seam_source_ids: tuple[str, ...] | None
    evidence_stages: tuple[EvidenceStage, ...]
    observation_hashes: tuple[str, ...]
    snapshot_reference: SemanticSnapshotReference

    def __post_init__(self) -> None:
        """Normalize ordered collections and validate the bound snapshot."""
        if not isinstance(self.snapshot_reference, SemanticSnapshotReference):
            raise TypeError(
                refusal(
                    "snapshot_reference must be a SemanticSnapshotReference",
                    found=type(self.snapshot_reference).__name__,
                )
            )
        for field_name in (
            "patch_ids",
            "source_owner_ids",
            "aggregate_owner_ids",
            "evidence_stages",
            "observation_hashes",
        ):
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))
        for field_name in (
            "effective_domain_ids",
            "outer_source_ids",
            "hole_source_ids",
            "seam_source_ids",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, tuple(value))


@dataclass(frozen=True, slots=True)
class SemanticEvidenceFacade:
    """Read immutable facts and collapse repeated evidence to one contribution."""

    snapshot: SemanticFactsSnapshot
    _patches: tuple[SourcedPatch, ...]

    def __init__(
        self,
        snapshot: SemanticFactsSnapshot,
        patches: Sequence[SourcedPatch] = (),
    ):
        """Bind a snapshot and any patch records for repeatable evaluation."""
        object.__setattr__(self, "snapshot", snapshot)
        object.__setattr__(self, "_patches", tuple(patches))

    def material_kind(self, semantic_id: str) -> str | None:
        """Read a material kind when the entity is present in the snapshot."""
        entity = self.snapshot.facts.get("entities", {}).get(semantic_id)
        return entity.get("material_kind") if isinstance(entity, Mapping) else None

    def evaluate(self, patch: SourcedPatch) -> EvidenceResult:
        """Classify one logical patch at its existing evidence call time."""
        for result in self.results((patch,)):
            if result.contribution_id == patch.contribution_id:
                return result
        raise RuntimeError(f"missing evidence result for {patch.contribution_id!r}")

    def results(
        self, patches: Sequence[SourcedPatch] = ()
    ) -> tuple[EvidenceResult, ...]:
        """Validate references, merge duplicate patches, and classify in ID order."""
        records_to_evaluate = (*self._patches, *tuple(patches))
        for patch in records_to_evaluate:
            _validate_interface_intent(patch)
        expected_reference = self.snapshot.reference()
        for patch in records_to_evaluate:
            _validate_snapshot_reference(patch, expected_reference)
        grouped: dict[str, list[SourcedPatch]] = {}
        for patch in records_to_evaluate:
            grouped.setdefault(patch.contribution_id, []).append(patch)
        results: list[EvidenceResult] = []
        for contribution_id, records in sorted(grouped.items()):
            first = records[0]
            if any(
                record.semantic_identity() != first.semantic_identity()
                for record in records[1:]
            ):
                raise ValueError(
                    f"conflicting semantic evidence for {contribution_id!r}"
                )
            classification, source_owner_ids = _classify_patch(self.snapshot, first)
            stages: tuple[EvidenceStage, ...] = tuple(
                stage
                for stage in _STAGE_ORDER
                if any(record.evidence_stage == stage for record in records)
            )
            patch_ids = tuple(dict.fromkeys(record.patch_id for record in records))
            observation_hashes = tuple(
                dict.fromkeys(
                    canonical_sha256(record.observation)
                    for record in records
                    if record.observation is not None
                )
            )
            results.append(
                EvidenceResult(
                    contribution_id=contribution_id,
                    patch_ids=patch_ids,
                    classification=classification,
                    source_owner_ids=source_owner_ids,
                    aggregate_owner_ids=first.aggregate_owner_ids,
                    material_id=first.material_id,
                    material_kind=first.material_kind,
                    effective_domain_ids=first.effective_domain_ids,
                    side=first.side,
                    sheet_owner_id=first.sheet_owner_id,
                    outer_source_ids=first.outer_source_ids,
                    hole_source_ids=first.hole_source_ids,
                    seam_source_ids=first.seam_source_ids,
                    evidence_stages=stages,
                    observation_hashes=observation_hashes,
                    snapshot_reference=expected_reference,
                )
            )
        return tuple(results)


def _validate_interface_intent(patch: SourcedPatch) -> None:
    if patch.interface_intent not in _INTERFACE_INTENTS:
        raise ValueError(f"unsupported interface intent {patch.interface_intent!r}")


def _validate_snapshot_reference(
    patch: SourcedPatch,
    expected_reference: SemanticSnapshotReference,
) -> None:
    if patch.snapshot_reference is None:
        raise ValueError(
            f"semantic evidence {patch.patch_id!r} has no snapshot reference"
        )
    if patch.snapshot_reference != expected_reference:
        raise ValueError(
            f"semantic evidence {patch.patch_id!r} belongs to a different snapshot"
        )


class _PatchClass(NamedTuple):
    """The interface kind a patch is classified as and its owners, in the order that kind names them."""

    kind: str
    owner_ids: tuple[str, ...]


def _classify_patch(
    snapshot: SemanticFactsSnapshot, patch: SourcedPatch
) -> _PatchClass:
    domains = patch.effective_domain_ids
    if patch.interface_intent == "conductor_solution":
        if domains is None or len(domains) != 1:
            raise ValueError(
                refusal(
                    "conductor_solution evidence requires one effective domain",
                    found=domains,
                )
            )
        material_kind = _material_kind(snapshot, domains[0])
        return _PatchClass(
            conductor_solution_interface_kind(material_kind), patch.source_owner_ids
        )
    if patch.interface_intent == "solution_solution":
        if domains is None or len(domains) != 2:
            raise ValueError(
                refusal(
                    "solution_solution evidence requires two ordered effective domains",
                    found=domains,
                )
            )
        lower_kind = _material_kind(snapshot, domains[0])
        upper_kind = _material_kind(snapshot, domains[1])
        kind = solution_interface_kind(lower_kind, upper_kind)
        return _PatchClass(
            kind,
            solution_interface_owner_ids(
                kind,
                domains[0],
                domains[1],
                lower_is_vacuum=lower_kind == "vacuum",
            ),
        )
    if patch.interface_intent == "metal_metal_contact":
        return _PatchClass("MM", patch.source_owner_ids)
    if patch.interface_intent == "unknown":
        return _PatchClass("unknown", patch.source_owner_ids)
    raise AssertionError(f"unvalidated interface intent {patch.interface_intent!r}")


def _material_kind(snapshot: SemanticFactsSnapshot, semantic_id: str) -> str:
    entity = snapshot.facts.get("entities", {}).get(semantic_id)
    value = entity.get("material_kind") if isinstance(entity, Mapping) else None
    if not isinstance(value, str):
        raise ValueError(
            refusal(f"{semantic_id} has no snapshotted material kind", found=value)
        )
    return value
