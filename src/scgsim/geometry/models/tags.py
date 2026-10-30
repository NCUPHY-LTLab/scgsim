"""Canonical geometry records. Keep source facts distinct from compiler records; reexports share these exact definitions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from scgsim.geometry.models.common import (
    DimensionLiteral,
    GmshDimTag,
    RouteLiteral,
    SolverUseLiteral,
    TagSourceKindLiteral,
)


@dataclass(frozen=True)
class TagPlanRecord:
    """Solver-neutral physical tag plan before backend entity tags exist.

    Tags are planned before OCC construction. `source_record_kind` and
    `source_record_id` must point to a live `SurfacePlanRecord` or
    `VolumePlanRecord`. After backend construction, the same source id is used
    to recover OCC dim-tags and create final physical groups. Together with
    `BackendEntityTagRecord`, this is the tag ledger: plan ids are stable,
    OCC tags are lowering results, and physical names are solver-facing labels.
    """

    physical_name: str
    dimension: DimensionLiteral
    source_record_kind: TagSourceKindLiteral
    source_record_id: str
    role: str
    solver_use: SolverUseLiteral = "solver_active"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected_dimension = 2 if self.source_record_kind == "surface" else 3
        if self.dimension != expected_dimension:
            raise ValueError(
                f"{self.source_record_kind} tag {self.physical_name!r} "
                f"must use dimension {expected_dimension}"
            )


@dataclass(frozen=True)
class BackendEntityTagRecord:
    """Backend dim-tag recovered for one live planned source record.

    Two live source ids mapping to the same backend dim-tag means
    canonicalization failed earlier; do not accept that as a backend shortcut.
    """

    source_record_kind: TagSourceKindLiteral
    source_record_id: str
    dim_tag: GmshDimTag
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FinalPhysicalGroupRecord:
    """Final solver-neutral group plan after backend tags are known.

    `physical_name` is part of the downstream contract. Interface surface names
    currently carry the reviewed grammar consumed by gsim, for example
    `MA__...`, `MS__...`, `SA__...`, and `MS_MA__...`.
    """

    physical_name: str
    dimension: int
    route: RouteLiteral
    role: str
    source_record_id: str
    net_id: str | None = None
    solver_use: SolverUseLiteral | None = None
    entity_tags: tuple[int, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.solver_use is None:
            object.__setattr__(self, "solver_use", "solver_active")
        if not self.entity_tags:
            raise ValueError("physical groups require backend entity tags")


@dataclass(frozen=True)
class StructuredFinalPhysicalGroupRecord(FinalPhysicalGroupRecord):
    """A/B-only solver-live structured surface authority."""

    representation: str = ""
    surface_id: str | None = None
    interface_type: str | None = None
    contact_kind: str | None = None
    face_kind: str | None = None
    owner_semantic_ids: tuple[str, ...] = ()
    adjacent_solution_volume_ids: tuple[str, ...] = ()
    conductor_component_id: str | None = None
    equipotential_id: str | None = None
    source_provenance: Mapping[str, Any] = field(default_factory=dict)
    physical_attribute: Mapping[str, Any] = field(default_factory=dict)
    conductor_component_ids: tuple[str, ...] = ()
    conductor_component_ids_by_source: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        components = self.conductor_component_ids
        bindings = self.conductor_component_ids_by_source
        if (
            not isinstance(components, tuple)
            or any(not isinstance(value, str) or not value for value in components)
            or tuple(sorted(set(components))) != components
        ):
            raise ValueError(
                "conductor_component_ids must contain sorted unique nonempty IDs"
            )
        if not isinstance(bindings, Mapping) or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
            for key, value in bindings.items()
        ):
            raise ValueError(
                "component source bindings require nonempty source and component IDs"
            )
        if not components and not bindings:
            # Existing singleton records remain readable without additive fields.
            return
        if not components or not bindings or set(bindings.values()) != set(components):
            raise ValueError("component source bindings must cover every component exactly")
        source_ids = self.source_provenance.get("source_record_ids")
        if not isinstance(source_ids, (tuple, list)) or (
            any(not isinstance(value, str) or not value for value in source_ids)
            or len(source_ids) != len(set(source_ids))
            or set(source_ids) != set(bindings)
        ):
            raise ValueError(
                "component source bindings must cover every grouped source exactly"
            )
        expected_singular = components[0] if len(components) == 1 else None
        if self.conductor_component_id != expected_singular:
            raise ValueError("singular and plural conductor component identities disagree")
        if len(components) > 1 and (
            not isinstance(self.net_id, str)
            or not self.net_id
        ):
            raise ValueError(
                "multiple conductor components require one Net identity"
            )
        if self.equipotential_id is not None and (
            not isinstance(self.equipotential_id, str) or not self.equipotential_id
        ):
            raise ValueError("equipotential_id must be a nonempty string when provided")
        if len(components) > 1:
            sources = self.source_provenance.get("sources")
            if (
                not isinstance(sources, (tuple, list))
                or len(sources) != len(source_ids)
                or any(
                    not isinstance(source, Mapping)
                    or source.get("surface_contribution_schema")
                    != "scgsim.surface-contributions.v1"
                    or not isinstance(
                        source.get("surface_contribution_ledger"), (tuple, list)
                    )
                    or not source["surface_contribution_ledger"]
                    for source in sources
                )
            ):
                raise ValueError(
                    "multiple conductor components require complete source contribution ledgers"
                )
