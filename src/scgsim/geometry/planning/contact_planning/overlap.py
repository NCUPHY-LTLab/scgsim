"""Validate volumetric conductor overlaps against explicit source attachments."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from scgsim.geometry.models.common import RouteLiteral
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.planning.contact_planning.geometry import contact_patch_loops
from scgsim.geometry.planning.domain_planning.occupancy import (
    entity_occupied_region,
    route_conductor_contact_footprints,
)
from scgsim.geometry.primitives.constants import TOPOLOGY_EPS_UM
from scgsim.geometry.primitives.entities import entity_z_range_um
from scgsim.geometry.primitives.entity_validation import resolve_contact_pad_attachment
from scgsim.geometry.primitives.grid import boolean_polygon_region
from scgsim.geometry.primitives.spatial import same_coordinate_z


def validate_volumetric_conductor_contacts(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral | None,
    entities: Mapping[str, SemanticEntitySpec],
    representation_context: RepresentationContext | None = None,
) -> tuple[
    tuple[
        SemanticEntitySpec,
        SemanticEntitySpec,
        tuple[tuple[tuple[float, float], ...], ...],
    ],
    ...,
]:
    """Reject body overlap without explicit stacked-contact normalization."""
    import gdstk  # noqa: PLC0415 -- defer the optional polygon engine until this operation reaches native geometry.

    footprint_regions: dict[str, list[Any]] = {}
    for footprint in route_conductor_contact_footprints(
        build_input, route, representation_context=representation_context
    ):
        footprint_regions.setdefault(footprint.semantic_id, []).extend(
            entity_occupied_region(gdstk, footprint)
        )
    ordered = tuple(sorted(entities.values(), key=lambda entity: entity.semantic_id))
    normalized: list[
        tuple[
            SemanticEntitySpec,
            SemanticEntitySpec,
            tuple[tuple[tuple[float, float], ...], ...],
        ]
    ] = []
    # An authored pad must attach to a real finite overlap with its face metal;
    # its label alone cannot keep a disconnected PEC body as a normalized pad.
    for pad in ordered:
        if pad.part_role != "contact_pad":
            continue
        resolve_contact_pad_attachment(pad, ordered)
    for index, lower in enumerate(ordered):
        lower_region = tuple(footprint_regions[lower.semantic_id])
        lower_min, lower_max = entity_z_range_um(lower)
        for upper in ordered[index + 1 :]:
            upper_min, upper_max = entity_z_range_um(upper)
            if min(lower_max, upper_max) - max(lower_min, upper_min) <= TOPOLOGY_EPS_UM:
                continue
            overlap = boolean_polygon_region(
                gdstk, lower_region, tuple(footprint_regions[upper.semantic_id]), "and"
            )
            if not overlap:
                continue
            validate_volumetric_overlap_ownership(lower, upper, ordered)
            normalized.append(
                (
                    lower,
                    upper,
                    contact_patch_loops(overlap, lower.semantic_id, upper.semantic_id),
                )
            )
    return tuple(normalized)


def validate_volumetric_overlap_ownership(
    lower: SemanticEntitySpec,
    upper: SemanticEntitySpec,
    entities: Sequence[SemanticEntitySpec],
    *,
    finite_overlap: Callable[[SemanticEntitySpec, SemanticEntitySpec], bool]
    | None = None,
) -> None:
    """Require explicit stacked-contact ownership of actual body overlap."""
    if not lower.net_id or not upper.net_id:
        raise ValueError(
            f"{lower.semantic_id}/{upper.semantic_id} volumetric overlap requires resolved nets"
        )
    if lower.net_id != upper.net_id:
        raise ValueError(
            f"{lower.semantic_id}/{upper.semantic_id} volumetric overlap shorts different nets"
        )
    if not _has_explicit_volumetric_normalization(
        lower, upper, entities, finite_overlap=finite_overlap
    ):
        raise ValueError(
            f"{lower.semantic_id}/{upper.semantic_id} volumetric overlap requires explicit same-net UBM/M1/In normalization provenance"
        )


def _has_explicit_volumetric_normalization(
    lower: SemanticEntitySpec,
    upper: SemanticEntitySpec,
    entities: Sequence[SemanticEntitySpec],
    *,
    finite_overlap: Callable[[SemanticEntitySpec, SemanticEntitySpec], bool]
    | None = None,
) -> bool:
    lower_z_min, lower_z_max = entity_z_range_um(lower)
    upper_z_min, upper_z_max = entity_z_range_um(upper)
    if not (
        same_coordinate_z(lower_z_min, upper_z_min)
        and same_coordinate_z(lower_z_max, upper_z_max)
    ):
        return False
    return any(
        (
            pad.part_role == "contact_pad"
            and face.part_role == "face_metal"
            and (
                resolve_contact_pad_attachment(
                    pad, entities, finite_overlap=finite_overlap
                )
                is face
            )
            for pad, face in ((lower, upper), (upper, lower))
        )
    )
