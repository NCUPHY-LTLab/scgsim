"""What each source polygon was before the grid: its Entity, its requested rings, and the audits the input records.

The grid projection (``grid_projection``) compares what the grid realized against these preimages.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypedDict, cast

from scgsim.geometry.models.input import (
    GeometryBuildInput,
    LayoutPolygonSpec,
    SemanticEntitySpec,
)
from scgsim.geometry.source.recipes import SourceGridAudit, SourcePolygonAudit


class _SourcePolygonIdentity(TypedDict):
    source_entity_id: str
    source_polygon_id: str


_SourceRing = Sequence[Sequence[float]]


@dataclass(frozen=True)
class PolygonPreimage:
    """Requested source rings retained with their original container types."""

    exterior: _SourceRing
    holes: Sequence[_SourceRing]


@dataclass(frozen=True)
class SourcePolygonInputs:
    """Entity ownership and physical source preimages for layout polygons.

    ``recorded_audits`` are the source audits the projected input records: the ones
    handed to this projection, else the ones the input already carries.
    """

    owners: Mapping[str, SemanticEntitySpec]
    preimages: Mapping[str, PolygonPreimage]
    recorded_audits: Sequence[SourceGridAudit]


def source_polygon_inputs(
    build_input: GeometryBuildInput,
    fresh_audits: tuple[SourceGridAudit, ...],
    *,
    use_source_preimage: bool,
) -> SourcePolygonInputs:
    """Each polygon's owning Entity and preimage, from the audits handed over and those the input carries.

    ``use_source_preimage`` says whether audited requests stand for the polygons they
    made; an input projected again on its own grid with no new audits has none.
    """
    stored_audits = cast(
        Sequence[SourceGridAudit],
        build_input.metadata.get("geometry_grid_source_audits", ()),
    )
    earliest_preimages = _earliest_preimages((*fresh_audits, *stored_audits))
    owners = {
        polygon_id: entity
        for entity in build_input.entities
        for polygon_id in entity.polygon_ids
    }
    preimages = {
        polygon.polygon_id: _polygon_preimage(
            polygon,
            owners.get(polygon.polygon_id),
            earliest_preimages if use_source_preimage else None,
        )
        for polygon in build_input.polygons
    }
    return SourcePolygonInputs(
        owners=owners,
        preimages=preimages,
        recorded_audits=fresh_audits or stored_audits,
    )


def _earliest_preimages(
    audits: Sequence[SourceGridAudit],
) -> dict[tuple[str, str], SourcePolygonAudit]:
    """Each audited source polygon as its first audit requested it, by Entity and polygon id."""
    earliest_preimages: dict[tuple[str, str], SourcePolygonAudit] = {}
    for audit in audits:
        for source_polygon in audit.get("requested_polygons", ()):
            earliest_preimages.setdefault(
                (audit["source_entity_id"], source_polygon["source_polygon_id"]),
                source_polygon,
            )
    return earliest_preimages


def _polygon_preimage(
    polygon: LayoutPolygonSpec,
    owner: SemanticEntitySpec | None,
    earliest_preimages: Mapping[tuple[str, str], SourcePolygonAudit] | None,
) -> PolygonPreimage:
    """What a polygon was before the grid: its audited request, or the polygon itself.

    Without ``earliest_preimages`` (an input projected again on its own grid) every
    polygon is its own preimage; with them, a polygon a declared recipe made must have one.
    """
    # Geometry metadata is the serialized boundary for primitive source identity.
    identity = cast(
        _SourcePolygonIdentity | None,
        polygon.metadata.get("geometry_grid_source_identity"),
    )
    if identity is None and "source_polygon_id" in polygon.metadata:
        identity = {
            "source_entity_id": cast(str, polygon.metadata["source_semantic_id"]),
            "source_polygon_id": cast(str, polygon.metadata["source_polygon_id"]),
        }
    if identity is None and owner is not None:
        identity = {
            "source_entity_id": owner.semantic_id,
            "source_polygon_id": polygon.polygon_id,
        }
    saved_preimage = (
        None
        if earliest_preimages is None or identity is None
        else earliest_preimages.get(
            (identity["source_entity_id"], identity["source_polygon_id"])
        )
    )
    if (
        earliest_preimages is not None
        and "geometry_grid_source_identity" in polygon.metadata
        and saved_preimage is None
    ):
        raise ValueError(
            f"declared physical source preimage is unavailable: {polygon.polygon_id}"
        )
    if saved_preimage:
        # JSON audit payloads may contain lists; preserve those public metadata values.
        return PolygonPreimage(
            exterior=cast(_SourceRing, saved_preimage["exterior"]),
            holes=cast(Sequence[_SourceRing], saved_preimage["holes"]),
        )
    return PolygonPreimage(exterior=polygon.exterior, holes=polygon.holes)
