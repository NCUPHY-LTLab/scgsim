"""The mesh features of the sidecar: what is true of a set of edges for the mesher, never a size.

Owns the five feature records (film sidewall, band boundary, arc, refinement curve, port edge) and their facts.
Each names its source in the table its kind refers to.  Part of the one definition with :mod:`.records`.
"""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from .markers import Doc, Length
from .values import (
    BandFamilyRef,
    EdgeRef,
    EntityRef,
    PortRef,
    Positive,
    RefinementRequestRef,
)


class _MeshFeatureEdges(TypedDict):
    """The edges every mesh feature covers."""

    edges: Annotated[list[EdgeRef], Length(min_items=1)]


class FilmSidewall(_MeshFeatureEdges):
    """Every edge of the sidewall faces of a finite body; thickness_um is the z extent of the sidewall column."""

    kind: Literal["film_sidewall"]
    source: Annotated[EntityRef, Doc("The body.")]
    facts: FilmSidewallFacts


class FilmSidewallFacts(TypedDict):
    """The facts of a film sidewall."""

    thickness_um: Positive


class BandBoundary(_MeshFeatureEdges):
    """The edges of an inset family; margins_um are the request's margins, ascending, and narrowest_band_um the
    smallest difference between consecutive margins with 0 prepended."""

    kind: Literal["band_boundary"]
    source: Annotated[BandFamilyRef, Doc("The band family.")]
    facts: BandBoundaryFacts


class BandBoundaryFacts(TypedDict):
    """The facts of a band boundary."""

    margins_um: Annotated[list[Positive], Length(min_items=1)]
    narrowest_band_um: Positive


class ArcFeature(_MeshFeatureEdges):
    """The circular-arc edges of one carrier radius."""

    kind: Literal["arc"]
    source: Annotated[
        EntityRef | None,
        Doc("The one Entity that owns the arcs; null when no single Entity does."),
    ]
    facts: ArcFacts


class ArcFacts(TypedDict):
    """The facts of an arc feature."""

    radius_um: Positive


class RefinementCurve(_MeshFeatureEdges):
    """The refinement-only curves of a refinement request."""

    kind: Literal["refinement_curve"]
    source: Annotated[RefinementRequestRef, Doc("The refinement request.")]
    facts: NoFacts


class PortEdge(_MeshFeatureEdges):
    """The boundary edges of the faces of a lumped-port sheet."""

    kind: Literal["port_edge"]
    source: Annotated[PortRef, Doc("The port sheet.")]
    facts: NoFacts


class NoFacts(TypedDict):
    """The facts of a feature kind that has none: an empty object."""


MeshFeature = FilmSidewall | BandBoundary | ArcFeature | RefinementCurve | PortEdge
