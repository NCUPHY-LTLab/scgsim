"""The feature recipe: which curves get which element size, from the sidecar's geometry facts alone.

The rules are measured, not derived.  On 1 to 4 mm chips the only near-flat tetrahedra sit at the 0.2 um metal
sidewalls, and sizing those curves from the film thickness removes them before any repair is needed.  With
Surface-EPR bands, the band edges are the other place: bands 0.5 um wide next to elements tens of micrometres
long leave slivers that local repair does not clear, so every band edge is sized from the width of its bands
(``experiments/meshing-benchmark/band_edge.md``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

from scgsim.refusal import found_text, refused

from ..features import Feature, SidecarFeatures
from ..sizing import BoundingBox, SizeField, SizeRequests, coarse_size, ladder_size

_WHERE_RECIPE = "the sizing recipe of the meshing call"

_WHERE_FEATURE_RECIPE = "the FeatureRecipe passed to set_mesh"


@dataclass(frozen=True)
class FeatureRecipe:
    """Sizing parameters of the feature rules.

    The film and outline defaults were measured on 2026-11-05 and the band-edge default on 2026-11-10
    (Gmsh 4.15.2).  Changing a number means a new version; the version and every parameter are recorded in
    each quality report.
    """

    name: ClassVar[str] = "features"
    version: ClassVar[str] = "scgsim.meshing.features/2"

    # Coarse size away from every feature: the bounding-box diagonal over this number.
    max_size_divisor: float = 20.0
    # Size grows by 1 um per um of distance from a refined curve.  Gradation of 1.5 or more created new
    # near-flat tetrahedra at the edge of the refined region.
    gradation: float = 1.0
    # A film wall of thickness t gets elements of at most this multiple of t, but only when that is finer
    # than this fraction of the coarse size; a thicker wall is already resolved and gets no size.
    film_sidewall_thickness_multiple: float = 25.0
    film_sidewall_ceiling_fraction: float = 0.5
    # Conductor-outline rule, off by default: a Palace AMR comparison found that resolving the edges of
    # the conductors in the initial mesh needs fewer degrees of freedom for a 0.1 % capacitance, and the
    # validation decides the default.  When on, the outline edges of the conductors (their film walls,
    # arcs and the port edges; never the outer domain, the substrate walls or band edges) get this size ...
    conductor_outline: bool = False
    conductor_edge_size_um: float = 5.0
    # ... growing linearly to the coarse size over this distance.
    conductor_edge_grow_um: float = 300.0
    # The largest mesh element on each Surface-EPR band boundary curve is this multiple of the narrowest
    # band width of its face, rounded down the ladder.  Of the measured multiples (band_edge.md), 4 alone
    # meshed the banded flip-chip Disk Qubit (PDK 49e539f) clean in its first mesh, within a minute; 2
    # asked for 2 to 2.4 times the elements and reached the time limit there, 8 needed a repair round
    # there.  All three meshed SimplePad on Route A with bands clean.
    band_mesh_size_multiple: float = 4.0

    def __post_init__(self) -> None:
        """Reject sizes that cannot define a field."""
        if not (self.conductor_edge_size_um > 0 and self.conductor_edge_grow_um > 0):
            raise ValueError(
                refused(
                    "the conductor-outline size and growth distance must be positive",
                    found_text(
                        {
                            "conductor_edge_size_um": self.conductor_edge_size_um,
                            "conductor_edge_grow_um": self.conductor_edge_grow_um,
                        }
                    ),
                    _WHERE_FEATURE_RECIPE,
                )
            )
        multiple = self.band_mesh_size_multiple
        if (
            isinstance(multiple, bool)
            or not isinstance(multiple, int | float)
            or not math.isfinite(multiple)
            or multiple <= 0
        ):
            raise ValueError(
                refused(
                    "band_mesh_size_multiple",
                    f"must be a finite positive number, got {multiple!r}",
                    _WHERE_RECIPE,
                )
            )

    def size_requests(
        self, features: SidecarFeatures, bounding_box: BoundingBox
    ) -> SizeRequests:
        """The a priori per-curve sizes of ``features`` in a model of the given extent.

        A curve covered by several features takes the smallest size any of them asks for.
        """
        max_size = coarse_size(bounding_box, self.max_size_divisor)
        curve_sizes: dict[int, float] = {}
        for tag, covering in features.features_by_curve().items():
            size = self._covering_size(covering, max_size)
            if size is not None:
                curve_sizes[tag] = size
        return SizeRequests(
            max_size,
            self.gradation,
            MappingProxyType(curve_sizes),
            self._outline_fields(features, max_size),
        )

    def _covering_size(
        self, covering: tuple[Feature, ...], max_size: float
    ) -> float | None:
        """The smallest size the rules ask for a curve covered by ``covering``; ``None`` when no rule asks."""
        sizes = [
            size
            for feature in covering
            if (size := _KIND_RULES[feature.kind](feature, max_size, self)) is not None
        ]
        return min(sizes) if sizes else None

    def _outline_fields(
        self, features: SidecarFeatures, max_size: float
    ) -> tuple[SizeField, ...]:
        """The conductor-outline field, when the rule is on and finer than the coarse size."""
        if not self.conductor_outline or self.conductor_edge_size_um >= max_size:
            return ()
        curves = conductor_outline_curves(features)
        if not curves:
            return ()
        return (
            SizeField(curves, self.conductor_edge_size_um, self.conductor_edge_grow_um),
        )


def conductor_outline_curves(features: SidecarFeatures) -> tuple[int, ...]:
    """Native tags of the conductors' outline edges: film walls, arcs and port edges, in ascending order.

    A feature counts when a conductor owns it (``source``).  Port edges are the exception: their source is
    the port sheet, not a conductor, but they are the outline where the port lands on the conductors.
    Band edges, the outer domain and the walls of a substrate are never included.
    """
    tags: set[int] = set()
    for feature in features.features:
        if feature.kind == "port_edge" or (
            feature.kind in ("film_sidewall", "arc")
            and feature.source in features.conductors
        ):
            tags.update(feature.native_tags)
    return tuple(sorted(tags))


def _film_sidewall_size(
    feature: Feature, max_size: float, recipe: FeatureRecipe
) -> float | None:
    """Elements at a thin film wall: a multiple of its thickness, on the ladder; nothing for a thick one.

    The rule binds only when the thickness multiple is finer than the ceiling (a fraction of the coarse
    size).  A wall at or above the ceiling, such as a 10 um substrate in a 1 mm model, is already resolved
    relative to the coarse size, so sizing it would only add elements; the repair loop still catches
    anything such a wall leaves.
    """
    wanted = recipe.film_sidewall_thickness_multiple * feature.fact("thickness_um")
    if wanted >= recipe.film_sidewall_ceiling_fraction * max_size:
        return None
    return ladder_size(wanted, max_size)


def _band_boundary_size(
    feature: Feature, max_size: float, recipe: FeatureRecipe
) -> float:
    """The largest mesh element on a band boundary curve: the multiple times the narrowest band width.

    The narrowest band width is that of the curve's face, and the size is rounded down the ladder.  Unlike
    a film wall, a band boundary curve is always sized.  Its bands are as narrow as the margins between
    them (0.5 um for margins 0.5 and 1 um) and sit at the conductor edges, where Route A has no film wall
    whose size would reach them; with no size of their own, SimplePad on Route A met elements up to 61 um
    long there and stalled.  The sidecar states the narrowest band per banded face, not per curve, so
    every boundary curve of a face takes the same size.
    """
    wanted = recipe.band_mesh_size_multiple * feature.fact("narrowest_band_um")
    return ladder_size(wanted, max_size)


def _no_size(
    _feature: Feature, _max_size: float, _recipe: FeatureRecipe
) -> float | None:
    """No a priori size: a mesh with only the coarse size was clean at these curves in features/1."""
    return None


_SizeRule = Callable[[Feature, float, FeatureRecipe], float | None]

# One hook per feature kind, so a later version changes the answer for one kind in one place.
_KIND_RULES: Mapping[str, _SizeRule] = MappingProxyType(
    {
        "film_sidewall": _film_sidewall_size,
        "band_boundary": _band_boundary_size,
        "arc": _no_size,
        "port_edge": _no_size,
    }
)
