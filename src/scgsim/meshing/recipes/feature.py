"""The feature recipe: which curves get which element size, from the sidecar's geometry facts alone.

The rules are measured, not derived: on 1 to 4 mm chips the only near-flat tetrahedra sit at the 0.2 um metal
sidewalls, and sizing those curves from the film thickness removes them before any repair is needed.  The
repair loop reuses the sizes this recipe asks for as its floor for those curves, so one number defines "fine
enough for this film" everywhere.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

from ..features import Feature, SidecarFeatures
from ..sizing import BoundingBox, SizeField, SizeRequests, coarse_size, ladder_size


@dataclass(frozen=True)
class FeatureRecipe:
    """Sizing parameters of the feature rules.

    The defaults were measured on 2026-11-05 (Gmsh 4.15.2).  Changing a number means a new version; the
    version and every parameter are recorded in each quality report.
    """

    name: ClassVar[str] = "features"
    version: ClassVar[str] = "scgsim.meshing.features/1"

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

    def __post_init__(self) -> None:
        """Reject sizes that cannot define a field."""
        if not (self.conductor_edge_size_um > 0 and self.conductor_edge_grow_um > 0):
            raise ValueError(
                "the conductor-outline size and growth distance must be positive."
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
        "band_boundary": _no_size,
        "arc": _no_size,
        "refinement_curve": _no_size,
        "port_edge": _no_size,
    }
)
