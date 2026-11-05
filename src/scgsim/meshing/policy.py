"""The sizing rules: which curves get which element size, from the sidecar's geometry facts alone.

Owns the versioned :class:`RuleSet` and the a priori step that turns features into per-curve sizes.  The
rules are measured, not derived: on 1 to 4 mm chips the only near-flat tetrahedra sit at the 0.2 um metal
sidewalls, and sizing those curves from the film thickness removes them before any repair is needed.  The
repair loop (``repair``) reuses the same ladder and the same film rule, so one number defines "fine enough
for this film" everywhere.

Sizes are micrometres, as in the sidecar.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .features import Feature, SidecarFeatures

RULES_VERSION = "scgsim.meshing.rules/1"


@dataclass(frozen=True)
class RuleSet:
    """A named, versioned set of sizing parameters.

    The defaults were measured on 2026-11-05 (Gmsh 4.15.2, HXT).  Changing a number means a new version;
    the version and every parameter are recorded in each quality report.
    """

    version: str = RULES_VERSION
    # Coarse size away from every feature: the bounding-box diagonal over this number.
    max_size_divisor: float = 20.0
    # Size grows by 1 um per um of distance from a refined curve.  Gradation of 1.5 or more created new
    # near-flat tetrahedra at the edge of the refined region.
    gradation: float = 1.0
    # A film wall of thickness t gets elements of at most this multiple of t ...
    film_sidewall_thickness_multiple: float = 25.0
    # ... and never coarser than this fraction of the coarse size.
    film_sidewall_ceiling_fraction: float = 0.5
    # Repair: a bad element's longest edge, times this, is the first size tried for the curves around it.
    repair_edge_fraction: float = 0.5
    # Repair: only curves whose local edge is at least this fraction of the longest local edge of the
    # element's candidates are refined; the others cannot shorten the element's long edge.
    repair_culprit_fraction: float = 0.5
    # Repair: shorter curves (the 0.2 um vertical edges of a film wall) are ignored, since refining a
    # curve cannot shorten an element edge below its length.
    repair_min_curve_length_um: float = 1.0


@dataclass(frozen=True)
class BoundingBox:
    """The axis-aligned bounding box of the model, micrometres."""

    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]

    @property
    def diagonal(self) -> float:
        """Length of the box diagonal."""
        return math.dist(self.minimum, self.maximum)


@dataclass(frozen=True)
class SizeRequests:
    """What the mesher is asked for: a coarse size, a gradation and the finer size of some curves."""

    max_size: float
    gradation: float
    curve_sizes: Mapping[int, float]  # native curve tag -> element size


SizeRule = Callable[[Feature, float, RuleSet], float | None]


def size_requests(
    features: SidecarFeatures, bounding_box: BoundingBox, ruleset: RuleSet
) -> SizeRequests:
    """The a priori per-curve sizes of ``features`` in a model of the given extent.

    A curve covered by several features takes the smallest size any of them asks for.
    """
    max_size = coarse_size(bounding_box, ruleset)
    curve_sizes: dict[int, float] = {}
    for tag, covering in features.features_by_curve().items():
        size = feature_size(covering, max_size, ruleset)
        if size is not None:
            curve_sizes[tag] = size
    return SizeRequests(max_size, ruleset.gradation, MappingProxyType(curve_sizes))


def coarse_size(bounding_box: BoundingBox, ruleset: RuleSet) -> float:
    """The global maximum element size of a model."""
    return bounding_box.diagonal / ruleset.max_size_divisor


def feature_size(
    covering: tuple[Feature, ...], max_size: float, ruleset: RuleSet
) -> float | None:
    """The smallest size the rules ask for a curve covered by ``covering``; ``None`` when no rule asks."""
    sizes = [
        size
        for feature in covering
        if (size := _KIND_RULES[feature.kind](feature, max_size, ruleset)) is not None
    ]
    return min(sizes) if sizes else None


def ladder_size(target: float, max_size: float) -> float:
    """The largest ``max_size / 2**k`` (k >= 1) that is at most ``target``.

    Sizes on one ladder keep the number of distinct size fields small: Gmsh evaluates every field at every
    point, and rounding down never makes a mesh coarser than the target.
    """
    if not target > 0:
        raise ValueError(f"a ladder size needs a positive target, got {target!r}.")
    exponent = 1
    while max_size / 2**exponent > target:
        exponent += 1
    return max_size / 2**exponent


def _film_sidewall_size(
    feature: Feature, max_size: float, ruleset: RuleSet
) -> float | None:
    """Elements at a film wall: a multiple of its thickness, no coarser than the ceiling, on the ladder."""
    wanted = min(
        ruleset.film_sidewall_thickness_multiple * feature.fact("thickness_um"),
        ruleset.film_sidewall_ceiling_fraction * max_size,
    )
    return ladder_size(wanted, max_size)


def _no_size(_feature: Feature, _max_size: float, _ruleset: RuleSet) -> float | None:
    """No a priori size: HXT with only the coarse size meshed these curves cleanly in rules/1."""
    return None


# One hook per feature kind, so a later rule set changes the answer for one kind in one place.
_KIND_RULES: Mapping[str, SizeRule] = MappingProxyType(
    {
        "film_sidewall": _film_sidewall_size,
        "band_boundary": _no_size,
        "arc": _no_size,
        "refinement_curve": _no_size,
        "port_edge": _no_size,
    }
)
