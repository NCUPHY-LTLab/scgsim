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
    # A film wall of thickness t gets elements of at most this multiple of t, but only when that is finer
    # than this fraction of the coarse size; a thicker wall is already resolved and gets no size.
    film_sidewall_thickness_multiple: float = 25.0
    film_sidewall_ceiling_fraction: float = 0.5
    # Repair: a bad element's longest edge, times this, is the first size tried for the curves around it.
    repair_edge_fraction: float = 0.5
    # Repair: only curves whose local edge is at least this fraction of the longest local edge of the
    # element's candidates are refined; the others cannot shorten the element's long edge.
    repair_culprit_fraction: float = 0.5
    # Repair: shorter curves (the 0.2 um vertical edges of a film wall) are ignored, since refining a
    # curve cannot shorten an element edge below its length.
    repair_min_curve_length_um: float = 1.0
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
class SizeField:
    """``size`` on ``curves``, growing linearly to the coarse size over ``distance`` micrometres."""

    curves: tuple[int, ...]  # native curve tags
    size: float
    distance: float


@dataclass(frozen=True)
class SizeRequests:
    """What the mesher is asked for: a coarse size and where it must be finer.

    ``curve_sizes`` are graded by ``gradation`` (size per micrometre of distance).  ``fields`` grow over a
    distance of their own and stay as they are while the repair loop refines ``curve_sizes``; the finest
    size at a point wins.
    """

    max_size: float
    gradation: float
    curve_sizes: Mapping[int, float]  # native curve tag -> element size
    fields: tuple[SizeField, ...] = ()


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
    return SizeRequests(
        max_size,
        ruleset.gradation,
        MappingProxyType(curve_sizes),
        _outline_fields(features, max_size, ruleset),
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


def _outline_fields(
    features: SidecarFeatures, max_size: float, ruleset: RuleSet
) -> tuple[SizeField, ...]:
    """The conductor-outline field, when the rule is on and finer than the coarse size."""
    if not ruleset.conductor_outline or ruleset.conductor_edge_size_um >= max_size:
        return ()
    curves = conductor_outline_curves(features)
    if not curves:
        return ()
    return (
        SizeField(
            curves, ruleset.conductor_edge_size_um, ruleset.conductor_edge_grow_um
        ),
    )


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
    """Elements at a thin film wall: a multiple of its thickness, on the ladder; nothing for a thick one.

    The rule binds only when the thickness multiple is finer than the ceiling (a fraction of the coarse
    size).  A wall at or above the ceiling, such as a 10 um substrate in a 1 mm model, is already resolved
    relative to the coarse size, so sizing it would only add elements; the repair loop still catches
    anything such a wall leaves.
    """
    wanted = ruleset.film_sidewall_thickness_multiple * feature.fact("thickness_um")
    if wanted >= ruleset.film_sidewall_ceiling_fraction * max_size:
        return None
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
