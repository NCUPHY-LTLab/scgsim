"""The vocabulary every sizing recipe and the repair loop share: the model's extent and the sizes asked for.

Owns :class:`BoundingBox`, :class:`SizeRequests` (what a recipe hands to the mesher) and the size ladder.  A
recipe (``recipes``) decides the initial sizes; the repair loop (``repair``) refines them on the same
ladder, so one rule defines "a step finer" for every recipe.

Sizes are micrometres, as in the sidecar.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass


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


def coarse_size(bounding_box: BoundingBox, divisor: float) -> float:
    """The global maximum element size of a model: its bounding-box diagonal over ``divisor``."""
    return bounding_box.diagonal / divisor


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
