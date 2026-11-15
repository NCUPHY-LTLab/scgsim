"""The vocabulary every sizing recipe and the repair loop share: the model's extent and the sizes asked for.

Owns :class:`BoundingBox`, :class:`SizeRequests` (what a recipe hands to the mesher, and the local sizes the
repair loop adds) and the size ladder.  A recipe (``recipes``) decides the initial sizes; the repair loop
(``repair``) refines around bad elements on the same ladder, so one rule defines "a step finer" for every
recipe.

Sizes are micrometres, as in the sidecar.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from scgsim.refusal import internal

_UM2_PER_MM2 = 1e6


@dataclass(frozen=True)
class BoundingBox:
    """The axis-aligned bounding box of the model, micrometres."""

    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]

    @property
    def diagonal(self) -> float:
        """Length of the box diagonal."""
        return math.dist(self.minimum, self.maximum)

    @property
    def footprint_mm2(self) -> float:
        """Area of the box in the layout plane (x by y), square millimetres."""
        width = self.maximum[0] - self.minimum[0]
        depth = self.maximum[1] - self.minimum[1]
        return width * depth / _UM2_PER_MM2


@dataclass(frozen=True)
class SizeField:
    """``size`` on ``curves``, growing linearly to the coarse size over ``distance`` micrometres."""

    curves: tuple[int, ...]  # native curve tags
    size: float
    distance: float


@dataclass(frozen=True)
class LocalSize:
    """``size`` within ``core_radius`` of ``centre``, growing by the gradation to the coarse size beyond.

    The repair loop places one at each bad element, so the refinement stays there; see
    :func:`scgsim.meshing.repair.refine_locally` for where it goes and how wide the core is.
    """

    centre: tuple[float, float, float]
    size: float
    # Radius of the fine core; None for a core as wide as ``size``.
    radius: float | None = None

    @property
    def core_radius(self) -> float:
        """Distance from ``centre`` within which the size is ``size``."""
        return self.size if self.radius is None else self.radius


@dataclass(frozen=True)
class SizeRequests:
    """What the mesher is asked for: a coarse size and where it must be finer.

    ``curve_sizes`` are graded by ``gradation`` (size per micrometre of distance).  ``fields`` grow over a
    distance of their own.  Both come from the recipe and stay as they are while the repair loop adds
    ``local_sizes`` around bad elements, graded by the same ``gradation``.  The finest size at a point wins.
    """

    max_size: float
    gradation: float
    curve_sizes: Mapping[int, float]  # native curve tag -> element size
    fields: tuple[SizeField, ...] = ()
    local_sizes: tuple[LocalSize, ...] = ()


def coarse_size(bounding_box: BoundingBox, divisor: float) -> float:
    """The global maximum element size of a model: its bounding-box diagonal over ``divisor``."""
    return bounding_box.diagonal / divisor


def ladder_size(target: float, max_size: float) -> float:
    """The largest ``max_size / 2**k`` (k >= 1) that is at most ``target``.

    Sizes on one ladder keep the number of distinct size fields small: Gmsh evaluates every field at every
    point, and rounding down never makes a mesh coarser than the target.
    """
    if not target > 0:
        raise ValueError(
            internal(f"a ladder size needs a positive target, got {target!r}")
        )
    exponent = 1
    while max_size / 2**exponent > target:
        exponent += 1
    return max_size / 2**exponent
