"""The global-size recipe: one maximum element size for the whole model, no size near any curve."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

from ..features import SidecarFeatures
from ..sizing import BoundingBox, SizeRequests, coarse_size


@dataclass(frozen=True)
class GlobalSizeRecipe:
    """Only the coarse size: the bounding-box diagonal over ``max_size_divisor``.

    The cheap reference.  On 1 to 4 mm chips it leaves near-flat tetrahedra at the 0.2 um metal sidewalls
    (min SICN 0.002 to 0.0005 in the benchmark), so it is meant to run with ``Budgets(max_rounds=0)`` as a
    comparison, or with repair rounds that find the walls from the bad elements.
    """

    name: ClassVar[str] = "global-size"
    version: ClassVar[str] = "scgsim.meshing.global-size/1"

    max_size_divisor: float = 20.0
    # Size per micrometre of distance for the curves a repair round refines; this recipe asks for none.
    gradation: float = 1.0

    def size_requests(
        self, features: SidecarFeatures, bounding_box: BoundingBox
    ) -> SizeRequests:
        """The coarse size alone; ``features`` is part of the recipe interface and is not read."""
        return SizeRequests(
            coarse_size(bounding_box, self.max_size_divisor),
            self.gradation,
            MappingProxyType({}),
        )
