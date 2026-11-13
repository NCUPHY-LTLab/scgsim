"""The uniform-boundary recipe: every curve of the model gets the same fine size."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

from scgsim.refusal import refused

from ..features import SidecarFeatures
from ..sizing import BoundingBox, SizeRequests

_WHERE_RECIPE = "the sizing recipe of the meshing call"

_CURVE_DIMENSION = 1


@dataclass(frozen=True)
class UniformBoundaryRecipe:
    """``near_um`` on every curve of the model, growing to ``far_um`` over ``far_um`` micrometres.

    The uniform policy a Palace run applies by default, here on the module's own Distance/Threshold field,
    so its element counts differ from a Palace mesh.  It ignores what the curves are: a 4 mm outline and a
    0.2 um sidewall edge get the same size.
    """

    name: ClassVar[str] = "uniform-boundary"
    version: ClassVar[str] = "scgsim.meshing.uniform-boundary/1"

    near_um: float = 5.0
    far_um: float = 300.0

    def __post_init__(self) -> None:
        """Reject sizes whose growth cannot be defined."""
        if not 0 < self.near_um < self.far_um:
            raise ValueError(
                refused(
                    "the sizes",
                    f"must satisfy 0 < near_um < far_um, got {self.near_um!r} and {self.far_um!r}",
                    _WHERE_RECIPE,
                )
            )

    def size_requests(
        self, features: SidecarFeatures, bounding_box: BoundingBox
    ) -> SizeRequests:
        """Every curve of the sidecar's binding at ``near_um``; the extent is not read.

        The gradation makes the size reach ``far_um`` over ``far_um`` micrometres:
        ``DistMax = (max - size) / gradation = far_um``.
        """
        curve_count = len(features.binding.ids_by_dimension[_CURVE_DIMENSION])
        gradation = (self.far_um - self.near_um) / self.far_um
        curves = {tag: self.near_um for tag in range(1, curve_count + 1)}
        return SizeRequests(self.far_um, gradation, MappingProxyType(curves))
