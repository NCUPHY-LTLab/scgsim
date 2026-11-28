"""Sizing recipes: each one decides the initial element sizes of a model, and nothing else.

A recipe turns the sidecar's geometry facts and the model's extent into :class:`SizeRequests`.  Repair, the
quality threshold and the report belong to the shared pipeline (``scgsim.meshing.repair``), so every recipe
is audited the same way; only :class:`GmshDefaultRecipe` is report-only (see ``mesh_sgb``).  Recipes never
import each other.

* :class:`FeatureRecipe` sizes the curves the sidecar's facts call for (the default).
* :class:`GlobalSizeRecipe` sets only the global maximum size.
* :class:`UniformBoundaryRecipe` sizes every curve of the model alike.
* :class:`GmshDefaultRecipe` sets nothing: Gmsh's own defaults, meshed once and never refused for quality.
"""

from __future__ import annotations

from typing import ClassVar, Protocol

from ..features import SidecarFeatures
from ..sizing import BoundingBox, SizeRequests
from .feature import FeatureRecipe
from .global_size import GlobalSizeRecipe
from .gmsh_default import GmshDefaultRecipe
from .uniform_boundary import UniformBoundaryRecipe


class SizingRecipe(Protocol):
    """What the pipeline needs of a recipe: an identity and the initial sizes.

    A recipe that is a dataclass has its fields recorded as parameters in the quality report.
    """

    # Short name of the strategy, and the version of its rules: a change in the sizes it asks for is a
    # new version.
    name: ClassVar[str]
    version: ClassVar[str]

    def size_requests(
        self, features: SidecarFeatures, bounding_box: BoundingBox
    ) -> SizeRequests:
        """The initial sizes of a model with these features and this extent."""
        ...


__all__ = [
    "FeatureRecipe",
    "GlobalSizeRecipe",
    "GmshDefaultRecipe",
    "SizingRecipe",
    "UniformBoundaryRecipe",
]
