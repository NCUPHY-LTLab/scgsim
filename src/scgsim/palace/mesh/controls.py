"""Geometry-mesh controls: the choices Palace passes through to ``scgsim.meshing``.

Palace sizes nothing itself.  The recipe, the mesher options, the budgets and the quality threshold are
those of :func:`scgsim.meshing.mesh_sgb`; this module only keeps them together, validates the types and
writes them down for the manifest.  Geometry order is independent of the finite-element order of
``set_numerical(order=...)``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from numbers import Real
from typing import Any

from scgsim.geometry.models.refusals import Declared
from scgsim.meshing import Budgets, FeatureRecipe, MesherOptions, SizingRecipe
from scgsim.meshing.report import recipe_record
from scgsim.refusal import found_text

_WHERE_MESH_CONTROLS = "the call to set_mesh"

# Provisional (Human decision H2, issue 65): the minimum signed inverse condition number (minSICN) every tetrahedron must reach.
DEFAULT_QUALITY_THRESHOLD = 0.01


@dataclass(frozen=True)
class MeshControls:
    """The meshing choices of one Palace simulation."""

    recipe: SizingRecipe = FeatureRecipe()  # noqa: RUF009 -- a frozen dataclass is a safe default
    mesher: MesherOptions = MesherOptions()  # noqa: RUF009 -- a frozen dataclass is a safe default
    budgets: Budgets = Budgets()  # noqa: RUF009 -- a frozen dataclass is a safe default
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD

    def __post_init__(self) -> None:
        """Reject a value of the wrong type or a threshold that is not a finite number."""
        # SizingRecipe is a protocol without runtime checks: a recipe is anything with the two identity
        # fields and the sizing method.
        if not (
            callable(getattr(self.recipe, "size_requests", None))
            and hasattr(self.recipe, "name")
            and hasattr(self.recipe, "version")
        ):
            raise TypeError(
                Declared(
                    "recipe must be a scgsim.meshing sizing recipe",
                    _WHERE_MESH_CONTROLS,
                ).refusal(found_text(type(self.recipe).__name__))
            )
        if not isinstance(self.mesher, MesherOptions):
            raise TypeError(
                Declared(
                    "mesher must be scgsim.meshing.MesherOptions", _WHERE_MESH_CONTROLS
                ).refusal(found_text(type(self.mesher).__name__))
            )
        if not isinstance(self.budgets, Budgets):
            raise TypeError(
                Declared(
                    "budgets must be scgsim.meshing.Budgets", _WHERE_MESH_CONTROLS
                ).refusal(found_text(type(self.budgets).__name__))
            )
        threshold = self.quality_threshold
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, Real)
            or not math.isfinite(threshold)
        ):
            raise ValueError(
                Declared(
                    "quality_threshold must be a finite number", _WHERE_MESH_CONTROLS
                ).refusal(found_text(threshold))
            )

    def updated(
        self,
        *,
        recipe: SizingRecipe | None = None,
        mesher: MesherOptions | None = None,
        budgets: Budgets | None = None,
        quality_threshold: float | None = None,
    ) -> MeshControls:
        """These controls with every supplied value replaced; omitted values are retained."""
        changes = {
            name: value
            for name, value in {
                "recipe": recipe,
                "mesher": mesher,
                "budgets": budgets,
                "quality_threshold": quality_threshold,
            }.items()
            if value is not None
        }
        return replace(self, **changes)

    def to_payload(self) -> dict[str, Any]:
        """The JSON form recorded in the mesh manifest (recipe name, version and parameters, not its class)."""
        return {
            "recipe": recipe_record(self.recipe),
            "mesher": asdict(self.mesher),
            "budgets": asdict(self.budgets),
            "quality_threshold": float(self.quality_threshold),
        }
