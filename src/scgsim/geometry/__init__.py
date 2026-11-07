"""Source-facing semantic geometry API; input ownership remains with caller/components.

The layout check (``simulation_drc`` and its report types) loads on first use. It reaches the whole SGB
package, which the AEDT runtime source bundle, built from this module's imports, does not ship.
"""

import importlib
from typing import Any

from scgsim.geometry.models.common import (
    Coordinate,
    CurveKindLiteral,
    CurveOrientationLiteral,
    PathInput,
    PolygonRing,
    RouteLiteral,
    Vector3D,
)
from scgsim.geometry.models.input import (
    BoundaryCurveChainSpec,
    GdsBoundaryReconstructionSpec,
    GeometryBuildInput,
    LayoutPolygonSpec,
    SemanticEntitySpec,
    SourceCurveSpec,
    VacuumRegionSpec,
)
from scgsim.geometry.models.junctions import (
    JunctionTerminalBinding,
    RectangularJunctionSpec,
)
from scgsim.geometry.models.lumped import LumpedSupport
from scgsim.geometry.models.physical import (
    ApproximationProfile,
    PhysicalLayerDeclaration,
    PhysicalRepresentationPlan,
    interpret_physical_layers,
)
from scgsim.geometry.models.regions import PortSheetOverlapRecord, PortSheetRegionRecord
from scgsim.geometry.planning.physical import interpret_physical_geometry
from scgsim.geometry.source.adapter import (
    build_gds_stack_geometry_input,
    build_gdsfactory_geometry_input,
)
from scgsim.geometry.source.curves import bind_source_curves
from scgsim.geometry.source.junctions import with_rectangular_junctions
from scgsim.geometry.source.plan import GeometryPlan, GeometryPlanSnapshot
from scgsim.geometry.source.stack import build_component_stack
from scgsim.geometry.source.summary import InputSummary, summarize_geometry_input
from scgsim.geometry.source.vacuum import apply_vacuum_region_to_stack

__all__ = [
    "JunctionTerminalBinding",
    "RectangularJunctionSpec",
    "with_rectangular_junctions",
    "interpret_physical_geometry",
    "interpret_physical_layers",
    "ApproximationProfile",
    "PhysicalLayerDeclaration",
    "PhysicalRepresentationPlan",
    "BoundaryCurveChainSpec",
    "Coordinate",
    "CurveKindLiteral",
    "CurveOrientationLiteral",
    "GdsBoundaryReconstructionSpec",
    "GeometryBuildInput",
    "GeometryPlan",
    "GeometryPlanSnapshot",
    "InputSummary",
    "LayoutCheckError",
    "LayoutCheckReport",
    "LayoutFinding",
    "LayoutPolygonSpec",
    "LumpedSupport",
    "PathInput",
    "PolygonRing",
    "PortSheetOverlapRecord",
    "PortSheetRegionRecord",
    "RouteLiteral",
    "SemanticEntitySpec",
    "SkippedRule",
    "SourceCurveSpec",
    "VacuumRegionSpec",
    "Vector3D",
    "apply_vacuum_region_to_stack",
    "bind_source_curves",
    "build_component_stack",
    "build_gds_stack_geometry_input",
    "build_gdsfactory_geometry_input",
    "simulation_drc",
    "summarize_geometry_input",
]


_LAYOUT_CHECK_NAMES = frozenset(
    {
        "LayoutCheckError",
        "LayoutCheckReport",
        "LayoutFinding",
        "SkippedRule",
        "simulation_drc",
    }
)


def __getattr__(name: str) -> Any:
    """Load the layout check on first use (see the module docstring)."""
    if name in _LAYOUT_CHECK_NAMES:
        return getattr(importlib.import_module("scgsim.geometry.drc"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
