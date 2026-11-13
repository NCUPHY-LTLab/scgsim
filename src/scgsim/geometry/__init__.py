"""Source-facing semantic geometry API; input ownership remains with caller/components.

One folder per stage, in the order the work flows for a layout; ``models`` and ``grid`` are what the stages share:

- ``models``: the typed records that cross stages.
- ``source``: from a layout and a stack to a prepared snapshot.
- ``grid``: the grid engine the stages call: noding, tagged regions, overlays, loops and polygon operations.
- ``interpretation``: from a prepared input to its physical representation and vacuum enclosure.
- ``junctions``: the rectangular partition around a lumped port.
- ``surface_epr``: from Surface-EPR specs to the inset requests of the SGB build.
- ``sgb``: the geometry builder.
- ``planning``: what only AEDT's native construction still uses.
"""

from scgsim.geometry.interpretation.physical import interpret_physical_geometry
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
from scgsim.geometry.source.adapter import build_gds_stack_geometry_input
from scgsim.geometry.source.junctions import with_rectangular_junctions
from scgsim.geometry.source.plan import GeometryPlan, GeometryPlanSnapshot
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
    "LayoutPolygonSpec",
    "LumpedSupport",
    "PathInput",
    "PolygonRing",
    "PortSheetOverlapRecord",
    "PortSheetRegionRecord",
    "RouteLiteral",
    "SemanticEntitySpec",
    "SourceCurveSpec",
    "VacuumRegionSpec",
    "Vector3D",
    "apply_vacuum_region_to_stack",
    "build_gds_stack_geometry_input",
    "summarize_geometry_input",
]
