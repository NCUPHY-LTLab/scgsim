"""Direct exports of canonical geometry definitions; no compatibility wrappers."""

from scgsim.geometry.models.common import (
    DEFAULT_INTERFACE_SOLVER_USE,
    HIGH_COUNT_LOCAL_CONDUCTOR_PART_ROLES,
    ConductorPartRoleLiteral,
    ConductorRepresentationLiteral,
    Coordinate,
    CurveKindLiteral,
    CurveOrientationLiteral,
    InterfaceKindLiteral,
    PathInput,
    PolygonRing,
    RouteLiteral,
    SolverUseLiteral,
    Vector3D,
)
from scgsim.geometry.models.construction import (
    ConstructionBodyPlanRecord,
    SurfacePartitionRecord,
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
from scgsim.geometry.models.lumped import LumpedSupport
from scgsim.geometry.models.physical import (
    ApproximationProfile,
    PhysicalLayerDeclaration,
    PhysicalRepresentationPlan,
)
from scgsim.geometry.models.regions import PortSheetOverlapRecord, PortSheetRegionRecord
from scgsim.geometry.models.topology import (
    InterfacePlanRecord,
    MMContactRecord,
    SurfacePlanRecord,
)

__all__ = [
    "ApproximationProfile",
    "PhysicalLayerDeclaration",
    "PhysicalRepresentationPlan",
    "DEFAULT_INTERFACE_SOLVER_USE",
    "HIGH_COUNT_LOCAL_CONDUCTOR_PART_ROLES",
    "ConductorPartRoleLiteral",
    "ConductorRepresentationLiteral",
    "ConstructionBodyPlanRecord",
    "Coordinate",
    "CurveKindLiteral",
    "CurveOrientationLiteral",
    "BoundaryCurveChainSpec",
    "GdsBoundaryReconstructionSpec",
    "SourceCurveSpec",
    "GeometryBuildInput",
    "InterfaceKindLiteral",
    "InterfacePlanRecord",
    "LayoutPolygonSpec",
    "LumpedSupport",
    "MMContactRecord",
    "PathInput",
    "PolygonRing",
    "PortSheetOverlapRecord",
    "PortSheetRegionRecord",
    "RouteLiteral",
    "SemanticEntitySpec",
    "SolverUseLiteral",
    "SurfacePartitionRecord",
    "SurfacePlanRecord",
    "VacuumRegionSpec",
    "Vector3D",
]
