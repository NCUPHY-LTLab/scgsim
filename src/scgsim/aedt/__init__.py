"""Public AEDT API: build a request, hand it to AEDT, and read results back.

Re-exports the user-facing pieces of the subpackages. The flow runs through them
in this order: `specs` (immutable requests and their JSON payloads),
`preparation` (turn a geometry into a request plus a hand-off folder), `runtime`
(execute the hand-off inside AEDT, run as `python -m scgsim.aedt.run`),
`results` (resolve and verify what AEDT wrote), and `epr` (energy participation
ratio analysis of Eigenmode results). Import from the owning submodule for anything
not listed in `__all__`.
"""

from scgsim.aedt.epr.models import EprResult
from scgsim.aedt.specs.epr import (
    EprAnalysisRequest,
    ExpressionCacheConvergence,
    NativeExpressionDefinition,
    NormalizedSurfaceEprTotal,
    PlanarJunction,
    SavedSolution,
    SurfaceEprSpec,
)
from scgsim.aedt.specs.geometry import (
    PreparedPlanarGeometry,
)
from scgsim.geometry.models.lumped import LumpedSupport

from .eigenmode import EigenmodeSim
from .epr.analysis import combine_epr_mode, combine_surface_epr_mode, reanalyze_epr
from .epr.geometry import planar_junction_from_port
from .epr.workflow import analyze_epr
from .preparation.geometry import (
    prepare_hfss_driven_from_geometry,
    prepare_hfss_eigenmode_from_geometry,
    prepare_q3d_from_geometry,
)
from .preparation.handoff import HandoffPlan, prepare_handoff
from .preparation.planar import prepare_planar_geometry_input
from .presentation.epr import plot_epr_result, show_epr
from .results.epr import resolve_epr_result, resolve_saved_solution
from .results.prepared import PreparedModel, resolve_prepared_model
from .results.resolve import ResolvedRun, resolve_results
from .specs.common import (
    AedtResources,
    HfssDrivenMode,
    LayerImport,
    LumpedRlc,
    LumpedTerminalPort,
    MatrixRunControl,
    ModalPort,
    ObjectBinding,
    PdkMaterial,
    TerminalPort,
)
from .specs.hfss import (
    EigenmodeRunControl,
    FrequencySweepSpec,
    HfssDrivenSpec,
    HfssEigenmodeSpec,
    HfssRunControl,
    LengthMeshSpec,
)
from .specs.hfss_prepared import (
    HfssDrivenGeometrySpec,
    HfssEprAnalysisSpec,
    HfssEprSpec,
)
from .specs.modeling import PhysicalLayerSpec
from .specs.parse import AedtSpec, HfssSpec, parse_aedt_spec
from .specs.q2d import Q2dConductorSpec, Q2dRectangleSpec, Q2dSpec
from .specs.q3d import Q3dBodySpec, Q3dNetSpec, Q3dSpec

__all__ = [
    "AedtResources",
    "AedtSpec",
    "EigenmodeRunControl",
    "EigenmodeSim",
    "EprAnalysisRequest",
    "EprResult",
    "ExpressionCacheConvergence",
    "FrequencySweepSpec",
    "HandoffPlan",
    "HfssDrivenMode",
    "HfssDrivenSpec",
    "HfssDrivenGeometrySpec",
    "LumpedSupport",
    "LumpedRlc",
    "LumpedTerminalPort",
    "HfssEigenmodeSpec",
    "HfssEprAnalysisSpec",
    "HfssEprSpec",
    "HfssRunControl",
    "HfssSpec",
    "LayerImport",
    "LengthMeshSpec",
    "MatrixRunControl",
    "ModalPort",
    "NativeExpressionDefinition",
    "NormalizedSurfaceEprTotal",
    "ObjectBinding",
    "PdkMaterial",
    "PhysicalLayerSpec",
    "PlanarJunction",
    "PreparedPlanarGeometry",
    "PreparedModel",
    "Q2dConductorSpec",
    "Q2dRectangleSpec",
    "Q2dSpec",
    "Q3dBodySpec",
    "Q3dNetSpec",
    "Q3dSpec",
    "ResolvedRun",
    "SavedSolution",
    "SurfaceEprSpec",
    "TerminalPort",
    "analyze_epr",
    "combine_epr_mode",
    "combine_surface_epr_mode",
    "parse_aedt_spec",
    "planar_junction_from_port",
    "plot_epr_result",
    "prepare_handoff",
    "prepare_hfss_eigenmode_from_geometry",
    "prepare_hfss_driven_from_geometry",
    "prepare_planar_geometry_input",
    "prepare_q3d_from_geometry",
    "reanalyze_epr",
    "resolve_epr_result",
    "resolve_results",
    "resolve_prepared_model",
    "resolve_saved_solution",
    "show_epr",
]
