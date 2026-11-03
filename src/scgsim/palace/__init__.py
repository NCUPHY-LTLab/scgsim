"""Palace workflow contracts, offline diagnostics, and handoff utilities.

Ownership: SCGSim Development.
Failure intent: all public entrypoints are fail-closed and reject incomplete
state without silently changing behavior. Mesh-quality warnings describe the
stored mesh and are not workflow failures or acceptance thresholds.
"""

from scgsim.palace.mesh.quality import MeshQualityReport, check_mesh_quality

from .preparation.eigenmode import EigenmodeSim
from .preparation.electrostatic import ElectrostaticSim
from .presentation.amr import PalaceAmrReport, show_amr_analysis
from .presentation.reports import (
    PalaceTrustReport,
    PhysicsQuantitiesReport,
    SimulationBenchmarkReport,
    inspect_run_trustworthiness,
)
from .results.amr_analysis import analyze_amr
from .results.amr_models import CapacitanceReference, PalaceAmrAnalysis
from .results.epr import (
    PalaceEprResult,
    epr_result,
    reanalyze_epr,
    resolve_epr_result,
    show_epr,
)
from .results.report_data import (
    PalaceFailureDiagnosis,
    PalaceResultSelection,
    PassCostRecord,
    SurfaceMaskEprRecord,
    SurfaceMaskEprSeriesSnapshot,
)
from .results.resolve import (
    PalaceCost,
    PalacePerformance,
    PalaceProvenance,
    PalaceReturnedReceipt,
    ParsedTable,
    ResolvedPalaceResult,
    resolve_palace_result,
)

__all__ = [
    "CapacitanceReference",
    "EigenmodeSim",
    "ElectrostaticSim",
    "MeshQualityReport",
    "PalaceAmrAnalysis",
    "PalaceAmrReport",
    "PalaceCost",
    "PalaceEprResult",
    "PalaceFailureDiagnosis",
    "PalacePerformance",
    "PalaceProvenance",
    "PalaceResultSelection",
    "PalaceReturnedReceipt",
    "PalaceTrustReport",
    "ParsedTable",
    "PassCostRecord",
    "PhysicsQuantitiesReport",
    "ResolvedPalaceResult",
    "SimulationBenchmarkReport",
    "SurfaceMaskEprRecord",
    "SurfaceMaskEprSeriesSnapshot",
    "analyze_amr",
    "check_mesh_quality",
    "epr_result",
    "inspect_run_trustworthiness",
    "reanalyze_epr",
    "resolve_epr_result",
    "resolve_palace_result",
    "show_amr_analysis",
    "show_epr",
]
