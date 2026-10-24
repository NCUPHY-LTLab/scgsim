"""Palace workflow contracts, offline diagnostics, and handoff utilities.

Ownership: SCGSim Development.
Failure intent: all public entrypoints are fail-closed and reject incomplete
state without silently changing behavior. Mesh-quality warnings describe the
stored mesh and are not workflow failures or acceptance thresholds.
"""

from scgsim.palace.mesh.quality import MeshQualityReport, check_mesh_quality

from .preparation.eigenmode import EigenmodeSim
from .preparation.electrostatic import ElectrostaticSim
from .presentation.reports import (
    PalaceTrustReport,
    PhysicsQuantitiesReport,
    SimulationBenchmarkReport,
    inspect_run_trustworthiness,
)
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
    "EigenmodeSim",
    "ElectrostaticSim",
    "MeshQualityReport",
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
    "check_mesh_quality",
    "epr_result",
    "inspect_run_trustworthiness",
    "reanalyze_epr",
    "resolve_epr_result",
    "resolve_palace_result",
    "show_epr",
]
