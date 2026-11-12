"""Palace workflow contracts, result views, and handoff utilities.

Ownership: SCGSim Development.
Failure intent: all public entrypoints are fail-closed and reject incomplete
state without silently changing behavior.
"""

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
from .results.report_models import (
    PalaceFailureDiagnosis,
    PalaceResultSelection,
    PassCostRecord,
    SurfaceMarginEprRecord,
    SurfaceMarginEprSeriesSnapshot,
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
from .simulation.drc import SimulationDrcError, SimulationDrcReport, StageResult
from .simulation.eigenmode import EigenmodeSim
from .simulation.electrostatic import ElectrostaticSim

__all__ = [
    "EigenmodeSim",
    "ElectrostaticSim",
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
    "SimulationDrcError",
    "SimulationDrcReport",
    "StageResult",
    "SurfaceMarginEprRecord",
    "SurfaceMarginEprSeriesSnapshot",
    "epr_result",
    "inspect_run_trustworthiness",
    "reanalyze_epr",
    "resolve_epr_result",
    "resolve_palace_result",
    "show_epr",
]
