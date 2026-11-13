"""Run a Palace simulation from a prepared plan to a run folder, and read the run Palace returns.

One folder per stage, in the order the work flows:

- ``simulation``: the stage a user drives, the two simulation classes and their lifecycle.
- ``source``: from a bound plan to the input of the geometry build.
- ``mesh``: from that input to ``palace.msh`` with its physical groups and manifest.
- ``config``: from the mesh's groups and the problem's settings to ``config.json`` and the index map.
- ``run``: the run folder handed to Palace, and what Palace returns.
- ``results``: from a finished run folder to a resolved result.
- ``reports``: from a resolved result to notebook views.

Every public entry point refuses incomplete state with an error; none fills in a missing setting silently.
"""

from .reports.reports import (
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
