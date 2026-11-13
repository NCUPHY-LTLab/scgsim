"""Run a Palace simulation from a prepared plan to a run folder, and read the run Palace returns.

One folder per stage, in the order the work flows:

- ``simulation``: the stage a user drives, the two simulation classes and their lifecycle.
- ``source``: from a bound plan to the input of the geometry build.
- ``mesh``: from that input to ``palace.msh`` with its physical groups and manifest.
- ``config``: from the mesh's groups and the problem's settings to ``config.json`` and the index map.
- ``run``: the run folder handed to Palace, and what Palace returns.
- ``results``: from a finished run folder to a resolved result.
- ``reports``: from a resolved result to notebook views.

A public entry point refuses state it cannot run from (no bound plan, a stage called out of order, a name that
binds nothing) with an error that says which call fixes it.  A setting with a documented default, such as Route B
and the numerical controls, keeps that default until the caller sets it.
"""

from .reports.epr import show_epr
from .reports.reports import inspect_run_trustworthiness
from .reports.resolved_result import ResolvedPalaceResult, resolve_palace_result
from .reports.trust import PalaceTrustReport
from .reports.views import PhysicsQuantitiesReport, SimulationBenchmarkReport
from .results.epr import epr_result, reanalyze_epr, resolve_epr_result
from .results.epr_records import PalaceEprResult
from .results.inspection_records import (
    PalaceFailureDiagnosis,
    PalaceResultSelection,
    PassCostRecord,
    SurfaceMarginEprRecord,
    SurfaceMarginEprSeriesSnapshot,
)
from .results.resolved_records import (
    PalaceCost,
    PalacePerformance,
    PalaceProvenance,
    PalaceReturnedReceipt,
    ParsedTable,
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
