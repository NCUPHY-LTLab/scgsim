"""Prepare, run and read back AEDT (HFSS, Q3D, Q2D) calculations.

In: an AEDT request spec for ``prepare_handoff``, with the request's controls. HFSS Eigenmode and the Q3D C/G preparation can also start from a ``GeometryPlan`` snapshot; Q2D starts from a ``Q2dSpec`` of explicit rectangles and conductor groups.
Out: a portable handoff that an AEDT run reads; from a finished run the detached results (``ResolvedRun``, ``EprResult``); from a preparation-only run, which never invokes the solver, the ``PreparedModel``.
Next: ``python -m scgsim.aedt.run`` executes the handoff; notebooks read a finished run through ``resolve_results``, a preparation-only run through ``resolve_prepared_model``, and plot with the plot helpers.
"""

from importlib import import_module
from typing import TYPE_CHECKING

from scgsim.aedt.epr.models import EprResult
from scgsim.aedt.specs.common import (
    LumpedRlc,
)
from scgsim.aedt.specs.epr import (
    EprAnalysisRequest,
    ExpressionCacheConvergence,
    ExpressionCacheConvergenceSet,
    NativeExpressionDefinition,
    NormalizedSurfaceEprTotal,
    PlanarJunction,
    SavedSolution,
)

from .drc import SimulationDrcReport, StageResult
from .eigenmode import EigenmodeSim
from .epr.reanalysis import reanalyze_epr
from .epr.workflow import analyze_epr
from .preparation.geometry import (
    prepare_hfss_driven_from_geometry,
    prepare_hfss_eigenmode_from_geometry,
    prepare_q3d_from_geometry,
)
from .preparation.handoff import HandoffPlan, prepare_handoff
from .preparation.planar_junctions import planar_junction_from_port
from .preparation.sgb_model import PreparedSgbModel, prepare_sgb_model
from .presentation.epr import plot_epr_result, show_epr
from .results.epr import resolve_epr_result, resolve_saved_solution
from .results.prepared import PreparedModel, resolve_prepared_model
from .results.resolve import ResolvedRun, resolve_results
from .runtime.control import request_q3d_abort_and_save, request_q3d_stop_and_save
from .specs.common import (
    AedtResources,
    LumpedTerminalPort,
    MatrixRunControl,
    ModalPort,
    PdkMaterial,
    TerminalPort,
)
from .specs.hfss import (
    EigenmodeRunControl,
    FrequencySweepSpec,
    HfssDrivenGeometrySpec,
    HfssEprAnalysisSpec,
    HfssEprSpec,
    HfssRunControl,
    HfssSpec,
)
from .specs.parse import AedtSpec, parse_aedt_spec
from .specs.q2d import Q2dConductorSpec, Q2dRectangleSpec, Q2dSpec
from .specs.q3d import Q3dNetSpec, Q3dSpec

if TYPE_CHECKING:
    from .batch import (
        NativeBatchPackage,
        collect_native_batch,
        prepare_native_batch,
        run_native_batch,
    )


__all__ = [
    "AedtResources",
    "AedtSpec",
    "EigenmodeRunControl",
    "EigenmodeSim",
    "EprAnalysisRequest",
    "EprResult",
    "ExpressionCacheConvergence",
    "ExpressionCacheConvergenceSet",
    "FrequencySweepSpec",
    "HandoffPlan",
    "HfssDrivenGeometrySpec",
    "LumpedRlc",
    "LumpedTerminalPort",
    "HfssEprAnalysisSpec",
    "HfssEprSpec",
    "HfssRunControl",
    "HfssSpec",
    "MatrixRunControl",
    "ModalPort",
    "NativeExpressionDefinition",
    "NormalizedSurfaceEprTotal",
    "PdkMaterial",
    "PlanarJunction",
    "PreparedModel",
    "PreparedSgbModel",
    "Q2dConductorSpec",
    "Q2dRectangleSpec",
    "Q2dSpec",
    "Q3dNetSpec",
    "Q3dSpec",
    "ResolvedRun",
    "SavedSolution",
    "SimulationDrcReport",
    "StageResult",
    "TerminalPort",
    "analyze_epr",
    "parse_aedt_spec",
    "planar_junction_from_port",
    "plot_epr_result",
    "prepare_handoff",
    "prepare_hfss_eigenmode_from_geometry",
    "prepare_hfss_driven_from_geometry",
    "prepare_q3d_from_geometry",
    "prepare_sgb_model",
    "reanalyze_epr",
    "request_q3d_stop_and_save",
    "request_q3d_abort_and_save",
    "resolve_epr_result",
    "resolve_results",
    "resolve_prepared_model",
    "resolve_saved_solution",
    "show_epr",
]


__all__ += [
    "NativeBatchPackage",
    "prepare_native_batch",
    "run_native_batch",
    "collect_native_batch",
]


def __getattr__(name: str):
    # Keep the standalone batch CLI unimported until a facade export is used.
    if name in {
        "NativeBatchPackage",
        "prepare_native_batch",
        "run_native_batch",
        "collect_native_batch",
    }:
        value = getattr(import_module(".batch", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
