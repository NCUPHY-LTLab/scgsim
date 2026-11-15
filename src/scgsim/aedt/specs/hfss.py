"""Public HFSS request facade; definitions live with their solver responsibilities."""

from __future__ import annotations

from scgsim.aedt.specs.common import (
    DRIVEN_GEOMETRY_SCHEMA_VERSION,
    EPR_ANALYSIS_SCHEMA_VERSION_V2,
    EPR_ANALYSIS_SCHEMA_VERSION_V3,
    EPR_EIGENMODE_SCHEMA_VERSION_V2,
    EPR_EIGENMODE_SCHEMA_VERSION_V3,
    EPR_EIGENMODE_SCHEMA_VERSION_V4,
    LOCKED_PYAEDT,
    OFFICIAL_PYAEDT_SOURCE_URL,
    POINT_COUNT,
    LumpedRlc,
    LumpedTerminalPort,
    ModalPort,
    PdkMaterial,
    TerminalPort,
)
from scgsim.aedt.specs.hfss_common import (
    EigenmodeRunControl,
    FrequencySweepSpec,
    HfssRunControl,
)
from scgsim.aedt.specs.hfss_driven import HfssDrivenGeometrySpec
from scgsim.aedt.specs.hfss_epr import HfssEprAnalysisSpec, HfssEprSpec
from scgsim.aedt.specs.modeling import (
    Modeling,
)

HfssSpec = HfssDrivenGeometrySpec | HfssEprSpec | HfssEprAnalysisSpec

__all__ = [
    "DRIVEN_GEOMETRY_SCHEMA_VERSION",
    "EPR_ANALYSIS_SCHEMA_VERSION_V2",
    "EPR_ANALYSIS_SCHEMA_VERSION_V3",
    "EPR_EIGENMODE_SCHEMA_VERSION_V2",
    "EPR_EIGENMODE_SCHEMA_VERSION_V3",
    "EPR_EIGENMODE_SCHEMA_VERSION_V4",
    "EigenmodeRunControl",
    "FrequencySweepSpec",
    "HfssDrivenGeometrySpec",
    "HfssEprAnalysisSpec",
    "HfssEprSpec",
    "HfssRunControl",
    "HfssSpec",
    "LOCKED_PYAEDT",
    "LumpedRlc",
    "LumpedTerminalPort",
    "ModalPort",
    "Modeling",
    "OFFICIAL_PYAEDT_SOURCE_URL",
    "PdkMaterial",
    "POINT_COUNT",
    "TerminalPort",
]
