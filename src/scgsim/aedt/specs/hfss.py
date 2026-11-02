"""Public HFSS request facade; definitions live with their solver responsibilities."""

from __future__ import annotations

from scgsim.aedt.specs.common import (
    DRIVEN_GEOMETRY_SCHEMA_VERSION,
    EIGENMODE_SCHEMA_VERSION,
    EPR_ANALYSIS_SCHEMA_VERSION,
    EPR_ANALYSIS_SCHEMA_VERSION_V2,
    EPR_ANALYSIS_SCHEMA_VERSION_V3,
    EPR_EIGENMODE_SCHEMA_VERSION,
    EPR_EIGENMODE_SCHEMA_VERSION_V2,
    EPR_EIGENMODE_SCHEMA_VERSION_V3,
    EPR_EIGENMODE_SCHEMA_VERSION_V4,
    LOCKED_PYAEDT,
    OFFICIAL_PYAEDT_SOURCE_URL,
    POINT_COUNT,
    SCHEMA_VERSION,
    HfssDrivenMode,
    LayerImport,
    LumpedRlc,
    LumpedTerminalPort,
    ModalPort,
    ObjectBinding,
    PdkMaterial,
    TerminalPort,
)
from scgsim.aedt.specs.hfss_common import (
    EigenmodeRunControl,
    FrequencySweepSpec,
    HfssRunControl,
    LengthMeshSpec,
)
from scgsim.aedt.specs.hfss_driven import HfssDrivenGeometrySpec, HfssDrivenSpec
from scgsim.aedt.specs.hfss_eigenmode import HfssEigenmodeSpec
from scgsim.aedt.specs.hfss_epr import HfssEprAnalysisSpec, HfssEprSpec
from scgsim.aedt.specs.modeling import (
    Modeling,
    PhysicalLayerDeclaration,
    PhysicalLayerSpec,
    PhysicalRepresentationPlan,
)

HfssSpec = (
    HfssDrivenGeometrySpec
    | HfssDrivenSpec
    | HfssEigenmodeSpec
    | HfssEprSpec
    | HfssEprAnalysisSpec
)

__all__ = [
    "DRIVEN_GEOMETRY_SCHEMA_VERSION",
    "EIGENMODE_SCHEMA_VERSION",
    "EPR_ANALYSIS_SCHEMA_VERSION",
    "EPR_ANALYSIS_SCHEMA_VERSION_V2",
    "EPR_ANALYSIS_SCHEMA_VERSION_V3",
    "EPR_EIGENMODE_SCHEMA_VERSION",
    "EPR_EIGENMODE_SCHEMA_VERSION_V2",
    "EPR_EIGENMODE_SCHEMA_VERSION_V3",
    "EPR_EIGENMODE_SCHEMA_VERSION_V4",
    "EigenmodeRunControl",
    "FrequencySweepSpec",
    "HfssDrivenMode",
    "HfssDrivenGeometrySpec",
    "HfssDrivenSpec",
    "HfssEigenmodeSpec",
    "HfssEprAnalysisSpec",
    "HfssEprSpec",
    "HfssRunControl",
    "HfssSpec",
    "LayerImport",
    "LengthMeshSpec",
    "LOCKED_PYAEDT",
    "LumpedRlc",
    "LumpedTerminalPort",
    "ModalPort",
    "Modeling",
    "ObjectBinding",
    "OFFICIAL_PYAEDT_SOURCE_URL",
    "PhysicalLayerDeclaration",
    "PhysicalLayerSpec",
    "PhysicalRepresentationPlan",
    "PdkMaterial",
    "POINT_COUNT",
    "SCHEMA_VERSION",
    "TerminalPort",
]
