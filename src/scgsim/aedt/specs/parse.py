"""Turn a request payload into the right spec class from its `schema_version`.

The schema version names the family. The family is never guessed from the other
keys, so an unknown or mistyped version fails loudly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scgsim.aedt.specs.common import (
    DRIVEN_GEOMETRY_SCHEMA_VERSION,
    EIGENMODE_SCHEMA_VERSION,
    EPR_ANALYSIS_SCHEMA_VERSION_V2,
    EPR_EIGENMODE_SCHEMA_VERSION_V2,
    EPR_EIGENMODE_SCHEMA_VERSION_V3,
    Q2D_SCHEMA_VERSION,
    Q2D_SCHEMA_VERSION_V2,
    Q3D_SCHEMA_VERSION,
    SCHEMA_VERSION,
)
from scgsim.aedt.specs.hfss import HfssDrivenSpec, HfssEigenmodeSpec
from scgsim.aedt.specs.hfss_prepared import (
    HfssDrivenGeometrySpec,
    HfssEprAnalysisSpec,
    HfssEprSpec,
)
from scgsim.aedt.specs.q2d import Q2dSpec
from scgsim.aedt.specs.q3d import Q3dSpec

HfssSpec = (
    HfssDrivenGeometrySpec
    | HfssDrivenSpec
    | HfssEigenmodeSpec
    | HfssEprSpec
    | HfssEprAnalysisSpec
)
AedtSpec = HfssSpec | Q3dSpec | Q2dSpec


def parse_aedt_spec(
    payload: dict[str, Any], *, base_dir: Path | None = None
) -> AedtSpec:
    """Dispatch one explicit AEDT schema without inferring solver family."""
    if payload.get("schema_version") == DRIVEN_GEOMETRY_SCHEMA_VERSION:
        return HfssDrivenGeometrySpec.from_payload(payload)
    if payload.get("schema_version") == SCHEMA_VERSION:
        return HfssDrivenSpec.from_payload(payload, base_dir=base_dir)
    if payload.get("schema_version") == EIGENMODE_SCHEMA_VERSION:
        return HfssEigenmodeSpec.from_payload(payload, base_dir=base_dir)
    if payload.get("schema_version") in {
        EPR_EIGENMODE_SCHEMA_VERSION_V2,
        EPR_EIGENMODE_SCHEMA_VERSION_V3,
    }:
        return HfssEprSpec.from_payload(payload)
    if payload.get("schema_version") == EPR_ANALYSIS_SCHEMA_VERSION_V2:
        return HfssEprAnalysisSpec.from_payload(payload, base_dir=base_dir)
    if payload.get("schema_version") == Q3D_SCHEMA_VERSION:
        return Q3dSpec.from_payload(payload, base_dir=base_dir)
    if payload.get("schema_version") in {Q2D_SCHEMA_VERSION, Q2D_SCHEMA_VERSION_V2}:
        return Q2dSpec.from_payload(payload)
    raise ValueError(f"unsupported AEDT schema {payload.get('schema_version')!r}")
