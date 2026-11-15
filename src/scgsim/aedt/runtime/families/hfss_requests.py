"""Name the HFSS model families and validate their explicit solve controls."""

from __future__ import annotations

from scgsim.aedt.specs.hfss import (
    EigenmodeRunControl,
    HfssDrivenGeometrySpec,
    HfssEprAnalysisSpec,
    HfssEprSpec,
    HfssRunControl,
)

HfssModelSpec = HfssDrivenGeometrySpec
HfssEigenmodeModelSpec = HfssEprSpec | HfssEprAnalysisSpec


def driven_control(spec: HfssDrivenGeometrySpec) -> HfssRunControl:
    """Require the complete driven controls at a native setup or export boundary."""
    if spec.run_control is None:
        raise ValueError("HFSS solving requires an explicit complete run_control")
    return spec.run_control


def eigenmode_control(spec: HfssEigenmodeModelSpec) -> EigenmodeRunControl:
    """Require the complete eigenmode controls at a native readback boundary."""
    if spec.run_control is None:
        raise ValueError("Eigenmode solving requires an explicit complete run_control")
    return spec.run_control
