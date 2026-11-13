"""Offline dispatch for family-specific native result readback checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scgsim.aedt.io import file_sha256
from scgsim.aedt.results.hfss_readback import (
    validate_eigenmode_readback,
    validate_hfss_setup_and_convergence,
    validate_modal_native_evidence,
)
from scgsim.aedt.results.q2d_readback import validate_q2d_readback
from scgsim.aedt.results.q3d_readback import validate_q3d_readback
from scgsim.aedt.results.receipt_validation import contained
from scgsim.aedt.results.terminal_readback import (
    validate_body_terminal_readback,
    validate_terminal_native_evidence,
)
from scgsim.aedt.specs.hfss import HfssDrivenGeometrySpec, HfssEigenmodeSpec
from scgsim.aedt.specs.q2d import Q2dSpec
from scgsim.aedt.specs.q3d import Q3dSpec


def validate_result_readback(root: Path, receipt: dict[str, Any], spec: Any) -> None:
    """Check the receipt's connected versions, then the native readback of the spec's family (Q2D, Q3D, Eigenmode or scattering)."""
    connected = receipt.get("connected")
    if not isinstance(connected, dict) or connected != {
        "aedt_version": spec.aedt_version,
        "pyaedt_version": spec.pyaedt_version,
    }:
        raise RuntimeError("completed receipt has invalid connected version identity")
    readback = receipt.get("result_readback")
    if isinstance(spec, Q2dSpec):
        validate_q2d_readback(root, receipt, spec)
        _validate_diagnostics(root, receipt.get("diagnostics"))
        return
    if isinstance(spec, Q3dSpec):
        validate_q3d_readback(root, receipt, spec)
        _validate_diagnostics(root, receipt.get("diagnostics"))
        return
    if isinstance(spec, HfssEigenmodeSpec):
        validate_eigenmode_readback(receipt, spec)
        validate_hfss_setup_and_convergence(root, receipt, spec)
        _validate_diagnostics(root, receipt.get("diagnostics"))
        return
    key = "terminal_st" if spec.mode == "terminal" else "modal_s"
    if not isinstance(readback, dict) or not isinstance(readback.get(key), dict):
        raise TypeError("completed receipt has no canonical result readback")
    frequency = readback[key]
    if frequency != {
        "records": int(spec.run_control.sweep.points),
        "frequency_unit": "GHz",
        "first_frequency_ghz": spec.run_control.sweep.start_ghz,
        "last_frequency_ghz": spec.run_control.sweep.stop_ghz,
        "strictly_increasing": True,
    }:
        raise RuntimeError("completed receipt result frequency readback is invalid")
    touchstone = readback.get("touchstone")
    ports = receipt.get("ports")
    port_key = "terminal_excitation" if spec.mode == "terminal" else "modal_excitation"
    port_order = (
        [record.get(port_key) for record in ports]
        if isinstance(ports, list)
        and len(ports) == 2
        and all(isinstance(record, dict) for record in ports)
        and [record.get("index") for record in ports] == [1, 2]
        else None
    )
    if (
        not isinstance(touchstone, dict)
        or touchstone.get("path") != f"{spec.mode}.s2p"
        or touchstone.get("ports") != 2
        or touchstone.get("records") != int(spec.run_control.sweep.points)
        or touchstone.get("frequency_unit") != "GHz"
        or touchstone.get("first_frequency_ghz") != spec.run_control.sweep.start_ghz
        or touchstone.get("last_frequency_ghz") != spec.run_control.sweep.stop_ghz
        or touchstone.get("strictly_increasing") is not True
        or touchstone.get("port_order") != port_order
        or not all(isinstance(name, str) and name for name in port_order or [])
        or len(set(port_order or [])) != 2
        or not isinstance(touchstone.get("bytes"), int)
        or touchstone["bytes"] <= 0
    ):
        raise RuntimeError(
            "completed receipt is missing a two-port Touchstone readback"
        )
    if spec.mode == "terminal":
        if isinstance(spec, HfssDrivenGeometrySpec):
            validate_body_terminal_readback(receipt, spec)
        else:
            validate_terminal_native_evidence(ports, spec)
    else:
        validate_modal_native_evidence(ports, spec)
    validate_hfss_setup_and_convergence(root, receipt, spec)
    _validate_diagnostics(root, receipt.get("diagnostics"))


def _validate_diagnostics(root: Path, diagnostics: Any) -> None:
    if not isinstance(diagnostics, dict) or diagnostics.get("batch_log") != "batch.log":
        raise RuntimeError("completed receipt diagnostics are invalid")
    batch_log = contained(root, "batch.log")
    if (
        diagnostics.get("present") is not True
        or not batch_log.is_file()
        or diagnostics.get("sha256") != file_sha256(batch_log)
        or not isinstance(diagnostics.get("physics_warnings"), list)
    ):
        raise RuntimeError("completed receipt diagnostics hash is invalid")
