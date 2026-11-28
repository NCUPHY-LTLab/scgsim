"""Strict offline validation for HFSS native model, setup, and solver readback."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

from scgsim.aedt.results.convergence.hfss import read_hfss_convergence
from scgsim.aedt.specs.hfss import (
    EigenmodeRunControl,
    HfssEprSpec,
    HfssRunControl,
    HfssSpec,
)


def validate_hfss_setup(receipt: dict[str, Any], spec: Any) -> None:
    """Bind the native setup record to the requested HFSS controls."""
    if isinstance(spec, HfssEprSpec):
        run_control = cast(EigenmodeRunControl, spec.run_control)
        native = {
            "minimum_frequency": f"{run_control.minimum_frequency_ghz:g}GHz",
            "num_modes": run_control.num_modes,
            "maximum_delta_frequency_percent": (
                run_control.maximum_delta_frequency_percent
            ),
            "maximum_passes": run_control.maximum_passes,
            "minimum_passes": run_control.minimum_passes,
            "minimum_converged_passes": run_control.minimum_converged_passes,
            "percent_refinement": run_control.percent_refinement,
        }
    else:
        run_control = cast(HfssRunControl, spec.run_control)
        native = {
            "solve_type": "Broadband",
            "low_frequency": f"{run_control.sweep.start_ghz:g}GHz",
            "high_frequency": f"{run_control.sweep.stop_ghz:g}GHz",
            "maximum_delta_s": run_control.maximum_delta_s,
            "maximum_passes": run_control.maximum_passes,
            "minimum_passes": run_control.minimum_passes,
            "minimum_converged_passes": run_control.minimum_converged_passes,
            "percent_refinement": run_control.percent_refinement,
        }
    setup = receipt.get("setup")
    expected = {"name": run_control.setup_name, "native": native}
    if (
        not isinstance(setup, dict)
        or (
            {key: setup.get(key) for key in expected}
            if isinstance(spec, HfssEprSpec)
            else setup
        )
        != expected
    ):
        raise RuntimeError("HFSS native setup evidence is invalid")


def validate_hfss_setup_and_convergence(
    root: Path, receipt: dict[str, Any], spec: HfssSpec
) -> None:
    """Verify HFSS setup and convergence artifacts against the completed receipt."""
    validate_hfss_setup(receipt, spec)
    if receipt.get("convergence") != read_hfss_convergence(root, spec):
        raise RuntimeError("HFSS native convergence evidence is invalid")


def validate_eigenmode_readback(receipt: dict[str, Any], spec: HfssEprSpec) -> None:
    """Verify that the receipt contains the requested native Eigenmode outputs."""
    readback = receipt.get("result_readback")
    # Lumped loads belong to the geometry; an Eigenmode receipt has no ports field.
    if "ports" in receipt:
        raise RuntimeError("an HFSS Eigenmode receipt must not contain ports")
    readback = readback.get("final_modes") if isinstance(readback, dict) else None
    values = readback.get("eigenmodes") if isinstance(readback, dict) else None
    if not isinstance(values, dict):
        raise TypeError("completed receipt has no Eigenmode result readback")
    run_control = cast(EigenmodeRunControl, spec.run_control)
    modes = run_control.num_modes
    frequencies = values.get("frequencies_ghz")
    q_factors = values.get("q_factors")
    if (
        values.get("modes") != modes
        or values.get("frequency_unit") != "GHz"
        or values.get("mode_indices") != list(range(1, modes + 1))
        or not isinstance(frequencies, list)
        or len(frequencies) != modes
        or not all(
            isinstance(value, (int, float)) and math.isfinite(value) and value > 0
            for value in frequencies
        )
        or not isinstance(q_factors, list)
        or len(q_factors) != modes
        or not all(
            isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
            for value in q_factors
        )
        or values.get("native_export") != "eigenmodes.eig"
        or not isinstance(values.get("native_export_bytes"), int)
        or values["native_export_bytes"] <= 0
    ):
        raise RuntimeError("completed receipt Eigenmode result readback is invalid")
