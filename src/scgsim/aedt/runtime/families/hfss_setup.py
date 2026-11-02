"""Own HFSS driven and eigenmode setup creation and readback."""

from __future__ import annotations

from typing import Any

from scgsim.aedt.runtime.families.hfss_requests import (
    HfssModelSpec,
    driven_control,
    eigenmode_control,
)
from scgsim.aedt.runtime.native.common import (
    saved_setup_properties,
)
from scgsim.aedt.runtime.native.common import (
    saved_setup_properties as _saved_setup_properties,
)
from scgsim.aedt.runtime.native.hfss_eigenmode import (
    create_eigenmode_setup,
    read_eigenmode_setup,
)
from scgsim.aedt.specs.hfss import (
    HfssEigenmodeSpec,
    HfssEprAnalysisSpec,
    HfssEprSpec,
)


def setup_hfss(hfss: Any, spec: HfssModelSpec) -> None:
    """Create the explicit driven or eigenmode setup with its authored controls."""
    if isinstance(spec, HfssEigenmodeSpec):
        create_eigenmode_setup(hfss, spec.run_control)
        return
    control = driven_control(spec)
    if hfss.setup_names:
        raise RuntimeError("new V1 design must not inherit a setup")
    setup = hfss.create_setup(control.setup_name)
    setup.props["SolveType"] = (
        "DrivenTerminal" if spec.mode == "terminal" else "DrivenModal"
    )
    if not setup.enable_adaptive_setup_broadband(
        f"{control.sweep.start_ghz}GHz",
        f"{control.sweep.stop_ghz}GHz",
        max_passes=control.maximum_passes,
        max_delta_s=control.maximum_delta_s,
    ):
        raise RuntimeError("adaptive setup initialization failed")
    setup.props["MinimumPasses"] = control.minimum_passes
    setup.props["MinimumConvergedPasses"] = control.minimum_converged_passes
    setup.props["PercentRefinement"] = control.percent_refinement
    if not setup.update():
        raise RuntimeError("HFSS Driven adaptive setup update failed")
    point_count = int(control.sweep.points)
    sweep = hfss.create_linear_count_sweep(
        control.setup_name,
        "GHz",
        control.sweep.start_ghz,
        control.sweep.stop_ghz,
        num_of_freq_points=point_count,
        name=control.sweep_name,
        save_fields=False,
        sweep_type="Fast",
    )
    if sweep is None:
        raise RuntimeError("Fast sweep creation failed")
    actual = (
        str(sweep.props.get("Type")),
        str(sweep.props.get("RangeStart")),
        str(sweep.props.get("RangeEnd")),
        int(sweep.props.get("RangeCount")),
    )
    expected = (
        "Fast",
        f"{control.sweep.start_ghz}GHz",
        f"{control.sweep.stop_ghz}GHz",
        point_count,
    )
    if actual != expected:
        raise RuntimeError(f"Fast sweep readback mismatch: {actual!r} != {expected!r}")


def read_hfss_setup(hfss: Any, spec: HfssModelSpec) -> dict[str, Any]:
    """Compare the saved native HFSS setup with the explicit run controls."""
    if isinstance(spec, HfssEigenmodeSpec):
        return read_eigenmode_setup(hfss, spec.run_control)
    control = driven_control(spec)
    raw = _saved_setup_properties(hfss, control.setup_name)
    frequencies = raw.get("MultipleAdaptiveFreqsSetup")
    if not isinstance(frequencies, dict):
        raise TypeError("HFSS Driven saved setup lacks broadband frequencies")
    native = {
        "solve_type": raw.get("SolveType"),
        "low_frequency": frequencies.get("Low"),
        "high_frequency": frequencies.get("High"),
        "maximum_delta_s": raw.get("MaxDeltaS"),
        "maximum_passes": raw.get("MaximumPasses"),
        "minimum_passes": raw.get("MinimumPasses"),
        "minimum_converged_passes": raw.get("MinimumConvergedPasses"),
        "percent_refinement": raw.get("PercentRefinement"),
    }
    expected = {
        "solve_type": "Broadband",
        "low_frequency": f"{control.sweep.start_ghz:g}GHz",
        "high_frequency": f"{control.sweep.stop_ghz:g}GHz",
        "maximum_delta_s": control.maximum_delta_s,
        "maximum_passes": control.maximum_passes,
        "minimum_passes": control.minimum_passes,
        "minimum_converged_passes": control.minimum_converged_passes,
        "percent_refinement": control.percent_refinement,
    }
    if native != expected:
        raise RuntimeError(f"HFSS saved setup readback mismatch: {native!r}")
    return {"name": control.setup_name, "native": native}


def read_body_eigenmode_setup(
    app: Any, spec: HfssEprSpec | HfssEprAnalysisSpec
) -> dict[str, Any]:
    """Verify the saved body-model eigenmode controls and field-saving policy."""
    control = eigenmode_control(spec)
    setup = read_eigenmode_setup(app, control)
    raw = saved_setup_properties(app, control.setup_name)
    creating_setup = isinstance(spec, HfssEprSpec)
    saved_field_properties = {
        "basis": (
            "completed Eigenmode LastAdaptive Fields evidence is checked after solve"
            if creating_setup
            else "existing saved-field analysis does not re-author setup field policy"
        ),
        "serialized": {
            key: raw[key]
            for key in ("SaveAnyFields", "SaveRadFieldsOnly")
            if key in raw
        },
        "status": (
            "unexposed_pending_postsolve"
            if creating_setup
            else "not_reauthored_existing_saved_solution"
        ),
    }
    return {
        "name": setup["name"],
        "native": setup["native"],
        "saved_fields": saved_field_properties,
    }
