"""Own Q3D setup creation and native control readback."""

from __future__ import annotations

from typing import Any, cast

from scgsim.aedt.runtime.native.common import (
    saved_setup_properties as _saved_setup_properties,
)
from scgsim.aedt.specs.common import MatrixRunControl
from scgsim.aedt.specs.q3d import Q3dSpec


def setup_q3d(app: Any, spec: Q3dSpec, control: MatrixRunControl) -> None:
    """Create only the requested capacitance and optional AC resistance/inductance setup."""
    if app.setup_names:
        raise RuntimeError("new Q3D design must not inherit a setup")
    setup = app.create_setup(control.setup_name)
    setup.dc_enabled = False
    setup.ac_rl_enabled = spec.solve_ac_rl
    setup.capacitance_enabled = True
    setup.props["AdaptiveFreq"] = f"{control.frequency_ghz:g}GHz"
    setup.props["Cap"]["MaxPass"] = control.maximum_passes
    setup.props["Cap"]["MinPass"] = control.minimum_passes
    setup.props["Cap"]["MinConvPass"] = control.minimum_converged_passes
    setup.props["Cap"]["PerError"] = control.convergence_percent
    setup.props["Cap"]["PerRefine"] = control.percent_refinement
    if spec.solve_ac_rl:
        setup.props["AC"]["MaxPass"] = control.maximum_passes
        setup.props["AC"]["MinPass"] = control.minimum_passes
        setup.props["AC"]["MinConvPass"] = control.minimum_converged_passes
        setup.props["AC"]["PerError"] = control.convergence_percent
        setup.props["AC"]["PerRefine"] = control.percent_refinement
    if not setup.update():
        raise RuntimeError("Q3D setup update failed")


def read_q3d_setup(
    app: Any, spec: Q3dSpec, control: MatrixRunControl
) -> dict[str, Any]:
    """Compare saved capacitance and AC resistance/inductance controls with the request."""
    raw = _saved_setup_properties(app, control.setup_name)
    cap = raw.get("Cap")
    ac = raw.get("AC")
    if not isinstance(cap, dict) or (spec.solve_ac_rl != isinstance(ac, dict)):
        raise TypeError("Q3D saved setup does not match its capacitance/AC-RL mode")
    if not spec.solve_ac_rl and ("AC" in raw or "DC" in raw):
        raise RuntimeError("Q3D capacitance-only setup saved AC-RL or DC controls")
    native = {
        "adaptive_frequency": raw.get("AdaptiveFreq"),
        "capacitance_maximum_passes": cap.get("MaxPass"),
        "capacitance_minimum_passes": cap.get("MinPass"),
        "capacitance_minimum_converged_passes": cap.get("MinConvPass"),
        "capacitance_convergence_percent": cap.get("PerError"),
        "capacitance_percent_refinement": cap.get("PerRefine"),
        "dc_enabled": "DC" in raw,
    }
    if spec.solve_ac_rl:
        ac = cast(dict[str, Any], ac)
        native.update(
            {
                "ac_rl_maximum_passes": ac.get("MaxPass"),
                "ac_rl_minimum_passes": ac.get("MinPass"),
                "ac_rl_minimum_converged_passes": ac.get("MinConvPass"),
                "ac_rl_convergence_percent": ac.get("PerError"),
                "ac_rl_percent_refinement": ac.get("PerRefine"),
            }
        )
    expected = {
        "adaptive_frequency": f"{control.frequency_ghz:g}GHz",
        "capacitance_maximum_passes": control.maximum_passes,
        "capacitance_minimum_passes": control.minimum_passes,
        "capacitance_minimum_converged_passes": (control.minimum_converged_passes),
        "capacitance_convergence_percent": control.convergence_percent,
        "capacitance_percent_refinement": control.percent_refinement,
        "dc_enabled": False,
    }
    if spec.solve_ac_rl:
        expected.update(
            {
                "ac_rl_maximum_passes": control.maximum_passes,
                "ac_rl_minimum_passes": control.minimum_passes,
                "ac_rl_minimum_converged_passes": (control.minimum_converged_passes),
                "ac_rl_convergence_percent": control.convergence_percent,
                "ac_rl_percent_refinement": control.percent_refinement,
            }
        )
    if native != expected:
        raise RuntimeError(f"Q3D saved setup readback mismatch: {native!r}")
    return {"name": control.setup_name, "native": native}
