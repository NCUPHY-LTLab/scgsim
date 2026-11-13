"""Validate and lower Palace numerical controls into config fields."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

from .config_models import positive_finite


@dataclass(frozen=True)
class _SolverSettings:
    order: int
    tolerance: float
    max_iterations: int
    solver_type: str
    preconditioner: str
    device: str
    ksp_type: str | None


@dataclass(frozen=True)
class _AmrSettings:
    max_passes: int | None
    nonconformal: bool
    nc_levels: int | None
    tolerance: float | None
    update_fraction: float | None
    save_iterations: bool | None
    save_mesh: bool | None
    estimator_mg: bool | None
    output_paraview: bool | None
    output_grid_function: bool | None


@dataclass(frozen=True)
class _ControlSettings:
    solver: _SolverSettings
    amr: _AmrSettings


def configure_numerical_controls(  # noqa: PLR0913 - stable named notebook controls API.
    *,
    order: int = 1,
    tolerance: float = 1e-6,
    max_iterations: int = 400,
    solver_type: str = "Default",
    preconditioner: str = "Default",
    device: str = "CPU",
    ksp_type: str | None = None,
    amr_max_passes: int | None = 0,
    amr_nonconformal: bool = False,
    amr_nc_levels: int | None = None,
    amr_tolerance: float | None = 1e-2,
    amr_update_fraction: float | None = None,
    save_adapt_iterations: bool | None = None,
    save_adapt_mesh: bool | None = None,
    estimator_mg: bool | None = None,
    output_paraview: bool | None = None,
    output_grid_function: bool | None = None,
) -> dict[str, Any]:
    """Validate notebook numerical controls for Palace config synthesis.

    Mesh controls are not numerical controls: ``set_mesh`` holds the recipe, mesher options, budgets and
    quality threshold of ``scgsim.meshing``.

    ``amr_max_passes`` becomes ``Model.Refinement.MaxIts``.
    ``amr_nonconformal`` becomes ``Model.Refinement.Nonconformal`` and defaults
    to ``False`` so generated configs do not inherit Palace's ``true`` default.
    ``amr_nc_levels`` optionally sets ``MaxNCLevels``; zero means no native limit.
    An omitted ``ksp_type`` retains SCGSim's problem-specific default; explicit
    ``"Default"`` selects Palace's own problem-specific Krylov default.
    """
    solver = _SolverSettings(
        order, tolerance, max_iterations, solver_type, preconditioner, device, ksp_type
    )
    max_iterations = _validate_solver_controls(solver)
    solver = _SolverSettings(
        order, tolerance, max_iterations, solver_type, preconditioner, device, ksp_type
    )
    amr = _AmrSettings(
        amr_max_passes,
        amr_nonconformal,
        amr_nc_levels,
        amr_tolerance,
        amr_update_fraction,
        save_adapt_iterations,
        save_adapt_mesh,
        estimator_mg,
        output_paraview,
        output_grid_function,
    )
    _validate_amr_controls(amr)
    return _assemble_controls(_ControlSettings(solver, amr))


def _validate_solver_controls(
    settings: _SolverSettings,
) -> int:
    order = settings.order
    tolerance = settings.tolerance
    max_iterations = settings.max_iterations
    solver_type = settings.solver_type
    preconditioner = settings.preconditioner
    device = settings.device
    if not isinstance(order, int) or isinstance(order, bool) or not 1 <= order <= 4:
        raise ValueError("order must be an integer from 1 through 4.")
    if not positive_finite(tolerance):
        raise ValueError("tolerance must be finite and positive.")
    if (
        not isinstance(max_iterations, int)
        or isinstance(max_iterations, bool)
        or max_iterations <= 0
    ):
        raise ValueError("max_iterations must be a positive integer.")
    if settings.ksp_type is not None and settings.ksp_type not in {
        "Default",
        "CG",
        "MINRES",
        "GMRES",
        "FGMRES",
        "BiCGSTAB",
    }:
        raise ValueError("unsupported ksp_type.")
    if solver_type not in {"Default", "SuperLU", "STRUMPACK", "MUMPS"}:
        raise ValueError("unsupported solver_type.")
    if preconditioner not in {"Default", "AMS", "BoomerAMG"}:
        raise ValueError("unsupported preconditioner.")
    if device not in {"CPU", "GPU"}:
        raise ValueError("unsupported device.")
    if solver_type != "Default" and preconditioner != "Default":
        raise ValueError("non-default preconditioner requires Default solver.")
    return 1 if solver_type == "MUMPS" else max_iterations


def _validate_amr_controls(
    settings: _AmrSettings,
) -> None:
    max_passes = settings.max_passes
    nonconformal = settings.nonconformal
    tolerance = settings.tolerance
    update_fraction = settings.update_fraction
    if max_passes is not None:
        if not isinstance(max_passes, int) or isinstance(max_passes, bool):
            raise TypeError("amr_max_passes must be an integer.")
        if max_passes < 0:
            raise ValueError("amr_max_passes must be >= 0.")
    if not isinstance(nonconformal, bool):
        raise TypeError("amr_nonconformal must be a bool.")
    if tolerance is not None and not positive_finite(tolerance):
        raise ValueError("amr_tolerance must be finite and positive.")
    _validate_nc_levels(settings.nc_levels)
    _validate_update_fraction(update_fraction)
    _validate_optional_boolean(settings.save_iterations, "save_adapt_iterations")
    _validate_optional_boolean(settings.save_mesh, "save_adapt_mesh")
    _validate_optional_boolean(settings.estimator_mg, "estimator_mg")
    _validate_optional_boolean(settings.output_paraview, "output_paraview")
    _validate_optional_boolean(settings.output_grid_function, "output_grid_function")


def _validate_nc_levels(value: int | None) -> None:
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("amr_nc_levels must be an integer or None.")
    if value < 0:
        raise ValueError("amr_nc_levels must be >= 0; zero selects no native limit.")


def _validate_update_fraction(value: float | None) -> None:
    if value is None:
        return
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError("amr_update_fraction must be a real number.")
    fraction = float(value)
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError(
            "amr_update_fraction must be finite and strictly between 0 and 1."
        )


def _validate_optional_boolean(value: bool | None, name: str) -> None:
    if value is not None and not isinstance(value, bool):
        raise TypeError(f"{name} must be bool or None.")


def _assemble_controls(
    settings: _ControlSettings,
) -> dict[str, Any]:
    solver = settings.solver
    amr = settings.amr
    controls = {
        "order": int(solver.order),
        "tolerance": float(solver.tolerance),
        "max_iterations": int(solver.max_iterations),
        "solver_type": str(solver.solver_type),
        "preconditioner": str(solver.preconditioner),
        "device": str(solver.device),
        "amr_max_passes": int(amr.max_passes) if amr.max_passes is not None else None,
        "amr_nonconformal": bool(amr.nonconformal),
        "amr_tolerance": float(amr.tolerance) if amr.tolerance is not None else None,
        "amr_update_fraction": (
            float(amr.update_fraction) if amr.update_fraction is not None else None
        ),
        "save_adapt_iterations": amr.save_iterations,
        "estimator_mg": amr.estimator_mg,
        "output_paraview": amr.output_paraview,
        "output_grid_function": amr.output_grid_function,
    }
    if solver.ksp_type is not None:
        controls["ksp_type"] = solver.ksp_type
    if amr.nc_levels is not None:
        controls["amr_nc_levels"] = amr.nc_levels
    if amr.save_mesh is not None:
        controls["save_adapt_mesh"] = amr.save_mesh
    return controls


def linear_solver(
    numerical: Mapping[str, Any], *, problem: str | None = None
) -> dict[str, Any]:
    """Lower problem-specific defaults while retaining explicit solver choices.

    Electrostatic uses conjugate gradients (CG) with BoomerAMG by default.
    Explicit solver/preconditioner paths and other problems retain GMRES.
    Explicit ``ksp_type="Default"`` delegates Krylov selection to Palace.
    """
    use_electrostatic_defaults = (
        problem == "Electrostatic"
        and numerical["solver_type"] == "Default"
        and numerical["preconditioner"] == "Default"
    )
    linear = {
        "Type": numerical["solver_type"],
        "KSPType": numerical.get(
            "ksp_type", "CG" if use_electrostatic_defaults else "GMRES"
        ),
        "Tol": float(numerical["tolerance"]),
        "MaxIts": int(numerical["max_iterations"]),
    }
    if numerical["solver_type"] == "MUMPS":
        linear.update(
            {
                "MaxIts": 1,
                "MGMaxLevels": 1,
                "PCMatReal": False,
                "ComplexCoarseSolve": True,
            }
        )
    if (
        numerical["solver_type"] == "Default"
        and numerical["preconditioner"] != "Default"
    ):
        linear["Type"] = numerical["preconditioner"]
    if use_electrostatic_defaults:
        linear["Type"] = "BoomerAMG"
    if numerical.get("estimator_mg") is not None:
        linear["EstimatorMG"] = bool(numerical["estimator_mg"])
    return linear


def build_refinement(numerical: Mapping[str, Any]) -> dict[str, Any]:
    """Translate AMR controls into Palace's Model.Refinement block."""
    max_passes = numerical.get("amr_max_passes")
    if max_passes is None:
        raise ValueError(
            "amr_max_passes must be explicitly set before config generation."
        )
    refinement = {
        "UniformLevels": 0,
        "MaxIts": int(max_passes),
        "Nonconformal": bool(numerical.get("amr_nonconformal", False)),
    }
    if numerical.get("amr_nc_levels") is not None:
        refinement["MaxNCLevels"] = int(numerical["amr_nc_levels"])
    if numerical.get("amr_tolerance") is not None:
        refinement["Tol"] = float(numerical["amr_tolerance"])
    if numerical.get("save_adapt_iterations") is not None:
        refinement["SaveAdaptIterations"] = bool(numerical["save_adapt_iterations"])
    if numerical.get("save_adapt_mesh") is not None:
        refinement["SaveAdaptMesh"] = bool(numerical["save_adapt_mesh"])
    if numerical.get("amr_update_fraction") is not None:
        refinement["UpdateFraction"] = float(numerical["amr_update_fraction"])
    return refinement


def build_output_formats(numerical: Mapping[str, Any]) -> dict[str, Any]:
    """Translate optional field-output switches into Palace's output block."""
    output_formats = {
        "Paraview": numerical.get("output_paraview"),
        "GridFunction": numerical.get("output_grid_function"),
    }
    outputs = {
        key: bool(value) for key, value in output_formats.items() if value is not None
    }
    if not outputs:
        return {}
    return {"OutputFormats": outputs}


__all__ = [
    "build_output_formats",
    "build_refinement",
    "configure_numerical_controls",
    "linear_solver",
]
