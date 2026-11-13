"""The numerical controls of a Palace solve, and their lowering into ``config.json`` fields.

``NumericalControls`` is what ``set_numerical`` stores: the linear solver, the FEM order, adaptive mesh
refinement (AMR) and the field outputs.  It checks its values when it is built.  ``linear_solver``,
``build_refinement`` and ``build_output_formats`` lower it into ``Solver.Linear``, ``Model.Refinement`` and
``Problem.OutputFormats``.  Mesh controls are not numerical controls: ``set_mesh`` holds those.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal

_KSP_TYPES = {"Default", "CG", "MINRES", "GMRES", "FGMRES", "BiCGSTAB"}
_SOLVER_TYPES = {"Default", "SuperLU", "STRUMPACK", "MUMPS"}
_PRECONDITIONERS = {"Default", "AMS", "BoomerAMG"}


@dataclass(frozen=True)
class NumericalControls:
    """The solver, FEM, AMR and output controls ``write_config()`` lowers; checked when built.

    ``amr_max_passes`` is Palace ``Model.Refinement.MaxIts`` (0 disables AMR; None: not chosen yet, which
    ``write_config()`` refuses).  ``amr_nonconformal`` is ``Model.Refinement.Nonconformal``, always written so
    a config does not inherit Palace's ``true`` default.  ``amr_nc_levels`` sets ``MaxNCLevels``; 0 means no
    limit.  ``ksp_type`` None keeps SCGSim's problem-specific Krylov default; ``"Default"`` selects Palace's.
    Every other None leaves its Palace key out.  With the MUMPS direct solver ``max_iterations`` is held as 1.
    """

    order: int = 1
    tolerance: float = 1e-6
    max_iterations: int = 400
    solver_type: str = "Default"
    preconditioner: str = "Default"
    device: str = "CPU"
    ksp_type: str | None = None
    amr_max_passes: int | None = 0
    amr_nonconformal: bool = False
    amr_nc_levels: int | None = None
    amr_tolerance: float | None = 1e-2
    amr_update_fraction: float | None = None
    save_adapt_iterations: bool | None = None
    save_adapt_mesh: bool | None = None
    estimator_mg: bool | None = None
    output_paraview: bool | None = None
    output_grid_function: bool | None = None

    def __post_init__(self) -> None:
        """Refuse an invalid control, then hold the numbers in the types Palace reads."""
        _check_solver_controls(self)
        _check_amr_controls(self)
        # A frozen dataclass normalizes its own fields through object.__setattr__.
        object.__setattr__(self, "tolerance", float(self.tolerance))
        if self.solver_type == "MUMPS":
            object.__setattr__(self, "max_iterations", 1)
        if self.amr_tolerance is not None:
            object.__setattr__(self, "amr_tolerance", float(self.amr_tolerance))
        if self.amr_update_fraction is not None:
            object.__setattr__(
                self, "amr_update_fraction", float(self.amr_update_fraction)
            )

    def check_ready_for(self, problem: Literal["Eigenmode", "Electrostatic"]) -> None:
        """Refuse controls ``problem``'s config cannot be written from: AMR settings still left open.

        Raises:
            ValueError: ``amr_max_passes`` was never chosen, or Eigenmode AMR is enabled without its
                tolerance and update fraction.
        """
        if self.amr_max_passes is None:
            raise ValueError(
                f"{problem} requires explicit amr_max_passes before config generation: call "
                "set_numerical(amr_max_passes=...) before write_config(); 0 disables AMR."
            )
        if (
            problem == "Eigenmode"
            and self.amr_max_passes > 0
            and (self.amr_tolerance is None or self.amr_update_fraction is None)
        ):
            raise ValueError(
                "Enabled Eigenmode AMR requires explicit amr_tolerance and amr_update_fraction: call "
                "set_numerical(amr_tolerance=..., amr_update_fraction=...) before write_config()."
            )

    def to_payload(self) -> dict[str, Any]:
        """The controls as ``metadata/palace_numerical_controls.json`` records them; unset optional keys left out."""
        payload: dict[str, Any] = {
            "order": self.order,
            "tolerance": self.tolerance,
            "max_iterations": self.max_iterations,
            "solver_type": self.solver_type,
            "preconditioner": self.preconditioner,
            "device": self.device,
            "amr_max_passes": self.amr_max_passes,
            "amr_nonconformal": self.amr_nonconformal,
            "amr_tolerance": self.amr_tolerance,
            "amr_update_fraction": self.amr_update_fraction,
            "save_adapt_iterations": self.save_adapt_iterations,
            "estimator_mg": self.estimator_mg,
            "output_paraview": self.output_paraview,
            "output_grid_function": self.output_grid_function,
        }
        if self.ksp_type is not None:
            payload["ksp_type"] = self.ksp_type
        if self.amr_nc_levels is not None:
            payload["amr_nc_levels"] = self.amr_nc_levels
        if self.save_adapt_mesh is not None:
            payload["save_adapt_mesh"] = self.save_adapt_mesh
        return payload


def _positive_finite(value: Any) -> bool:
    """Whether a non-boolean real number is finite and positive."""
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _check_solver_controls(controls: NumericalControls) -> None:
    """Refuse an invalid FEM order, linear tolerance, iteration limit, solver, preconditioner or device."""
    order = controls.order
    if not isinstance(order, int) or isinstance(order, bool) or not 1 <= order <= 4:
        raise ValueError(
            "order must be an integer from 1 through 4: call set_numerical(order=1..4)."
        )
    if not _positive_finite(controls.tolerance):
        raise ValueError(
            "tolerance must be finite and positive: call set_numerical(tolerance=...)."
        )
    if (
        not isinstance(controls.max_iterations, int)
        or isinstance(controls.max_iterations, bool)
        or controls.max_iterations <= 0
    ):
        raise ValueError(
            "max_iterations must be a positive integer: call set_numerical(max_iterations=...)."
        )
    _check_choice("ksp_type", controls.ksp_type, _KSP_TYPES, optional=True)
    _check_choice("solver_type", controls.solver_type, _SOLVER_TYPES)
    _check_choice("preconditioner", controls.preconditioner, _PRECONDITIONERS)
    _check_choice("device", controls.device, {"CPU", "GPU"})
    if controls.solver_type != "Default" and controls.preconditioner != "Default":
        raise ValueError(
            "non-default preconditioner requires Default solver: call set_numerical() with "
            "solver_type='Default' or preconditioner='Default'."
        )


def _check_choice(
    name: str, value: str | None, choices: set[str], *, optional: bool = False
) -> None:
    """Refuse a control outside its fixed choices."""
    if optional and value is None:
        return
    if value not in choices:
        raise ValueError(
            f"unsupported {name}: call set_numerical({name}=...) with one of {sorted(choices)}."
        )


def _check_amr_controls(controls: NumericalControls) -> None:
    """Refuse an invalid AMR pass limit, conformity, tolerance, NC level, update fraction or output switch."""
    max_passes = controls.amr_max_passes
    if max_passes is not None:
        if not isinstance(max_passes, int) or isinstance(max_passes, bool):
            raise TypeError(
                "amr_max_passes must be an integer: call set_numerical(amr_max_passes=...)."
            )
        if max_passes < 0:
            raise ValueError(
                "amr_max_passes must be >= 0: call set_numerical(amr_max_passes=...)."
            )
    if not isinstance(controls.amr_nonconformal, bool):
        raise TypeError(
            "amr_nonconformal must be a bool: call set_numerical(amr_nonconformal=True or False)."
        )
    if controls.amr_tolerance is not None and not _positive_finite(
        controls.amr_tolerance
    ):
        raise ValueError(
            "amr_tolerance must be finite and positive: call set_numerical(amr_tolerance=...)."
        )
    _check_nc_levels(controls.amr_nc_levels)
    _check_update_fraction(controls.amr_update_fraction)
    for name in (
        "save_adapt_iterations",
        "save_adapt_mesh",
        "estimator_mg",
        "output_paraview",
        "output_grid_function",
    ):
        value = getattr(controls, name)
        if value is not None and not isinstance(value, bool):
            raise TypeError(
                f"{name} must be bool or None: call set_numerical({name}=True, False or None)."
            )


def _check_nc_levels(value: int | None) -> None:
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(
            "amr_nc_levels must be an integer or None: call set_numerical(amr_nc_levels=...)."
        )
    if value < 0:
        raise ValueError(
            "amr_nc_levels must be >= 0; zero selects no native limit: call set_numerical(amr_nc_levels=...)."
        )


def _check_update_fraction(value: float | None) -> None:
    if value is None:
        return
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(
            "amr_update_fraction must be a real number: call set_numerical(amr_update_fraction=...)."
        )
    fraction = float(value)
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError(
            "amr_update_fraction must be finite and strictly between 0 and 1: call "
            "set_numerical(amr_update_fraction=...)."
        )


def linear_solver(
    numerical: NumericalControls, *, problem: str | None = None
) -> dict[str, Any]:
    """Lower problem-specific defaults while retaining explicit solver choices.

    Electrostatic uses conjugate gradients (CG) with BoomerAMG by default.
    Explicit solver/preconditioner paths and other problems retain GMRES.
    Explicit ``ksp_type="Default"`` delegates Krylov selection to Palace.
    """
    use_electrostatic_defaults = (
        problem == "Electrostatic"
        and numerical.solver_type == "Default"
        and numerical.preconditioner == "Default"
    )
    default_ksp = "CG" if use_electrostatic_defaults else "GMRES"
    linear: dict[str, Any] = {
        "Type": numerical.solver_type,
        "KSPType": default_ksp if numerical.ksp_type is None else numerical.ksp_type,
        "Tol": numerical.tolerance,
        "MaxIts": numerical.max_iterations,
    }
    if numerical.solver_type == "MUMPS":
        linear.update(
            {
                "MaxIts": 1,
                "MGMaxLevels": 1,
                "PCMatReal": False,
                "ComplexCoarseSolve": True,
            }
        )
    if numerical.solver_type == "Default" and numerical.preconditioner != "Default":
        linear["Type"] = numerical.preconditioner
    if use_electrostatic_defaults:
        linear["Type"] = "BoomerAMG"
    if numerical.estimator_mg is not None:
        linear["EstimatorMG"] = numerical.estimator_mg
    return linear


def build_refinement(numerical: NumericalControls) -> dict[str, Any]:
    """Translate AMR controls into Palace's ``Model.Refinement`` block.

    Raises:
        ValueError: ``amr_max_passes`` was never chosen (``NumericalControls.check_ready_for`` says so first).
    """
    if numerical.amr_max_passes is None:
        raise ValueError(
            "amr_max_passes must be explicitly set before config generation: call "
            "set_numerical(amr_max_passes=...) before write_config()."
        )
    refinement: dict[str, Any] = {
        "UniformLevels": 0,
        "MaxIts": numerical.amr_max_passes,
        "Nonconformal": numerical.amr_nonconformal,
    }
    if numerical.amr_nc_levels is not None:
        refinement["MaxNCLevels"] = numerical.amr_nc_levels
    if numerical.amr_tolerance is not None:
        refinement["Tol"] = numerical.amr_tolerance
    if numerical.save_adapt_iterations is not None:
        refinement["SaveAdaptIterations"] = numerical.save_adapt_iterations
    if numerical.save_adapt_mesh is not None:
        refinement["SaveAdaptMesh"] = numerical.save_adapt_mesh
    if numerical.amr_update_fraction is not None:
        refinement["UpdateFraction"] = numerical.amr_update_fraction
    return refinement


def build_output_formats(numerical: NumericalControls) -> dict[str, Any]:
    """Translate optional field-output switches into Palace's output block."""
    output_formats = {
        "Paraview": numerical.output_paraview,
        "GridFunction": numerical.output_grid_function,
    }
    outputs = {key: value for key, value in output_formats.items() if value is not None}
    if not outputs:
        return {}
    return {"OutputFormats": outputs}


__all__ = [
    "NumericalControls",
    "build_output_formats",
    "build_refinement",
    "linear_solver",
]
