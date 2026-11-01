"""Validate the numerical controls of a Palace run.

``configure_numerical_controls`` turns the notebook's keyword arguments into the one
dict that ``config.py`` writes into ``config.json``: FEM (finite element) order,
linear solver, mesh sizes, adaptive mesh refinement (AMR) and output formats. It
raises on the first invalid value and fills in nothing the caller did not choose
except the defaults in its signature.

Not here: writing the config (``config.py``) and the Gmsh controls themselves
(``palace/mesh/controls.py``).
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any


def configure_numerical_controls(  # noqa: PLR0913 - public keyword-only API, one argument per notebook control
    *,
    order: int = 1,
    tolerance: float = 1e-6,
    max_iterations: int = 400,
    solver_type: str = "Default",
    preconditioner: str = "Default",
    device: str = "CPU",
    refined_mesh_size: float = 5.0,
    max_mesh_size: float = 300.0,
    algorithm_3d: str = "Delaunay",
    threads: int = 1,
    surface_threads: int = 1,
    geometry_order: int = 1,
    high_order_optimize: bool = True,
    amr_max_passes: int = 0,
    amr_nonconformal: bool = False,
    amr_tolerance: float = 1e-2,
    amr_update_fraction: float | None = None,
    save_adapt_iterations: bool | None = None,
    save_adapt_mesh: bool | None = None,
    estimator_mg: bool | None = None,
    output_paraview: bool | None = None,
    output_grid_function: bool | None = None,
) -> dict[str, Any]:
    """Validate notebook numerical controls for Palace config synthesis.

    ``amr_max_passes`` becomes ``Model.Refinement.MaxIts``.
    ``amr_nonconformal`` becomes ``Model.Refinement.Nonconformal`` and defaults
    to ``False`` so generated configs do not inherit Palace's ``true`` default.
    """
    _check_solver_controls(
        order, tolerance, max_iterations, solver_type, preconditioner, device
    )
    if solver_type == "MUMPS":
        # MUMPS is a direct solver: one pass, no iteration.
        max_iterations = 1
    refined, maximum = _check_mesh_sizes(refined_mesh_size, max_mesh_size)
    _check_amr_controls(
        amr_max_passes, amr_nonconformal, amr_tolerance, amr_update_fraction
    )
    _check_optional_flags(
        save_adapt_iterations=save_adapt_iterations,
        save_adapt_mesh=save_adapt_mesh,
        estimator_mg=estimator_mg,
        output_paraview=output_paraview,
        output_grid_function=output_grid_function,
    )

    from ..mesh.controls import mesh_controls

    controls = {
        **mesh_controls(
            algorithm_3d=algorithm_3d,
            threads=threads,
            surface_threads=surface_threads,
            geometry_order=geometry_order,
            high_order_optimize=high_order_optimize,
        ),
        "order": int(order),
        "tolerance": float(tolerance),
        "max_iterations": int(max_iterations),
        "solver_type": str(solver_type),
        "preconditioner": str(preconditioner),
        "device": str(device),
        "refined_mesh_size": refined,
        "max_mesh_size": maximum,
        "amr_max_passes": int(amr_max_passes),
        "amr_nonconformal": bool(amr_nonconformal),
        "amr_tolerance": float(amr_tolerance),
        "amr_update_fraction": (
            float(amr_update_fraction) if amr_update_fraction is not None else None
        ),
        "save_adapt_iterations": save_adapt_iterations,
        "estimator_mg": estimator_mg,
        "output_paraview": output_paraview,
        "output_grid_function": output_grid_function,
    }
    if save_adapt_mesh is not None:
        controls["save_adapt_mesh"] = save_adapt_mesh
    return controls


def _check_solver_controls(  # noqa: PLR0913 - one argument per control being checked
    order: int,
    tolerance: float,
    max_iterations: int,
    solver_type: str,
    preconditioner: str,
    device: str,
) -> None:
    if not isinstance(order, int) or isinstance(order, bool) or not 1 <= order <= 4:
        raise ValueError("order must be an integer from 1 through 4.")
    if not _positive_finite(tolerance):
        raise ValueError("tolerance must be finite and positive.")
    if (
        not isinstance(max_iterations, int)
        or isinstance(max_iterations, bool)
        or max_iterations <= 0
    ):
        raise ValueError("max_iterations must be a positive integer.")
    if solver_type not in {"Default", "SuperLU", "STRUMPACK", "MUMPS"}:
        raise ValueError("unsupported solver_type.")
    if preconditioner not in {"Default", "AMS", "BoomerAMG"}:
        raise ValueError("unsupported preconditioner.")
    if device not in {"CPU", "GPU"}:
        raise ValueError("unsupported device.")
    if solver_type != "Default" and preconditioner != "Default":
        raise ValueError("non-default preconditioner requires Default solver.")


def _check_mesh_sizes(
    refined_mesh_size: float, max_mesh_size: float
) -> tuple[float, float]:
    if not _positive_finite(refined_mesh_size):
        raise ValueError("refined_mesh_size must be finite and > 0.")
    if not _positive_finite(max_mesh_size):
        raise ValueError("max_mesh_size must be finite and > 0.")
    refined = float(refined_mesh_size)
    maximum = float(max_mesh_size)
    if maximum < refined:
        raise ValueError("max_mesh_size must be >= refined_mesh_size.")
    return refined, maximum


def _check_amr_controls(
    amr_max_passes: int,
    amr_nonconformal: bool,
    amr_tolerance: float,
    amr_update_fraction: float | None,
) -> None:
    """AMR is adaptive mesh refinement."""
    if not isinstance(amr_max_passes, int) or isinstance(amr_max_passes, bool):
        raise TypeError("amr_max_passes must be an integer.")
    if amr_max_passes < 0:
        raise ValueError("amr_max_passes must be >= 0.")
    if not isinstance(amr_nonconformal, bool):
        raise TypeError("amr_nonconformal must be a bool.")
    if not _positive_finite(amr_tolerance):
        raise ValueError("amr_tolerance must be finite and positive.")
    if amr_update_fraction is not None:
        if not isinstance(amr_update_fraction, Real) or isinstance(
            amr_update_fraction, bool
        ):
            raise TypeError("amr_update_fraction must be a real number.")
        amr_update = float(amr_update_fraction)
        if not math.isfinite(amr_update) or not 0.0 < amr_update < 1.0:
            raise ValueError(
                "amr_update_fraction must be finite and strictly between 0 and 1."
            )


def _check_optional_flags(**flags: bool | None) -> None:
    """Each flag is a bool, or None to leave Palace's default."""
    for name, value in flags.items():
        if value is not None and not isinstance(value, bool):
            raise TypeError(f"{name} must be bool or None.")


def _positive_finite(value: Any) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )
