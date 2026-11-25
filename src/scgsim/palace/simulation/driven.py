"""The Driven problem: the plan's lumped ports excited one at a time over a frequency sweep, and the Driven config.

Everything else (the plan with its vacuum, Surface EPR and lumped elements, route, mesh controls, ``mesh()``,
``simulation_drc``, the summaries and the handoff) is ``PalaceSimulation`` in ``core``; this file shows only what
the Driven problem adds.  Palace solves the field at each frequency with one port excited and the others
terminated in their impedance, and writes the S-matrix (``port-S``).  ``resolve_palace_result(...).driven_result()``
reads the returned run back (``scgsim.palace.results.driven``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from scgsim.contracts.palace_run_folder import (
    INDEX_MAP,
    MATERIAL_RESOLUTION,
    NUMERICAL_CONTROLS,
    PORT_INFORMATION,
)
from scgsim.geometry.source.plan import GeometryPlanSnapshot
from scgsim.refusal import Declared

from ..config.config import build_driven_config
from ..config.config_records import DrivenSweep, PortConfigBuildResult
from ..config.convergence_plan import PlanContext
from ..config.files import MetadataFile
from ..config.index import IndexEntry
from ..config.numerical import ProblemDefaults
from ..source.lumped_elements import (
    driven_ports,
    driven_realization,
    driven_report,
    summarized_elements,
)
from ..source.ports import RequestedPort
from .core import PalaceSimulation, index_map


@dataclass
class DrivenSim(PalaceSimulation):
    """Prepare an SGB-authored Driven (S-parameter) mesh, config and manual handoff.

    The ports are the plan's: each ``LumpedPort`` is a Palace ``LumpedPort`` of its impedance, excited by
    itself, so port *i* is row and column *i* of the S-matrix, in the order the plan declares them; the plan's
    other lumped elements are written with their R, L and C, not excited
    (``scgsim.palace.source.lumped_elements``).  The calls run in this order; each later call needs the
    earlier ones::

        plan.set_surface_epr(specs)  # optional, on the plan: without it, no Surface EPR
        plan.set_lumped_elements([
            LumpedPort("chip/o_lumped_left", reference_net="ground", impedance_ohm=50.0),
            LumpedPort("chip/o_lumped_right", reference_net="ground", impedance_ohm=50.0),
        ])
        sim = DrivenSim()
        sim.set_plan(plan.prepare())
        sim.set_output_dir("runs/through")
        sim.set_driven(start=2e9, stop=10e9, points=81, adaptive_tolerance=1e-3)
        sim.mesh()
        sim.write_config()
        sim.prepare_handoff(
            profile="direct-local", executable="palace", resources={"processes": 8}
        )
    """

    sweep: DrivenSweep | None = None

    problem = "Driven"
    # Second-order elements and no AMR by default (both accepted on issue 187): the run solves on the mesh mesh()
    # writes, where first-order elements make the phase of a through line's S21 about 7 % too large and
    # second-order ones about 1 %.  Enabled AMR refines nonconformally, as Eigenmode does, with the tolerance
    # left for the user to choose.
    numerical_defaults: ClassVar[ProblemDefaults] = ProblemDefaults(
        order=2, amr_max_passes=0, amr_nonconformal=True, amr_tolerance=None
    )

    def set_driven(
        self,
        *,
        start: float,
        stop: float,
        points: int,
        adaptive_tolerance: float | None = None,
        adaptive_max_samples: int | None = None,
    ) -> None:
        """Set the frequency sweep: ``points`` frequencies from ``start`` to ``stop`` (Hz), both ends included.

        Without ``adaptive_tolerance`` Palace solves the field at every frequency, once per port.  With it,
        Palace's adaptive fast sweep solves a few frequencies per port and evaluates a reduced model built
        from them at every frequency, adding solves until the model agrees with them to that relative error;
        ``adaptive_max_samples`` caps those solves per port (Palace's own cap when omitted).  Whether the
        adaptive sweep converged is written only to Palace's log.  The values are checked (``DrivenSweep``)
        before any is stored.

        Raises:
            TypeError: a value is not a number, or ``points`` or ``adaptive_max_samples`` is not an integer.
            ValueError: ``start`` is not above 0, ``stop`` is not above ``start``, ``points`` is below 2,
                ``adaptive_tolerance`` is not above 0, or ``adaptive_max_samples`` is below 1 or is given
                without ``adaptive_tolerance``.
        """
        self.sweep = DrivenSweep(
            start_hz=start,
            stop_hz=stop,
            points=points,
            adaptive_tolerance=adaptive_tolerance,
            adaptive_max_samples=adaptive_max_samples,
        )
        self._fall_back("set_driven")

    def set_numerical(  # noqa: PLR0913 -- the public keyword controls of set_numerical
        self,
        *,
        order: int = 2,
        tolerance: float = 1e-6,
        max_iterations: int = 400,
        solver_type: str = "Default",
        preconditioner: str = "Default",
        device: str = "CPU",
        ksp_type: str | None = None,
        amr_max_passes: int | None = None,
        amr_nonconformal: bool | None = None,
        amr_nc_levels: int | None = None,
        amr_tolerance: float | None = None,
        amr_update_fraction: float | None = None,
        save_adapt_iterations: bool | None = None,
        save_adapt_mesh: bool | None = None,
        estimator_mg: bool | None = None,
        output_paraview: bool | None = None,
        output_grid_function: bool | None = None,
    ) -> None:
        """Set solver, FEM, AMR, and output controls for the next config write; stored as ``NumericalControls``.

        ``order`` is the FEM order of the Nédélec space; a ``DrivenSim`` solves with second-order elements,
        because at first order the phase of the S-parameters comes out too coarse on the mesh ``mesh()``
        writes.  ``amr_max_passes`` is Palace ``MaxIts``; a ``DrivenSim`` starts at 0, no AMR.  An AMR pass solves the whole sweep again, every port at every
        frequency, so refine on a short sweep.  Enabled AMR needs ``amr_tolerance`` and
        ``amr_update_fraction`` stated.  As in ``EigenmodeSim.set_numerical``, an omitted AMR, Krylov or
        nonconformal-level control keeps its current value; explicit ``ksp_type="Default"`` uses Palace's
        default.

        Raises:
            ValueError: a control is out of its range or choices.
            TypeError: a control has the wrong type.
        """
        self._set_numerical(
            self.numerical.with_given(
                order=order,
                tolerance=tolerance,
                max_iterations=max_iterations,
                solver_type=solver_type,
                preconditioner=preconditioner,
                device=device,
                ksp_type=ksp_type,
                amr_max_passes=amr_max_passes,
                amr_nonconformal=amr_nonconformal,
                amr_nc_levels=amr_nc_levels,
                amr_tolerance=amr_tolerance,
                amr_update_fraction=amr_update_fraction,
                save_adapt_iterations=save_adapt_iterations,
                save_adapt_mesh=save_adapt_mesh,
                estimator_mg=estimator_mg,
                output_paraview=output_paraview,
                output_grid_function=output_grid_function,
            )
        )

    def write_config(self) -> Path:
        """Write ``config.json`` and its metadata for the current mesh, the plan's ports and the sweep.

        Returns:
            The path of the written ``config.json``.

        Raises:
            ValueError: ``mesh()`` has not built the current mesh, ``set_driven`` has not set the sweep, AMR is
                left open, or the config refuses the bindings.
            FileNotFoundError: a file of the current mesh is gone from the run folder.
        """
        common = self._start_config()
        if self.sweep is None:
            raise ValueError(
                Declared(
                    "the simulation",
                    "the call order: call set_driven(start=..., stop=..., points=...) before write_config()",
                ).refusal("has no frequency sweep to write a config for")
            )
        result = build_driven_config(
            common, ports=self._resolved_ports, sweep=self.sweep
        )
        return self._finish_config(
            result.config, self._metadata(result), result.index_entries
        )

    def _metadata(self, result: PortConfigBuildResult) -> tuple[MetadataFile, ...]:
        """The metadata files written with ``config.json``, by run-folder path, in the order they are written."""
        return (
            MetadataFile(INDEX_MAP, index_map(result.index_entries)),
            MetadataFile(
                MATERIAL_RESOLUTION,
                {"schema_version": 1, "solution_volumes": result.material_resolution},
            ),
            MetadataFile(
                PORT_INFORMATION,
                {"schema_version": 1, "ports": result.port_information},
            ),
            MetadataFile(NUMERICAL_CONTROLS, self.numerical.to_payload()),
        )

    # ---- The core's questions ------------------------------------------------------------------------

    def _plan_context(self, entries: Sequence[IndexEntry]) -> PlanContext:
        """The config's entries; ``set_convergence`` refuses every Driven criterion, so nothing reads them yet."""
        return PlanContext(problem=self.problem, entries=entries)

    def _lumped_ports(self, plan: GeometryPlanSnapshot) -> list[RequestedPort]:
        """Every lumped element of ``plan``, in its number's order, each port excited: element n is ``P<n>``."""
        return driven_ports(plan)

    def _lumped_element_report(self, plan: GeometryPlanSnapshot) -> tuple[str, ...]:
        """Each port as an excitation, each other element as a not excited ``LumpedPort``, each bare sheet as not modeled."""
        return driven_report(plan)

    def _problem_summary(self, plan: GeometryPlanSnapshot) -> dict[str, Any]:
        """The lumped elements as Driven realizes them, the ports among them first, and the sweep."""
        return {
            "lumped_elements": summarized_elements(plan, driven_realization),
            "sweep": None if self.sweep is None else self.sweep.to_payload(),
        }


__all__ = ["DrivenSim"]
