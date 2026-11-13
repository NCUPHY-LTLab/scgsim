"""The Electrostatic problem: terminals, grounds, the boundary policies and the Electrostatic config.

Everything else (plan, vacuum, route, Surface EPR, mesh controls, ``mesh()``, ``simulation_drc``, the summaries and
the handoff) is ``PalaceSimulation`` in ``core``; this file shows only what the Electrostatic problem adds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

from scgsim.contracts.palace_run_folder import (
    INDEX_MAP,
    MATERIAL_RESOLUTION,
    NUMERICAL_CONTROLS,
)
from scgsim.geometry.source.plan import GeometryPlanSnapshot

from ..config.config import build_electrostatic_config
from ..config.config_records import (
    ElectrostaticConfigBuildResult,
    ElectrostaticSettings,
    TerminalBinding,
)
from ..config.numerical import NumericalControls, ProblemDefaults
from ..source.transforms import validate_nonempty_string
from .core import PalaceSimulation, index_map


@dataclass
class ElectrostaticSim(PalaceSimulation):
    """Prepare explicit SGB geometry and manual Palace electrostatic handoff files.

    The calls run in this order; each later call needs the earlier ones::

        sim = ElectrostaticSim()
        sim.set_plan(plan.prepare())
        sim.set_output_dir("runs/pads")
        sim.set_route("B")  # optional: Route B is the default
        sim.set_surface_epr(specs)  # optional: without it, no Surface EPR
        sim.add_terminal("pad", net_id="pad")
        sim.add_ground(net_id="ground")
        sim.set_electrostatic(save_fields=1)
        sim.mesh()
        sim.write_config()
        sim.prepare_handoff(
            profile="direct-local", executable="palace", resources={"processes": 1}
        )
    """

    terminals: list[TerminalBinding] = field(default_factory=list)
    ground_net_ids: list[str] = field(default_factory=list)
    save_fields: int = 0
    unassigned_conductor_policy: Literal["ground", "error"] = "ground"
    exterior_boundary_policy: Literal["none", "ground"] = "none"

    problem = "Electrostatic"
    # AMR off by default, conformal when enabled, with a tolerance of 0.01.
    numerical_defaults: ClassVar[ProblemDefaults] = ProblemDefaults(
        amr_max_passes=0, amr_nonconformal=False, amr_tolerance=1e-2
    )

    def add_terminal(self, name: str, *, net_id: str) -> None:
        """Bind one exact conductor net to a named Electrostatic terminal; a repeated name replaces it.

        Raises:
            ValueError: ``name`` or ``net_id`` is empty.
        """
        binding = TerminalBinding(
            name=validate_nonempty_string(name, "terminal name"),
            net_id=validate_nonempty_string(net_id, "net_id"),
        )
        self.terminals = [item for item in self.terminals if item.name != binding.name]
        self.terminals.append(binding)
        self._fall_back("add_terminal")

    def add_ground(self, *, net_id: str) -> None:
        """Bind exact SGB conductor-net identity to physical Palace Ground.

        Raises:
            ValueError: ``net_id`` is empty.
        """
        ground_net_id = validate_nonempty_string(net_id, "net_id")
        if ground_net_id not in self.ground_net_ids:
            self.ground_net_ids.append(ground_net_id)
        self._fall_back("add_ground")

    def set_electrostatic(
        self,
        *,
        save_fields: int = 0,
        unassigned_conductor_policy: Literal["ground", "error"] = "ground",
        exterior_boundary_policy: Literal["none", "ground"] = "none",
    ) -> None:
        """Set Electrostatic boundary policies and saved-field controls.

        All values are checked (``ElectrostaticSettings``) before any is stored.

        Raises:
            ValueError: a policy is not one of its values, an exterior ground is asked for while
                unassigned conductors are grounded, or ``save_fields`` is negative.
            TypeError: ``save_fields`` is not an integer.
        """
        settings = ElectrostaticSettings(
            save_fields=save_fields,
            unassigned_conductor_policy=unassigned_conductor_policy,
            exterior_boundary_policy=exterior_boundary_policy,
        )
        self.save_fields = settings.save_fields
        self.unassigned_conductor_policy = settings.unassigned_conductor_policy
        self.exterior_boundary_policy = settings.exterior_boundary_policy
        self._fall_back("set_electrostatic")

    def set_numerical(  # noqa: PLR0913 -- the public keyword controls of set_numerical
        self,
        *,
        order: int = 1,
        tolerance: float = 1e-6,
        max_iterations: int = 400,
        solver_type: str = "Default",
        preconditioner: str = "Default",
        device: str = "CPU",
        ksp_type: str | None = None,
        amr_max_passes: int = 0,
        amr_nonconformal: bool = False,
        amr_nc_levels: int | None = None,
        amr_tolerance: float = 1e-2,
        amr_update_fraction: float | None = None,
        save_adapt_iterations: bool | None = None,
        save_adapt_mesh: bool | None = None,
        estimator_mg: bool | None = None,
        output_paraview: bool | None = None,
        output_grid_function: bool | None = None,
    ) -> None:
        """Set solver, FEM, AMR, and output controls for the next config write; stored as ``NumericalControls``.

        ``amr_max_passes`` is Palace ``MaxIts``. ``amr_nonconformal`` is Palace
        ``Model.Refinement.Nonconformal`` and defaults to ``False``. Set
        ``True`` only to opt into nonconformal AMR.
        ``amr_nc_levels`` sets native ``MaxNCLevels``; zero means no limit.
        Omitted Krylov and nonconformal-level controls retain current choices.
        Explicit ``ksp_type="Default"`` uses Palace's problem-specific default.

        How it differs from ``EigenmodeSim.set_numerical``: here an omitted ``amr_max_passes``,
        ``amr_nonconformal``, ``amr_tolerance`` or ``amr_update_fraction`` takes its default (0, False, 0.01,
        None); there each keeps its current value.

        Raises:
            ValueError: a control is out of its range or choices.
            TypeError: a control has the wrong type.
        """
        current = self.numerical
        self._set_numerical(
            NumericalControls(
                order=order,
                tolerance=tolerance,
                max_iterations=max_iterations,
                solver_type=solver_type,
                preconditioner=preconditioner,
                device=device,
                ksp_type=current.ksp_type if ksp_type is None else ksp_type,
                amr_max_passes=amr_max_passes,
                amr_nonconformal=amr_nonconformal,
                amr_nc_levels=current.amr_nc_levels
                if amr_nc_levels is None
                else amr_nc_levels,
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
        """Write ``config.json`` and its metadata for the current mesh, terminals, grounds and policies.

        Returns:
            The path of the written ``config.json``.

        Raises:
            ValueError: ``mesh()`` has not built the current mesh, AMR is left open, or the config refuses
                the bindings.
            FileNotFoundError: a file of the current mesh is gone from the run folder.
        """
        mesh, common = self._start_config()
        result = build_electrostatic_config(
            common,
            terminals=self.terminals,
            ground_net_ids=self.ground_net_ids,
            electrostatic=self._electrostatic_settings(),
        )
        return self._finish_config(mesh, result.config, self._metadata(result))

    def _electrostatic_settings(self) -> ElectrostaticSettings:
        """The saved fields and policies as the config reads them, checked again in case an attribute was assigned."""
        return ElectrostaticSettings(
            save_fields=self.save_fields,
            unassigned_conductor_policy=self.unassigned_conductor_policy,
            exterior_boundary_policy=self.exterior_boundary_policy,
        )

    def _metadata(
        self, result: ElectrostaticConfigBuildResult
    ) -> tuple[tuple[str, dict[str, Any]], ...]:
        """The metadata files written with ``config.json``, by run-folder path, in the order they are written."""
        return (
            (
                INDEX_MAP,
                index_map(
                    result.index_entries,
                    ground_boundary=result.ground_boundary_resolution,
                ),
            ),
            (
                MATERIAL_RESOLUTION,
                {"schema_version": 1, "solution_volumes": result.material_resolution},
            ),
            (NUMERICAL_CONTROLS, self.numerical.to_payload()),
        )

    # ---- The core's questions ------------------------------------------------------------------------

    def _lumped_ports(self) -> None:
        """None: an Electrostatic model cuts no lumped ports, and the plan's sheets stay as authored."""
        return None

    def _problem_summary(self) -> dict[str, Any]:
        """The terminals, grounds, saved fields and boundary policies."""
        return {
            "terminals": [
                {"name": item.name, "net_id": item.net_id} for item in self.terminals
            ],
            "ground_net_ids": list(self.ground_net_ids),
            "save_fields": self.save_fields,
            "unassigned_conductor_policy": self.unassigned_conductor_policy,
            "exterior_boundary_policy": self.exterior_boundary_policy,
        }

    def _add_drc_inputs(self, snapshot: GeometryPlanSnapshot) -> None:
        """Nothing: terminals and grounds are bound to the mesh by ``write_config()``, after meshing."""


__all__ = ["ElectrostaticSim"]
