"""The Electrostatic problem: terminals, grounds, the boundary policies and the Electrostatic config.

Everything else (plan, vacuum, Surface EPR, mesh controls, ``mesh()``, ``simulation_drc``, the summaries and
the handoff) is ``PalaceSimulation`` in ``core``; this file shows only what the Electrostatic problem adds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

from scgsim.geometry.source.plan import GeometryPlanSnapshot

from ..preparation.config import (
    TerminalBinding,
    build_electrostatic_config,
    configure_numerical_controls,
)
from ..preparation.transforms import validate_non_negative_int, validate_nonempty_string
from ..results.index_contract import INDEX_MAP_SCHEMA_VERSION
from .core import PalaceSimulation


@dataclass
class ElectrostaticSim(PalaceSimulation):
    """Prepare explicit SGB geometry and manual Palace electrostatic handoff files.

    The calls run in this order; each later call needs the earlier ones::

        sim = ElectrostaticSim()
        sim.set_plan(plan.prepare())
        sim.set_output_dir("runs/pads")
        sim.set_surface_epr(representation="B", specs={})  # required before mesh()
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
    numerical: dict[str, Any] = field(default_factory=configure_numerical_controls)

    problem = "Electrostatic"
    # TODO(human): why Electrostatic refuses a mesh with no set_surface_epr() while Eigenmode accepts one
    # (Issue #199 keeps both rules as they are).
    mesh_requires_surface_epr: ClassVar[bool] = True

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

        Raises:
            ValueError: a policy is not one of its values, or an exterior ground is asked for while
                unassigned conductors are grounded.
        """
        if unassigned_conductor_policy not in {"ground", "error"}:
            raise ValueError("unassigned_conductor_policy must be 'ground' or 'error'.")
        if exterior_boundary_policy not in {"none", "ground"}:
            raise ValueError("exterior_boundary_policy must be 'none' or 'ground'.")
        if (
            exterior_boundary_policy == "ground"
            and unassigned_conductor_policy != "error"
        ):
            raise ValueError(
                "exterior ground requires unassigned_conductor_policy='error'."
            )
        self.save_fields = validate_non_negative_int(save_fields, "save_fields")
        self.unassigned_conductor_policy = unassigned_conductor_policy
        self.exterior_boundary_policy = exterior_boundary_policy
        self._fall_back("set_electrostatic")

    def set_numerical(  # noqa: PLR0913 - preserve the public keyword controls.
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
        """Set solver, FEM, AMR, and output controls for the next config write.

        ``amr_max_passes`` is Palace ``MaxIts``. ``amr_nonconformal`` is Palace
        ``Model.Refinement.Nonconformal`` and defaults to ``False``. Set
        ``True`` only to opt into nonconformal AMR.
        ``amr_nc_levels`` sets native ``MaxNCLevels``; zero means no limit.
        Omitted Krylov and nonconformal-level controls retain current choices.
        Explicit ``ksp_type="Default"`` uses Palace's problem-specific default.
        """
        current = self.numerical
        self.numerical = configure_numerical_controls(
            order=order,
            tolerance=tolerance,
            max_iterations=max_iterations,
            solver_type=solver_type,
            preconditioner=preconditioner,
            device=device,
            ksp_type=current.get("ksp_type") if ksp_type is None else ksp_type,
            amr_nc_levels=(
                current.get("amr_nc_levels") if amr_nc_levels is None else amr_nc_levels
            ),
            amr_max_passes=amr_max_passes,
            amr_nonconformal=amr_nonconformal,
            amr_tolerance=amr_tolerance,
            amr_update_fraction=amr_update_fraction,
            save_adapt_iterations=save_adapt_iterations,
            save_adapt_mesh=save_adapt_mesh,
            estimator_mg=estimator_mg,
            output_paraview=output_paraview,
            output_grid_function=output_grid_function,
        )
        self._fall_back("set_numerical")

    def write_config(self) -> Path:
        """Write ``config.json`` and its metadata for the current mesh, terminals, grounds and policies.

        Returns:
            The path of the written ``config.json``.

        Raises:
            ValueError: ``mesh()`` has not built the current mesh, or the config refuses the bindings.
            FileNotFoundError: a file of the current mesh is gone from the run folder.
        """
        mesh, materials = self._start_config()
        surface_epr_specs = self._require("write_config", self.surface_epr_specs)
        result = build_electrostatic_config(
            groups=mesh.groups,
            terminals=self.terminals,
            ground_net_ids=self.ground_net_ids,
            materials=materials,
            save_fields=self.save_fields,
            unassigned_conductor_policy=self.unassigned_conductor_policy,
            exterior_boundary_policy=self.exterior_boundary_policy,
            mesh_path=mesh.mesh_path,
            surface_epr_specs=surface_epr_specs,
            numerical=self.numerical,
        )
        index_map = {
            "schema_version": INDEX_MAP_SCHEMA_VERSION,
            "entries": [
                *result.terminal_index_map,
                *result.domain_energy_index_map,
                *result.surface_epr_index_map,
            ],
            "ground_boundary": result.ground_boundary_resolution,
        }
        return self._finish_config(
            mesh,
            result.config,
            metadata=(
                ("palace_index_map.json", index_map),
                (
                    "palace_material_resolution.json",
                    {
                        "schema_version": 1,
                        "solution_volumes": result.material_resolution,
                    },
                ),
                ("palace_numerical_controls.json", self.numerical),
            ),
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
