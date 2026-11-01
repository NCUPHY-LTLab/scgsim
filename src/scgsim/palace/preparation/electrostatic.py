"""Mutable Electrostatic preparation state and its ordered failure lifecycle.

Owns: the Electrostatic-only inputs (terminals, grounds, boundary policies), the
mesh entry point and ``config.json`` writing.  Setters shared with Eigenmode live in
``_sim_common.py``.  This class prepares geometry and manual handoff files; it never
runs Palace.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

from scgsim.geometry.models.input import VacuumRegionSpec
from scgsim.geometry.models.physical import ApproximationProfile
from scgsim.geometry.source.plan import GeometryPlanSnapshot

from ..execution.handoff import HandoffPlan
from ..mesh.build import MeshBuildResult
from ._sim_common import PreparationControls
from .config import (
    build_electrostatic_config,
)
from .electrostatic_boundaries import TerminalBinding
from .numerical import configure_numerical_controls
from .transforms import (
    RouteAThinFilm,
    validate_non_negative_int,
    validate_nonempty_string,
)
from .workflow import persist_problem_files, prepare_mesh_input


@dataclass
class ElectrostaticSim(PreparationControls):
    """Prepare explicit SGB geometry and manual Palace electrostatic handoff files."""

    _PROBLEM: ClassVar[Literal["Electrostatic", "Eigenmode"]] = "Electrostatic"

    component: Any | None = None
    stack: Mapping[str, Any] | None = None
    output_dir: Path | None = None
    airbox: dict[str, float] = field(default_factory=dict)
    route: Literal["A", "B"] = "B"
    route_a_thin_film: RouteAThinFilm | None = None
    approximation_profile: ApproximationProfile | None = None
    terminals: list[TerminalBinding] = field(default_factory=list)
    ground_net_ids: list[str] = field(default_factory=list)
    surface_epr_specs: dict[str, dict[str, Any]] | None = None
    save_fields: int = 0
    unassigned_conductor_policy: Literal["ground", "error"] = "ground"
    exterior_boundary_policy: Literal["none", "ground"] = "none"
    numerical: dict[str, Any] = field(default_factory=configure_numerical_controls)
    _materials: dict[str, Mapping[str, Any]] | None = field(default=None, init=False)
    vacuum_region: VacuumRegionSpec | None = None
    indium_ground_bumps: dict[str, Any] | None = None
    _mesh_result: MeshBuildResult | None = field(default=None, init=False)
    config_path: Path | None = field(default=None, init=False)
    handoff_plan: HandoffPlan | None = field(default=None, init=False)
    _plan_snapshot: GeometryPlanSnapshot | None = field(
        default=None, init=False, repr=False
    )

    def _invalidate_mesh(self) -> None:
        self._mesh_result = None
        self.config_path = None
        self.handoff_plan = None

    def set_plan(self, snapshot: GeometryPlanSnapshot) -> None:
        """Consume one detached notebook geometry and stack source."""
        if not isinstance(snapshot, GeometryPlanSnapshot):
            raise TypeError("set_plan requires a GeometryPlanSnapshot")
        if "VACUUM_REGION" in snapshot.stack["solution_regions"] and (
            self.vacuum_region is not None or self.airbox
        ):
            raise ValueError("GeometryPlan owns the outer vacuum envelope")
        trial = ElectrostaticSim()
        trial.set_stack(snapshot.stack)
        self.component = None
        self.stack = trial.stack
        self._materials = trial._materials
        self._plan_snapshot = snapshot
        self._invalidate_mesh()

    def add_terminal(self, name: str, *, net_id: str) -> None:
        """Bind a named Palace Terminal to one conductor net; a repeated name replaces it."""
        binding = TerminalBinding(
            name=validate_nonempty_string(name, "terminal name"),
            net_id=validate_nonempty_string(net_id, "net_id"),
        )
        self.terminals = [item for item in self.terminals if item.name != binding.name]
        self.terminals.append(binding)
        self._invalidate_config()

    def add_ground(self, *, net_id: str) -> None:
        """Bind exact SGB conductor-net identity to physical Palace Ground."""
        ground_net_id = validate_nonempty_string(net_id, "net_id")
        if ground_net_id not in self.ground_net_ids:
            self.ground_net_ids.append(ground_net_id)
        self._invalidate_config()

    def set_electrostatic(
        self,
        *,
        save_fields: int = 0,
        unassigned_conductor_policy: Literal["ground", "error"] = "ground",
        exterior_boundary_policy: Literal["none", "ground"] = "none",
    ) -> None:
        """Set the Electrostatic controls: saved fields and how unassigned conductors and the outer boundary are grounded."""
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
        self._invalidate_config()

    def mesh(self) -> Path:
        """Build the Palace mesh for the current inputs. Replaces any earlier mesh."""
        self._invalidate_mesh()
        self._check_mesh_inputs_set()
        if self.surface_epr_specs is None:
            raise ValueError("set_surface_epr() must run before mesh().")
        if self._plan_snapshot is not None:
            return self._mesh_from_plan(self._plan_snapshot)
        return self._mesh_from_component()

    def _mesh_from_plan(self, snapshot: GeometryPlanSnapshot) -> Path:
        self._check_plan_has_no_bump_fill()
        prepared = prepare_mesh_input(
            component=None,
            stack=snapshot.stack,
            route=self.route,
            route_a_thin_film=self.route_a_thin_film,
            approximation_profile=self.approximation_profile,
            airbox=self.airbox,
            vacuum_region=self._plan_vacuum_region(),
            indium_ground_bumps=None,
            build_input=snapshot.geometry_input,
        )
        return self._build_mesh(
            prepared, component=None, source_gds_bytes=snapshot.gds_bytes
        )

    def _mesh_from_component(self) -> Path:
        prepared = prepare_mesh_input(
            component=self.component,
            stack=self.stack,
            route=self.route,
            route_a_thin_film=self.route_a_thin_film,
            approximation_profile=self.approximation_profile,
            airbox=self.airbox,
            vacuum_region=self.vacuum_region,
            indium_ground_bumps=self.indium_ground_bumps,
        )
        return self._build_mesh(
            prepared,
            component=prepared.component,
            indium_ground_bump_fill=prepared.indium_ground_bump_fill,
        )

    def write_config(self) -> Path:
        """Write config.json and the index map for the current mesh."""
        self._invalidate_config()
        if (
            self._mesh_result is None
            or self._materials is None
            or self.surface_epr_specs is None
        ):
            raise ValueError(
                "set_stack(), set_surface_epr(), and mesh() must run before write_config()."
            )
        if not self._mesh_result.mesh_path.is_file():
            raise FileNotFoundError(
                "current palace.msh is missing; mesh() must be rerun."
            )
        result = build_electrostatic_config(
            groups=self._mesh_result.groups,
            terminals=self.terminals,
            ground_net_ids=self.ground_net_ids,
            materials=self._materials,
            save_fields=self.save_fields,
            unassigned_conductor_policy=self.unassigned_conductor_policy,
            exterior_boundary_policy=self.exterior_boundary_policy,
            mesh_path=self._mesh_result.mesh_path,
            surface_epr_specs=self.surface_epr_specs,
            numerical=self.numerical,
        )
        metadata_dir = self._mesh_result.output_dir / "metadata"
        index_payload = {
            "schema_version": 2,
            "entries": [
                *result.terminal_index_map,
                *result.domain_energy_index_map,
                *result.surface_epr_index_map,
            ],
            "ground_boundary": result.ground_boundary_resolution,
        }
        material_payload = {
            "schema_version": 1,
            "solution_volumes": result.material_resolution,
        }
        config_path = self._mesh_result.output_dir / "config.json"
        self.config_path = persist_problem_files(
            metadata_files=(
                (metadata_dir / "palace_index_map.json", index_payload),
                (metadata_dir / "palace_material_resolution.json", material_payload),
                (metadata_dir / "palace_numerical_controls.json", self.numerical),
            ),
            config_path=config_path,
            config=result.config,
        )
        return config_path


__all__ = ["ElectrostaticSim"]
