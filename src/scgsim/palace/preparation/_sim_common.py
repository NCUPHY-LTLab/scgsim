"""Setters and mesh scaffolding shared by the Electrostatic and Eigenmode classes.

Owns: the inputs every Palace preparation class takes the same way (geometry, stack,
output directory, airbox or vacuum region, indium bumps, Surface EPR, mesh and
numerical controls) and the mesh build that follows from them.  The two classes
differ in what they add: terminals and grounds for Electrostatic, ports and the
mode request for Eigenmode.  Those stay in ``electrostatic.py`` and ``eigenmode.py``.

Every setter ends by clearing what depends on it: ``_invalidate_mesh`` when the
mesh would change, ``_invalidate_config`` when only ``config.json`` would.  The
two invalidators are defined by the concrete class because Eigenmode also clears
its resolved ports.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Literal

from scgsim.geometry.models.input import VacuumRegionSpec
from scgsim.geometry.models.physical import ApproximationProfile
from scgsim.geometry.source.plan import GeometryPlanSnapshot
from scgsim.palace.mesh.quality import MeshQualityReport, check_mesh_quality

from ..execution.handoff import HandoffPlan, prepare_handoff
from ..mesh.build import MeshBuildResult, build_route_mesh
from ..mesh.controls import MESH_CONTROL_KEYS
from .config import _MODEL_L0_M
from .inputs import (
    _load_stack,
    _non_negative_number,
    _validate_stack_material_kinds,
)
from .numerical import configure_numerical_controls
from .surface_epr import normalize_surface_epr_specs
from .transforms import (
    RouteAThinFilm,
    normalize_route_a_thin_film,
    validate_positive_number,
)
from .workflow import PreparedMeshInput

_GEOMETRY_PLAN_OWNS_SOURCE = (
    "GeometryPlan owns the paired source; use set_plan(new_snapshot) or a new Sim"
)


class PreparationControls:
    """Mixin: the setters and mesh steps both Palace preparation classes share."""

    _PROBLEM: ClassVar[Literal["Electrostatic", "Eigenmode"]]

    component: Any | None
    stack: Mapping[str, Any] | None
    output_dir: Path | None
    airbox: dict[str, float]
    route: Literal["A", "B"]
    route_a_thin_film: RouteAThinFilm | None
    approximation_profile: ApproximationProfile | None
    surface_epr_specs: dict[str, dict[str, Any]] | None
    numerical: dict[str, Any]
    vacuum_region: VacuumRegionSpec | None
    indium_ground_bumps: dict[str, Any] | None
    config_path: Path | None
    handoff_plan: HandoffPlan | None
    _materials: dict[str, Mapping[str, Any]] | None
    _mesh_result: MeshBuildResult | None
    _plan_snapshot: GeometryPlanSnapshot | None

    def _invalidate_mesh(self) -> None:
        raise NotImplementedError

    def _invalidate_config(self) -> None:
        self.config_path = None
        self.handoff_plan = None

    # ------------------------------------------------------------- source inputs

    def set_geometry(self, component: Any) -> None:
        """Use a component (it must provide write_gds) as the source geometry."""
        if self._plan_snapshot is not None:
            raise ValueError(_GEOMETRY_PLAN_OWNS_SOURCE)
        if component is None or not callable(getattr(component, "write_gds", None)):
            raise TypeError("component must provide write_gds(path).")
        self.component = component
        self._plan_snapshot = None
        self._invalidate_mesh()

    def set_stack(self, stack: Mapping[str, Any] | str | Path) -> None:
        """Load the layer stack; it must define explicit materials. Clears the mesh."""
        if self._plan_snapshot is not None:
            raise ValueError(_GEOMETRY_PLAN_OWNS_SOURCE)
        payload = _load_stack(stack)
        materials = payload.get("materials")
        if not isinstance(materials, Mapping) or not materials:
            raise ValueError(
                "stack must define a non-empty explicit materials mapping."
            )
        resolved = {
            str(key): value
            for key, value in materials.items()
            if isinstance(value, Mapping)
        }
        if len(resolved) != len(materials):
            raise TypeError("every explicit stack material must be a mapping.")
        _validate_stack_material_kinds(payload, resolved)
        self.stack = payload
        self._materials = resolved
        self._plan_snapshot = None
        self._invalidate_mesh()

    def set_output_dir(self, path: str | Path) -> None:
        """Choose the run directory the mesh, config and handoff are written to."""
        if not isinstance(path, (str, Path)):
            raise TypeError("output path must be a path string or Path.")
        self.output_dir = Path(path).expanduser().resolve()
        self._invalidate_mesh()

    def set_airbox(
        self,
        *,
        margin_x: float,
        margin_y: float,
        z_above: float | None = None,
        z_below: float | None = None,
    ) -> None:
        """Pad the device with vacuum: equal in-plane margins, optional z extensions. Excludes set_vacuum_region()."""
        self._check_plan_leaves_vacuum_envelope_free()
        if self.vacuum_region is not None:
            raise ValueError(
                "set_airbox is mutually exclusive with set_vacuum_region()."
            )
        airbox = {
            "margin_x": validate_positive_number(margin_x, "margin_x"),
            "margin_y": validate_positive_number(margin_y, "margin_y"),
        }
        if z_above is not None:
            airbox["z_above"] = validate_positive_number(z_above, "z_above")
        if z_below is not None:
            airbox["z_below"] = validate_positive_number(z_below, "z_below")
        self.airbox = airbox
        self._invalidate_mesh()

    def set_vacuum_region(
        self,
        padding: float | list[float] | tuple[float, ...] | Mapping[str, Any] = 0.0,
    ) -> None:
        """Set six-direction padding for a route-aware generated vacuum region."""
        self._check_plan_leaves_vacuum_envelope_free()
        if self.airbox:
            raise ValueError(
                "set_vacuum_region is mutually exclusive with set_airbox()."
            )
        self.vacuum_region = VacuumRegionSpec.from_padding(padding)
        self._invalidate_mesh()

    def _check_plan_leaves_vacuum_envelope_free(self) -> None:
        if self._plan_owns_vacuum_envelope():
            raise ValueError("GeometryPlan owns the outer vacuum envelope")

    def _plan_owns_vacuum_envelope(self) -> bool:
        return (
            self._plan_snapshot is not None
            and "VACUUM_REGION" in self._plan_snapshot.stack["solution_regions"]
        )

    def set_indium_ground_bumps(
        self, *, fill: bool, fill_pitch_um: float, fill_clearance_um: float
    ) -> None:
        """Request public-PDK authored-plus-ground-fill bumps for the next mesh."""
        if not isinstance(fill, bool):
            raise TypeError("fill must be a bool.")
        self.indium_ground_bumps = {
            "fill": fill,
            "fill_pitch_um": validate_positive_number(fill_pitch_um, "fill_pitch_um"),
            "fill_clearance_um": _non_negative_number(
                fill_clearance_um, "fill_clearance_um"
            ),
        }
        self._invalidate_mesh()

    def set_surface_epr(
        self,
        *,
        representation: str,
        specs: Mapping[str, Mapping[str, Any]],
        route_a_thin_film: RouteAThinFilm | None = None,
        approximation_profile: ApproximationProfile | None = None,
    ) -> None:
        """Configure Surface EPR and the required Route-A thin-film lowering."""
        route = (
            representation.strip().upper() if isinstance(representation, str) else ""
        )
        if route not in {"A", "B"}:
            raise ValueError("Surface EPR representation must be 'A' or 'B'.")
        if self._materials is None:
            raise ValueError("set_stack() must run before set_surface_epr().")
        normalized_thin_film = (
            normalize_route_a_thin_film(route, route_a_thin_film)
            if route_a_thin_film is not None
            else None
        )
        if approximation_profile is not None and not isinstance(
            approximation_profile, ApproximationProfile
        ):
            raise TypeError("approximation_profile must be ApproximationProfile")
        normalized_specs = normalize_surface_epr_specs(specs, materials=self._materials)
        self.route = route  # type: ignore[assignment]
        self.route_a_thin_film = normalized_thin_film
        if approximation_profile is not None:
            self.approximation_profile = approximation_profile
        self.surface_epr_specs = normalized_specs
        self._invalidate_mesh()

    # ------------------------------------------------- mesh and numerical controls

    def set_mesh(  # noqa: PLR0913 - public keyword-only API, one argument per mesh control
        self,
        *,
        refined_mesh_size: float | None = None,
        max_mesh_size: float | None = None,
        algorithm_3d: str | None = None,
        threads: int | None = None,
        surface_threads: int | None = None,
        geometry_order: int | None = None,
        high_order_optimize: bool | None = None,
    ) -> None:
        """Set geometry-mesh controls; every omitted value retains its setting.

        Geometry order is independent of ``set_numerical(order=...)``. Elevated
        geometry is optimized with Gmsh HighOrder once when enabled.
        """
        proposed = dict(self.numerical)
        proposed.update(
            {
                key: value
                for key, value in {
                    "refined_mesh_size": refined_mesh_size,
                    "max_mesh_size": max_mesh_size,
                    "algorithm_3d": algorithm_3d,
                    "threads": threads,
                    "surface_threads": surface_threads,
                    "geometry_order": geometry_order,
                    "high_order_optimize": high_order_optimize,
                }.items()
                if value is not None
            }
        )
        numerical = configure_numerical_controls(**proposed)
        if numerical != self.numerical:
            self.numerical = numerical
            self._invalidate_mesh()

    def set_numerical(  # noqa: PLR0913 - public keyword-only API, one argument per solver control
        self,
        *,
        order: int = 1,
        tolerance: float = 1e-6,
        max_iterations: int = 400,
        solver_type: str = "Default",
        preconditioner: str = "Default",
        device: str = "CPU",
        refined_mesh_size: float | None = None,
        max_mesh_size: float | None = None,
        amr_max_passes: int = 0,
        amr_nonconformal: bool = False,
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
        """
        if (refined_mesh_size is None) != (max_mesh_size is None):
            raise ValueError(
                "refined_mesh_size and max_mesh_size must be provided together."
            )
        previous_mesh_sizes = {
            "refined_mesh_size": self.numerical["refined_mesh_size"],
            "max_mesh_size": self.numerical["max_mesh_size"],
        }
        mesh_sizes = previous_mesh_sizes
        if refined_mesh_size is not None:
            mesh_sizes = {
                "refined_mesh_size": refined_mesh_size,
                "max_mesh_size": max_mesh_size,
            }
        self.numerical = configure_numerical_controls(
            order=order,
            tolerance=tolerance,
            max_iterations=max_iterations,
            solver_type=solver_type,
            preconditioner=preconditioner,
            device=device,
            **mesh_sizes,
            **{key: self.numerical[key] for key in MESH_CONTROL_KEYS},
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
        # Mesh sizes change the mesh; every other control only changes config.json.
        if (
            self.numerical["refined_mesh_size"]
            != previous_mesh_sizes["refined_mesh_size"]
            or self.numerical["max_mesh_size"] != previous_mesh_sizes["max_mesh_size"]
        ):
            self._invalidate_mesh()
        else:
            self._invalidate_config()

    # ------------------------------------------------------------------ mesh steps

    def _check_mesh_inputs_set(self) -> None:
        if (
            (self.component is None and self._plan_snapshot is None)
            or self.stack is None
            or self.output_dir is None
        ):
            raise ValueError(
                "set_geometry(), set_stack(), and set_output_dir() must run before mesh()."
            )

    def _check_plan_has_no_bump_fill(self) -> None:
        if self.indium_ground_bumps is not None and self.indium_ground_bumps["fill"]:
            raise ValueError(
                "GeometryPlan source already owns its geometry; add authored bumps before prepare()"
            )

    def _plan_vacuum_region(self) -> VacuumRegionSpec | None:
        """A plan that already carries its vacuum region leaves nothing to add."""
        return None if self._plan_owns_vacuum_envelope() else self.vacuum_region

    def _build_mesh(
        self,
        prepared: PreparedMeshInput,
        *,
        component: Any,
        **source: Any,
    ) -> Path:
        """Run the route mesh build; ``source`` carries the plan or component extras."""
        if prepared.materials is not None:
            self._materials = prepared.materials
        self._mesh_result = build_route_mesh(
            component=component,
            stack=prepared.stack,
            route=self.route,
            output_dir=self.output_dir,
            refined_mesh_size=self.numerical["refined_mesh_size"],
            max_mesh_size=self.numerical["max_mesh_size"],
            **{key: self.numerical[key] for key in MESH_CONTROL_KEYS},
            source_geometry_input=prepared.geometry_input,
            representation_context=prepared.physical_plan.context,
            **source,
        )
        return self._mesh_result.mesh_path

    def mesh_summary(self) -> dict[str, Any]:
        """Detach actual mesh observations and the current FEM topology estimate."""
        from ..mesh.summary import summary_from_files

        if self._mesh_result is None:
            raise RuntimeError("mesh_summary requires a generated mesh.")
        return summary_from_files(
            self._mesh_result.mesh_path,
            self._mesh_result.mesh_manifest_path,
            problem=self._PROBLEM,
            fem_order=self.numerical["order"],
        )

    def check_mesh_quality(self) -> MeshQualityReport:
        """Inspect the current mesh without changing simulation state."""
        if self._mesh_result is None:
            raise ValueError("mesh() must run before check_mesh_quality().")
        return check_mesh_quality(
            self._mesh_result.mesh_path, length_scale_m=_MODEL_L0_M
        )

    def prepare_handoff(
        self,
        *,
        profile: str,
        executable: str,
        resources: Mapping[str, Any] | None = None,
        setup_commands: tuple[str, ...] = (),
        petsc_options: tuple[str, ...] = (),
    ) -> HandoffPlan:
        """Seal the run directory for a manual Palace run (see execution/handoff.py)."""
        if self._mesh_result is None or self.config_path is None:
            raise ValueError(
                "mesh() and write_config() must run before prepare_handoff()."
            )
        self.handoff_plan = prepare_handoff(
            profile=profile,
            mesh_result=self._mesh_result,
            config_path=self.config_path,
            executable=executable,
            resources=resources,
            setup_commands=setup_commands,
            petsc_options=petsc_options,
            problem=self._PROBLEM,
        )
        return self.handoff_plan
