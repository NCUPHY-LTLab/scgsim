"""Mutable Eigenmode preparation state and its ordered failure lifecycle.

This class prepares geometry and manual handoff files; it never runs Palace.
"""

# long-file: the setters, mesh(), its geometry half and simulation_drc share the simulation's private state; a split would reach into it.

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from scgsim.geometry.models.input import VacuumRegionSpec
from scgsim.geometry.models.physical import ApproximationProfile
from scgsim.geometry.source.plan import GeometryPlanSnapshot
from scgsim.geometry.source.summary import InputSummary, summarize_geometry_input
from scgsim.geometry.surface_epr.specs import normalize_surface_epr_specs
from scgsim.meshing import Budgets, MesherOptions, SizingRecipe
from scgsim.palace.mesh.quality import MeshQualityReport, check_mesh_quality

from ..execution.handoff import HandoffPlan, prepare_handoff
from ..mesh.build import (
    MeshBuildResult,
    RouteGeometry,
    build_route_geometry,
    mesh_route_geometry,
)
from ..mesh.controls import MeshControls
from ..mesh.summary import summary_from_files
from ..results.epr import PalaceEprResult, reanalyze_epr, show_epr
from ..results.index_contract import INDEX_MAP_SCHEMA_VERSION
from .config import (
    MODEL_L0_M,
    LayoutPortBinding,
    build_eigenmode_config,
    configure_numerical_controls,
)
from .drc import SimulationDrcReport, check_meshing_stage, run_stages
from .inputs import (
    load_stack,
    non_negative_number,
    validate_stack_material_kinds,
)
from .layout_snapshot import (
    DRC_VACUUM_PADDING_UM,
    prepare_layout_snapshot,
    snapshot_owns_vacuum,
)
from .ports import RequestedPort, resolve_layout_ports
from .transforms import (
    RouteAThinFilm,
    normalize_route_a_thin_film,
    validate_non_negative_int,
    validate_nonempty_string,
    validate_positive_number,
)
from .workflow import persist_problem_files, prepare_mesh_input

# The geometry build never reads a port's inductance, but ``add_port`` requires a positive one.
_PORT_INDUCTANCE_H = 1.0e-9


@dataclass(frozen=True)
class _GeometryRequest:
    """The resolved ports and the geometry keywords that ``mesh()`` and the modeling check both build from."""

    ports: list[LayoutPortBinding]
    source: dict[str, Any]


@dataclass
class EigenmodeSim:
    """Prepare an SGB-authored Eigenmode mesh, config, and manual handoff."""

    component: Any | None = None
    stack: Mapping[str, Any] | None = None
    output_dir: Path | None = None
    airbox: dict[str, float] = field(default_factory=dict)
    route: Literal["A", "B"] = "B"
    route_a_thin_film: RouteAThinFilm | None = None
    approximation_profile: ApproximationProfile | None = None
    ports: list[RequestedPort] = field(default_factory=list)
    surface_epr_specs: dict[str, dict[str, Any]] | None = None
    epr_request: dict[str, tuple[str, ...] | None] | None = None
    num_modes: int = 10
    target_hz: float | None = None
    eigenmode_tolerance: float = 1e-6
    save_fields: int = 0
    mesh_controls: MeshControls = field(default_factory=MeshControls)
    numerical: dict[str, Any] = field(
        default_factory=lambda: configure_numerical_controls(
            amr_nonconformal=True,
            amr_max_passes=None,
            amr_tolerance=None,
            amr_update_fraction=None,
        )
    )
    _materials: dict[str, Mapping[str, Any]] | None = field(default=None, init=False)
    _resolved_ports: list[LayoutPortBinding] = field(default_factory=list, init=False)
    vacuum_region: VacuumRegionSpec | None = None
    indium_ground_bumps: dict[str, Any] | None = None
    _mesh_result: MeshBuildResult | None = field(default=None, init=False)
    config_path: Path | None = field(default=None, init=False)
    handoff_plan: HandoffPlan | None = field(default=None, init=False)
    _plan_snapshot: GeometryPlanSnapshot | None = field(
        default=None, init=False, repr=False
    )

    @staticmethod
    def reanalyze_epr(
        result: PalaceEprResult,
        *,
        surface_defaults: Mapping[str, Mapping[str, Any]] | None = None,
        group_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> PalaceEprResult:
        """Recompute detached EPR loss values with explicit overrides."""
        return reanalyze_epr(
            result, surface_defaults=surface_defaults, group_overrides=group_overrides
        )

    @staticmethod
    def show_epr(result: PalaceEprResult, *, mode: int | None = None) -> Any:
        """Display detached EPR values without opening the solver."""
        return show_epr(result, mode=mode)

    def _invalidate_mesh(self) -> None:
        self._mesh_result = None
        self._resolved_ports = []
        self.config_path = None
        self.handoff_plan = None

    def _invalidate_config(self) -> None:
        self.config_path = None
        self.handoff_plan = None

    def set_geometry(self, component: Any) -> None:
        """Bind standalone authored geometry and invalidate derived mesh state."""
        if self._plan_snapshot is not None:
            raise ValueError(
                "GeometryPlan owns the paired source; use set_plan(new_snapshot) or a new Sim"
            )
        if component is None or not callable(getattr(component, "write_gds", None)):
            raise TypeError("component must provide write_gds(path).")
        self.component = component
        self._plan_snapshot = None
        self._invalidate_mesh()

    def set_plan(self, snapshot: GeometryPlanSnapshot) -> None:
        """Consume one detached source/stack pair from a notebook GeometryPlan."""
        if not isinstance(snapshot, GeometryPlanSnapshot):
            raise TypeError("set_plan requires a GeometryPlanSnapshot")
        if "VACUUM_REGION" in snapshot.stack["solution_regions"] and (
            self.vacuum_region is not None or self.airbox
        ):
            raise ValueError("GeometryPlan owns the outer vacuum envelope")
        trial = EigenmodeSim()
        trial.set_stack(snapshot.stack)
        self.component = None
        self.stack = trial.stack
        self._materials = trial._materials
        self._plan_snapshot = snapshot
        self._invalidate_mesh()

    def input_summary(self) -> InputSummary:
        """Show the bound Plan source and Palace settings before mesh work."""

        if self._plan_snapshot is None:
            raise ValueError(
                "input_summary requires a bound GeometryPlan snapshot; normalize "
                "a raw Component explicitly first"
            )
        data = summarize_geometry_input(
            self._plan_snapshot.geometry_input,
            prepared_stack=self._plan_snapshot.stack,
        ).data
        data["backend"] = {
            "name": "palace",
            "route": self.route,
            "route_a_thin_film": self.route_a_thin_film,
            "approximation_profile": self.approximation_profile.to_payload()
            if self.approximation_profile is not None
            else None,
            "airbox": self.airbox,
            "ports": [
                {
                    "name": item.name,
                    "layer": item.layer,
                    "inductance_h": item.inductance,
                }
                for item in self.ports
            ],
            "surface_epr_specs": self.surface_epr_specs,
            "epr_request": self.epr_request,
            "num_modes": self.num_modes,
            "target_hz": self.target_hz,
            "eigenmode_tolerance": self.eigenmode_tolerance,
            "save_fields": self.save_fields,
            "numerical": self.numerical,
            "mesh": self.mesh_controls.to_payload(),
            "indium_ground_bumps": self.indium_ground_bumps,
        }
        return InputSummary(data)

    def set_stack(self, stack: Mapping[str, Any] | str | Path) -> None:
        """Validate standalone stack materials before replacing the source state."""
        if self._plan_snapshot is not None:
            raise ValueError(
                "GeometryPlan owns the paired source; use set_plan(new_snapshot) or a new Sim"
            )
        payload = load_stack(stack)
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
        validate_stack_material_kinds(payload, resolved)
        self.stack = payload
        self._materials = resolved
        self._plan_snapshot = None
        self._invalidate_mesh()

    def set_output_dir(self, path: str | Path) -> None:
        """Select the output directory and invalidate derived artifacts."""
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
        """Validate all envelope controls before updating mesh-dependent state."""
        if (
            self._plan_snapshot is not None
            and "VACUUM_REGION" in self._plan_snapshot.stack["solution_regions"]
        ):
            raise ValueError("GeometryPlan owns the outer vacuum envelope")
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
        if self._plan_snapshot is not None and snapshot_owns_vacuum(
            self._plan_snapshot
        ):
            raise ValueError("GeometryPlan owns the outer vacuum envelope")
        if self.airbox:
            raise ValueError(
                "set_vacuum_region is mutually exclusive with set_airbox()."
            )
        self.vacuum_region = VacuumRegionSpec.from_padding(padding)
        self._invalidate_mesh()

    def set_indium_ground_bumps(
        self, *, fill: bool, fill_pitch_um: float, fill_clearance_um: float
    ) -> None:
        """Request public-PDK authored-plus-ground-fill bumps for the next mesh."""
        if not isinstance(fill, bool):
            raise TypeError("fill must be a bool.")
        self.indium_ground_bumps = {
            "fill": fill,
            "fill_pitch_um": validate_positive_number(fill_pitch_um, "fill_pitch_um"),
            "fill_clearance_um": non_negative_number(
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
        """Configure Surface EPR and the required Route-A thin-film lowering.

        Surface EPR is geometry: its inset margins become bands that ``mesh()`` cuts into the model, so
        this runs before ``mesh()``.  Calling it again invalidates the mesh, config and handoff.

        Each spec (``"MA"``, ``"MS"``, ``"SA"``) may carry:

        * ``inset_margins_um``: strictly increasing positive margins in micrometres.  ``mesh()`` asks SGB
          for bands at these margins on every face of the type, in one geometry build.
        * ``face_kinds``: restrict the spec to some of ``"sheet"``, ``"top"``, ``"bottom"`` and
          ``"interface"``.  With margins, a kind SGB cannot band (``"sidewall"``) raises ``ValueError``.
        * ``inset_margins_by_entity_um`` (``MA`` and ``MS``): ``{entity_id: [margins]}`` gives chosen
          conductors their own margins; the rest of the type keeps ``inset_margins_um``, if given.  This
          builds the geometry twice (once to find the conductors' faces), and a named entity with no face
          of the type raises ``ValueError`` from ``mesh()``.

        One face has one set of margins: a Route A sheet reached by both ``MA`` and ``MS`` must get the same
        margins from both (compared on the model grid), or ``mesh()`` raises.
        """
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

    def add_port(
        self,
        name: str,
        *,
        layer: str,
        layout_sheet: bool = False,
        inductance: float,
    ) -> None:
        """Bind one authored GDSFactory port to its target logical metal."""
        if layout_sheet is not True:
            raise ValueError("Eigenmode V1 supports only layout_sheet=True ports.")
        name = validate_nonempty_string(name, "port name")
        layer = validate_nonempty_string(layer, "port target layer")
        if any(port.name == name for port in self.ports):
            raise ValueError(f"layout_sheet port {name!r} is already configured.")
        self.ports.append(
            RequestedPort(
                name=name,
                layer=layer,
                inductance=validate_positive_number(inductance, "inductance"),
            )
        )
        self._invalidate_mesh()

    def set_epr_request(
        self,
        *,
        surface_interfaces: Sequence[str] | None = None,
        bulk_domain_ids: Sequence[str] | None = None,
        port_names: Sequence[str] | None = None,
    ) -> None:
        """Select reported EPR quantities without changing the physical solve."""

        def selected(values: Sequence[str] | None, name: str) -> tuple[str, ...] | None:
            if values is None:
                return None
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
                raise TypeError(f"{name} must be a sequence of IDs or None")
            normalized = tuple(values)
            if any(not isinstance(value, str) or not value for value in normalized):
                raise ValueError(f"{name} must contain non-empty text IDs")
            if len(normalized) != len(set(normalized)):
                raise ValueError(f"{name} must not repeat IDs")
            return normalized

        surfaces = selected(surface_interfaces, "surface_interfaces")
        if surfaces is not None and set(surfaces) - {"MA", "MS", "SA"}:
            raise ValueError("surface_interfaces supports only MA, MS, and SA")
        request = {
            "surface_interfaces": surfaces,
            "bulk_domain_ids": selected(bulk_domain_ids, "bulk_domain_ids"),
            "port_names": selected(port_names, "port_names"),
        }
        self.epr_request = request
        self._invalidate_config()

    def set_eigenmode(
        self,
        *,
        num_modes: int = 10,
        target: float | None = None,
        tolerance: float = 1e-6,
        save: int = 0,
    ) -> None:
        """Validate mode controls atomically, then invalidate config and handoff."""
        if (
            not isinstance(num_modes, int)
            or isinstance(num_modes, bool)
            or num_modes < 1
        ):
            raise ValueError("num_modes must be a positive integer.")
        if target is not None:
            target = validate_positive_number(target, "target")
        normalized_tolerance = validate_positive_number(tolerance, "tolerance")
        normalized_save = validate_non_negative_int(save, "save")
        self.num_modes = num_modes
        self.target_hz = target
        self.eigenmode_tolerance = normalized_tolerance
        self.save_fields = normalized_save
        self._invalidate_config()

    def set_mesh(
        self,
        *,
        recipe: SizingRecipe | None = None,
        mesher: MesherOptions | None = None,
        budgets: Budgets | None = None,
        quality_threshold: float | None = None,
    ) -> None:
        """Set geometry-mesh controls; every omitted value retains its setting.

        These are the controls of ``scgsim.meshing.mesh_sgb``: ``recipe`` (``FeatureRecipe``,
        ``GlobalSizeRecipe`` or ``UniformBoundaryRecipe``) chooses the initial element sizes, ``mesher``
        (``MesherOptions``) the Gmsh algorithms and the geometry order, ``budgets`` when the repair loop
        gives up, and ``quality_threshold`` the minimum signed inverse condition number (minSICN) of every tetrahedron (default
        0.01).  A mesh that cannot reach the threshold raises ``MeshRepairError`` from ``mesh()``.
        Geometry order is independent of ``set_numerical(order=...)``.
        """
        controls = self.mesh_controls.updated(
            recipe=recipe,
            mesher=mesher,
            budgets=budgets,
            quality_threshold=quality_threshold,
        )
        if controls != self.mesh_controls:
            self.mesh_controls = controls
            self._invalidate_mesh()

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
        amr_max_passes: int | None = None,
        amr_nonconformal: bool | None = None,
        amr_nc_levels: int | None = None,
        amr_tolerance: float | None = None,
        amr_update_fraction: float | None = None,
        save_adapt_iterations: bool | None = None,
        save_adapt_mesh: bool | None = None,
        save_adapt_diagnostics: bool | None = None,
        estimator_mg: bool | None = None,
        output_paraview: bool | None = None,
        output_grid_function: bool | None = None,
    ) -> None:
        """Set solver, FEM, AMR, and output controls for the next config write.

        ``amr_max_passes`` is Palace ``MaxIts``. ``amr_nonconformal`` is Palace
        ``Model.Refinement.Nonconformal``. Omitted AMR controls retain their
        settings. New Eigenmode requests use NC and require explicit case
        controls before config generation; max passes zero disables AMR.
        ``amr_nc_levels`` sets native ``MaxNCLevels``; zero means no limit.
        Omitted Krylov and nonconformal-level controls retain current choices.
        Explicit ``ksp_type="Default"`` uses Palace's problem-specific default.
        """
        self.numerical = configure_numerical_controls(
            order=order,
            tolerance=tolerance,
            max_iterations=max_iterations,
            solver_type=solver_type,
            preconditioner=preconditioner,
            device=device,
            ksp_type=self.numerical.get("ksp_type") if ksp_type is None else ksp_type,
            amr_nc_levels=(
                self.numerical.get("amr_nc_levels")
                if amr_nc_levels is None
                else amr_nc_levels
            ),
            amr_max_passes=(
                self.numerical["amr_max_passes"]
                if amr_max_passes is None
                else amr_max_passes
            ),
            amr_nonconformal=(
                self.numerical["amr_nonconformal"]
                if amr_nonconformal is None
                else amr_nonconformal
            ),
            amr_tolerance=(
                self.numerical["amr_tolerance"]
                if amr_tolerance is None
                else amr_tolerance
            ),
            amr_update_fraction=(
                self.numerical["amr_update_fraction"]
                if amr_update_fraction is None
                else amr_update_fraction
            ),
            save_adapt_iterations=save_adapt_iterations,
            save_adapt_mesh=save_adapt_mesh,
            save_adapt_diagnostics=(
                self.numerical.get("save_adapt_diagnostics")
                if save_adapt_diagnostics is None
                else save_adapt_diagnostics
            ),
            estimator_mg=estimator_mg,
            output_paraview=output_paraview,
            output_grid_function=output_grid_function,
        )
        self._invalidate_config()

    def mesh(self) -> Path:
        """Invalidate prior derived state, prepare the source, build the geometry, check it, and mesh it.

        Surface EPR is geometry: ``set_surface_epr`` runs before ``mesh()``, which cuts the inset
        margins of its specs into the model as bands.  Without Surface EPR no bands are requested.

        Before the mesher starts, the built model goes through the meshing stage of ``simulation_drc``: a
        refusing fine edge or face raises :class:`~scgsim.palace.SimulationDrcError` carrying the same
        report, and the geometry stays in the output folder; warnings do not stop the mesh.
        """
        self._invalidate_mesh()
        if (
            (self.component is None and self._plan_snapshot is None)
            or self.stack is None
            or self.output_dir is None
        ):
            raise ValueError(
                "set_geometry(), set_stack(), and set_output_dir() must run before mesh()."
            )
        request = self._geometry_request(self.stack)
        geometry = build_route_geometry(**request.source, output_dir=self.output_dir)
        check_meshing_stage(geometry)
        self._mesh_result = mesh_route_geometry(geometry, self.mesh_controls)
        self._resolved_ports = request.ports
        return self._mesh_result.mesh_path

    @classmethod
    def simulation_drc(  # noqa: PLR0913 -- the layout, its PDK data, and the plan settings the simulation takes
        cls,
        component: Any,
        *,
        layer_stack: Any,
        material_records: Mapping[str, Mapping[str, Any]],
        instances: Mapping[str, Any] | None = None,
        nets: Mapping[str, Sequence[str]] | None = None,
        surface_epr_specs: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> SimulationDrcReport:
        """Check that an ``EigenmodeSim`` can model and mesh ``component``, in two stages, and return their report.

        The geometry stage runs the preparation and SGB geometry build of ``mesh()``, with fixed settings:
        Route B, Palace's default airbox, a 1 um vacuum region when the layout has none of its own, zero
        coupon padding, and every authored port with a fixed inductance (the geometry never reads it).  A
        layout it refuses raises the error ``mesh()`` raises, unchanged.  With ``surface_epr_specs`` the bands
        are cut as ``set_surface_epr(representation="B", specs=...)`` would cut them.

        Only a model that built goes to the meshing stage, ``scgsim.meshing.check_fine_structure``: the fine
        edges and faces the mesher cannot use, each with its location, size, owner and fix.  A refusal there
        raises :class:`~scgsim.palace.SimulationDrcError` carrying both stages; otherwise the
        :class:`~scgsim.palace.SimulationDrcReport` is returned, warnings included.  Nothing is meshed, and
        the geometry is written to a temporary folder that is removed.  ``mesh()`` runs the same meshing
        stage on the model it builds, before the mesher starts.

        A pass means the layout can be modeled with these settings.  It does not promise that other
        settings (Route A, junctions, lumped supports, another vacuum region) will model; run that
        simulation's own ``mesh()`` for that.

        Args:
            component: The layout Component under check.
            layer_stack: The PDK layer stack, as ``GeometryPlan`` takes it.
            material_records: The PDK material records, as ``GeometryPlan`` takes them.
            instances: ``{"chip/device": reference, ...}`` for an assembly, as ``GeometryPlan.register_instance``.
            nets: ``{"Ground": ("chip/plane", ...), ...}``, the Net assignment the simulation's plan is given, as
                ``GeometryPlan.set_nets`` takes it.  Without it no conductor has a Net, so any two conductors
                that touch are refused as a different-Net contact.
            surface_epr_specs: The specs ``set_surface_epr`` takes; their margins become bands.
        """
        snapshot = prepare_layout_snapshot(
            component,
            layer_stack=layer_stack,
            material_records=material_records,
            instances=instances,
            nets=nets,
        )
        sim = cls()
        sim.set_plan(snapshot)
        if not snapshot_owns_vacuum(snapshot):
            # Without vacuum above it a conductor has no metal-air face, so MA and SA bands would cut nothing.
            sim.set_vacuum_region(DRC_VACUUM_PADDING_UM)
        for port in snapshot.stack["metadata"].get("port_sheet_source_layers", ()):
            sim.add_port(
                port["name"],
                layer=port["target_layer"],
                layout_sheet=True,
                inductance=_PORT_INDUCTANCE_H,
            )
        if surface_epr_specs is not None:
            sim.set_surface_epr(representation="B", specs=surface_epr_specs)
        return run_stages(sim._build_geometry)  # noqa: SLF001 -- the check is a classmethod of the same class

    def _build_geometry(self, output_dir: str | Path) -> RouteGeometry:
        """Run the preparation and SGB geometry build of ``mesh()`` into ``output_dir``, then stop.

        This is the modeling path ``mesh()`` shares, used by ``simulation_drc``.  The simulation's own output folder and mesh state are left as they were.
        """
        if (
            self.component is None and self._plan_snapshot is None
        ) or self.stack is None:
            raise ValueError(
                "set_geometry() or set_plan(), and set_stack(), must run before the geometry build."
            )
        return build_route_geometry(
            **self._geometry_request(self.stack).source, output_dir=output_dir
        )

    def _geometry_request(self, stack: Mapping[str, Any]) -> _GeometryRequest:
        """Resolve ports and prepare the geometry input, everything before the SGB build, for a plan or a Component."""
        if self._plan_snapshot is None:
            return self._component_geometry_request(stack)
        if self.indium_ground_bumps is not None and self.indium_ground_bumps["fill"]:
            raise ValueError(
                "GeometryPlan source already owns its geometry; add authored bumps before prepare()"
            )
        available_records = self._plan_snapshot.stack["metadata"].get(
            "port_sheet_source_layers", ()
        )
        available = {record["name"]: record for record in available_records}
        if len(available) != len(available_records):
            raise ValueError("GeometryPlan port names must be unique")
        resolved, source_records = [], []
        for requested in self.ports:
            record = available.get(requested.name)
            if record is None or record["target_layer"] != requested.layer:
                raise ValueError(
                    f"GeometryPlan has no matching authored port {requested.name!r}"
                )
            resolved.append(
                LayoutPortBinding(
                    index=len(resolved) + 1,
                    name=requested.name,
                    target_layer=requested.layer,
                    source_layer=f"{record['layer']}/{record['datatype']}",
                    direction=tuple(record["direction"]),
                    inductance=requested.inductance,
                )
            )
            source_records.append({**record, "port_index": len(resolved)})
        source_stack = self._plan_snapshot.stack
        prepared = prepare_mesh_input(
            component=None,
            stack=source_stack,
            route=self.route,
            route_a_thin_film=self.route_a_thin_film,
            approximation_profile=self.approximation_profile,
            airbox=self.airbox,
            vacuum_region=(
                None
                if "VACUUM_REGION" in source_stack["solution_regions"]
                else self.vacuum_region
            ),
            indium_ground_bumps=None,
            build_input=self._plan_snapshot.geometry_input,
            port_sheet_source_layers=source_records,
        )
        if prepared.materials is not None:
            self._materials = prepared.materials
        return _GeometryRequest(
            resolved,
            {
                "component": None,
                "stack": prepared.stack,
                "route": self.route,
                "surface_epr_specs": self.surface_epr_specs,
                "source_gds_bytes": self._plan_snapshot.gds_bytes,
                "source_geometry_input": prepared.geometry_input,
                "representation_context": prepared.physical_plan.context,
            },
        )

    def _component_geometry_request(self, stack: Mapping[str, Any]) -> _GeometryRequest:
        resolved, source_records = (
            resolve_layout_ports(self.component, self.ports) if self.ports else ([], [])
        )
        prepared = prepare_mesh_input(
            component=self.component,
            stack=stack,
            route=self.route,
            route_a_thin_film=self.route_a_thin_film,
            approximation_profile=self.approximation_profile,
            airbox=self.airbox,
            vacuum_region=self.vacuum_region,
            indium_ground_bumps=self.indium_ground_bumps,
            port_sheet_source_layers=source_records,
        )
        if prepared.materials is not None:
            self._materials = prepared.materials
        return _GeometryRequest(
            resolved,
            {
                "component": prepared.component,
                "stack": prepared.stack,
                "route": self.route,
                "surface_epr_specs": self.surface_epr_specs,
                "indium_ground_bump_fill": prepared.indium_ground_bump_fill,
                "source_geometry_input": prepared.geometry_input,
                "representation_context": prepared.physical_plan.context,
            },
        )

    def mesh_summary(self) -> dict[str, Any]:
        """Detach actual mesh observations and the current FEM topology estimate."""
        if self._mesh_result is None:
            raise RuntimeError("mesh_summary requires a generated mesh.")
        return summary_from_files(
            self._mesh_result.mesh_path,
            self._mesh_result.mesh_manifest_path,
            problem="Eigenmode",
            fem_order=self.numerical["order"],
        )

    def check_mesh_quality(self) -> MeshQualityReport:
        """Inspect the current mesh without changing simulation state."""

        if self._mesh_result is None:
            raise ValueError("mesh() must run before check_mesh_quality().")
        return check_mesh_quality(
            self._mesh_result.mesh_path, length_scale_m=MODEL_L0_M
        )

    def write_config(self) -> Path:
        """Validate current mesh bindings and persist metadata before config."""
        self._invalidate_config()
        if self._mesh_result is None or self._materials is None:
            raise ValueError("set_stack() and mesh() are required.")
        if not self._mesh_result.mesh_path.is_file():
            raise FileNotFoundError(
                "current palace.msh is missing; mesh() must be rerun."
            )
        result = build_eigenmode_config(
            groups=self._mesh_result.groups,
            ports=self._resolved_ports,
            materials=self._materials,
            surface_epr_specs=self.surface_epr_specs or {},
            numerical=self.numerical,
            num_modes=self.num_modes,
            target_hz=self.target_hz,
            eigenmode_tolerance=self.eigenmode_tolerance,
            save_fields=self.save_fields,
            mesh_path=self._mesh_result.mesh_path,
        )
        request = self.epr_request
        if request is not None:
            available = {
                "surface_interfaces": {
                    row["interface_type"]
                    for item in result.index_entries
                    if item["section"] == "Boundaries.Postprocessing.Dielectric"
                    for row in item["metadata"]["rows"]
                },
                "bulk_domain_ids": {
                    item["entry_name"]
                    for item in result.index_entries
                    if item["section"] == "Domains.Postprocessing.Energy"
                },
                "port_names": {
                    item["port_name"]
                    for item in result.index_entries
                    if item["section"] == "Boundaries.LumpedPort"
                },
            }
            for name, configured in available.items():
                chosen = request[name]
                if chosen is not None and set(chosen) - configured:
                    raise ValueError(f"EPR request selects an unknown {name} binding")
        index_payload = {
            "schema_version": INDEX_MAP_SCHEMA_VERSION,
            "entries": result.index_entries,
        }
        json.loads(self._mesh_result.mesh_manifest_path.read_text(encoding="utf-8"))
        if request is not None:
            index_payload["epr_request"] = {
                "schema_version": "scgsim.palace.epr-selection.v1",
                **{
                    name: None if values is None else list(values)
                    for name, values in request.items()
                },
            }
        metadata = self._mesh_result.output_dir / "metadata"
        config_path = self._mesh_result.output_dir / "config.json"
        self.config_path = persist_problem_files(
            metadata_files=(
                (
                    metadata / "palace_index_map.json",
                    index_payload,
                ),
                (
                    metadata / "palace_material_resolution.json",
                    {
                        "schema_version": 1,
                        "solution_volumes": result.material_resolution,
                    },
                ),
                (
                    metadata / "port_information.json",
                    {"schema_version": 1, "ports": result.port_information},
                ),
                (metadata / "palace_numerical_controls.json", self.numerical),
            ),
            config_path=config_path,
            config=result.config,
        )
        return config_path

    def prepare_handoff(
        self,
        *,
        profile: str,
        executable: str,
        resources: Mapping[str, Any] | None = None,
        setup_commands: tuple[str, ...] = (),
        petsc_options: tuple[str, ...] = (),
    ) -> HandoffPlan:
        """Package the current mesh and config for explicit manual execution."""
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
            problem="Eigenmode",
        )
        return self.handoff_plan


__all__ = ["EigenmodeSim"]
