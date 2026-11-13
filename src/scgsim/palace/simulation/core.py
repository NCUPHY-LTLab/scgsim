"""What every Palace simulation shares: the plan it is bound to, the geometry settings, ``mesh()`` and the handoff.

``PalaceSimulation`` is the base of ``EigenmodeSim`` and ``ElectrostaticSim``.  It owns the calls that do
not depend on the problem: binding a plan, the output folder, the vacuum region, the route, Surface EPR,
the mesh controls, ``input_summary()``, ``mesh()``, ``simulation_drc``, ``mesh_summary()`` and ``prepare_handoff()``.
A problem class adds its own setters, its numerical defaults and ``write_config()``, and answers what the
core asks of it: the class attributes and methods under "What a problem adds" below.

Which call needs which stage, and how far each call falls back, is the table in ``lifecycle``; this module
only applies it.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal, TypeVar

from scgsim.contracts.palace_run_folder import CONFIG, INDEX_MAP_SCHEMA_VERSION
from scgsim.geometry.models.input import VacuumRegionSpec
from scgsim.geometry.models.physical import ApproximationProfile
from scgsim.geometry.source.plan import GeometryPlanSnapshot
from scgsim.geometry.source.summary import InputSummary, summarize_geometry_input
from scgsim.meshing import Budgets, MesherOptions, SizingRecipe
from scgsim.refusal import Declared, found_text

from ..config.config_records import ConfigInput
from ..config.files import MetadataFile, persist_problem_files
from ..config.index import IndexEntry
from ..config.numerical import NumericalControls, ProblemDefaults
from ..mesh.build import (
    MeshBuildResult,
    RouteGeometry,
    build_route_geometry,
    mesh_route_geometry,
)
from ..mesh.controls import MeshControls
from ..mesh.summary import MeshSummary, summary_from_files
from ..run.handoff import HandoffPlan, prepare_handoff
from ..source.inputs import stack_materials
from ..source.layout_ports import LayoutGeometryRequest, layout_geometry_request
from ..source.layout_snapshot import (
    DRC_VACUUM_PADDING_UM,
    prepare_layout_snapshot,
    snapshot_owns_vacuum,
)
from ..source.ports import LayoutPortBinding, RequestedPort
from ..source.surface_epr import SurfaceEprSpecs
from ..source.workflow import (
    SET_ROUTE,
    SIMULATION_DRC,
    GeometrySettings,
    RouteCall,
    checked_route_setting,
    route_profile,
)
from .drc import SimulationDrcReport, check_meshing_stage, run_stages
from .lifecycle import CALLS, Call, Stage

_WHERE_MESH_AGAIN = "the run folder: call mesh() again"
_WHERE_NUMERICAL_ARG = "the numerical argument of the simulation"
_WHERE_SURFACE_EPR = "the call to set_surface_epr"

_WHERE_SET_OUTPUT_DIR = "the call to set_output_dir"
_WHERE_SET_PLAN = "the call to set_plan"

_Output = TypeVar("_Output")


@dataclass
class PalaceSimulation(ABC):
    """The shared lifecycle of a Palace simulation; use ``EigenmodeSim`` or ``ElectrostaticSim``.

    The simulation prepares geometry, mesh, config and manual handoff files; it never runs Palace.
    """

    output_dir: Path | None = None
    route: Literal["A", "B"] = "B"
    approximation_profile: ApproximationProfile | None = None
    vacuum_region: VacuumRegionSpec | None = None
    surface_epr_specs: SurfaceEprSpecs | None = None
    mesh_controls: MeshControls = field(default_factory=MeshControls)
    # Filled with the problem's ``numerical_defaults`` on construction and by every ``set_numerical``.
    numerical: NumericalControls = field(default_factory=NumericalControls)
    config_path: Path | None = field(default=None, init=False)
    handoff_plan: HandoffPlan | None = field(default=None, init=False)
    _stage: Stage = field(default=Stage.UNBOUND, init=False, repr=False)
    _plan: GeometryPlanSnapshot | None = field(default=None, init=False, repr=False)
    # The materials Surface EPR specs and the config are checked against: the bound stack's, then those of
    # the stack ``mesh()`` prepared, which can add a material (a set vacuum region adds its vacuum).
    _materials: dict[str, Mapping[str, Any]] | None = field(
        default=None, init=False, repr=False
    )
    _mesh_result: MeshBuildResult | None = field(default=None, init=False, repr=False)
    _resolved_ports: list[LayoutPortBinding] = field(
        default_factory=list, init=False, repr=False
    )

    # ---- What a problem adds ------------------------------------------------------------------------

    problem: ClassVar[Literal["Eigenmode", "Electrostatic"]]
    """The Palace ``Problem.Type`` this class writes; the mesh summary and the handoff record it."""

    numerical_defaults: ClassVar[ProblemDefaults]
    """The AMR defaults of this problem, for every control a ``NumericalControls`` leaves to the problem."""

    @abstractmethod
    def _lumped_ports(self) -> list[RequestedPort] | None:
        """The authored port sheets the geometry cuts as lumped ports, or None for a problem with none."""

    @abstractmethod
    def _problem_summary(self) -> dict[str, Any]:
        """The problem's own settings, as ``input_summary()`` shows them beside the shared ones."""

    @abstractmethod
    def _add_drc_inputs(self, snapshot: GeometryPlanSnapshot) -> None:
        """Set the problem inputs ``simulation_drc`` needs before the geometry build of ``snapshot``."""

    @abstractmethod
    def write_config(self) -> Path:
        """Write ``config.json`` and its metadata for the current mesh; see the problem class."""

    def __post_init__(self) -> None:
        """Fill the problem's defaults into the numerical controls the constructor was given.

        Raises:
            TypeError: ``numerical`` is not a ``NumericalControls``.
        """
        self.numerical = self._with_problem_defaults(self.numerical)

    # ---- Binding and geometry settings --------------------------------------------------------------

    def set_plan(self, snapshot: GeometryPlanSnapshot) -> None:
        """Bind the prepared ``GeometryPlan`` snapshot every later call builds from.

        Raises:
            TypeError: ``snapshot`` is not a ``GeometryPlanSnapshot``.
            ValueError: the plan has its own vacuum and ``set_vacuum_region`` was called too.
        """
        if not isinstance(snapshot, GeometryPlanSnapshot):
            raise TypeError(
                Declared(
                    "set_plan() takes a GeometryPlanSnapshot: pass plan.prepare(), not the GeometryPlan",
                    _WHERE_SET_PLAN,
                ).refusal(found_text(type(snapshot).__name__))
            )
        if snapshot_owns_vacuum(snapshot) and self.vacuum_region is not None:
            raise ValueError(_TWO_VACUUMS)
        self._plan = snapshot
        self._materials = stack_materials(snapshot.stack)
        self._fall_back("set_plan")
        self._stage = Stage.BOUND

    def set_output_dir(self, path: str | Path) -> None:
        """Select the run folder ``mesh()`` writes into; it must be empty when ``mesh()`` runs.

        Raises:
            TypeError: ``path`` is not a string or ``Path``.
        """
        if not isinstance(path, (str, Path)):
            raise TypeError(
                Declared(
                    "set_output_dir() takes a path string or Path",
                    _WHERE_SET_OUTPUT_DIR,
                ).refusal(found_text(type(path).__name__))
            )
        self.output_dir = Path(path).expanduser().resolve()
        self._fall_back("set_output_dir")

    def set_vacuum_region(
        self,
        padding: float | list[float] | tuple[float, ...] | Mapping[str, Any] = 0.0,
    ) -> None:
        """Wrap a plan that has no vacuum of its own in a vacuum region padded by ``padding`` (um, six directions).

        Raises:
            ValueError: the bound plan has its own vacuum.
        """
        if self._plan is not None and snapshot_owns_vacuum(self._plan):
            raise ValueError(_TWO_VACUUMS)
        self.vacuum_region = VacuumRegionSpec.from_padding(padding)
        self._fall_back("set_vacuum_region")

    def set_route(
        self,
        route: Literal["A", "B"],
        *,
        approximation_profile: ApproximationProfile | None = None,
    ) -> None:
        """Choose how the conductors are modelled: Route ``"A"`` (films as zero-thickness sheets) or ``"B"`` (3D volumes).

        A simulation that never calls ``set_route`` is Route B.  One call states the whole route setting: it
        replaces the route and ``approximation_profile`` together, so nothing survives from an earlier call or
        the constructor.

        ``approximation_profile`` is the ``ApproximationProfile`` the preparation interprets the plan's
        physical layers with, used as given.  Without one, each route has its default:

        * ``set_route("A")`` uses ``thin_film`` (no parameters): the film's thickness is removed and what is
          above it moves down.
        * ``set_route("A", approximation_profile=profile)`` uses ``substrate_face`` or
          ``metal_gap_equivalent`` with their ``film_bindings``.
        * ``set_route("B")`` uses the identity profile ``route_b``, which is also what no call gives.

        Calling it again discards the mesh, config and handoff.

        Raises:
            ValueError: ``route`` is not ``"A"`` or ``"B"``.
            TypeError: ``approximation_profile`` is not an ``ApproximationProfile``.
        """
        self._set_route(route, approximation_profile, SET_ROUTE)

    def _set_route(
        self,
        route: Any,
        approximation_profile: Any,
        call: RouteCall,
    ) -> None:
        """Store the route setting ``call`` states; a refusal names ``call``, where the setting is changed."""
        # Check every value before storing any, so a refused call leaves the earlier setting whole.
        self.route, self.approximation_profile = checked_route_setting(
            route, approximation_profile, call
        )
        self._fall_back("set_route")

    def set_surface_epr(self, specs: Mapping[str, Mapping[str, Any]] | None) -> None:
        """Request Surface EPR with ``specs`` by interface type, or turn it off again with ``None``.

        Surface EPR is computed only when this was called with specs.  Without it, ``mesh()`` cuts no bands
        and ``write_config()`` writes no Surface-EPR rows.  Surface EPR is geometry: the inset margins of the
        specs become bands that ``mesh()`` cuts into the model, so this runs before ``mesh()``; calling it
        again, ``None`` included, discards the mesh, config and handoff.

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

        Raises:
            ValueError: no plan is bound (the specs are checked against its materials), ``specs`` is an empty
                mapping, or a spec is invalid.
            TypeError: ``specs`` or a spec is not a mapping, or a field has the wrong type.
        """
        materials = self._require("set_surface_epr", self._materials)
        if specs is None:
            self.surface_epr_specs = None
        else:
            _refuse_empty_specs(specs)
            self.surface_epr_specs = SurfaceEprSpecs.normalized(
                specs, materials=materials
            )
        self._fall_back("set_surface_epr")

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
        ``GlobalSizeRecipe``, ``UniformBoundaryRecipe`` or the report-only ``GmshDefaultRecipe``) chooses
        the initial element sizes, ``mesher`` (``MesherOptions``) the Gmsh algorithms and the geometry
        order, ``budgets`` when the repair loop gives up, and ``quality_threshold`` the minimum signed
        inverse condition number (minSICN) of every tetrahedron (default 0.01).  A mesh that cannot reach
        the threshold raises ``MeshRepairError`` from ``mesh()``, except with ``GmshDefaultRecipe``, whose
        mesh is delivered with its quality reported, not enforced.  Geometry order is independent of
        ``set_numerical(order=...)``.  An unchanged setting keeps the current mesh.
        """
        controls = self.mesh_controls.updated(
            recipe=recipe,
            mesher=mesher,
            budgets=budgets,
            quality_threshold=quality_threshold,
        )
        if controls != self.mesh_controls:
            self.mesh_controls = controls
            self._fall_back("set_mesh")

    # ---- Inspecting the input -----------------------------------------------------------------------

    def input_summary(self) -> InputSummary:
        """Show the input ``mesh()`` builds from, and the Palace settings, before mesh work.

        The source half is the geometry input and stack ``mesh()`` prepares from the bound plan, so it holds the
        vacuum region, the requested ports and the route's conductors as the simulation was given them.  The
        Palace half shows the route and the approximation profile the preparation uses (the route's default
        included) and the Surface-EPR specs, ``None`` (shown as "not requested") when ``set_surface_epr`` was
        not called.
        A setup ``mesh()`` refuses while preparing is refused here with the same error, an input with no vacuum,
        planned or set, included.  Nothing is built or written, and the simulation is left as it was.

        Raises:
            ValueError: no plan is bound, or the preparation refuses the setup.
        """
        plan = self._require("input_summary", self._plan)
        prepared = self._geometry_request(plan).source
        data = summarize_geometry_input(
            prepared.geometry_input, prepared_stack=prepared.stack
        ).data
        data["backend"] = {
            "name": "palace",
            "route": self.route,
            "approximation_profile": route_profile(
                self.route, self.approximation_profile
            ).to_payload(),
            "surface_epr_specs": None
            if self.surface_epr_specs is None
            else self.surface_epr_specs.to_payload(),
            **self._problem_summary(),
            "numerical": self.numerical.to_payload(),
            "mesh": self.mesh_controls.to_payload(),
        }
        return InputSummary(data)

    # ---- Meshing ------------------------------------------------------------------------------------

    def mesh(self) -> Path:
        """Discard prior derived state, prepare the source, build the geometry, check it, and mesh it.

        Surface EPR is geometry: ``set_surface_epr`` runs before ``mesh()``, which cuts the inset
        margins of its specs into the model as bands.  When it was not called, no bands are requested.

        Before the mesher starts, the built model goes through the meshing stage of ``simulation_drc``: a
        refusing fine edge or face raises :class:`~scgsim.palace.SimulationDrcError` carrying the same
        report, and the geometry stays in the output folder; warnings do not stop the mesh.

        Returns:
            The path of the written ``palace.msh``.

        Raises:
            ValueError: no plan is bound, no output folder is set, or the preparation or geometry build
                refuses the setup.
            SimulationDrcError: the meshing stage refuses the built model.
        """
        self._fall_back("mesh")
        plan = self._require("mesh", self._plan)
        if self.output_dir is None:
            raise ValueError(
                Declared(
                    "No output folder to mesh into", _WHERE_SET_OUTPUT_DIR
                ).refusal("call set_output_dir() before mesh()")
            )
        request = self._geometry_request(plan)
        if request.materials is not None:
            self._materials = request.materials
        geometry = build_route_geometry(request.source, output_dir=self.output_dir)
        check_meshing_stage(geometry)
        self._mesh_result = mesh_route_geometry(geometry, self.mesh_controls)
        self._resolved_ports = request.ports
        self._stage = Stage.MESHED
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
        route: Literal["A", "B"] = "B",
        approximation_profile: ApproximationProfile | None = None,
    ) -> SimulationDrcReport:
        """Check that this simulation can model and mesh ``component``, in two stages, and return their report.

        The geometry stage runs the preparation and SGB geometry build of ``mesh()`` on the route the
        simulation will use: ``route`` and ``approximation_profile`` are the arguments of ``set_route``, with
        its defaults and refusals, and a refusal names this call.  Without them the check is Route B.  The
        other settings are fixed: zero coupon padding, the problem's own fixed inputs (``EigenmodeSim``:
        every authored port with a fixed inductance, which the geometry never reads; ``ElectrostaticSim``:
        no terminals, which are not needed before meshing) and, when the layout has no vacuum of its own, a
        vacuum region padded by (x, y, z) = (0, 0, 1000) um: it ends at the layout's planar extent and
        reaches 1000 um below and above.  A conductor that reaches that extent only at a point, such as a
        lone round pad with no die outline or ground plane around it, pinches this vacuum and is refused.  A
        layout it refuses raises the error ``mesh()`` raises, unchanged.  With ``surface_epr_specs`` the
        bands are cut as ``set_surface_epr(surface_epr_specs)`` would cut them, and the specs are checked as
        it checks them; without them no bands are cut.

        Only a model that built goes to the meshing stage, ``scgsim.meshing.check_fine_structure``: the fine
        edges and faces the mesher cannot use, each with its location, size, owner and fix.  A refusal there
        raises :class:`~scgsim.palace.SimulationDrcError` carrying both stages; otherwise the
        :class:`~scgsim.palace.SimulationDrcReport` is returned, warnings included.  Nothing is meshed, and
        the geometry is written to a temporary folder that is removed.  ``mesh()`` runs the same meshing
        stage on the model it builds, before the mesher starts.

        A pass means the layout can be modeled on the route and profile checked, with these fixed settings.
        It does not promise that other settings (another route or profile, terminals, junctions, lumped
        supports, another vacuum region) will model; run that simulation's own ``mesh()`` for that.  Route C
        is not a Palace route, so it is refused here as ``set_route`` refuses it.

        Args:
            component: The layout Component under check.
            layer_stack: The PDK layer stack, as ``GeometryPlan`` takes it.
            material_records: The PDK material records, as ``GeometryPlan`` takes them.
            instances: ``{"chip/device": reference, ...}`` for an assembly, as ``GeometryPlan.register_instance``.
            nets: ``{"Ground": ("chip/plane", ...), ...}``, the Net assignment the simulation's plan is given, as
                ``GeometryPlan.set_nets`` takes it.  Without it no conductor has a Net, so any two conductors
                that touch are refused as a different-Net contact.
            surface_epr_specs: The specs ``set_surface_epr`` takes; their margins become bands.
            route: ``"A"`` or ``"B"``, as ``set_route`` takes it.
            approximation_profile: The ``ApproximationProfile`` ``set_route`` takes; None for the route's
                default (``thin_film`` for Route A, ``route_b`` for Route B).

        Raises:
            ValueError: ``route`` is not ``"A"`` or ``"B"``.
            TypeError: ``approximation_profile`` is not an ``ApproximationProfile``.
        """
        sim = cls()
        # The route is checked before the layout is read, so a refused route costs no preparation.
        sim._set_route(route, approximation_profile, SIMULATION_DRC)  # noqa: SLF001 -- the check is a classmethod of the same class
        snapshot = prepare_layout_snapshot(
            component,
            layer_stack=layer_stack,
            material_records=material_records,
            instances=instances,
            nets=nets,
        )
        sim._bind_for_drc(snapshot, surface_epr_specs)  # noqa: SLF001 -- the check is a classmethod of the same class
        return run_stages(sim._build_geometry)  # noqa: SLF001 -- the check is a classmethod of the same class

    def _bind_for_drc(
        self,
        snapshot: GeometryPlanSnapshot,
        surface_epr_specs: Mapping[str, Mapping[str, Any]] | None,
    ) -> None:
        """Bind ``snapshot`` with the fixed settings ``simulation_drc`` documents, on the route already set."""
        self.set_plan(snapshot)
        if not snapshot_owns_vacuum(snapshot):
            # Without vacuum above it a conductor has no metal-air face, so MA and SA bands would cut nothing.
            self.set_vacuum_region(DRC_VACUUM_PADDING_UM)
        self._add_drc_inputs(snapshot)
        if surface_epr_specs is not None:
            self.set_surface_epr(surface_epr_specs)

    def _build_geometry(self, output_dir: str | Path) -> RouteGeometry:
        """Run the preparation and SGB geometry build of ``mesh()`` into ``output_dir``, then stop.

        This is the modeling path ``mesh()`` shares, used by ``simulation_drc``.  The simulation's own
        output folder and mesh state are left as they were.
        """
        plan = self._require("mesh", self._plan)
        return build_route_geometry(
            self._geometry_request(plan).source, output_dir=output_dir
        )

    def _geometry_request(self, plan: GeometryPlanSnapshot) -> LayoutGeometryRequest:
        """Resolve ports and prepare the geometry input of ``plan``, everything before the SGB build.

        ``mesh()``, ``simulation_drc`` and ``input_summary()`` read this one request; it changes no simulation state.
        """
        return layout_geometry_request(
            plan=plan,
            ports=self._lumped_ports(),
            settings=GeometrySettings(
                route=self.route,
                approximation_profile=self.approximation_profile,
                vacuum_region=self.vacuum_region,
                surface_epr_specs=self.surface_epr_specs,
            ),
        )

    def mesh_summary(self) -> MeshSummary:
        """Detach actual mesh observations and the current FEM topology estimate.

        Raises:
            RuntimeError: ``mesh()`` has not built the current mesh.
        """
        mesh = self._require("mesh_summary", self._mesh_result)
        return summary_from_files(
            mesh.mesh_path,
            mesh.mesh_manifest_path,
            problem=self.problem,
            fem_order=self.numerical.order,
        )

    # ---- Config and handoff -------------------------------------------------------------------------

    def _set_numerical(self, controls: NumericalControls) -> None:
        """Store the controls a ``set_numerical`` call built; the mesh stays, the config is discarded."""
        self.numerical = self._with_problem_defaults(controls)
        self._fall_back("set_numerical")

    def _with_problem_defaults(self, controls: NumericalControls) -> NumericalControls:
        """``controls`` with the problem's defaults filled in: the one place the constructor and ``set_numerical`` share."""
        if not isinstance(controls, NumericalControls):
            raise TypeError(
                Declared("numerical", _WHERE_NUMERICAL_ARG).refusal(
                    f"must be NumericalControls: pass {type(self).__name__}(numerical=NumericalControls(...)) or call set_numerical(...)"
                )
            )
        return controls.with_defaults(self.numerical_defaults)

    def _start_config(self) -> ConfigInput:
        """Discard the previous config and return the shared input of a new config of the current mesh.

        Raises:
            ValueError: ``mesh()`` has not built the current mesh, or the numerical controls leave AMR open.
            FileNotFoundError: a file of that mesh is gone from the run folder.
        """
        self._fall_back("write_config")
        mesh = self._require("write_config", self._mesh_result)
        materials = self._require("write_config", self._materials)
        _check_mesh_files(mesh)
        return ConfigInput(
            problem=self.problem,
            groups=mesh.groups,
            mesh_path=mesh.mesh_path,
            materials=materials,
            surface_epr_specs=self.surface_epr_specs,
            numerical=self.numerical,
        )

    def _finish_config(
        self, config: Mapping[str, Any], metadata: Sequence[MetadataFile]
    ) -> Path:
        """Write each metadata file in order, then ``config.json`` last, and reach ``CONFIG_WRITTEN``."""
        mesh = self._require("write_config", self._mesh_result)
        config_path = mesh.output_dir / CONFIG
        self.config_path = persist_problem_files(
            run_folder=mesh.output_dir,
            metadata_files=metadata,
            config_path=config_path,
            config=config,
        )
        self._stage = Stage.CONFIG_WRITTEN
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
        """Write the run script, metadata and ``.tar.gz`` archive that hand the run folder to a manual Palace run.

        Raises:
            ValueError: ``write_config()`` has not written the config of the current mesh, or a handoff
                argument is invalid.
            FileNotFoundError: ``config.json`` or ``palace.msh`` is gone from the run folder.
        """
        mesh = self._require("prepare_handoff", self._mesh_result)
        config_path = self._require("prepare_handoff", self.config_path)
        self.handoff_plan = prepare_handoff(
            profile=profile,
            mesh_result=mesh,
            config_path=config_path,
            executable=executable,
            resources=resources,
            setup_commands=setup_commands,
            petsc_options=petsc_options,
            problem=self.problem,
        )
        self._stage = Stage.HANDED_OFF
        return self.handoff_plan

    # ---- The lifecycle table, applied ---------------------------------------------------------------

    def _require(self, call: Call, output: _Output | None) -> _Output:
        """Return ``output``, held by the stage ``call`` needs; refuse ``call`` if that stage is not reached.

        The stage is the rule; the None test only tells the type checker what the stage guarantees.
        """
        rule = CALLS[call]
        if self._stage < rule.needs or output is None:
            raise rule.error(rule.refusal)
        return output

    def _fall_back(self, call: Call) -> None:
        """Discard every output of a stage after the one ``call`` falls back to."""
        keep = CALLS[call].falls_back_to
        if keep is None or self._stage <= keep:
            return
        if keep < Stage.MESHED:
            self._mesh_result = None
            self._resolved_ports = []
        if keep < Stage.CONFIG_WRITTEN:
            self.config_path = None
        self.handoff_plan = None
        self._stage = keep


_TWO_VACUUMS = Declared(
    "the GeometryPlan",
    "the plan or the simulation: drop sim.set_vacuum_region(), or prepare the plan without "
    "plan.set_vacuum_region()",
).refusal("has its own vacuum (VACUUM_REGION), so the simulation cannot add one")


def _refuse_empty_specs(specs: Mapping[str, Mapping[str, Any]]) -> None:
    """Refuse an empty mapping: "no Surface EPR" has one spelling, no call (or ``None``)."""
    if isinstance(specs, Mapping) and not specs:
        raise ValueError(
            Declared(
                "set_surface_epr({}) requests no Surface EPR", _WHERE_SURFACE_EPR
            ).refusal(
                "to have none, do not call set_surface_epr(), or call set_surface_epr(None) to turn it off again"
            )
        )


def _check_mesh_files(mesh: MeshBuildResult) -> None:
    """Refuse a mesh whose ``palace.msh`` or mesh manifest is gone from the run folder or no longer reads.

    The config is written beside the mesh and the handoff seals both with the manifest, so a run folder
    changed after ``mesh()`` is refused before anything is written into it.
    """
    for path in (mesh.mesh_path, mesh.mesh_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(
                Declared(f"The current {path.name}", _WHERE_MESH_AGAIN).refusal(
                    f"is missing from {mesh.output_dir}: call mesh() again"
                )
            )
    # Only a check: parsing refuses a manifest that no longer reads as JSON; its content is not needed here.
    json.loads(mesh.mesh_manifest_path.read_text(encoding="utf-8"))


def index_map(entries: Sequence[IndexEntry], **extra: Any) -> dict[str, Any]:
    """The ``palace_index_map.json`` payload: the schema, the entries, then the problem's ``extra`` records."""
    return {
        "schema_version": INDEX_MAP_SCHEMA_VERSION,
        "entries": [entry.to_payload() for entry in entries],
        **extra,
    }


__all__ = ["PalaceSimulation"]
