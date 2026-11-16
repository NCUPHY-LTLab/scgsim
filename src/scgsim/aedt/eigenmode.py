"""``EigenmodeSim``: configure an Eigenmode or EPR run on a geometry plan and prepare its handoff.

The source is a ``GeometryPlan`` snapshot only (``set_plan``): it carries the layout, the stack with its vacuum
region and the plan's Surface-EPR specs.  ``prepare_handoff`` builds the plan's SGB model for AEDT
(:func:`prepare_sgb_model`) and writes the HFSS Eigenmode handoff that carries it (issue #67);
``prepared_model`` is then the model in that handoff.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from scgsim.aedt.epr.models import (
    EprResult,
)
from scgsim.aedt.specs.common import (
    LumpedRlc,
)
from scgsim.aedt.specs.epr import (
    EprAnalysisRequest,
    ExpressionCacheConvergence,
    ExpressionCacheConvergenceSet,
    PlanarJunction,
)
from scgsim.geometry import (
    GeometryPlanSnapshot,
    InputSummary,
    summarize_geometry_input,
)
from scgsim.geometry.source.plan import GeometryPlan, simulation_plan

from .drc import (
    SimulationDrcReport,
    check_eigenmode_model,
    fixed_port_rlcs,
    run_geometry_stage,
)
from .epr.reanalysis import reanalyze_epr
from .preparation.geometry import prepare_hfss_eigenmode_from_geometry
from .preparation.handoff import HandoffPlan
from .preparation.sgb_handoff import SGB_MODEL_DIR
from .preparation.sgb_model import PreparedSgbModel, prepare_sgb_model
from .presentation.epr import show_epr
from .results.resolve import ResolvedRun, resolve_results
from .specs.common import AedtResources
from .specs.hfss import EigenmodeRunControl
from .specs.modeling import Modeling, modeling_from_value
from .versions import normalize_aedt_version


@dataclass
class EigenmodeSim:
    """Configure ordinary and EPR Eigenmode on one ``GeometryPlan`` snapshot, given by ``set_plan``."""

    modeling: Modeling
    aedt_version: str = field(kw_only=True)
    output_dir: Path | None = None
    project_name: str | None = None
    design_name: str | None = None
    run_control: EigenmodeRunControl | None = None
    resources: AedtResources | None = None
    junctions: tuple[PlanarJunction, ...] = ()
    lumped_rlcs: tuple[LumpedRlc, ...] = ()
    epr_request: EprAnalysisRequest | None = None
    expression_convergence: (
        ExpressionCacheConvergence | ExpressionCacheConvergenceSet | None
    ) = None
    prepared_model: PreparedSgbModel | None = field(default=None, init=False)
    handoff_plan: HandoffPlan | None = field(default=None, init=False)
    _plan_snapshot: GeometryPlanSnapshot | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Normalize the public modeling selector at construction time."""
        modeling_from_value(self.modeling)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "aedt_version":
            value = normalize_aedt_version(value)
            if "aedt_version" in self.__dict__ and self.aedt_version != value:
                self.handoff_plan = None
        object.__setattr__(self, name, value)

    def _invalidate_model(self) -> None:
        self.prepared_model = None
        self.handoff_plan = None

    def set_plan(self, snapshot: GeometryPlanSnapshot) -> None:
        """Bind the prepared ``GeometryPlan`` snapshot the model is built from, and discard the prepared model.

        The snapshot brings the plan's vacuum region (AEDT's Region) and Surface-EPR specs
        (``GeometryPlan.set_vacuum_region``, ``GeometryPlan.set_surface_epr``).
        """
        if not isinstance(snapshot, GeometryPlanSnapshot):
            raise TypeError("set_plan requires a GeometryPlanSnapshot")
        self._plan_snapshot = snapshot
        self._invalidate_model()

    @property
    def surface_epr_specs(self) -> dict[str, dict[str, Any]] | None:
        """The normalized Surface-EPR specs of the bound plan, None when it has none or none is bound.

        With specs, their inset margins become bands of the SGB model and each spec's rows are selected in
        AEDT; without them, neither.
        """
        if self._plan_snapshot is None:
            return None
        return self._plan_snapshot.surface_epr_specs

    def set_output_dir(self, path: str | Path) -> None:
        """Set the resolved output directory and invalidate any prepared handoff."""
        if not isinstance(path, (str, Path)):
            raise TypeError("output_dir must be a path string or Path")
        resolved = Path(path).expanduser().resolve()
        self.output_dir = resolved
        self.handoff_plan = None

    def set_eigenmode(
        self,
        *,
        project_name: str,
        design_name: str,
        run_control: EigenmodeRunControl | None = None,
        resources: AedtResources | None = None,
    ) -> None:
        """Declare native model identity and an optional complete solve setup."""
        if (
            not isinstance(project_name, str)
            or not project_name.strip()
            or Path(project_name).name != project_name
            or Path(project_name).suffix
        ):
            raise ValueError("project_name must be a plain project stem")
        if not isinstance(design_name, str) or not design_name.strip():
            raise ValueError("design_name must be non-empty text")
        if run_control is not None and not isinstance(run_control, EigenmodeRunControl):
            raise TypeError("run_control must be EigenmodeRunControl or None")
        modeling_from_value(self.modeling)
        if resources is not None and not isinstance(resources, AedtResources):
            raise TypeError("resources must be AedtResources or None")
        self.project_name, self.design_name, self.run_control = (
            project_name,
            design_name,
            run_control,
        )
        self.resources = resources
        self.handoff_plan = None

    def add_lumped_rlc(self, treatment: LumpedRlc) -> None:
        """Apply an electrical treatment to one lumped-port sheet of the plan."""
        if not isinstance(treatment, LumpedRlc):
            raise TypeError("treatment must be LumpedRlc")
        if treatment.port_sheet_id in {item.port_sheet_id for item in self.lumped_rlcs}:
            raise ValueError("port sheet already has an RLC treatment")
        self.lumped_rlcs = (*self.lumped_rlcs, treatment)
        self._invalidate_model()

    def add_junction(self, junction: PlanarJunction) -> None:
        """Append one junction after rejecting duplicate source identities."""
        if not isinstance(junction, PlanarJunction):
            raise TypeError("junction must be PlanarJunction")
        if junction.junction_id in {item.junction_id for item in self.junctions}:
            raise ValueError("junction ID is already configured")
        self.junctions = (*self.junctions, junction)
        self._invalidate_model()

    def set_epr_request(self, request: EprAnalysisRequest | None) -> None:
        """Set the optional EPR request and discard a stale handoff plan."""
        if request is not None and not isinstance(request, EprAnalysisRequest):
            raise TypeError("request must be EprAnalysisRequest or None")
        self.epr_request = request
        self.handoff_plan = None

    def set_expression_cache_convergence(
        self, control: ExpressionCacheConvergence | ExpressionCacheConvergenceSet | None
    ) -> None:
        """Set the optional expression-cache convergence control."""
        if control is not None and not isinstance(
            control, (ExpressionCacheConvergence, ExpressionCacheConvergenceSet)
        ):
            raise TypeError(
                "control must be a unit expression control, collection, or None"
            )
        self.expression_convergence = control
        self.handoff_plan = None

    def input_summary(self) -> InputSummary:
        """Show the bound plan's normalized input and the AEDT settings without writing GDS or preparing HFSS."""
        if self._plan_snapshot is None:
            raise ValueError("input_summary() requires set_plan(plan.prepare()) first")
        data = summarize_geometry_input(
            self._plan_snapshot.geometry_input,
            prepared_stack=self._plan_snapshot.stack,
        ).data
        data["backend"] = {
            "name": "aedt",
            "aedt_version": self.aedt_version,
            "modeling": self.modeling,
            "project_name": self.project_name,
            "design_name": self.design_name,
            "run_control": (
                None if self.run_control is None else self.run_control.to_payload()
            ),
            "resources": None
            if self.resources is None
            else self.resources.to_payload(),
            "surface_epr_specs": self.surface_epr_specs,
            "junctions": [item.to_payload() for item in self.junctions],
            "lumped_rlcs": [item.to_payload() for item in self.lumped_rlcs],
            "epr_request": (
                None if self.epr_request is None else self.epr_request.to_payload()
            ),
            "expression_convergence": (
                None
                if self.expression_convergence is None
                else self.expression_convergence.to_payload()
            ),
        }
        return InputSummary(data)

    def prepare_handoff(self) -> HandoffPlan:
        """Build the plan's SGB model for AEDT, then write the HFSS Eigenmode handoff that carries it."""
        snapshot = self._plan_snapshot
        if (
            snapshot is None
            or self.output_dir is None
            or self.project_name is None
            or self.design_name is None
        ):
            raise ValueError(
                "set_plan(), output_dir, project_name, and design_name are required"
            )
        with TemporaryDirectory(prefix="scgsim-aedt-model-") as build_dir:
            model = self.prepared_model
            if model is None:
                model = prepare_sgb_model(
                    snapshot.geometry_input,
                    prepared_stack=snapshot.stack,
                    modeling=self.modeling,
                    output_dir=build_dir,
                    junctions=self.junctions,
                    lumped_rlcs=self.lumped_rlcs,
                    surface_epr_specs=self.surface_epr_specs,
                )
            plan = prepare_hfss_eigenmode_from_geometry(
                aedt_version=self.aedt_version,
                model=model,
                project_name=self.project_name,
                design_name=self.design_name,
                run_control=self.run_control,
                output_dir=self.output_dir,
                epr_request=self.epr_request,
                expression_convergence=self.expression_convergence,
                resources=self.resources,
            )
        # The build folder is gone; the model now lives in the handoff, which travels with it.
        self.prepared_model = replace(model, model_dir=plan.run_dir / SGB_MODEL_DIR)
        self.handoff_plan = plan
        return plan

    @classmethod
    def simulation_drc(
        cls,
        plan: GeometryPlan,
        *,
        modeling: Modeling,
    ) -> SimulationDrcReport:
        """Check that this simulation can build its AEDT model of ``plan``, without AEDT, and return the report.

        ``plan`` is the ``GeometryPlan`` the simulation will be given, unprepared: the check prepares it, as
        ``set_plan(plan.prepare())`` is given it, and leaves the caller's plan as it was.  Its Nets,
        instances, coupon padding, vacuum region, Surface-EPR specs, port sheets and rectangular junctions
        are used as the simulation uses them; the check adds none of its own.  A plan without ``set_vacuum_region`` is
        refused with the error ``prepare_handoff()`` raises, since AEDT's Region is the plan's vacuum box.

        The check runs the geometry half of ``prepare_handoff()`` and of the HFSS import, everything before
        AEDT is started (:mod:`scgsim.aedt.drc`): the plan's SGB model on the route ``modeling`` realizes
        (``"thin_film"`` Route A, ``"solid"`` Route C) with its STEP and import guard, the boundaries the
        import plans on the guarded faces, and the selection of every Surface-EPR row.  A step that refuses
        raises the error that step raises in the simulation, unchanged.  Every port sheet of the plan, the
        sheet of each rectangular junction included, gets a parallel inductance the geometry never reads, so
        the import treats it; a rectangular junction's sheet is cut as its ``RectangularJunctionSpec``
        derives it.  The bands of the plan's Surface-EPR specs are cut as ``prepare_handoff()`` cuts them; a
        plan without specs gets no bands.

        AEDT meshes the model itself, so the report has a geometry stage only.  The model is built in a
        temporary folder that is removed.  A pass means that model can be built and imported with these
        port treatments; it does not promise the other ``modeling``, the junctions' widths and Nets (which
        ``prepare_handoff()`` compares), nor that AEDT meshes or solves it.

        Args:
            plan: The ``GeometryPlan`` the simulation will be given, before ``prepare()``.
            modeling: ``"solid"`` or ``"thin_film"``, as ``EigenmodeSim`` takes it; there is no default.

        Raises:
            TypeError: ``plan`` is not a ``GeometryPlan``.
            ValueError: ``modeling`` is not ``"solid"`` or ``"thin_film"``, or a step of the pipeline
                refuses the plan.
        """
        plan = simulation_plan(plan, f"{cls.__name__}.simulation_drc")
        # The modeling is checked before the plan is prepared, so a refused one costs no preparation.
        modeling = modeling_from_value(modeling)
        snapshot = plan.prepare()
        return run_geometry_stage(
            lambda folder: prepare_sgb_model(
                snapshot.geometry_input,
                prepared_stack=snapshot.stack,
                modeling=modeling,
                output_dir=folder,
                lumped_rlcs=fixed_port_rlcs(snapshot),
                surface_epr_specs=snapshot.surface_epr_specs,
            ),
            check_eigenmode_model,
        )

    def resolve_results(self, run_dir: str | Path | None = None) -> ResolvedRun:
        """Resolve completed result evidence from the plan or explicit run path."""
        if run_dir is None:
            if self.handoff_plan is None:
                raise ValueError("prepare_handoff() or an explicit run_dir is required")
            run_dir = self.handoff_plan.run_dir
        return resolve_results(run_dir)

    @staticmethod
    def reanalyze_epr(
        result: EprResult,
        *,
        surface_defaults: Mapping[str, Mapping[str, Any]] | None = None,
        group_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> EprResult:
        """Reanalyze a detached EPR result with presentation-only overrides."""
        return reanalyze_epr(
            result, surface_defaults=surface_defaults, group_overrides=group_overrides
        )

    @staticmethod
    def show_epr(
        result: EprResult,
        *,
        mode: int | None = None,
        native_pass: int | None = None,
        theme: str = "light",
    ) -> Any:
        """Display one detached EPR result without running the solver."""
        return show_epr(result, mode=mode, native_pass=native_pass, theme=theme)
