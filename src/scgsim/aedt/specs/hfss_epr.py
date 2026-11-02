"""HFSS Eigenmode EPR requests and selection validation."""

from __future__ import annotations

from collections.abc import (
    Mapping,
)
from dataclasses import (
    dataclass,
    field,
)
from pathlib import (
    Path,
)
from types import (
    MappingProxyType,
)
from typing import (
    Any,
    Literal,
    cast,
)

from scgsim.aedt.epr.selection import surface_integral_groups
from scgsim.aedt.specs.common import (
    EPR_ANALYSIS_SCHEMA_VERSION,
    EPR_ANALYSIS_SCHEMA_VERSION_V2,
    EPR_ANALYSIS_SCHEMA_VERSION_V3,
    EPR_EIGENMODE_SCHEMA_VERSION,
    EPR_EIGENMODE_SCHEMA_VERSION_V2,
    EPR_EIGENMODE_SCHEMA_VERSION_V3,
    EPR_EIGENMODE_SCHEMA_VERSION_V4,
    LOCKED_PYAEDT,
    OFFICIAL_PYAEDT_SOURCE_URL,
    LumpedRlc,
    project_filename,
    project_name_from_payload,
    text_value,
)
from scgsim.aedt.specs.epr import (
    EprAnalysisRequest,
    ExpressionCacheConvergence,
    ExpressionCacheConvergenceSet,
    NormalizedSurfaceEprTotal,
    SavedSolution,
    iter_expression_convergence,
)
from scgsim.aedt.specs.geometry import PreparedPlanarGeometry
from scgsim.aedt.specs.hfss_common import EigenmodeRunControl
from scgsim.aedt.specs.modeling import (
    Modeling,
    PhysicalLayerDeclaration,
    PhysicalLayerSpec,
    PhysicalRepresentationPlan,
    modeling_payload,
    physical_layers_from_payload,
    request_modeling,
    source_plan_layers,
)
from scgsim.aedt.versions import (
    normalize_aedt_version,
)


@dataclass(frozen=True)
class HfssEprSpec:
    """One body-first Eigenmode model with optional solve setup.

    An explicit null run_control declares model-only preparation; EPR analysis
    and cache requests retain their complete setup requirement.
    """

    modeling: Modeling
    project_name: str
    design_name: str
    geometry: Any
    run_control: EigenmodeRunControl | None = None
    physical_layers: tuple[PhysicalLayerDeclaration | PhysicalLayerSpec, ...] = ()
    physical_representation: PhysicalRepresentationPlan | None = None
    _historical_physical: bool = False
    _historical_modeling: bool = False
    epr_request: Any = None
    aedt_version: str = field(kw_only=True)
    pyaedt_version: str = LOCKED_PYAEDT
    _legacy_payload: bool = False
    expression_convergence: Any = None

    @property
    def historical_physical(self) -> bool:
        """Report whether this request decodes historical physical data."""
        return self._historical_physical

    @property
    def historical_modeling(self) -> bool:
        """Report whether this request decodes historical modeling data."""
        return self._historical_modeling

    @property
    def legacy_payload(self) -> bool:
        """Report whether this request came from the legacy payload schema."""
        return self._legacy_payload

    @property
    def lumped_rlcs(self) -> tuple[LumpedRlc, ...]:
        """Return lumped RLC treatments attached to the geometry."""
        return self.geometry.lumped_rlcs

    @property
    def mode(self) -> Literal["eigenmode"]:
        """Return the solver mode declared by this request."""
        return "eigenmode"

    def __post_init__(self) -> None:
        request_modeling(self)
        object.__setattr__(
            self, "aedt_version", normalize_aedt_version(self.aedt_version)
        )
        if self.pyaedt_version != LOCKED_PYAEDT:
            raise ValueError("EPR V1 requires PyAEDT 1.3.0")
        project = project_filename(self.project_name, "project_name")
        object.__setattr__(self, "project_name", project.stem)
        object.__setattr__(
            self, "design_name", text_value(self.design_name, "design_name")
        )
        if not isinstance(self.geometry, PreparedPlanarGeometry):
            raise TypeError("geometry must be PreparedPlanarGeometry")
        if not self._historical_modeling and self.geometry.modeling != self.modeling:
            raise ValueError("request and prepared geometry modeling differ")
        if not self._historical_modeling:
            declared = source_plan_layers(self.geometry.source)
            if self.physical_layers and {
                item.physical_layer_id: item for item in self.physical_layers
            } != {item.physical_layer_id: item for item in declared}:
                raise ValueError(
                    "request physical layers differ from prepared source map"
                )
            object.__setattr__(self, "physical_layers", declared)

        if self.run_control is not None and not isinstance(
            self.run_control, EigenmodeRunControl
        ):
            raise TypeError("run_control must be EigenmodeRunControl or None")
        if self.run_control is None and (
            self.epr_request is not None
            or self.expression_convergence is not None
            or self._legacy_payload
            or self._historical_modeling
        ):
            raise ValueError(
                "EPR analysis/cache and historical requests require run_control"
            )
        if self.epr_request is not None and not isinstance(
            self.epr_request, EprAnalysisRequest
        ):
            raise TypeError("epr_request must be EprAnalysisRequest or None")
        if self.expression_convergence is not None and not isinstance(
            self.expression_convergence,
            (ExpressionCacheConvergence, ExpressionCacheConvergenceSet),
        ):
            raise TypeError(
                "expression_convergence must be a unit control, collection, or None"
            )
        if self._legacy_payload and self.expression_convergence is not None:
            raise ValueError("legacy EPR payload cannot add expression convergence")
        _validate_epr_selection(self)

    def to_payload(self) -> dict[str, Any]:
        """Return the canonical HfssEprSpec payload."""
        schema_version = (
            EPR_EIGENMODE_SCHEMA_VERSION
            if self._legacy_payload
            else (
                (
                    EPR_EIGENMODE_SCHEMA_VERSION_V4
                    if isinstance(
                        self.expression_convergence, ExpressionCacheConvergenceSet
                    )
                    else EPR_EIGENMODE_SCHEMA_VERSION_V3
                )
                if self.expression_convergence is not None
                else EPR_EIGENMODE_SCHEMA_VERSION_V2
            )
        )
        return {
            **modeling_payload(self),
            "schema_version": schema_version,
            "mode": self.mode,
            "aedt": {"requested_version": self.aedt_version},
            "pyaedt": {
                "locked_version": self.pyaedt_version,
                "official_source": OFFICIAL_PYAEDT_SOURCE_URL,
            },
            "project": {"name": self.project_name, "design": self.design_name},
            "run_control": (
                None if self.run_control is None else self.run_control.to_payload()
            ),
            "geometry": self.geometry.to_payload(),
            "epr": (
                None if self.epr_request is None else self.epr_request.to_payload()
            ),
            **(
                {"expression_convergence": self.expression_convergence.to_payload()}
                if self.expression_convergence is not None
                else {}
            ),
        }

    @classmethod
    def from_payload(
        cls, payload: dict[str, Any], *, allow_historical_modeling: bool = False
    ) -> HfssEprSpec:
        """Decode a HfssEprSpec from its canonical payload."""
        schema = _validate_hfss_epr_payload(payload)
        run = payload.get("run_control")
        if run is not None and not isinstance(run, dict):
            raise TypeError("run_control must be a JSON object or null")
        project = payload.get("project")
        if not isinstance(project, dict):
            raise TypeError("project must be a JSON object")
        geometry = PreparedPlanarGeometry.from_payload(payload.get("geometry"))
        return cls(
            modeling=cast(Modeling, payload.get("modeling")),
            physical_layers=physical_layers_from_payload(
                payload, allow_historical=allow_historical_modeling
            ),
            physical_representation=(
                PhysicalRepresentationPlan.from_payload(
                    payload["physical_representation"]
                )
                if "physical_representation" in payload
                else None
            ),
            _historical_physical=allow_historical_modeling
            and "physical_representation" not in payload,
            _historical_modeling=allow_historical_modeling
            and "modeling" not in payload,
            project_name=project_name_from_payload(project.get("name")),
            design_name=text_value(project.get("design"), "project.design"),
            geometry=geometry,
            run_control=None if run is None else EigenmodeRunControl(**run),
            epr_request=(
                None
                if payload.get("epr") is None
                else EprAnalysisRequest.from_payload(payload.get("epr"))
            ),
            expression_convergence=(
                None
                if schema
                not in {
                    EPR_EIGENMODE_SCHEMA_VERSION_V3,
                    EPR_EIGENMODE_SCHEMA_VERSION_V4,
                }
                else (
                    ExpressionCacheConvergenceSet
                    if schema == EPR_EIGENMODE_SCHEMA_VERSION_V4
                    else ExpressionCacheConvergence
                ).from_payload(payload.get("expression_convergence"))
            ),
            aedt_version=payload.get("aedt", {}).get("requested_version"),
            pyaedt_version=text_value(
                payload.get("pyaedt", {}).get("locked_version"),
                "pyaedt.locked_version",
            ),
            _legacy_payload=schema == EPR_EIGENMODE_SCHEMA_VERSION,
        )


def _validate_hfss_epr_payload(payload: dict[str, Any]) -> str:
    schema = payload.get("schema_version")
    supported = {
        EPR_EIGENMODE_SCHEMA_VERSION,
        EPR_EIGENMODE_SCHEMA_VERSION_V2,
        EPR_EIGENMODE_SCHEMA_VERSION_V3,
        EPR_EIGENMODE_SCHEMA_VERSION_V4,
    }
    if schema not in supported:
        raise ValueError("unsupported HFSS EPR schema")
    expected = {
        "schema_version",
        "mode",
        "aedt",
        "pyaedt",
        "project",
        "run_control",
        "geometry",
        "epr",
    }
    if "modeling" in payload:
        expected.update({"modeling", "physical_layers"})
    if "physical_representation" in payload:
        expected.add("physical_representation")
    if schema in {EPR_EIGENMODE_SCHEMA_VERSION_V3, EPR_EIGENMODE_SCHEMA_VERSION_V4}:
        expected.add("expression_convergence")
        if payload.get("expression_convergence") is None:
            raise ValueError(
                "HFSS EPR V3/V4 requires an expression convergence configuration"
            )
    if set(payload) != expected or payload.get("mode") != "eigenmode":
        raise ValueError("HFSS EPR payload members are not canonical")
    return cast(str, schema)


@dataclass(frozen=True)
class HfssEprAnalysisSpec:
    """One saved-copy-only Eigenmode EPR analysis request."""

    project_name: str
    design_name: str
    geometry: Any
    run_control: EigenmodeRunControl
    epr_request: Any
    saved_solution: Any
    aedt_version: str = field(kw_only=True)
    pyaedt_version: str = LOCKED_PYAEDT
    _legacy_payload: bool = False
    surface_field_selectors: Mapping[str, Literal["surface", "adjacent"]] | None = (
        field(default=None, kw_only=True)
    )

    @property
    def legacy_payload(self) -> bool:
        """Report whether this analysis came from the legacy payload schema."""
        return self._legacy_payload

    @property
    def mode(self) -> Literal["eigenmode"]:
        """Return the solver mode declared by this request."""
        return "eigenmode"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "aedt_version", normalize_aedt_version(self.aedt_version)
        )
        if self.pyaedt_version != LOCKED_PYAEDT:
            raise ValueError("EPR V1 requires PyAEDT 1.3.0")
        project = project_filename(self.project_name, "project_name")
        object.__setattr__(self, "project_name", project.stem)
        object.__setattr__(
            self, "design_name", text_value(self.design_name, "design_name")
        )
        if not isinstance(self.geometry, PreparedPlanarGeometry):
            raise TypeError("geometry must be PreparedPlanarGeometry")
        if not isinstance(self.run_control, EigenmodeRunControl):
            raise TypeError("run_control must be EigenmodeRunControl")
        if not isinstance(self.saved_solution, SavedSolution):
            raise TypeError("saved_solution must be SavedSolution")
        if not isinstance(self.epr_request, EprAnalysisRequest):
            raise TypeError("epr_request must be EprAnalysisRequest")
        _validate_epr_selection(self)
        requested = self.surface_field_selectors
        if requested is None:
            requested = {}
        if not isinstance(requested, Mapping) or any(
            not isinstance(key, str) for key in requested
        ):
            raise TypeError(
                "surface_field_selectors must be a contribution-ID mapping or None"
            )
        selectors = dict(requested)
        if any(value not in ("surface", "adjacent") for value in selectors.values()):
            raise ValueError("surface field selectors must be 'surface' or 'adjacent'")
        selected = (
            {item.contribution_id for item in self.geometry.contributions}
            if self.epr_request.surface_contribution_ids is None
            else set(self.epr_request.surface_contribution_ids)
        )
        if not set(selectors) <= selected:
            raise ValueError(
                "surface field selectors reference unknown or unselected contributions"
            )
        if self._legacy_payload and selectors:
            raise ValueError(
                "legacy saved-analysis payload cannot add surface field selectors"
            )
        object.__setattr__(self, "surface_field_selectors", MappingProxyType(selectors))
        if self.saved_solution.project_path.stem != self.project_name:
            raise ValueError("saved solution project does not match project_name")
        identity = self.saved_solution.identity
        expected = {
            "model_source_sha256": self.geometry.model_sha256,
            "project_name": self.project_name,
            "design_name": self.design_name,
            "setup_name": self.run_control.setup_name,
        }
        if any(identity.get(name) != value for name, value in expected.items()):
            raise ValueError(
                "saved solution identity does not match the analysis model"
            )

    def to_payload(self) -> dict[str, Any]:
        """Return the canonical HfssEprAnalysisSpec payload."""
        return {
            "schema_version": (
                EPR_ANALYSIS_SCHEMA_VERSION
                if self._legacy_payload
                else (
                    EPR_ANALYSIS_SCHEMA_VERSION_V3
                    if self.surface_field_selectors
                    else EPR_ANALYSIS_SCHEMA_VERSION_V2
                )
            ),
            "mode": self.mode,
            "aedt": {"requested_version": self.aedt_version},
            "pyaedt": {
                "locked_version": self.pyaedt_version,
                "official_source": OFFICIAL_PYAEDT_SOURCE_URL,
            },
            "project": {"name": self.project_name, "design": self.design_name},
            "run_control": self.run_control.to_payload(),
            "geometry": self.geometry.to_payload(),
            "epr": self.epr_request.to_payload(),
            "saved_solution": self.saved_solution.to_payload(),
            **(
                {"surface_field_selectors": dict(self.surface_field_selectors)}
                if self.surface_field_selectors
                else {}
            ),
        }

    @classmethod
    def from_payload(
        cls, payload: dict[str, Any], *, base_dir: Path | None
    ) -> HfssEprAnalysisSpec:
        """Decode a HfssEprAnalysisSpec from its canonical payload."""
        schema = payload.get("schema_version")
        if schema not in {
            EPR_ANALYSIS_SCHEMA_VERSION,
            EPR_ANALYSIS_SCHEMA_VERSION_V2,
            EPR_ANALYSIS_SCHEMA_VERSION_V3,
        }:
            raise ValueError("unsupported HFSS EPR analysis schema")
        if base_dir is None:
            raise ValueError("HFSS EPR analysis requires a bound base directory")
        expected = {
            "schema_version",
            "mode",
            "aedt",
            "pyaedt",
            "project",
            "run_control",
            "geometry",
            "epr",
            "saved_solution",
        }
        if schema == EPR_ANALYSIS_SCHEMA_VERSION_V3:
            expected.add("surface_field_selectors")
            if (
                not isinstance(payload.get("surface_field_selectors"), Mapping)
                or not payload["surface_field_selectors"]
            ):
                raise ValueError(
                    "saved-analysis v3 requires nonempty surface field selectors"
                )
        if set(payload) != expected or payload.get("mode") != "eigenmode":
            raise ValueError("HFSS EPR analysis payload members are not canonical")
        project = payload.get("project")
        run = payload.get("run_control")
        if not isinstance(project, dict) or not isinstance(run, dict):
            raise TypeError("project and run_control must be JSON objects")
        return cls(
            project_name=project_name_from_payload(project.get("name")),
            design_name=text_value(project.get("design"), "project.design"),
            geometry=PreparedPlanarGeometry.from_payload(payload.get("geometry")),
            run_control=EigenmodeRunControl(**run),
            epr_request=EprAnalysisRequest.from_payload(payload.get("epr")),
            saved_solution=SavedSolution.from_payload(
                base_dir / "saved", payload.get("saved_solution")
            ),
            aedt_version=payload.get("aedt", {}).get("requested_version"),
            pyaedt_version=text_value(
                payload.get("pyaedt", {}).get("locked_version"),
                "pyaedt.locked_version",
            ),
            _legacy_payload=schema == EPR_ANALYSIS_SCHEMA_VERSION,
            surface_field_selectors=(
                payload["surface_field_selectors"]
                if schema == EPR_ANALYSIS_SCHEMA_VERSION_V3
                else None
            ),
        )


def _validate_epr_selection(spec: HfssEprSpec | HfssEprAnalysisSpec) -> None:
    request = spec.epr_request
    if request is None:
        if getattr(spec, "expression_convergence", None) is not None:
            raise ValueError("expression convergence requires an EPR analysis request")
        return
    if any(
        item.field_side == "sidewall" for item in spec.geometry.contributions
    ) or any(
        item["contribution"]["side"] == "sidewall"
        for item in spec.geometry.surface_bindings
    ):
        raise ValueError(
            "sidewall Surface-EPR is excluded; reprepare a horizontal-only "
            "handoff before EPR execution"
        )
    run_control = cast(EigenmodeRunControl, spec.run_control)
    modes = (
        tuple(range(1, run_control.num_modes + 1))
        if request.mode_indices is None
        else request.mode_indices
    )
    if not modes or any(mode > run_control.num_modes for mode in modes):
        raise ValueError("EPR mode_indices must select configured Eigenmodes")
    surface_ids = {item.contribution_id for item in spec.geometry.contributions}
    selected_surfaces = (
        surface_ids
        if request.surface_contribution_ids is None
        else set(request.surface_contribution_ids)
    )
    if not selected_surfaces <= surface_ids:
        raise ValueError("EPR request selects an unknown surface contribution")
    junction_ids = {item.junction_id for item in spec.geometry.junctions}
    selected_junctions = (
        junction_ids if request.junction_ids is None else set(request.junction_ids)
    )
    if not selected_junctions <= junction_ids:
        raise ValueError("EPR request selects an unknown junction")
    source_domains = {
        item["semantic_id"] for item in spec.geometry.source["solution_regions"]
    }
    native_region = spec.geometry.source.get("native_region")
    if native_region is not None:
        source_domains.difference_update(native_region["logical_vacuum_ids"])
        source_domains.add("Region")
    if (
        request.bulk_domain_ids is not None
        and not set(request.bulk_domain_ids) <= source_domains
    ):
        raise ValueError("EPR request selects an unknown bulk domain")
    groups = surface_integral_groups(spec.geometry, request)
    convergence = getattr(spec, "expression_convergence", None)
    for control in iter_expression_convergence(convergence):
        if control.mode not in modes:
            raise ValueError("expression convergence mode must be selected for EPR")
        target = control.target
        if isinstance(target, NormalizedSurfaceEprTotal):
            selected_groups = [
                group
                for group in groups
                if group["interface_kind"] == target.interface_kind
                and group.get("evaluation_kind", "requested_margin")
                == target.evaluation_kind
                and float(group["margin_um"]) == target.margin_um
            ]
            if not selected_groups:
                raise ValueError(
                    "normalized surface convergence target selects no prepared owner groups"
                )
