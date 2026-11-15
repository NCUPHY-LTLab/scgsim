"""HFSS Eigenmode requests on an SGB model, and the check of their EPR selection against it.

Both requests carry the :class:`PreparedSgbModel` AEDT imports (issue #67).  A request payload is a handoff
payload: its model lives in the handoff's ``model/`` folder, so a payload is decoded against the directory
that holds it (``base_dir``) and every model file is checked again there.  A payload of the native builder
(one that carries ``geometry``) is refused with the reason (issue #67 decision 6).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

from scgsim.aedt.epr.sgb_source import sgb_epr_source
from scgsim.aedt.preparation.sgb_handoff import SGB_MODEL_DIR, bind_sgb_model
from scgsim.aedt.preparation.sgb_model import PreparedSgbModel
from scgsim.aedt.specs.common import (
    EPR_ANALYSIS_SCHEMA_VERSION_V2,
    EPR_ANALYSIS_SCHEMA_VERSION_V3,
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
from scgsim.aedt.specs.hfss_common import EigenmodeRunControl
from scgsim.aedt.specs.modeling import Modeling
from scgsim.aedt.versions import normalize_aedt_version

NATIVE_PAYLOAD_REFUSAL = (
    "this HFSS Eigenmode request carries a natively built geometry; AEDT runs and reads only SGB models "
    "since issue #67, prepare it again"
)


@dataclass(frozen=True)
class HfssEprSpec:
    """One HFSS Eigenmode model built from an SGB model, with an optional solve setup and EPR request.

    An explicit null run_control declares model-only preparation; EPR analysis and cache requests require
    the complete setup.
    """

    modeling: Modeling
    project_name: str
    design_name: str
    model: PreparedSgbModel
    run_control: EigenmodeRunControl | None = None
    epr_request: EprAnalysisRequest | None = None
    aedt_version: str = field(kw_only=True)
    pyaedt_version: str = LOCKED_PYAEDT
    expression_convergence: (
        ExpressionCacheConvergence | ExpressionCacheConvergenceSet | None
    ) = None

    @property
    def lumped_rlcs(self) -> tuple[LumpedRlc, ...]:
        """The generic RLC treatments of the model's port sheets."""
        return self.model.lumped_rlcs

    @property
    def mode(self) -> Literal["eigenmode"]:
        """Return the solver mode declared by this request."""
        return "eigenmode"

    def __post_init__(self) -> None:
        _check_identity(self)
        if self.model.modeling != self.modeling:
            raise ValueError("request and prepared model modeling differ")
        if self.run_control is not None and not isinstance(
            self.run_control, EigenmodeRunControl
        ):
            raise TypeError("run_control must be EigenmodeRunControl or None")
        if self.run_control is None and (
            self.epr_request is not None or self.expression_convergence is not None
        ):
            raise ValueError("EPR analysis/cache requests require run_control")
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
        validate_epr_selection(self)

    def to_payload(self) -> dict[str, Any]:
        """Return the canonical HfssEprSpec payload; its model names the handoff's ``model/`` folder."""
        schema_version = (
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
        return {
            "schema_version": schema_version,
            "mode": self.mode,
            "modeling": self.modeling,
            "aedt": {"requested_version": self.aedt_version},
            "pyaedt": {
                "locked_version": self.pyaedt_version,
                "official_source": OFFICIAL_PYAEDT_SOURCE_URL,
            },
            "project": {"name": self.project_name, "design": self.design_name},
            "run_control": (
                None if self.run_control is None else self.run_control.to_payload()
            ),
            "model": model_payload(self.model),
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
        cls, payload: dict[str, Any], *, base_dir: Path | None
    ) -> HfssEprSpec:
        """Decode a HfssEprSpec from its canonical payload, binding its model under ``base_dir``."""
        schema = _validate_hfss_epr_payload(payload)
        run = payload.get("run_control")
        if run is not None and not isinstance(run, dict):
            raise TypeError("run_control must be a JSON object or null")
        project = payload.get("project")
        if not isinstance(project, dict):
            raise TypeError("project must be a JSON object")
        return cls(
            modeling=cast(Modeling, payload.get("modeling")),
            project_name=project_name_from_payload(project.get("name")),
            design_name=text_value(project.get("design"), "project.design"),
            model=_bound_model(payload, base_dir),
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
        )


def _validate_hfss_epr_payload(payload: dict[str, Any]) -> str:
    schema = payload.get("schema_version")
    supported = {
        EPR_EIGENMODE_SCHEMA_VERSION_V2,
        EPR_EIGENMODE_SCHEMA_VERSION_V3,
        EPR_EIGENMODE_SCHEMA_VERSION_V4,
    }
    if schema not in supported:
        raise ValueError("unsupported HFSS EPR schema")
    if "geometry" in payload:
        raise ValueError(NATIVE_PAYLOAD_REFUSAL)
    expected = {
        "schema_version",
        "mode",
        "modeling",
        "aedt",
        "pyaedt",
        "project",
        "run_control",
        "model",
        "epr",
    }
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
    """One saved-copy-only Eigenmode EPR analysis of a solved SGB model.

    ``surface_field_selectors`` are keyed by sidecar Surface-EPR group: ``surface`` reads a row's field on
    its faces, ``adjacent`` just off them.
    """

    project_name: str
    design_name: str
    model: PreparedSgbModel
    run_control: EigenmodeRunControl
    epr_request: EprAnalysisRequest
    saved_solution: SavedSolution
    aedt_version: str = field(kw_only=True)
    pyaedt_version: str = LOCKED_PYAEDT
    surface_field_selectors: Mapping[str, Literal["surface", "adjacent"]] | None = (
        field(default=None, kw_only=True)
    )

    @property
    def mode(self) -> Literal["eigenmode"]:
        """Return the solver mode declared by this request."""
        return "eigenmode"

    def __post_init__(self) -> None:
        _check_identity(self)
        if not isinstance(self.run_control, EigenmodeRunControl):
            raise TypeError("run_control must be EigenmodeRunControl")
        if not isinstance(self.saved_solution, SavedSolution):
            raise TypeError("saved_solution must be SavedSolution")
        if not isinstance(self.epr_request, EprAnalysisRequest):
            raise TypeError("epr_request must be EprAnalysisRequest")
        groups = validate_epr_selection(self)
        requested = self.surface_field_selectors
        if requested is None:
            requested = {}
        if not isinstance(requested, Mapping) or any(
            not isinstance(key, str) for key in requested
        ):
            raise TypeError(
                "surface_field_selectors must be a Surface-EPR group mapping or None"
            )
        selectors = dict(requested)
        if any(value not in ("surface", "adjacent") for value in selectors.values()):
            raise ValueError("surface field selectors must be 'surface' or 'adjacent'")
        unknown = sorted(set(selectors) - groups)
        if unknown:
            raise ValueError(
                f"surface field selectors name groups no selected Surface-EPR row carries: {unknown}"
            )
        object.__setattr__(self, "surface_field_selectors", MappingProxyType(selectors))
        if self.saved_solution.project_path.stem != self.project_name:
            raise ValueError("saved solution project does not match project_name")
        identity = self.saved_solution.identity
        expected = {
            "model_source_sha256": self.model.identity_sha256,
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
                EPR_ANALYSIS_SCHEMA_VERSION_V3
                if self.surface_field_selectors
                else EPR_ANALYSIS_SCHEMA_VERSION_V2
            ),
            "mode": self.mode,
            "aedt": {"requested_version": self.aedt_version},
            "pyaedt": {
                "locked_version": self.pyaedt_version,
                "official_source": OFFICIAL_PYAEDT_SOURCE_URL,
            },
            "project": {"name": self.project_name, "design": self.design_name},
            "run_control": self.run_control.to_payload(),
            "model": model_payload(self.model),
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
        """Decode a HfssEprAnalysisSpec from its canonical payload, binding its model under ``base_dir``."""
        schema = payload.get("schema_version")
        if schema not in {
            EPR_ANALYSIS_SCHEMA_VERSION_V2,
            EPR_ANALYSIS_SCHEMA_VERSION_V3,
        }:
            raise ValueError("unsupported HFSS EPR analysis schema")
        if base_dir is None:
            raise ValueError("HFSS EPR analysis requires a bound base directory")
        if "geometry" in payload:
            raise ValueError(NATIVE_PAYLOAD_REFUSAL)
        expected = {
            "schema_version",
            "mode",
            "aedt",
            "pyaedt",
            "project",
            "run_control",
            "model",
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
            model=_bound_model(payload, base_dir),
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
            surface_field_selectors=(
                payload["surface_field_selectors"]
                if schema == EPR_ANALYSIS_SCHEMA_VERSION_V3
                else None
            ),
        )


def model_payload(model: PreparedSgbModel) -> dict[str, Any]:
    """The model as a request payload names it: in the handoff's own ``model/`` folder."""
    payload = model.to_payload()
    payload["model_dir"] = SGB_MODEL_DIR
    return payload


def _bound_model(payload: Mapping[str, Any], base_dir: Path | None) -> PreparedSgbModel:
    if base_dir is None:
        raise ValueError(
            "an HFSS Eigenmode request is read against the handoff directory that holds its model"
        )
    return bind_sgb_model(payload.get("model") or {}, Path(base_dir))


def _check_identity(spec: HfssEprSpec | HfssEprAnalysisSpec) -> None:
    object.__setattr__(spec, "aedt_version", normalize_aedt_version(spec.aedt_version))
    if spec.pyaedt_version != LOCKED_PYAEDT:
        raise ValueError("EPR V1 requires PyAEDT 1.3.0")
    project = project_filename(spec.project_name, "project_name")
    object.__setattr__(spec, "project_name", project.stem)
    object.__setattr__(spec, "design_name", text_value(spec.design_name, "design_name"))
    if not isinstance(spec.model, PreparedSgbModel):
        raise TypeError("model must be PreparedSgbModel")


def validate_epr_selection(spec: HfssEprSpec | HfssEprAnalysisSpec) -> set[str]:
    """Check the request's selection against the model; return the Surface-EPR groups its rows carry.

    The modes must be configured Eigenmodes, the junctions and bulk domains the model's, and every
    normalized-surface convergence target must select a Surface-EPR entry.  Reading the model's EPR
    source also refuses an unknown or empty Surface-EPR group and a film without its thickness.
    """
    request = spec.epr_request
    if request is None:
        if getattr(spec, "expression_convergence", None) is not None:
            raise ValueError("expression convergence requires an EPR analysis request")
        return set()
    run_control = cast(EigenmodeRunControl, spec.run_control)
    modes = (
        tuple(range(1, run_control.num_modes + 1))
        if request.mode_indices is None
        else request.mode_indices
    )
    if not modes or any(mode > run_control.num_modes for mode in modes):
        raise ValueError("EPR mode_indices must select configured Eigenmodes")
    source = sgb_epr_source(spec.model, surface_epr_groups=request.surface_epr_groups)
    if request.junction_ids is not None and not set(request.junction_ids) <= set(
        source.junctions
    ):
        raise ValueError("EPR request selects an unknown junction")
    if request.bulk_domain_ids is not None and not set(request.bulk_domain_ids) <= set(
        source.domains
    ):
        raise ValueError("EPR request selects an unknown bulk domain")
    for control in iter_expression_convergence(
        getattr(spec, "expression_convergence", None)
    ):
        if control.mode not in modes:
            raise ValueError("expression convergence mode must be selected for EPR")
        target = control.target
        if isinstance(target, NormalizedSurfaceEprTotal) and not any(
            entry.interface_type == target.interface_kind
            and entry.evaluation_kind == target.evaluation_kind
            and entry.margin_um == float(target.margin_um)
            for entry in source.entries
        ):
            raise ValueError(
                "normalized surface convergence target selects no Surface-EPR entry"
            )
    return {row.group for row in source.rows.values()}
