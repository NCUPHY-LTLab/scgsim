"""HFSS requests built on a `PreparedPlanarGeometry` instead of a GDS file.

The three requests here (body-first Driven Terminal, Eigenmode, and saved-copy
Eigenmode analysis) carry the geometry as already-normalized bodies, so they
never import a GDS or choose layers. The GDS-import requests, run controls and
the length-mesh exception stay in `hfss.py`; schema dispatch is in `parse.py`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from scgsim.aedt._data import stripped_text
from scgsim.aedt.epr.selection import surface_integral_groups
from scgsim.aedt.specs.common import (
    DRIVEN_GEOMETRY_SCHEMA_VERSION,
    EPR_ANALYSIS_SCHEMA_VERSION_V2,
    EPR_EIGENMODE_SCHEMA_VERSION_V2,
    EPR_EIGENMODE_SCHEMA_VERSION_V3,
    LOCKED_PYAEDT,
    OFFICIAL_PYAEDT_SOURCE_URL,
    REQUIRED_AEDT_VERSION,
    LumpedRlc,
    LumpedTerminalPort,
    TerminalPort,
    project_filename,
    project_name_from_payload,
)
from scgsim.aedt.specs.epr import (
    EprAnalysisRequest,
    ExpressionCacheConvergence,
    NormalizedSurfaceEprTotal,
    SavedSolution,
)
from scgsim.aedt.specs.geometry import PreparedPlanarGeometry
from scgsim.aedt.specs.hfss import (
    EigenmodeRunControl,
    FrequencySweepSpec,
    HfssRunControl,
)
from scgsim.aedt.specs.modeling import (
    Modeling,
    PhysicalLayerDeclaration,
    PhysicalRepresentationPlan,
    modeling_payload,
    physical_fields_from_payload,
    physical_layers_from_payload,
    request_modeling,
    source_plan_layers,
)


@dataclass(frozen=True)
class HfssEprSpec:
    """An Eigenmode model built from prepared geometry, optionally with EPR.

    An explicit null `run_control` declares model-only preparation; EPR analysis
    and cache requests (`epr_request`, `expression_convergence`) still require a
    complete setup. `expression_convergence` selects schema v3.
    """

    modeling: Modeling
    project_name: str
    design_name: str
    geometry: Any
    run_control: EigenmodeRunControl | None = None
    physical_layers: tuple[PhysicalLayerDeclaration, ...] = ()
    physical_representation: PhysicalRepresentationPlan | None = None
    epr_request: Any = None
    aedt_version: str = REQUIRED_AEDT_VERSION
    pyaedt_version: str = LOCKED_PYAEDT
    expression_convergence: Any = None

    @property
    def lumped_rlcs(self) -> tuple[LumpedRlc, ...]:
        """Lumped RLC treatments carried by the prepared geometry."""
        return self.geometry.lumped_rlcs

    @property
    def mode(self) -> Literal["eigenmode"]:
        """Always `"eigenmode"`; written into the payload as `mode`."""
        return "eigenmode"

    def __post_init__(self) -> None:
        request_modeling(self)
        _normalize_epr_identity(self)
        self._bind_prepared_geometry()
        self._check_run_control_and_epr()
        _validate_epr_selection(self)

    def _bind_prepared_geometry(self) -> None:
        """Check the geometry and adopt the physical layers it was prepared with."""
        if not isinstance(self.geometry, PreparedPlanarGeometry):
            raise TypeError("geometry must be PreparedPlanarGeometry")
        if self.geometry.modeling != self.modeling:
            raise ValueError("request and prepared geometry modeling differ")
        declared = source_plan_layers(self.geometry.source)
        if self.physical_layers and {
            item.physical_layer_id: item for item in self.physical_layers
        } != {item.physical_layer_id: item for item in declared}:
            raise ValueError("request physical layers differ from prepared source map")
        object.__setattr__(self, "physical_layers", declared)

    def _check_run_control_and_epr(self) -> None:
        """Validate the optional solve setup and the EPR selections that need it."""
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
            self.expression_convergence, ExpressionCacheConvergence
        ):
            raise TypeError(
                "expression_convergence must be ExpressionCacheConvergence or None"
            )

    def to_payload(self) -> dict[str, Any]:
        """Serialize as EPR schema v2, or v3 when an expression convergence is set."""
        schema_version = (
            EPR_EIGENMODE_SCHEMA_VERSION_V3
            if self.expression_convergence is not None
            else EPR_EIGENMODE_SCHEMA_VERSION_V2
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
    def from_payload(cls, payload: dict[str, Any]) -> HfssEprSpec:
        """Rebuild from an EPR v2 or v3 payload, checking the member set matches that
        version exactly.
        """
        schema = payload.get("schema_version")
        if schema not in {
            EPR_EIGENMODE_SCHEMA_VERSION_V2,
            EPR_EIGENMODE_SCHEMA_VERSION_V3,
        }:
            raise ValueError(
                f"HFSS EPR schema {schema!r} is unsupported; only "
                f"{EPR_EIGENMODE_SCHEMA_VERSION_V2!r} and "
                f"{EPR_EIGENMODE_SCHEMA_VERSION_V3!r} are supported"
            )
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
        if schema == EPR_EIGENMODE_SCHEMA_VERSION_V3:
            expected.add("expression_convergence")
            if payload.get("expression_convergence") is None:
                raise ValueError(
                    "HFSS EPR V3 requires an expression convergence configuration"
                )
        if set(payload) != expected or payload.get("mode") != "eigenmode":
            raise ValueError("HFSS EPR payload members are not canonical")
        run = payload.get("run_control")
        if run is not None and not isinstance(run, dict):
            raise TypeError("run_control must be a JSON object or null")
        project = payload.get("project")
        if not isinstance(project, dict):
            raise TypeError("project must be a JSON object")
        geometry = PreparedPlanarGeometry.from_payload(payload.get("geometry"))
        return cls(
            modeling=payload.get("modeling"),
            **physical_fields_from_payload(payload),
            project_name=project_name_from_payload(project.get("name")),
            design_name=stripped_text(project.get("design"), "project.design"),
            geometry=geometry,
            run_control=None if run is None else EigenmodeRunControl(**run),
            epr_request=(
                None
                if payload.get("epr") is None
                else EprAnalysisRequest.from_payload(payload.get("epr"))
            ),
            expression_convergence=(
                None
                if schema != EPR_EIGENMODE_SCHEMA_VERSION_V3
                else ExpressionCacheConvergence.from_payload(
                    payload.get("expression_convergence")
                )
            ),
            aedt_version=stripped_text(
                payload.get("aedt", {}).get("requested_version"),
                "aedt.requested_version",
            ),
            pyaedt_version=stripped_text(
                payload.get("pyaedt", {}).get("locked_version"),
                "pyaedt.locked_version",
            ),
        )


@dataclass(frozen=True)
class HfssEprAnalysisSpec:
    """EPR analysis of an already solved Eigenmode model.

    Works only on a saved copy of the solution: `saved_solution` must belong to
    this project, design, setup and model digest, so the analysis cannot silently
    read a different solve.
    """

    project_name: str
    design_name: str
    geometry: Any
    run_control: EigenmodeRunControl
    epr_request: Any
    saved_solution: Any
    aedt_version: str = REQUIRED_AEDT_VERSION
    pyaedt_version: str = LOCKED_PYAEDT

    @property
    def mode(self) -> Literal["eigenmode"]:
        """Always `"eigenmode"`; written into the payload as `mode`."""
        return "eigenmode"

    def __post_init__(self) -> None:
        _normalize_epr_identity(self)
        if not isinstance(self.geometry, PreparedPlanarGeometry):
            raise TypeError("geometry must be PreparedPlanarGeometry")
        if not isinstance(self.run_control, EigenmodeRunControl):
            raise TypeError("run_control must be EigenmodeRunControl")
        if not isinstance(self.saved_solution, SavedSolution):
            raise TypeError("saved_solution must be SavedSolution")
        if not isinstance(self.epr_request, EprAnalysisRequest):
            raise TypeError("epr_request must be EprAnalysisRequest")
        _validate_epr_selection(self)
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
        """Serialize as analysis schema v2, with the saved-solution record."""
        return {
            "schema_version": EPR_ANALYSIS_SCHEMA_VERSION_V2,
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
        }

    @classmethod
    def from_payload(
        cls, payload: dict[str, Any], *, base_dir: Path | None
    ) -> HfssEprAnalysisSpec:
        """Rebuild against `base_dir`, whose `saved` folder must hold the saved solution."""
        schema = payload.get("schema_version")
        if schema != EPR_ANALYSIS_SCHEMA_VERSION_V2:
            raise ValueError(
                f"HFSS EPR analysis schema {schema!r} is unsupported; only "
                f"{EPR_ANALYSIS_SCHEMA_VERSION_V2!r} is supported"
            )
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
        if set(payload) != expected or payload.get("mode") != "eigenmode":
            raise ValueError("HFSS EPR analysis payload members are not canonical")
        project = payload.get("project")
        run = payload.get("run_control")
        if not isinstance(project, dict) or not isinstance(run, dict):
            raise TypeError("project and run_control must be JSON objects")
        return cls(
            project_name=project_name_from_payload(project.get("name")),
            design_name=stripped_text(project.get("design"), "project.design"),
            geometry=PreparedPlanarGeometry.from_payload(payload.get("geometry")),
            run_control=EigenmodeRunControl(**run),
            epr_request=EprAnalysisRequest.from_payload(payload.get("epr")),
            saved_solution=SavedSolution.from_payload(
                base_dir / "saved", payload.get("saved_solution")
            ),
            aedt_version=stripped_text(
                payload.get("aedt", {}).get("requested_version"),
                "aedt.requested_version",
            ),
            pyaedt_version=stripped_text(
                payload.get("pyaedt", {}).get("locked_version"),
                "pyaedt.locked_version",
            ),
        )


def _normalize_epr_identity(spec: HfssEprSpec | HfssEprAnalysisSpec) -> None:
    """Pin the tool versions and normalize project and design names in place."""
    if (
        str(spec.aedt_version) != REQUIRED_AEDT_VERSION
        or str(spec.pyaedt_version) != LOCKED_PYAEDT
    ):
        raise ValueError("EPR V1 requires AEDT 2024.2 and PyAEDT 1.3.0")
    project = project_filename(spec.project_name, "project_name")
    object.__setattr__(spec, "project_name", project.stem)
    object.__setattr__(
        spec, "design_name", stripped_text(spec.design_name, "design_name")
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
    modes = (
        tuple(range(1, spec.run_control.num_modes + 1))
        if request.mode_indices is None
        else request.mode_indices
    )
    if not modes or any(mode > spec.run_control.num_modes for mode in modes):
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
    if convergence is None:
        return
    if convergence.mode not in modes:
        raise ValueError("expression convergence mode must be selected for EPR")
    target = convergence.target
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


@dataclass(frozen=True)
class HfssDrivenGeometrySpec:
    """A Driven Terminal model built from prepared geometry, with two ports.

    Ports are wave ports on the -X and +X Region faces, or lumped ports on
    neutral supports. The geometry must not carry junctions, lumped RLC treatments
    or EPR requests, which belong to Eigenmode. A null `run_control` prepares the
    model without a solve setup.
    """

    modeling: Modeling
    project_name: str
    design_name: str
    geometry: Any
    ports: tuple[TerminalPort | LumpedTerminalPort, ...]
    run_control: HfssRunControl | None = None
    physical_layers: tuple[PhysicalLayerDeclaration, ...] = ()
    physical_representation: PhysicalRepresentationPlan | None = None
    aedt_version: str = REQUIRED_AEDT_VERSION
    pyaedt_version: str = LOCKED_PYAEDT

    @property
    def mode(self) -> Literal["terminal"]:
        """Always `"terminal"`; written into the payload as `mode`."""
        return "terminal"

    def __post_init__(self) -> None:
        self._bind_prepared_geometry()
        request_modeling(self)
        self._normalize_identity_and_run_control()
        self._validate_ports()

    def _bind_prepared_geometry(self) -> None:
        """Check the geometry is plain Driven geometry and adopt its physical layers."""
        if not isinstance(self.geometry, PreparedPlanarGeometry):
            raise TypeError("geometry must be PreparedPlanarGeometry")
        if self.geometry.modeling != self.modeling:
            raise ValueError("request and prepared geometry modeling differ")
        if (
            self.geometry.junctions
            or self.geometry.lumped_rlcs
            or self.geometry.contributions
        ):
            raise ValueError(
                "Driven geometry cannot contain Eigenmode treatments or EPR requests"
            )
        declared = source_plan_layers(self.geometry.source)
        if self.physical_layers and self.physical_layers != declared:
            raise ValueError("request physical layers differ from prepared source map")
        object.__setattr__(self, "physical_layers", declared)

    def _normalize_identity_and_run_control(self) -> None:
        """Pin tool versions, normalize names, and type-check the optional setup."""
        if (
            self.aedt_version != REQUIRED_AEDT_VERSION
            or self.pyaedt_version != LOCKED_PYAEDT
        ):
            raise ValueError("V1 requires AEDT 2024.2 and PyAEDT 1.3.0")
        object.__setattr__(
            self,
            "project_name",
            project_filename(self.project_name, "project_name").stem,
        )
        object.__setattr__(
            self, "design_name", stripped_text(self.design_name, "design_name")
        )
        if self.run_control is not None and not isinstance(
            self.run_control, HfssRunControl
        ):
            raise TypeError("run_control must be HfssRunControl or None")

    def _validate_ports(self) -> None:
        """Check both ports against the prepared source and against each other."""
        ports = tuple(self.ports)
        if (
            tuple(port.index for port in ports) != (1, 2)
            or len({port.name for port in ports}) != 2
        ):
            raise ValueError("V1 requires unique ordered ports 1 and 2")
        source = self.geometry.source
        supports = {
            item["support_id"]: item for item in source.get("lumped_supports", ())
        }
        conductor_ids = {item["semantic_id"] for item in source["conductors"]}
        conductor_ids.update(
            item["source_semantic_id"] for item in source["conductors"]
        )
        for port in ports:
            if isinstance(port, LumpedTerminalPort):
                _check_lumped_terminal(port, supports)
            elif isinstance(port, TerminalPort):
                _check_wave_terminal(port, conductor_ids, source)
            else:
                raise TypeError(
                    "Driven Terminal requires Wave or Lumped terminal records"
                )
        if all(isinstance(port, TerminalPort) for port in ports):
            if {port.side for port in ports} != {"-X", "+X"}:
                raise ValueError("Wave V1 requires -X/+X port sides")
        references = tuple(
            port.reference_entity_ids
            if isinstance(port, LumpedTerminalPort)
            else port.reference_objects
            for port in ports
        )
        if references[0] != references[1]:
            raise ValueError(
                f"Terminal ports {ports[0].name!r} and {ports[1].name!r} "
                "must share ordered source reference conductors"
            )
        object.__setattr__(self, "ports", ports)

    def to_payload(self) -> dict[str, Any]:
        """Serialize the prepared geometry, ports and optional run control."""
        return {
            **modeling_payload(self),
            "schema_version": DRIVEN_GEOMETRY_SCHEMA_VERSION,
            "mode": self.mode,
            "aedt": {"requested_version": self.aedt_version},
            "pyaedt": {
                "locked_version": self.pyaedt_version,
                "official_source": OFFICIAL_PYAEDT_SOURCE_URL,
            },
            "project": {"name": self.project_name, "design": self.design_name},
            "geometry": self.geometry.to_payload(),
            "ports": [port.to_payload() for port in self.ports],
            "run_control": None
            if self.run_control is None
            else self.run_control.to_payload(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> HfssDrivenGeometrySpec:
        """Rebuild from a body-first Driven payload with exactly the canonical members."""
        if (
            payload.get("schema_version") != DRIVEN_GEOMETRY_SCHEMA_VERSION
            or payload.get("mode") != "terminal"
        ):
            raise ValueError("unsupported body-first Driven schema")
        expected = {
            "modeling",
            "physical_layers",
            "schema_version",
            "mode",
            "aedt",
            "pyaedt",
            "project",
            "geometry",
            "ports",
            "run_control",
        }
        if "physical_representation" in payload:
            expected.add("physical_representation")
        if set(payload) != expected:
            raise ValueError("body-first Driven payload members are not canonical")
        run = payload["run_control"]
        if run is not None:
            run = dict(run)
            run["sweep"] = FrequencySweepSpec(**run["sweep"])
        return cls(
            modeling=payload["modeling"],
            project_name=payload["project"]["name"],
            design_name=payload["project"]["design"],
            geometry=PreparedPlanarGeometry.from_payload(payload["geometry"]),
            ports=tuple(
                LumpedTerminalPort.from_payload(item)
                if item.get("kind") == "lumped"
                else TerminalPort(**item)
                for item in payload["ports"]
            ),
            run_control=None if run is None else HfssRunControl(**run),
            physical_layers=physical_layers_from_payload(payload),
            physical_representation=PhysicalRepresentationPlan.from_payload(
                payload["physical_representation"]
            ),
            aedt_version=payload["aedt"]["requested_version"],
            pyaedt_version=payload["pyaedt"]["locked_version"],
        )


def _check_lumped_terminal(
    port: LumpedTerminalPort, supports: Mapping[str, Mapping[str, Any]]
) -> None:
    """A lumped port must name the contacts its neutral support was authored with."""
    support = supports.get(port.support_id)
    if support is None:
        raise ValueError(f"terminal support {port.support_id!r} is missing")
    original = support["source"]
    if port.signal_entity_ids != tuple(
        original["terminal_a_entity_ids"]
    ) or port.reference_entity_ids != tuple(original["terminal_b_entity_ids"]):
        raise ValueError(
            "lumped terminal ownership differs from ordered source contacts"
        )


def _check_wave_terminal(
    port: TerminalPort, conductor_ids: set[str], source: Mapping[str, Any]
) -> None:
    """A wave port sits on a Region face, so that face must have zero padding."""
    if not port.signal_entity_ids or not set(port.signal_entity_ids).issubset(
        conductor_ids
    ):
        raise ValueError("body-first Wave requires explicit source Signal Entities")
    if not set(port.reference_objects).issubset(conductor_ids):
        raise ValueError("Wave references must bind source conductor Entities")
    padding = source["native_region"]["padding_um"]
    # Match the existing native Region order, without extending Wave support.
    index = {"+X": 0, "-X": 1, "+Y": 2, "-Y": 3, "+Z": 4, "-Z": 5}[port.side]
    if padding[index] != 0:
        raise ValueError("Wave port side requires zero Region padding")
