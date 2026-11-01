"""Mutable Eigenmode preparation state and its ordered failure lifecycle.

Owns: the Eigenmode-only inputs (lumped ports, mode request, EPR selection), the
mesh entry point and ``config.json`` writing.  Setters shared with Electrostatic
live in ``_sim_common.py``.  This class prepares geometry and manual handoff files;
it never runs Palace.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any, ClassVar, Literal

from scgsim.geometry.models.input import VacuumRegionSpec
from scgsim.geometry.models.physical import ApproximationProfile
from scgsim.geometry.source.plan import GeometryPlanSnapshot
from scgsim.geometry.source.summary import InputSummary, summarize_geometry_input

from ..execution.handoff import HandoffPlan
from ..mesh.build import MeshBuildResult
from ..results.epr import PalaceEprResult, reanalyze_epr, show_epr
from ._sim_common import PreparationControls
from .config import (
    LayoutPortBinding,
    build_eigenmode_config,
)
from .numerical import configure_numerical_controls
from .transforms import (
    RouteAThinFilm,
    validate_non_negative_int,
    validate_nonempty_string,
    validate_positive_number,
)
from .workflow import persist_problem_files, prepare_mesh_input


@dataclass(frozen=True)
class _RequestedPort:
    name: str
    layer: str
    inductance: float


@dataclass
class EigenmodeSim(PreparationControls):
    """Prepare an SGB-authored Eigenmode mesh, config, and manual handoff."""

    _PROBLEM: ClassVar[Literal["Electrostatic", "Eigenmode"]] = "Eigenmode"

    component: Any | None = None
    stack: Mapping[str, Any] | None = None
    output_dir: Path | None = None
    airbox: dict[str, float] = field(default_factory=dict)
    route: Literal["A", "B"] = "B"
    route_a_thin_film: RouteAThinFilm | None = None
    approximation_profile: ApproximationProfile | None = None
    ports: list[_RequestedPort] = field(default_factory=list)
    surface_epr_specs: dict[str, dict[str, Any]] | None = None
    epr_request: dict[str, tuple[str, ...] | None] | None = None
    num_modes: int = 10
    target_hz: float | None = None
    eigenmode_tolerance: float = 1e-6
    save_fields: int = 0
    numerical: dict[str, Any] = field(default_factory=configure_numerical_controls)
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
        """Re-weight stored surface participation with other film assumptions, offline."""
        return reanalyze_epr(
            result, surface_defaults=surface_defaults, group_overrides=group_overrides
        )

    @staticmethod
    def show_epr(result: PalaceEprResult, *, mode: int | None = None) -> Any:
        """Plot one mode with separate surface, bulk and port axes."""
        return show_epr(result, mode=mode)

    def _invalidate_mesh(self) -> None:
        self._mesh_result = None
        self._resolved_ports = []
        self.config_path = None
        self.handoff_plan = None

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
            "indium_ground_bumps": self.indium_ground_bumps,
        }
        return InputSummary(data)

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
            _RequestedPort(
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
        """Set the Eigenmode controls: mode count, target frequency, tolerance and saved fields."""
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

    def mesh(self) -> Path:
        """Build the Palace mesh for the current inputs. Replaces any earlier mesh."""
        self._invalidate_mesh()
        self._check_mesh_inputs_set()
        if self._plan_snapshot is not None:
            return self._mesh_from_plan(self._plan_snapshot)
        return self._mesh_from_component()

    def _mesh_from_plan(self, snapshot: GeometryPlanSnapshot) -> Path:
        self._check_plan_has_no_bump_fill()
        resolved, source_records = _resolve_plan_ports(snapshot, self.ports)
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
            port_sheet_source_layers=source_records,
        )
        mesh_path = self._build_mesh(
            prepared, component=None, source_gds_bytes=snapshot.gds_bytes
        )
        self._resolved_ports = resolved
        return mesh_path

    def _mesh_from_component(self) -> Path:
        resolved, source_records = (
            _resolve_layout_ports(self.component, self.ports)
            if self.ports
            else ([], [])
        )
        prepared = prepare_mesh_input(
            component=self.component,
            stack=self.stack,
            route=self.route,
            route_a_thin_film=self.route_a_thin_film,
            approximation_profile=self.approximation_profile,
            airbox=self.airbox,
            vacuum_region=self.vacuum_region,
            indium_ground_bumps=self.indium_ground_bumps,
            port_sheet_source_layers=source_records,
        )
        mesh_path = self._build_mesh(
            prepared,
            component=prepared.component,
            indium_ground_bump_fill=prepared.indium_ground_bump_fill,
        )
        self._resolved_ports = resolved
        return mesh_path

    def write_config(self) -> Path:
        """Write config.json and the index map for the current mesh."""
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
            _check_epr_request_is_configured(request, result.index_entries)
        index_payload = {"schema_version": 1, "entries": result.index_entries}
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
                (metadata / "palace_index_map.json", index_payload),
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


def _check_epr_request_is_configured(
    request: Mapping[str, tuple[str, ...] | None],
    index_entries: Sequence[Mapping[str, Any]],
) -> None:
    """Every EPR selection must name something the config actually contains."""
    available = {
        "surface_interfaces": {
            item["metadata"]["interface_type"]
            for item in index_entries
            if item["section"] == "Boundaries.Postprocessing.Dielectric"
        },
        "bulk_domain_ids": {
            item["entry_name"]
            for item in index_entries
            if item["section"] == "Domains.Postprocessing.Energy"
        },
        "port_names": {
            item["port_name"]
            for item in index_entries
            if item["section"] == "Boundaries.LumpedPort"
        },
    }
    for name, configured in available.items():
        chosen = request[name]
        if chosen is not None and set(chosen) - configured:
            raise ValueError(f"EPR request selects an unknown {name} binding")


def _resolve_plan_ports(
    snapshot: GeometryPlanSnapshot, requested_ports: Sequence[_RequestedPort]
) -> tuple[list[LayoutPortBinding], list[dict[str, Any]]]:
    """Match requested ports to the ports the GeometryPlan already authored."""
    available_records = snapshot.stack["metadata"].get("port_sheet_source_layers", ())
    available = {record["name"]: record for record in available_records}
    if len(available) != len(available_records):
        raise ValueError("GeometryPlan port names must be unique")
    resolved: list[LayoutPortBinding] = []
    source_records: list[dict[str, Any]] = []
    for requested in requested_ports:
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
    return resolved, source_records


def _resolve_layout_ports(
    component: Any, ports: list[_RequestedPort]
) -> tuple[list[LayoutPortBinding], list[dict[str, Any]]]:
    resolved: list[LayoutPortBinding] = []
    records: list[dict[str, Any]] = []
    used_layers: set[tuple[int, int]] = set()
    component_ports = getattr(component, "ports", None)
    if component_ports is None:
        raise TypeError("component must expose authored ports.")
    for index, requested in enumerate(ports, start=1):
        try:
            port = component_ports[requested.name]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"component has no authored port {requested.name!r}."
            ) from exc
        source = _port_layer(port)
        if source in used_layers:
            raise ValueError(
                "layout_sheet ports must use distinct authored GDS layers."
            )
        used_layers.add(source)
        orientation = getattr(port, "orientation", None)
        if not isinstance(orientation, Real) or isinstance(orientation, bool):
            raise TypeError(
                f"authored port {requested.name!r} needs numeric orientation."
            )
        angle = math.radians(float(orientation))
        direction = (round(math.cos(angle), 15), round(math.sin(angle), 15), 0.0)
        length = math.hypot(direction[0], direction[1])
        if not math.isfinite(length) or length == 0.0:
            raise ValueError(f"authored port {requested.name!r} has invalid direction.")
        direction = (direction[0] / length, direction[1] / length, 0.0)
        source_layer = f"{source[0]}/{source[1]}"
        resolved.append(
            LayoutPortBinding(
                index=index,
                name=requested.name,
                target_layer=requested.layer,
                source_layer=source_layer,
                direction=direction,
                inductance=requested.inductance,
            )
        )
        records.append(
            {
                "layer": source[0],
                "datatype": source[1],
                "name": requested.name,
                "source": "palace_lumped_port_sheet",
                "port_index": index,
                "target_layer": requested.layer,
                "direction": list(direction),
                "direction_sign_convention": "gdsfactory_port_orientation_outward",
            }
        )
    return resolved, records


def _port_layer(port: Any) -> tuple[int, int]:
    raw = getattr(port, "layer", None)
    if isinstance(raw, (tuple, list)) and len(raw) == 2:
        return (int(raw[0]), int(raw[1]))
    if hasattr(raw, "layer") and hasattr(raw, "datatype"):
        return (int(raw.layer), int(raw.datatype))
    try:
        info = port.kcl.layout.get_info(int(str(raw)))
        return (int(info.layer), int(info.datatype))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("layout_sheet port requires an exact GDS layer.") from exc


__all__ = ["EigenmodeSim"]
