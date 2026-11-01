"""HFSS requests that import GDS pieces, plus the run controls every HFSS request shares.

Owns the Driven (two-port, Terminal or Modal) and port-free Eigenmode requests
that name a GDS file with layer imports and object bindings, the adaptive and
sweep controls, and the uniform-CPW length-mesh exception. Requests built on an
already-prepared body geometry are in `hfss_prepared.py`; schema dispatch is in
`parse.py`. Nothing here talks to AEDT.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from numbers import Number
from pathlib import Path
from typing import Any, Literal

from scgsim.aedt._data import stripped_text
from scgsim.aedt.specs.common import (
    EIGENMODE_SCHEMA_VERSION,
    LOCKED_PYAEDT,
    OFFICIAL_PYAEDT_SOURCE_URL,
    POINT_COUNT,
    REQUIRED_AEDT_VERSION,
    SCHEMA_VERSION,
    HfssDrivenMode,
    LayerImport,
    ModalPort,
    ObjectBinding,
    PdkMaterial,
    TerminalPort,
    adaptive_controls,
    checked_region_padding,
    coerced_finite_number,
    normalize_gds_spec,
    project_name_from_payload,
)
from scgsim.aedt.specs.modeling import (
    Modeling,
    PhysicalLayerDeclaration,
    PhysicalRepresentationPlan,
    effective_record,
    modeling_payload,
    physical_fields_from_payload,
    request_modeling,
    source_record_payload,
    verify_effective_records,
)


def _direct_hfss_enclosure(spec: Any) -> dict[str, Any]:
    """Project selected GDS geometry's original full envelope exactly once."""
    import gdstk

    library = gdstk.read_gds(spec.gds_path)
    scale = library.unit * 1e6
    selected = []
    for record in spec.effective_layer_imports:
        polygons = tuple(
            polygon
            for cell in library.top_level()
            for polygon in cell.get_polygons(
                layer=record["native_import_layer"], datatype=0
            )
        )
        if polygons:
            selected.append(
                (
                    record,
                    tuple(point for polygon in polygons for point in polygon.points),
                )
            )
    if not selected:
        raise ValueError("selected HFSS import layers have no source polygons")
    xy = {
        f"{axis}_{edge}_um": operation(
            float(point[index]) * scale for _, points in selected for point in points
        )
        for index, axis in enumerate(("x", "y"))
        for edge, operation in (("min", min), ("max", max))
    }
    source_bounds = {
        **xy,
        "z_min_um": min(record["z_min_um"] for record, _ in selected),
        "z_max_um": max(record["z_max_um"] for record, _ in selected),
    }
    effective_bounds = {
        **xy,
        "z_min_um": min(record["effective_z_min_um"] for record, _ in selected),
        "z_max_um": max(record["effective_z_max_um"] for record, _ in selected),
    }
    keys = (
        "x_plus_um",
        "x_minus_um",
        "y_plus_um",
        "y_minus_um",
        "z_plus_um",
        "z_minus_um",
    )
    return spec.physical_representation.project_enclosure(
        source_body_bounds_um=source_bounds,
        effective_body_bounds_um=effective_bounds,
        source_padding_um=dict(zip(keys, spec.region_padding_um, strict=True)),
    )


def _materials_from_payload(payload: Mapping[str, Any]) -> dict[str, PdkMaterial]:
    """Decode the `materials` object, one PDK material per string material_id."""
    raw_materials = payload.get("materials")
    if not isinstance(raw_materials, dict):
        raise TypeError("materials must be a JSON object")
    materials: dict[str, PdkMaterial] = {}
    for material_id, item in raw_materials.items():
        if not isinstance(material_id, str) or not isinstance(item, dict):
            raise TypeError("materials must map string material_id values to objects")
        materials[material_id] = PdkMaterial(**item)
    return materials


def _run_control_from_payload(
    run: Mapping[str, Any], sweep: FrequencySweepSpec
) -> HfssRunControl:
    """Decode a Driven run control; an absent adaptive control takes its default."""
    return HfssRunControl(
        stripped_text(run.get("setup_name"), "setup_name"),
        stripped_text(run.get("sweep_name"), "sweep_name"),
        sweep,
        maximum_passes=run.get("maximum_passes", 99),
        minimum_passes=run.get("minimum_passes", 1),
        minimum_converged_passes=run.get("minimum_converged_passes", 1),
        percent_refinement=run.get("percent_refinement", 30.0),
        maximum_delta_s=run.get("maximum_delta_s", 0.02),
    )


def _effective_layer_imports(spec: Any) -> tuple[dict[str, Any], ...]:
    """Pair each requested layer import with its native layer number and effective Z.

    AEDT numbers imported layers 1..N in ascending (layer, datatype) order, so
    that order is the native import layer. A layer holding a substrate binding
    whose object is named `<layer_name>_...` is the retained domain, which the
    Thin Film approximation must not collapse to a sheet.
    """
    mapping = request_modeling(spec)
    pairs = {
        pair: index + 1
        for index, pair in enumerate(
            sorted((item.layer, item.datatype) for item in spec.layer_imports)
        )
    }
    domain_pairs = {
        (item.layer, item.datatype)
        for item in spec.layer_imports
        if any(
            binding.role == "substrate"
            and binding.layer == item.layer
            and binding.object_name.startswith(f"{item.layer_name}_")
            for binding in spec.object_bindings
        )
    }
    return tuple(
        {
            **effective_record(
                item,
                mapping,
                retained_domain=(item.layer, item.datatype) in domain_pairs,
            ),
            "native_import_layer": pairs[(item.layer, item.datatype)],
        }
        for item in spec.layer_imports
    )


@dataclass(frozen=True)
class FrequencySweepSpec:
    """One HFSS Fast frequency sweep from `start_ghz` to `stop_ghz`.

    `points` is the number of output frequency points, an exact whole number.
    """

    start_ghz: float
    stop_ghz: float
    points: int = POINT_COUNT

    def __post_init__(self) -> None:
        start = coerced_finite_number(self.start_ghz, "start_ghz")
        stop = coerced_finite_number(self.stop_ghz, "stop_ghz")
        if start <= 0 or stop <= start:
            raise ValueError("frequency sweep requires 0 < start_ghz < stop_ghz")
        if isinstance(self.points, bool) or not isinstance(self.points, Number):
            raise ValueError("points must be an exact whole-number numeric value")
        try:
            native_points = int(self.points)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "points must be an exact whole-number numeric value"
            ) from exc
        if self.points != native_points:
            raise ValueError("points must be an exact whole-number numeric value")
        object.__setattr__(self, "start_ghz", start)
        object.__setattr__(self, "stop_ghz", stop)

    def to_payload(self) -> dict[str, Any]:
        """Serialize the sweep; `points` keeps the authored numeric type so historical
        payload identity is unchanged.
        """
        return {
            "start_ghz": self.start_ghz,
            "stop_ghz": self.stop_ghz,
            # Preserve authored count representation in historical payload identity.
            "points": self.points,
        }


@dataclass(frozen=True)
class HfssRunControl:
    """A Driven setup: adaptive meshing, its convergence test, and a Fast sweep.

    The adaptive solve runs between `minimum_passes` and `maximum_passes`, refining
    `percent_refinement` percent per pass. A pass is converged when the largest
    S-parameter change since the previous pass is below `maximum_delta_s`;
    `minimum_converged_passes` such passes are required.
    """

    setup_name: str
    sweep_name: str
    sweep: FrequencySweepSpec
    maximum_passes: int = 99
    minimum_passes: int = 1
    minimum_converged_passes: int = 1
    percent_refinement: float = 30.0
    maximum_delta_s: float = 0.02

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "setup_name", stripped_text(self.setup_name, "setup_name")
        )
        object.__setattr__(
            self, "sweep_name", stripped_text(self.sweep_name, "sweep_name")
        )
        refinement = adaptive_controls(
            self.maximum_passes,
            self.minimum_passes,
            self.minimum_converged_passes,
            self.percent_refinement,
        )
        delta = coerced_finite_number(self.maximum_delta_s, "maximum_delta_s")
        if delta <= 0:
            raise ValueError("maximum_delta_s must be > 0")
        object.__setattr__(self, "percent_refinement", refinement)
        object.__setattr__(self, "maximum_delta_s", delta)

    def to_payload(self) -> dict[str, Any]:
        """Serialize the setup, sweep and adaptive-pass controls."""
        return {
            "setup_name": self.setup_name,
            "sweep_name": self.sweep_name,
            "sweep": self.sweep.to_payload(),
            "maximum_passes": self.maximum_passes,
            "minimum_passes": self.minimum_passes,
            "minimum_converged_passes": self.minimum_converged_passes,
            "percent_refinement": self.percent_refinement,
            "maximum_delta_s": self.maximum_delta_s,
        }


@dataclass(frozen=True)
class HfssDrivenSpec:
    """A two-port Driven HFSS model imported from a GDS file.

    `mode` is "terminal" (ports defined by conductor references) or "modal"
    (ports defined by an integration line). The GDS layers named in
    `layer_imports` are imported, `object_bindings` give each imported object a role
    and material, and `ports` are on the -X and +X faces. `region_padding_um` pads
    the solution Region on +X, -X, +Y, -Y, +Z, -Z and must be zero on each port's
    face. `length_mesh` optionally adds the uniform-CPW length mesh.
    """

    mode: HfssDrivenMode
    gds_path: Path | str
    modeling: Modeling
    project_name: str
    design_name: str
    materials: Mapping[str, PdkMaterial]
    vacuum_material_id: str
    layer_imports: tuple[LayerImport, ...]
    object_bindings: tuple[ObjectBinding, ...]
    ports: tuple[TerminalPort, TerminalPort] | tuple[ModalPort, ModalPort]
    run_control: HfssRunControl
    region_padding_um: tuple[float, float, float, float, float, float]
    physical_layers: tuple[PhysicalLayerDeclaration, ...] = ()
    physical_representation: PhysicalRepresentationPlan | None = None
    length_mesh: LengthMeshSpec | None = None
    aedt_version: str = REQUIRED_AEDT_VERSION
    pyaedt_version: str = LOCKED_PYAEDT

    def __post_init__(self) -> None:
        request_modeling(self)
        if self.mode not in {"terminal", "modal"}:
            raise ValueError("mode must be terminal or modal")
        grounds, signals = normalize_gds_spec(self)
        if len(self.ports) != 2 or tuple(port.index for port in self.ports) != (1, 2):
            raise ValueError("V1 requires exactly ordered ports 1 and 2")
        if len({port.name for port in self.ports}) != 2 or {
            port.side for port in self.ports
        } != {"-X", "+X"}:
            raise ValueError("V1 requires unique ports on distinct -X and +X faces")
        if self.mode == "terminal" and not all(
            isinstance(port, TerminalPort) for port in self.ports
        ):
            raise ValueError("terminal mode requires TerminalPort entries")
        if self.mode == "modal" and not all(
            isinstance(port, ModalPort) for port in self.ports
        ):
            raise ValueError("modal mode requires ModalPort entries")
        if (
            self.mode == "terminal"
            and all(isinstance(port, TerminalPort) for port in self.ports)
            and any(
                not set(port.reference_objects).issubset(grounds) for port in self.ports
            )
        ):
            raise ValueError("terminal references must name declared ground objects")
        if (
            self.mode == "terminal"
            and all(isinstance(port, TerminalPort) for port in self.ports)
            and self.ports[0].reference_objects != self.ports[1].reference_objects
        ):
            raise ValueError(
                "terminal ports must share one ordered global reference conductor tuple"
            )
        padding = checked_region_padding(self.region_padding_um)
        # PyAEDT create_region consumes this native order unchanged.
        side_index = {"+X": 0, "-X": 1, "+Y": 2, "-Y": 3, "+Z": 4, "-Z": 5}
        if any(padding[side_index[port.side]] != 0 for port in self.ports):
            raise ValueError(
                "region padding must be exactly zero on each declared port side"
            )
        object.__setattr__(self, "region_padding_um", padding)
        if self.length_mesh is not None and (
            not set(self.length_mesh.ground_objects).issubset(grounds)
            or not set(self.length_mesh.signal_objects).issubset(signals)
        ):
            raise ValueError(
                "length mesh targets must be declared ground/signal objects"
            )

    @property
    def effective_ports(self) -> tuple[TerminalPort, ...] | tuple[ModalPort, ...]:
        """Native port coordinates derived from immutable authored endpoints."""
        mapping = request_modeling(self)
        return tuple(
            replace(
                port,
                integration_line_um=tuple(
                    (x, y, mapping.map_z(z)) for x, y, z in port.integration_line_um
                ),
            )
            if isinstance(port, ModalPort)
            else port
            for port in self.ports
        )

    @property
    def effective_region(self) -> dict[str, Any]:
        """The solution Region in effective coordinates, projected from the selected GDS
        geometry.
        """
        return _direct_hfss_enclosure(self)

    @property
    def effective_layer_imports(self) -> tuple[dict[str, Any], ...]:
        """Layer imports as AEDT will receive them, after the Z-map is applied."""
        return _effective_layer_imports(self)

    def to_payload(self) -> dict[str, Any]:
        """Serialize with the effective layer imports and ports of the bound plan."""
        return {
            **modeling_payload(self),
            "schema_version": SCHEMA_VERSION,
            "mode": self.mode,
            "aedt": {"requested_version": self.aedt_version},
            "pyaedt": {
                "locked_version": self.pyaedt_version,
                "official_source": OFFICIAL_PYAEDT_SOURCE_URL,
            },
            "project": {"name": self.project_name, "design": self.design_name},
            "materials": {
                material_id: item.to_payload()
                for material_id, item in self.materials.items()
            },
            "vacuum_material_id": self.vacuum_material_id,
            "gds": {"path": self.gds_path.as_posix()},
            "layer_imports": list(self.effective_layer_imports),
            "object_bindings": [item.to_payload() for item in self.object_bindings],
            "ports": [item.to_payload() for item in self.ports],
            "effective_ports": [item.to_payload() for item in self.effective_ports],
            "run_control": self.run_control.to_payload(),
            "region_padding_um": list(self.region_padding_um),
            "length_mesh": self.length_mesh.to_payload() if self.length_mesh else None,
        }

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        base_dir: Path | None = None,
    ) -> HfssDrivenSpec:
        """Rebuild from a driven v1 payload, then verify stored effective layers and ports
        against the plan.
        """
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported HFSS driven schema")
        mode = stripped_text(payload.get("mode"), "mode")
        gds = Path(stripped_text(payload.get("gds", {}).get("path"), "gds.path"))
        if base_dir is not None and not gds.is_absolute():
            gds = base_dir / gds
        imports = tuple(
            LayerImport(**source_record_payload(item))
            for item in payload.get("layer_imports", ())
        )
        bindings = tuple(
            ObjectBinding(**item) for item in payload.get("object_bindings", ())
        )
        materials = _materials_from_payload(payload)
        port_type = TerminalPort if mode == "terminal" else ModalPort
        ports = tuple(port_type(**item) for item in payload.get("ports", ()))
        run = payload.get("run_control", {})
        sweep = FrequencySweepSpec(**run.get("sweep", {}))
        length = payload.get("length_mesh")
        result = cls(
            modeling=payload.get("modeling"),
            **physical_fields_from_payload(payload),
            mode=mode,  # type: ignore[arg-type]
            gds_path=gds,
            project_name=project_name_from_payload(
                payload.get("project", {}).get("name")
            ),
            design_name=stripped_text(
                payload.get("project", {}).get("design"), "project.design"
            ),
            materials=materials,
            vacuum_material_id=stripped_text(
                payload.get("vacuum_material_id"), "vacuum_material_id"
            ),
            layer_imports=imports,
            object_bindings=bindings,
            ports=ports,  # type: ignore[arg-type]
            run_control=_run_control_from_payload(run, sweep),
            region_padding_um=tuple(payload.get("region_padding_um", ())),  # type: ignore[arg-type]
            length_mesh=LengthMeshSpec(**length) if length is not None else None,
            aedt_version=stripped_text(
                payload.get("aedt", {}).get("requested_version"),
                "aedt.requested_version",
            ),
            pyaedt_version=stripped_text(
                payload.get("pyaedt", {}).get("locked_version"), "pyaedt.locked_version"
            ),
        )
        verify_effective_records(
            payload["layer_imports"], result.effective_layer_imports
        )
        verify_effective_records(
            payload["effective_ports"],
            [item.to_payload() for item in result.effective_ports],
        )
        return result


@dataclass(frozen=True)
class EigenmodeRunControl:
    """An Eigenmode setup: how many modes to find, from where, and when to stop.

    AEDT searches for `num_modes` modes above `minimum_frequency_ghz`. The adaptive
    solve runs between `minimum_passes` and `maximum_passes`, refining
    `percent_refinement` percent per pass, and a pass is converged when the
    frequency change since the previous pass is below
    `maximum_delta_frequency_percent`; `minimum_converged_passes` such passes are
    required.
    """

    setup_name: str
    minimum_frequency_ghz: float
    num_modes: int
    maximum_passes: int
    maximum_delta_frequency_percent: float
    minimum_passes: int = 1
    minimum_converged_passes: int = 1
    percent_refinement: float = 30.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "setup_name", stripped_text(self.setup_name, "setup_name")
        )
        minimum = coerced_finite_number(
            self.minimum_frequency_ghz, "minimum_frequency_ghz"
        )
        delta = coerced_finite_number(
            self.maximum_delta_frequency_percent,
            "maximum_delta_frequency_percent",
        )
        if minimum <= 0 or delta <= 0:
            raise ValueError("Eigenmode frequency and convergence percent must be > 0")
        if not isinstance(self.num_modes, int) or self.num_modes <= 0:
            raise ValueError("num_modes must be a positive integer")
        refinement = adaptive_controls(
            self.maximum_passes,
            self.minimum_passes,
            self.minimum_converged_passes,
            self.percent_refinement,
        )
        object.__setattr__(self, "minimum_frequency_ghz", minimum)
        object.__setattr__(self, "maximum_delta_frequency_percent", delta)
        object.__setattr__(self, "percent_refinement", refinement)

    def to_payload(self) -> dict[str, Any]:
        """Serialize the Eigenmode setup and adaptive-pass controls."""
        return {
            "setup_name": self.setup_name,
            "minimum_frequency_ghz": self.minimum_frequency_ghz,
            "num_modes": self.num_modes,
            "maximum_passes": self.maximum_passes,
            "maximum_delta_frequency_percent": self.maximum_delta_frequency_percent,
            "minimum_passes": self.minimum_passes,
            "minimum_converged_passes": self.minimum_converged_passes,
            "percent_refinement": self.percent_refinement,
        }


@dataclass(frozen=True)
class HfssEigenmodeSpec:
    """A port-free HFSS Eigenmode model imported from a GDS file.

    Eigenmode finds the resonant frequencies and fields of the structure itself, so
    there are no ports. Layer imports, bindings and Region padding work as in
    `HfssDrivenSpec`; `run_control` is the Eigenmode setup.
    """

    gds_path: Path | str
    modeling: Modeling
    project_name: str
    design_name: str
    materials: Mapping[str, PdkMaterial]
    vacuum_material_id: str
    layer_imports: tuple[LayerImport, ...]
    object_bindings: tuple[ObjectBinding, ...]
    run_control: EigenmodeRunControl
    region_padding_um: tuple[float, float, float, float, float, float]
    physical_layers: tuple[PhysicalLayerDeclaration, ...] = ()
    physical_representation: PhysicalRepresentationPlan | None = None
    length_mesh: LengthMeshSpec | None = None
    aedt_version: str = REQUIRED_AEDT_VERSION
    pyaedt_version: str = LOCKED_PYAEDT

    @property
    def mode(self) -> Literal["eigenmode"]:
        """Always `"eigenmode"`; written into the payload as `mode`."""
        return "eigenmode"

    def __post_init__(self) -> None:
        request_modeling(self)
        grounds, signals = normalize_gds_spec(self)
        object.__setattr__(
            self, "region_padding_um", checked_region_padding(self.region_padding_um)
        )
        if self.length_mesh is not None and (
            not set(self.length_mesh.ground_objects).issubset(grounds)
            or not set(self.length_mesh.signal_objects).issubset(signals)
        ):
            raise ValueError(
                "length mesh targets must be declared ground/signal objects"
            )

    @property
    def effective_region(self) -> dict[str, Any]:
        """The solution Region in effective coordinates, projected from the selected GDS
        geometry.
        """
        return _direct_hfss_enclosure(self)

    @property
    def effective_layer_imports(self) -> tuple[dict[str, Any], ...]:
        """Layer imports as AEDT will receive them, after the Z-map is applied."""
        return _effective_layer_imports(self)

    def to_payload(self) -> dict[str, Any]:
        """Serialize with the effective layer imports of the bound plan."""
        return {
            **modeling_payload(self),
            "schema_version": EIGENMODE_SCHEMA_VERSION,
            "mode": self.mode,
            "aedt": {"requested_version": self.aedt_version},
            "pyaedt": {
                "locked_version": self.pyaedt_version,
                "official_source": OFFICIAL_PYAEDT_SOURCE_URL,
            },
            "project": {"name": self.project_name, "design": self.design_name},
            "materials": {
                material_id: item.to_payload()
                for material_id, item in self.materials.items()
            },
            "vacuum_material_id": self.vacuum_material_id,
            "gds": {"path": self.gds_path.as_posix()},
            "layer_imports": list(self.effective_layer_imports),
            "object_bindings": [item.to_payload() for item in self.object_bindings],
            "run_control": self.run_control.to_payload(),
            "region_padding_um": list(self.region_padding_um),
            "length_mesh": self.length_mesh.to_payload() if self.length_mesh else None,
        }

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        base_dir: Path | None = None,
    ) -> HfssEigenmodeSpec:
        """Rebuild from an Eigenmode v1 payload, then verify stored effective layers
        against the plan.
        """
        if payload.get("schema_version") != EIGENMODE_SCHEMA_VERSION:
            raise ValueError("unsupported HFSS Eigenmode schema")
        gds = Path(stripped_text(payload.get("gds", {}).get("path"), "gds.path"))
        if base_dir is not None and not gds.is_absolute():
            gds = base_dir / gds
        raw_materials = payload.get("materials")
        if not isinstance(raw_materials, dict):
            raise TypeError("materials must be a JSON object")
        materials = {
            material_id: PdkMaterial(**item)
            for material_id, item in raw_materials.items()
        }
        run = payload.get("run_control")
        if not isinstance(run, dict):
            raise TypeError("run_control must be a JSON object")
        length = payload.get("length_mesh")
        result = cls(
            modeling=payload.get("modeling"),
            **physical_fields_from_payload(payload),
            gds_path=gds,
            project_name=project_name_from_payload(
                payload.get("project", {}).get("name")
            ),
            design_name=stripped_text(
                payload.get("project", {}).get("design"), "project.design"
            ),
            materials=materials,
            vacuum_material_id=stripped_text(
                payload.get("vacuum_material_id"), "vacuum_material_id"
            ),
            layer_imports=tuple(
                LayerImport(**source_record_payload(item))
                for item in payload.get("layer_imports", ())
            ),
            object_bindings=tuple(
                ObjectBinding(**item) for item in payload.get("object_bindings", ())
            ),
            run_control=EigenmodeRunControl(**run),
            region_padding_um=tuple(payload.get("region_padding_um", ())),  # type: ignore[arg-type]
            length_mesh=LengthMeshSpec(**length) if length is not None else None,
            aedt_version=stripped_text(
                payload.get("aedt", {}).get("requested_version"),
                "aedt.requested_version",
            ),
            pyaedt_version=stripped_text(
                payload.get("pyaedt", {}).get("locked_version"), "pyaedt.locked_version"
            ),
        )
        verify_effective_records(
            payload["layer_imports"], result.effective_layer_imports
        )
        return result


@dataclass(frozen=True)
class LengthMeshSpec:
    """A length-based mesh on the signal and ground objects of a uniform CPW.

    CPW is a coplanar waveguide, and MTL a multiconductor transmission line
    (`uniform_cpw_mtl` is always true). `maximum_length_um` caps the mesh element
    length in micrometres on the named objects.
    TODO(human): state the ordinary ground mesh rule this exception overrides.
    """

    signal_objects: tuple[str, ...]
    ground_objects: tuple[str, ...]
    maximum_length_um: float
    uniform_cpw_mtl: Literal[True] = True

    def __post_init__(self) -> None:
        signals = tuple(
            stripped_text(value, "signal_object") for value in self.signal_objects
        )
        grounds = tuple(
            stripped_text(value, "ground_object") for value in self.ground_objects
        )
        if not signals or not grounds:
            raise ValueError(
                "length mesh requires explicit signal_objects and ground_objects"
            )
        maximum = coerced_finite_number(self.maximum_length_um, "maximum_length_um")
        if maximum <= 0:
            raise ValueError("maximum_length_um must be > 0")
        if self.uniform_cpw_mtl is not True:
            raise ValueError("length mesh requires uniform_cpw_mtl=True")
        object.__setattr__(self, "signal_objects", signals)
        object.__setattr__(self, "ground_objects", grounds)
        object.__setattr__(self, "maximum_length_um", maximum)

    def to_payload(self) -> dict[str, Any]:
        """Serialize the signal and ground object lists and the maximum element length."""
        return {
            "uniform_cpw_mtl": True,
            "signal_objects": list(self.signal_objects),
            "ground_objects": list(self.ground_objects),
            "maximum_length_um": self.maximum_length_um,
        }
