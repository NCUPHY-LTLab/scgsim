"""HFSS Driven Terminal/Modal request records and payload decoding."""

from __future__ import annotations

from collections.abc import (
    Mapping,
)
from dataclasses import (
    dataclass,
    field,
    replace,
)
from pathlib import (
    Path,
)
from typing import (
    Any,
    Literal,
    cast,
)

from scgsim.aedt.specs.common import (
    DRIVEN_GEOMETRY_SCHEMA_VERSION,
    LOCKED_PYAEDT,
    OFFICIAL_PYAEDT_SOURCE_URL,
    SCHEMA_VERSION,
    HfssDrivenMode,
    LayerImport,
    LumpedTerminalPort,
    ModalPort,
    ObjectBinding,
    PdkMaterial,
    TerminalPort,
    normalize_gds_spec,
    project_filename,
    project_name_from_payload,
    text_value,
    validate_region_padding,
)
from scgsim.aedt.specs.geometry import PreparedPlanarGeometry
from scgsim.aedt.specs.hfss_common import (
    FrequencySweepSpec,
    HfssRunControl,
    LengthMeshSpec,
    decode_pdk_materials,
    direct_hfss_enclosure,
)
from scgsim.aedt.specs.modeling import (
    Modeling,
    PhysicalLayerDeclaration,
    PhysicalLayerSpec,
    PhysicalRepresentationPlan,
    effective_record,
    modeling_payload,
    physical_layers_from_payload,
    request_modeling,
    source_plan_layers,
    source_record_payload,
    verify_effective_records,
)
from scgsim.aedt.versions import (
    normalize_aedt_version,
)


@dataclass(frozen=True)
class HfssDrivenSpec:
    """The complete two-port, one-mode, one-setup public CPW handoff spec."""

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
    physical_layers: tuple[PhysicalLayerDeclaration | PhysicalLayerSpec, ...] = ()
    physical_representation: PhysicalRepresentationPlan | None = None
    length_mesh: LengthMeshSpec | None = None
    aedt_version: str = field(kw_only=True)
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
                not set(port.reference_objects).issubset(grounds)
                for port in cast(tuple[TerminalPort, ...], self.ports)
            )
        ):
            raise ValueError("terminal references must name declared ground objects")
        if (
            self.mode == "terminal"
            and all(isinstance(port, TerminalPort) for port in self.ports)
            and cast(tuple[TerminalPort, ...], self.ports)[0].reference_objects
            != cast(tuple[TerminalPort, ...], self.ports)[1].reference_objects
        ):
            raise ValueError(
                "terminal ports must share one ordered global reference conductor tuple"
            )
        padding = validate_region_padding(self.region_padding_um)
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
        ports = tuple(
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
        return cast(tuple[TerminalPort, ...] | tuple[ModalPort, ...], ports)

    @property
    def effective_region(self) -> dict[str, Any] | None:
        """Return the projected solver region, when available."""
        return direct_hfss_enclosure(self)

    @property
    def effective_layer_imports(self) -> tuple[dict[str, Any], ...]:
        """Return effective layer imports derived from the source model."""
        mapping = request_modeling(self)
        pairs = {
            pair: index + 1
            for index, pair in enumerate(
                sorted((item.layer, item.datatype) for item in self.layer_imports)
            )
        }
        domain_pairs = {
            (item.layer, item.datatype)
            for item in self.layer_imports
            if any(
                binding.role == "substrate"
                and binding.layer == item.layer
                and binding.object_name.startswith(f"{item.layer_name}_")
                for binding in self.object_bindings
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
            for item in self.layer_imports
        )

    def to_payload(self) -> dict[str, Any]:
        """Return the canonical HfssDrivenSpec payload."""
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
            "gds": {"path": Path(self.gds_path).as_posix()},
            "layer_imports": (list(self.effective_layer_imports)),
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
        """Decode a HfssDrivenSpec from its canonical payload."""
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported HFSS driven schema")
        mode = text_value(payload.get("mode"), "mode")
        gds = Path(text_value(payload.get("gds", {}).get("path"), "gds.path"))
        if base_dir is not None and not gds.is_absolute():
            gds = base_dir / gds
        imports, bindings = _decode_driven_layer_bindings(payload)
        materials = decode_pdk_materials(payload.get("materials"))
        ports = _decode_driven_ports(mode, payload)
        run = payload.get("run_control", {})
        sweep = FrequencySweepSpec(**run.get("sweep", {}))
        length = payload.get("length_mesh")
        result = cls(
            modeling=cast(Modeling, payload.get("modeling")),
            physical_layers=physical_layers_from_payload(payload),
            physical_representation=(
                PhysicalRepresentationPlan.from_payload(
                    payload["physical_representation"]
                )
                if "physical_representation" in payload
                else None
            ),
            mode=mode,  # type: ignore[arg-type]
            gds_path=gds,
            project_name=project_name_from_payload(
                payload.get("project", {}).get("name")
            ),
            design_name=text_value(
                payload.get("project", {}).get("design"), "project.design"
            ),
            materials=materials,
            vacuum_material_id=text_value(
                payload.get("vacuum_material_id"), "vacuum_material_id"
            ),
            layer_imports=imports,
            object_bindings=bindings,
            ports=ports,  # type: ignore[arg-type]
            run_control=HfssRunControl(
                text_value(run.get("setup_name"), "setup_name"),
                text_value(run.get("sweep_name"), "sweep_name"),
                sweep,
                maximum_passes=run.get("maximum_passes", 99),
                minimum_passes=run.get("minimum_passes", 1),
                minimum_converged_passes=run.get("minimum_converged_passes", 1),
                percent_refinement=run.get("percent_refinement", 30.0),
                maximum_delta_s=run.get("maximum_delta_s", 0.02),
            ),
            region_padding_um=tuple(payload.get("region_padding_um", ())),  # type: ignore[arg-type]
            length_mesh=LengthMeshSpec(**length) if length is not None else None,
            aedt_version=payload.get("aedt", {}).get("requested_version"),
            pyaedt_version=text_value(
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


def _decode_driven_ports(mode: str, payload: dict[str, Any]):
    port_type = TerminalPort if mode == "terminal" else ModalPort
    return tuple(port_type(**item) for item in payload.get("ports", ()))


def _decode_driven_layer_bindings(
    payload: dict[str, Any],
) -> tuple[tuple[LayerImport, ...], tuple[ObjectBinding, ...]]:
    imports = tuple(
        LayerImport(**source_record_payload(item))
        for item in payload.get("layer_imports", ())
    )
    bindings = tuple(
        ObjectBinding(**item) for item in payload.get("object_bindings", ())
    )
    return imports, bindings


@dataclass(frozen=True)
class HfssDrivenGeometrySpec:
    """Body-first Terminal model with optional setup; no synthetic solve controls."""

    modeling: Modeling
    project_name: str
    design_name: str
    geometry: Any
    ports: tuple[TerminalPort | LumpedTerminalPort, ...]
    run_control: HfssRunControl | None = None
    physical_layers: tuple[PhysicalLayerDeclaration | PhysicalLayerSpec, ...] = ()
    physical_representation: PhysicalRepresentationPlan | None = None
    aedt_version: str = field(kw_only=True)
    pyaedt_version: str = LOCKED_PYAEDT

    @property
    def mode(self) -> Literal["terminal"]:
        """Return the solver mode declared by this request."""
        return "terminal"

    def __post_init__(self) -> None:
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
        request_modeling(self)
        object.__setattr__(
            self, "aedt_version", normalize_aedt_version(self.aedt_version)
        )
        if self.pyaedt_version != LOCKED_PYAEDT:
            raise ValueError("V1 requires PyAEDT 1.3.0")
        object.__setattr__(
            self,
            "project_name",
            project_filename(self.project_name, "project_name").stem,
        )
        object.__setattr__(
            self, "design_name", text_value(self.design_name, "design_name")
        )
        if self.run_control is not None and not isinstance(
            self.run_control, HfssRunControl
        ):
            raise TypeError("run_control must be HfssRunControl or None")
        _validate_driven_geometry_ports(self)

    def to_payload(self) -> dict[str, Any]:
        """Return the canonical HfssDrivenGeometrySpec payload."""
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
        """Decode a HfssDrivenGeometrySpec from its canonical payload."""
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
            physical_representation=(
                PhysicalRepresentationPlan.from_payload(
                    payload["physical_representation"]
                )
                if "physical_representation" in payload
                else None
            ),
            aedt_version=payload["aedt"]["requested_version"],
            pyaedt_version=payload["pyaedt"]["locked_version"],
        )


def _validate_driven_geometry_ports(spec: HfssDrivenGeometrySpec) -> None:
    ports = tuple(spec.ports)
    if (
        tuple(port.index for port in ports) != (1, 2)
        or len({port.name for port in ports}) != 2
    ):
        raise ValueError("V1 requires unique ordered ports 1 and 2")
    supports = {
        item["support_id"]: item
        for item in spec.geometry.source.get("lumped_supports", ())
    }
    conductor_ids = {item["semantic_id"] for item in spec.geometry.source["conductors"]}
    conductor_ids.update(
        item["source_semantic_id"] for item in spec.geometry.source["conductors"]
    )
    _validate_driven_port_bindings(spec, ports, supports, conductor_ids)
    if all(isinstance(port, TerminalPort) for port in ports):
        terminal_ports = cast(tuple[TerminalPort, ...], ports)
        if {port.side for port in terminal_ports} != {"-X", "+X"}:
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
    object.__setattr__(spec, "ports", ports)


def _validate_driven_port_bindings(
    spec: HfssDrivenGeometrySpec,
    ports: tuple[TerminalPort | LumpedTerminalPort, ...],
    supports: dict[str, Any],
    conductor_ids: set[str],
) -> None:
    for port in ports:
        if isinstance(port, LumpedTerminalPort):
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
        elif isinstance(port, TerminalPort):
            if not port.signal_entity_ids or not set(port.signal_entity_ids).issubset(
                conductor_ids
            ):
                raise ValueError(
                    "body-first Wave requires explicit source Signal Entities"
                )
            if not set(port.reference_objects).issubset(conductor_ids):
                raise ValueError("Wave references must bind source conductor Entities")
            padding = spec.geometry.source["native_region"]["padding_um"]
            # Match the existing native Region order, without extending Wave support.
            index = {"+X": 0, "-X": 1, "+Y": 2, "-Y": 3, "+Z": 4, "-Z": 5}[port.side]
            if padding[index] != 0:
                raise ValueError("Wave port side requires zero Region padding")
        else:
            raise TypeError("Driven Terminal requires Wave or Lumped terminal records")
