"""HFSS Eigenmode request records and payload decoding."""

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
from typing import (
    Any,
    Literal,
    cast,
)

from scgsim.aedt.specs.common import (
    EIGENMODE_SCHEMA_VERSION,
    LOCKED_PYAEDT,
    OFFICIAL_PYAEDT_SOURCE_URL,
    LayerImport,
    ObjectBinding,
    PdkMaterial,
    normalize_gds_spec,
    project_name_from_payload,
    text_value,
    validate_region_padding,
)
from scgsim.aedt.specs.hfss_common import (
    EigenmodeRunControl,
    LengthMeshSpec,
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
    source_record_payload,
    verify_effective_records,
)


@dataclass(frozen=True)
class HfssEigenmodeSpec:
    """One port-free HFSS Eigenmode model with explicit native setup controls."""

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
    physical_layers: tuple[PhysicalLayerDeclaration | PhysicalLayerSpec, ...] = ()
    physical_representation: PhysicalRepresentationPlan | None = None
    _historical_physical: bool = False
    _historical_modeling: bool = False
    length_mesh: LengthMeshSpec | None = None
    aedt_version: str = field(kw_only=True)
    pyaedt_version: str = LOCKED_PYAEDT

    @property
    def historical_physical(self) -> bool:
        """Report whether this request decodes historical physical data."""
        return self._historical_physical

    @property
    def historical_modeling(self) -> bool:
        """Report whether this request decodes historical modeling data."""
        return self._historical_modeling

    @property
    def mode(self) -> Literal["eigenmode"]:
        """Return the solver mode declared by this request."""
        return "eigenmode"

    def __post_init__(self) -> None:
        request_modeling(self)
        grounds, signals = normalize_gds_spec(self)
        object.__setattr__(
            self, "region_padding_um", validate_region_padding(self.region_padding_um)
        )
        if self.length_mesh is not None and (
            not set(self.length_mesh.ground_objects).issubset(grounds)
            or not set(self.length_mesh.signal_objects).issubset(signals)
        ):
            raise ValueError(
                "length mesh targets must be declared ground/signal objects"
            )

    @property
    def effective_region(self) -> dict[str, Any] | None:
        """Return the projected solver region, when available."""
        return direct_hfss_enclosure(self)

    @property
    def effective_layer_imports(self) -> tuple[dict[str, Any], ...]:
        """Return effective layer imports derived from the source model."""
        mapping = request_modeling(self)
        if mapping is None:
            raise ValueError("historical requests have no new execution modeling map")
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
        """Return the canonical HfssEigenmodeSpec payload."""
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
            "gds": {"path": Path(self.gds_path).as_posix()},
            "layer_imports": (
                list(self.effective_layer_imports)
                if self.modeling is not None
                else [item.to_payload() for item in self.layer_imports]
            ),
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
        allow_historical_modeling: bool = False,
    ) -> HfssEigenmodeSpec:
        """Decode a HfssEigenmodeSpec from its canonical payload."""
        if payload.get("schema_version") != EIGENMODE_SCHEMA_VERSION:
            raise ValueError("unsupported HFSS Eigenmode schema")
        gds = Path(text_value(payload.get("gds", {}).get("path"), "gds.path"))
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
            aedt_version=payload.get("aedt", {}).get("requested_version"),
            pyaedt_version=text_value(
                payload.get("pyaedt", {}).get("locked_version"), "pyaedt.locked_version"
            ),
        )
        if result.modeling is not None:
            verify_effective_records(
                payload["layer_imports"], result.effective_layer_imports
            )
        return result
