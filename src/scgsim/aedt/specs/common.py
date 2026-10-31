"""Records and schema constants shared by every AEDT request family.

Owns the small immutable pieces that more than one request is built from: lumped
RLC treatments, solve resources, GDS layer imports, PDK materials, object
bindings, the three kinds of port, matrix (Q3D/Q2D) run control, and the
validators they share. Each record checks its own fields on construction and
converts to and from a JSON payload with `to_payload`.

Whole requests are in `hfss`, `hfss_prepared`, `q3d` and `q2d`. Nothing here
talks to AEDT; the tool versions are pinned only so a request can refuse to run
against a different AEDT or PyAEDT.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from scgsim.aedt._data import _stripped_text


@dataclass(frozen=True)
class LumpedRlc:
    """A resistor, inductor and capacitor placed on one neutral lumped support.

    A neutral support is the sheet a lumped element occupies between two groups of
    conductors. `topology` says whether the enabled components are wired in series
    or in parallel; AEDT spells these "Serial" and "Parallel" (see `_rlc`). A
    component left as None is absent, and at least one must be enabled. Units are
    ohm, henry and farad. Values only have to be finite; this record does not
    reject zero or negative values.
    """

    support_id: str
    topology: Literal["series", "parallel"]
    resistance_ohm: float | None = None
    inductance_h: float | None = None
    capacitance_f: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "support_id", _stripped_text(self.support_id, "support_id")
        )
        if self.topology not in {"series", "parallel"}:
            raise ValueError("lumped topology must be series or parallel")
        for key in ("resistance_ohm", "inductance_h", "capacitance_f"):
            value = getattr(self, key)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError(f"{key} must be numeric or None")
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError(f"{key} must be finite")
                object.__setattr__(self, key, value)
        if all(
            getattr(self, key) is None
            for key in ("resistance_ohm", "inductance_h", "capacitance_f")
        ):
            raise ValueError("lumped RLC requires an enabled component")

    def to_payload(self) -> dict[str, Any]:
        """Serialize all three components, writing null for a disabled one."""
        return {
            "support_id": self.support_id,
            "topology": self.topology,
            "resistance_ohm": self.resistance_ohm,
            "inductance_h": self.inductance_h,
            "capacitance_f": self.capacitance_f,
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> LumpedRlc:
        """Rebuild from a payload; an unknown key raises TypeError from the constructor."""
        return cls(**dict(value))


HfssDrivenMode = Literal["terminal", "modal"]

LayerRole = Literal["ground", "signal", "substrate"]

PdkMaterialKind = Literal["vacuum", "dielectric", "superconductor"]

Side = Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z"]

# Schema identifiers. A payload's `schema_version` selects its parser (`parse.py`),
# and a changed payload shape gets a new version instead of an edit. Older versions
# stay listed because saved hand-offs still have to load.
SCHEMA_VERSION = "scgsim.aedt.hfss-driven.v1"

EIGENMODE_SCHEMA_VERSION = "scgsim.aedt.hfss-eigenmode.v1"

EPR_EIGENMODE_SCHEMA_VERSION = "scgsim.aedt.hfss-eigenmode-epr.v1"

EPR_ANALYSIS_SCHEMA_VERSION = "scgsim.aedt.hfss-eigenmode-epr-analysis.v1"

EPR_EIGENMODE_SCHEMA_VERSION_V2 = "scgsim.aedt.hfss-eigenmode-epr.v2"

EPR_EIGENMODE_SCHEMA_VERSION_V3 = "scgsim.aedt.hfss-eigenmode-epr.v3"

DRIVEN_GEOMETRY_SCHEMA_VERSION = "scgsim.aedt.hfss-driven-geometry.v1"

EPR_ANALYSIS_SCHEMA_VERSION_V2 = "scgsim.aedt.hfss-eigenmode-epr-analysis.v2"

Q3D_SCHEMA_VERSION = "scgsim.aedt.q3d.v3"

Q3D_GEOMETRY_SOURCE_SCHEMA_VERSION = "scgsim.aedt.q3d-geometry-source.v3"

Q2D_SCHEMA_VERSION = "scgsim.aedt.q2d.v1"
Q2D_SCHEMA_VERSION_V2 = "scgsim.aedt.q2d.v2"

OFFICIAL_PYAEDT_SOURCE_URL = "https://github.com/ansys/pyaedt/tree/v1.3.0"

LOCKED_PYAEDT = "1.3.0"

REQUIRED_AEDT_VERSION = "2024.2"

# Default number of frequency points in a Fast sweep.
POINT_COUNT = 20_000

# Curved-surface approximation level given to AEDT when meshing curved GDS edges.
# TODO(human): why level 9 and not another value.
SURFACE_APPROXIMATION_LEVEL = 9


@dataclass(frozen=True)
class AedtResources:
    """How much of this machine one AEDT solve may use, independent of geometry.

    `cores` is the number of solver cores; `ram_limit_percent` (1 to 100) is the
    RAM limit AEDT is asked to respect, as a percentage.
    """

    cores: int
    ram_limit_percent: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.cores, bool)
            or not isinstance(self.cores, int)
            or self.cores <= 0
        ):
            raise ValueError("cores must be a positive integer")
        if (
            isinstance(self.ram_limit_percent, bool)
            or not isinstance(self.ram_limit_percent, int)
            or not 1 <= self.ram_limit_percent <= 100
        ):
            raise ValueError("ram_limit_percent must be an integer from 1 to 100")

    def to_payload(self) -> dict[str, int]:
        """Serialize as `cores` and `ram_limit_percent`."""
        return {"cores": self.cores, "ram_limit_percent": self.ram_limit_percent}


def _project_filename(value: Any, field: str) -> Path:
    project = Path(_stripped_text(value, field))
    if project.parent != Path(".") or project.name in {".", ".."}:
        raise ValueError("project_name must be a single project filename")
    if not project.stem:
        raise ValueError("project_name must normalize to a non-empty stem")
    return project


def _project_name_from_payload(value: Any) -> str:
    """Adapt one canonical payload name through constructor normalization."""
    return f"{_project_filename(value, 'project.name').name}.aedt"


def _coerced_finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _nonnegative_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _adaptive_controls(
    maximum_passes: Any,
    minimum_passes: Any,
    minimum_converged_passes: Any,
    percent_refinement: Any,
) -> float:
    """Check the adaptive-pass counts and return the validated refinement percent."""
    for field, value in (
        ("maximum_passes", maximum_passes),
        ("minimum_passes", minimum_passes),
        ("minimum_converged_passes", minimum_converged_passes),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{field} must be a positive integer")
        if field != "maximum_passes" and value > maximum_passes:
            raise ValueError(f"{field} must not exceed maximum_passes")
    refinement = _coerced_finite_number(percent_refinement, "percent_refinement")
    if not 0 < refinement <= 100:
        raise ValueError("percent_refinement must be in (0, 100]")
    return refinement


@dataclass(frozen=True)
class LayerImport:
    """One GDS (layer, datatype) pair and where AEDT should put it in Z.

    `layer_name` is the PyAEDT layer the polygons are imported into. The metal
    occupies `z_min_um` to `z_max_um`, in micrometres; equal values make a
    zero-thickness conductor sheet. `physical_layer_id` ties the import to a
    declared physical layer when a modeling plan is used.
    """

    layer: int
    datatype: int
    layer_name: str
    z_min_um: float
    z_max_um: float
    physical_layer_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer", _nonnegative_int(self.layer, "layer"))
        object.__setattr__(
            self, "datatype", _nonnegative_int(self.datatype, "datatype")
        )
        object.__setattr__(
            self, "layer_name", _stripped_text(self.layer_name, "layer_name")
        )
        low = _coerced_finite_number(self.z_min_um, "z_min_um")
        high = _coerced_finite_number(self.z_max_um, "z_max_um")
        if high < low:
            raise ValueError(
                "z_max_um must be >= z_min_um; equal values are conductor sheets"
            )
        object.__setattr__(self, "z_min_um", low)
        object.__setattr__(self, "z_max_um", high)

    def to_payload(self) -> dict[str, Any]:
        """Serialize with `physical_layer_id` first, and only when it is set."""
        return {
            **(
                {"physical_layer_id": self.physical_layer_id}
                if self.physical_layer_id is not None
                else {}
            ),
            "layer": self.layer,
            "datatype": self.datatype,
            "layer_name": self.layer_name,
            "z_min_um": self.z_min_um,
            "z_max_um": self.z_max_um,
        }


@dataclass(frozen=True)
class PdkMaterial:
    """A PDK material identity, without numeric properties.

    The PDK owns what the material is (vacuum, dielectric or superconductor); AEDT
    only needs a name. Dielectrics and vacuum name an AEDT library material through
    `library_name`. A superconductor must not name an AEDT material at all.
    """

    material_id: str
    kind: PdkMaterialKind
    is_superconducting: bool
    library_name: str | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "material_id", _stripped_text(self.material_id, "material_id")
        )
        if self.kind not in {"vacuum", "dielectric", "superconductor"}:
            raise ValueError(
                "material kind must be vacuum, dielectric, or superconductor"
            )
        if not isinstance(self.is_superconducting, bool):
            raise TypeError("is_superconducting must be boolean")
        if self.is_superconducting != (self.kind == "superconductor"):
            raise ValueError(
                "PDK material kind and is_superconducting must agree exactly"
            )
        library_name = (
            None
            if self.library_name is None
            else _stripped_text(self.library_name, "library_name")
        )
        if self.is_superconducting and library_name is not None:
            raise ValueError("superconducting PDK material must not name AEDT material")
        if not self.is_superconducting and library_name is None:
            raise ValueError(
                "non-superconducting PDK material requires AEDT library name"
            )
        object.__setattr__(self, "library_name", library_name)

    def to_payload(self) -> dict[str, Any]:
        """Serialize identity, kind, the superconducting flag and the AEDT library name
        (null for a superconductor).
        """
        return {
            "material_id": self.material_id,
            "kind": self.kind,
            "is_superconducting": self.is_superconducting,
            "library_name": self.library_name,
        }


@dataclass(frozen=True)
class ObjectBinding:
    """What an imported object is, by name, once AEDT has read the GDS.

    The GDS import produces objects named after their layer. Each binding records
    the object's GDS layer number, its role (ground, signal or substrate) and the
    PDK material it uses, so the model can be checked against what AEDT created.
    """

    object_name: str
    layer: int
    role: LayerRole
    material_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "object_name", _stripped_text(self.object_name, "object_name")
        )
        object.__setattr__(self, "layer", _nonnegative_int(self.layer, "binding.layer"))
        if self.role not in {"ground", "signal", "substrate"}:
            raise ValueError("role must be ground, signal, or substrate")
        object.__setattr__(
            self, "material_id", _stripped_text(self.material_id, "material_id")
        )

    def to_payload(self) -> dict[str, Any]:
        """Serialize the object name with its layer, role and material."""
        return {
            "object_name": self.object_name,
            "layer": self.layer,
            "role": self.role,
            "material_id": self.material_id,
        }


@dataclass(frozen=True)
class LumpedTerminalPort:
    """A lumped port on a neutral support, for a Driven Terminal solve.

    The port sits on a lumped support rather than on a Region face. Its signal
    entities must equal the support's terminal-A conductors and its reference
    entities the terminal-B conductors, in the authored order. `impedance_ohm` is the
    port impedance, `renormalize` asks AEDT to renormalize results to it, and
    de-embedding is not supported (only 0 is accepted).
    """

    index: int
    name: str
    support_id: str
    signal_entity_ids: tuple[str, ...]
    reference_entity_ids: tuple[str, ...]
    impedance_ohm: float
    renormalize: bool = False
    deembed_um: float = 0.0

    def __post_init__(self) -> None:
        if self.index not in {1, 2}:
            raise ValueError("two-port V1 uses port indices 1 and 2")
        for key in ("name", "support_id"):
            object.__setattr__(self, key, _stripped_text(getattr(self, key), key))
        for key in ("signal_entity_ids", "reference_entity_ids"):
            values = tuple(_stripped_text(value, key) for value in getattr(self, key))
            if not values:
                raise ValueError(f"{key} requires explicit conductor Entities")
            object.__setattr__(self, key, values)
        if isinstance(self.impedance_ohm, bool) or not isinstance(
            self.impedance_ohm, (int, float)
        ):
            raise TypeError("impedance_ohm must be explicitly numeric")
        if self.deembed_um != 0:
            raise NotImplementedError("Lumped Terminal supports deembed_um=0 only")
        if not isinstance(self.renormalize, bool):
            raise TypeError("renormalize must be bool")

    def to_payload(self) -> dict[str, Any]:
        """Serialize with `kind: "lumped"`, which is how a port list tells lumped ports
        from wave ports.
        """
        return {
            "kind": "lumped",
            "index": self.index,
            "name": self.name,
            "support_id": self.support_id,
            "signal_entity_ids": list(self.signal_entity_ids),
            "reference_entity_ids": list(self.reference_entity_ids),
            "impedance_ohm": self.impedance_ohm,
            "renormalize": self.renormalize,
            "deembed_um": self.deembed_um,
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> LumpedTerminalPort:
        """Rebuild from a payload whose `kind` must be `"lumped"`."""
        record = dict(value)
        if record.pop("kind") != "lumped":
            raise ValueError("lumped terminal kind differs")
        return cls(**record)


@dataclass(frozen=True)
class TerminalPort:
    """A wave port on a Region face, for a Driven Terminal solve.

    `side` is the Region face the port sits on. `reference_objects` are the ground
    conductors the port voltage is measured against, and both ports of a model must
    share the same ordered list. `deembed_um` de-embeds the port by that
    distance in micrometres.
    """

    index: int
    name: str
    side: Side
    reference_objects: tuple[str, ...]
    deembed_um: float = 0.0
    signal_entity_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.index not in {1, 2}:
            raise ValueError("two-port V1 uses port indices 1 and 2")
        object.__setattr__(self, "name", _stripped_text(self.name, "port.name"))
        if self.side not in {"+X", "-X", "+Y", "-Y", "+Z", "-Z"}:
            raise ValueError("port.side is invalid")
        references = tuple(
            _stripped_text(value, "reference_object")
            for value in self.reference_objects
        )
        if not references:
            raise ValueError("terminal port requires explicit reference_objects")
        object.__setattr__(self, "reference_objects", references)
        object.__setattr__(
            self,
            "signal_entity_ids",
            tuple(
                _stripped_text(value, "signal_entity_id")
                for value in self.signal_entity_ids
            ),
        )
        deembed = _coerced_finite_number(self.deembed_um, "deembed_um")
        if deembed < 0:
            raise ValueError("deembed_um must be >= 0")
        object.__setattr__(self, "deembed_um", deembed)

    def to_payload(self) -> dict[str, Any]:
        """Serialize the wave port; `signal_entity_ids` is written only when non-empty."""
        return {
            "index": self.index,
            "name": self.name,
            "side": self.side,
            "reference_objects": list(self.reference_objects),
            "deembed_um": self.deembed_um,
            **(
                {"signal_entity_ids": list(self.signal_entity_ids)}
                if self.signal_entity_ids
                else {}
            ),
        }


@dataclass(frozen=True)
class ModalPort:
    """A wave port on a Region face, for a Driven Modal solve.

    `integration_line_um` is the line, as two XYZ points in micrometres, that fixes
    the port's voltage path and polarity. Modal ports have no reference objects.
    """

    index: int
    name: str
    side: Side
    integration_line_um: tuple[tuple[float, float, float], tuple[float, float, float]]

    def __post_init__(self) -> None:
        if self.index not in {1, 2}:
            raise ValueError("two-port V1 uses port indices 1 and 2")
        object.__setattr__(self, "name", _stripped_text(self.name, "port.name"))
        if self.side not in {"+X", "-X", "+Y", "-Y", "+Z", "-Z"}:
            raise ValueError("port.side is invalid")
        if len(self.integration_line_um) != 2 or any(
            len(point) != 3 for point in self.integration_line_um
        ):
            raise ValueError("integration_line_um must be two xyz points")
        object.__setattr__(
            self,
            "integration_line_um",
            tuple(
                tuple(
                    _coerced_finite_number(value, "integration_line_um")
                    for value in point
                )
                for point in self.integration_line_um
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        """Serialize the integration line as two XYZ points in micrometres."""
        return {
            "index": self.index,
            "name": self.name,
            "side": self.side,
            "integration_line_um": [list(point) for point in self.integration_line_um],
        }


def _padding(
    values: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    """Check the six Region padding values, in PyAEDT's face order +X,-X,+Y,-Y,+Z,-Z."""
    if len(values) != 6:
        raise ValueError(
            "region_padding_um requires six PyAEDT face values: +X,-X,+Y,-Y,+Z,-Z"
        )
    result = tuple(
        _coerced_finite_number(value, "region_padding_um") for value in values
    )
    if any(value < 0 for value in result):
        raise ValueError("region_padding_um values must be >= 0")
    return result


def _padding_2d(
    values: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Check the four Q2D Region padding values, ordered +X,-X,+Y,-Y."""
    if len(values) != 4:
        raise ValueError("Q2D region_padding_um requires +X,-X,+Y,-Y values")
    result = tuple(
        _coerced_finite_number(value, "region_padding_um") for value in values
    )
    if any(value < 0 for value in result):
        raise ValueError("region_padding_um values must be >= 0")
    return result


def _normalize_common_spec(spec: Any) -> dict[str, PdkMaterial]:
    """Check tool versions, names, materials and vacuum; normalize them in place.

    Shared by every request family. Returns the validated material mapping.
    """
    if (
        str(spec.aedt_version) != REQUIRED_AEDT_VERSION
        or str(spec.pyaedt_version) != LOCKED_PYAEDT
    ):
        raise ValueError("V1 requires AEDT 2024.2 and PyAEDT 1.3.0")
    project = _project_filename(spec.project_name, "project_name")
    object.__setattr__(spec, "project_name", project.stem)
    object.__setattr__(
        spec, "design_name", _stripped_text(spec.design_name, "design_name")
    )
    if not isinstance(spec.materials, Mapping):
        raise TypeError("materials must be a PDK material_id mapping")
    materials = dict(spec.materials)
    if (
        not materials
        or any(not isinstance(item, PdkMaterial) for item in materials.values())
        or set(materials) != {item.material_id for item in materials.values()}
    ):
        raise ValueError("materials must map each PDK material_id to its record")
    vacuum_id = _stripped_text(spec.vacuum_material_id, "vacuum_material_id")
    vacuum = materials.get(vacuum_id)
    if (
        vacuum is None
        or vacuum.kind != "vacuum"
        or vacuum.is_superconducting
        or vacuum.library_name is None
        or vacuum.library_name.lower() != "vacuum"
    ):
        raise ValueError(
            "vacuum_material_id must reference non-superconducting PDK vacuum library_name 'vacuum'"
        )
    object.__setattr__(spec, "materials", materials)
    object.__setattr__(spec, "vacuum_material_id", vacuum_id)
    return materials


def _normalize_gds_spec(spec: Any) -> tuple[set[str], set[str]]:
    """Normalize the shared PDK/GDS model contract for 3D AEDT families."""
    materials = _normalize_common_spec(spec)
    gds = Path(spec.gds_path)
    if not str(gds) or gds.name in {"", "."}:
        raise ValueError("gds_path must name a GDS file")
    object.__setattr__(spec, "gds_path", gds)
    imports = tuple(spec.layer_imports)
    if (
        not imports
        or len({(item.layer, item.datatype) for item in imports}) != len(imports)
        or len({item.layer_name for item in imports}) != len(imports)
        or any(
            other.layer_name.startswith(item.layer_name)
            for item in imports
            for other in imports
            if item is not other
        )
    ):
        raise ValueError(
            "layer_imports must contain unique non-prefixing numeric and destination layer mappings"
        )
    object.__setattr__(spec, "layer_imports", imports)
    bindings = tuple(spec.object_bindings)
    if (
        not bindings
        or len({item.object_name for item in bindings}) != len(bindings)
        or {item.layer for item in bindings} != {item.layer for item in imports}
        or any(item.material_id not in materials for item in bindings)
    ):
        raise ValueError(
            "object_bindings must be unique and refer to declared import layers"
        )
    object.__setattr__(spec, "object_bindings", bindings)
    grounds = {item.object_name for item in bindings if item.role == "ground"}
    signals = {item.object_name for item in bindings if item.role == "signal"}
    substrates = [item for item in bindings if item.role == "substrate"]
    if not grounds or not signals or not substrates:
        raise ValueError(
            "HFSS model requires explicit signal, ground, and substrate bindings"
        )
    if any(
        not materials[item.material_id].is_superconducting
        for item in bindings
        if item.role in {"signal", "ground"}
    ):
        raise ValueError("signal and ground bindings require PDK superconductors")
    if any(
        materials[item.material_id].is_superconducting
        or materials[item.material_id].kind != "dielectric"
        for item in substrates
    ):
        raise ValueError(
            "substrate bindings require non-superconducting PDK dielectrics"
        )
    return grounds, signals


@dataclass(frozen=True)
class MatrixRunControl:
    """One Q3D or Q2D matrix setup with its adaptive meshing policy.

    The matrix is solved at the single frequency `frequency_ghz`. AEDT refines the
    mesh pass by pass: it runs between `minimum_passes` and `maximum_passes`,
    refines `percent_refinement` percent per pass, and counts a pass as converged
    when the change since the previous pass is below `convergence_percent`;
    `minimum_converged_passes` such passes are required.
    """

    setup_name: str
    frequency_ghz: float
    maximum_passes: int
    convergence_percent: float = 1.0
    minimum_passes: int = 1
    minimum_converged_passes: int = 1
    percent_refinement: float = 30.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "setup_name", _stripped_text(self.setup_name, "setup_name")
        )
        frequency = _coerced_finite_number(self.frequency_ghz, "frequency_ghz")
        if frequency <= 0:
            raise ValueError("matrix frequency_ghz must be > 0")
        refinement = _adaptive_controls(
            self.maximum_passes,
            self.minimum_passes,
            self.minimum_converged_passes,
            self.percent_refinement,
        )
        convergence = _coerced_finite_number(
            self.convergence_percent, "convergence_percent"
        )
        if convergence <= 0:
            raise ValueError("matrix convergence_percent must be > 0")
        object.__setattr__(self, "frequency_ghz", frequency)
        object.__setattr__(self, "convergence_percent", convergence)
        object.__setattr__(self, "percent_refinement", refinement)

    def to_payload(self) -> dict[str, Any]:
        """Serialize the setup name, frequency and adaptive-pass controls."""
        return {
            "setup_name": self.setup_name,
            "frequency_ghz": self.frequency_ghz,
            "maximum_passes": self.maximum_passes,
            "convergence_percent": self.convergence_percent,
            "minimum_passes": self.minimum_passes,
            "minimum_converged_passes": self.minimum_converged_passes,
            "percent_refinement": self.percent_refinement,
        }
