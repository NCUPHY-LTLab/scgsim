"""Q3D v3 request and source-geometry contract; no older Q3D parser."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any, Literal, cast, get_args

from scgsim.aedt.specs.common import (
    LOCKED_PYAEDT,
    OFFICIAL_PYAEDT_SOURCE_URL,
    Q3D_GEOMETRY_SOURCE_SCHEMA_VERSION,
    Q3D_SCHEMA_VERSION,
    LayerRole,
    MatrixRunControl,
    PdkMaterial,
    Side,
    finite_number,
    normalize_common_spec,
    project_name_from_payload,
    text_value,
    validate_region_padding,
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
class Q3dTerminalPair:
    """Caller-selected global terminal sides on exact source Entity identities.

    Each selected Entity must lower to exactly one body. Final Net ownership
    is validated by Q3dNetSpec after resolving these source identities.
    """

    source_entity_id: str
    source_side: Side
    sink_entity_id: str
    sink_side: Side

    def __post_init__(self) -> None:
        for name in ("source_entity_id", "sink_entity_id"):
            object.__setattr__(self, name, text_value(getattr(self, name), name))
        if self.source_side not in get_args(Side) or self.sink_side not in get_args(
            Side
        ):
            raise ValueError("Q3D source and sink sides are invalid")


@dataclass(frozen=True)
class Q3dNetSpec:
    """One exact connected Q3D net and optional signal source/sink."""

    name: str
    net_type: Literal["Signal", "Ground"]
    object_names: tuple[str, ...]
    source_object: str | None = None
    source_side: Side | None = None
    sink_object: str | None = None
    sink_side: Side | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", text_value(self.name, "net.name"))
        if self.net_type not in {"Signal", "Ground"}:
            raise ValueError("Q3D net_type must be Signal or Ground")
        objects = tuple(
            text_value(value, "net.object_name") for value in self.object_names
        )
        if not objects or len(set(objects)) != len(objects):
            raise ValueError("Q3D net object_names must be nonempty and unique")
        object.__setattr__(self, "object_names", objects)
        terminal_values = (
            self.source_object,
            self.source_side,
            self.sink_object,
            self.sink_side,
        )
        if self.net_type == "Ground":
            if any(value is not None for value in terminal_values):
                raise ValueError("Q3D Ground nets must not define source or sink")
            return
        if all(value is None for value in terminal_values):
            return
        if any(value is None for value in terminal_values):
            raise ValueError(
                "Q3D Signal terminals must be all absent or a complete source/sink pair"
            )
        source = text_value(self.source_object, "net.source_object")
        sink = text_value(self.sink_object, "net.sink_object")
        if source not in objects or sink not in objects:
            raise ValueError("Q3D source and sink objects must belong to their net")
        if self.source_side not in {
            "+X",
            "-X",
            "+Y",
            "-Y",
            "+Z",
            "-Z",
        } or self.sink_side not in {
            "+X",
            "-X",
            "+Y",
            "-Y",
            "+Z",
            "-Z",
        }:
            raise ValueError("Q3D source and sink sides are invalid")
        object.__setattr__(self, "source_object", source)
        object.__setattr__(self, "sink_object", sink)

    def to_payload(self) -> dict[str, Any]:
        """Return this request record in its canonical payload shape."""
        return {
            "name": self.name,
            "net_type": self.net_type,
            "object_names": list(self.object_names),
            "source_object": self.source_object,
            "source_side": self.source_side,
            "sink_object": self.sink_object,
            "sink_side": self.sink_side,
        }


@dataclass(frozen=True)
class Q3dBodySpec:
    """One normalized connected planar body, including holes, swept along +Z.

    Body identity is independent of final Net ownership. Optional source fields
    are absent for explicitly authored source-free bodies, never inferred.
    """

    body_id: str
    exterior_um: tuple[tuple[float, float], ...]
    holes_um: tuple[tuple[tuple[float, float], ...], ...]
    z_min_um: float
    z_max_um: float
    material_id: str
    physical_role: LayerRole
    net_id: str | None
    physical_layer_id: str | None = None
    source_entity_id: str | None = None
    source_polygon_id: str | None = None
    source_occurrence_path: str | None = None
    source_local_entity_id: str | None = None
    source_level: str | None = None
    source_layer_datatype: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        for field in ("body_id", "material_id"):
            object.__setattr__(self, field, text_value(getattr(self, field), field))
        if self.physical_role not in {"signal", "ground", "substrate"}:
            raise ValueError("Q3D body physical_role is invalid")
        for field in (
            "net_id",
            "source_entity_id",
            "source_polygon_id",
            "source_occurrence_path",
            "source_local_entity_id",
            "source_level",
        ):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, text_value(value, field))

        def ring(value: Any) -> tuple[tuple[float, float], ...]:
            """Describe the ring represented by this request."""
            points = tuple(tuple(point) for point in value)
            if len(points) < 3 or any(len(point) != 2 for point in points):
                raise ValueError("Q3D body rings require at least three XY points")
            return tuple((float(x), float(y)) for x, y in points)

        object.__setattr__(self, "exterior_um", ring(self.exterior_um))
        object.__setattr__(
            self, "holes_um", tuple(ring(hole) for hole in self.holes_um)
        )
        low, high = (
            finite_number(self.z_min_um, "z_min_um"),
            finite_number(self.z_max_um, "z_max_um"),
        )
        if high <= low:
            raise ValueError("Q3D bodies require positive thickness")
        object.__setattr__(self, "z_min_um", low)
        object.__setattr__(self, "z_max_um", high)
        pair = self.source_layer_datatype
        if pair is not None:
            if len(pair) != 2 or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in pair
            ):
                raise ValueError(
                    "source_layer_datatype must be a nonnegative integer pair"
                )
            object.__setattr__(self, "source_layer_datatype", tuple(pair))

    def to_payload(self) -> dict[str, Any]:
        """Return this request record in its canonical payload shape."""
        payload = asdict(self)
        if self.physical_layer_id is None:
            payload.pop("physical_layer_id")
        payload["exterior_um"] = [list(point) for point in self.exterior_um]
        payload["holes_um"] = [
            [list(point) for point in hole] for hole in self.holes_um
        ]
        if self.source_layer_datatype is not None:
            payload["source_layer_datatype"] = list(self.source_layer_datatype)
        return payload


def _is_sha256_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validated_geometry_files(
    files: Any, source: Mapping[str, Any]
) -> dict[str, dict[str, str]]:
    if not isinstance(files, Mapping) or set(files) != {
        "canonical_gds",
        "stack",
        "trace",
    }:
        raise ValueError("Q3D geometry_source files are invalid")
    copied_files: dict[str, dict[str, str]] = {}
    for key, reference in files.items():
        if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
            raise ValueError(f"Q3D geometry_source file reference is invalid: {key}")
        path = reference["path"]
        digest = reference["sha256"]
        if not isinstance(path, str) or not path:
            raise ValueError(f"Q3D geometry_source file path is invalid: {key}")
        if not _is_sha256_digest(digest):
            raise ValueError(f"Q3D geometry_source file hash is invalid: {key}")
        copied_files[key] = {"path": path, "sha256": digest}
    if copied_files["canonical_gds"]["sha256"] != source["source_gds_sha256"]:
        raise ValueError("Q3D source GDS digest differs from its file reference")
    if copied_files["stack"]["sha256"] != source["source_stack_sha256"]:
        raise ValueError("Q3D source stack digest differs from its file reference")
    return copied_files


def _validated_physical_ground_nets(
    value: Any, nets: tuple[Q3dNetSpec, ...]
) -> list[str]:
    net_by_name = {net.name: net for net in nets}
    if (
        not isinstance(value, list)
        or any(not isinstance(name, str) or not name for name in value)
        or len(set(value)) != len(value)
        or not set(value) <= set(net_by_name)
    ):
        raise ValueError("Q3D geometry_source physical_ground_nets are invalid")
    return value


def _q3d_geometry_source(
    value: Mapping[str, Any],
    *,
    bodies: tuple[Q3dBodySpec, ...],
    nets: tuple[Q3dNetSpec, ...],
) -> dict[str, Any]:
    """Bind canonical source attachments and normalized body declarations."""
    source = dict(value)
    expected_keys = {
        "schema_version",
        "source_gds_sha256",
        "source_geometry_sha256",
        "source_stack_sha256",
        "source_dbu_um",
        "files",
        "physical_ground_nets",
        "bodies",
    }
    if set(source) != expected_keys:
        raise ValueError("Q3D geometry_source fields are invalid")
    if source["schema_version"] != Q3D_GEOMETRY_SOURCE_SCHEMA_VERSION:
        raise ValueError("Q3D geometry_source schema is unsupported")
    for field in (
        "source_gds_sha256",
        "source_geometry_sha256",
        "source_stack_sha256",
    ):
        digest = source[field]
        if not _is_sha256_digest(digest):
            raise ValueError(f"Q3D geometry_source {field} is invalid")
    copied_files = _validated_geometry_files(source["files"], source)
    physical_ground_nets = _validated_physical_ground_nets(
        source["physical_ground_nets"], nets
    )

    dbu = finite_number(source["source_dbu_um"], "geometry_source.source_dbu_um")
    if dbu <= 0:
        raise ValueError("source_dbu_um must be positive")
    if source["bodies"] != [body.to_payload() for body in bodies]:
        raise ValueError("Q3D geometry_source bodies differ from the declared bodies")
    if any(
        body.physical_role != "ground"
        for body in bodies
        if body.net_id in physical_ground_nets
    ):
        raise ValueError("Q3D physical ground bodies must retain the ground role")
    source["files"] = copied_files
    source["bodies"] = [body.to_payload() for body in bodies]
    source["physical_ground_nets"] = list(physical_ground_nets)
    return source


def _validate_run_control(spec: Q3dSpec) -> None:
    if spec.run_control is not None and not isinstance(
        spec.run_control, MatrixRunControl
    ):
        raise TypeError("run_control must be MatrixRunControl or None")


def _validated_bodies(
    records: tuple[Q3dBodySpec, ...], materials: Mapping[str, PdkMaterial]
) -> tuple[Q3dBodySpec, ...]:
    bodies = tuple(records)
    if not bodies or any(not isinstance(body, Q3dBodySpec) for body in bodies):
        raise TypeError("Q3D bodies must contain Q3dBodySpec records")
    if len({body.body_id for body in bodies}) != len(bodies):
        raise ValueError("Q3D body identities must be unique")
    for body in bodies:
        material = materials[body.material_id]
        if body.physical_role == "substrate":
            if material.kind != "dielectric" or body.net_id is not None:
                raise ValueError(
                    "Q3D substrate bodies require dielectric material and no Net"
                )
        elif not material.is_superconducting or body.net_id is None:
            raise ValueError(
                "Q3D conductor bodies require a PDK superconductor and final Net"
            )
    return bodies


def _validated_nets(
    records: tuple[Q3dNetSpec, ...], bodies: tuple[Q3dBodySpec, ...]
) -> tuple[Q3dNetSpec, ...]:
    nets = tuple(records)
    if (
        not nets
        or len({net.name for net in nets}) != len(nets)
        or not any(net.net_type == "Signal" for net in nets)
    ):
        raise ValueError("Q3D requires at least one unique Signal net")
    owners = {name: net for net in nets for name in net.object_names}
    conductors = {
        body.body_id: body for body in bodies if body.physical_role != "substrate"
    }
    if len(owners) != sum(len(net.object_names) for net in nets) or set(owners) != set(
        conductors
    ):
        raise ValueError("Q3D nets must cover every conductor body exactly once")
    if any(owners[name].name != body.net_id for name, body in conductors.items()):
        raise ValueError("Q3D body final Net differs from Net membership")
    return nets


def _validate_net_terminals(solve_ac_rl: bool, nets: tuple[Q3dNetSpec, ...]) -> None:
    if not isinstance(solve_ac_rl, bool):
        raise TypeError("solve_ac_rl must be boolean")
    if solve_ac_rl and any(
        net.net_type == "Signal" and net.source_object is None for net in nets
    ):
        raise ValueError("Q3D AC/RL requires complete terminals for every Signal net")


def _validate_geometry_source(
    spec: Q3dSpec, bodies: tuple[Q3dBodySpec, ...], nets: tuple[Q3dNetSpec, ...]
) -> None:
    if spec.geometry_source is None:
        return
    if not isinstance(spec.geometry_source, Mapping):
        raise TypeError("geometry_source must be a JSON object")
    object.__setattr__(
        spec,
        "geometry_source",
        _q3d_geometry_source(spec.geometry_source, bodies=bodies, nets=nets),
    )


def _validate_grounded_region(
    grounded_region_net: str | None,
    solve_ac_rl: bool,
    nets: tuple[Q3dNetSpec, ...],
) -> None:
    if grounded_region_net is None:
        return
    text_value(grounded_region_net, "grounded_region_net")
    if solve_ac_rl:
        raise ValueError("grounded_region_net requires solve_ac_rl=False")
    if grounded_region_net in {net.name for net in nets}:
        raise ValueError("grounded_region_net must name a new dedicated enclosure net")


@dataclass(frozen=True)
class Q3dSpec:
    """One body-backed Q3D model with optional matrix setup (v3 only).

    Explicit run_control=None declares model preparation without solve setup.
    Non-null controls retain the existing C/G and optional AC R/L payload.
    """

    modeling: Modeling
    project_name: str
    design_name: str
    materials: Mapping[str, PdkMaterial]
    vacuum_material_id: str
    bodies: tuple[Q3dBodySpec, ...]
    nets: tuple[Q3dNetSpec, ...]
    run_control: MatrixRunControl | None
    region_padding_um: tuple[float, float, float, float, float, float]
    physical_layers: tuple[PhysicalLayerDeclaration | PhysicalLayerSpec, ...] = ()
    physical_representation: PhysicalRepresentationPlan | None = None
    solve_ac_rl: bool = True
    grounded_region_net: str | None = None
    aedt_version: str = dataclass_field(kw_only=True)
    pyaedt_version: str = LOCKED_PYAEDT
    schema_version: Literal["scgsim.aedt.q3d.v3"] = Q3D_SCHEMA_VERSION
    geometry_source: Mapping[str, Any] | None = None

    @property
    def mode(self) -> Literal["q3d"]:
        """Return the solver family represented by this request."""
        return "q3d"

    def __post_init__(self) -> None:
        request_modeling(self)
        if self.schema_version != Q3D_SCHEMA_VERSION:
            raise ValueError("unsupported Q3D schema")
        materials = normalize_common_spec(self)
        _validate_run_control(self)
        bodies = _validated_bodies(self.bodies, materials)
        object.__setattr__(self, "bodies", bodies)
        object.__setattr__(
            self, "region_padding_um", validate_region_padding(self.region_padding_um)
        )
        nets = _validated_nets(self.nets, bodies)
        _validate_net_terminals(self.solve_ac_rl, nets)
        _validate_geometry_source(self, bodies, nets)
        _validate_grounded_region(self.grounded_region_net, self.solve_ac_rl, nets)
        object.__setattr__(self, "nets", nets)

    @property
    def effective_bodies(self) -> tuple[dict[str, Any], ...]:
        """Return source bodies projected through the declared physical model."""
        mapping = request_modeling(self)
        return tuple(effective_record(item, mapping) for item in self.bodies)

    @property
    def effective_region(self) -> dict[str, Any] | None:
        """Project the complete originally requested envelope with the common plan."""
        mapping = request_modeling(self)

        def bounds(records):
            """Describe the bounds represented by this request."""
            points = [point for body in records for point in body["exterior_um"]]
            return {
                "x_min_um": min(point[0] for point in points),
                "x_max_um": max(point[0] for point in points),
                "y_min_um": min(point[1] for point in points),
                "y_max_um": max(point[1] for point in points),
                "z_min_um": min(body["z_min_um"] for body in records),
                "z_max_um": max(body["z_max_um"] for body in records),
            }

        original = [item.to_payload() for item in self.bodies]
        effective = [
            {
                **record,
                "z_min_um": record["effective_z_min_um"],
                "z_max_um": record["effective_z_max_um"],
            }
            for record in self.effective_bodies
        ]
        keys = (
            "x_plus_um",
            "x_minus_um",
            "y_plus_um",
            "y_minus_um",
            "z_plus_um",
            "z_minus_um",
        )
        return mapping.project_enclosure(
            source_body_bounds_um=bounds(original),
            effective_body_bounds_um=bounds(effective),
            source_padding_um=dict(zip(keys, self.region_padding_um, strict=True)),
        )

    @property
    def effective_region_padding_um(self) -> tuple[float, ...]:
        """Return the effective enclosure padding for this request."""
        trace = self.effective_region
        if trace is None:
            return self.region_padding_um
        return tuple(
            trace["effective_padding_um"][key]
            for key in (
                "x_plus_um",
                "x_minus_um",
                "y_plus_um",
                "y_minus_um",
                "z_plus_um",
                "z_minus_um",
            )
        )

    def to_payload(self) -> dict[str, Any]:
        """Return this request record in its canonical payload shape."""
        payload = {
            **modeling_payload(self),
            "schema_version": self.schema_version,
            "mode": self.mode,
            "aedt": {"requested_version": self.aedt_version},
            "pyaedt": {
                "locked_version": self.pyaedt_version,
                "official_source": OFFICIAL_PYAEDT_SOURCE_URL,
            },
            "project": {"name": self.project_name, "design": self.design_name},
            "materials": {
                key: item.to_payload() for key, item in self.materials.items()
            },
            "vacuum_material_id": self.vacuum_material_id,
            "bodies": (list(self.effective_bodies)),
            "nets": [net.to_payload() for net in self.nets],
            "run_control": None
            if self.run_control is None
            else self.run_control.to_payload(),
            "region_padding_um": list(self.region_padding_um),
            "solve_ac_rl": self.solve_ac_rl,
            "grounded_region_net": self.grounded_region_net,
        }
        if self.geometry_source is not None:
            payload["geometry_source"] = dict(self.geometry_source)
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        base_dir: Path | None = None,
    ) -> Q3dSpec:
        """Construct this request record from its serialized payload."""
        if payload.get("schema_version") != Q3D_SCHEMA_VERSION:
            raise ValueError("unsupported Q3D schema")
        if {"gds", "layer_imports", "object_bindings"} & set(payload):
            raise ValueError("Q3D v3 does not accept GDS-piece inputs")
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
            project_name=project_name_from_payload(
                payload.get("project", {}).get("name")
            ),
            design_name=text_value(
                payload.get("project", {}).get("design"), "project.design"
            ),
            materials={
                key: PdkMaterial(**item) for key, item in payload["materials"].items()
            },
            vacuum_material_id=payload["vacuum_material_id"],
            bodies=tuple(
                Q3dBodySpec(**source_record_payload(body)) for body in payload["bodies"]
            ),
            nets=tuple(Q3dNetSpec(**net) for net in payload["nets"]),
            run_control=(
                None
                if payload["run_control"] is None
                else MatrixRunControl(**payload["run_control"])
            ),
            region_padding_um=tuple(payload["region_padding_um"]),
            solve_ac_rl=payload.get("solve_ac_rl", True),
            grounded_region_net=payload.get("grounded_region_net"),
            aedt_version=payload["aedt"]["requested_version"],
            pyaedt_version=payload["pyaedt"]["locked_version"],
            geometry_source=payload.get("geometry_source"),
        )
        verify_effective_records(payload["bodies"], result.effective_bodies)
        return result
