"""Q3D request: extract the capacitance matrix (and optionally AC R/L) of a 3D model.

Q3D solves electrostatics (capacitance C and conductance G) and, optionally,
AC resistance and inductance (R and L) between named nets. The request describes
the conductors as extruded planar bodies, groups them into nets, and records the
layout source they came from. Only schema v3 is read. Nothing here talks to AEDT.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from scgsim.aedt._data import _stripped_text
from scgsim.aedt.specs.common import (
    LOCKED_PYAEDT,
    OFFICIAL_PYAEDT_SOURCE_URL,
    Q3D_GEOMETRY_SOURCE_SCHEMA_VERSION,
    Q3D_SCHEMA_VERSION,
    REQUIRED_AEDT_VERSION,
    LayerRole,
    MatrixRunControl,
    PdkMaterial,
    Side,
    _coerced_finite_number,
    _normalize_common_spec,
    _padding,
    _project_name_from_payload,
)
from scgsim.aedt.specs.modeling import (
    Modeling,
    PhysicalLayerDeclaration,
    PhysicalRepresentationPlan,
    _effective_record,
    _modeling_payload,
    _physical_fields_from_payload,
    _request_modeling,
    _source_record_payload,
    _verify_effective_records,
)


@dataclass(frozen=True)
class Q3dNetSpec:
    """One Q3D net: the conductor bodies that are electrically one node.

    A `Signal` net may name a `source_object` and `sink_object` with the Region
    side each sits on; AC R/L needs them to define where current enters and leaves.
    A `Ground` net never has terminals. Source and sink are all given or all absent.
    """

    name: str
    net_type: Literal["Signal", "Ground"]
    object_names: tuple[str, ...]
    source_object: str | None = None
    source_side: Side | None = None
    sink_object: str | None = None
    sink_side: Side | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _stripped_text(self.name, "net.name"))
        if self.net_type not in {"Signal", "Ground"}:
            raise ValueError("Q3D net_type must be Signal or Ground")
        objects = tuple(
            _stripped_text(value, "net.object_name") for value in self.object_names
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
        source = _stripped_text(self.source_object, "net.source_object")
        sink = _stripped_text(self.sink_object, "net.sink_object")
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
        """Serialize the net with its source and sink terminal fields, which are null when
        unused.
        """
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
    """One connected planar body, including holes, extruded along +Z.

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
            object.__setattr__(self, field, _stripped_text(getattr(self, field), field))
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
                object.__setattr__(self, field, _stripped_text(value, field))

        def ring(value: Any) -> tuple[tuple[float, float], ...]:
            points = tuple(tuple(point) for point in value)
            if len(points) < 3 or any(len(point) != 2 for point in points):
                raise ValueError("Q3D body rings require at least three XY points")
            return tuple((float(x), float(y)) for x, y in points)

        object.__setattr__(self, "exterior_um", ring(self.exterior_um))
        object.__setattr__(
            self, "holes_um", tuple(ring(hole) for hole in self.holes_um)
        )
        low, high = (
            _coerced_finite_number(self.z_min_um, "z_min_um"),
            _coerced_finite_number(self.z_max_um, "z_max_um"),
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
        """Serialize every field, leaving out `physical_layer_id` when it is unset; points
        become `[x, y]` lists.
        """
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
        if not _is_sha256(source[field]):
            raise ValueError(f"Q3D geometry_source {field} is invalid")
    copied_files = _checked_source_files(source["files"])
    if copied_files["canonical_gds"]["sha256"] != source["source_gds_sha256"]:
        raise ValueError("Q3D source GDS digest differs from its file reference")
    if copied_files["stack"]["sha256"] != source["source_stack_sha256"]:
        raise ValueError("Q3D source stack digest differs from its file reference")

    physical_ground_nets = source["physical_ground_nets"]
    net_names = {net.name for net in nets}
    if (
        not isinstance(physical_ground_nets, list)
        or any(not isinstance(name, str) or not name for name in physical_ground_nets)
        or len(set(physical_ground_nets)) != len(physical_ground_nets)
        or not set(physical_ground_nets) <= net_names
    ):
        raise ValueError("Q3D geometry_source physical_ground_nets are invalid")

    dbu = _coerced_finite_number(
        source["source_dbu_um"], "geometry_source.source_dbu_um"
    )
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


def _is_sha256(value: Any) -> bool:
    """True for a 64-character lowercase hexadecimal digest string."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _checked_source_files(files: Any) -> dict[str, dict[str, str]]:
    """Validate the canonical GDS, stack and trace file references of a source."""
    if not isinstance(files, Mapping) or set(files) != {
        "canonical_gds",
        "stack",
        "trace",
    }:
        raise ValueError("Q3D geometry_source files are invalid")
    copied: dict[str, dict[str, str]] = {}
    for key, reference in files.items():
        if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
            raise ValueError(f"Q3D geometry_source file reference is invalid: {key}")
        path = reference["path"]
        if not isinstance(path, str) or not path:
            raise ValueError(f"Q3D geometry_source file path is invalid: {key}")
        if not _is_sha256(reference["sha256"]):
            raise ValueError(f"Q3D geometry_source file hash is invalid: {key}")
        copied[key] = {"path": path, "sha256": reference["sha256"]}
    return copied


@dataclass(frozen=True)
class Q3dSpec:
    """A Q3D model built from extruded bodies, with an optional matrix setup.

    `bodies` are the conductors and substrate; `nets` partition the conductor
    bodies. `region_padding_um` pads the solution Region on +X, -X, +Y, -Y, +Z and
    -Z. `run_control=None` declares preparation without a solve setup; otherwise the
    setup solves C and G, plus AC R/L when `solve_ac_rl` is true. A
    `grounded_region_net` names a new net for the Region boundary and is only
    allowed when `solve_ac_rl` is false. `geometry_source` links the model back to
    its GDS and stack files by digest.
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
    physical_layers: tuple[PhysicalLayerDeclaration, ...] = ()
    physical_representation: PhysicalRepresentationPlan | None = None
    solve_ac_rl: bool = True
    grounded_region_net: str | None = None
    aedt_version: str = REQUIRED_AEDT_VERSION
    pyaedt_version: str = LOCKED_PYAEDT
    schema_version: Literal["scgsim.aedt.q3d.v3"] = Q3D_SCHEMA_VERSION
    geometry_source: Mapping[str, Any] | None = None

    @property
    def mode(self) -> Literal["q3d"]:
        """Always `"q3d"`; written into the payload as `mode`."""
        return "q3d"

    def __post_init__(self) -> None:
        _request_modeling(self)
        if self.schema_version != Q3D_SCHEMA_VERSION:
            raise ValueError("unsupported Q3D schema")
        materials = _normalize_common_spec(self)
        if self.run_control is not None and not isinstance(
            self.run_control, MatrixRunControl
        ):
            raise TypeError("run_control must be MatrixRunControl or None")
        bodies = self._checked_bodies(materials)
        object.__setattr__(self, "bodies", bodies)
        object.__setattr__(self, "region_padding_um", _padding(self.region_padding_um))
        nets = self._checked_nets(bodies)
        if self.geometry_source is not None:
            if not isinstance(self.geometry_source, Mapping):
                raise TypeError("geometry_source must be a JSON object")
            object.__setattr__(
                self,
                "geometry_source",
                _q3d_geometry_source(self.geometry_source, bodies=bodies, nets=nets),
            )
        self._check_grounded_region_net(nets)
        object.__setattr__(self, "nets", nets)

    def _checked_bodies(
        self, materials: Mapping[str, PdkMaterial]
    ) -> tuple[Q3dBodySpec, ...]:
        """Bodies are unique, and each material matches the body's physical role."""
        bodies = tuple(self.bodies)
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

    def _checked_nets(self, bodies: tuple[Q3dBodySpec, ...]) -> tuple[Q3dNetSpec, ...]:
        """Nets partition the conductor bodies exactly and agree with each body's Net."""
        nets = tuple(self.nets)
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
        if len(owners) != sum(len(net.object_names) for net in nets) or set(
            owners
        ) != set(conductors):
            raise ValueError("Q3D nets must cover every conductor body exactly once")
        if any(owners[name].name != body.net_id for name, body in conductors.items()):
            raise ValueError("Q3D body final Net differs from Net membership")
        if not isinstance(self.solve_ac_rl, bool):
            raise TypeError("solve_ac_rl must be boolean")
        # AC R/L needs a source and sink on every Signal net to define the current path.
        if self.solve_ac_rl and any(
            net.net_type == "Signal" and net.source_object is None for net in nets
        ):
            raise ValueError(
                "Q3D AC/RL requires complete terminals for every Signal net"
            )
        return nets

    def _check_grounded_region_net(self, nets: tuple[Q3dNetSpec, ...]) -> None:
        """The grounded-Region enclosure Net must be new and excludes AC R/L.

        Per docs/specs/aedt-runtime.qmd, the grounded Region is a
        capacitance/conductance-only construction and makes no DC or AC R/L
        return-path claim, so the combination fails closed.
        """
        grounded_region_net = self.grounded_region_net
        if grounded_region_net is None:
            return
        _stripped_text(grounded_region_net, "grounded_region_net")
        if self.solve_ac_rl:
            raise ValueError("grounded_region_net requires solve_ac_rl=False")
        if grounded_region_net in {net.name for net in nets}:
            raise ValueError(
                "grounded_region_net must name a new dedicated enclosure net"
            )

    @property
    def effective_bodies(self) -> tuple[dict[str, Any], ...]:
        """Bodies with the physical-layer plan applied, carrying both source and effective
        Z.
        """
        mapping = _request_modeling(self)
        return tuple(_effective_record(item, mapping) for item in self.bodies)

    @property
    def effective_region(self) -> dict[str, Any]:
        """Project the complete originally requested envelope with the common plan."""
        mapping = _request_modeling(self)

        def bounds(records):
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
        """Region padding in effective (native) space, ordered +X, -X, +Y, -Y, +Z, -Z."""
        trace = self.effective_region
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
        """Serialize with the effective bodies of the bound plan."""
        payload = {
            **_modeling_payload(self),
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
            "bodies": list(self.effective_bodies),
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
        """Rebuild from a v3 payload and verify the stored effective bodies against the
        plan.
        """
        if payload.get("schema_version") != Q3D_SCHEMA_VERSION:
            raise ValueError("unsupported Q3D schema")
        if {"gds", "layer_imports", "object_bindings"} & set(payload):
            raise ValueError("Q3D v3 does not accept GDS-piece inputs")
        result = cls(
            modeling=payload.get("modeling"),
            **_physical_fields_from_payload(payload),
            project_name=_project_name_from_payload(
                payload.get("project", {}).get("name")
            ),
            design_name=_stripped_text(
                payload.get("project", {}).get("design"), "project.design"
            ),
            materials={
                key: PdkMaterial(**item) for key, item in payload["materials"].items()
            },
            vacuum_material_id=payload["vacuum_material_id"],
            bodies=tuple(
                Q3dBodySpec(**_source_record_payload(body))
                for body in payload["bodies"]
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
        _verify_effective_records(payload["bodies"], result.effective_bodies)
        return result
