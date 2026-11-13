"""Assemble the detached PDK stack from component-local source declarations."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from scgsim.geometry.models.refusals import Declared
from scgsim.geometry.source.normalization import (
    coupon_domain_bounds,
    gds_layer,
    normalize_material_records,
    semantic_layer_records,
    solution_regions,
)

_PATH_SEGMENT = re.compile(r"[A-Za-z][A-Za-z0-9_.-]*\Z")

_FORBIDDEN_AUTHOR_KEYS = {
    "net_id",
    "signal_group",
    "semantic_group_id",
    "physical_group_id",
    "physical_group_name",
}


def source_path(value: Any) -> str:
    """Validate one slash-separated component or occurrence identity."""
    if (
        not isinstance(value, str)
        or not value
        or any(_PATH_SEGMENT.fullmatch(part) is None for part in value.split("/"))
    ):
        raise ValueError(
            "instance and entity paths need non-empty slash-separated names"
        )
    return value


def occurrence_transform(
    reference: Any,
) -> tuple[float, float, float, float, float, float]:
    """The affine map ``(a, b, c, d, x, y)`` of a reference in its parent; only rigid rotation and mirror."""
    from klayout import db as kdb  # noqa: PLC0415 - transforms are needed only during occurrence registration

    transform = reference.dcplx_trans
    origin = transform * kdb.DPoint(0.0, 0.0)
    x_axis = transform * kdb.DPoint(1.0, 0.0)
    y_axis = transform * kdb.DPoint(0.0, 1.0)
    values = (
        float(x_axis.x - origin.x),
        float(y_axis.x - origin.x),
        float(x_axis.y - origin.y),
        float(y_axis.y - origin.y),
        float(origin.x),
        float(origin.y),
    )
    if not all(math.isfinite(item) for item in values):
        raise ValueError("instance transform must be finite")
    a, b, c, d, _, _ = values
    if not math.isclose(abs(a * d - b * c), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("GeometryPlan supports rigid rotation and mirror only")
    return values


def composed_transform(
    parent: tuple[float, ...], child: tuple[float, ...]
) -> tuple[float, float, float, float, float, float]:
    """The affine map of ``child`` placed in ``parent``: first the child's, then the parent's."""
    a, b, c, d, x, y = parent
    e, f, g, h, u, v = child
    return (
        a * e + b * g,
        a * f + b * h,
        c * e + d * g,
        c * f + d * h,
        a * u + b * v + x,
        c * u + d * v + y,
    )


def transform_source_point(
    transform: tuple[float, ...], value: Sequence[float]
) -> list[float]:
    """Transform one local XY selector into its assembled source position."""
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError("selector and port point must be a two-item coordinate")
    x, y = float(value[0]), float(value[1])
    a, b, c, d, u, v = transform
    result = [a * x + b * y + u, c * x + d * y + v]
    if not all(math.isfinite(item) for item in result):
        raise ValueError("transformed point is not finite")
    return result


# A direction component within this of zero, beside a unit one, is float noise from a rotation's
# cosine and sine (cos(pi/2) is 6e-17), never an angle a layout draws.
_DIRECTION_FLOAT_NOISE = 1e-12
_QUARTER_TURN_DIRECTIONS = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))


def transform_source_direction(
    transform: tuple[float, ...], angle_degrees: float
) -> list[float]:
    """Transform an outward port angle into a normalized XY direction.

    A port angle that is a whole number of quarter turns uses its exact cosine and sine, and a
    component the transform leaves within float noise of zero is zero, so a port on a quarter
    turn points exactly along an axis: the rectangle cut from its sheet is then axis-aligned.
    """
    x, y = _port_unit_direction(float(angle_degrees))
    a, b, c, d, _, _ = transform
    result = [a * x + b * y, c * x + d * y]
    magnitude = math.hypot(*result)
    if not math.isfinite(magnitude) or magnitude == 0:
        raise ValueError("transformed port direction is invalid")
    result = [
        0.0 if abs(value / magnitude) <= _DIRECTION_FLOAT_NOISE else value
        for value in result
    ]
    magnitude = math.hypot(*result)
    return [result[0] / magnitude, result[1] / magnitude, 0.0]


def _port_unit_direction(angle_degrees: float) -> tuple[float, float]:
    """The unit vector of a port angle, exact on the quarter turns."""
    turns, remainder = divmod(angle_degrees, 90.0)
    if remainder == 0.0:
        return _QUARTER_TURN_DIRECTIONS[int(turns) % 4]
    angle = math.radians(angle_degrees)
    return math.cos(angle), math.sin(angle)


def component_semantics(component: Any, path: str) -> Mapping[str, Any]:
    """Read and validate one component's supported local semantics record."""
    try:
        raw = component.info["component_semantics"]
    except (AttributeError, KeyError) as exc:
        raise ValueError(f"{path} has no component_semantics v2 declaration") from exc
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 2:
        raise ValueError(f"{path} requires component_semantics schema_version=2")
    if set(raw) - {
        "schema_version",
        "conductor_regions",
        "ports",
        "ground_plane_contributions",
        "boundary_curves",
        "boundary_reconstruction",
        "metadata",
    }:
        raise ValueError(f"{path} has unknown component_semantics v2 fields")
    metadata = raw.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError(f"{path} component metadata must be a mapping")
    if "signal_group" in metadata:
        raise ValueError(f"{path} cannot author a public Group ID")
    records = raw.get("conductor_regions", ())
    ports = raw.get("ports", ())
    contributions = raw.get("ground_plane_contributions", ())
    if any(
        isinstance(value, (str, bytes)) or not isinstance(value, Sequence)
        for value in (records, ports, contributions)
    ):
        raise TypeError(
            f"{path} conductor_regions, ports, and ground_plane_contributions must be sequences"
        )
    return raw


@dataclass(frozen=True)
class GeometryPlanStackAssembler:
    """Build stack records from one plan's declared component occurrences."""

    component: Any
    root_id: str
    instances: Mapping[str, tuple[Any, Any, tuple[float, ...]]]
    nets: Mapping[str, tuple[str, ...]]
    layer_stack: Any
    material_records: Mapping[str, Mapping[str, Any]]
    coupon_padding_um: float

    def assemble(self) -> dict[str, Any]:
        """Resolve PDK regions, transform source records, and attach provenance."""
        levels = getattr(self.layer_stack, "layers", None)
        if not isinstance(levels, Mapping):
            raise TypeError("layer_stack must expose layers")
        bounds = coupon_domain_bounds(self.component, self.coupon_padding_um)
        materials = normalize_material_records(self.material_records)
        regions = solution_regions(levels, materials=materials, bounds=bounds)
        if not regions:
            raise ValueError("plan needs at least one PDK solution region")

        net_owners = {
            entity_id: net
            for net, members in self.nets.items()
            for entity_id in members
        }
        declarations, ports, contributions = self._component_records(net_owners)
        layers = semantic_layer_records(
            declarations,
            levels=levels,
            materials=materials,
            solution_region_ids=regions,
            require_net=False,
        )
        by_id = {record["semantic_id"]: record for record in layers}
        if len(by_id) != len(layers):
            raise ValueError("GeometryPlan Entity IDs must be unique")
        self._validate_physical_planes(layers, regions, levels)
        self._apply_ground_contributions(contributions, by_id)
        return {
            "solution_regions": regions,
            "materials": materials,
            "layers": layers,
            "metadata": {
                "adapter": "scgsim.sgb.GeometryPlan",
                "component_semantics_schema_version": 2,
                "coupon_domain_bounds_um": dict(bounds),
                "coupon_padding_um": self.coupon_padding_um,
                "top_cell_name": self.component.name,
                "port_sheet_source_layers": ports,
            },
        }

    def _component_records(
        self, net_owners: Mapping[str, str]
    ) -> tuple[
        list[dict[str, Any]], list[dict[str, Any]], list[tuple[str, dict[str, Any]]]
    ]:
        declarations: list[dict[str, Any]] = []
        ports: list[dict[str, Any]] = []
        contributions: list[tuple[str, dict[str, Any]]] = []
        contribution_ids: set[tuple[str, str]] = set()
        for path, (cell, _, transform) in self.instances.items():
            semantics = component_semantics(cell, path)
            declarations.extend(
                _entity_declaration(path, raw, transform, net_owners)
                for raw in semantics.get("conductor_regions", ())
            )
            for raw in semantics.get("ground_plane_contributions", ()):
                contribution = _ground_contribution(
                    path, raw, transform, contribution_ids
                )
                contributions.append(contribution)
            for raw in semantics.get("ports", ()):
                ports.append(_source_port(path, cell, raw, transform, len(ports) + 1))
        return declarations, ports, contributions

    def _validate_physical_planes(
        self,
        layers: list[dict[str, Any]],
        regions: dict[str, Any],
        levels: Mapping[str, Any],
    ) -> None:
        physical_planes: set[str] = set()
        for record in layers:
            if (
                record["geometry"].get("geometry_source")
                != "die_face_minus_ground_mask"
            ):
                continue
            if record["metadata"]["source_occurrence_path"] != self.root_id:
                # GDSFactory's layer mapping is needed only for occurrence Ground.
                from gdsfactory.pdk import get_layer_tuple  # noqa: PLC0415 - occurrence Ground only
                from gdsfactory.technology.layer_stack import (  # noqa: PLC0415 - occurrence Ground only
                    LogicalLayer,
                )

                bounds_ref = record["geometry"].get("plane_bounds_ref")
                if bounds_ref not in regions or bounds_ref not in levels:
                    raise ValueError(
                        "occurrence Ground requires its declared plane_bounds_ref"
                    )
                source_layer = levels[bounds_ref].layer
                if not isinstance(source_layer, LogicalLayer):
                    raise ValueError(
                        "occurrence Ground bounds require a source logical layer"
                    )
                regions[bounds_ref]["metadata"]["source_layer_datatype"] = list(
                    get_layer_tuple(source_layer.layer)
                )
            level = record["metadata"]["logical_layer_id"]
            if level in physical_planes:
                raise ValueError(
                    f"multiple ground planes claim physical level {level!r}"
                )
            physical_planes.add(level)

    def _apply_ground_contributions(
        self,
        contributions: list[tuple[str, dict[str, Any]]],
        by_id: Mapping[str, dict[str, Any]],
    ) -> None:
        for plane_id, contribution in contributions:
            target = by_id.get(f"{self.root_id}/{plane_id}")
            if (
                target is None
                or target["geometry"].get("geometry_source")
                != "die_face_minus_ground_mask"
            ):
                raise ValueError(
                    f"ground contribution targets no root derived plane {plane_id!r}"
                )
            if target["metadata"]["logical_layer_id"] != contribution["level"]:
                raise ValueError(
                    f"ground contribution {contribution['source_local_id']!r} level differs from target plane"
                )
            geometry = target["geometry"]
            if "include_layer" in geometry or "include_selector_points_um" in geometry:
                raise ValueError(
                    "root ground plane cannot mix global includes with source contributions"
                )
            geometry.setdefault("source_occurrence_includes", []).append(contribution)
            target["metadata"].setdefault(
                "ground_plane_contribution_sources", []
            ).append(
                {
                    "source_occurrence_path": contribution["source_occurrence_path"],
                    "source_local_id": contribution["source_local_id"],
                    "level": contribution["level"],
                    "layer": contribution["layer"],
                }
            )


def _entity_declaration(
    path: str,
    raw: Any,
    transform: tuple[float, ...],
    net_owners: Mapping[str, str],
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise TypeError(f"{path} conductor records must be mappings")
    if set(raw) & _FORBIDDEN_AUTHOR_KEYS:
        raise ValueError(f"{path} v2 entity cannot author Net or Group")
    metadata = raw.get("metadata", {})
    if not isinstance(metadata, Mapping) or set(metadata) & _FORBIDDEN_AUTHOR_KEYS:
        raise ValueError(f"{path} v2 entity metadata cannot author Net or Group")
    local_id = source_path(raw.get("semantic_id"))
    if "/" in local_id:
        raise ValueError("component-local entity ID must be one segment")
    entity_id = f"{path}/{local_id}"
    geometry = dict(raw.get("geometry", {}))
    for key in ("selector_point_um",):
        if key in geometry:
            geometry[key] = transform_source_point(transform, geometry[key])
    for key in ("include_selector_points_um", "exclude_selector_points_um"):
        if key in geometry:
            geometry[key] = [
                transform_source_point(transform, point) for point in geometry[key]
            ]
    return {
        **raw,
        "semantic_id": entity_id,
        "net_id": net_owners.get(entity_id),
        "geometry": geometry,
        "metadata": {
            **metadata,
            "source_occurrence_path": path,
            "source_local_entity_id": local_id,
            "source_semantic_id": entity_id,
        },
    }


def _ground_contribution(
    path: str,
    raw: Any,
    transform: tuple[float, ...],
    contribution_ids: set[tuple[str, str]],
) -> tuple[str, dict[str, Any]]:
    if not isinstance(raw, Mapping) or set(raw) != {
        "local_id",
        "plane_id",
        "level",
        "layer",
        "selector_point_um",
    }:
        raise ValueError(
            f"{path} ground contribution needs local_id, plane_id, level, layer, selector_point_um"
        )
    local_id, plane_id, level = (
        source_path(raw[key]) for key in ("local_id", "plane_id", "level")
    )
    if any("/" in value for value in (local_id, plane_id, level)):
        raise ValueError("ground contribution IDs must be component-local names")
    key = (path, local_id)
    if key in contribution_ids:
        raise ValueError(f"duplicate ground contribution {path}/{local_id}")
    contribution_ids.add(key)
    layer = gds_layer(
        raw["layer"],
        Declared(
            f"ground contribution {path}/{local_id}",
            "the component's component_semantics ground_plane_contributions",
        ),
        "layer",
    )
    return plane_id, {
        "source_occurrence_path": path,
        "source_local_id": local_id,
        "level": level,
        "layer": list(layer),
        "selector_point_um": transform_source_point(
            transform, raw["selector_point_um"]
        ),
    }


def _source_port(
    path: str,
    cell: Any,
    raw: Any,
    transform: tuple[float, ...],
    port_index: int,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != {
        "name",
        "layer",
        "target_layer",
    }:
        raise ValueError(f"{path} port requires name, layer, target_layer")
    name = source_path(raw["name"])
    if "/" in name:
        raise ValueError("component-local port name must be one segment")
    port = cell.ports[name]
    layer = gds_layer(
        raw["layer"],
        Declared(f"port {path}/{name}", "the component's component_semantics ports"),
        "layer",
    )
    raw_port_layer = port.layer
    if hasattr(raw_port_layer, "layer") and hasattr(raw_port_layer, "datatype"):
        port_layer = raw_port_layer
    else:
        port_layer = port.kcl.layout.get_info(int(str(raw_port_layer)))
    if (int(port_layer.layer), int(port_layer.datatype)) != layer:
        raise ValueError(f"{path}/{name} authored port layer differs")
    return {
        "layer": layer[0],
        "datatype": layer[1],
        "name": f"{path}/{name}",
        "source": "palace_lumped_port_sheet",
        "port_index": port_index,
        "target_layer": raw["target_layer"],
        "direction": transform_source_direction(transform, float(port.orientation)),
        "direction_sign_convention": "gdsfactory_port_orientation_outward",
        "selector_point_um": transform_source_point(transform, port.center),
        "source_occurrence_path": path,
    }
