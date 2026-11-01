"""Compose the semantic level stack of a `GeometryPlan` from its registered cells.

Owns: reading each registered cell's `component_semantics` (schema version 2),
moving its local points into root coordinates, and assembling the stack mapping
that `build_gds_stack_geometry_input` consumes.

Does not own: instance registration, nets, GDS writing or the snapshot
(`plan.py`), or lowering the stack into geometry (`adapter.py`).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from scgsim.geometry.source._normalization import (
    coupon_domain_bounds,
    gds_layer,
    normalized_materials,
    semantic_layer_records,
    solution_regions,
)

# One instance: (cell, GDS reference or None for the root, affine transform to root).
InstanceRecord = tuple[Any, Any, tuple[float, ...]]

_DERIVED_GROUND = "die_face_minus_ground_mask"


_PATH_SEGMENT = re.compile(r"[A-Za-z][A-Za-z0-9_.-]*\Z")


_FORBIDDEN_AUTHOR_KEYS = {
    "net_id",
    "signal_group",
    "semantic_group_id",
    "physical_group_id",
    "physical_group_name",
}


def checked_entity_path(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(_PATH_SEGMENT.fullmatch(part) is None for part in value.split("/"))
    ):
        raise ValueError(
            "instance and entity paths need non-empty slash-separated names"
        )
    return value


def affine(reference: Any) -> tuple[float, float, float, float, float, float]:
    from klayout import db as kdb

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


def compose(
    parent: tuple[float, ...], child: tuple[float, ...]
) -> tuple[float, float, float, float, float, float]:
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


def transformed_point(
    transform: tuple[float, ...], value: Sequence[float]
) -> list[float]:
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError("selector and port point must be a two-item coordinate")
    x, y = float(value[0]), float(value[1])
    a, b, c, d, u, v = transform
    result = [a * x + b * y + u, c * x + d * y + v]
    if not all(math.isfinite(item) for item in result):
        raise ValueError("transformed point is not finite")
    return result


def _direction(transform: tuple[float, ...], angle_degrees: float) -> list[float]:
    angle = math.radians(float(angle_degrees))
    x, y = math.cos(angle), math.sin(angle)
    a, b, c, d, _, _ = transform
    result = [a * x + b * y, c * x + d * y]
    magnitude = math.hypot(*result)
    if not math.isfinite(magnitude) or magnitude == 0:
        raise ValueError("transformed port direction is invalid")
    return [result[0] / magnitude, result[1] / magnitude, 0.0]


def local_semantics(component: Any, path: str) -> Mapping[str, Any]:
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


def build_plan_stack(  # noqa: PLR0913 - keyword-only; each argument is a separate plan input
    *,
    component: Any,
    layer_stack: Any,
    material_records: Mapping[str, Mapping[str, Any]],
    coupon_padding_um: float,
    instances: Mapping[str, InstanceRecord],
    nets: Mapping[str, tuple[str, ...]],
    root_id: str,
) -> dict[str, Any]:
    """Assemble the stack mapping: solution regions, materials, layers, metadata."""
    levels = getattr(layer_stack, "layers", None)
    if not isinstance(levels, Mapping):
        raise TypeError("layer_stack must expose layers")
    bounds = coupon_domain_bounds(component, coupon_padding_um)
    materials = normalized_materials(material_records)
    regions = solution_regions(levels, materials=materials, bounds=bounds)
    if not regions:
        raise ValueError("plan needs at least one PDK solution region")
    owners = {entity_id: net for net, members in nets.items() for entity_id in members}
    declarations: list[dict[str, Any]] = []
    ports: list[dict[str, Any]] = []
    contributions: list[tuple[str, dict[str, Any]]] = []
    contribution_ids: set[tuple[str, str]] = set()
    for path, (cell, _, transform) in instances.items():
        semantics = local_semantics(cell, path)
        declarations.extend(_conductor_declarations(path, semantics, transform, owners))
        contributions.extend(
            _ground_contributions(path, semantics, transform, contribution_ids)
        )
        ports.extend(
            _port_sheet_sources(path, cell, semantics, transform, first=len(ports))
        )
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
    _bind_occurrence_ground_planes(layers, regions, levels, root_id)
    _attach_ground_contributions(by_id, contributions, root_id)
    return {
        "solution_regions": regions,
        "materials": materials,
        "layers": layers,
        "metadata": {
            "adapter": "scgsim.sgb.GeometryPlan",
            "component_semantics_schema_version": 2,
            "coupon_domain_bounds_um": dict(bounds),
            "coupon_padding_um": coupon_padding_um,
            "top_cell_name": component.name,
            "port_sheet_source_layers": ports,
        },
    }


def _conductor_declarations(
    path: str,
    semantics: Mapping[str, Any],
    transform: tuple[float, ...],
    owners: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Conductor regions of one cell, qualified by `path` and moved to root frame."""
    declarations: list[dict[str, Any]] = []
    for raw in semantics.get("conductor_regions", ()):
        if not isinstance(raw, Mapping):
            raise TypeError(f"{path} conductor records must be mappings")
        if set(raw) & _FORBIDDEN_AUTHOR_KEYS:
            raise ValueError(f"{path} v2 entity cannot author Net or Group")
        metadata = raw.get("metadata", {})
        if not isinstance(metadata, Mapping) or set(metadata) & _FORBIDDEN_AUTHOR_KEYS:
            raise ValueError(f"{path} v2 entity metadata cannot author Net or Group")
        local_id = checked_entity_path(raw.get("semantic_id"))
        if "/" in local_id:
            raise ValueError("component-local entity ID must be one segment")
        entity_id = f"{path}/{local_id}"
        geometry = dict(raw.get("geometry", {}))
        for key in ("selector_point_um",):
            if key in geometry:
                geometry[key] = transformed_point(transform, geometry[key])
        for key in ("include_selector_points_um", "exclude_selector_points_um"):
            if key in geometry:
                geometry[key] = [
                    transformed_point(transform, point) for point in geometry[key]
                ]
        declarations.append(
            {
                **raw,
                "semantic_id": entity_id,
                "net_id": owners.get(entity_id),
                "geometry": geometry,
                "metadata": {
                    **metadata,
                    "source_occurrence_path": path,
                    "source_local_entity_id": local_id,
                    "source_semantic_id": entity_id,
                },
            }
        )
    return declarations


def _ground_contributions(
    path: str,
    semantics: Mapping[str, Any],
    transform: tuple[float, ...],
    contribution_ids: set[tuple[str, str]],
) -> list[tuple[str, dict[str, Any]]]:
    """Ground-plane contributions of one cell as (target plane id, record) pairs."""
    contributions: list[tuple[str, dict[str, Any]]] = []
    for raw in semantics.get("ground_plane_contributions", ()):
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
            checked_entity_path(raw[key]) for key in ("local_id", "plane_id", "level")
        )
        if any("/" in value for value in (local_id, plane_id, level)):
            raise ValueError("ground contribution IDs must be component-local names")
        key = (path, local_id)
        if key in contribution_ids:
            raise ValueError(f"duplicate ground contribution {path}/{local_id}")
        contribution_ids.add(key)
        layer = gds_layer(raw["layer"], f"{path}/{local_id} ground contribution")
        contributions.append(
            (
                plane_id,
                {
                    "source_occurrence_path": path,
                    "source_local_id": local_id,
                    "level": level,
                    "layer": list(layer),
                    "selector_point_um": transformed_point(
                        transform, raw["selector_point_um"]
                    ),
                },
            )
        )
    return contributions


def _port_sheet_sources(
    path: str,
    cell: Any,
    semantics: Mapping[str, Any],
    transform: tuple[float, ...],
    *,
    first: int,
) -> list[dict[str, Any]]:
    """Port-sheet source records of one cell; `first` ports already exist plan-wide."""
    ports: list[dict[str, Any]] = []
    for raw in semantics.get("ports", ()):
        if not isinstance(raw, Mapping) or set(raw) != {
            "name",
            "layer",
            "target_layer",
        }:
            raise ValueError(f"{path} port requires name, layer, target_layer")
        name = checked_entity_path(raw["name"])
        if "/" in name:
            raise ValueError("component-local port name must be one segment")
        port = cell.ports[name]
        layer = gds_layer(raw["layer"], f"{path}/{name} port")
        raw_port_layer = port.layer
        if hasattr(raw_port_layer, "layer") and hasattr(raw_port_layer, "datatype"):
            port_layer = raw_port_layer
        else:
            port_layer = port.kcl.layout.get_info(int(str(raw_port_layer)))
        if (int(port_layer.layer), int(port_layer.datatype)) != layer:
            raise ValueError(f"{path}/{name} authored port layer differs")
        ports.append(
            {
                "layer": layer[0],
                "datatype": layer[1],
                "name": f"{path}/{name}",
                "source": "palace_lumped_port_sheet",
                "port_index": first + len(ports) + 1,
                "target_layer": raw["target_layer"],
                "direction": _direction(transform, float(port.orientation)),
                "direction_sign_convention": "gdsfactory_port_orientation_outward",
                "selector_point_um": transformed_point(transform, port.center),
                "source_occurrence_path": path,
            }
        )
    return ports


def _bind_occurrence_ground_planes(
    layers: Sequence[Mapping[str, Any]],
    regions: Mapping[str, Any],
    levels: Mapping[str, Any],
    root_id: str,
) -> None:
    """Allow one derived ground plane per physical level.

    A ground plane declared in a child cell is cut by the layer its
    `plane_bounds_ref` level names, so that GDS layer is written into the
    referenced solution region's metadata for the later polygon pass.
    """
    physical_planes: set[str] = set()
    for record in layers:
        if record["geometry"].get("geometry_source") != _DERIVED_GROUND:
            continue
        if record["metadata"]["source_occurrence_path"] != root_id:
            from gdsfactory.pdk import get_layer_tuple
            from gdsfactory.technology.layer_stack import LogicalLayer

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
            raise ValueError(f"multiple ground planes claim physical level {level!r}")
        physical_planes.add(level)


def _attach_ground_contributions(
    by_id: Mapping[str, Any],
    contributions: Sequence[tuple[str, dict[str, Any]]],
    root_id: str,
) -> None:
    """Record each contribution on the root derived plane it targets."""
    for plane_id, contribution in contributions:
        target = by_id.get(f"{root_id}/{plane_id}")
        if (
            target is None
            or target["geometry"].get("geometry_source") != _DERIVED_GROUND
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
        target["metadata"].setdefault("ground_plane_contribution_sources", []).append(
            {
                "source_occurrence_path": contribution["source_occurrence_path"],
                "source_local_id": contribution["source_local_id"],
                "level": contribution["level"],
                "layer": contribution["layer"],
            }
        )
