"""Construct source-backed planar bodies and realize their material catalog."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from scgsim.aedt.preparation.planar_source import (
    native_entity_name,
    native_name,
)
from scgsim.aedt.runtime.native.common import (
    read_native_material,
    realize_source_material,
)
from scgsim.aedt.specs.epr import PlanarJunction
from scgsim.geometry.models.junctions import JunctionGeometry
from scgsim.semantics.route_a import geometry_z_range


def polygon_sheet(
    app: Any, polygon: Mapping[str, Any], *, name: str, z_um: float
) -> Any:
    """Create one covered source polygon and subtract its declared holes in order."""
    points = [[float(x), float(y), z_um] for x, y in polygon["exterior"]]
    value = app.modeler.create_polyline(
        points,
        cover_surface=True,
        close_surface=True,
        name=name,
        non_model=False,
    )
    if value is False or value is None:
        raise RuntimeError(f"failed to create planar source polygon {name!r}")
    for index, ring in enumerate(polygon["holes"]):
        hole_name = native_name("hole", name, index)
        hole = app.modeler.create_polyline(
            [[float(x), float(y), z_um] for x, y in ring],
            cover_surface=True,
            close_surface=True,
            name=hole_name,
            non_model=False,
        )
        if hole is False or hole is None:
            raise RuntimeError(f"failed to create planar hole {hole_name!r}")
        if not app.modeler.subtract(name, hole_name, keep_originals=False):
            raise RuntimeError(f"failed to subtract planar hole {hole_name!r}")
    return app.modeler[name]


def swept_z_solid(
    app: Any,
    polygon: Mapping[str, Any],
    *,
    name: str,
    z_min_um: float,
    z_max_um: float,
) -> Any:
    """Sweep a planar loop along explicit global +Z, independent of its normal."""

    if z_max_um <= z_min_um:
        raise ValueError(f"solid {name!r} has empty Z extent")
    sheet = polygon_sheet(app, polygon, name=name, z_um=z_min_um)
    body = app.modeler.sweep_along_vector(
        sheet.name, ["0um", "0um", f"{z_max_um - z_min_um:.17g}um"]
    )
    if body is False or body is None:
        raise RuntimeError(f"failed to sweep solid {name!r} along positive Z")
    bounds = body.bounding_box
    scale = max(1.0, abs(z_min_um), abs(z_max_um))
    if (
        len(bounds) != 6
        or not math.isclose(
            float(bounds[2]), z_min_um, rel_tol=0.0, abs_tol=1e-9 * scale
        )
        or not math.isclose(
            float(bounds[5]), z_max_um, rel_tol=0.0, abs_tol=1e-9 * scale
        )
    ):
        raise RuntimeError(f"solid {name!r} native Z range differs from source")
    return body


def _bounds_box(app: Any, entity: Mapping[str, Any], source: Mapping[str, Any]) -> Any:
    geometry = entity["geometry"]
    bounds = geometry.get("domain_bounds_um")
    if not isinstance(bounds, Mapping):
        raise ValueError(
            f"solution domain {entity['semantic_id']!r} lacks domain_bounds_um"
        )
    required = ("x_min_um", "x_max_um", "y_min_um", "y_max_um")
    if any(key not in bounds for key in required):
        raise ValueError(
            f"solution domain {entity['semantic_id']!r} bounds are incomplete"
        )
    z_min = geometry.get("z_min_um", geometry.get("z_um"))
    z_max = geometry.get("z_max_um")
    if z_max is None and z_min is not None and geometry.get("thickness_um") is not None:
        z_max = float(z_min) + float(geometry["thickness_um"])
    if z_min is None or z_max is None:
        raise ValueError(
            f"solution domain {entity['semantic_id']!r} lacks a finite z range"
        )
    origin = [float(bounds["x_min_um"]), float(bounds["y_min_um"]), float(z_min)]
    size = [
        float(bounds["x_max_um"]) - origin[0],
        float(bounds["y_max_um"]) - origin[1],
        float(z_max) - origin[2],
    ]
    if any(value <= 0.0 for value in size):
        raise ValueError(f"solution domain {entity['semantic_id']!r} has empty bounds")
    native_material = str(entity["material_id"])
    value = app.modeler.create_box(
        origin,
        size,
        name=native_entity_name(source, "domain", entity["semantic_id"]),
        material=native_material,
    )
    if value is False or value is None:
        raise RuntimeError(
            f"failed to create solution domain {entity['semantic_id']!r}"
        )
    return value


def solution_body(
    app: Any, entity: Mapping[str, Any], source: Mapping[str, Any]
) -> Any:
    """Create a bounded box or swept source loop for one solution-domain body."""
    geometry = entity["geometry"]
    if "outer_loop" not in geometry:
        return _bounds_box(app, entity, source)
    z_min, z_max = geometry_z_range(geometry, entity["semantic_id"])
    if z_max <= z_min:
        raise ValueError(
            f"solution domain {entity['semantic_id']!r} has empty Z extent"
        )
    polygon = {
        "exterior": geometry["outer_loop"],
        "holes": geometry.get("hole_loops", ()),
    }
    name = native_entity_name(source, "domain", entity["semantic_id"])
    body = swept_z_solid(app, polygon, name=name, z_min_um=z_min, z_max_um=z_max)
    body.material_name = str(entity["material_id"])
    return body


def validate_native_modeling_source(source: Mapping[str, Any]) -> None:
    """Require the source to carry its effective-layer ``modeling`` provenance."""
    if not isinstance(source.get("modeling"), Mapping):
        raise ValueError("prepared EPR effective-layer provenance is unavailable")


def native_conductor_geometry(
    source: Mapping[str, Any], entity: Mapping[str, Any]
) -> tuple[str, float, float]:
    """Read the producer's effective native shape and Z without remapping it."""
    if not isinstance(source.get("modeling"), Mapping):
        raise TypeError("prepared EPR modeling provenance is unavailable")
    physical_layer = entity.get("physical_layer")
    if not isinstance(physical_layer, Mapping):
        raise ValueError(
            f"EPR conductor {entity.get('semantic_id')!r} lacks "
            "effective-layer provenance"
        )
    representation = physical_layer.get("representation")
    if representation not in {"solid", "sheet"}:
        raise ValueError("prepared EPR conductor representation is invalid")
    if entity.get("representation") != representation:
        raise ValueError(
            f"EPR conductor {entity.get('semantic_id')!r} representation "
            "differs from its effective layer"
        )
    return (
        str(representation),
        float(physical_layer["effective_z_min_um"]),
        float(physical_layer["effective_z_max_um"]),
    )


def material_readback(app: Any, materials: Mapping[str, Any]) -> dict[str, Any]:
    """Read native definitions for the declared vacuum and dielectric materials."""
    observed: dict[str, Any] = {}
    for material_id, record in materials.items():
        if not isinstance(record, Mapping) or record.get("kind") not in {
            "vacuum",
            "dielectric",
        }:
            continue
        observed[material_id] = read_native_material(app, material_id)
    return observed


def native_solution_materials(source: Mapping[str, Any]) -> dict[str, Any]:
    """Install only materials carried by native solution-domain bodies."""

    catalog = source["materials"]
    material_ids = {str(item["material_id"]) for item in source["solution_regions"]}
    material_ids.add(str(source["native_region"]["material_id"]))
    missing = material_ids - set(catalog)
    if missing:
        raise ValueError(f"native solution materials are missing: {sorted(missing)!r}")
    return {material_id: catalog[material_id] for material_id in sorted(material_ids)}


def install_material_catalog(app: Any, materials: Mapping[str, Any]) -> dict[str, Any]:
    """Install the declared vacuum and dielectric materials (a vacuum defaults to permittivity 1); conductors are skipped."""
    observed: dict[str, Any] = {}
    for material_id, record in materials.items():
        if not isinstance(material_id, str) or not isinstance(record, Mapping):
            raise TypeError("prepared material catalog must be string-keyed mappings")
        kind = record.get("kind")
        if kind == "conductor":
            continue
        if kind not in {"vacuum", "dielectric"}:
            raise ValueError(f"unsupported prepared material kind {kind!r}")
        requested = dict(record)
        if "permittivity" not in requested:
            if kind != "vacuum":
                raise ValueError(
                    f"prepared material {material_id!r} requires positive permittivity"
                )
            # A vacuum may omit its permittivity; a dielectric may not.
            requested["permittivity"] = 1.0
        observed[material_id] = realize_source_material(app, material_id, requested)
    return observed


def junction_geometry(junction: PlanarJunction) -> JunctionGeometry:
    """Pass the neutral geometry fields without electrical solver parameters."""
    return JunctionGeometry(
        junction_id=junction.junction_id,
        source_polygon_id=junction.source_polygon_id,
        terminal_a_net=junction.terminal_a_net,
        terminal_b_net=junction.terminal_b_net,
        direction_xy=junction.direction_xy,
        width_um=junction.width_um,
        metadata=junction.metadata,
    )
