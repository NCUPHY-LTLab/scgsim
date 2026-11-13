"""Return a detached stack that holds the declared vacuum region."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

from scgsim.geometry.models.input import VacuumRegionSpec
from scgsim.geometry.models.refusals import Declared
from scgsim.semantics.route_a import geometry_z_range, record_geometry

_CALL = Declared(
    "apply_vacuum_region_to_stack", "the call to apply_vacuum_region_to_stack"
)
_STACK = Declared(
    "the stack", "the stack: the PDK layer stack's levels, or the stack file"
)
_MATERIALS = Declared(
    "the stack's materials", "the material records the stack was built from"
)


def _region(semantic_id: object) -> Declared:
    """A solution region of the stack, which the stack declares."""
    return Declared(
        f"solution region {semantic_id!r}",
        "the stack: its PDK layer stack level, or the stack file's solution_regions",
    )


def apply_vacuum_region_to_stack(
    stack: Mapping[str, Any], vacuum_region: VacuumRegionSpec
) -> Mapping[str, Any]:
    """Return a detached stack with one canonical automatic vacuum region.

    The steps: check the stack holds no explicit airbox, make the vacuum material, bound
    the stack's bodies, and record the region as those bounds grown by the padding.
    """
    raw_regions, materials = _checked_regions_without_airbox(stack, vacuum_region)
    regions = {str(key): dict(value) for key, value in raw_regions.items()}
    work = dict(stack)
    material_catalog = dict(materials)
    vacuum_id = _materialize_locked_vacuum_material(regions, material_catalog)
    bounds = _auto_vacuum_bounds(regions, material_catalog, stack.get("layers", ()))
    regions["VACUUM_REGION"] = _vacuum_region_record(
        regions.get("VACUUM_REGION"), vacuum_id, bounds, vacuum_region
    )
    work["solution_regions"] = regions
    work["materials"] = material_catalog
    return {**work, "solution_regions": regions}


class _StackParts(NamedTuple):
    """The stack's solution regions and materials."""

    solution_regions: Mapping[str, Any]
    materials: Mapping[str, Any]


def _checked_regions_without_airbox(stack: Any, vacuum_region: Any) -> _StackParts:
    """The stack's solution regions and materials, which must hold no explicit airbox."""
    if not isinstance(stack, Mapping):
        raise TypeError(_CALL.expected("stack", "a mapping", stack))
    if not isinstance(vacuum_region, VacuumRegionSpec):
        raise TypeError(
            _CALL.expected("vacuum_region", "a VacuumRegionSpec", vacuum_region)
        )
    raw_regions = stack.get("solution_regions")
    materials = stack.get("materials")
    if not isinstance(raw_regions, Mapping):
        raise TypeError(_STACK.expected("solution_regions", "a mapping", raw_regions))
    if not isinstance(materials, Mapping):
        raise TypeError(_STACK.expected("materials", "a mapping", materials))
    for semantic_id, region in raw_regions.items():
        if not isinstance(region, Mapping):
            raise TypeError(
                _region(semantic_id).expected("record", "a mapping", region)
            )
        if not isinstance(region.get("is_airbox", False), bool):
            raise TypeError(
                _region(semantic_id).expected(
                    "is_airbox", "a boolean", region.get("is_airbox")
                )
            )
        if region.get("is_airbox", False):
            raise ValueError(
                _region(semantic_id).refusal(
                    "it is an explicit airbox; set_vacuum_region cannot be used with explicit airbox solution regions: remove one of them"
                )
            )
    return _StackParts(raw_regions, materials)


def _vacuum_region_base(region: Any, vacuum_id: str) -> dict[str, Any]:
    """The record the VACUUM_REGION starts from: the stack's own, of the vacuum material, or a new one."""
    if region is None:
        return {
            "role": "solution_region",
            "material_id": vacuum_id,
            "geometry_kind": "domain",
            "metadata": {},
            "geometry": {},
        }
    if not isinstance(region, Mapping):
        raise TypeError(
            _region("VACUUM_REGION").expected("record", "a mapping", region)
        )
    if _material_id_for_region(region) != vacuum_id:
        raise ValueError(
            _region("VACUUM_REGION").expected(
                "material_id",
                f"the vacuum material {vacuum_id!r}",
                _material_id_for_region(region),
            )
        )
    return dict(region)


def _vacuum_region_record(
    region: Any,
    vacuum_id: str,
    bounds: Mapping[str, float],
    vacuum_region: VacuumRegionSpec,
) -> dict[str, Any]:
    """The VACUUM_REGION record: the stack's own if it has one, around the bounds grown by the padding."""
    region = _vacuum_region_base(region, vacuum_id)
    region.update(
        {
            "material_id": vacuum_id,
            "material_kind": "vacuum",
            "is_auto_vacuum_region": True,
            "geometry": {
                **dict(region.get("geometry", {})),
                "domain": "VACUUM_REGION",
                "domain_bounds_um": {
                    "x_min_um": bounds["x_min_um"] - vacuum_region.x_minus_um,
                    "x_max_um": bounds["x_max_um"] + vacuum_region.x_plus_um,
                    "y_min_um": bounds["y_min_um"] - vacuum_region.y_minus_um,
                    "y_max_um": bounds["y_max_um"] + vacuum_region.y_plus_um,
                },
                "z_min_um": bounds["z_min_um"] - vacuum_region.z_minus_um,
                "z_max_um": bounds["z_max_um"] + vacuum_region.z_plus_um,
            },
            "metadata": {
                **dict(region.get("metadata", {})),
                "source": "set_vacuum_region",
                "is_auto_vacuum_region": True,
                "vacuum_region_padding_um": {
                    "x_minus_um": vacuum_region.x_minus_um,
                    "x_plus_um": vacuum_region.x_plus_um,
                    "y_minus_um": vacuum_region.y_minus_um,
                    "y_plus_um": vacuum_region.y_plus_um,
                    "z_minus_um": vacuum_region.z_minus_um,
                    "z_plus_um": vacuum_region.z_plus_um,
                },
            },
        }
    )
    return region


def _materialize_locked_vacuum_material(
    regions: Mapping[str, Mapping[str, Any]], materials: dict[str, Any]
) -> str:
    record = materials.get("vacuum")
    if record is None:
        materials["vacuum"] = {
            "kind": "vacuum",
            "permittivity": 1.0,
            "loss_tangent": 0.0,
        }
    elif not isinstance(record, Mapping):
        raise TypeError(_MATERIALS.expected("material 'vacuum'", "a mapping", record))
    else:
        normalized = dict(record)
        if normalized.get("permittivity") is None:
            normalized["permittivity"] = 1.0
        if normalized.get("loss_tangent") is None:
            normalized["loss_tangent"] = 0.0
        if normalized.get("kind") != "vacuum":
            raise ValueError(
                _MATERIALS.expected(
                    "material 'vacuum' kind", "vacuum", normalized.get("kind")
                )
            )
        if normalized["permittivity"] != 1.0:
            raise ValueError(
                _MATERIALS.expected(
                    "material 'vacuum' permittivity", "1.0", normalized["permittivity"]
                )
            )
        if normalized["loss_tangent"] != 0.0:
            raise ValueError(
                _MATERIALS.expected(
                    "material 'vacuum' loss_tangent", "0.0", normalized["loss_tangent"]
                )
            )
        materials["vacuum"] = normalized
    for semantic_id, region in regions.items():
        material_id = _material_id_for_region(region)
        material = materials.get(material_id)
        if not isinstance(material, Mapping):
            raise TypeError(
                _region(semantic_id).expected(
                    "material", "a material of the stack", material_id
                )
            )
        if material.get("kind") == "vacuum" and material_id != "vacuum":
            raise ValueError(
                _region(semantic_id).expected(
                    "material",
                    "the canonical material 'vacuum', as it is a vacuum",
                    material_id,
                )
            )
    return "vacuum"


def _auto_vacuum_bounds(
    regions: Mapping[str, Mapping[str, Any]],
    materials: Mapping[str, Any],
    layers: Any,
) -> dict[str, float]:
    bounds: list[dict[str, float]] = []
    for semantic_id, region in regions.items():
        region_bounds = _region_envelope_bounds(str(semantic_id), region, materials)
        if region_bounds is None:
            continue
        bounds.append(region_bounds)
    if not bounds:
        raise ValueError(
            _STACK.refusal(
                "it has no non-vacuum solution region to put the vacuum envelope around"
            )
        )
    if isinstance(layers, (str, bytes)) or not isinstance(layers, Sequence):
        raise TypeError(_STACK.expected("layers", "a sequence", layers))
    layer_ranges = []
    for index, layer in enumerate(layers):
        if not isinstance(layer, Mapping):
            raise TypeError(_STACK.expected(f"layer {index}", "a mapping", layer))
        layer_ranges.append(
            geometry_z_range(
                record_geometry(layer, str(layer.get("semantic_id", index))),
                str(layer.get("semantic_id", index)),
            )
        )
    return {
        "x_min_um": min(item["x_min_um"] for item in bounds),
        "x_max_um": max(item["x_max_um"] for item in bounds),
        "y_min_um": min(item["y_min_um"] for item in bounds),
        "y_max_um": max(item["y_max_um"] for item in bounds),
        "z_min_um": min(
            *(item["z_min_um"] for item in bounds),
            *(item[0] for item in layer_ranges),
        ),
        "z_max_um": max(
            *(item["z_max_um"] for item in bounds),
            *(item[1] for item in layer_ranges),
        ),
    }


def _region_envelope_bounds(
    semantic_id: str,
    region: Mapping[str, Any],
    materials: Mapping[str, Any],
) -> dict[str, float] | None:
    material_id = _material_id_for_region(region)
    kind = _material_kind(material_id, materials)
    metadata = region.get("metadata")
    if (
        kind == "vacuum"
        and isinstance(metadata, Mapping)
        and bool(metadata.get("is_auto_vacuum_region"))
    ):
        return None

    geometry = region.get("geometry")
    if not isinstance(geometry, Mapping):
        raise TypeError(
            _region(semantic_id).expected("geometry", "a mapping", geometry)
        )
    domain = geometry.get("domain_bounds_um")
    if not isinstance(domain, Mapping):
        raise TypeError(
            _region(semantic_id).expected("domain_bounds_um", "a mapping", domain)
        )
    coordinate_names = ("x_min_um", "x_max_um", "y_min_um", "y_max_um")
    invalid_coordinates = [
        name for name in coordinate_names if not _finite(domain.get(name))
    ]
    if invalid_coordinates:
        raise ValueError(
            _region(semantic_id).expected(
                f"domain_bounds_um {invalid_coordinates!r}", "finite", domain
            )
        )

    z_min, z_max = _region_z_interval(semantic_id, geometry)
    return {
        "x_min_um": float(domain["x_min_um"]),
        "x_max_um": float(domain["x_max_um"]),
        "y_min_um": float(domain["y_min_um"]),
        "y_max_um": float(domain["y_max_um"]),
        "z_min_um": z_min,
        "z_max_um": z_max,
    }


def _region_z_interval(
    semantic_id: str, geometry: Mapping[str, Any]
) -> tuple[float, float]:
    """A solution region's Z interval: from its base to its top, or to its base plus its thickness."""
    z_min = geometry.get("z_min_um", geometry.get("z_um"))
    if z_min is None or not _finite(z_min):
        raise ValueError(_region(semantic_id).expected("z_min_um", "finite", z_min))
    z_min = float(z_min)
    z_max = geometry.get("z_max_um")
    if z_max is None:
        thickness = geometry.get("thickness_um")
        if thickness is None or not _finite(thickness):
            raise ValueError(
                _region(semantic_id).expected(
                    "z_max_um or thickness_um", "finite", thickness
                )
            )
        z_max = z_min + float(thickness)
    elif not _finite(z_max):
        raise ValueError(_region(semantic_id).expected("z_max_um", "finite", z_max))
    else:
        z_max = float(z_max)
    if not z_min < z_max:
        raise ValueError(
            _region(semantic_id).refusal(
                f"its thickness is not positive: z from {z_min} to {z_max} um"
            )
        )
    return z_min, z_max


def _material_id_for_region(region: Mapping[str, Any]) -> str:
    material_id = region.get("material_id", region.get("name"))
    if not isinstance(material_id, str) or not material_id:
        raise ValueError(
            _region(region.get("semantic_id")).expected(
                "material_id", "a non-empty string", material_id
            )
        )
    return material_id


def _material_kind(material_id: str, materials: Mapping[str, Any]) -> str:
    material = materials.get(material_id)
    if not isinstance(material, Mapping):
        raise TypeError(
            _MATERIALS.expected(f"material {material_id!r}", "a mapping", material)
        )
    kind = material.get("kind")
    if kind not in {"vacuum", "dielectric", "conductor"}:
        raise ValueError(
            _MATERIALS.expected(
                f"material {material_id!r} kind",
                "vacuum, dielectric or conductor",
                kind,
            )
        )
    return str(kind)


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
