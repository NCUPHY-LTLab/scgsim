"""Private Palace problem-input parsing shared by the public problem classes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def non_negative_number(value: Any, name: str) -> float:
    """Validate the finite nonnegative scalar used by problem inputs."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite non-negative number.")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number.")
    return result


def stack_materials(stack: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """The explicit materials of a plan's stack, each kind checked against the regions and layers that use it."""
    materials = stack.get("materials")
    if not isinstance(materials, Mapping) or not materials:
        raise ValueError("stack must define a non-empty explicit materials mapping.")
    resolved = {
        str(key): value
        for key, value in materials.items()
        if isinstance(value, Mapping)
    }
    if len(resolved) != len(materials):
        raise TypeError("every explicit stack material must be a mapping.")
    _validate_stack_material_kinds(stack, resolved)
    return resolved


def _validate_stack_material_kinds(
    stack: Mapping[str, Any], materials: Mapping[str, Mapping[str, Any]]
) -> None:
    """Require explicit material kinds for solution regions and metal layers."""

    def material_kind(material_id: Any, owner: str) -> str:
        if not isinstance(material_id, str) or material_id not in materials:
            raise ValueError(f"{owner} must reference an explicit stack material_id.")
        kind = materials[material_id].get("kind")
        if kind not in {"vacuum", "dielectric", "conductor"}:
            raise ValueError(
                f"material {material_id!r} requires kind vacuum, dielectric, or conductor."
            )
        return str(kind)

    regions = stack.get("solution_regions")
    if not isinstance(regions, Mapping) or not regions:
        raise ValueError("stack must define explicit solution_regions.")
    for name, region in regions.items():
        if not isinstance(region, Mapping) or material_kind(
            region.get("material_id"), f"solution region {name!r}"
        ) not in {"vacuum", "dielectric"}:
            raise ValueError(
                f"solution region {name!r} must use vacuum or dielectric material kind."
            )
    layers = stack.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("stack must define explicit layers.")
    for layer in layers:
        if not isinstance(layer, Mapping):
            raise TypeError("stack layers must be mappings.")
        if (
            layer.get("role") == "metal"
            and material_kind(layer.get("material_id"), "metal layer") != "conductor"
        ):
            raise ValueError("metal layer material must have conductor kind.")


__all__ = ["non_negative_number", "stack_materials"]
