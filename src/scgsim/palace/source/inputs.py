"""The explicit materials of a plan's stack, checked, for the simulation classes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from scgsim.geometry.models.refusals import Declared
from scgsim.refusal import found_text

_WHERE_STACK = "the stack file"


def stack_materials(stack: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """The explicit materials of a plan's stack, each kind checked against the regions and layers that use it."""
    materials = stack.get("materials")
    if not isinstance(materials, Mapping) or not materials:
        raise ValueError(
            Declared(
                "stack must define a non-empty explicit materials mapping", _WHERE_STACK
            ).refusal(found_text(materials))
        )
    resolved = {
        str(key): value
        for key, value in materials.items()
        if isinstance(value, Mapping)
    }
    if len(resolved) != len(materials):
        raise TypeError(
            Declared(
                "every explicit stack material must be a mapping", _WHERE_STACK
            ).refusal(
                found_text(
                    sorted(
                        str(key)
                        for key, value in materials.items()
                        if not isinstance(value, Mapping)
                    )
                )
            )
        )
    _validate_stack_material_kinds(stack, resolved)
    return resolved


def _validate_stack_material_kinds(
    stack: Mapping[str, Any], materials: Mapping[str, Mapping[str, Any]]
) -> None:
    """Require explicit material kinds for solution regions and metal layers."""

    def material_kind(material_id: Any, owner: str) -> str:
        if not isinstance(material_id, str) or material_id not in materials:
            raise ValueError(
                Declared(f"{owner}", _WHERE_STACK).refusal(
                    "must reference an explicit stack material_id"
                )
            )
        kind = materials[material_id].get("kind")
        if kind not in {"vacuum", "dielectric", "conductor"}:
            raise ValueError(
                Declared(f"material {material_id!r}", _WHERE_STACK).refusal(
                    "requires kind vacuum, dielectric, or conductor"
                )
            )
        return str(kind)

    regions = stack.get("solution_regions")
    if not isinstance(regions, Mapping) or not regions:
        raise ValueError(
            Declared(
                "stack must define explicit solution_regions", _WHERE_STACK
            ).refusal(found_text(regions))
        )
    for name, region in regions.items():
        if not isinstance(region, Mapping) or material_kind(
            region.get("material_id"), f"solution region {name!r}"
        ) not in {"vacuum", "dielectric"}:
            raise ValueError(
                Declared(f"solution region {name!r}", _WHERE_STACK).refusal(
                    "must use vacuum or dielectric material kind"
                )
            )
    layers = stack.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError(
            Declared("stack must define explicit layers", _WHERE_STACK).refusal(
                found_text(layers)
            )
        )
    for layer in layers:
        if not isinstance(layer, Mapping):
            raise TypeError(
                Declared("stack layers must be mappings", _WHERE_STACK).refusal(
                    found_text(layer)
                )
            )
        if (
            layer.get("role") == "metal"
            and material_kind(layer.get("material_id"), "metal layer") != "conductor"
        ):
            raise ValueError(
                Declared(
                    "metal layer material must have conductor kind", _WHERE_STACK
                ).refusal(
                    found_text(
                        {
                            "layer": layer.get("name"),
                            "material_id": layer.get("material_id"),
                        }
                    )
                )
            )


__all__ = ["stack_materials"]
