"""Read solution volumes (the dielectric and vacuum domains) out of SGB groups.

SGB (the geometry builder) names each domain volume and gives it Palace attributes and
one material id. These helpers turn that record into the Palace ``Materials`` rows,
the ``Postprocessing.Energy`` rows, and the material resolution record, and refuse a
volume that does not carry an explicit, valid material.

Not here: boundary synthesis (``electrostatic_boundaries.py``) and the config
assembly (``config.py``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

from .._values import optional_stripped_string


def resolve_volume_materials(
    domain_volumes: list[tuple[str, Mapping[str, Any]]],
    materials: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Which material each solution volume got (``palace_material_resolution.json``)."""
    return [
        {
            "solution_volume": name,
            "attributes": sorted(physical_group_values(volume.get("phys_group"))),
            "physical_attribute": volume.get("physical_attribute", {}),
            "material": _material_identity(volume, materials),
            "source_provenance": volume.get("source_provenance", {}),
        }
        for name, volume in domain_volumes
    ]


def domain_energy_rows(
    domain_volumes: list[tuple[str, Mapping[str, Any]]],
    materials: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [
        {
            "Index": index,
            "Attributes": sorted(physical_group_values(info.get("phys_group"))),
        }
        for index, (_, info) in enumerate(domain_volumes, start=1)
    ]
    index_rows = [
        {
            "section": "Domains.Postprocessing.Energy",
            "index": index,
            "entry_name": name,
            "role": "domain_energy",
            "physical_name": name,
            "physical_names": [name],
            "attributes": row["Attributes"],
            "metadata": {
                "physical_attribute": info.get("physical_attribute", {}),
                "material": _material_identity(info, materials),
                "source_provenance": info.get("source_provenance", {}),
                "representation": info.get("representation"),
                "route": info.get("route"),
            },
        }
        for index, ((name, info), row) in enumerate(
            zip(domain_volumes, rows, strict=True), start=1
        )
    ]
    return rows, index_rows


def domains_from_solution_volumes(
    groups: Mapping[str, Mapping[str, Any]],
) -> list[tuple[str, Mapping[str, Any]]]:
    """Sort solver volumes as deterministic domains."""
    return sorted(
        ((name, dict(info)) for name, info in groups.items()), key=lambda item: item[0]
    )


def materials_from_solution_volumes(
    *,
    domain_volumes: list[tuple[str, Mapping[str, Any]]],
    materials: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build Palace-domain material rows from explicit material records."""
    if not isinstance(materials, Mapping) or not materials:
        raise ValueError(
            "stack must include top-level materials mapping for config generation."
        )

    material_rows: list[dict[str, Any]] = []
    for name, volume in domain_volumes:
        attrs = physical_group_values(volume.get("phys_group"))
        physical_attribute = volume.get("physical_attribute")
        if not isinstance(physical_attribute, Mapping):
            raise TypeError(f"SGB solution volume {name!r} missing physical_attribute.")
        material_ids = physical_attribute.get("material_ids")
        if not isinstance(material_ids, (list, tuple)) or len(material_ids) != 1:
            raise ValueError(
                f"SGB solution volume {name!r} must define a single material_ids entry."
            )
        material_id = optional_stripped_string(material_ids[0])
        if not material_id:
            raise ValueError(
                f"SGB solution volume {name!r} must define a non-empty material id."
            )
        material = materials.get(material_id)
        if material is None:
            raise ValueError(
                f"SGB solution volume {name!r} references unknown material id {material_id!r}."
            )
        if not isinstance(material, Mapping):
            raise TypeError(f"SGB material record {material_id!r} must be a mapping.")
        if material.get("kind") not in {"vacuum", "dielectric"}:
            raise ValueError(
                f"SGB solution material {material_id!r} must have vacuum or dielectric kind."
            )
        permittivity = material.get("permittivity")
        loss_tangent = material.get("loss_tangent")
        if (
            not isinstance(permittivity, Real)
            or not math.isfinite(float(permittivity))
            or float(permittivity) <= 0.0
        ):
            raise ValueError(
                f"Material {material_id!r} requires finite positive permittivity."
            )
        if (
            not isinstance(loss_tangent, Real)
            or not math.isfinite(float(loss_tangent))
            or float(loss_tangent) < 0.0
        ):
            raise ValueError(
                f"Material {material_id!r} requires finite non-negative loss_tangent."
            )
        if material_rows and [
            x for x in material_rows if list(x["Attributes"]) == list(attrs)
        ]:
            # allow only deterministic duplicate attributes if metadata diverges.
            raise ValueError(
                f"SGB solution volume {name!r} duplicates physical attributes {list(attrs)}."
            )
        material_rows.append(
            {
                "Attributes": sorted(int(item) for item in attrs),
                "Permittivity": float(permittivity),
                "LossTan": float(loss_tangent),
            }
        )

    return material_rows


def _material_identity(
    volume: Mapping[str, Any], materials: Mapping[str, Mapping[str, Any]]
) -> dict[str, str]:
    ids = volume.get("physical_attribute", {}).get("material_ids")
    if (
        not isinstance(ids, (list, tuple))
        or len(ids) != 1
        or not isinstance(ids[0], str)
    ):
        raise ValueError("solution volume must carry one explicit material id.")
    material_id = ids[0]
    material = materials.get(material_id)
    if not isinstance(material, Mapping) or material.get("kind") not in {
        "vacuum",
        "dielectric",
    }:
        raise ValueError(
            f"solution material {material_id!r} must carry vacuum or dielectric kind."
        )
    return {"material_id": material_id, "kind": str(material["kind"])}


def physical_group_values(value: Any) -> tuple[int, ...]:
    raw = value if isinstance(value, (list, tuple, set)) else (value,)
    values = [
        int(item)
        for item in raw
        if isinstance(item, int) and not isinstance(item, bool)
    ]
    if not values:
        raise ValueError("Physical group must have at least one integer attribute.")
    return tuple(values)
