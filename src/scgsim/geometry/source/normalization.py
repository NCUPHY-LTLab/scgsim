"""Normalize the declared layers, materials and solution regions of a stack, for the stack adapter and the plan alike."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any, NamedTuple

from scgsim.geometry.models.physical import (
    PHYSICAL_LAYER_FIELD,
    PhysicalLayerDeclaration,
)
from scgsim.geometry.models.refusals import Declared


def solution_regions(
    levels: Mapping[str, Any],
    *,
    materials: Mapping[str, Mapping[str, Any]],
    bounds: Mapping[str, float],
) -> dict[str, dict[str, Any]]:
    """Normalize declared solution-region layers and their domain bounds."""
    result: dict[str, dict[str, Any]] = {}
    for level_name, level in levels.items():
        info = _level_info(level, level_name)
        if info.get("simulation_role") != "solution_region":
            continue
        owner = level_declared(level_name)
        include = info.get("include_in_component_simulation")
        if not isinstance(include, bool):
            raise TypeError(
                owner.expected("include_in_component_simulation", "a boolean", include)
            )
        if not include:
            continue
        semantic_id = _identifier(level_name, owner, "id")
        facts = _level_facts(level, level_name)
        material_id, z_min_um, thickness_um, priority = (
            facts.material_id,
            facts.z_um,
            facts.thickness_um,
            facts.priority,
        )
        _require_material(material_id, materials, owner)
        authored_geometry = info.get("geometry", {})
        if not isinstance(authored_geometry, Mapping):
            raise TypeError(owner.expected("geometry", "a mapping", authored_geometry))
        source_outline = {
            key: copy.deepcopy(authored_geometry[key])
            for key in ("outer_loop", "hole_loops")
            if key in authored_geometry
        }
        result[semantic_id] = {
            "role": "solution_region",
            "material_id": material_id,
            "priority": priority,
            "geometry_kind": "domain",
            "geometry": {
                "z_min_um": z_min_um,
                "z_max_um": z_min_um + thickness_um,
                "domain_bounds_um": dict(bounds),
                **source_outline,
            },
            "metadata": {
                "pdk_level_id": semantic_id,
                "pdk_semantic_authority": "layer_stack",
                PHYSICAL_LAYER_FIELD: _physical_declaration(
                    info, level_name, material_id
                ),
            },
        }
    return result


def semantic_layer_records(
    declarations: Sequence[Mapping[str, Any]],
    *,
    levels: Mapping[str, Any],
    materials: Mapping[str, Mapping[str, Any]],
    solution_region_ids: Mapping[str, Any],
    require_net: bool = True,
) -> list[dict[str, Any]]:
    """Normalize component conductor declarations against stack records."""
    result = []
    semantic_ids: set[str] = set()
    for declaration in declarations:
        semantic_id = _declared_conductor_id(declaration, require_net=require_net)
        if semantic_id in semantic_ids:
            raise ValueError(
                conductor_declared(semantic_id).refusal("it is declared twice")
            )
        semantic_ids.add(semantic_id)
        level = _conductor_level(declaration, semantic_id, levels, materials)
        host, auto_host = _conductor_host(level, semantic_id, solution_region_ids)
        layer, datatype = gds_layer(
            declaration["gds_layer"], conductor_declared(semantic_id)
        )
        geometry = _conductor_geometry(declaration, semantic_id, level)
        component_metadata = mapping(
            declaration.get("metadata", {}), conductor_declared(semantic_id), "metadata"
        )
        part_role = _identifier(
            level.info.get("part_role"), level_declared(level.name), "part_role"
        )
        record: dict[str, Any] = {
            "layer": layer,
            "datatype": datatype,
            "semantic_id": semantic_id,
            "role": "metal",
            "material_id": level.facts.material_id,
            "priority": level.facts.priority,
            "part_role": part_role,
            "host_void_semantic_id": host,
            "geometry_kind": "layout_extrusion",
            "geometry": geometry,
            "route_representations": {},
            "metadata": {
                "logical_layer_id": level.name,
                "source_layer_name": level.name,
                "pdk_semantic_authority": "layer_stack",
                "component_semantic_authority": "component.info",
                **component_metadata,
                PHYSICAL_LAYER_FIELD: _physical_declaration(
                    level.info, level.name, level.facts.material_id
                ),
                **(
                    {"host_reference_origin": "generated_background"}
                    if auto_host
                    else {}
                ),
            },
            "net_id": (
                _identifier(
                    declaration["net_id"], conductor_declared(semantic_id), "net_id"
                )
                if require_net or declaration.get("net_id") is not None
                else None
            ),
        }
        result.append(record)
    return result


_CONDUCTOR_FIELDS = {
    "semantic_id",
    "level",
    "gds_layer",
    "net_id",
    "geometry",
    "metadata",
}
_COMPONENT_GEOMETRY_FIELDS = {
    "geometry_source",
    "selector_point_um",
    "mask_layer",
    "include_layer",
    "include_selector_points_um",
}


def _declared_conductor_id(declaration: Any, *, require_net: bool) -> str:
    """The semantic id of one component conductor declaration, which may carry only its own fields."""
    owner = conductor_declared(
        declaration.get("semantic_id") if isinstance(declaration, Mapping) else None
    )
    if not isinstance(declaration, Mapping):
        raise TypeError(owner.expected("declaration", "a mapping", declaration))
    unknown = set(declaration) - _CONDUCTOR_FIELDS
    if unknown:
        raise ValueError(
            owner.expected(
                "fields",
                f"only {sorted(_CONDUCTOR_FIELDS)}; the others are the PDK's",
                sorted(unknown),
            )
        )
    required = (
        ("semantic_id", "level", "gds_layer", "net_id")
        if require_net
        else ("semantic_id", "level", "gds_layer")
    )
    missing = [field for field in required if field not in declaration]
    if missing:
        raise ValueError(owner.refusal(f"it lacks fields {missing!r}"))
    return _identifier(declaration["semantic_id"], owner, "semantic_id")


@dataclass(frozen=True)
class _ConductorLevel:
    """The PDK level a conductor declaration names, with the facts the stack records for it."""

    name: str
    info: dict[str, Any]
    facts: LevelFacts


def _conductor_level(
    declaration: Mapping[str, Any],
    semantic_id: str,
    levels: Mapping[str, Any],
    materials: Mapping[str, Mapping[str, Any]],
) -> _ConductorLevel:
    """The declared level, which the PDK must declare a conductor of a known material."""
    level_name = _identifier(
        declaration["level"], conductor_declared(semantic_id), "level"
    )
    level = _level(levels, level_name)
    level_info = _level_info(level, level_name)
    if level_info.get("simulation_role") != "conductor":
        raise ValueError(
            level_declared(level_name).expected(
                "simulation_role", "conductor", level_info.get("simulation_role")
            )
        )
    facts = _level_facts(level, level_name)
    _require_material(facts.material_id, materials, conductor_declared(semantic_id))
    return _ConductorLevel(level_name, level_info, facts)


class _ConductorHost(NamedTuple):
    """The solution region a conductor's voids belong to, and whether it is the generated vacuum."""

    host_void_semantic_id: str
    generated: bool


def _conductor_host(
    level: _ConductorLevel, semantic_id: str, solution_region_ids: Mapping[str, Any]
) -> _ConductorHost:
    """The solution region a conductor's voids belong to, and whether it is the generated vacuum."""
    auto_host = "host_void_semantic_id" not in level.info
    host = (
        "VACUUM_REGION"
        if auto_host
        else _identifier(
            level.info["host_void_semantic_id"],
            level_declared(level.name),
            "host_void_semantic_id",
        )
    )
    if host not in solution_region_ids and not auto_host:
        raise ValueError(
            level_declared(level.name).expected(
                "host_void_semantic_id", "a solution region of the stack", host
            )
        )
    return _ConductorHost(host, auto_host)


def _conductor_geometry(
    declaration: Mapping[str, Any], semantic_id: str, level: _ConductorLevel
) -> dict[str, Any]:
    """The conductor's geometry: its level's Z, then the PDK's geometry, then the component's own fields.

    A component that names another geometry source than the PDK's keeps none of the PDK's.
    """
    pdk_geometry = mapping(
        level.info.get("geometry", {}), level_declared(level.name), "geometry"
    )
    owner = conductor_declared(semantic_id)
    component_geometry = mapping(declaration.get("geometry", {}), owner, "geometry")
    unknown_geometry = set(component_geometry) - _COMPONENT_GEOMETRY_FIELDS
    if unknown_geometry:
        raise ValueError(
            owner.expected(
                "geometry fields",
                f"only {sorted(_COMPONENT_GEOMETRY_FIELDS)}; the others are the PDK's",
                sorted(unknown_geometry),
            )
        )
    if "geometry_source" in component_geometry and component_geometry[
        "geometry_source"
    ] != pdk_geometry.get("geometry_source"):
        pdk_geometry = {}
    return {
        "z_um": level.facts.z_um,
        "thickness_um": level.facts.thickness_um,
        "route_ab_fused_selector_mode": True,
        **pdk_geometry,
        **component_geometry,
    }


def coupon_domain_bounds(component: Any, padding_um: float) -> dict[str, float]:
    """Read component bounds and expand them by the requested padding."""
    owner = Declared("the plan's component", "the component passed to GeometryPlan")
    try:
        bbox = component.dbbox()
    except AttributeError as exc:
        raise TypeError(owner.refusal("it must expose dbbox()")) from exc
    if bbox is None:
        raise ValueError(owner.refusal("its dbbox() must return physical-unit bounds"))
    try:
        x_min, y_min, x_max, y_max = (
            float(bbox.left),
            float(bbox.bottom),
            float(bbox.right),
            float(bbox.top),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError(
            owner.refusal("its dbbox() must expose left, bottom, right and top")
        ) from exc
    if not all(isfinite(value) for value in (x_min, y_min, x_max, y_max)):
        raise ValueError(
            owner.expected("dbbox()", "finite", (x_min, y_min, x_max, y_max))
        )
    if x_min >= x_max or y_min >= y_max:
        raise ValueError(
            owner.expected(
                "dbbox()", "a positive X and Y extent", (x_min, y_min, x_max, y_max)
            )
        )
    return {
        "x_min_um": x_min - padding_um,
        "y_min_um": y_min - padding_um,
        "x_max_um": x_max + padding_um,
        "y_max_um": y_max + padding_um,
    }


def level_declared(name: object) -> Declared:
    """A level of the PDK layer stack, which the PDK declares."""
    return Declared(f"PDK level {name!r}", "the PDK layer stack")


def conductor_declared(semantic_id: object) -> Declared:
    """A component's conductor region, which the component declares."""
    return Declared(
        f"conductor region {semantic_id!r}",
        "the component's component_semantics conductor_regions",
    )


_LAYER_STACK = Declared("the PDK layer stack", "the PDK layer stack")


@dataclass(frozen=True)
class LevelFacts:
    """What a PDK level gives an Entity: its material, base Z, thickness and overlap priority (higher wins)."""

    material_id: str
    z_um: float
    thickness_um: float
    priority: int


def _level(levels: Mapping[str, Any], name: Any) -> Any:
    name = _identifier(name, _LAYER_STACK, "level name")
    try:
        return levels[name]
    except KeyError as exc:
        raise ValueError(
            _LAYER_STACK.refusal(f"it declares no level {name!r}")
        ) from exc


def _level_info(level: Any, name: Any) -> dict[str, Any]:
    return mapping(getattr(level, "info", None), level_declared(name), "info")


def _level_facts(level: Any, name: Any) -> LevelFacts:
    """Read one PDK level; return `priority` where a HIGHER value wins overlaps.

    gdsfactory `LayerLevel.mesh_order` is the opposite: the LOWER value wins.
    This is the single place that converts the PDK convention into
    `SemanticEntitySpec.priority` (higher wins), by negation. Material kind is
    decided before priority (dielectric over vacuum), so mesh_order ranks
    overlapping solution regions of the same kind only.
    """
    owner = level_declared(name)
    material_id = _identifier(getattr(level, "material", None), owner, "material")
    z_um = finite_number(getattr(level, "zmin", None), owner, "zmin")
    thickness_um = _positive_number(
        getattr(level, "thickness", None), owner, "thickness"
    )
    try:
        priority = -int(getattr(level, "mesh_order", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            owner.expected(
                "mesh_order", "an integer", getattr(level, "mesh_order", None)
            )
        ) from exc
    return LevelFacts(material_id, z_um, thickness_um, priority)


def _require_material(
    material_id: str, materials: Mapping[str, Mapping[str, Any]], owner: Declared
) -> None:
    if material_id not in materials:
        raise ValueError(
            owner.expected(
                "material", "a material of the plan's material records", material_id
            )
        )


def gds_layer(value: Any, owner: Declared, field: str = "gds_layer") -> tuple[int, int]:
    """Read an integer GDS layer and datatype pair; ``owner`` declares it in ``field``."""
    if (
        isinstance(value, str | bytes)
        or not isinstance(value, Sequence)
        or len(value) != 2
    ):
        raise TypeError(owner.expected(field, "a (layer, datatype) pair", value))
    layer, datatype = value
    if any(
        isinstance(item, bool) or not isinstance(item, int)
        for item in (layer, datatype)
    ):
        raise TypeError(owner.expected(field, "two integers", value))
    return layer, datatype


def _physical_declaration(
    info: Mapping[str, Any], level_name: str, material_id: str
) -> dict[str, Any]:
    """Read the level's ``physical_layer`` declaration; a level that still carries ``aedt_modeling`` is refused."""
    owner = level_declared(level_name)
    if "aedt_modeling" in info:
        raise ValueError(
            owner.refusal("it carries aedt_modeling; replace it with physical_layer")
        )
    raw = info.get(PHYSICAL_LAYER_FIELD)
    if not isinstance(raw, Mapping):
        raise ValueError(owner.expected(PHYSICAL_LAYER_FIELD, "a mapping", raw))
    declaration = PhysicalLayerDeclaration.from_payload(raw)
    if (
        declaration.source_level_id != level_name
        or declaration.material_id != material_id
    ):
        raise ValueError(
            owner.expected(
                "physical_layer source_level_id and material_id",
                f"{level_name!r} and {material_id!r}",
                (declaration.source_level_id, declaration.material_id),
            )
        )
    return declaration.to_payload()


def mapping(value: Any, owner: Declared, field: str) -> dict[str, Any]:
    """Copy ``field`` of ``owner``, which must be a mapping, into an ordinary dictionary."""
    if not isinstance(value, Mapping):
        raise TypeError(owner.expected(field, "a mapping", value))
    return dict(value)


def _identifier(value: Any, owner: Declared, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(owner.expected(field, "a non-empty string", value))
    return value


def finite_number(value: Any, owner: Declared, field: str) -> float:
    """Read ``field`` of ``owner`` as a finite number; a boolean is refused."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(owner.expected(field, "a finite number", value))
    result = float(value)
    if not isfinite(result):
        raise ValueError(owner.expected(field, "a finite number", value))
    return result


def _positive_number(value: Any, owner: Declared, field: str) -> float:
    result = finite_number(value, owner, field)
    if result <= 0.0:
        raise ValueError(owner.expected(field, "a finite positive number", value))
    return result


def normalize_material_records(
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Normalize material records with the established stack-material mapping."""
    owner = Declared("material_records", "the material records passed to GeometryPlan")
    if not isinstance(records, Mapping):
        raise TypeError(owner.expected("material_records", "a mapping", records))
    kinds = {
        "superconductor": "conductor",
        "conductive": "conductor",
        "conductor": "conductor",
        "vacuum": "vacuum",
        "dielectric": "dielectric",
    }
    result: dict[str, dict[str, Any]] = {}
    for name, record in records.items():
        if not isinstance(record, Mapping):
            raise TypeError(owner.expected(f"record {name!r}", "a mapping", record))
        raw_kind = record.get("material_kind", record.get("kind"))
        if raw_kind not in kinds:
            continue
        normalized = kinds[raw_kind]
        item: dict[str, Any] = {"kind": normalized}
        if normalized != "conductor":
            permittivity = record.get(
                "relative_permittivity", record.get("permittivity")
            )
            if "relative_permittivity" in record or "permittivity" in record:
                item["permittivity"] = (
                    float(permittivity)
                    if isinstance(permittivity, (int, float))
                    and not isinstance(permittivity, bool)
                    else copy.deepcopy(permittivity)
                )
        if "loss_tangent" in record:
            loss_tangent = record["loss_tangent"]
            item["loss_tangent"] = (
                float(loss_tangent)
                if isinstance(loss_tangent, (int, float))
                and not isinstance(loss_tangent, bool)
                else copy.deepcopy(loss_tangent)
            )
        # These standard source fields are carried verbatim, including unsupported
        # shapes, so the backend cannot silently replace them with library values.
        for field in ("permeability", "conductivity", "aedt_library_name"):
            if field in record:
                item[field] = copy.deepcopy(record[field])
        result[str(name)] = item
    if not result:
        raise ValueError(
            owner.refusal(
                "none has a supported kind (superconductor, conductive, conductor, "
                "vacuum or dielectric)"
            )
        )
    return result
