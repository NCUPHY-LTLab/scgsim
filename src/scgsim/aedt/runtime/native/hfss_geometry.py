"""Construct or bind source-backed HFSS planar bodies and declared junctions."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from scgsim.aedt.data import detached
from scgsim.aedt.preparation.planar_junctions import junction_polygon
from scgsim.aedt.preparation.planar_source import native_entity_name, native_name
from scgsim.aedt.runtime.native.common import (
    native_boundary_type,
    native_object_boolean_property,
    native_object_evidence,
    native_object_property,
    resolve_native_assignment,
)
from scgsim.aedt.runtime.native.junction_readback import native_junction_live_readback
from scgsim.aedt.runtime.native.planar_bodies import (
    install_material_catalog,
    junction_geometry,
    material_readback,
    native_conductor_geometry,
    native_solution_materials,
    polygon_sheet,
    solution_body,
    swept_z_solid,
    validate_native_modeling_source,
)
from scgsim.aedt.runtime.native.planar_region import (
    create_epr_region,
    implicit_closed_enclosure,
    verified_region_bounds,
)
from scgsim.aedt.specs.geometry import PreparedPlanarGeometry
from scgsim.geometry.junctions.terminal import junction_terminal_line


def _native_lumped_supports(
    app: Any, prepared: PreparedPlanarGeometry, *, saved: bool = False
) -> list[dict[str, Any]]:
    """Bind neutral authored supports without inventing conductor electrodes."""
    records = []
    for support in prepared.source.get("lumped_supports", ()):
        value = detached(support)
        effective = value["effective"]
        name = native_name("lumped_support", value["support_id"])
        if saved:
            obj = app.modeler[name]
        else:
            obj = polygon_sheet(app, effective, name=name, z_um=effective["z_um"])
        evidence = native_object_evidence(app, name)
        if evidence["native_object_type"] != "Sheet" or len(obj.faces) != 1:
            raise RuntimeError(
                f"neutral support is not one native Sheet face: {value['support_id']!r}"
            )
        records.append({**value, **evidence, "object_name": obj.name})
    return records


def native_conductor_objects(
    prepared: PreparedPlanarGeometry,
    native: Mapping[str, Any],
    entity_ids: Sequence[str],
) -> list[str]:
    """Resolve explicit source Entities to all their created conductor bodies."""
    source_entities = prepared.source["conductors"]
    names = []
    for entity_id in entity_ids:
        semantic_ids = {
            item["semantic_id"]
            for item in source_entities
            if entity_id in {item["semantic_id"], item["source_semantic_id"]}
        }
        matches = [
            item["object_name"]
            for item in native["objects"]
            if item["kind"] == "conductor" and item["semantic_id"] in semantic_ids
        ]
        if not matches:
            raise RuntimeError(f"source conductor has no native bodies: {entity_id!r}")
        for name in matches:
            if name not in names:
                names.append(name)
    return names


def prepare_native_planar_geometry(
    app: Any, prepared: PreparedPlanarGeometry
) -> dict[str, Any]:
    """Create bodies and live evidence; runtime completes lines after native save."""

    if not isinstance(prepared, PreparedPlanarGeometry):
        raise TypeError("prepared must be PreparedPlanarGeometry")
    if prepared.modeling not in {"solid", "thin_film"}:
        raise ValueError("new EPR native construction requires explicit modeling")
    source = detached(prepared.source)
    if (
        source.get("native_region", {}).get("method")
        != "single_region_absolute_offset.v1"
    ):
        raise ValueError("EPR geometry predates single Region; reprepare the handoff")
    if source["native_region"].get("outer_boundary_policy") != "hfss_default.v1":
        raise ValueError("EPR Region boundary policy requires re-preparation")
    validate_native_modeling_source(source)
    polygons = {item["polygon_id"]: item for item in source["polygons"]}
    junction_regions = {
        item["source_polygon_id"]: item for item in source["junction_regions"]
    }
    polygons.update(
        {
            source_id: {
                "polygon_id": source_id,
                "layer": item["source_layer"],
                "exterior": item["exterior"],
                "holes": item["holes"],
                "object_name": None,
                "net_name": None,
                "port_name": item["port_sheet_id"],
            }
            for source_id, item in junction_regions.items()
        }
    )
    bindings: list[dict[str, Any]] = []
    phase_seconds: dict[str, float] = {}

    started = time.perf_counter()
    material_readback = install_material_catalog(app, native_solution_materials(source))
    phase_seconds["materials_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    _create_solution_bodies(app, source, bindings)
    phase_seconds["interior_solution_bodies_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    _create_conductors(app, prepared, source, polygons, bindings)
    phase_seconds["conductors_seconds"] = time.perf_counter() - started

    junction_bindings: list[dict[str, Any]] = []
    started = time.perf_counter()
    junction_bindings = _create_junctions(
        app, prepared, source, polygons, junction_regions
    )
    phase_seconds["junctions_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    bindings.insert(0, create_epr_region(app, source))
    enclosure = implicit_closed_enclosure(app, source)
    phase_seconds["region_boundary_seconds"] = time.perf_counter() - started

    surface_selections: list[dict[str, Any]] = []

    object_names = [item["object_name"] for item in bindings] + [
        item["object_name"] for item in junction_bindings
    ]
    if len(object_names) != len(set(object_names)):
        raise RuntimeError("native planar object names are not unique")
    return {
        "schema_version": "scgsim.aedt.epr-native-binding.v1",
        "modeling": prepared.modeling,
        "source_sha256": prepared.source_sha256,
        "objects": bindings,
        **(
            {"lumped_supports": _native_lumped_supports(app, prepared)}
            if source.get("lumped_supports")
            else {}
        ),
        "junctions": junction_bindings,
        "surface_selections": surface_selections,
        "material_readback": material_readback,
        "closed_enclosure": enclosure,
        "geometry_phases": {
            key: round(value, 6) for key, value in phase_seconds.items()
        },
    }


def bind_saved_planar_geometry(
    app: Any, prepared: PreparedPlanarGeometry
) -> dict[str, Any]:
    """Bind deterministic native objects in a saved project without creating CAD."""

    if not isinstance(prepared, PreparedPlanarGeometry):
        raise TypeError("prepared must be PreparedPlanarGeometry")
    source = detached(prepared.source)
    validate_native_modeling_source(source)
    if (
        source.get("native_region", {}).get("method")
        != "single_region_absolute_offset.v1"
    ):
        raise ValueError("saved EPR geometry predates single Region; reprepare")
    objects = _bind_saved_solution_bodies(app, source)
    region_plan = source["native_region"]

    _bind_saved_conductors(app, prepared, source, objects)

    junctions: list[dict[str, Any]] = []
    for junction in prepared.junctions:
        name = native_name("junction", junction.junction_id)
        evidence = native_object_evidence(app, name)
        if evidence["native_object_type"] != "Sheet":
            raise RuntimeError(f"saved junction {name!r} is not a sheet")
        partition_readback = (
            native_junction_live_readback(app, source, junction)
            if junction.junction_id in source.get("junction_partitions", {})
            else None
        )
        junctions.append(
            {
                "junction_id": junction.junction_id,
                "source_polygon_id": junction.source_polygon_id,
                "object_name": name,
                "boundary_name": native_name("junction_rlc", junction.junction_id),
                **(
                    {"_partition_live_readback": partition_readback}
                    if partition_readback is not None
                    else {}
                ),
                **evidence,
            }
        )

    selections: list[dict[str, Any]] = []
    if region_plan.get("outer_boundary_policy") != "hfss_default.v1":
        raise ValueError("saved EPR Region boundary policy requires re-preparation")
    enclosure = {
        **implicit_closed_enclosure(app, source),
        "binding_stage": "saved_project_readback_without_cad_creation",
    }
    return {
        "schema_version": "scgsim.aedt.epr-native-binding.v1",
        "modeling": prepared.modeling,
        "source_sha256": prepared.source_sha256,
        "objects": objects,
        **(
            {"lumped_supports": _native_lumped_supports(app, prepared, saved=True)}
            if source.get("lumped_supports")
            else {}
        ),
        "junctions": junctions,
        "surface_selections": selections,
        "material_readback": material_readback(app, native_solution_materials(source)),
        "closed_enclosure": enclosure,
        "binding_stage": "saved_project_readback_without_cad_creation",
    }


def _create_solution_bodies(
    app: Any, source: Mapping[str, Any], bindings: list[dict[str, Any]]
) -> None:
    """Create interior solution bodies in declared source order."""
    for entity in source["solution_regions"]:
        if entity["metadata"].get("is_auto_vacuum_region"):
            continue
        obj = solution_body(app, entity, source)
        evidence = native_object_evidence(app, obj.name)
        if evidence["native_object_type"] != "Solid":
            raise RuntimeError(
                f"solution domain {entity['semantic_id']!r} is not solid"
            )
        bindings.append(
            {
                "kind": "solution_domain",
                "semantic_id": entity["semantic_id"],
                "material_id": entity["material_id"],
                "object_name": obj.name,
                **evidence,
            }
        )


def _create_conductors(
    app: Any,
    prepared: PreparedPlanarGeometry,
    source: Mapping[str, Any],
    polygons: Mapping[str, Any],
    bindings: list[dict[str, Any]],
) -> None:
    """Create source conductors and verify PEC assignments in source order."""
    entities = source["conductors"]
    junction_polygons = {item.source_polygon_id for item in prepared.junctions}
    for entity in entities:
        representation, z_min_um, z_max_um = native_conductor_geometry(source, entity)
        is_sheet = representation == "sheet"
        z_um = z_min_um
        thickness_um = z_max_um - z_min_um
        for polygon_id in entity["polygon_ids"]:
            if polygon_id in junction_polygons:
                continue
            name = native_entity_name(
                source, "conductor", entity["semantic_id"], polygon_id
            )
            boundary_name: str | None = None
            if is_sheet:
                obj = polygon_sheet(app, polygons[polygon_id], name=name, z_um=z_um)
                boundary_name = native_name(
                    "pec_boundary", entity["semantic_id"], polygon_id
                )
                boundary = app.assign_perfect_e(obj.name, name=boundary_name)
                if boundary is False or boundary is None:
                    raise RuntimeError(f"failed to assign Perfect E to {obj.name!r}")
                if native_boundary_type(app, boundary_name) != "Perfect E":
                    raise RuntimeError(f"Perfect E readback failed for {obj.name!r}")
            else:
                if thickness_um <= 0.0:
                    raise ValueError(
                        f"finite conductor {entity['semantic_id']!r} requires positive thickness"
                    )
                obj = swept_z_solid(
                    app,
                    polygons[polygon_id],
                    name=name,
                    z_min_um=z_min_um,
                    z_max_um=z_max_um,
                )
                obj.material_name = "pec"
                obj.solve_inside = False
            evidence = native_object_evidence(app, obj.name)
            expected_type = "Sheet" if is_sheet else "Solid"
            if evidence["native_object_type"] != expected_type:
                raise RuntimeError(f"native conductor type mismatch for {obj.name!r}")
            observed: dict[str, Any] = {}
            if not is_sheet:
                observed = {
                    "native_material_name": native_object_property(
                        obj, "Material"
                    ).strip('"'),
                    "native_solve_inside": native_object_boolean_property(
                        obj, "Solve Inside"
                    ),
                }
                if (
                    observed["native_material_name"].casefold() != "pec"
                    or observed["native_solve_inside"] is not False
                ):
                    raise RuntimeError(
                        f"native finite PEC readback mismatch for {obj.name!r}"
                    )
            bindings.append(
                {
                    "kind": "conductor",
                    "semantic_id": entity["semantic_id"],
                    "source_polygon_id": polygon_id,
                    "modeling": prepared.modeling,
                    "physical_layer": detached(entity["physical_layer"]),
                    "object_name": obj.name,
                    "boundary_name": boundary_name,
                    "observed": observed,
                    **evidence,
                }
            )


def _create_junctions(
    app: Any,
    prepared: PreparedPlanarGeometry,
    source: Mapping[str, Any],
    polygons: Mapping[str, Any],
    junction_regions: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Create declared junction sheets, assignments and live readback records."""
    entities = source["conductors"]
    junction_bindings: list[dict[str, Any]] = []
    for junction in prepared.junctions:
        polygon, partition = junction_polygon(source, junction, polygons)
        region = junction_regions.get(junction.source_polygon_id)
        if partition is not None:
            owner_ids = [end["source_entity_id"] for end in partition["ends"]]
        elif region is None:
            owner_ids = [
                item["semantic_id"]
                for item in entities
                if junction.source_polygon_id in item["polygon_ids"]
            ]
        else:
            owner_ids = list(region["host_semantic_ids"])
        owners = [item for item in entities if item["semantic_id"] in owner_ids]
        if not owners or {item["net_id"] for item in owners} != {
            junction.terminal_a_net,
            junction.terminal_b_net,
        }:
            raise RuntimeError(
                f"junction {junction.junction_id!r} source owners do not match its nets"
            )
        z_values = [native_conductor_geometry(source, owner)[1] for owner in owners]
        if max(z_values) - min(z_values) > 1e-9:
            raise RuntimeError(
                f"junction {junction.junction_id!r} owners are not coplanar"
            )
        z_um = partition["z_um"] if partition is not None else z_values[0]
        name = native_name("junction", junction.junction_id)
        sheet = polygon_sheet(app, polygon, name=name, z_um=z_um)
        line, span_um = junction_terminal_line(
            polygon, junction_geometry(junction), z_um
        )
        boundary_name = native_name("junction_rlc", junction.junction_id)
        boundary = app.assign_lumped_rlc_to_sheet(
            sheet.name,
            line,
            name=boundary_name,
            rlc_type="Parallel",
            inductance=junction.inductance_h,
            capacitance=(junction.capacitance_f or None),
        )
        if boundary is False or boundary is None:
            raise RuntimeError(f"failed to assign junction {junction.junction_id!r}")
        evidence = native_object_evidence(app, sheet.name)
        partition_readback = (
            native_junction_live_readback(app, source, junction)
            if partition is not None
            else None
        )
        junction_bindings.append(
            {
                "junction_id": junction.junction_id,
                "source_polygon_id": junction.source_polygon_id,
                "object_name": sheet.name,
                "boundary_name": boundary_name,
                "terminal_a_net": junction.terminal_a_net,
                "terminal_b_net": junction.terminal_b_net,
                "integration_line_um": line,
                "terminal_span_um": span_um,
                "width_um": junction.width_um,
                "inductance_h": junction.inductance_h,
                "capacitance_f": junction.capacitance_f,
                **(
                    {"_partition_live_readback": partition_readback}
                    if partition_readback is not None
                    else {}
                ),
                **(
                    {
                        "central_polygon_id": partition["central_polygon_id"],
                        "partition_method": partition["method"],
                    }
                    if partition is not None
                    else {}
                ),
                **evidence,
            }
        )
    return junction_bindings


def _bind_saved_conductors(
    app: Any,
    prepared: PreparedPlanarGeometry,
    source: Mapping[str, Any],
    objects: list[dict[str, Any]],
) -> None:
    """Match saved conductor identities and verify their PEC implementation."""
    junction_polygon_ids = {item.source_polygon_id for item in prepared.junctions}
    for entity in source["conductors"]:
        representation, _, _ = native_conductor_geometry(source, entity)
        is_sheet = representation == "sheet"
        for polygon_id in entity["polygon_ids"]:
            if polygon_id in junction_polygon_ids:
                continue
            name = native_entity_name(
                source, "conductor", entity["semantic_id"], polygon_id
            )
            evidence = native_object_evidence(app, name)
            expected_type = "Sheet" if is_sheet else "Solid"
            if evidence["native_object_type"] != expected_type:
                raise RuntimeError(f"saved conductor {name!r} has wrong object type")
            boundary_name: str | None = None
            observed: dict[str, Any] = {}
            if is_sheet:
                boundary_name = native_name(
                    "pec_boundary", entity["semantic_id"], polygon_id
                )
                _verify_saved_sheet_pec(app, name, boundary_name)
            else:
                obj = app.modeler.get_object_from_name(name)
                if obj is None:
                    raise RuntimeError(f"saved conductor {name!r} is unavailable")
                observed = {
                    "native_material_name": native_object_property(
                        obj, "Material"
                    ).strip('"'),
                    "native_solve_inside": native_object_boolean_property(
                        obj, "Solve Inside"
                    ),
                }
                if (
                    observed["native_material_name"].casefold() != "pec"
                    or observed["native_solve_inside"] is not False
                ):
                    raise RuntimeError(f"saved conductor {name!r} PEC readback differs")
            objects.append(
                {
                    "kind": "conductor",
                    "semantic_id": entity["semantic_id"],
                    "source_polygon_id": polygon_id,
                    "modeling": prepared.modeling,
                    **(
                        {"physical_layer": detached(entity["physical_layer"])}
                        if isinstance(entity.get("physical_layer"), Mapping)
                        else {}
                    ),
                    "object_name": name,
                    "boundary_name": boundary_name,
                    "observed": observed,
                    **evidence,
                }
            )


def _bind_saved_solution_bodies(
    app: Any, source: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Verify saved interior bodies and the Region before binding conductors."""
    objects: list[dict[str, Any]] = []
    for domain in source["solution_regions"]:
        if domain["metadata"].get("is_auto_vacuum_region"):
            continue
        name = native_entity_name(source, "domain", domain["semantic_id"])
        evidence = native_object_evidence(app, name)
        if evidence["native_object_type"] != "Solid":
            raise RuntimeError(f"saved solution domain {name!r} is not solid")
        objects.append(
            {
                "kind": "solution_domain",
                "semantic_id": domain["semantic_id"],
                "material_id": domain["material_id"],
                "object_name": name,
                **evidence,
            }
        )
    region_plan = source["native_region"]
    saved_region = app.modeler.get_object_from_name("Region")
    if saved_region is None:
        raise RuntimeError("saved EPR Region is unavailable")
    region_evidence = native_object_evidence(app, "Region")
    if region_evidence["native_object_type"] != "Solid":
        raise RuntimeError("saved EPR Region is not solid")
    saved_material = native_object_property(saved_region, "Material").strip('"')
    if saved_material.casefold() != str(region_plan["material_id"]).casefold():
        raise RuntimeError("saved EPR Region vacuum material differs")
    if not native_object_boolean_property(saved_region, "Solve Inside"):
        raise RuntimeError("saved EPR Region does not solve inside vacuum")
    objects.insert(
        0,
        {
            "kind": "solution_domain",
            "semantic_id": "Region",
            "material_id": region_plan["material_id"],
            "logical_vacuum_ids": list(region_plan["logical_vacuum_ids"]),
            "object_name": "Region",
            "padding_um": list(region_plan["padding_um"]),
            "native_solve_inside": True,
            "native_bounding_box_um": list(
                verified_region_bounds(saved_region, region_plan)
            ),
            **region_evidence,
        },
    )

    return objects


def _verify_saved_sheet_pec(app: Any, name: str, boundary_name: str) -> None:
    """Verify saved Perfect E kind, assignment IDs and object target."""
    if native_boundary_type(app, boundary_name) != "Perfect E":
        raise RuntimeError(f"saved conductor {name!r} lacks Perfect E")
    assigned = app.oboundary.GetBoundaryAssignment(boundary_name)
    if assigned is None:
        raise RuntimeError(f"saved conductor {name!r} lacks PEC assignment")
    try:
        raw_ids = [int(value) for value in assigned]
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"saved conductor {name!r} has invalid PEC IDs") from exc
    if not raw_ids or len(raw_ids) != len(set(raw_ids)):
        raise RuntimeError(f"saved conductor {name!r} has invalid PEC IDs")
    _, _, assigned_objects = resolve_native_assignment(app, raw_ids)
    if assigned_objects != {name}:
        raise RuntimeError(f"saved conductor {name!r} PEC target differs")
