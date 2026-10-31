"""Prepare and re-bind the native HFSS planar model for a request.

Owns the two entry points of native geometry: ``prepare_native_planar_geometry``
builds bodies, materials, enclosure and declared junctions in a live design, and
``bind_saved_planar_geometry`` re-binds the same evidence against the saved project.
Construction does not depend on whether an EPR (energy-participation ratio)
analysis was requested. The pieces live in sibling modules: ``hfss_bodies``,
``hfss_planar_topology``, ``hfss_junction_readback`` and ``hfss_enclosure``.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from scgsim.aedt._data import detached
from scgsim.aedt.preparation.planar import (
    _junction_polygon,
    _junction_terminal_line,
    _native_entity_name,
    _native_name,
)
from scgsim.aedt.runtime.native.common import (
    _native_boundary_type,
    _native_object_boolean_property,
    _native_object_evidence,
    _resolve_native_assignment,
    native_object_property,
)
from scgsim.aedt.runtime.native.hfss_bodies import (
    _install_material_catalog,
    _material_readback,
    _native_conductor_geometry,
    _native_solution_materials,
    _polygon_sheet,
    _prepared_modeling_identity,
    _solution_body,
    _swept_z_solid,
    _validate_native_modeling_source,
)
from scgsim.aedt.runtime.native.hfss_enclosure import (
    _create_epr_region,
    _implicit_closed_enclosure,
    _verified_region_bounds,
)
from scgsim.aedt.runtime.native.hfss_junction_readback import (
    _native_junction_live_readback,
)
from scgsim.aedt.specs.epr import PlanarJunction
from scgsim.aedt.specs.geometry import PreparedPlanarGeometry


def _native_lumped_supports(
    app: Any, prepared: PreparedPlanarGeometry, *, saved: bool = False
) -> list[dict[str, Any]]:
    """Bind neutral authored supports without inventing conductor electrodes."""
    records = []
    for support in prepared.source.get("lumped_supports", ()):
        value = detached(support)
        effective = value["effective"]
        name = _native_name("lumped_support", value["support_id"])
        if saved:
            obj = app.modeler[name]
        else:
            obj = _polygon_sheet(app, effective, name=name, z_um=effective["z_um"])
        evidence = _native_object_evidence(app, name)
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
    """Create bodies and live evidence; runtime completes lines after native save.

    Steps run in a fixed order because AEDT assigns object names and boundary
    state as it goes: materials, solution domains, conductors, junctions, then the
    Region and enclosure.
    """
    source = _validated_native_source(prepared)
    polygons, junction_regions = _polygons_with_junction_regions(source)
    phase_seconds: dict[str, float] = {}

    started = time.perf_counter()
    material_readback = _install_material_catalog(
        app, _native_solution_materials(source)
    )
    phase_seconds["materials_seconds"] = time.perf_counter() - started

    bindings: list[dict[str, Any]] = []
    started = time.perf_counter()
    bindings.extend(_create_solution_domains(app, source))
    phase_seconds["interior_solution_bodies_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    bindings.extend(_create_conductors(app, prepared, source, polygons))
    phase_seconds["conductors_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    junction_bindings = _create_junctions(
        app, prepared, source, polygons, junction_regions
    )
    phase_seconds["junctions_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    bindings.insert(0, _create_epr_region(app, source))
    enclosure = _implicit_closed_enclosure(app, source)
    phase_seconds["region_boundary_seconds"] = time.perf_counter() - started

    surface_selections: list[dict[str, Any]] = []

    object_names = [item["object_name"] for item in bindings] + [
        item["object_name"] for item in junction_bindings
    ]
    if len(object_names) != len(set(object_names)):
        raise RuntimeError("native planar object names are not unique")
    return {
        "schema_version": "scgsim.aedt.epr-native-binding.v1",
        **_prepared_modeling_identity(prepared),
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


def _validated_native_source(prepared: PreparedPlanarGeometry) -> dict[str, Any]:
    """Return a detached copy of the source after the checks native construction needs."""
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
    _validate_native_modeling_source(prepared, source)
    return source


def _polygons_with_junction_regions(
    source: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Index polygons by id, adding each junction region as a polygon of its own.

    Returns ``(polygons, junction_regions)``; junction regions are keyed by the
    polygon they were cut from.
    """
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
    return polygons, junction_regions


def _create_solution_domains(
    app: Any, source: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Create the interior solution bodies; the automatic vacuum region is skipped."""
    bindings = []
    for entity in source["solution_regions"]:
        if entity["metadata"].get("is_auto_vacuum_region"):
            continue
        obj = _solution_body(app, entity, source)
        evidence = _native_object_evidence(app, obj.name)
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
    return bindings


def _create_conductors(
    app: Any,
    prepared: PreparedPlanarGeometry,
    source: Mapping[str, Any],
    polygons: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Create one native PEC (perfect electric conductor) object per conductor polygon.

    A sheet conductor becomes a Perfect E boundary on a sheet; a finite conductor
    becomes a solid with material ``pec`` and ``solve_inside`` off. Junction
    polygons are skipped here and built by ``_create_junctions``.
    """
    bindings = []
    junction_polygons = {item.source_polygon_id for item in prepared.junctions}
    for entity in source["conductors"]:
        representation, z_min_um, z_max_um = _native_conductor_geometry(source, entity)
        is_sheet = representation == "sheet"
        z_um = z_min_um
        thickness_um = z_max_um - z_min_um
        for polygon_id in entity["polygon_ids"]:
            if polygon_id in junction_polygons:
                continue
            name = _native_entity_name(
                source, "conductor", entity["semantic_id"], polygon_id
            )
            boundary_name: str | None = None
            if is_sheet:
                obj = _polygon_sheet(app, polygons[polygon_id], name=name, z_um=z_um)
                boundary_name = _native_name(
                    "pec_boundary", entity["semantic_id"], polygon_id
                )
                boundary = app.assign_perfect_e(obj.name, name=boundary_name)
                if boundary is False or boundary is None:
                    raise RuntimeError(f"failed to assign Perfect E to {obj.name!r}")
                if _native_boundary_type(app, boundary_name) != "Perfect E":
                    raise RuntimeError(f"Perfect E readback failed for {obj.name!r}")
            else:
                if thickness_um <= 0.0:
                    raise ValueError(
                        f"finite conductor {entity['semantic_id']!r} requires positive thickness"
                    )
                obj = _swept_z_solid(
                    app,
                    polygons[polygon_id],
                    name=name,
                    z_min_um=z_min_um,
                    z_max_um=z_max_um,
                )
                obj.material_name = "pec"
                obj.solve_inside = False
            evidence = _native_object_evidence(app, obj.name)
            expected_type = "Sheet" if is_sheet else "Solid"
            if evidence["native_object_type"] != expected_type:
                raise RuntimeError(f"native conductor type mismatch for {obj.name!r}")
            observed: dict[str, Any] = {}
            if not is_sheet:
                observed = {
                    "native_material_name": native_object_property(
                        obj, "Material"
                    ).strip('"'),
                    "native_solve_inside": _native_object_boolean_property(
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
                    **_prepared_modeling_identity(prepared),
                    "physical_layer": detached(entity["physical_layer"]),
                    "object_name": obj.name,
                    "boundary_name": boundary_name,
                    "observed": observed,
                    **evidence,
                }
            )
    return bindings


def _junction_owner_z_um(
    source: Mapping[str, Any],
    junction: PlanarJunction,
    partition: Mapping[str, Any] | None,
    region: Mapping[str, Any] | None,
) -> float:
    """Check that the junction touches exactly its two nets on one plane; return that plane's z."""
    entities = source["conductors"]
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
    z_values = [_native_conductor_geometry(source, owner)[1] for owner in owners]
    if max(z_values) - min(z_values) > 1e-9:
        raise RuntimeError(f"junction {junction.junction_id!r} owners are not coplanar")
    return partition["z_um"] if partition is not None else z_values[0]


def _create_junctions(
    app: Any,
    prepared: PreparedPlanarGeometry,
    source: Mapping[str, Any],
    polygons: Mapping[str, Any],
    junction_regions: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Create each junction sheet with its lumped RLC (resistor/inductor/capacitor) boundary.

    Partitioned junctions also get a live CAD readback here; the terminal line is
    completed after the native save.
    """
    junction_bindings: list[dict[str, Any]] = []
    for junction in prepared.junctions:
        polygon, partition = _junction_polygon(source, junction, polygons)
        region = junction_regions.get(junction.source_polygon_id)
        z_um = _junction_owner_z_um(source, junction, partition, region)
        name = _native_name("junction", junction.junction_id)
        sheet = _polygon_sheet(app, polygon, name=name, z_um=z_um)
        line, span_um = _junction_terminal_line(polygon, junction, z_um)
        boundary_name = _native_name("junction_rlc", junction.junction_id)
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
        evidence = _native_object_evidence(app, sheet.name)
        partition_readback = (
            _native_junction_live_readback(app, source, junction)
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


def bind_saved_planar_geometry(
    app: Any, prepared: PreparedPlanarGeometry
) -> dict[str, Any]:
    """Bind deterministic native objects in a saved project without creating CAD.

    Mirrors ``prepare_native_planar_geometry`` step for step (domains, Region,
    conductors, junctions, enclosure), but only reads the saved project back.
    """
    if not isinstance(prepared, PreparedPlanarGeometry):
        raise TypeError("prepared must be PreparedPlanarGeometry")
    source = detached(prepared.source)
    _validate_native_modeling_source(prepared, source)
    if (
        source.get("native_region", {}).get("method")
        != "single_region_absolute_offset.v1"
    ):
        raise ValueError("saved EPR geometry predates single Region; reprepare")
    objects = _bind_saved_solution_domains(app, source)
    region_plan = source["native_region"]
    objects.insert(0, _bind_saved_region(app, region_plan))
    objects.extend(_bind_saved_conductors(app, prepared, source))
    junctions = _bind_saved_junctions(app, prepared, source)

    selections: list[dict[str, Any]] = []
    boundary_policy = region_plan.get("outer_boundary_policy")
    if boundary_policy == "hfss_default.v1":
        enclosure = {
            **_implicit_closed_enclosure(app, source),
            "binding_stage": "saved_project_readback_without_cad_creation",
        }
    else:
        raise ValueError("saved EPR Region boundary policy is unknown")
    return {
        "schema_version": "scgsim.aedt.epr-native-binding.v1",
        **_prepared_modeling_identity(prepared),
        "source_sha256": prepared.source_sha256,
        "objects": objects,
        **(
            {"lumped_supports": _native_lumped_supports(app, prepared, saved=True)}
            if source.get("lumped_supports")
            else {}
        ),
        "junctions": junctions,
        "surface_selections": selections,
        "material_readback": _material_readback(
            app, _native_solution_materials(source)
        ),
        "closed_enclosure": enclosure,
        "binding_stage": "saved_project_readback_without_cad_creation",
    }


def _bind_saved_solution_domains(
    app: Any, source: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Check each saved interior solution body is a solid; the automatic vacuum region is skipped."""
    objects: list[dict[str, Any]] = []
    for domain in source["solution_regions"]:
        if domain["metadata"].get("is_auto_vacuum_region"):
            continue
        name = _native_entity_name(source, "domain", domain["semantic_id"])
        evidence = _native_object_evidence(app, name)
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
    return objects


def _bind_saved_region(app: Any, region_plan: Mapping[str, Any]) -> dict[str, Any]:
    """Check the saved ``Region`` is a vacuum solid that solves inside, and record its bounds."""
    saved_region = app.modeler.get_object_from_name("Region")
    if saved_region is None:
        raise RuntimeError("saved EPR Region is unavailable")
    region_evidence = _native_object_evidence(app, "Region")
    if region_evidence["native_object_type"] != "Solid":
        raise RuntimeError("saved EPR Region is not solid")
    saved_material = native_object_property(saved_region, "Material").strip('"')
    if saved_material.casefold() != str(region_plan["material_id"]).casefold():
        raise RuntimeError("saved EPR Region vacuum material differs")
    if not _native_object_boolean_property(saved_region, "Solve Inside"):
        raise RuntimeError("saved EPR Region does not solve inside vacuum")
    return {
        "kind": "solution_domain",
        "semantic_id": "Region",
        "material_id": region_plan["material_id"],
        "logical_vacuum_ids": list(region_plan["logical_vacuum_ids"]),
        "object_name": "Region",
        "padding_um": list(region_plan["padding_um"]),
        "native_solve_inside": True,
        "native_bounding_box_um": list(
            _verified_region_bounds(saved_region, region_plan)
        ),
        **region_evidence,
    }


def _bind_saved_conductors(
    app: Any, prepared: PreparedPlanarGeometry, source: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Check each saved conductor object and its PEC (perfect electric conductor) state."""
    objects: list[dict[str, Any]] = []
    junction_polygon_ids = {item.source_polygon_id for item in prepared.junctions}
    for entity in source["conductors"]:
        representation, _, _ = _native_conductor_geometry(source, entity)
        is_sheet = representation == "sheet"
        for polygon_id in entity["polygon_ids"]:
            if polygon_id in junction_polygon_ids:
                continue
            name = _native_entity_name(
                source, "conductor", entity["semantic_id"], polygon_id
            )
            evidence = _native_object_evidence(app, name)
            expected_type = "Sheet" if is_sheet else "Solid"
            if evidence["native_object_type"] != expected_type:
                raise RuntimeError(f"saved conductor {name!r} has wrong object type")
            boundary_name: str | None = None
            observed: dict[str, Any] = {}
            if is_sheet:
                boundary_name = _native_name(
                    "pec_boundary", entity["semantic_id"], polygon_id
                )
                if _native_boundary_type(app, boundary_name) != "Perfect E":
                    raise RuntimeError(f"saved conductor {name!r} lacks Perfect E")
                assigned = app.oboundary.GetBoundaryAssignment(boundary_name)
                if assigned is None:
                    raise RuntimeError(f"saved conductor {name!r} lacks PEC assignment")
                try:
                    raw_ids = [int(value) for value in assigned]
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"saved conductor {name!r} has invalid PEC IDs"
                    ) from exc
                if not raw_ids or len(raw_ids) != len(set(raw_ids)):
                    raise RuntimeError(f"saved conductor {name!r} has invalid PEC IDs")
                _, _, assigned_objects = _resolve_native_assignment(app, raw_ids)
                if assigned_objects != {name}:
                    raise RuntimeError(f"saved conductor {name!r} PEC target differs")
            else:
                obj = app.modeler.get_object_from_name(name)
                if obj is None:
                    raise RuntimeError(f"saved conductor {name!r} is unavailable")
                observed = {
                    "native_material_name": native_object_property(
                        obj, "Material"
                    ).strip('"'),
                    "native_solve_inside": _native_object_boolean_property(
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
                    **_prepared_modeling_identity(prepared),
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
    return objects


def _bind_saved_junctions(
    app: Any, prepared: PreparedPlanarGeometry, source: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Check each saved junction sheet; partitioned junctions get a live CAD readback."""
    junctions: list[dict[str, Any]] = []
    for junction in prepared.junctions:
        name = _native_name("junction", junction.junction_id)
        evidence = _native_object_evidence(app, name)
        if evidence["native_object_type"] != "Sheet":
            raise RuntimeError(f"saved junction {name!r} is not a sheet")
        partition_readback = (
            _native_junction_live_readback(app, source, junction)
            if junction.junction_id in source.get("junction_partitions", {})
            else None
        )
        junctions.append(
            {
                "junction_id": junction.junction_id,
                "source_polygon_id": junction.source_polygon_id,
                "object_name": name,
                "boundary_name": _native_name("junction_rlc", junction.junction_id),
                **(
                    {"_partition_live_readback": partition_readback}
                    if partition_readback is not None
                    else {}
                ),
                **evidence,
            }
        )
    return junctions
