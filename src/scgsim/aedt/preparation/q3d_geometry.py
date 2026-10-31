"""Lower a Q3D geometry snapshot into a sealed `Q3dSpec`.

Owns: checking that the caller's Net, material and ground declarations match the
GeometryPlan, turning every imported Entity into body declarations, and writing the
canonical source attachments (original GDS, stack, trace) that travel in the handoff.
Does not own: classification or enclosure projection (the physical-geometry planner)
or the handoff cohort itself (`handoff.py`). AEDT is never opened here.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from scgsim.aedt._data import canonical_sha256, detached
from scgsim.aedt._io import file_sha256, write_json
from scgsim.aedt.preparation.modeling import _interpret_geometry
from scgsim.aedt.specs.common import MatrixRunControl, PdkMaterial
from scgsim.aedt.specs.modeling import Modeling
from scgsim.aedt.specs.q3d import Q3dBodySpec, Q3dNetSpec, Q3dSpec
from scgsim.geometry import GeometryPlanSnapshot
from scgsim.semantics.route_a import geometry_z_range


def lower_q3d_geometry(  # noqa: PLR0913 - public keyword-only preparation API
    snapshot: GeometryPlanSnapshot,
    *,
    modeling: Modeling,
    directory: Path,
    project_name: str,
    design_name: str,
    materials: Mapping[str, PdkMaterial],
    net_types: Mapping[str, Literal["Signal", "Ground"]],
    physical_ground_nets: Sequence[str],
    run_control: MatrixRunControl | None = None,
    region_padding_um: Sequence[float],
) -> Q3dSpec:
    """Write canonical source provenance and body declarations; never open AEDT."""
    if not isinstance(snapshot, GeometryPlanSnapshot):
        raise TypeError("snapshot must be GeometryPlanSnapshot")
    source = snapshot.geometry_input
    if source.boundary_curves or source.boundary_reconstruction:
        raise NotImplementedError(
            "AEDT finite Q3D lowering does not support active source curve intent"
        )
    stack = snapshot.stack
    _, _, physical_plan = _interpret_geometry(source, stack, modeling)
    layer_bindings = {
        key: value["physical_layer_id"] for key, value in physical_plan.entities.items()
    }

    net_ids, ground_nets = _declared_nets(source, net_types, physical_ground_nets)
    _check_material_bindings(source, stack["materials"], materials)
    vacuum_id = _vacuum_material_id(source, stack["materials"], materials)

    dbu_um = source.metadata["source_dbu_um"]
    bodies, members = _body_declarations(source, layer_bindings, ground_nets, net_ids)
    geometry_source = _write_source_attachments(
        snapshot, directory, dbu_um=dbu_um, ground_nets=ground_nets, bodies=bodies
    )
    return Q3dSpec(
        modeling=modeling,
        physical_layers=physical_plan.declarations,
        physical_representation=physical_plan,
        project_name=project_name,
        design_name=design_name,
        materials=materials,
        vacuum_material_id=vacuum_id,
        bodies=tuple(bodies),
        nets=tuple(
            Q3dNetSpec(net_id, net_types[net_id], tuple(members[net_id]))
            for net_id in sorted(net_ids)
        ),
        run_control=run_control,
        region_padding_um=region_padding_um,
        solve_ac_rl=False,
        geometry_source=geometry_source,
    )


def _declared_nets(source, net_types, physical_ground_nets):
    """Return the final conductor Nets and the selected physical ground Nets.

    The caller owns Net membership and native Net types independently of which
    Nets are physical ground, so both must match the GeometryPlan exactly.
    """
    conductors = [
        entity for entity in source.entities if entity.material_kind == "conductor"
    ]
    if any(entity.net_id is None for entity in conductors):
        raise ValueError(
            "Q3D conductors require explicit final GeometryPlan Net membership"
        )
    net_ids = {entity.net_id for entity in conductors}
    if set(net_types) != net_ids:
        raise ValueError("net_types must cover exactly the final conductor Nets")
    ground_nets = tuple(physical_ground_nets)
    if isinstance(physical_ground_nets, str | bytes) or not set(ground_nets) <= net_ids:
        raise ValueError("physical_ground_nets must select final conductor Nets")
    return net_ids, ground_nets


def _check_material_bindings(source, source_materials, materials) -> None:
    """Require each explicit material to restate the source stack's material."""
    for entity in source.entities:
        material = materials[entity.material_id]
        record = source_materials[entity.material_id]
        # The caller's conductor materials are superconductors; the stack only
        # records them as generic conductors.
        expected_kind = (
            "superconductor"
            if entity.material_kind == "conductor"
            else entity.material_kind
        )
        if (
            material.material_id != entity.material_id
            or material.kind != expected_kind
            or record["kind"] != entity.material_kind
        ):
            raise ValueError(
                f"Q3D explicit material binding differs from source {entity.material_id!r}"
            )


def _vacuum_material_id(source, source_materials, materials) -> str:
    """Find the single vacuum material, preferring one an Entity already uses."""
    vacuum_ids = {
        entity.material_id
        for entity in source.entities
        if entity.material_kind == "vacuum"
    }
    if not vacuum_ids:
        vacuum_ids = {
            key for key, value in source_materials.items() if value["kind"] == "vacuum"
        }
    if len(vacuum_ids) != 1:
        raise ValueError("Q3D source requires exactly one vacuum material identity")
    vacuum_id = next(iter(vacuum_ids))
    if (
        materials[vacuum_id].kind != "vacuum"
        or materials[vacuum_id].material_id != vacuum_id
    ):
        raise ValueError("Q3D vacuum binding differs from source")
    return vacuum_id


def _body_declarations(source, layer_bindings, ground_nets, net_ids):
    """Declare one body per source polygon of every conductor and dielectric Entity.

    Returns the bodies and, for each Net, the ids of its conductor bodies.
    """
    imported = [
        entity
        for entity in source.entities
        if entity.material_kind in {"conductor", "dielectric"}
    ]
    polygons = {polygon.polygon_id: polygon for polygon in source.polygons}
    bodies = []
    members = {net_id: [] for net_id in sorted(net_ids)}
    for entity in sorted(imported, key=lambda value: value.semantic_id):
        zmin, zmax = geometry_z_range(entity.geometry, entity.semantic_id)
        if zmax <= zmin:
            raise ValueError(
                f"Q3D entity {entity.semantic_id!r} requires positive finite thickness"
            )
        role = (
            "substrate"
            if entity.material_kind == "dielectric"
            else "ground"
            if entity.net_id in ground_nets
            else "signal"
        )
        for polygon_id, exterior, holes in _entity_polygons(entity, polygons):
            body_id = f"Q3D_BODY_{len(bodies) + 1:06d}"
            source_pair = (
                (entity.geometry["gds_layer"], entity.geometry["gds_datatype"])
                if "gds_layer" in entity.geometry
                else None
            )
            bodies.append(
                Q3dBodySpec(
                    body_id=body_id,
                    physical_layer_id=layer_bindings.get(entity.semantic_id),
                    exterior_um=exterior,
                    holes_um=holes,
                    z_min_um=zmin,
                    z_max_um=zmax,
                    material_id=entity.material_id,
                    physical_role=role,
                    net_id=entity.net_id
                    if entity.material_kind == "conductor"
                    else None,
                    source_entity_id=entity.semantic_id,
                    source_polygon_id=polygon_id,
                    source_occurrence_path=entity.metadata.get(
                        "source_occurrence_path"
                    ),
                    source_local_entity_id=entity.metadata.get(
                        "source_local_entity_id"
                    ),
                    source_level=entity.metadata.get(
                        "logical_layer_id", entity.metadata.get("pdk_level_id")
                    ),
                    source_layer_datatype=source_pair,
                )
            )
            if entity.material_kind == "conductor":
                members[entity.net_id].append(body_id)
    return bodies, members


def _entity_polygons(entity, polygons):
    """List `(source polygon id, exterior, holes)` for one Entity's body geometry.

    Entities without source polygons fall back to the loop or domain box in their
    geometry; a conductor must always have source polygons.
    """
    if entity.polygon_ids:
        return [
            (pid, polygons[pid].exterior, polygons[pid].holes)
            for pid in sorted(entity.polygon_ids)
        ]
    if entity.material_kind == "conductor":
        raise ValueError(f"Q3D conductor {entity.semantic_id!r} has no source polygons")
    geometry = entity.geometry
    if "outer_loop" in geometry:
        return [(None, geometry["outer_loop"], geometry.get("hole_loops", ()))]
    bounds = geometry["domain_bounds_um"]
    xmin, ymin = bounds["x_min_um"], bounds["y_min_um"]
    xmax, ymax = bounds["x_max_um"], bounds["y_max_um"]
    return [(None, ((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)), ())]


def _write_source_attachments(snapshot, directory, *, dbu_um, ground_nets, bodies):
    """Write the original GDS, stack and trace, and return the sealed source record."""
    source = snapshot.geometry_input
    stack = snapshot.stack
    directory.mkdir(parents=True, exist_ok=True)
    original_path = directory / "source.gds"
    original_path.write_bytes(snapshot.gds_bytes)
    stack_path = directory / "geometry_stack.json"
    write_json(stack_path, stack)
    source_payload = detached(asdict(source))
    # These adapter paths identify temporary snapshot transport, not source
    # semantics. Keep the detached trace portable without changing the snapshot.
    source_payload["metadata"]["gds_file"] = "geometry/source.gds"
    source_payload["metadata"]["stack_file"] = "metadata/geometry_stack.json"
    geometry_source = {
        "schema_version": "scgsim.aedt.q3d-geometry-source.v3",
        "source_gds_sha256": hashlib.sha256(snapshot.gds_bytes).hexdigest(),
        "source_geometry_sha256": canonical_sha256(source_payload),
        "source_stack_sha256": file_sha256(stack_path),
        "source_dbu_um": dbu_um,
        "physical_ground_nets": list(ground_nets),
        "bodies": [body.to_payload() for body in bodies],
    }
    trace_path = directory / "geometry_trace.json"
    write_json(
        trace_path,
        {
            "geometry_source": geometry_source,
            "source_geometry": source_payload,
            "source_occurrences": detached(snapshot.source_occurrences),
            "body_geometry": "normalized source exterior/holes without fragmentation; no locator sheets",
        },
    )
    geometry_source["files"] = {
        key: {"path": str(path), "sha256": file_sha256(path)}
        for key, path in (
            ("canonical_gds", original_path),
            ("stack", stack_path),
            ("trace", trace_path),
        )
    }
    return geometry_source
