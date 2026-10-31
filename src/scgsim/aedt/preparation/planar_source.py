"""Detached source record of a planar HFSS model, checked against the prepared stack.

Owns: turning a normalized `GeometryBuildInput` plus its prepared stack into the plain
`source` mapping stored in `PreparedPlanarGeometry`: conductors, solution regions,
the native EPR `Region` and the junction partition records. Does not own: surface
contributions (`planar_surfaces.py`), junction partitioning (`junction_partition.py`)
or the entry point that assembles the prepared geometry (`planar.py`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from scgsim.aedt._data import canonical_sha256
from scgsim.aedt.specs.modeling import _native_physical_record
from scgsim.geometry import GeometryBuildInput
from scgsim.semantics.route_a import geometry_z_range

# Keys of Entity metadata that identify where a conductor came from in the source.
_CONDUCTOR_METADATA_KEYS = (
    "semantic_group_id",
    "source_semantic_id",
    "source_occurrence_path",
    "source_local_entity_id",
    "split_polygon_index",
    "ground_bump_id",
)

_PADDING_FACE_KEYS = (
    "x_plus_um",
    "x_minus_um",
    "y_plus_um",
    "y_minus_um",
    "z_plus_um",
    "z_minus_um",
)


def _plain_geometry_source(value: Any) -> Any:
    """Convert nested mappings and sequences to plain JSON-compatible values."""
    if isinstance(value, Mapping):
        return {str(key): _plain_geometry_source(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain_geometry_source(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"geometry source contains unsupported {type(value).__name__}")


def _source_payload(
    build_input: GeometryBuildInput,
    prepared_stack: Mapping[str, Any],
    *,
    source_dbu_um: float,
) -> dict[str, Any]:
    """Build the detached `source` record, in the key order the hashes cover."""
    materials, solution_regions, layers, metadata = _stack_sections(prepared_stack)
    material_catalog = _plain_geometry_source(materials)
    _check_entity_materials(build_input, materials)
    conductors = _conductor_records(build_input, layers)
    normalized_regions = _solution_region_records(build_input, solution_regions)
    native_region = _native_region(normalized_regions, metadata)
    return {
        "schema_version": "scgsim.aedt.epr-planar-source.v1",
        "modeling": _plain_geometry_source(metadata.get("physical_representation")),
        "source_dbu_um": source_dbu_um,
        "prepared_stack_sha256": canonical_sha256(
            _plain_geometry_source(prepared_stack)
        ),
        "source_stack_sha256": metadata["physical_source_stack_sha256"],
        "materials": material_catalog,
        "solution_regions": normalized_regions,
        "native_region": native_region,
        "native_entity_name_policy": (
            "readable.v1"
            if metadata.get("component_semantics_schema_version") == 2
            else "hash.v1"
        ),
        "surface_evaluation_policy": "unmasked_plus_requested.v1",
        **(
            {
                "junction_partitions": _plain_geometry_source(
                    build_input.metadata["aedt_junction_partitions"]
                ),
                "authored_input_sha256": build_input.metadata[
                    "aedt_authored_input_sha256"
                ],
            }
            if "aedt_junction_partitions" in build_input.metadata
            else {}
        ),
        "conductors": conductors,
        "junction_regions": _junction_region_records(build_input),
        "polygons": _polygon_records(build_input),
    }


def _stack_sections(prepared_stack: Mapping[str, Any]):
    """Return `(materials, solution_regions, layers, metadata)` or raise TypeError."""
    materials = prepared_stack.get("materials")
    solution_regions = prepared_stack.get("solution_regions")
    layers = prepared_stack.get("layers")
    metadata = prepared_stack.get("metadata", {})
    if (
        not isinstance(materials, Mapping)
        or not isinstance(solution_regions, Mapping)
        or isinstance(layers, (str, bytes))
        or not isinstance(layers, Sequence)
        or not isinstance(metadata, Mapping)
    ):
        raise TypeError(
            "prepared_stack must contain materials, solution_regions, layers, and metadata"
        )
    return materials, solution_regions, layers, metadata


def _check_entity_materials(build_input, materials: Mapping[str, Any]) -> None:
    """Require every Entity's material and material kind to match the stack."""
    for entity in build_input.entities:
        material = materials.get(entity.material_id)
        if not isinstance(material, Mapping):
            raise ValueError(
                f"geometry entity {entity.semantic_id!r} material is absent from prepared_stack"
            )
        if material.get("kind") != entity.material_kind:
            raise ValueError(
                f"geometry entity {entity.semantic_id!r} material kind contradicts prepared_stack"
            )


def _conductor_records(build_input, layers: Sequence[Any]) -> list[dict[str, Any]]:
    """Describe each conductor Entity and bind it to its prepared-stack layer."""
    source_layers = {
        str(record.get("semantic_id")): record
        for record in layers
        if isinstance(record, Mapping) and isinstance(record.get("semantic_id"), str)
    }
    conductors = []
    for item in build_input.entities:
        if item.material_kind != "conductor":
            continue
        source_id = _stack_source_id(item)
        if source_id not in source_layers:
            raise ValueError(
                f"geometry conductor {item.semantic_id!r} has no prepared-stack source record"
            )
        source_record = source_layers[source_id]
        if source_record.get("material_id") != item.material_id:
            raise ValueError(
                f"geometry conductor {item.semantic_id!r} material contradicts prepared_stack"
            )
        conductors.append(
            {
                "semantic_id": item.semantic_id,
                "source_semantic_id": source_id,
                "role": item.role,
                "material_id": item.material_id,
                "material_kind": item.material_kind,
                "part_role": item.part_role,
                "representation": item.metadata["physical_representation"][
                    "representation"
                ],
                "physical_layer": _native_physical_record(
                    item.metadata["physical_representation"]
                ),
                "net_id": item.net_id,
                "polygon_ids": list(item.polygon_ids),
                "geometry": _plain_geometry_source(item.geometry),
                "metadata": {
                    key: _plain_geometry_source(item.metadata[key])
                    for key in _CONDUCTOR_METADATA_KEYS
                    if key in item.metadata
                },
            }
        )
    return conductors


def _stack_source_id(item) -> str:
    """Name the prepared-stack layer a conductor Entity was split from.

    An Entity split from a larger polygon group looks up its group, not itself.
    """
    source_id = str(
        item.metadata.get("source_semantic_id", item.semantic_id)
        if "split_polygon_index" in item.metadata
        else item.semantic_id
    )
    if (
        "split_polygon_index" in item.metadata
        and "source_semantic_id" not in item.metadata
    ):
        source_id = str(item.metadata.get("semantic_group_id", item.semantic_id))
    return source_id


def _solution_region_records(
    build_input, solution_regions: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Describe each vacuum or dielectric Entity as a solution-domain region."""
    normalized_regions = []
    for entity in build_input.entities:
        if entity.material_kind not in {"vacuum", "dielectric"}:
            continue
        semantic_id = entity.semantic_id
        record = solution_regions.get(semantic_id)
        if record is not None and not isinstance(record, Mapping):
            raise TypeError("prepared_stack solution regions must be mappings")
        if (
            record is not None
            and record.get("material_id", semantic_id) != entity.material_id
        ):
            raise ValueError(
                f"solution region {semantic_id!r} material contradicts normalized geometry"
            )
        generated_auto = bool(entity.metadata.get("is_auto_vacuum_region"))
        geometry = (
            entity.geometry
            if record is None or generated_auto
            else record.get("geometry", record)
        )
        geometry_z_range(geometry, semantic_id)
        normalized_regions.append(
            {
                "semantic_id": semantic_id,
                "material_id": entity.material_id,
                "material_kind": entity.material_kind,
                "representation": "solution_domain",
                "geometry": _plain_geometry_source(geometry),
                "metadata": _plain_geometry_source(entity.metadata),
            }
        )
    missing_regions = set(solution_regions) - {
        item["semantic_id"] for item in normalized_regions
    }
    if missing_regions:
        raise ValueError(
            f"prepared-stack solution regions lack normalized geometry: {sorted(missing_regions)!r}"
        )
    if any(item["semantic_id"] == "Region" for item in normalized_regions):
        raise ValueError(
            "source solution-domain id 'Region' is reserved for native EPR CAD"
        )
    return normalized_regions


def _native_region(
    normalized_regions: list[dict[str, Any]], metadata: Mapping[str, Any]
) -> dict[str, Any]:
    """Describe the single native HFSS `Region` that encloses the planner's vacuum.

    The planner splits the surrounding vacuum into auto-vacuum components. They
    must agree on material, group, padding and envelope, because native HFSS
    builds one `Region` from the whole envelope.
    """
    auto_vacuum_regions = _auto_vacuum_regions(normalized_regions)
    vacuum_material_ids = {item["material_id"] for item in auto_vacuum_regions}
    vacuum_groups = {
        item["metadata"].get("auto_vacuum_group_id") for item in auto_vacuum_regions
    }
    padding_records = {
        canonical_sha256(item["metadata"].get("vacuum_region_padding_um"))
        for item in auto_vacuum_regions
    }
    if (
        len(vacuum_material_ids) != 1
        or len(vacuum_groups) != 1
        or None in vacuum_groups
        or len(padding_records) != 1
    ):
        raise ValueError("EPR auto vacuum components disagree on Region identity")
    envelope_trace = metadata["physical_representation"]["enclosures"][
        auto_vacuum_regions[0]["semantic_id"]
    ]
    padding_um = _six_face_padding_um(envelope_trace["effective_padding_um"])
    loops = {
        canonical_sha256(item["metadata"].get("auto_vacuum_envelope_outer_loop"))
        for item in auto_vacuum_regions
    }
    if len(loops) != 1:
        raise ValueError("EPR auto vacuum components disagree on envelope")
    envelope_loop = auto_vacuum_regions[0]["metadata"].get(
        "auto_vacuum_envelope_outer_loop"
    )
    if not isinstance(envelope_loop, Sequence) or len(envelope_loop) < 3:
        raise ValueError("EPR Region requires an auto vacuum envelope loop")
    # Complement slabs omit dielectric and metal occupancy. Native Region
    # encloses those bodies too, so its bounds belong to the complete envelope.
    envelope_bounds = auto_vacuum_regions[0]["metadata"][
        "auto_vacuum_envelope_bounds_um"
    ]
    return {
        "method": "single_region_absolute_offset.v1",
        "outer_boundary_policy": "hfss_default.v1",
        "name": "Region",
        "material_id": next(iter(vacuum_material_ids)),
        "logical_vacuum_ids": sorted(
            item["semantic_id"] for item in auto_vacuum_regions
        ),
        "padding_um": padding_um,
        **(
            {"source_effective_envelope": _plain_geometry_source(envelope_trace)}
            if isinstance(envelope_trace, Mapping)
            else {}
        ),
        "envelope_outer_loop_um": _plain_geometry_source(envelope_loop),
        "z_range_um": [
            envelope_bounds["z_min_um"],
            envelope_bounds["z_max_um"],
        ],
    }


def _auto_vacuum_regions(normalized_regions: list[dict[str, Any]]):
    vacuum_regions = [
        item for item in normalized_regions if item["material_kind"] == "vacuum"
    ]
    auto_vacuum_regions = [
        item for item in vacuum_regions if item["metadata"].get("is_auto_vacuum_region")
    ]
    if not auto_vacuum_regions:
        raise ValueError("EPR Region requires planner-owned auto vacuum components")
    return auto_vacuum_regions


def _six_face_padding_um(padding: Any) -> list[float]:
    """Return the X/Y/Z plus and minus padding, which must all be given and valid."""
    if not isinstance(padding, Mapping) or set(padding) != set(_PADDING_FACE_KEYS):
        raise ValueError("EPR Region requires exact six-face source padding")
    padding_um = [float(padding[key]) for key in _PADDING_FACE_KEYS]
    if any(not math.isfinite(value) or value < 0.0 for value in padding_um):
        raise ValueError("EPR Region padding must be finite and nonnegative")
    return padding_um


def _junction_region_records(build_input) -> list[dict[str, Any]]:
    return [
        {
            "port_sheet_id": item.port_sheet_id,
            "source_polygon_id": item.source_polygon_id,
            "source_layer": item.source_layer,
            "exterior": [list(point) for point in item.exterior],
            "holes": [[list(point) for point in ring] for ring in item.holes],
            "host_semantic_ids": list(
                dict.fromkeys(overlap.host_semantic_id for overlap in item.overlaps)
            ),
            "metadata": _plain_geometry_source(item.metadata),
        }
        for item in build_input.port_sheet_regions
    ]


def _polygon_records(build_input) -> list[dict[str, Any]]:
    return [
        {
            "polygon_id": item.polygon_id,
            "layer": item.layer,
            "exterior": [list(point) for point in item.exterior],
            "holes": [[list(point) for point in ring] for ring in item.holes],
            "object_name": item.object_name,
            "net_name": item.net_name,
            "port_name": item.port_name,
        }
        for item in build_input.polygons
    ]
