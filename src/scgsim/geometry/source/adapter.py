"""Lower a GDS file plus a stack JSON into canonical `GeometryBuildInput` records.

Owns: reading the stack file and GDS, turning stack layer records into semantic
entities and layout polygons, and splitting multi-polygon layers into entities.
Polygon holes and source IDs are preserved.

Does not own: solver ownership of any entity (the compiler decides that), the
paired-plan consistency checks (`plan_source_checks.py`) or port-sheet lowering
(`port_sheets.py`).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from scgsim.geometry.models.common import PathInput
from scgsim.geometry.models.input import (
    GeometryBuildInput,
    LayoutPolygonSpec,
    SemanticEntitySpec,
)
from scgsim.geometry.source._occurrence import (
    collect_polygons_by_layer,
    entity_from_layer_record,
    gds_polygons_for_entity,
    is_record_sequence,
    resolve_material_kind,
)
from scgsim.geometry.source.ground import derived_ground_polygons
from scgsim.geometry.source.plan_source_checks import verify_paired_plan_sources
from scgsim.geometry.source.port_sheets import port_sheet_regions_from_stack_metadata
from scgsim.geometry.source.stack import build_component_stack

if TYPE_CHECKING:
    from gdsfactory import Component
    from gdsfactory.technology import LayerStack

_COMPONENT_SEMANTICS_PLAN_SCHEMA = 2


def build_gds_stack_geometry_input(
    *,
    gds_file: PathInput,
    stack_file: PathInput,
    top_cell_name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> GeometryBuildInput:
    """Level 0 adapter: build GeometryBuildInput from GDS plus stack JSON.

    This is the canonical frontend contract for v1. Tool-specific adapters
    such as GDSFactory, gsim, or direct KLayout support should
    lower into this same semantic stack shape instead of inventing their own
    geometry semantics.

    `gds_file` is the layout source path. The adapter must load this file,
    select `top_cell_name` when provided, apply an explicit hierarchy/flattening
    policy, and convert layer/datatype shapes to `LayoutPolygonSpec` records
    without leaking raw KLayout objects.

    `top_cell_name` optionally selects the GDS cell to adapt. If it is omitted
    and a later KLayout-backed extraction sees multiple plausible top cells, it
    should fail fast rather than guessing silently.

    `stack_file` is a path to a project stackup JSON file. It is the reviewed
    semantic contract for layer/datatype, material, z-range, route
    representation, and interface-recognition metadata. Unsupported suffixes
    fail clearly.

    Minimal JSON schema for this first slice:

    - `layers`: sequence of layer records. Each record maps one GDS
      `layer`/`datatype` pair to a semantic entity using `semantic_id`, `role`,
      `material_id`, optional `priority`, optional `geometry_kind`, optional
      `part_role`, optional `net_id`, optional `polygon_ids`, optional `labels`,
      optional `attached_face_metal_semantic_id`, optional
      `route_representations`, optional `host_void_semantic_id`, and either
      `z_um` plus `thickness_um` or a `geometry` mapping.
    - `solution_regions`: mapping from solution-region semantic id, such as
      `AIR` or `substrate`, to metadata. Each region may define `material_id`,
      `priority`, `geometry_kind`, and `geometry`; missing values default to
      the semantic id, `0`, `domain`, and the metadata mapping itself.
    - `metadata.port_sheet_source_layers`: optional sequence of Palace
      lumped-port sheet source layers. These become `PortSheetRegionRecord`s,
      not backend-live `SurfacePlanRecord`s.
    - `metadata`: optional mapping copied into `GeometryBuildInput.metadata`.

    `metadata` is copied to `GeometryBuildInput.metadata` for adapter version,
    source GDS/stack file paths, selected top cell, unit convention, KLayout
    database unit, flattening policy, and stack-file dialect/provenance.

    The implementation must return a fully frontend-normalized
    `GeometryBuildInput`: `polygons` from KLayout geometry, `entities` from the
    resolved stack-file material semantics, and `solution_regions` from
    solver-domain definitions such as AIR, substrate, dielectric, or enclosure
    boxes. It must not emit solver config or assume Ansys physical names are the
    final semantic ids.
    """
    import gdstk

    gds_path = Path(gds_file)
    if not gds_path.is_file():
        raise FileNotFoundError(gds_path)

    stack_mapping, stack_path = _load_stack_mapping(stack_file)
    raw_layers, solution_regions, materials, stack_metadata = _stack_sections(
        stack_mapping
    )
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping when provided")

    effective_top_cell_name = top_cell_name or stack_metadata.get("top_cell_name")
    library = gdstk.read_gds(str(gds_path))
    cell = _select_gds_cell(
        library,
        str(effective_top_cell_name) if effective_top_cell_name else None,
    )
    cell_bounds = _cell_bounds_um(cell)
    polygons_by_layer = collect_polygons_by_layer(cell)
    if (
        stack_metadata.get("component_semantics_schema_version")
        == _COMPONENT_SEMANTICS_PLAN_SCHEMA
    ):
        plan_metadata = verify_paired_plan_sources(
            gds_path=gds_path,
            library=library,
            top_cell=cell,
            raw_layers=raw_layers,
            stack_metadata=stack_metadata,
            solution_regions=solution_regions,
            materials=materials,
        )
        if plan_metadata is not None:
            # Preserved quirk: the original inline check overwrote `metadata`
            # with the last derived ground plane's metadata (see
            # `verify_paired_plan_sources`).
            metadata = plan_metadata

    entities, polygons = _lower_stack_entities(
        raw_layers,
        solution_regions=solution_regions,
        materials=materials,
        polygons_by_layer=polygons_by_layer,
        cell_bounds=cell_bounds,
    )

    combined_metadata = {
        **dict(stack_metadata),
        **dict(metadata or {}),
        "adapter": "gds_stack",
        "gds_file": str(gds_path),
        "stack_file": str(stack_path),
        "selected_cell_name": cell.name,
        "cell_bounds_um": cell_bounds,
        "source_dbu_um": float(library.precision) * 1e6,
    }
    from scgsim.geometry.source.intents import route_a_sheet_interfaces

    combined_metadata["interface_intents_2d"] = route_a_sheet_interfaces(
        entities, polygons
    )
    port_sheet_regions = port_sheet_regions_from_stack_metadata(
        stack_metadata,
        polygons_by_layer=polygons_by_layer,
        host_entities=entities,
        host_polygons=polygons,
    )

    from scgsim.geometry.source.curves import boundary_binding

    return GeometryBuildInput(
        polygons=tuple(polygons),
        entities=tuple(entities),
        port_sheet_regions=port_sheet_regions,
        solution_regions=dict(solution_regions),
        metadata=combined_metadata,
        boundary_curves=tuple(
            boundary_binding(value)
            for value in stack_metadata.get("boundary_curves", ())
        ),
        boundary_reconstruction=tuple(
            boundary_binding(value, reconstruction=True)
            for value in stack_metadata.get("boundary_reconstruction", ())
        ),
    )


def _stack_sections(
    stack_mapping: Mapping[str, Any],
) -> tuple[Any, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Return the stack's layers, solution regions, materials and metadata."""
    raw_layers = stack_mapping.get("layers")
    if not is_record_sequence(raw_layers):
        raise TypeError("stack_file must define sequence 'layers'")

    solution_regions = stack_mapping.get("solution_regions")
    if not isinstance(solution_regions, Mapping):
        raise TypeError("stack_file must define mapping 'solution_regions'")
    materials = stack_mapping.get("materials")
    if not isinstance(materials, Mapping):
        raise TypeError("stack_file must define mapping 'materials'")

    stack_metadata = stack_mapping.get("metadata", {})
    if not isinstance(stack_metadata, Mapping):
        raise TypeError("stack_file 'metadata' must be a mapping when provided")
    return raw_layers, solution_regions, materials, stack_metadata


def _lower_stack_entities(
    raw_layers: Any,
    *,
    solution_regions: Mapping[str, Any],
    materials: Mapping[str, Any],
    polygons_by_layer: Mapping[tuple[int, int], tuple[Any, ...]],
    cell_bounds: Mapping[str, float],
) -> tuple[list[SemanticEntitySpec], list[LayoutPolygonSpec]]:
    """Solution-region entities first, then layer entities in stack order."""
    polygons: list[LayoutPolygonSpec] = []
    entities: list[SemanticEntitySpec] = [
        solution_region_entity_from_record(
            semantic_id,
            record,
            materials=materials,
            cell_bounds_um=cell_bounds,
        )
        for semantic_id, record in solution_regions.items()
    ]
    domain_bounds_by_semantic_id = {
        entity.semantic_id: entity.geometry["domain_bounds_um"]
        for entity in entities
        if isinstance(entity.geometry.get("domain_bounds_um"), Mapping)
    }

    for record in raw_layers:
        for entity, entity_polygons in _entities_and_polygons_from_layer_record(
            record,
            materials=materials,
            polygons_by_layer=polygons_by_layer,
            cell_bounds_um=cell_bounds,
            domain_bounds_by_semantic_id=domain_bounds_by_semantic_id,
        ):
            polygons.extend(entity_polygons)
            entities.append(entity)
    return entities, polygons


def build_gdsfactory_geometry_input(  # noqa: PLR0913 - public keyword-only API; one argument per input
    *,
    component: Component,
    layer_stack: LayerStack,
    materials: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    top_cell_name: str | None = None,
    work_dir: PathInput | None = None,
    padding_um: float = 100.0,
) -> GeometryBuildInput:
    """Build GeometryBuildInput from GDSFactory layout and technology objects.

    This adapter intentionally lowers GDSFactory/gsim objects into the reviewed
    Level-0 contract: a GDS file plus stack JSON, then delegates to
    `build_gds_stack_geometry_input`. The adapter does not invent a second
    semantic language.

    `layer_stack.layers[*].info` supplies PDK-owned material, fabrication,
    host-volume, and 3D-integration semantics. `component.info` supplies
    component-owned conductor identities, nets, and topology selectors.
    Missing or mixed authority fails before geometry lowering.

    `work_dir` makes the generated GDS and stack JSON reviewable. Without it,
    temporary files are used only long enough to call the Level-0 adapter.
    """
    if materials is not None and not isinstance(materials, Mapping):
        raise TypeError("materials must be a mapping when provided")

    if work_dir is None:
        with TemporaryDirectory() as tmp:
            return _build_gdsfactory_geometry_input_from_dir(
                component=component,
                layer_stack=layer_stack,
                materials=materials,
                metadata=metadata,
                top_cell_name=top_cell_name,
                work_dir=Path(tmp),
                padding_um=padding_um,
            )
    return _build_gdsfactory_geometry_input_from_dir(
        component=component,
        layer_stack=layer_stack,
        materials=materials,
        metadata=metadata,
        top_cell_name=top_cell_name,
        work_dir=Path(work_dir),
        padding_um=padding_um,
    )


def _build_gdsfactory_geometry_input_from_dir(  # noqa: PLR0913 - forwards the public keyword-only API plus work_dir
    *,
    component: Any,
    layer_stack: Any,
    materials: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None,
    top_cell_name: str | None,
    work_dir: Path,
    padding_um: float,
) -> GeometryBuildInput:
    work_dir.mkdir(parents=True, exist_ok=True)
    component_name = str(getattr(component, "name", "component") or "component")
    safe_name = component_name.replace("/", "_").replace(":", "_")
    gds_path = work_dir / f"{safe_name}.gds"
    stack_path = work_dir / f"{safe_name}.stack.json"

    write_gds = getattr(component, "write_gds", None)
    if write_gds is None:
        raise TypeError("component must provide write_gds(path)")
    try:
        write_gds(gds_path)
    except TypeError:
        write_gds(str(gds_path))

    stack_mapping = _semantic_stack_mapping_from_layer_stack(
        component,
        layer_stack,
        materials=materials,
        source_gds=gds_path,
        padding_um=padding_um,
    )
    stack_path.write_text(
        json.dumps(stack_mapping, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return build_gds_stack_geometry_input(
        gds_file=gds_path,
        stack_file=stack_path,
        top_cell_name=top_cell_name or component_name,
        metadata={
            **dict(metadata or {}),
            "adapter": "gdsfactory",
            "component_name": component_name,
            "generated_gds_file": str(gds_path),
            "generated_stack_file": str(stack_path),
        },
    )


def _semantic_stack_mapping_from_layer_stack(
    component: Any,
    layer_stack: Any,
    *,
    materials: Mapping[str, Any] | None,
    source_gds: Path,
    padding_um: float,
) -> dict[str, Any]:
    stack = build_component_stack(
        component=component,
        layer_stack=layer_stack,
        material_records=materials,  # type: ignore[arg-type]
        coupon_padding_um=padding_um,
    )
    stack["metadata"].update(
        {
            "schema": "semantic_geometry_stack_v1",
            "units": "um",
            "source": str(source_gds),
            "adapter": "gdsfactory",
            "material_names": sorted(str(key) for key in (materials or {})),
        }
    )
    return stack


def _load_stack_mapping(
    stack_file: PathInput,
) -> tuple[Mapping[str, Any], Path]:
    stack_path = Path(stack_file)
    if not stack_path.is_file():
        raise FileNotFoundError(stack_path)
    if stack_path.suffix.lower() != ".json":
        raise ValueError(
            "unsupported stack_file suffix "
            f"{stack_path.suffix!r}; only JSON is supported for now"
        )

    data = json.loads(stack_path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise TypeError("JSON stack_file root must be a mapping")
    return data, stack_path


def _select_gds_cell(library: Any, top_cell_name: str | None) -> Any:
    if top_cell_name is not None:
        for cell in library.cells:
            if cell.name == top_cell_name:
                return cell
        raise ValueError(f"GDS top_cell_name not found: {top_cell_name!r}")

    candidates = [
        cell
        for cell in library.cells
        if cell.name != "$$$CONTEXT_INFO$$$"
        and cell.get_polygons(apply_repetitions=True)
    ]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        names = ", ".join(sorted(cell.name for cell in candidates))
        raise ValueError(f"top_cell_name is required; candidates: {names}")

    top_cells = [
        cell
        for cell in library.top_level()
        if cell.get_polygons(apply_repetitions=True)
    ]
    if len(top_cells) == 1:
        return top_cells[0]
    names = ", ".join(sorted(cell.name for cell in top_cells))
    raise ValueError(f"top_cell_name is required; candidates: {names}")


def _cell_bounds_um(cell: Any) -> dict[str, float]:
    bbox = cell.bounding_box()
    if bbox is None:
        raise ValueError(f"GDS cell {cell.name!r} has no bounding box")
    (x_min, y_min), (x_max, y_max) = bbox
    return {
        "x_min_um": float(x_min),
        "y_min_um": float(y_min),
        "x_max_um": float(x_max),
        "y_max_um": float(y_max),
    }


def solution_region_entity_from_record(
    semantic_id: Any,
    record: Any,
    *,
    materials: Mapping[str, Any],
    cell_bounds_um: Mapping[str, float],
) -> SemanticEntitySpec:
    """Build a solution-region entity from a stack record; default bounds are padded."""
    if not isinstance(semantic_id, str):
        raise TypeError("solution region ids must be strings")
    if not isinstance(record, Mapping):
        raise TypeError("stack_file 'solution_regions' values must be mappings")

    geometry = record.get("geometry", record)
    if not isinstance(geometry, Mapping):
        raise TypeError("solution region 'geometry' must be a mapping")

    geometry = dict(geometry)
    padding_um = float(geometry.get("padding_um", 0.0))
    geometry.setdefault(
        "domain_bounds_um",
        {
            "x_min_um": cell_bounds_um["x_min_um"] - padding_um,
            "y_min_um": cell_bounds_um["y_min_um"] - padding_um,
            "x_max_um": cell_bounds_um["x_max_um"] + padding_um,
            "y_max_um": cell_bounds_um["y_max_um"] + padding_um,
        },
    )

    return SemanticEntitySpec(
        semantic_id=semantic_id,
        role=record.get("role", "solution_region"),
        material_id=record.get("material_id", semantic_id),
        material_kind=resolve_material_kind(
            record,
            material_id=record.get("material_id", semantic_id),
            materials=materials,
            context=f"solution region {semantic_id!r}",
        ),
        priority=record.get("priority", 0),
        geometry_kind=record.get("geometry_kind", "domain"),
        geometry=geometry,
        metadata=record.get("metadata", {}),
    )


def _entities_and_polygons_from_layer_record(
    record: Any,
    *,
    materials: Mapping[str, Any],
    polygons_by_layer: Mapping[tuple[int, int], tuple[Any, ...]],
    cell_bounds_um: Mapping[str, float],
    domain_bounds_by_semantic_id: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[SemanticEntitySpec, tuple[LayoutPolygonSpec, ...]], ...]:
    entity = entity_from_layer_record(record, materials=materials)
    geometry_source = str(entity.geometry.get("geometry_source", "gds_polygon"))
    if geometry_source == "die_face_minus_ground_mask":
        occurrence_polygons = entity.geometry.get("source_occurrence_polygons_um")
        if occurrence_polygons is not None:
            entity_polygons = tuple(
                LayoutPolygonSpec(
                    polygon_id=f"{entity.semantic_id}__P{index:04d}",
                    layer=f"{record['layer']}/{record['datatype']}",
                    exterior=region["exterior"],
                    holes=region["holes"],
                    object_name=entity.semantic_id,
                    net_name=entity.net_id,
                    metadata={"source": "die_face_minus_ground_mask"},
                )
                for index, region in enumerate(occurrence_polygons)
            )
        else:
            entity_polygons = derived_ground_polygons(
                entity,
                polygons_by_layer=polygons_by_layer,
                cell_bounds_um=cell_bounds_um,
                domain_bounds_by_semantic_id=domain_bounds_by_semantic_id,
            )
        if len(entity_polygons) > 1:
            return tuple(
                _split_polygon_entity(entity, polygon, index)
                for index, polygon in enumerate(entity_polygons)
            )
        return (
            (_entity_with_polygon_geometry(entity, entity_polygons), entity_polygons),
        )
    elif geometry_source == "gds_polygon":
        entity_polygons = gds_polygons_for_entity(
            entity,
            polygons_by_layer=polygons_by_layer,
        )
    else:
        entity_polygons = ()

    if entity.geometry.get("split_polygons_as_entities") or (
        entity.geometry.get("route_ab_fused_selector_mode", False)
        and entity.geometry.get("selector_point_um") is None
    ):
        return tuple(
            _split_polygon_entity(entity, polygon, index)
            for index, polygon in enumerate(entity_polygons)
        )
    entity = _entity_with_polygon_geometry(entity, entity_polygons)
    return ((entity, entity_polygons),)


def _split_polygon_entity(
    entity: SemanticEntitySpec,
    polygon: LayoutPolygonSpec,
    index: int,
) -> tuple[SemanticEntitySpec, tuple[LayoutPolygonSpec, ...]]:
    semantic_id = f"{entity.semantic_id}_{index:04d}"
    split_polygon = LayoutPolygonSpec(
        polygon_id=f"{semantic_id}__P0000",
        layer=polygon.layer,
        exterior=polygon.exterior,
        holes=polygon.holes,
        object_name=semantic_id,
        net_name=polygon.net_name,
        port_name=polygon.port_name,
        metadata={
            **dict(polygon.metadata),
            "source_semantic_id": entity.semantic_id,
            "split_polygon_index": index,
        },
    )
    geometry = {
        **dict(entity.geometry),
        "outer_loop": split_polygon.exterior,
        "hole_loops": split_polygon.holes,
    }
    # Selection lists are consumed by this adapter; duplicating them onto each
    # split entity would turn generated-fill metadata quadratic.
    geometry.pop("include_selector_points_um", None)
    geometry.pop("exclude_selector_points_um", None)
    metadata = {
        **dict(entity.metadata),
        **(
            {"source_semantic_id": entity.semantic_id}
            if "source_occurrence_path" in entity.metadata
            else {"semantic_group_id": entity.semantic_id}
        ),
        "split_polygon_index": index,
    }
    return (
        SemanticEntitySpec(
            semantic_id=semantic_id,
            role=entity.role,
            material_id=entity.material_id,
            material_kind=entity.material_kind,
            priority=entity.priority,
            geometry_kind=entity.geometry_kind,
            part_role=entity.part_role,
            attached_face_metal_semantic_id=entity.attached_face_metal_semantic_id,
            net_id=entity.net_id,
            polygon_ids=(split_polygon.polygon_id,),
            labels=entity.labels,
            host_void_semantic_id=entity.host_void_semantic_id,
            requires_construction_body=entity.requires_construction_body,
            route_representations=entity.route_representations,
            geometry=geometry,
            metadata=metadata,
        ),
        (split_polygon,),
    )


def _entity_with_polygon_geometry(
    entity: SemanticEntitySpec,
    polygons: tuple[LayoutPolygonSpec, ...],
) -> SemanticEntitySpec:
    if not polygons:
        return entity
    geometry = dict(entity.geometry)
    geometry.pop("include_selector_points_um", None)
    geometry.pop("exclude_selector_points_um", None)
    if len(polygons) == 1:
        geometry.setdefault("outer_loop", polygons[0].exterior)
        geometry.setdefault("hole_loops", polygons[0].holes)
    polygon_ids = tuple(polygon.polygon_id for polygon in polygons)
    return SemanticEntitySpec(
        semantic_id=entity.semantic_id,
        role=entity.role,
        material_id=entity.material_id,
        material_kind=entity.material_kind,
        priority=entity.priority,
        geometry_kind=entity.geometry_kind,
        part_role=entity.part_role,
        attached_face_metal_semantic_id=entity.attached_face_metal_semantic_id,
        net_id=entity.net_id,
        polygon_ids=polygon_ids,
        labels=entity.labels,
        host_void_semantic_id=entity.host_void_semantic_id,
        requires_construction_body=entity.requires_construction_body,
        route_representations=entity.route_representations,
        geometry=geometry,
        metadata=entity.metadata,
    )
