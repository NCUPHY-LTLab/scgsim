"""Load a GDS file with its stack JSON and run the ordered source-normalization pipeline."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from scgsim.diagnostics import diagnostic_stage, progress
from scgsim.geometry.grid.grid import geometry_grid_scope, source_recipe_map
from scgsim.geometry.grid.grid_cache import with_grid_work_cache
from scgsim.geometry.models.common import PathInput
from scgsim.geometry.models.grid import GeometryGridContext
from scgsim.geometry.models.input import (
    BoundaryCurveChainSpec,
    GdsBoundaryReconstructionSpec,
    GeometryBuildInput,
    LayoutPolygonSpec,
    SemanticEntitySpec,
)
from scgsim.geometry.models.regions import PortSheetRegionRecord
from scgsim.geometry.source.adapter_records import (
    AdapterSource,
    LoadedLibrary,
    StackRecords,
    lower_layer_record,
    record_sequence,
    solution_region_entity,
)
from scgsim.geometry.source.curves import (
    DrawnWalks,
    boundary_binding,
    gds_walks,
    refuse_reconstructions_without_circle,
    with_drawn_rings,
)
from scgsim.geometry.source.grid import apply_geometry_grid
from scgsim.geometry.source.intents import route_a_sheet_interfaces
from scgsim.geometry.source.occurrence import (
    failed_source_recipe,
    read_polygons_by_layer,
    run_source_recipe,
    source_recipe_job,
    with_occurrence_extraction_cache,
)
from scgsim.geometry.source.ports import port_sheet_regions_from_stack_metadata

if TYPE_CHECKING:
    from gdstk import Cell


@with_grid_work_cache
@diagnostic_stage("geometry.source.adapter")
@with_occurrence_extraction_cache
def build_gds_stack_geometry_input(
    *,
    gds_file: PathInput,
    stack_file: PathInput,
    top_cell_name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> GeometryBuildInput:
    """Build a ``GeometryBuildInput`` from a GDS file and a stack JSON file.

    `gds_file` is the layout. It is read with gdstk, and its database unit is the
    model grid.

    `top_cell_name` selects the GDS cell to adapt (the stack's own
    `top_cell_name` metadata applies when it is omitted). Without either, the
    one cell that holds polygons is used; when several cells do, the call
    refuses and names them.

    `stack_file` is a path to a project stackup JSON file. It is the reviewed
    semantic contract for layer/datatype, material, z-range, route
    representation, and interface-recognition metadata. Unsupported suffixes
    fail clearly. A stack that `GeometryPlan` wrote (a snapshot's `stack`)
    is read only with the GDS it was written for (the snapshot's `gds_bytes`);
    another GDS is refused, naming both digests.

    The stack JSON holds:

    - `layers`: sequence of layer records. Each record maps one GDS
      `layer`/`datatype` pair to a semantic entity using `semantic_id`, `role`,
      `material_id`, optional `priority` (higher wins), optional
      `geometry_kind`, optional `part_role`, optional `net_id`, optional `polygon_ids`, optional `labels`,
      optional `attached_face_metal_semantic_id`, optional
      `route_representations`, optional `host_void_semantic_id`, and either
      `z_um` plus `thickness_um` or a `geometry` mapping.
    - `solution_regions`: mapping from solution-region semantic id, such as
      `AIR` or `substrate`, to metadata. Each region may define `material_id`,
      `priority`, `geometry_kind`, and `geometry`; missing values default to
      the semantic id, `0`, `domain`, and the metadata mapping itself.
    - `materials`: mapping from material id to its record.
    - `metadata.port_sheet_source_layers`: optional sequence of Palace
      lumped-port sheet source layers. These become `PortSheetRegionRecord`s,
      not backend-live `SurfacePlanRecord`s.
    - `metadata`: optional mapping copied into `GeometryBuildInput.metadata`.

    `metadata` is copied to `GeometryBuildInput.metadata` together with the
    source GDS and stack file paths, the selected cell name, its bounds and the
    database unit.

    The result holds `polygons` read from the GDS cell, `entities` from the
    stack's layers and solution regions, and `solution_regions` from the stack's
    solution regions. It carries no solver configuration and takes no Ansys
    physical name as a final semantic id.
    """

    import gdstk  # noqa: PLC0415 - optional native loader runs before path validation.

    gds_path = Path(gds_file)
    if not gds_path.is_file():
        raise FileNotFoundError(gds_path)
    stack_mapping, stack_path = _load_stack_mapping(stack_file)
    stack = stack_records(stack_mapping, metadata)
    check_plan_pair(stack, gds_path.read_bytes())
    # read_gds parses Cell objects; Library's general type also permits RawCells
    # inserted by callers, which this freshly loaded source never contains.
    library = cast("LoadedLibrary", gdstk.read_gds(str(gds_path)))
    return lower_layout(
        stack,
        library,
        top_cell_name or stack.metadata.get("top_cell_name"),
        metadata=metadata,
        source_files={"gds_file": str(gds_path), "stack_file": str(stack_path)},
    )


def stack_records(
    stack_mapping: Mapping[str, Any], metadata: Mapping[str, Any] | None
) -> StackRecords:
    """The stack's containers, each checked to have the shape the lowering reads."""
    raw_layers = stack_mapping.get("layers")
    if not record_sequence(raw_layers):
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
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping when provided")
    return StackRecords(raw_layers, solution_regions, materials, stack_metadata)


def check_plan_pair(stack: StackRecords, gds_bytes: bytes) -> None:
    """Refuse a GeometryPlan stack read with a GDS other than the one it was written for.

    A stack GeometryPlan wrote carries the occurrence, include and port polygons it
    already read from its GDS; lowering it with another GDS would mix two layouts.
    A stack written by hand carries no component-semantics version and is read
    from its GDS alone.
    """
    if "component_semantics_schema_version" not in stack.metadata:
        return
    version = stack.metadata["component_semantics_schema_version"]
    if version != 2:
        raise ValueError(
            f"stack metadata component_semantics_schema_version is {version!r}; "
            "only 2, which GeometryPlan writes, is supported"
        )
    recorded = stack.metadata.get("gds_sha256")
    actual = hashlib.sha256(gds_bytes).hexdigest()
    if recorded != actual:
        raise ValueError(
            "GeometryPlan stack_file was written for the GDS with SHA-256 "
            f"{recorded!r}, but gds_file has SHA-256 {actual!r}. Pass the GDS saved "
            "with this stack (the snapshot's gds_bytes), or prepare the plan again."
        )


def layout_grid(library: LoadedLibrary) -> GeometryGridContext:
    """The model grid of a layout: the database unit its GDS is written in, in micrometres.

    Every XY coordinate the source stage writes lies on this grid; it is read
    here and nowhere else.
    """
    return GeometryGridContext(
        float(Decimal(str(library.precision)) * Decimal("1000000"))
    )


def lower_layout(
    stack: StackRecords,
    library: LoadedLibrary,
    top_cell_name: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
    source_files: Mapping[str, str] | None = None,
) -> GeometryBuildInput:
    """Lower a loaded layout and its stack records onto the layout's grid.

    ``source_files`` names the files the two were read from, for the input's metadata.
    """
    context = layout_grid(library)
    with geometry_grid_scope(context):
        source = _read_selected_source(
            stack, library, context, str(top_cell_name) if top_cell_name else None
        )
        return _SourceLowering(source, metadata, dict(source_files or {})).build()


def _read_selected_source(
    stack: StackRecords,
    library: LoadedLibrary,
    context: GeometryGridContext,
    top_cell_name: str | None,
) -> AdapterSource:
    cell = _select_gds_cell(library, top_cell_name)
    cell_bounds = _cell_bounds_um(cell)
    polygons_by_layer = read_polygons_by_layer(cell)
    return AdapterSource(stack, library, cell, cell_bounds, polygons_by_layer, context)


@dataclass
class _SourceLowering:
    source: AdapterSource
    effective_metadata: Mapping[str, Any] | None
    source_files: Mapping[str, str]
    polygons: list[LayoutPolygonSpec] = field(default_factory=list)
    entities: list[SemanticEntitySpec] = field(default_factory=list)
    domain_bounds: dict[str, Mapping[str, Any]] = field(init=False)

    def build(self) -> GeometryBuildInput:
        self.lower_solution_regions()
        self.lower_layers()
        metadata = self.combined_metadata()
        ports = port_sheet_regions_from_stack_metadata(
            self.source.stack.metadata,
            polygons_by_layer=self.source.polygons_by_layer,
            host_entities=self.entities,
            host_polygons=self.polygons,
        )
        gridded = apply_geometry_grid(
            self.geometry_input(metadata, ports), self.source.grid_context
        )
        # Reconstruction indices name rings as drawn in the top cell's GDS; the grid
        # has since rounded and noded them, so each polygon keeps its drawn rings.
        drawn = DrawnWalks.read(
            gds_walks(self.source.polygons_by_layer),
            self.source.grid_context.grid_size_um,
        )
        built = with_drawn_rings(gridded, lambda _polygon: drawn)
        refuse_reconstructions_without_circle(built, lambda _polygon: drawn)
        return built

    def lower_solution_regions(self) -> None:
        self.entities = [
            solution_region_entity(
                semantic_id,
                record,
                materials=self.source.stack.materials,
                cell_bounds_um=self.source.cell_bounds_um,
            )
            for semantic_id, record in self.source.stack.solution_regions.items()
        ]
        self.domain_bounds = {
            entity.semantic_id: entity.geometry["domain_bounds_um"]
            for entity in self.entities
            if isinstance(entity.geometry.get("domain_bounds_um"), Mapping)
        }

    def lower_layers(self) -> None:
        with diagnostic_stage(
            "geometry.source.lower_layers",
            counts={"layers": len(self.source.stack.layers)},
        ):
            for rows in source_recipe_map(run_source_recipe, self.layer_jobs()):
                for lowered in rows:
                    self.polygons.extend(lowered.polygons)
                    self.entities.append(lowered.entity)

    def layer_jobs(self):
        for diagnostic_index, record in enumerate(self.source.stack.layers):
            progress(diagnostic_index, len(self.source.stack.layers))
            try:
                job = source_recipe_job(
                    lower_layer_record,
                    record,
                    materials=self.source.stack.materials,
                    source_polygons=self.source.polygons_by_layer,
                    cell_bounds_um=self.source.cell_bounds_um,
                    domain_bounds_by_semantic_id=self.domain_bounds,
                )
            except Exception as error:  # noqa: BLE001 - marshal the original worker error in source order.
                yield failed_source_recipe(error)
                return
            yield job

    def combined_metadata(self) -> dict[str, Any]:
        metadata = {
            **dict(self.source.stack.metadata),
            **dict(self.effective_metadata or {}),
            "adapter": "gds_stack",
            **self.source_files,
            "selected_cell_name": self.source.cell.name,
            "cell_bounds_um": self.source.cell_bounds_um,
            "source_dbu_um": self.source.grid_context.grid_size_um,
        }
        metadata["interface_intents_2d"] = route_a_sheet_interfaces(
            self.entities, self.polygons
        )
        return metadata

    def geometry_input(
        self, metadata: Mapping[str, Any], ports: tuple[PortSheetRegionRecord, ...]
    ) -> GeometryBuildInput:
        # boundary_binding's reconstruction flag selects the actual record class;
        # these casts rely on that validation and check nothing twice.
        return GeometryBuildInput(
            polygons=tuple(self.polygons),
            entities=tuple(self.entities),
            port_sheet_regions=ports,
            solution_regions=dict(self.source.stack.solution_regions),
            metadata=metadata,
            boundary_curves=tuple(
                cast("BoundaryCurveChainSpec", boundary_binding(value))
                for value in self.source.stack.metadata.get("boundary_curves", ())
            ),
            boundary_reconstruction=tuple(
                cast(
                    "GdsBoundaryReconstructionSpec",
                    boundary_binding(value, reconstruction=True),
                )
                for value in self.source.stack.metadata.get(
                    "boundary_reconstruction", ()
                )
            ),
        )


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


def _select_gds_cell(library: LoadedLibrary, top_cell_name: str | None) -> Cell:
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


def _cell_bounds_um(cell: Cell) -> dict[str, float]:
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
