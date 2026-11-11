"""Load public GDS/component inputs and run the ordered source-normalization pipeline."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, cast

from scgsim.diagnostics import diagnostic_stage, progress
from scgsim.geometry.models.common import PathInput
from scgsim.geometry.models.input import (
    BoundaryCurveChainSpec,
    GdsBoundaryReconstructionSpec,
    GeometryBuildInput,
    LayoutPolygonSpec,
    SemanticEntitySpec,
)
from scgsim.geometry.models.regions import PortSheetRegionRecord
from scgsim.geometry.primitives.grid import geometry_grid_scope, source_recipe_map
from scgsim.geometry.primitives.grid_cache import with_grid_work_cache
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
    with_drawn_rings,
)
from scgsim.geometry.source.grid import apply_geometry_grid, source_grid_context
from scgsim.geometry.source.intents import route_a_sheet_interfaces
from scgsim.geometry.source.occurrence import (
    failed_source_recipe,
    read_polygons_by_layer,
    run_source_recipe,
    source_recipe_job,
    with_occurrence_extraction_cache,
)
from scgsim.geometry.source.occurrence_validation import validate_occurrence_sources
from scgsim.geometry.source.ports import port_sheet_regions_from_stack_metadata
from scgsim.geometry.source.stack import build_component_stack

if TYPE_CHECKING:
    from gdsfactory import Component
    from gdsfactory.technology import LayerStack
    from gdstk import Cell

    from scgsim.geometry.models.grid import GeometryGridContext


@with_grid_work_cache
@diagnostic_stage("geometry.source.adapter")
@with_occurrence_extraction_cache
def build_gds_stack_geometry_input(
    *,
    gds_file: PathInput,
    stack_file: PathInput,
    top_cell_name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    grid_size_um: float | None = None,
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
      `material_id`, optional `priority` (higher wins), optional
      `geometry_kind`, optional `part_role`, optional `net_id`, optional `polygon_ids`, optional `labels`,
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

    import gdstk  # noqa: PLC0415 - optional native loader runs before path validation.

    stack = _read_stack_records(Path(gds_file), stack_file, metadata)
    effective_top_cell_name = top_cell_name or stack.metadata.get("top_cell_name")
    # read_gds parses Cell objects; Library's general type also permits RawCells
    # inserted by callers, which this freshly loaded source never contains.
    library = cast("LoadedLibrary", gdstk.read_gds(str(stack.gds_path)))
    source_dbu_um = float(Decimal(str(library.precision)) * Decimal("1000000"))
    context = source_grid_context(source_dbu_um, grid_size_um)
    with geometry_grid_scope(context):
        source = _read_selected_source(
            stack, library, source_dbu_um, context, effective_top_cell_name
        )
        effective_metadata = validate_occurrence_sources(source, metadata)
        return _SourceLowering(source, effective_metadata).build()


def build_gdsfactory_geometry_input(  # noqa: PLR0913 - preserve the public adapter contract.
    *,
    component: Component,
    layer_stack: LayerStack,
    materials: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    top_cell_name: str | None = None,
    work_dir: PathInput | None = None,
    padding_um: float = 100.0,
    grid_size_um: float | None = None,
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
    adapter = _ComponentAdapter(
        component,
        layer_stack,
        materials,
        metadata,
        top_cell_name,
        padding_um,
        grid_size_um,
    )
    if work_dir is None:
        with TemporaryDirectory() as tmp:
            return adapter.build_in(Path(tmp))
    return adapter.build_in(Path(work_dir))


def _read_stack_records(
    gds_path: Path, stack_file: PathInput, metadata: Mapping[str, Any] | None
) -> StackRecords:
    if not gds_path.is_file():
        raise FileNotFoundError(gds_path)
    stack_mapping, stack_path = _load_stack_mapping(stack_file)
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
    return StackRecords(
        gds_path, stack_path, raw_layers, solution_regions, materials, stack_metadata
    )


def _read_selected_source(
    stack: StackRecords,
    library: LoadedLibrary,
    source_dbu_um: float,
    context: GeometryGridContext,
    effective_top_cell_name: Any,
) -> AdapterSource:
    cell = _select_gds_cell(
        library, str(effective_top_cell_name) if effective_top_cell_name else None
    )
    cell_bounds = _cell_bounds_um(cell)
    polygons_by_layer = read_polygons_by_layer(cell)
    return AdapterSource(
        stack, library, cell, cell_bounds, polygons_by_layer, source_dbu_um, context
    )


@dataclass
class _SourceLowering:
    source: AdapterSource
    effective_metadata: Mapping[str, Any] | None
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
        return with_drawn_rings(gridded, lambda _polygon: drawn)

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
            "gds_file": str(self.source.stack.gds_path),
            "stack_file": str(self.source.stack.stack_path),
            "selected_cell_name": self.source.cell.name,
            "cell_bounds_um": self.source.cell_bounds_um,
            "source_dbu_um": self.source.source_dbu_um,
        }
        metadata["interface_intents_2d"] = route_a_sheet_interfaces(
            self.entities, self.polygons
        )
        return metadata

    def geometry_input(
        self, metadata: Mapping[str, Any], ports: tuple[PortSheetRegionRecord, ...]
    ) -> GeometryBuildInput:
        # boundary_binding's reconstruction flag selects the actual record class;
        # these casts retain its existing validation without duplicate checks.
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


@dataclass(frozen=True)
class _ComponentAdapter:
    component: Component
    layer_stack: LayerStack
    materials: Mapping[str, Any] | None
    metadata: Mapping[str, Any] | None
    top_cell_name: str | None
    padding_um: float
    grid_size_um: float | None

    def build_in(self, work_dir: Path) -> GeometryBuildInput:
        work_dir.mkdir(parents=True, exist_ok=True)
        component_name = str(
            getattr(self.component, "name", "component") or "component"
        )
        safe_name = component_name.replace("/", "_").replace(":", "_")
        gds_path = work_dir / f"{safe_name}.gds"
        stack_path = work_dir / f"{safe_name}.stack.json"
        self.write_gds(gds_path)
        stack_mapping = self.stack_mapping(gds_path)
        stack_path.write_text(
            json.dumps(stack_mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return build_gds_stack_geometry_input(
            gds_file=gds_path,
            stack_file=stack_path,
            top_cell_name=self.top_cell_name or component_name,
            grid_size_um=self.grid_size_um,
            metadata={
                **dict(self.metadata or {}),
                "adapter": "gdsfactory",
                "component_name": component_name,
                "generated_gds_file": str(gds_path),
                "generated_stack_file": str(stack_path),
            },
        )

    def write_gds(self, gds_path: Path) -> None:
        write_gds = getattr(self.component, "write_gds", None)
        if write_gds is None:
            raise TypeError("component must provide write_gds(path)")
        try:
            write_gds(gds_path)
        except TypeError:
            write_gds(str(gds_path))

    def stack_mapping(self, source_gds: Path) -> dict[str, Any]:
        # Preserve source material validation after genuine GDS emission, including
        # the existing None/malformed-input failure path owned by the stack builder.
        material_records = cast("Mapping[str, Mapping[str, Any]]", self.materials)
        stack = build_component_stack(
            component=self.component,
            layer_stack=self.layer_stack,
            material_records=material_records,
            coupon_padding_um=self.padding_um,
        )
        stack["metadata"].update(
            {
                "schema": "semantic_geometry_stack_v1",
                "units": "um",
                "source": str(source_gds),
                "adapter": "gdsfactory",
                "material_names": sorted(str(key) for key in (self.materials or {})),
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


# Palace preparation still imports this source-owned helper at its original path.
