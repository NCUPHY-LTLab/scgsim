"""Bind recorded occurrence lineage and ownership to the paired native GDS."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from math import cos, isfinite, sin
from typing import TYPE_CHECKING, Any

from scgsim.geometry.models.input import LayoutPolygonSpec
from scgsim.geometry.primitives.grid import (
    current_grid_size_um,
    named_source,
    polygon_boolean,
    source_place,
    source_recipe_map,
    source_refusal,
)
from scgsim.geometry.source.adapter_records import AdapterSource, record_sequence
from scgsim.geometry.source.ground import layout_polygon_region
from scgsim.geometry.source.occurrence import (
    SourceRecipeJob,
    entity_from_layer_record,
    failed_source_recipe,
    occurrence_ground_polygons_for_entity,
    occurrence_include_polygon,
    occurrence_polygons_for_entity,
    occurrence_port_polygon,
    run_source_recipe,
    source_recipe_job,
)

if TYPE_CHECKING:
    from gdstk import Cell, Reference

# Any is confined to extensible stack records until their existing validators run.


def validate_occurrence_sources(
    source: AdapterSource, metadata: Mapping[str, Any] | None
) -> Mapping[str, Any] | None:
    """Validate schema-two source lineage in source order before lowering records."""
    if source.stack.metadata.get("component_semantics_schema_version") != 2:
        return metadata
    if (
        source.stack.metadata.get("gds_sha256")
        != hashlib.sha256(source.stack.gds_path.read_bytes()).hexdigest()
    ):
        raise ValueError("GeometryPlan GDS identity differs from its paired stack")
    source_cells = {
        source_cell.name: source_cell for source_cell in source.library.cells
    }
    occurrences = source.stack.metadata.get("source_occurrences")
    if not record_sequence(occurrences):
        raise ValueError("GeometryPlan stack needs source occurrences")
    by_path = {row.get("path"): row for row in occurrences if isinstance(row, Mapping)}
    if len(by_path) != len(occurrences):
        raise ValueError("GeometryPlan source occurrence paths must be unique")
    root_path = (
        next(path for path, row in by_path.items() if row.get("parent_path") is None)
        if sum(row.get("parent_path") is None for row in by_path.values()) == 1
        else None
    )
    validate_plan_occurrences(
        by_path,
        top_cell=source.cell,
        source_cells=source_cells,
        source_dbu_um=source.source_dbu_um,
    )
    child_reference_indexes = {
        path: tuple(
            child["reference_index"]
            for child in occurrences
            if child["parent_path"] == path
        )
        for path in by_path
    }
    checks = _OccurrenceValidation(
        source, source_cells, by_path, root_path, child_reference_indexes, metadata
    )
    return checks.validate()


@dataclass
class _OccurrenceValidation:
    source: AdapterSource
    source_cells: Mapping[str, Cell]
    by_path: Mapping[Any, Mapping[str, Any]]
    root_path: Any
    child_reference_indexes: Mapping[Any, tuple[int, ...]]
    effective_metadata: Mapping[str, Any] | None
    contribution_ids: set[tuple[object, str]] = field(default_factory=set)
    physical_planes: set[Any] = field(default_factory=set)
    planes: dict[Any, Mapping[str, Any]] = field(default_factory=dict)
    pending_checks: list[Callable[[], SourceRecipeJob]] = field(default_factory=list)

    def validate(self) -> Mapping[str, Any] | None:
        self.index_planes()
        self.validate_layers()
        self.validate_ports()
        validate_ground_contribution_ownership(self.source.stack.layers)
        return self.effective_metadata

    def index_planes(self) -> None:
        for record in self.source.stack.layers:
            if not isinstance(record, Mapping):
                raise ValueError("GeometryPlan layers must be mappings")
            geometry = record.get("geometry", {})
            if geometry.get("geometry_source") == "die_face_minus_ground_mask":
                plane_metadata = record.get("metadata", {})
                level = plane_metadata.get("logical_layer_id")
                if level in self.physical_planes:
                    raise ValueError(
                        f"multiple ground planes claim physical level {level!r}"
                    )
                self.physical_planes.add(level)
                self.planes[record.get("semantic_id")] = record

    def flush_expected(self) -> None:
        def jobs():
            for factory in self.pending_checks:
                try:
                    job = factory()
                except Exception as error:  # noqa: BLE001 - marshal the original worker error in source order.
                    yield failed_source_recipe(error)
                    return
                yield job

        try:
            source_recipe_map(run_source_recipe, jobs())
        finally:
            self.pending_checks.clear()

    def validate_contribution(
        self, record: Mapping[str, Any], contribution: Any
    ) -> dict[str, Any]:
        if not isinstance(contribution, Mapping) or set(contribution) != {
            "source_occurrence_path",
            "source_local_id",
            "level",
            "layer",
            "selector_point_um",
            "polygon",
        }:
            raise ValueError("GeometryPlan ground contribution has invalid fields")
        if record.get("semantic_id") not in self.planes:
            raise ValueError(
                "GeometryPlan ground contribution targets no root derived plane"
            )
        if contribution["level"] != record["metadata"].get("logical_layer_id"):
            raise ValueError(
                "GeometryPlan ground contribution level differs from target plane"
            )
        if (
            not isinstance(contribution["source_local_id"], str)
            or not contribution["source_local_id"]
            or "/" in contribution["source_local_id"]
        ):
            raise ValueError("GeometryPlan ground contribution local ID is invalid")
        source_path = contribution.get("source_occurrence_path")
        source_occurrence = self.by_path.get(source_path)
        if (
            source_occurrence is None
            or source_occurrence.get("cell_name") not in self.source_cells
        ):
            raise ValueError(
                "GeometryPlan ground contribution has no source occurrence"
            )
        key = (source_path, contribution["source_local_id"])
        if key in self.contribution_ids:
            raise ValueError("GeometryPlan ground contribution local ID is duplicated")
        self.contribution_ids.add(key)
        self.flush_expected()
        expected = occurrence_include_polygon(
            contribution,
            cell=self.source_cells[source_occurrence["cell_name"]],
            transform=source_occurrence["transform"],
            excluded_reference_indexes=self.child_reference_indexes[source_path],
        )
        if contribution.get("polygon") != expected:
            raise ValueError("GeometryPlan ground contribution differs from GDS source")
        return {
            key: contribution[key]
            for key in ("source_occurrence_path", "source_local_id", "level", "layer")
        }

    def validate_layer(self, record: Any) -> None:
        if not isinstance(record, Mapping) or not isinstance(
            record.get("metadata"), Mapping
        ):
            raise ValueError("GeometryPlan layers need source occurrence metadata")
        path = record["metadata"].get("source_occurrence_path")
        occurrence = self.by_path.get(path)
        if occurrence is None or occurrence.get("cell_name") not in self.source_cells:
            raise ValueError("GeometryPlan layer has no recorded source occurrence")
        geometry = record.get("geometry", {})
        source_summaries = []
        for contribution in geometry.get("source_occurrence_includes", ()):
            source_summaries.append(self.validate_contribution(record, contribution))
        if (
            record["metadata"].get("ground_plane_contribution_sources", [])
            != source_summaries
        ):
            raise ValueError(
                "GeometryPlan ground contribution lineage metadata differs"
            )
        if geometry.get("geometry_source", "gds_polygon") != "gds_polygon" and not (
            geometry.get("geometry_source") == "die_face_minus_ground_mask"
            and path != self.root_path
        ):
            return
        local_ground = (
            geometry.get("geometry_source") == "die_face_minus_ground_mask"
            and path != self.root_path
        )
        selector = (
            occurrence_ground_polygons_for_entity
            if local_ground
            else occurrence_polygons_for_entity
        )
        self.pending_checks.append(
            partial(
                source_recipe_job,
                selector,
                entity_from_layer_record(record, materials=self.source.stack.materials),
                source_cell=self.source_cells[occurrence["cell_name"]],
                transform=occurrence["transform"],
                excluded_reference_indexes=self.child_reference_indexes[path],
                **(
                    {
                        "plane_layer": self.source.stack.solution_regions[
                            geometry["plane_bounds_ref"]
                        ]["metadata"]["source_layer_datatype"]
                    }
                    if local_ground
                    else {}
                ),
                expected=record["geometry"].get("source_occurrence_polygons_um"),
                comparison_error=f"GeometryPlan polygon source differs for {record['semantic_id']!r}",
            )
        )

    def validate_layers(self) -> None:
        try:
            for record in self.source.stack.layers:
                self.validate_layer(record)
        except Exception:
            self.flush_expected()
            raise
        self.flush_expected()

    def validate_ports(self) -> None:
        try:
            for record in self.source.stack.metadata.get(
                "port_sheet_source_layers", ()
            ):
                path = (
                    record.get("source_occurrence_path")
                    if isinstance(record, Mapping)
                    else None
                )
                occurrence = self.by_path.get(path)
                if (
                    occurrence is None
                    or occurrence.get("cell_name") not in self.source_cells
                ):
                    raise ValueError(
                        "GeometryPlan port has no recorded source occurrence"
                    )
                self.pending_checks.append(
                    partial(
                        source_recipe_job,
                        occurrence_port_polygon,
                        record,
                        source_cell=self.source_cells[occurrence["cell_name"]],
                        transform=occurrence["transform"],
                        excluded_reference_indexes=self.child_reference_indexes[path],
                        expected=record.get("source_occurrence_polygon_um"),
                        comparison_error=f"GeometryPlan port source differs for {record.get('name')!r}",
                    )
                )
        except Exception:
            self.flush_expected()
            raise
        self.flush_expected()


def validate_plan_occurrences(
    by_path: Mapping[Any, Mapping[str, Any]],
    *,
    top_cell: Cell,
    source_cells: Mapping[str, Cell],
    source_dbu_um: float,
) -> None:
    """Bind every qualified path to one reference in the paired written GDS."""
    roots = [path for path, row in by_path.items() if row.get("parent_path") is None]
    if len(roots) != 1:
        raise ValueError("GeometryPlan GDS needs exactly one recorded root occurrence")
    root = roots[0]
    claimed: set[tuple[str, int]] = set()
    for path, row in by_path.items():
        if not isinstance(path, str) or not path or not isinstance(row, Mapping):
            raise ValueError("GeometryPlan occurrence path is invalid")
        raw_transform = row.get("transform")
        if (
            not record_sequence(raw_transform)
            or len(raw_transform) != 6
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
                for value in raw_transform
            )
        ):
            raise ValueError(f"GeometryPlan occurrence {path!r} has invalid transform")
        transform = tuple(float(value) for value in raw_transform)
        if path == root:
            if (
                row.get("cell_name") != top_cell.name
                or row.get("reference_index") is not None
                or any(
                    abs(actual - expected) > 1e-9
                    for actual, expected in zip(
                        transform, (1.0, 0.0, 0.0, 1.0, 0.0, 0.0), strict=False
                    )
                )
            ):
                raise ValueError("GeometryPlan root differs from selected GDS top cell")
            continue
        parent_path = path.rpartition("/")[0]
        parent = by_path.get(parent_path)
        index = row.get("reference_index")
        if (
            not parent_path.startswith(root)
            or row.get("parent_path") != parent_path
            or parent is None
            or parent.get("cell_name") not in source_cells
            or row.get("cell_name") not in source_cells
            or isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
        ):
            raise ValueError(
                f"GeometryPlan occurrence {path!r} has no GDS parent reference"
            )
        references = source_cells[parent["cell_name"]].references
        if index >= len(references) or (parent_path, index) in claimed:
            raise ValueError(
                f"GeometryPlan occurrence {path!r} has duplicate or missing GDS reference"
            )
        claimed.add((parent_path, index))
        reference = references[index]
        target = getattr(reference.cell, "name", reference.cell)
        if target != row["cell_name"] or len(reference.repetition.get_offsets()) != 0:
            raise ValueError(
                f"GeometryPlan occurrence {path!r} differs from its GDS reference"
            )
        expected = _reference_transform(reference, parent["transform"])
        if any(abs(transform[i] - expected[i]) > 1e-9 for i in range(4)) or any(
            abs(transform[i] - expected[i]) > source_dbu_um / 2 + 1e-9 for i in (4, 5)
        ):
            raise ValueError(
                f"GeometryPlan occurrence {path!r} transform differs from GDS"
            )


def validate_ground_contribution_ownership(
    raw_layers: Sequence[Mapping[str, Any]],
) -> None:
    """A ground include cannot consume a separately named local conductor."""
    import gdstk  # noqa: PLC0415 - load optional native geometry only at ownership validation.

    for plane in raw_layers:
        for contribution in plane.get("geometry", {}).get(
            "source_occurrence_includes", ()
        ):
            region = contribution["polygon"]
            included = layout_polygon_region(
                gdstk,
                LayoutPolygonSpec(
                    polygon_id="source_ground_contribution",
                    layer="0/0",
                    exterior=region["exterior"],
                    holes=region["holes"],
                ),
            )
            for other in raw_layers:
                if (
                    other.get("metadata", {}).get("source_occurrence_path")
                    != contribution["source_occurrence_path"]
                    or other["metadata"].get("logical_layer_id")
                    != contribution["level"]
                    or other.get("geometry", {}).get("geometry_source", "gds_polygon")
                    != "gds_polygon"
                ):
                    continue
                for polygon in other["geometry"]["source_occurrence_polygons_um"]:
                    claimed = layout_polygon_region(
                        gdstk,
                        LayoutPolygonSpec(
                            polygon_id="named_local_conductor",
                            layer="0/0",
                            exterior=polygon["exterior"],
                            holes=polygon["holes"],
                        ),
                    )
                    overlap = polygon_boolean(
                        included, claimed, "and", precision=current_grid_size_um()
                    )
                    if overlap:
                        raise ValueError(
                            _contribution_overlap(contribution, other, overlap)
                        )


def _contribution_overlap(
    contribution: Mapping[str, Any], other: Mapping[str, Any], overlap: Sequence[Any]
) -> str:
    """The refusal of a Ground contribution whose include region overlaps a named conductor, at the overlap."""
    layer = contribution["layer"]
    with named_source(
        str(contribution["source_local_id"]), f"{int(layer[0])}/{int(layer[1])}"
    ):
        return source_refusal(
            f"ground contribution {contribution['source_local_id']!r} "
            f"also contains local Entity {other['semantic_id']!r}",
            [
                f"its include region overlaps {other['semantic_id']} at "
                f"{source_place([point for part in overlap for point in part.points])}"
            ],
            "Fix the layout or the declaration: select an include region clear of the "
            "named conductors, or draw them apart.",
        )


def _reference_transform(
    reference: Reference, parent_transform: Sequence[Any]
) -> tuple[float, ...]:
    """Compose the native reference with its recorded parent in the original order."""
    angle = float(reference.rotation or 0.0)
    scale = float(reference.magnification or 1.0)
    reflection = -1.0 if reference.x_reflection else 1.0
    local = (
        scale * cos(angle),
        -scale * sin(angle) * reflection,
        scale * sin(angle),
        scale * cos(angle) * reflection,
        float(reference.origin[0]),
        float(reference.origin[1]),
    )
    pa, pb, pc, pd, px, py = (float(value) for value in parent_transform)
    a, b, c, d, x, y = local
    return (
        pa * a + pb * c,
        pa * b + pb * d,
        pc * a + pd * c,
        pc * b + pd * d,
        pa * x + pb * y + px,
        pc * x + pd * y + py,
    )
