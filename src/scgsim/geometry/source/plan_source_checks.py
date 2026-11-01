"""Check that a paired GeometryPlan stack still matches the GDS it was written with.

Owns: re-deriving every recorded source polygon from the GDS and comparing it with
what the stack file stored, so an edited GDS or a hand-edited stack fails before
lowering.

Does not own: lowering stack records into geometry input (`adapter.py`) or the
plan format itself (`plan.py`).

Every check raises `ValueError`. The order of checks decides which error a user
sees first, so the steps below run in the order of the original single function.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import cos, isfinite, sin
from pathlib import Path
from typing import Any

from scgsim.geometry.models.input import LayoutPolygonSpec
from scgsim.geometry.source._occurrence import (
    entity_from_layer_record,
    is_record_sequence,
    occurrence_ground_polygons_for_entity,
    occurrence_include_polygon,
    occurrence_polygons_for_entity,
    occurrence_port_polygon,
)
from scgsim.geometry.source.ground import layout_polygon_region

_DERIVED_GROUND = "die_face_minus_ground_mask"
_GROUND_CONTRIBUTION_FIELDS = {
    "source_occurrence_path",
    "source_local_id",
    "level",
    "layer",
    "selector_point_um",
    "polygon",
}
_GROUND_CONTRIBUTION_SUMMARY_FIELDS = (
    "source_occurrence_path",
    "source_local_id",
    "level",
    "layer",
)


@dataclass(frozen=True)
class _OccurrenceIndex:
    """The stack's recorded cell occurrences, keyed by qualified path."""

    source_cells: Mapping[str, Any]
    by_path: Mapping[str, Mapping[str, Any]]
    root_path: str | None
    child_reference_indexes: Mapping[str, tuple[int, ...]]

    def source_cell_of(self, path: Any) -> Any | None:
        """The GDS cell of the occurrence at `path`, or None if unrecorded."""
        occurrence = self.by_path.get(path)
        if occurrence is None or occurrence.get("cell_name") not in self.source_cells:
            return None
        return self.source_cells[occurrence["cell_name"]]


def verify_paired_plan_sources(  # noqa: PLR0913 - keyword-only; each argument is a separate input of the check
    *,
    gds_path: Path,
    library: Any,
    top_cell: Any,
    raw_layers: Sequence[Any],
    stack_metadata: Mapping[str, Any],
    solution_regions: Mapping[str, Any],
    materials: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Verify a schema-2 GeometryPlan stack against its GDS.

    Returns the `metadata` of the last derived ground plane record, or None when
    there is none. The original inline code reused the name `metadata` for that
    value and later merged it into the geometry input's metadata in place of the
    caller's metadata, so the adapter still does the same.
    """
    # TODO(human): is it intended that the caller's `metadata` argument is dropped
    # whenever the stack has a derived ground plane record? Kept as found.
    if (
        stack_metadata.get("gds_sha256")
        != hashlib.sha256(gds_path.read_bytes()).hexdigest()
    ):
        raise ValueError("GeometryPlan GDS identity differs from its paired stack")
    index = _index_occurrences(library, top_cell, stack_metadata)
    last_plane_metadata, derived_planes = _claim_derived_ground_planes(raw_layers)
    contribution_ids: set[tuple[str, str]] = set()
    for record in raw_layers:
        _check_layer_record(
            record,
            index=index,
            derived_planes=derived_planes,
            contribution_ids=contribution_ids,
            solution_regions=solution_regions,
            materials=materials,
        )
    _check_port_sheet_sources(stack_metadata, index)
    _validate_plan_ground_contribution_ownership(raw_layers)
    return last_plane_metadata


def _index_occurrences(
    library: Any, top_cell: Any, stack_metadata: Mapping[str, Any]
) -> _OccurrenceIndex:
    source_cells = {source_cell.name: source_cell for source_cell in library.cells}
    occurrences = stack_metadata.get("source_occurrences")
    if not is_record_sequence(occurrences):
        raise ValueError("GeometryPlan stack needs source occurrences")
    by_path = {row.get("path"): row for row in occurrences if isinstance(row, Mapping)}
    if len(by_path) != len(occurrences):
        raise ValueError("GeometryPlan source occurrence paths must be unique")
    root_path = (
        next(path for path, row in by_path.items() if row.get("parent_path") is None)
        if sum(row.get("parent_path") is None for row in by_path.values()) == 1
        else None
    )
    _validate_plan_occurrences(
        by_path,
        top_cell=top_cell,
        source_cells=source_cells,
        source_dbu_um=float(library.precision) * 1e6,
    )
    # A cell's own polygons exclude the references that are separate occurrences.
    child_reference_indexes = {
        path: tuple(
            child["reference_index"]
            for child in occurrences
            if child["parent_path"] == path
        )
        for path in by_path
    }
    return _OccurrenceIndex(
        source_cells=source_cells,
        by_path=by_path,
        root_path=root_path,
        child_reference_indexes=child_reference_indexes,
    )


def _claim_derived_ground_planes(
    raw_layers: Sequence[Any],
) -> tuple[Any, dict[Any, Any]]:
    """Map each derived ground plane record by id; one plane per physical level."""
    physical_planes: set[str] = set()
    planes: dict[Any, Any] = {}
    last_plane_metadata = None
    for record in raw_layers:
        if not isinstance(record, Mapping):
            raise ValueError("GeometryPlan layers must be mappings")
        geometry = record.get("geometry", {})
        if geometry.get("geometry_source") == _DERIVED_GROUND:
            last_plane_metadata = record.get("metadata", {})
            level = last_plane_metadata.get("logical_layer_id")
            if level in physical_planes:
                raise ValueError(
                    f"multiple ground planes claim physical level {level!r}"
                )
            physical_planes.add(level)
            planes[record.get("semantic_id")] = record
    return last_plane_metadata, planes


def _check_layer_record(  # noqa: PLR0913 - keyword-only; per-layer check shares plan-wide state
    record: Any,
    *,
    index: _OccurrenceIndex,
    derived_planes: Mapping[Any, Any],
    contribution_ids: set[tuple[str, str]],
    solution_regions: Mapping[str, Any],
    materials: Mapping[str, Any],
) -> None:
    """Check one layer record's ground contributions and its stored polygons."""
    if not isinstance(record, Mapping) or not isinstance(
        record.get("metadata"), Mapping
    ):
        raise ValueError("GeometryPlan layers need source occurrence metadata")
    path = record["metadata"].get("source_occurrence_path")
    if index.source_cell_of(path) is None:
        raise ValueError("GeometryPlan layer has no recorded source occurrence")
    geometry = record.get("geometry", {})
    source_summaries = []
    for contribution in geometry.get("source_occurrence_includes", ()):
        _check_ground_contribution(
            contribution,
            record=record,
            index=index,
            derived_planes=derived_planes,
            contribution_ids=contribution_ids,
        )
        source_summaries.append(
            {key: contribution[key] for key in _GROUND_CONTRIBUTION_SUMMARY_FIELDS}
        )
    if (
        record["metadata"].get("ground_plane_contribution_sources", [])
        != source_summaries
    ):
        raise ValueError("GeometryPlan ground contribution lineage metadata differs")
    # Only drawn polygons and non-root derived ground planes are re-derived from
    # the GDS; other geometry sources carry no stored source polygons.
    geometry_source = geometry.get("geometry_source", "gds_polygon")
    local_ground = geometry_source == _DERIVED_GROUND and path != index.root_path
    if geometry_source != "gds_polygon" and not local_ground:
        return
    _check_layer_polygons(
        record,
        path=path,
        local_ground=local_ground,
        index=index,
        solution_regions=solution_regions,
        materials=materials,
    )


def _check_ground_contribution(
    contribution: Any,
    *,
    record: Mapping[str, Any],
    index: _OccurrenceIndex,
    derived_planes: Mapping[Any, Any],
    contribution_ids: set[tuple[str, str]],
) -> None:
    """One ground include must name a real source and equal the GDS polygon."""
    if (
        not isinstance(contribution, Mapping)
        or set(contribution) != _GROUND_CONTRIBUTION_FIELDS
    ):
        raise ValueError("GeometryPlan ground contribution has invalid fields")
    if record.get("semantic_id") not in derived_planes:
        raise ValueError(
            "GeometryPlan ground contribution targets no root derived plane"
        )
    if contribution["level"] != record["metadata"].get("logical_layer_id"):
        raise ValueError(
            "GeometryPlan ground contribution level differs from target plane"
        )
    local_id = contribution["source_local_id"]
    if not isinstance(local_id, str) or not local_id or "/" in local_id:
        raise ValueError("GeometryPlan ground contribution local ID is invalid")
    source_path = contribution.get("source_occurrence_path")
    source_cell = index.source_cell_of(source_path)
    if source_cell is None:
        raise ValueError("GeometryPlan ground contribution has no source occurrence")
    key = (source_path, local_id)
    if key in contribution_ids:
        raise ValueError("GeometryPlan ground contribution local ID is duplicated")
    contribution_ids.add(key)
    expected = occurrence_include_polygon(
        contribution,
        cell=source_cell,
        transform=index.by_path[source_path]["transform"],
        excluded_reference_indexes=index.child_reference_indexes[source_path],
    )
    if contribution.get("polygon") != expected:
        raise ValueError("GeometryPlan ground contribution differs from GDS source")


def _check_layer_polygons(  # noqa: PLR0913 - keyword-only; mirrors _check_layer_record
    record: Mapping[str, Any],
    *,
    path: str,
    local_ground: bool,
    index: _OccurrenceIndex,
    solution_regions: Mapping[str, Any],
    materials: Mapping[str, Any],
) -> None:
    """Compare the record's stored polygons with those re-read from the GDS."""
    geometry = record.get("geometry", {})
    selector = (
        occurrence_ground_polygons_for_entity
        if local_ground
        else occurrence_polygons_for_entity
    )
    # The entity is built first so its errors still come before a bad plane ref.
    entity = entity_from_layer_record(record, materials=materials)
    extra: dict[str, Any] = {}
    if local_ground:
        # A non-root ground plane is cut by the layer its plane bounds name.
        extra["plane_layer"] = solution_regions[geometry["plane_bounds_ref"]][
            "metadata"
        ]["source_layer_datatype"]
    expected = selector(
        entity,
        cell=index.source_cell_of(path),
        transform=index.by_path[path]["transform"],
        excluded_reference_indexes=index.child_reference_indexes[path],
        **extra,
    )
    if record["geometry"].get("source_occurrence_polygons_um") != expected:
        raise ValueError(
            f"GeometryPlan polygon source differs for {record['semantic_id']!r}"
        )


def _check_port_sheet_sources(
    stack_metadata: Mapping[str, Any], index: _OccurrenceIndex
) -> None:
    for record in stack_metadata.get("port_sheet_source_layers", ()):
        path = (
            record.get("source_occurrence_path")
            if isinstance(record, Mapping)
            else None
        )
        source_cell = index.source_cell_of(path)
        if source_cell is None:
            raise ValueError("GeometryPlan port has no recorded source occurrence")
        expected = occurrence_port_polygon(
            record,
            cell=source_cell,
            transform=index.by_path[path]["transform"],
            excluded_reference_indexes=index.child_reference_indexes[path],
        )
        if record.get("source_occurrence_polygon_um") != expected:
            raise ValueError(
                f"GeometryPlan port source differs for {record.get('name')!r}"
            )


def _validate_plan_occurrences(
    by_path: Mapping[str, Mapping[str, Any]],
    *,
    top_cell: Any,
    source_cells: Mapping[str, Any],
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
        transform = _occurrence_transform(path, row)
        if path == root:
            _check_root_occurrence(row, transform, top_cell)
            continue
        parent_path = path.rpartition("/")[0]
        parent = by_path.get(parent_path)
        reference_index = row.get("reference_index")
        if (
            not parent_path.startswith(root)
            or row.get("parent_path") != parent_path
            or parent is None
            or parent.get("cell_name") not in source_cells
            or row.get("cell_name") not in source_cells
            or isinstance(reference_index, bool)
            or not isinstance(reference_index, int)
            or reference_index < 0
        ):
            raise ValueError(
                f"GeometryPlan occurrence {path!r} has no GDS parent reference"
            )
        references = source_cells[parent["cell_name"]].references
        if (
            reference_index >= len(references)
            or (
                parent_path,
                reference_index,
            )
            in claimed
        ):
            raise ValueError(
                f"GeometryPlan occurrence {path!r} has duplicate or missing GDS reference"
            )
        claimed.add((parent_path, reference_index))
        reference = references[reference_index]
        target = getattr(reference.cell, "name", reference.cell)
        if target != row["cell_name"] or len(reference.repetition.get_offsets()) != 0:
            raise ValueError(
                f"GeometryPlan occurrence {path!r} differs from its GDS reference"
            )
        expected = _composed_reference_transform(parent["transform"], reference)
        # Rotation/scale terms must agree tightly; translation may differ by half a
        # database unit because the GDS stores integer coordinates.
        if any(abs(transform[i] - expected[i]) > 1e-9 for i in range(4)) or any(
            abs(transform[i] - expected[i]) > source_dbu_um / 2 + 1e-9 for i in (4, 5)
        ):
            raise ValueError(
                f"GeometryPlan occurrence {path!r} transform differs from GDS"
            )


def _occurrence_transform(path: str, row: Mapping[str, Any]) -> tuple[float, ...]:
    """The occurrence's 2D affine transform (a, b, c, d, x, y) as finite floats."""
    raw_transform = row.get("transform")
    if (
        not is_record_sequence(raw_transform)
        or len(raw_transform) != 6
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            for value in raw_transform
        )
    ):
        raise ValueError(f"GeometryPlan occurrence {path!r} has invalid transform")
    return tuple(float(value) for value in raw_transform)


def _check_root_occurrence(
    row: Mapping[str, Any], transform: tuple[float, ...], top_cell: Any
) -> None:
    """The root must be the selected top cell, placed at the identity transform."""
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


def _composed_reference_transform(
    parent_transform: Sequence[Any], reference: Any
) -> tuple[float, ...]:
    """Parent transform composed with the GDS reference's own placement."""
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


def _validate_plan_ground_contribution_ownership(
    raw_layers: Sequence[Mapping[str, Any]],
) -> None:
    """A ground include cannot consume a separately named local conductor."""
    import gdstk

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
                    if gdstk.boolean(included, claimed, "and", precision=1e-9):
                        raise ValueError(
                            f"ground contribution {contribution['source_local_id']!r} "
                            f"also contains local Entity {other['semantic_id']!r}"
                        )
