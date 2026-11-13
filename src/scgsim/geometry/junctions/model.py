"""Compose exact source incidence, model emission and final junction checks, and assemble the derived input."""

from __future__ import annotations

from dataclasses import replace

from scgsim.geometry.grid.scope import GridScope
from scgsim.geometry.junctions.core import (
    DEFAULT_FORMAT,
    JunctionPartitionFormat,
    canonical_sha256,
    plain_record,
)
from scgsim.geometry.junctions.emission import (
    check_emitted_partitions,
    emit_partitions,
    initialize_registry,
    rebuild_entity_support,
    rebuild_original_polygons,
)
from scgsim.geometry.junctions.incidence import (
    initialize_source,
    register_crossings,
    register_owned_seams,
    register_source_edges,
    slice_free_boundaries,
    trace_free_boundaries,
)
from scgsim.geometry.junctions.model_state import JunctionModel
from scgsim.geometry.models.grid import geometry_grid_context
from scgsim.geometry.models.input import GeometryBuildInput


def model_geometry(build_input, originals, records, output_format=DEFAULT_FORMAT):
    """Trace source incidences and owned seams before topology, then slice at two cuts."""
    state = JunctionModel(
        build_input,
        originals,
        records,
        output_format,
        GridScope(geometry_grid_context(build_input)),
    )
    initialize_source(state)
    register_source_edges(state)
    register_crossings(state)
    register_owned_seams(state)
    trace_free_boundaries(state)
    slice_free_boundaries(state)
    initialize_registry(state)
    rebuild_original_polygons(state)
    emit_partitions(state)
    rebuild_entity_support(state)
    check_emitted_partitions(state)
    return (state.model, tuple(state.entities))


def assembled_input(
    build_input: GeometryBuildInput,
    model,
    entities,
    records,
    output_format: JunctionPartitionFormat,
) -> GeometryBuildInput:
    """The input with the emitted model in place of its sources: derived records and the authored-input hash."""
    metadata = dict(build_input.metadata)
    metadata[output_format.source_hash_key] = canonical_sha256(
        plain_record(build_input)
    )
    metadata[output_format.metadata_key] = records
    original_ids = {p.polygon_id for p in build_input.polygons}
    authored_ids = {record["authored_polygon_id"] for record in records.values()}
    derived_polygons = tuple(
        (model[p.polygon_id] for p in build_input.polygons)
    ) + tuple(
        (
            p
            for pid, p in model.items()
            if pid not in original_ids and pid not in authored_ids
        )
    )
    regions = tuple(
        (
            replace(
                r,
                exterior=model[r.source_polygon_id].exterior,
                holes=model[r.source_polygon_id].holes,
            )
            if r.source_polygon_id in model
            else r
            for r in build_input.port_sheet_regions
        )
    )
    return replace(
        build_input,
        polygons=derived_polygons,
        entities=entities,
        port_sheet_regions=regions,
        metadata=metadata,
    )
